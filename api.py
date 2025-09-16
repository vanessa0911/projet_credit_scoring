# api.py
# ======================================================================================
# API de scoring (FastAPI) — robuste aux artefacts et aux environnements Windows
# Endpoints :
#   GET  /                    -> statut + méta
#   GET  /health              -> statut "ok"
#   GET  /expected_columns    -> liste des colonnes attendues (avant encodage)
#   GET  /value_domains       -> domaines de valeurs pour les variables qualitatives (si détectables)
#   POST /predict             -> {"data": { ...features... }}         -> proba + décision
#   POST /predict_proba_batch -> {"records": [ {...}, {...} ]}        -> probas + décisions
#
# Artefacts attendus dans ./artifacts :
#   - model_calibrated_isotonic.joblib
#   - model_calibrated_sigmoid.joblib
#   - model_baseline_logreg.joblib
#   - metadata.json  (peut contenir: chosen_model, decision_threshold, expected_input_columns)
#
# Notes robustesse :
#   - Si l’artefact du modèle "choisi" est manquant, on prend le premier disponible.
#   - expected_columns : si absent du metadata, on essaie de le déduire automatiquement du modèle.
#   - Lecture JSON tolérante (UTF-8 BOM + suppression virgules finales).
# ======================================================================================

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from joblib import load

# ---- pour /value_domains (option B)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ======================================================================================
# Constantes & chemins
# ======================================================================================

HERE = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
META_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

MODEL_CANDIDATES = {
    "isotonic": "model_calibrated_isotonic.joblib",
    "sigmoid": "model_calibrated_sigmoid.joblib",
    "baseline": "model_baseline_logreg.joblib",
}

# ======================================================================================
# Helpers
# ======================================================================================

def _safe_json_read(path: str) -> Dict[str, Any]:
    """Lecture JSON tolérante : UTF-8 (BOM ok), supprime une éventuelle virgule finale."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Tolérance minimale: supprimer les virgules avant } ou ]
        import re
        txt2 = re.sub(r',\s*([}\]])', r'\1', txt)
        return json.loads(txt2)

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _extract_expected_columns(model: Any) -> List[str]:
    """
    Essaie d'extraire la liste des colonnes d'entrée "brutes" attendues par le préprocess du modèle.
    On tente plusieurs cas : estimator direct, pipeline, calibrateur, etc.
    """
    # 1) Estimator/Pipeline récent avec feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in list(getattr(model, "feature_names_in_"))]

    # 2) Pipeline sklearn : chercher un step qui expose feature_names_in_ (souvent le preprocess)
    for attr in ("named_steps", "steps"):
        if hasattr(model, attr):
            steps = getattr(model, attr)
            # steps peut être dict (named_steps) ou liste (steps)
            if isinstance(steps, dict):
                for step in steps.values():
                    if hasattr(step, "feature_names_in_"):
                        return [str(c) for c in list(step.feature_names_in_)]
            elif isinstance(steps, list):
                for _, step in steps:
                    if hasattr(step, "feature_names_in_"):
                        return [str(c) for c in list(step.feature_names_in_)]

    # 3) CalibratedClassifierCV et autres wrappers
    for wrap_attr in ("base_estimator", "estimator", "classifier"):
        if hasattr(model, wrap_attr):
            inner = getattr(model, wrap_attr)
            cols = _extract_expected_columns(inner)
            if cols:
                return cols

    # 4) Rien trouvé
    return []

def _predict_proba_safely(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Renvoie proba classe positive (shape: (n_samples,))."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):  # certains wrappers exotiques
            proba = proba[0]
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
    if hasattr(model, "decision_function"):
        df = np.asarray(model.decision_function(X)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-df))  # sigmoïde
    raise RuntimeError("Le modèle ne supporte pas predict_proba/decision_function.")

def _build_dataframe_one(data: Dict[str, Any], expected_cols: List[str]) -> pd.DataFrame:
    """
    Construit un DataFrame 1 ligne avec exactement les colonnes attendues.
    - Cols manquantes -> None
    - Cols en trop -> ignorées
    - Si expected_cols vide -> on prend toutes les clés de `data` (ordre trié)
    """
    if expected_cols:
        row = {col: data.get(col, None) for col in expected_cols}
        return pd.DataFrame([row])
    cols = sorted(list(data.keys()))
    return pd.DataFrame([{c: data.get(c, None) for c in cols}])

def _decision_from_threshold(p: float, decision_threshold: Dict[str, Any] | float | None) -> Optional[int]:
    """Applique la décision (0/1) si un seuil est défini (1 = refuser, 0 = accepter)."""
    if decision_threshold is None:
        return None
    if isinstance(decision_threshold, dict) and "t_selected" in decision_threshold:
        t = float(decision_threshold["t_selected"])
        return int(p >= t)
    if isinstance(decision_threshold, (int, float)):
        return int(p >= float(decision_threshold))
    return None

# ---- helpers option B: extraire les domaines de valeurs depuis le modèle
def _find_column_transformer(model: Any) -> Optional[ColumnTransformer]:
    # Cherche un ColumnTransformer dans la pipeline
    for attr in ("named_steps", "steps"):
        if hasattr(model, attr):
            steps = getattr(model, attr)
            if isinstance(steps, dict):
                for step in steps.values():
                    if isinstance(step, ColumnTransformer):
                        return step
            elif isinstance(steps, list):
                for _, step in steps:
                    if isinstance(step, ColumnTransformer):
                        return step
    # Wrappers éventuels
    for wrap_attr in ("base_estimator", "estimator", "classifier"):
        if hasattr(model, wrap_attr):
            inner = getattr(model, wrap_attr)
            ct = _find_column_transformer(inner)
            if ct is not None:
                return ct
    return None

def _value_domains_from_model(model: Any) -> Dict[str, list]:
    """
    Retourne {col_categorielle: [valeurs connues, ...]} si OneHotEncoder(categories_) est dispo.
    - Gère OHE direct dans ColumnTransformer
    - Gère OHE dans un sous-pipeline nommé 'cat'
    """
    domains: Dict[str, list] = {}
    ct = _find_column_transformer(model)
    if ct is None or not hasattr(ct, "transformers_"):
        return domains

    for name, transf, cols in ct.transformers_:
        cols_list = list(cols) if cols is not None else []
        # cas 1: OneHotEncoder direct
        if isinstance(transf, OneHotEncoder) and hasattr(transf, "categories_"):
            for c, colname in enumerate(cols_list):
                try:
                    cats = list(transf.categories_[c])
                except Exception:
                    cats = []
                # cast en str (None -> "None" non souhaité -> on garde None)
                cleaned = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in cats]
                domains[str(colname)] = cleaned
            continue

        # cas 2: pipeline imbriqué avec step OHE (souvent nommé 'cat')
        if hasattr(transf, "named_steps"):
            # cherche un OneHotEncoder dans les steps
            for _, step in getattr(transf, "named_steps").items():
                if isinstance(step, OneHotEncoder) and hasattr(step, "categories_"):
                    for c, colname in enumerate(cols_list):
                        try:
                            cats = list(step.categories_[c])
                        except Exception:
                            cats = []
                        cleaned = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in cats]
                        domains[str(colname)] = cleaned
                    break
    return domains

# ======================================================================================
# Chargement modèle + méta
# ======================================================================================

def load_model_and_threshold() -> Tuple[Any, Dict[str, Any] | float | None, str, List[str]]:
    """
    Charge le modèle + le seuil de décision + nom du champion + colonnes d'entrée.
    """
    meta = _safe_json_read(META_PATH)

    chosen_key = str(meta.get("chosen_model", "") or "").lower()
    chosen_name = chosen_key if chosen_key in MODEL_CANDIDATES else "unknown"

    # Ordre de recherche : modèle "choisi" puis fallbacks
    priority_files = []
    if chosen_name in MODEL_CANDIDATES:
        priority_files.append(os.path.join(ARTIFACT_DIR, MODEL_CANDIDATES[chosen_name]))
    for _, fname in MODEL_CANDIDATES.items():
        path = os.path.join(ARTIFACT_DIR, fname)
        if path not in priority_files:
            priority_files.append(path)

    model_path = _first_existing(priority_files)
    if model_path is None:
        raise RuntimeError(
            "Aucun artefact modèle trouvé dans ./artifacts "
            f"(cherché : {', '.join(os.path.basename(p) for p in priority_files)})"
        )

    # Chargement du modèle
    try:
        model = load(model_path)
    except ModuleNotFoundError as e:
        missing_mod = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise RuntimeError(
            f"Impossible de charger le modèle ({os.path.basename(model_path)}). "
            f"Module manquant : {missing_mod}. "
            "Installe la librairie requise (ex: pip install catboost) "
            "ou réentraîne/sérialise un modèle sans cette dépendance."
        )
    except Exception as e:
        raise RuntimeError(f"Echec de chargement de {os.path.basename(model_path)} : {e}")

    # Seuil de décision
    dth = meta.get("decision_threshold", None)
    if isinstance(dth, dict) and "t_selected" in dth:
        decision_threshold = dth
    elif isinstance(dth, (int, float)):
        decision_threshold = float(dth)
    else:
        decision_threshold = None

    # Colonnes attendues (avant encodage)
    expected_cols: List[str] = meta.get("expected_input_columns") or []
    if not isinstance(expected_cols, list):
        expected_cols = []

    # Fallback : si absent du metadata, on essaie de déduire depuis le modèle
    if not expected_cols:
        expected_cols = _extract_expected_columns(model)

    return model, decision_threshold, chosen_name, expected_cols

# ======================================================================================
# FastAPI app
# ======================================================================================

app = FastAPI(title="Credit Scoring API", version="1.1.0")

# CORS : autorise front local (Streamlit) sur localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en local, on simplifie ; en prod, restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charge modèle & seuil au démarrage du module
MODEL: Any
DECISION_THRESHOLD: Dict[str, Any] | float | None
CHOSEN_NAME: str
EXPECTED_COLS: List[str]

try:
    MODEL, DECISION_THRESHOLD, CHOSEN_NAME, EXPECTED_COLS = load_model_and_threshold()
except Exception as e:
    raise RuntimeError(f"[API startup] Erreur de chargement des artefacts : {e}")

# ======================================================================================
# Schémas Pydantic (entrées)
# ======================================================================================

class PredictRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Features du client (avant encodage)")

class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Liste de clients (features brutes)")

# ======================================================================================
# Endpoints
# ======================================================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/")
def root() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "status": "ok",
        "endpoints": ["/predict", "/predict_proba_batch", "/expected_columns", "/value_domains", "/health", "/docs"],
    }
    if CHOSEN_NAME:
        info["chosen_model"] = CHOSEN_NAME
    if DECISION_THRESHOLD is not None:
        if isinstance(DECISION_THRESHOLD, dict):
            info["decision_threshold"] = DECISION_THRESHOLD
        else:
            info["decision_threshold"] = {"t_selected": float(DECISION_THRESHOLD)}
    if EXPECTED_COLS:
        info["n_expected_columns"] = len(EXPECTED_COLS)
    return info

@app.get("/expected_columns")
def expected_columns() -> Dict[str, Any]:
    return {"expected_columns": EXPECTED_COLS, "count": len(EXPECTED_COLS)}

@app.get("/value_domains")
def value_domains() -> Dict[str, Any]:
    """
    Renvoie les valeurs possibles pour les variables qualitatives (si détectables).
    Utilise OneHotEncoder.categories_ via le ColumnTransformer.
    """
    try:
        domains = _value_domains_from_model(MODEL)
        return {"domains": domains, "count": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible d'extraire les domaines: {e}")

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        X = _build_dataframe_one(req.data, EXPECTED_COLS)
        proba = float(_predict_proba_safely(MODEL, X)[0])
        decision = _decision_from_threshold(proba, DECISION_THRESHOLD)
        return {
            "probability_default": proba,
            "decision": decision,   # 1 = refuser, 0 = accepter (selon seuil)
            "threshold": DECISION_THRESHOLD,
            "model": CHOSEN_NAME,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de scoring : {e}")

@app.post("/predict_proba_batch")
def predict_proba_batch(req: BatchPredictRequest) -> Dict[str, Any]:
    try:
        if not req.records:
            return {"results": [], "count": 0, "model": CHOSEN_NAME, "threshold": DECISION_THRESHOLD}

        # Harmonise toutes les lignes sur les colonnes attendues
        if EXPECTED_COLS:
            rows = [{col: r.get(col, None) for col in EXPECTED_COLS} for r in req.records]
            X = pd.DataFrame(rows)
        else:
            # fallback : union des clés reçues
            all_keys = sorted({k for r in req.records for k in r.keys()})
            rows = [{k: r.get(k, None) for k in all_keys} for r in req.records]
            X = pd.DataFrame(rows)

        probas = _predict_proba_safely(MODEL, X).astype(float).tolist()
        decisions = [_decision_from_threshold(p, DECISION_THRESHOLD) for p in probas]
        results = [{"probability_default": p, "decision": d} for p, d in zip(probas, decisions)]
        return {"results": results, "count": len(results), "model": CHOSEN_NAME, "threshold": DECISION_THRESHOLD}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur batch : {e}")

# ======================================================================================
# Lanceur local (optionnel)
# ======================================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
