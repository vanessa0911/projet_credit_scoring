# api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute


# api.py (extraits à ajouter)
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd


# Charger votre modèle et votre pipeline comme déjà fait dans votre repo
# model = ...
# preprocessor = ...


try:
import shap
SHAP_OK = True
explainer = None # lazy init
except Exception:
SHAP_OK = False


app = FastAPI()


class Features(BaseModel):
features: dict


class Rows(BaseModel):
rows: list


@app.get("/global_importance")
def global_importance():
"""Retourne l'importance globale (SHAP moyen absolu si SHAP disponible, sinon importance du modèle si dispo)."""
global explainer
try:
# Option 1 : SHAP global (préféré)
if SHAP_OK:
if explainer is None:
# Construire un petit background pour SHAP (ex: 1k échantillons d'un dataset de ref)
# X_bg = pd.read_parquet("artifacts/ref_sample.parquet") # adapter
# X_bg_proc = preprocessor.transform(X_bg)
# explainer = shap.Explainer(model, X_bg_proc)
pass # implémenter selon vos artefacts
# importance = np.abs(explainer.expected_value) # ceci n'est pas la bonne métrique
# => Préférez calculer mean(|SHAP|) sur un set de ref
# shap_vals = explainer(X_bg_proc)
# gi = np.mean(np.abs(shap_vals.values), axis=0)
# return [{"feature": f, "importance": float(w)} for f, w in zip(feature_names, gi)]
raise NotImplementedError
else:
# Option 2 : importance du modèle (ex. feature_importances_)
if hasattr(model, "feature_importances_"):
imps = model.feature_importances_.ravel()
feature_names = preprocessor.get_feature_names_out()
return [{"feature": f, "importance": float(w)} for f, w in zip(feature_names, imps)]
except Exception:
pass
return []


@app.post("/shap_local")
def shap_local(payload: Features):
"""Explique un dossier (retourne les SHAP values + base_value)."""
if not SHAP_OK:
return {"detail": "SHAP non disponible côté serveur"}
# x = pd.DataFrame([payload.features])
# x_proc = preprocessor.transform(x)
# global explainer
# if explainer is None:
# # idem : initialiser sur un background
# X_bg_proc = ...
# explainer = shap.Explainer(model, X_bg_proc)
# sv = explainer(x_proc)
# return {
# "feature_names": list(preprocessor.get_feature_names_out()),
# "shap_values": sv.values[0].tolist(),
# "base_value": float(sv.base_values[0]) if hasattr(sv, "base_values") else None,
# }
return {"detail": "À implémenter selon vos artefacts"}



# ------------------------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="Credit Scoring API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Codespaces: ouvert large
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Constantes & cache
# ------------------------------------------------------------------------------
DATA_CSV = Path("data/application_train.csv")
ARTIFACTS_DIR = Path("artifacts")
REF_STATS_JSON = ARTIFACTS_DIR / "ref_stats.json"
INTERP_SUMMARY_JSON = ARTIFACTS_DIR / "interpretability_summary.json"

_expected_columns_cache: Optional[List[str]] = None
_top_features_cache: Optional[List[str]] = None

# ------------------------------------------------------------------------------
# Utilitaires colonnes attendues
# ------------------------------------------------------------------------------
def _load_expected_columns() -> List[str]:
    """
    Ordre de priorité:
      1) en-tête du CSV si présent et non vide
      2) ref_stats.json si non vide
      3) fallback minimal (3 colonnes)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) depuis le CSV
    if DATA_CSV.exists():
        try:
            cols = list(pd.read_csv(DATA_CSV, nrows=0).columns)
            if cols:
                _expected_columns_cache = cols
                return _expected_columns_cache
        except Exception:
            pass

    # 2) depuis les artefacts
    if REF_STATS_JSON.exists():
        try:
            with open(REF_STATS_JSON, "r", encoding="utf-8") as f:
                ref = json.load(f)
            cols: List[str] = []
            for key in ("numerical_columns", "categorical_columns", "all_columns"):
                v = ref.get(key)
                if isinstance(v, list):
                    cols.extend(v)
            cols = [c for c in dict.fromkeys(cols) if isinstance(c, str) and c.strip()]
            if cols:
                _expected_columns_cache = cols
                return _expected_columns_cache
        except Exception:
            pass

    # 3) fallback minimal
    _expected_columns_cache = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"]
    return _expected_columns_cache

# ------------------------------------------------------------------------------
# Utilitaires top-features (robustes sur n'importe quel CSV d'artefacts)
# ------------------------------------------------------------------------------
_DEFAULT_TOP10 = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_RATING_CLIENT",
    "CNT_FAM_MEMBERS",
]

def _pick_feature_and_importance_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Devine la colonne 'feature' et optionnellement la colonne d'importance.
    - Si aucune importance claire, on retournera (feature_col, None) et on gardera l'ordre.
    """
    cols_lower = [c.lower() for c in df.columns]
    # Essayer des noms fréquents pour la colonne feature
    feature_aliases = ["feature", "features", "variable", "name", "column"]
    feature_col = None
    for alias in feature_aliases:
        if alias in cols_lower:
            feature_col = df.columns[cols_lower.index(alias)]
            break
    if feature_col is None:
        # fallback: première colonne textuelle
        feature_col = df.columns[0]

    # Chercher une colonne d'importance
    imp_aliases = [
        "importance", "gain", "weight", "shap", "abs_importance",
        "feature_importance", "feature_importances", "importance_mean",
    ]
    imp_col: Optional[str] = None
    for alias in imp_aliases:
        if alias in cols_lower:
            imp_col = df.columns[cols_lower.index(alias)]
            break

    return feature_col, imp_col

def _load_top_features_from_any_csv(path: Path) -> Optional[List[str]]:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0 or df.shape[0] == 0:
            return None
        fcol, icol = _pick_feature_and_importance_columns(df)
        # Nettoyage
        df = df[[fcol] + ([icol] if icol and icol in df.columns else [])].dropna(subset=[fcol])
        df[fcol] = df[fcol].astype(str)
        if icol and icol in df.columns:
            # tri par importance absolue décroissante
            try:
                df = df.sort_values(by=icol, key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False)
            except Exception:
                # si non numérique, garder l'ordre
                pass
        feats = [f for f in df[fcol].tolist() if f.strip()]
        return feats[:10] if feats else None
    except Exception:
        return None

def _load_top_features_from_json(path: Path) -> Optional[List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        gi = obj.get("global_importance") or obj.get("global_importances")
        if isinstance(gi, dict) and gi:
            feats = [k for k, _ in sorted(gi.items(), key=lambda kv: abs(kv[1]), reverse=True)]
            feats = [f for f in feats if isinstance(f, str) and f.strip()]
            if feats:
                return feats[:10]
        tf = obj.get("top_features")
        if isinstance(tf, list) and tf:
            feats = [str(x) for x in tf if isinstance(x, (str, int, float))]
            if feats:
                return feats[:10]
    except Exception:
        pass
    return None

def _scan_artifacts_for_top10() -> Optional[List[str]]:
    """Parcourt tous les CSV de artifacts/ et retourne la première Top-10 plausible."""
    if not ARTIFACTS_DIR.exists():
        return None
    # Priorités légères : fichiers dont le nom contient "importance" ou "shap"
    csvs = sorted(ARTIFACTS_DIR.glob("*.csv"))
    priority = [p for p in csvs if any(k in p.name.lower() for k in ("importance", "importances", "shap"))]
    others = [p for p in csvs if p not in priority]
    for path in priority + others:
        feats = _load_top_features_from_any_csv(path)
        if feats:
            return feats
    # sinon, tenter le JSON
    if INTERP_SUMMARY_JSON.exists():
        feats = _load_top_features_from_json(INTERP_SUMMARY_JSON)
        if feats:
            return feats
    return None

def _load_top_features() -> List[str]:
    """NE RENVOIE JAMAIS une liste vide : scan CSV/JSON, sinon fallback."""
    global _top_features_cache
    if _top_features_cache is not None:
        return _top_features_cache

    feats = _scan_artifacts_for_top10()
    if feats:
        _top_features_cache = feats
        return _top_features_cache

    _top_features_cache = _DEFAULT_TOP10[:]
    return _top_features_cache

# ------------------------------------------------------------------------------
# Fallback de proba (pour dev/démo)
# ------------------------------------------------------------------------------
def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _fallback_probability(features: Dict[str, Any]) -> float:
    """
    Heuristique simple basée sur ratio CREDIT/INCOME (borne 0.01..0.99).
    """
    credit = None
    income = None
    for k, v in features.items():
        lk = str(k).lower()
        if credit is None and ("credit" in lk or "amt_credit" in lk):
            credit = _safe_float(v)
        if income is None and ("income" in lk or "amt_income_total" in lk):
            income = _safe_float(v)

    if credit is not None and income is not None and income > 0:
        ratio = credit / income
        prob = max(0.01, min(0.99, 0.2 + 0.6 * (ratio / (ratio + 1.0))))
    else:
        prob = 0.5
    return float(prob)

def _endpoint_list() -> List[str]:
    return [r.path for r in app.routes if isinstance(r, APIRoute)]

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "name": "Credit Scoring API", "endpoints": _endpoint_list()}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "name": "Credit Scoring API", "endpoints": _endpoint_list()}

@app.get("/expected_columns")
def expected_columns() -> List[str]:
    return _load_expected_columns()

@app.get("/top_features")
def top_features() -> List[str]:
    return _load_top_features()

@app.get("/debug/artifacts")
def debug_artifacts() -> Dict[str, Any]:
    """
    Aide au diagnostic : liste les fichiers dans artifacts/ et
    montre le premier aperçu de CSV utilisable si trouvé.
    """
    info: Dict[str, Any] = {
        "artifacts_dir_exists": ARTIFACTS_DIR.exists(),
        "csv_files": [],
        "json_files": [],
        "chosen_top10": _load_top_features(),
    }
    if ARTIFACTS_DIR.exists():
        info["csv_files"] = [p.name for p in ARTIFACTS_DIR.glob("*.csv")]
        info["json_files"] = [p.name for p in ARTIFACTS_DIR.glob("*.json")]
    return info

@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Le corps doit être un objet JSON {clé: valeur}.")
    prob = _fallback_probability(payload)
    threshold = 0.5
    decision = int(prob >= threshold)
    return {"probability": prob, "decision": decision, "threshold": threshold, "used_columns_count": len(_load_expected_columns())}

@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Le corps doit contenir une liste 'instances'.")
    probs = []
    for row in instances:
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail="Chaque instance doit être un objet JSON.")
        probs.append(_fallback_probability(row))
    return {"probabilities": probs, "count": len(probs)}

# ------------------------------------------------------------------------------
# Exécution directe
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
