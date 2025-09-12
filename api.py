# api.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from joblib import load
from scipy import sparse

# =============================================================================
# 0) Chemins ABSOLUS (robustes)
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()
ARTIFACT_DIR = BASE_DIR / "artifacts"

BASELINE_PATH = ARTIFACT_DIR / "model_baseline_logreg.joblib"
CALIB_ISO_PATH = ARTIFACT_DIR / "model_calibrated_isotonic.joblib"
CALIB_SIG_PATH = ARTIFACT_DIR / "model_calibrated_sigmoid.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
REF_STATS_PATH = ARTIFACT_DIR / "ref_stats.json"

# =============================================================================
# 1) Chargement modèle + seuil
# =============================================================================
def load_model_and_threshold() -> Tuple[Any, float, str]:
    """
    Charge le modèle choisi dans artifacts/ + lit le seuil de décision
    depuis metadata.json si présent.
    Retourne: (model, threshold, chosen_name)
    """
    chosen = "baseline"
    threshold = 0.5

    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            chosen = meta.get("chosen_model", "baseline")
            # decision_threshold peut être { "t_selected": 0.17, ... }
            th = meta.get("decision_threshold", {})
            if isinstance(th, dict):
                threshold = float(th.get("t_selected", threshold))
            elif isinstance(th, (int, float)):
                threshold = float(th)
        except Exception:
            pass

    # Sélection du bon artefact model
    if chosen == "isotonic" and CALIB_ISO_PATH.exists():
        model_path = CALIB_ISO_PATH
    elif chosen == "sigmoid" and CALIB_SIG_PATH.exists():
        model_path = CALIB_SIG_PATH
    else:
        # fallback
        chosen = "baseline"
        model_path = BASELINE_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefact modèle introuvable: {model_path}\n"
            "Assure-toi d’avoir entraîné et sauvegardé les artefacts."
        )

    model = load(model_path)
    return model, float(threshold), str(chosen)


MODEL, DECISION_THRESHOLD, CHOSEN_NAME = load_model_and_threshold()

# =============================================================================
# 2) Utilitaires pour accéder à la Pipeline même si le modèle est calibré
# =============================================================================
def get_pipeline_from_model(model: Any):
    """
    Retourne la Pipeline (préprocess + model) à expliquer / interroger
    même si le modèle est enveloppé dans un CalibratedClassifierCV.
    """
    # Cas simple: c'est déjà une Pipeline
    if hasattr(model, "named_steps"):
        return model

    # CalibratedClassifierCV (sklearn) expose .estimator (base estimator)
    for attr in ("estimator", "base_estimator"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "named_steps"):
            return inner

    # Impossible de récupérer une Pipeline
    return None


def get_preprocess_and_clf(pipe) -> Tuple[Any, Any]:
    """
    Retourne (preprocess, classifier) depuis une Pipeline scikit-learn
    en supposant des steps nommés "preprocess" et "model".
    """
    preprocess = None
    clf = None
    if pipe is not None and hasattr(pipe, "named_steps"):
        preprocess = pipe.named_steps.get("preprocess")
        clf = pipe.named_steps.get("model")
    return preprocess, clf


def get_expected_input_columns_from_preprocess(preprocess) -> List[str]:
    """
    Lit la configuration du ColumnTransformer (champ .transformers) pour
    récupérer les colonnes brutes attendues (num + cat).
    N'utilise PAS .transformers_ (pas besoin d’être 'fitted' pour lire la config).
    """
    cols: List[str] = []
    if preprocess is None:
        return cols

    transformers = getattr(preprocess, "transformers", None)
    if not transformers:
        return cols

    for name, _trans, cols_sel in transformers:
        if name in ("num", "cat"):
            # cols_sel est typiquement une liste de noms de colonnes
            if isinstance(cols_sel, (list, tuple)):
                cols.extend(list(cols_sel))
    # Par sécurité on enlève l'ID si jamais il s'y trouve
    cols = [c for c in cols if c != "SK_ID_CURR"]
    return cols


def get_full_feature_names(preprocess, cat_cols: List[str], num_cols: List[str]) -> Optional[np.ndarray]:
    """
    Construit la liste des noms de features APRÈS transformation:
      [num_cols] + OHE(cat_cols)
    Requiert un preprocess FIT (pour que .named_transformers_ et l'encodage existent).
    """
    if preprocess is None:
        return None

    try:
        # Récupère l'OHE de la branche 'cat'
        cat_pipe = preprocess.named_transformers_.get("cat")
        if cat_pipe is None:
            return None
        # Le OneHotEncoder peut être dans un Pipeline avec étape 'encoder'
        ohe = getattr(cat_pipe, "named_steps", {}).get("encoder", cat_pipe)
        cat_names = ohe.get_feature_names_out(cat_cols)
        full = np.concatenate([np.array(num_cols), np.array(cat_names)])
        return full
    except Exception:
        return None


def to_dense(X):
    """Convertit une matrice potentiellement sparse en dense numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


# =============================================================================
# 3) Colonnes attendues (brutes avant encodage)
# =============================================================================
def expected_input_columns() -> List[str]:
    """
    Déduit les colonnes brutes attendues par le préprocesseur.
    Fallback: si indisponible, essaie metadata.json ('expected_input_columns').
    """
    pipe = get_pipeline_from_model(MODEL)
    preprocess, _ = get_preprocess_and_clf(pipe)
    cols = get_expected_input_columns_from_preprocess(preprocess)
    if cols:
        return cols

    # Fallback: metadata.json
    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            cols_meta = meta.get("expected_input_columns")
            if isinstance(cols_meta, list) and cols_meta:
                return [c for c in cols_meta if c != "SK_ID_CURR"]
        except Exception:
            pass
    return []


EXPECTED_COLS = expected_input_columns()

# =============================================================================
# 4) Chargement des stats de référence (train) pour /ref_stats
# =============================================================================
def load_ref_stats() -> dict:
    """Charge artifacts/ref_stats.json (généré par make_ref_stats.py) si présent."""
    try:
        if REF_STATS_PATH.exists():
            return json.loads(REF_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

# =============================================================================
# 5) Pydantic models (IO)
# =============================================================================
class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    probability_default: float
    decision: str
    threshold: float
    missing_features: List[str]
    used_model: str

class ExplainResponse(BaseModel):
    probability_default: float
    decision: str
    threshold: float
    used_model: str
    bias: Optional[float] = None
    top_contributions: Optional[List[Dict[str, Any]]] = None
    note: Optional[str] = None

# =============================================================================
# 6) FastAPI app
# =============================================================================
app = FastAPI(title="Credit Scoring API", version="0.2.0")

# ---- petits helpers
def align_features(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aligne le dict reçu sur les colonnes attendues :
      - colonnes manquantes -> NaN
      - colonnes en trop -> ignorées
    """
    if not EXPECTED_COLS:
        raise HTTPException(
            status_code=500,
            detail="Colonnes attendues indisponibles. Vérifie l'artefact du préprocesseur ou metadata.json."
        )
    missing = [c for c in EXPECTED_COLS if c not in payload]
    row = {c: payload.get(c, np.nan) for c in EXPECTED_COLS}
    df = pd.DataFrame([row], columns=EXPECTED_COLS)
    return df, missing

def decide_label(prob: float, thr: float) -> str:
    return "refusé" if prob >= thr else "accordé"


# =============================================================================
# 7) Endpoints
# =============================================================================
@app.get("/health")
def health():
    ref_ok = REF_STATS_PATH.exists()
    return {
        "status": "ok",
        "used_model": CHOSEN_NAME,
        "threshold": DECISION_THRESHOLD,
        "expected_n_features": len(EXPECTED_COLS),
        "ref_stats_available": ref_ok,
    }

@app.get("/expected_features")
def expected_features():
    return {"expected_features": EXPECTED_COLS}

@app.get("/ref_stats")
def ref_stats():
    """
    Statistiques de référence calculées à partir du train (application_train.csv),
    générées via make_ref_stats.py et sauvegardées dans artifacts/ref_stats.json.
    """
    stats = load_ref_stats()
    if not stats:
        return {
            "available": False,
            "message": "ref_stats.json absent. Lance make_ref_stats.py pour le générer."
        }
    return {"available": True, "stats": stats}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not isinstance(req.features, dict) or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Le champ 'features' doit être un dictionnaire non vide.")

    X_input, missing = align_features(req.features)

    try:
        proba = float(MODEL.predict_proba(X_input)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {e}")

    decision = decide_label(proba, DECISION_THRESHOLD)

    return PredictResponse(
        probability_default=proba,
        decision=decision,
        threshold=float(DECISION_THRESHOLD),
        missing_features=missing,
        used_model=CHOSEN_NAME
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(
    req: PredictRequest,
    top_k: int = Query(8, ge=1, le=50, description="Nombre de contributions à retourner"),
):
    """
    Explication locale (TOP-K) en termes de contributions en log-odds.
    - Fonctionne lorsque le modèle interne est une Régression Logistique dans une Pipeline
      (même si le modèle exposé est calibré Isotonic/Sigmoid).
    - On explique la partie linéaire (pré-calibrage).
    """
    if not isinstance(req.features, dict) or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Le champ 'features' doit être un dictionnaire non vide.")

    # 1) Toujours renvoyer la prédiction API (calibrée ou non)
    try:
        X_input, _missing = align_features(req.features)
        proba = float(MODEL.predict_proba(X_input)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {e}")

    decision = decide_label(proba, DECISION_THRESHOLD)

    # 2) Tenter une explication log-odds via la Pipeline interne
    pipe = get_pipeline_from_model(MODEL)
    preprocess, clf = get_preprocess_and_clf(pipe)

    # pipeline valide + logistic regression ?
    ok_lr = hasattr(clf, "coef_") and hasattr(clf, "intercept_")
    if not (pipe and preprocess and ok_lr and EXPECTED_COLS):
        # Explication non dispo: renvoyer une note explicite
        return ExplainResponse(
            probability_default=proba,
            decision=decision,
            threshold=float(DECISION_THRESHOLD),
            used_model=CHOSEN_NAME,
            top_contributions=None,
            bias=None,
            note="Explication indisponible (modèle non linéaire ou artefact incompatible)."
        )

    # Colonnes brutes num/cat d'après la config
    num_cols: List[str] = []
    cat_cols: List[str] = []
    transformers = getattr(preprocess, "transformers", []) or []
    for name, _t, cols_sel in transformers:
        if name == "num":
            num_cols = list(cols_sel)
        elif name == "cat":
            cat_cols = list(cols_sel)

    # Features après transformation (nécessite preprocess FIT)
    full_names = get_full_feature_names(preprocess, cat_cols, num_cols)
    if full_names is None:
        return ExplainResponse(
            probability_default=proba,
            decision=decision,
            threshold=float(DECISION_THRESHOLD),
            used_model=CHOSEN_NAME,
            top_contributions=None,
            bias=None,
            note="Explication indisponible (noms de features transformées inaccessibles)."
        )

    # Transformer l'input avec le preprocess
    try:
        X_proc = preprocess.transform(X_input)  # sparse/dense
        X_proc = to_dense(X_proc).reshape(1, -1)
    except Exception as e:
        return ExplainResponse(
            probability_default=proba,
            decision=decision,
            threshold=float(DECISION_THRESHOLD),
            used_model=CHOSEN_NAME,
            top_contributions=None,
            bias=None,
            note=f"Explication indisponible (erreur transform): {e}"
        )

    # Contributions en log-odds = coef * x ; bias = intercept
    coef = np.asarray(clf.coef_).reshape(-1)      # shape (n_features,)
    bias = float(np.asarray(clf.intercept_).reshape(-1)[0])

    if X_proc.shape[1] != coef.shape[0]:
        return ExplainResponse(
            probability_default=proba,
            decision=decision,
            threshold=float(DECISION_THRESHOLD),
            used_model=CHOSEN_NAME,
            top_contributions=None,
            bias=None,
            note="Explication indisponible (mismatch dimensions features/coefs)."
        )

    contrib = X_proc.flatten() * coef  # (n_features,)
    # top-k par valeur absolue
    order = np.argsort(-np.abs(contrib))[:top_k]
    top = [
        {"feature": str(full_names[i]), "contribution": float(contrib[i])}
        for i in order
    ]

    return ExplainResponse(
        probability_default=proba,
        decision=decision,
        threshold=float(DECISION_THRESHOLD),
        used_model=CHOSEN_NAME,
        bias=bias,
        top_contributions=top,
        note="Contributions exprimées en log-odds sur la couche linéaire (avant calibration éventuelle)."
    )
