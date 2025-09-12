# api.py
from typing import Dict, Any, List, Tuple
import os
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load


# =========================
# Chemins des artefacts
# =========================
ARTIFACT_DIR = "artifacts"
BASELINE_PATH = os.path.join(ARTIFACT_DIR, "model_baseline_logreg.joblib")
CALIB_ISO_PATH = os.path.join(ARTIFACT_DIR, "model_calibrated_isotonic.joblib")
CALIB_SIG_PATH = os.path.join(ARTIFACT_DIR, "model_calibrated_sigmoid.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")


# =========================
# Helpers : déballer la Pipeline & récupérer le preprocess
# =========================
def unwrap_pipeline(est: Any):
    """
    Retourne la Pipeline interne (avec .named_steps) quel que soit l'enrobage.
    - Si 'est' est déjà une Pipeline -> retourne est
    - Si 'est' est un CalibratedClassifierCV -> retourne est.estimator (ou base_estimator) s'il s'agit d'une Pipeline
    """
    if hasattr(est, "named_steps"):
        return est

    inner = None
    if hasattr(est, "estimator"):
        inner = getattr(est, "estimator")
    elif hasattr(est, "base_estimator"):
        inner = getattr(est, "base_estimator")

    if inner is not None and hasattr(inner, "named_steps"):
        return inner

    raise AttributeError(
        "Impossible de retrouver la Pipeline interne (named_steps). "
        "Assure-toi que l'artefact est une sklearn.Pipeline ou un CalibratedClassifierCV(Pipeline)."
    )


def get_preprocess_step(est: Any):
    """
    Renvoie le step 'preprocess' (ColumnTransformer) depuis une Pipeline.
    """
    pipe_inner = unwrap_pipeline(est)
    if "preprocess" not in pipe_inner.named_steps:
        raise KeyError(
            "Le step 'preprocess' est introuvable dans la Pipeline. "
            "Vérifie que ta Pipeline est bien Pipeline([('preprocess', ...), ('model', ...)])"
        )
    return pipe_inner.named_steps["preprocess"]


def list_cols_from_column_transformer(ct) -> List[str]:
    """
    Retourne la liste des colonnes déclarées pour les blocs 'num' et 'cat'.
    Gère les deux cas :
      - avant fit  : ct.transformers
      - après fit  : ct.transformers_
    """
    items = getattr(ct, "transformers_", None)
    if items is None:
        items = getattr(ct, "transformers", [])

    cols: List[str] = []
    for t in items:
        # t est typiquement (name, transformer, cols_sel, [optional_weight])
        name = t[0]
        if name not in ("num", "cat"):
            continue
        cols_sel = t[2] if len(t) >= 3 else []
        if isinstance(cols_sel, (list, tuple, np.ndarray, pd.Index)):
            cols.extend(list(cols_sel))
        elif cols_sel is not None:
            cols.append(cols_sel)
    return cols


def get_output_feature_names_from_preprocess(preprocess) -> List[str]:
    """
    Reconstitue les noms de features après préprocessing :
      - numériques : noms identiques (après StandardScaler)
      - catégorielles : noms One-Hot via l'encoder
    """
    # Récupère les colonnes brutes num / cat
    items = getattr(preprocess, "transformers_", None) or getattr(preprocess, "transformers", [])
    num_cols, cat_cols = [], []
    for name, transformer, cols_sel, *rest in items:
        if name == "num":
            num_cols = list(cols_sel)
        elif name == "cat":
            cat_cols = list(cols_sel)

    # Encoder catégoriel (fit dans l'artefact baseline)
    try:
        ohe = preprocess.named_transformers_["cat"].named_steps["encoder"]
    except Exception as e:
        raise RuntimeError(f"Encoder catégoriel indisponible dans le préprocesseur: {e}")

    # Noms OHE (compat >=1.0 et anciens)
    try:
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
    except AttributeError:
        cat_feature_names = ohe.get_feature_names(cat_cols)

    full_names = list(num_cols) + list(cat_feature_names)
    return [str(n) for n in full_names]


# =========================
# Chargement du modèle de prédiction (calibré ou non) + seuil
# =========================
def load_model_and_threshold() -> Tuple[Any, float, str]:
    chosen = "baseline"
    threshold = 0.5

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        chosen = meta.get("chosen_model", "baseline")
        threshold = float(meta.get("decision_threshold", {}).get("t_selected", 0.5))

    if chosen == "isotonic" and os.path.exists(CALIB_ISO_PATH):
        model_path = CALIB_ISO_PATH
    elif chosen == "sigmoid" and os.path.exists(CALIB_SIG_PATH):
        model_path = CALIB_SIG_PATH
    else:
        model_path = BASELINE_PATH
        chosen = "baseline"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Impossible de trouver le modèle à {model_path}. "
            f"Vérifie que le dossier '{ARTIFACT_DIR}' contient bien les fichiers .joblib."
        )

    model = load(model_path)
    return model, threshold, chosen


# =========================
# Chargement du modèle d'explication (toujours la baseline logistique fit)
# =========================
def load_explainer_pipeline() -> Any:
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"Impossible de trouver la pipeline baseline à {BASELINE_PATH}. "
            "Elle est nécessaire pour calculer les contributions locales."
        )
    base = load(BASELINE_PATH)  # pipeline fit (préprocess + LogisticRegression)
    # vérifications rapides
    pipe_inner = unwrap_pipeline(base)
    assert "preprocess" in pipe_inner.named_steps and "model" in pipe_inner.named_steps, \
        "L'artefact baseline ne ressemble pas à Pipeline([('preprocess'), ('model')])."
    return base


# ===== charge les deux =====
model_predict, DECISION_THRESHOLD, CHOSEN_NAME = load_model_and_threshold()
model_explain = load_explainer_pipeline()  # pipeline logistique FIT pour les contributions


# =========================
# Colonnes attendues (on s'appuie sur la baseline FIT)
# =========================
def expected_input_columns() -> List[str]:
    preprocess = get_preprocess_step(model_explain)
    cols = list_cols_from_column_transformer(preprocess)
    # Par sécurité, ne pas exiger l'ID si tu l'as exclu à l'entraînement
    cols = [c for c in cols if c != "SK_ID_CURR"]
    return cols


EXPECTED_COLS = expected_input_columns()


# =========================
# Schémas I/O (Pydantic)
# =========================
class PredictRequest(BaseModel):
    features: Dict[str, Any]  # {nom_colonne: valeur}


class PredictResponse(BaseModel):
    probability_default: float
    decision: str
    threshold: float
    missing_features: List[str]
    used_model: str


class Contribution(BaseModel):
    feature: str
    contribution: float  # contribution en log-odds


class ExplainResponse(BaseModel):
    probability_default: float
    decision: str
    threshold: float
    used_model: str
    bias: float
    top_contributions: List[Contribution]


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Credit Scoring API", version="0.2.0")


@app.get("/")
def root():
    return {"message": "Credit Scoring API is up. See /docs for interactive Swagger UI."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "used_model": CHOSEN_NAME,
        "threshold": DECISION_THRESHOLD,
        "expected_n_features": len(EXPECTED_COLS),
    }


@app.get("/expected_features")
def expected_features():
    return {"expected_features": EXPECTED_COLS}


def align_features(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aligne le dict reçu sur les colonnes attendues :
      - colonnes manquantes -> NaN
      - colonnes en trop   -> ignorées
    """
    missing = [c for c in EXPECTED_COLS if c not in payload]
    row = {c: payload.get(c, np.nan) for c in EXPECTED_COLS}
    df = pd.DataFrame([row], columns=EXPECTED_COLS)
    return df, missing


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not isinstance(req.features, dict) or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Le champ 'features' doit être un dictionnaire non vide.")

    X_input, missing = align_features(req.features)

    try:
        proba = float(model_predict.predict_proba(X_input)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {e}")

    decision = "refusé" if proba >= DECISION_THRESHOLD else "accordé"

    return PredictResponse(
        probability_default=proba,
        decision=decision,
        threshold=float(DECISION_THRESHOLD),
        missing_features=missing,
        used_model=CHOSEN_NAME,
    )


# =========================
# EXPLICATION LOCALE (sans SHAP) via contributions logistiques
# =========================
def compute_logreg_contributions(est: Any, X_df: pd.DataFrame) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calcule les contributions 'locales' pour UNE ligne en utilisant la pipeline baseline FIT :
      - transforme X via preprocess
      - contributions = coef * valeur_transformée (par feature)
      - renvoie (intercept, liste triée par |contribution|)
    """
    pipe_inner = unwrap_pipeline(est)
    preprocess = pipe_inner.named_steps["preprocess"]
    model_step = pipe_inner.named_steps["model"]  # LogisticRegression

    # Transforme la ligne
    X_trans = preprocess.transform(X_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Coefficients & intercept (classe 1)
    coefs = model_step.coef_.ravel()
    intercept = float(model_step.intercept_[0])

    # Noms de features transformées
    feat_names = get_output_feature_names_from_preprocess(preprocess)

    # Contributions (même ordre que X_trans)
    row = X_trans[0]
    contrib_pairs = list(zip(feat_names, row * coefs))
    contrib_pairs_sorted = sorted(contrib_pairs, key=lambda t: abs(t[1]), reverse=True)
    return intercept, contrib_pairs_sorted


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest, top_k: int = 8):
    """
    Explique la prédiction pour une observation :
      - probabilité (du modèle choisi : baseline ou calibré)
      - décision selon threshold
      - top_k contributions en log-odds issues de la baseline logistique FIT
    """
    if not isinstance(req.features, dict) or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Le champ 'features' doit être un dictionnaire non vide.")

    # Aligner les features
    X_input, _ = align_features(req.features)

    # Proba + décision (calibré si choisi)
    try:
        proba = float(model_predict.predict_proba(X_input)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {e}")
    decision = "refusé" if proba >= DECISION_THRESHOLD else "accordé"

    # Contributions locales via baseline logistique FIT
    try:
        bias, pairs_sorted = compute_logreg_contributions(model_explain, X_input)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explication indisponible (artefact baseline incompatible) : {e}"
        )

    top_k = max(1, int(top_k))
    top = [{"feature": f, "contribution": float(v)} for f, v in pairs_sorted[:top_k]]

    return ExplainResponse(
        probability_default=proba,
        decision=decision,
        threshold=float(DECISION_THRESHOLD),
        used_model=CHOSEN_NAME,
        bias=float(bias),
        top_contributions=top
    )
