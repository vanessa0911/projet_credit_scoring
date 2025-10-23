# ================================================================
#  api.py — API FastAPI robuste (sans secrets obligatoires)
#  Lit les artefacts locaux si disponibles, sinon dégrade proprement.
# ================================================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import os, json, pathlib

# ------------------------------------------------
# Configuration: répertoires / fichiers artefacts
# ------------------------------------------------
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
EXPECTED_COLUMNS_PATH = os.getenv("EXPECTED_COLUMNS_PATH", os.path.join(ARTIFACTS_DIR, "expected_columns.json"))
KEY_FEATURES_PATH = os.getenv("KEY_FEATURES_PATH", os.path.join(ARTIFACTS_DIR, "key_features.json"))
GLOBAL_IMPORTANCE_PATH = os.getenv("GLOBAL_IMPORTANCE_PATH", os.path.join(ARTIFACTS_DIR, "global_importance.csv"))

pathlib.Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

def _read_json_list(path: str) -> List[str]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                # tolère {"columns":[...]} ou {"features":[...]}
                for k in ("columns", "features"):
                    if k in data and isinstance(data[k], list):
                        return [str(x) for x in data[k]]
    except Exception:
        pass
    return []

def _read_global_importance(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if {"feature", "importance"}.issubset(df.columns):
                # cast
                df["feature"] = df["feature"].astype(str)
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
                df = df.dropna(subset=["importance"])
                return df
    except Exception:
        pass
    return None

EXPECTED_COLUMNS: List[str] = _read_json_list(EXPECTED_COLUMNS_PATH)
if not EXPECTED_COLUMNS:
    # fallback minimal si artefact absent
    EXPECTED_COLUMNS = [
        "gender", "age", "income", "loan_amount", "loan_duration",
        "employment_years", "credit_history", "housing",
    ]

KEY_FEATURES: List[str] = _read_json_list(KEY_FEATURES_PATH)
if not KEY_FEATURES:
    # fallback "métier" si artefact absent
    KEY_FEATURES = ["income", "loan_amount", "loan_duration", "credit_history", "age"]

GLOBAL_IMPORTANCE = _read_global_importance(GLOBAL_IMPORTANCE_PATH)

# ------------------------------------------------
# App FastAPI
# ------------------------------------------------
app = FastAPI(
    title="Credit Scoring API",
    description="API pour la prédiction et l’explicabilité du risque crédit.",
    version="1.1.0",
)

class Features(BaseModel):
    features: Dict[str, Any]

class Rows(BaseModel):
    rows: List[Dict[str, Any]]

# ------------------------------------------------
# Modèle (placeholder) — à remplacer par ton pipeline
# ------------------------------------------------
def dummy_model_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Simulation : probabilité de défaut ~ loan_amount / 100k + bruit.
    Remplace par (preprocessor -> model.predict_proba) quand tes artefacts sont prêts.
    """
    amt = pd.to_numeric(df.get("loan_amount", pd.Series([10000]*len(df))), errors="coerce").fillna(10000)
    base = 0.2 + 0.3 * (amt / 100000.0)
    noise = np.random.normal(0, 0.05, size=len(df))
    proba = np.clip(base + noise, 0, 1)
    return proba

# ------------------------------------------------
# Endpoints
# ------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API Credit Scoring opérationnelle."}

@app.get("/expected_columns")
def expected_columns():
    return EXPECTED_COLUMNS

@app.get("/key_features")
def key_features():
    return {"features": KEY_FEATURES}

@app.post("/predict")
def predict(payload: Features):
    try:
        x = pd.DataFrame([payload.features])
        proba = float(dummy_model_predict_proba(x)[0])
        y_hat = int(proba >= 0.5)
        return {"proba": proba, "y_hat": y_hat}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Rows):
    try:
        X = pd.DataFrame(payload.rows)
        pred_proba = dummy_model_predict_proba(X)
        y_hat = (pred_proba >= 0.5).astype(int)
        return {"pred_proba": pred_proba.tolist(), "y_hat": y_hat.tolist()}
    except Exception as e:
        return {"error": str(e)}

try:
    import shap  # noqa
    SHAP_OK = True
except Exception:
    SHAP_OK = False

@app.get("/global_importance")
def global_importance():
    """
    Importance globale à partir de artifacts/global_importance.csv si dispo,
    sinon fallback (aléatoire stable sur EXPECTED_COLUMNS).
    """
    try:
        if GLOBAL_IMPORTANCE is not None and not GLOBAL_IMPORTANCE.empty:
            return GLOBAL_IMPORTANCE.sort_values("importance", ascending=False).to_dict(orient="records")
        # fallback
        rng = np.random.default_rng(42)
        imp = rng.random(len(EXPECTED_COLUMNS))
        df = pd.DataFrame({"feature": EXPECTED_COLUMNS, "importance": imp})
        return df.sort_values("importance", ascending=False).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.post("/shap_local")
def shap_local(payload: Features):
    """
    Placeholder SHAP local. Retourne des valeurs simulées si SHAP installé,
    sinon message explicite. À brancher sur ton vrai modèle plus tard.
    """
    if not SHAP_OK:
        return {"detail": "SHAP non disponible côté serveur."}
    try:
        n = len(EXPECTED_COLUMNS)
        rng = np.random.default_rng()
        shap_values = rng.normal(0, 0.02, n).tolist()
        base_value = 0.2
        return {
            "feature_names": EXPECTED_COLUMNS,
            "shap_values": shap_values,
            "base_value": base_value,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
