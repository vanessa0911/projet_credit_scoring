# ================================================================
#  api.py — API FastAPI complète pour le projet de Credit Scoring
# ================================================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# ================================================================
#  SECTION 1 — Configuration de base
# ================================================================

app = FastAPI(
    title="Credit Scoring API",
    description="API pour la prédiction et l’explicabilité du risque crédit.",
    version="1.0.0",
)

# 🔹 Simulation : colonnes attendues et variables clés (à ajuster selon ton dataset)
EXPECTED_COLUMNS: List[str] = [
    "gender",
    "age",
    "income",
    "loan_amount",
    "loan_duration",
    "employment_years",
    "credit_history",
    "housing",
    "marital_status",
    "dependents",
    "purpose",
    "savings_account",
    "checking_account",
    "existing_loans_count",
    "job_type",
    "foreign_worker",
]

KEY_FEATURES: List[str] = [
    "income",
    "loan_amount",
    "loan_duration",
    "credit_history",
    "age",
]

# ================================================================
#  SECTION 2 — Modèles Pydantic pour les entrées
# ================================================================

class Features(BaseModel):
    features: Dict[str, Any]

class Rows(BaseModel):
    rows: List[Dict[str, Any]]

# ================================================================
#  SECTION 3 — Placeholders de modèle (à remplacer par ton pipeline)
# ================================================================

# Ici, tu pourras charger ton vrai modèle et ton preprocess
# Exemple :
# import joblib
# model = joblib.load("artifacts/model.pkl")
# preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Pour l’instant, on simule un modèle simple :
def dummy_model_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Simulation : renvoie une probabilité de défaut basée sur un mix aléatoire
    pondéré par certaines variables.
    """
    base = 0.2 + 0.3 * (df.get("loan_amount", 10000) / 100000)
    noise = np.random.normal(0, 0.05, size=len(df))
    proba = np.clip(base + noise, 0, 1)
    return proba

# ================================================================
#  SECTION 4 — Endpoints
# ================================================================

@app.get("/")
def root():
    """Endpoint de santé."""
    return {"status": "ok", "message": "API Credit Scoring opérationnelle."}


@app.get("/expected_columns")
def expected_columns():
    """Retourne la liste des colonnes attendues."""
    return EXPECTED_COLUMNS


@app.get("/key_features")
def key_features():
    """Retourne la liste des variables clés identifiées (métier ou par importance)."""
    return {"features": KEY_FEATURES}


@app.post("/predict")
def predict(payload: Features):
    """
    Endpoint de prédiction pour un seul dossier.
    Renvoie la probabilité de défaut et la classe prédite.
    """
    try:
        # Conversion en DataFrame
        x = pd.DataFrame([payload.features])

        # Appel du modèle (ici simulé)
        proba = float(dummy_model_predict_proba(x)[0])

        # Décision binaire avec seuil 0.5
        y_hat = int(proba >= 0.5)

        return {"proba": proba, "y_hat": y_hat}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Rows):
    """
    Endpoint de scoring en lot (batch CSV).
    Renvoie la probabilité et la prédiction pour chaque ligne.
    """
    try:
        X = pd.DataFrame(payload.rows)
        pred_proba = dummy_model_predict_proba(X)
        y_hat = (pred_proba >= 0.5).astype(int)

        return {
            "pred_proba": pred_proba.tolist(),
            "y_hat": y_hat.tolist(),
        }

    except Exception as e:
        return {"error": str(e)}


# ================================================================
#  SECTION 5 — Explicabilité (placeholders SHAP)
# ================================================================

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False


@app.get("/global_importance")
def global_importance():
    """
    Importance globale (SHAP moyen ou fallback).
    Pour l’instant, renvoie des valeurs simulées.
    """
    try:
        importance = np.random.rand(len(EXPECTED_COLUMNS))
        df = pd.DataFrame({"feature": EXPECTED_COLUMNS, "importance": importance})
        df = df.sort_values("importance", ascending=False)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.post("/shap_local")
def shap_local(payload: Features):
    """
    Explication locale d’un dossier.
    Pour l’instant, renvoie des valeurs SHAP simulées.
    """
    if not SHAP_OK:
        return {"detail": "SHAP non disponible côté serveur."}

    try:
        x = pd.DataFrame([payload.features])
        n_features = len(EXPECTED_COLUMNS)
        shap_values = np.random.normal(0, 0.02, n_features).tolist()
        base_value = 0.2
        return {
            "feature_names": EXPECTED_COLUMNS,
            "shap_values": shap_values,
            "base_value": base_value,
        }

    except Exception as e:
        return {"error": str(e)}


# ================================================================
#  SECTION 6 — Lancement local (optionnel)
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
