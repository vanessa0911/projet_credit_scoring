# api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# App + CORS (nécessaire pour l'appel depuis le dashboard Streamlit en Codespaces)
# ------------------------------------------------------------------------------
app = FastAPI(title="Credit Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en Codespaces on ouvre largement; à restreindre si besoin
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

_expected_columns_cache: Optional[List[str]] = None


# ------------------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------------------
def _load_expected_columns() -> List[str]:
    """
    Stratégie simple et robuste pour obtenir les colonnes d'entrée attendues :
    1) Si data/application_train.csv existe, on lit l'en-tête (nrows=0)
    2) Sinon, si artifacts/ref_stats.json existe, on tente de combiner les listes connues
    3) Sinon, on renvoie [] (le front gère ce cas)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) depuis le CSV d'entraînement
    if DATA_CSV.exists():
        try:
            df_head = pd.read_csv(DATA_CSV, nrows=0)
            _expected_columns_cache = list(df_head.columns)
            return _expected_columns_cache
        except Exception:
            pass

    # 2) depuis les artefacts récap
    if REF_STATS_JSON.exists():
        try:
            with open(REF_STATS_JSON, "r", encoding="utf-8") as f:
                ref = json.load(f)
            cols: List[str] = []
            for key in ("numerical_columns", "categorical_columns", "all_columns"):
                if isinstance(ref.get(key), list):
                    cols.extend(ref[key])
            # dédoublonne en conservant l'ordre
            _expected_columns_cache = list(dict.fromkeys(cols))
            return _expected_columns_cache
        except Exception:
            pass

    # 3) défaut
    _expected_columns_cache = []
    return _expected_columns_cache


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _fallback_probability(features: Dict[str, Any]) -> float:
    """
    Fallback déterministe pour permettre au front de fonctionner sans modèle.
    Heuristique simple basée sur ratio CREDIT/INCOME si disponible.
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
        # borne entre 0.01 et 0.99 pour éviter 0/1 stricts
        prob = max(0.01, min(0.99, 0.2 + 0.6 * (ratio / (ratio + 1.0))))
    else:
        prob = 0.5
    return float(prob)


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "name": "Credit Scoring API",
        "endpoints": [
            "/",
            "/health",
            "/expected_columns",
            "/predict",
            "/predict_proba_batch",
        ],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/expected_columns")
def expected_columns() -> List[str]:
    return _load_expected_columns()


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Corps attendu: {feature_name: value, ...}
    Réponse: {"probability": float, "decision": int, "threshold": float, ...}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object with feature:value pairs.")

    # TODO: brancher ici le vrai pipeline (preprocess + model.predict_proba)
    prob = _fallback_probability(payload)
    threshold = 0.5
    decision = int(prob >= threshold)

    return {
        "probability": prob,
        "decision": decision,
        "threshold": threshold,
        "used_columns_count": len(_load_expected_columns()),
    }


@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Corps attendu: {"instances": [ {feature: value, ...}, ... ]}
    Réponse: {"probabilities": [float, ...], "count": int}
    """
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Body must contain a list under 'instances'.")

    probs: List[float] = []
    for row in instances:
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail="Each instance must be a JSON object.")
        probs.append(_fallback_probability(row))

    return {"probabilities": probs, "count": len(probs)}


# ------------------------------------------------------------------------------
# Entrée locale (facultatif)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Permet un run direct: python api.py
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
