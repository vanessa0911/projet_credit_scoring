# api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

# ------------------------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="Credit Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Codespaces: on ouvre largement; à restreindre si besoin
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
    Récupère la liste des colonnes d'entrée attendues.
    Ordre de priorité:
      1) en-tête du CSV d'entraînement (si présent)
      2) artefacts/ref_stats.json (si présent)
      3) fallback minimal (fonctionne sans dataset)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) depuis le CSV (le plus fiable si présent)
    if DATA_CSV.exists():
        try:
            df_head = pd.read_csv(DATA_CSV, nrows=0)
            _expected_columns_cache = list(df_head.columns)
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
                if isinstance(ref.get(key), list):
                    cols.extend(ref[key])
            # dédoublonne en conservant l'ordre
            _expected_columns_cache = list(dict.fromkeys(cols))
            return _expected_columns_cache
        except Exception:
            pass

    # 3) fallback minimal pour que le dashboard soit exploitable sans dataset
    _expected_columns_cache = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"]
    return _expected_columns_cache


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _fallback_probability(features: Dict[str, Any]) -> float:
    """
    Fallback déterministe pour dev/démo (sans modèle).
    Heuristique simple basée sur le ratio CREDIT/INCOME si disponible.
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


def _endpoint_list() -> List[str]:
    return [r.path for r in app.routes if isinstance(r, APIRoute)]


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "name": "Credit Scoring API",
        "endpoints": _endpoint_list(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "name": "Credit Scoring API",
        "endpoints": _endpoint_list(),
    }


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
# Entrée directe (facultatif)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
