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
    allow_origins=["*"],  # Pour Codespaces : on laisse tout ouvert
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
# Fonctions utilitaires
# ------------------------------------------------------------------------------
def _load_expected_columns() -> List[str]:
    """
    Récupère la liste des colonnes d'entrée attendues.
    Ordre de priorité :
      1) en-tête du CSV si présent et non vide
      2) artefacts/ref_stats.json si non vide
      3) fallback minimal (3 colonnes)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) depuis le CSV
    if DATA_CSV.exists():
        try:
            df_head = pd.read_csv(DATA_CSV, nrows=0)
            cols = list(df_head.columns)
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
            # Dédoublonner et filtrer le vide
            cols = [c for c in dict.fromkeys(cols) if isinstance(c, str) and c.strip()]
            if cols:
                _expected_columns_cache = cols
                return _expected_columns_cache
        except Exception:
            pass

    # 3) fallback minimal
    _expected_columns_cache = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"]
    return _expected_columns_cache


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _fallback_probability(features: Dict[str, Any]) -> float:
    """
    Fallback déterministe pour démonstration.
    Heuristique simple basée sur le ratio CREDIT/INCOME.
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
    """Retourne la liste des endpoints exposés (pour /health et /)."""
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
    Corps attendu : {feature_name: value, ...}
    Réponse : {"probability": float, "decision": int, "threshold": float}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Le corps doit être un objet JSON {clé: valeur}.")

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
    Corps attendu : {"instances": [ {feature: value, ...}, ... ]}
    Réponse : {"probabilities": [float, ...], "count": int}
    """
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Le corps doit contenir une liste 'instances'.")

    probs: List[float] = []
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
