# api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# App + CORS (nécessaire pour que le dashboard Streamlit atteigne l’API depuis le navigateur)
# ------------------------------------------------------------------------------
app = FastAPI(title="Credit Scoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en Codespaces on autorise tout : tu pourras restreindre ensuite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------------------
DATA_CSV = Path("data/application_train.csv")
ARTIFACTS_DIR = Path("artifacts")
REF_STATS_JSON = ARTIFACTS_DIR / "ref_stats.json"

_expected_columns_cache: Optional[List[str]] = None


def _load_expected_columns() -> List[str]:
    """
    Stratégie pour dev rapide :
    1) si data/application_train.csv est présent, on lit l'en-tête
    2) sinon si artifacts/ref_stats.json existe, on tente d'en déduire les colonnes
    3) sinon on renvoie une liste vide (le front gérera)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) CSV d’entraînement (le plus fiable)
    if DATA_CSV.exists():
        try:
            df_head = pd.read_csv(DATA_CSV, nrows=0)
            _expected_columns_cache = list(df_head.columns)
            return _expected_columns_cache
        except Exception:
            pass

    # 2) Artefacts (optionnel)
    if REF_STATS_JSON.exists():
        try:
            with open(REF_STATS_JSON, "r", encoding="utf-8") as f:
                ref = json.load(f)
            # essaie des clés courantes si elles existent
            cols = []
            for key in ("numerical_columns", "categorical_columns", "all_columns"):
                if key in ref and isinstance(ref[key], list):
                    cols.extend(ref[key])
            # dédoublonne tout en gardant l’ordre
            _expected_columns_cache = list(dict.fromkeys(cols))
            return _expected_columns_cache
        except Exception:
            pass

    # 3) par défaut : aucune (le front affichera un message)
    _expected_columns_cache = []
    return _expected_columns_cache


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/expected_columns")
def expected_columns() -> List[str]:
    cols = _load_expected_columns()
    return cols


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint minimaliste.
    - Si aucun modèle n’est dispo, renvoie une proba heuristique stable (fallback),
      pour ne pas casser le dashboard pendant l’intégration.
    - Le front s’attend à recevoir: {"probability": float, "decision": int}
    """
    cols = _load_expected_columns()

    # Heuristique simple (fallback) : si variables clés présentes on calcule une proba naïve
    # Tu pourras brancher ici ton vrai modèle quand il sera prêt.
    # Exemple d’heuristique : ratio CREDIT/INCOME si colonnes existent
    credit = None
    income = None
    # essaie quelques noms courants du dataset Home Credit
    for k in payload.keys():
        lk = k.lower()
        if credit is None and ("credit" in lk or "amt_credit" in lk):
            credit = _safe_float(payload[k], None)
        if income is None and ("income" in lk or "amt_income_total" in lk):
            income = _safe_float(payload[k], None)

    if credit is not None and income is not None and income > 0:
        ratio = credit / income
        # borne la proba entre 0.01 et 0.99
        prob = max(0.01, min(0.99, 0.2 + 0.6 * (ratio / (ratio + 1.0))))
    else:
        # fallback constant si pas de features utilisables
        prob = 0.5

    threshold = 0.5
    decision = int(prob >= threshold)

    return {
        "probability": float(prob),
        "decision": decision,
        "threshold": threshold,
        "used_columns_count": len(cols),
    }


@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attendu par le mode batch du dashboard.
    payload: {"instances": [ {feature: value, ...}, ... ]}
    """
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Body must contain a list under 'instances'.")

    results: List[float] = []
    for row in instances:
        results.append(predict(row)["probability"])  # réutilise la logique fallback

    return {"probabilities": results, "count": len(results)}
