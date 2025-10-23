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
# Configuration générale
# ------------------------------------------------------------------------------

app = FastAPI(title="Credit Scoring API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Codespaces : on ouvre à tout
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Chemins et caches
# ------------------------------------------------------------------------------

DATA_CSV = Path("data/application_train.csv")
ARTIFACTS_DIR = Path("artifacts")
REF_STATS_JSON = ARTIFACTS_DIR / "ref_stats.json"
GLOBAL_IMPORTANCE_CSV = ARTIFACTS_DIR / "global_importance.csv"
INTERP_SUMMARY_JSON = ARTIFACTS_DIR / "interpretability_summary.json"

_expected_columns_cache: Optional[List[str]] = None
_top_features_cache: Optional[List[str]] = None

# ------------------------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------------------------

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _endpoint_list() -> List[str]:
    """Retourne la liste des endpoints exposés."""
    return [r.path for r in app.routes if isinstance(r, APIRoute)]


# ------------------------------------------------------------------------------
# Colonnes attendues
# ------------------------------------------------------------------------------

def _load_expected_columns() -> List[str]:
    """
    Liste des colonnes d’entrée du modèle.
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
# Top features (les 10 plus importantes)
# ------------------------------------------------------------------------------

def _default_top_features() -> List[str]:
    """Fallback robuste (10 variables typiques Home Credit)."""
    return [
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


def _load_top_features_from_csv(path: Path) -> Optional[List[str]]:
    try:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "feature" in cols:
            fcol = df.columns[cols.index("feature")]
        else:
            fcol = df.columns[0]

        icandidates = ["importance", "gain", "weight", "abs_importance", "shap"]
        icol: Optional[str] = None
        for cand in icandidates:
            if cand in cols:
                icol = df.columns[cols.index(cand)]
                break

        if icol:
            df = df[[fcol, icol]].dropna()
            df = df.sort_values(by=icol, key=lambda s: s.abs(), ascending=False)
            feats = df[fcol].astype(str).tolist()
        else:
            feats = df[fcol].astype(str).tolist()

        feats = [f for f in feats if f and f.strip()]
        if feats:
            return feats[:10]
    except Exception:
        pass
    return None


def _load_top_features_from_json(path: Path) -> Optional[List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        gi = obj.get("global_importance") or obj.get("global_importances")
        if isinstance(gi, dict):
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


def _load_top_features() -> List[str]:
    """
    Renvoie la liste des 10 variables globalement les plus impactantes.
    Ordre : global_importance.csv > interpretability_summary.json > défaut.
    Ne renvoie jamais [].
    """
    global _top_features_cache
    if _top_features_cache is not None:
        return _top_features_cache

    # 1) CSV
    if GLOBAL_IMPORTANCE_CSV.exists():
        feats = _load_top_features_from_csv(GLOBAL_IMPORTANCE_CSV)
        if feats:
            _top_features_cache = feats
            return _top_features_cache

    # 2) JSON
    if INTERP_SUMMARY_JSON.exists():
        feats = _load_top_features_from_json(INTERP_SUMMARY_JSON)
        if feats:
            _top_features_cache = feats
            return _top_features_cache

    # 3) Fallback
    _top_features_cache = _default_top_features()
    return _top_features_cache


# ------------------------------------------------------------------------------
# Logique de prédiction (fallback simple)
# ------------------------------------------------------------------------------

def _fallback_probability(features: Dict[str, Any]) -> float:
    """
    Fallback heuristique : estime la probabilité de défaut
    à partir du ratio CREDIT / INCOME.
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


# ------------------------------------------------------------------------------
# Endpoints FastAPI
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


@app.get("/top_features")
def top_features() -> List[str]:
    return _load_top_features()


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
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
