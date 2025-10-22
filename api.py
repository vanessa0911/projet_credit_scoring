from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# Config répertoire artefacts
# =========================
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"           # adapte si ton modèle a un autre nom
EXPECTED_COLS_PATH = ARTIFACTS_DIR / "expected_columns.json"  # optionnel
REF_STATS_PATH = ARTIFACTS_DIR / "ref_stats.json"     # optionnel
GLOBAL_IMP_PATH = ARTIFACTS_DIR / "global_importance.csv"  # optionnel

# =========================
# FastAPI init + CORS
# =========================
app = FastAPI(title="Credit Scoring API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

# CORS permissif pour simplifier les tests (tu peux restreindre à l’URL du Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ex: ["https://*.githubpreview.dev"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Modèles d’entrées/sorties
# =========================
class PredictItem(BaseModel):
    # Utilise les noms de colonnes réels de ton modèle; ci-dessous un fallback minimal
    AMT_INCOME_TOTAL: Optional[float] = Field(None, description="Revenu annuel")
    AMT_CREDIT: Optional[float] = Field(None, description="Montant du crédit demandé")
    DAYS_BIRTH: Optional[float] = Field(None, description="Âge en jours négatifs (HomeCredit) ou âge transformé")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    proba: float
    decision: int
    threshold: float
    missing_columns: List[str] = []

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    expected_columns: List[str]

# =========================
# Chargement paresseux des artefacts
# =========================
_model = None
_expected_columns_cache: Optional[List[str]] = None

def load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable à {MODEL_PATH}. "
            "Entraîne/pose le fichier (joblib) ou ajuste MODEL_PATH dans api.py."
        )
    _model = joblib.load(MODEL_PATH)
    return _model

def get_expected_columns() -> List[str]:
    """
    Priorité :
    1) artifacts/expected_columns.json  (liste de noms)
    2) attribut .feature_names_in_ du modèle (scikit-learn >=1.0)
    3) Fallback minimal (à adapter selon ton projet)
    """
    global _expected_columns_cache
    if _expected_columns_cache is not None:
        return _expected_columns_cache

    # 1) Fichier JSON
    if EXPECTED_COLS_PATH.exists():
        try:
            _expected_columns_cache = json.loads(EXPECTED_COLS_PATH.read_text(encoding="utf-8"))
            if not isinstance(_expected_columns_cache, list):
                raise ValueError("expected_columns.json doit contenir une liste de chaînes.")
            _expected_columns_cache = [str(c) for c in _expected_columns_cache]
            return _expected_columns_cache
        except Exception as e:
            # On log/continue vers les autres méthodes
            print(f"[WARN] Lecture {EXPECTED_COLS_PATH} impossible: {e}")

    # 2) Depuis le modèle
    try:
        model = load_model()
        if hasattr(model, "feature_names_in_"):
            _expected_columns_cache = [str(c) for c in list(model.feature_names_in_)]
            return _expected_columns_cache
    except Exception as e:
        print(f"[WARN] Impossible d'inférer les colonnes depuis le modèle: {e}")

    # 3) Fallback : adapter aux colonnes de ton dashboard
    _expected_columns_cache = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
    return _expected_columns_cache

# =========================
# Utilitaires
