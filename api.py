# api.py
from __future__ import annotations
import os, json, hashlib, joblib
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_latest.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.npy")
REF_STATS_PATH = os.path.join(ARTIFACTS_DIR, "ref_stats.json")

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# CORS (ajuste la whitelist si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------- Chargement artefacts ------- #
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _hash_list(values: List[str]) -> str:
    m = hashlib.sha256()
    for v in values:
        m.update(v.encode("utf-8"))
    return m.hexdigest()[:12]

try:
    META = _load_json(META_PATH)
except FileNotFoundError:
    raise RuntimeError(f"metadata.json introuvable à {META_PATH}")

EXPECTED_COLUMNS: List[str] = META.get("expected_input_columns") or []
if not EXPECTED_COLUMNS and os.path.exists(FEATURES_PATH):
    EXPECTED_COLUMNS = np.load(FEATURES_PATH, allow_pickle=True).tolist()

if not EXPECTED_COLUMNS:
    raise RuntimeError("Aucune liste de colonnes attendues ('expected_input_columns').")

SCHEMA_HASH = _hash_list(EXPECTED_COLUMNS)
THRESHOLD: float = float(META.get("threshold", 0.5))
MODEL_VERSION: str = str(META.get("model_version", "unknown"))
TARGET_PREVALENCE: Optional[float] = META.get("target_prevalence")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modèle introuvable à {MODEL_PATH}")

MODEL = joblib.load(MODEL_PATH)

REF_STATS: Dict[str, Any] = {}
if os.path.exists(REF_STATS_PATH):
    REF_STATS = _load_json(REF_STATS_PATH)

# ------- Schémas Pydantic ------- #
class Record(BaseModel):
    data: Dict[str, Any] = Field(..., description="Une observation avec toutes les colonnes attendues")

class Records(BaseModel):
    records: List[Dict[str, Any]]

# ------- Normalisation d'entrée ------- #
def align_cast_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    # 1) aligne les colonnes attendues, renseigne None si manquantes
    row = {col: payload.get(col, None) for col in EXPECTED_COLUMNS}
    X = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

    # 2) cast léger basé sur ref_stats.json (si dispo)
    if REF_STATS and "columns" in REF_STATS:
        col_meta = REF_STATS["columns"]
        for col in EXPECTED_COLUMNS:
            role = (col_meta.get(col, {}).get("role") or "").lower()
            if role == "numeric":
                # essais de cast numérique non bloquant
                X[col] = pd.to_numeric(X[col], errors="coerce")
            elif role == "boolean":
                # map simples
                X[col] = X[col].map(
                    {True: 1, False: 0, "True": 1, "False": 0, "Y": 1, "N": 0}
                ).astype("float64")
            # categorical : on laisse tel quel (le pipeline encodera)

    return X

def align_cast_dataframe_batch(payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in payloads:
        rows.append({col: p.get(col, None) for col in EXPECTED_COLUMNS})
    X = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)

    if REF_STATS and "columns" in REF_STATS:
        col_meta = REF_STATS["columns"]
        for col in EXPECTED_COLUMNS:
            role = (col_meta.get(col, {}).get("role") or "").lower()
            if role == "numeric":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            elif role == "boolean":
                X[col] = X[col].map(
                    {True: 1, False: 0, "True": 1, "False": 0, "Y": 1, "N": 0}
                ).astype("float64")
    return X

# ------- Endpoints ------- #
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_COLUMNS),
        "schema_hash": SCHEMA_HASH,
        "target_prevalence": TARGET_PREVALENCE,
        "has_ref_stats": bool(REF_STATS),
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/expected_columns")
def expected_columns():
    return {"expected_columns": EXPECTED_COLUMNS, "count": len(EXPECTED_COLUMNS), "schema_hash": SCHEMA_HASH}

@app.post("/predict")
def predict(rec: Record):
    try:
        X = align_cast_dataframe(rec.data)
        proba = float(MODEL.predict_proba(X)[:, 1][0])
        decision = int(proba >= THRESHOLD)
        return {"probability": proba, "decision": decision, "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_proba_batch")
def predict_proba_batch(recs: Records):
    try:
        if not recs.records:
            raise ValueError("Payload vide.")
        X = align_cast_dataframe_batch(recs.records)
        proba = MODEL.predict_proba(X)[:, 1].astype(float).tolist()
        decisions = [int(p >= THRESHOLD) for p in proba]
        return {
            "probabilities": proba,
            "decisions": decisions,
            "count": len(proba),
            "model_version": MODEL_VERSION,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
