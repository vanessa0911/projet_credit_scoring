# make_ref_stats.py
"""
Génère des statistiques de référence (population d'entraînement) pour le dashboard :
- pour chaque variable numérique : count, mean, std, min, max, quantiles (p1..p99),
  et histogramme (edges + counts) à NBINS classes.
- ajoute des FEATURES INGENIÉRÉES si les colonnes sources existent (AGE_YEARS, PAYMENT_RATE, ...)

Sortie:
    artifacts/ref_stats.json

Usage :
    python make_ref_stats.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ====== CONFIG UTILISATEUR ======
# Chemin exact (Windows, avec espaces/accents) :
TRAIN_CSV = r"C:\Users\Maintenant Prêt\Desktop\Projet_credit_scoring\data\application_train.csv"

# Nombre de classes pour les histogrammes
NBINS = 40

# Colonnes à exclure explicitement (identifiants, cible…)
EXCLUDE_COLS = {"TARGET", "SK_ID_CURR"}

# ====== SORTIE ======
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
OUT_PATH = ARTIFACT_DIR / "ref_stats.json"


# ---------------- Utils ----------------
def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Division protégée: remplace /0 par NaN, cast en float."""
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom.replace({0: np.nan})
    return out.astype(float)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées si les colonnes sources existent.
    Ne lève pas d'erreur si des colonnes manquent.
    """
    out = df.copy()

    # 1) Âges/anciennetés (jours neg -> années positives)
    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = np.abs(pd.to_numeric(out["DAYS_BIRTH"], errors="coerce")) / 365.25
    if "DAYS_EMPLOYED" in out.columns:
        out["EMPLOY_YEARS"] = np.abs(pd.to_numeric(out["DAYS_EMPLOYED"], errors="coerce")) / 365.25
    if "DAYS_REGISTRATION" in out.columns:
        out["REG_YEARS"] = np.abs(pd.to_numeric(out["DAYS_REGISTRATION"], errors="coerce")) / 365.25

    # 2) Ratios de base
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(out.columns):
        out["CREDIT_INCOME_RATIO"] = safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(out.columns):
        out["ANNUITY_INCOME_RATIO"] = safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(out.columns):
        # RATE standard Home Credit
        out["PAYMENT_RATE"] = safe_div(out["AMT_ANNUITY"], out["AMT_CREDIT"])
        # Durée approximative (nombre de paiements) = crédit / annuité
        out["CREDIT_TERM_MONTHS"] = safe_div(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"} <= set(out.columns):
        out["CREDIT_GOODS_RATIO"] = safe_div(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])

    # 3) Composites EXT_SOURCE
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        tmp = out[ext_cols].apply(pd.to_numeric, errors="coerce")
        out["EXT_SOURCES_MEAN"] = tmp.mean(axis=1)
        out["EXT_SOURCES_SUM"] = tmp.sum(axis=1)
        out["EXT_SOURCES_NA"] = tmp.isna().sum(axis=1).astype(float)

    # 4) Démographie ménage
    if "AMT_INCOME_TOTAL" in out.columns:
        denom = 1.0
        if "CNT_FAM_MEMBERS" in out.columns:
            denom = pd.to_numeric(out["CNT_FAM_MEMBERS"], errors="coerce").fillna(0) + 1.0
        out["INCOME_PER_PERSON"] = safe_div(out["AMT_INCOME_TOTAL"], denom)

    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"} <= set(out.columns):
        out["CHILDREN_RATIO"] = safe_div(
            pd.to_numeric(out["CNT_CHILDREN"], errors="coerce"),
            (pd.to_numeric(out["CNT_FAM_MEMBERS"], errors="coerce") + 1.0),
        )

    # 5) Docs
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["DOC_COUNT"] = pd.to_numeric(out[doc_cols], errors="coerce").fillna(0).sum(axis=1).astype(float)

    # 6) Bool car/realty
    if "OWN_CAR" in out.columns:
        out["OWN_CAR_BOOL"] = (out["OWN_CAR"].astype(str).str.upper() == "Y").astype(float)
    if "OWN_REALTY" in out.columns:
        out["OWN_REALTY_BOOL"] = (out["OWN_REALTY"].astype(str).str.upper() == "Y").astype(float)

    # 7) Rapports âge/emploi
    if {"AGE_YEARS", "EMPLOY_YEARS"} <= set(out.columns):
        out["EMPLOY_TO_AGE_RATIO"] = safe_div(out["EMPLOY_YEARS"], out["AGE_YEARS"])

    # 8) Nombre de NaN par ligne (utile pour qualitatif cohorte)
    out["MISSING_COUNT_ROW"] = out.apply(lambda r: r.isna().sum(), axis=1).astype(float)

    return out


def compute_numeric_stats(series: pd.Series, nbins: int = NBINS) -> dict:
    """
    Calcule les stats pour une série numérique :
      - count, mean, std, min, max
      - quantiles p1, p5, p10, p25, p50, p75, p90, p95, p99
      - histogramme : edges (bornes) & counts
    Retourne {} si la série est vide après nettoyage.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    qs = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    q_vals = np.percentile(s.values, qs)
    counts, edges = np.histogram(s.values, bins=nbins)
    stats = {
        "type": "numeric",
        "count": int(s.shape[0]),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "quantiles": {f"p{int(p)}": float(v) for p, v in zip(qs, q_vals)},
        "hist": {
            "edges": [float(x) for x in edges],   # len(edges) = nbins + 1
            "counts": [int(x) for x in counts],   # len(counts) = nbins
        },
    }
    return stats


# ---------------- Main ----------------
def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Fichier introuvable : {TRAIN_CSV}\n"
            f"Vérifie le chemin, ou copie application_train.csv dans ce dossier."
        )

    print(f"Lecture du train : {TRAIN_CSV}")
    # Encodage: UTF-8 d'abord, fallback cp1252 si besoin
    try:
        df = pd.read_csv(TRAIN_CSV, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(TRAIN_CSV, low_memory=False, encoding="cp1252")

    # Retire la cible si présente
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    # Ajoute des features dérivées (si possible)
    df = add_engineered_features(df)

    # Colonnes numériques (pandas)
    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    # Retire les colonnes explicitement exclues
    num_cols = [c for c in num_cols if c not in EXCLUDE_COLS]

    print(f"Colonnes numériques retenues : {len(num_cols)}")

    features_stats: Dict[str, dict] = {}
    for col in num_cols:
        stats = compute_numeric_stats(df[col], nbins=NBINS)
        if stats:
            features_stats[col] = stats

    out = {
        "source": TRAIN_CSV,
        "n_rows": int(df.shape[0]),
        "n_features": len(features_stats),
        "features": features_stats,  # dict : { feature_name: {stats...}, ... }
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"OK ✅  Stats sauvegardées dans: {OUT_PATH.resolve()}")
    print(f"Features numériques: {len(features_stats)}")
    if len(features_stats) == 0:
        print("⚠️ Aucune feature numérique utile détectée. Vérifie le CSV.")


if __name__ == "__main__":
    main()
