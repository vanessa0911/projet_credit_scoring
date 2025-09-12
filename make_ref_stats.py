# make_ref_stats.py
"""
Génère des statistiques de référence (population d'entraînement) pour le dashboard :
- pour chaque variable numérique : count, mean, std, min, max, quantiles (p1..p99),
  et histogramme (edges + counts) à NBINS classes.
Sauvegarde en JSON : artifacts/ref_stats.json

Usage :
    python make_ref_stats.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ====== CONFIG UTILISATEUR ======
# Chemin exact fourni par toi (Windows, avec espaces/accents) :
TRAIN_CSV = r"C:\Users\Maintenant Prêt\Desktop\Projet_credit_scoring\data\application_train.csv"

# Nombre de classes pour les histogrammes
NBINS = 40

# Colonnes à exclure explicitement (identifiants, cible…)
EXCLUDE_COLS = {"TARGET", "SK_ID_CURR"}

# ====== SORTIE ======
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
OUT_PATH = ARTIFACT_DIR / "ref_stats.json"


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

    # quantiles
    qs = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    q_vals = np.percentile(s.values, qs)

    # histogramme
    counts, edges = np.histogram(s.values, bins=nbins)

    stats = {
        "type": "numeric",
        "count": int(s.shape[0]),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "quantiles": {
            f"p{int(p)}": float(v) for p, v in zip(qs, q_vals)
        },
        "hist": {
            "edges": [float(x) for x in edges],     # len(edges) = nbins + 1
            "counts": [int(x) for x in counts],     # len(counts) = nbins
        },
    }
    return stats


def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Fichier introuvable : {TRAIN_CSV}\n"
            f"Vérifie le chemin, ou copie application_train.csv dans ce dossier."
        )

    print(f"Lecture du train : {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV, low_memory=False)

    # On enlève la cible si présente
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    # Colonnes numériques (pandas)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Retire les colonnes explicitement exclues
    num_cols = [c for c in num_cols if c not in EXCLUDE_COLS]

    print(f"Colonnes numériques retenues : {len(num_cols)}")

    features_stats = {}
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
