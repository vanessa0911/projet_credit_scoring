#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère des statistiques de référence pour le dashboard Streamlit.
- Lit data/application_train.csv (modifiable via TRAIN_CSV)
- Calcule des stats numériques et catégorielles
- Écrit artifacts/ref_stats.json

Exécution :
    python make_ref_stats.py
"""

from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
from typing import Dict, Any

# --- chemins (chemins relatifs au repo) ---
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TRAIN_CSV = DATA_DIR / "application_train.csv"   # adapte si besoin

# --- colonnes utiles (optionnel) ---
# Si tu as une cible (target) dans ton CSV, renseigne son nom ici (sinon laisse None)
TARGET_COL = None  # ex: "TARGET"

def _is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _safe_float(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def compute_reference_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "numeric": {},
        "categorical": {},
        "columns": list(df.columns),
        "target_col": TARGET_COL,
        "note": "Stats générées pour l’affichage dans Streamlit (comparaison population, contrôles de qualité)."
    }

    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())

        if _is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            stats["numeric"][col] = {
                "count": int(desc.get("count", 0)),
                "missing_rate": missing_rate,
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "p05": _safe_float(desc.get("5%")),
                "p25": _safe_float(desc.get("25%")),
                "p50": _safe_float(desc.get("50%")),
                "p75": _safe_float(desc.get("75%")),
                "p95": _safe_float(desc.get("95%")),
                "max": _safe_float(desc.get("max")),
            }
        else:
            # Pour les catégorielles : top catégories + fréquence
            vc = s.astype("string").value_counts(dropna=True)
            top = vc.head(5)
            uniques = int(s.nunique(dropna=True))
            top_items = [
                {"value": str(idx), "count": int(cnt), "freq": float(cnt / max(1, len(s.dropna())))}
                for idx, cnt in top.items()
            ]
            stats["categorical"][col] = {
                "missing_rate": missing_rate,
                "unique_count": uniques,
                "top_values": top_items,
            }

    # Option : si une cible existe et est numérique/catégorielle, on ajoute un petit récap
    if TARGET_COL and TARGET_COL in df.columns:
        ts = df[TARGET_COL]
        if _is_numeric_dtype(ts):
            stats["target_summary"] = {
                "type": "numeric",
                "mean": _safe_float(ts.mean()),
                "std": _safe_float(ts.std()),
                "missing_rate": float(ts.isna().mean()),
            }
        else:
            tv = ts.astype("string").value_counts(dropna=True)
            stats["target_summary"] = {
                "type": "categorical",
                "distribution": [
                    {"value": str(k), "count": int(v), "freq": float(v / max(1, len(ts.dropna())))}
                    for k, v in tv.items()
                ],
                "missing_rate": float(ts.isna().mean()),
            }

    return stats

def main() -> None:
    # sécurités & création de dossiers
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {TRAIN_CSV.resolve()}.\n"
            "Place le dataset dans data/application_train.csv "
            "(ou modifie TRAIN_CSV dans make_ref_stats.py)."
        )

    # lecture rapide et sûre
    # low_memory=False pour éviter les dtypes incohérents sur gros CSV
    df = pd.read_csv(TRAIN_CSV, low_memory=False)

    # nettoyage simple : harmoniser colonnes (optionnel)
    # df.columns = [c.strip() for c in df.columns]

    stats = compute_reference_stats(df)

    out_path = ARTIFACTS_DIR / "ref_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Statistiques écrites → {out_path.as_posix()}")
    print(f"Colonnes détectées ({len(stats['columns'])}) : {', '.join(stats['columns'][:10])}"
          + ("..." if len(stats['columns']) > 10 else ""))

if __name__ == "__main__":
    main()
