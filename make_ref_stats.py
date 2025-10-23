# make_ref_stats.py
from __future__ import annotations
import os, json, argparse
import pandas as pd
import numpy as np

def percentiles(s: pd.Series):
    qs = [1,5,50,95,99]
    vals = np.nanpercentile(s.astype(float), qs)
    return dict(zip([f"p{q}" for q in qs], map(float, vals)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="CSV de référence (ex: application_train_features.csv)")
    ap.add_argument("--artifacts", type=str, default="artifacts", help="Dossier artefacts (output)")
    ap.add_argument("--expected", type=str, default=None, help="Chemin vers feature_names.npy (optionnel)")
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.expected and os.path.exists(args.expected):
        expected = np.load(args.expected, allow_pickle=True).tolist()
        df = df[[c for c in expected if c in df.columns]]  # ne garde que les colonnes attendues
    else:
        expected = list(df.columns)

    columns_meta = {}
    for c in expected:
        s = df[c]
        # heuristique rôle
        if pd.api.types.is_numeric_dtype(s):
            role = "numeric"
            meta = {
                "role": role,
                "missing_ratio": float(s.isna().mean()),
                "mean": float(np.nanmean(s)),
                "std": float(np.nanstd(s)),
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
            }
            meta.update(percentiles(s))
            # bornes UI (clamp sur [p1,p99])
            meta["ui_min"] = meta.get("p1", meta["min"])
            meta["ui_max"] = meta.get("p99", meta["max"])
        else:
            role = "categorical"
            vc = s.astype(str).value_counts(dropna=True).head(20)
            meta = {
                "role": role,
                "missing_ratio": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "top_values": vc.index.tolist(),
            }
        columns_meta[c] = meta

    ref = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_rows": int(len(df)),
        "columns": columns_meta,
    }

    out_path = os.path.join(args.artifacts, "ref_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
    print(f"[OK] Écrit: {out_path}")

if __name__ == "__main__":
    main()
