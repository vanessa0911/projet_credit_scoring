# make_global_importance.py
from __future__ import annotations
import os, argparse, json, joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_val", type=str, required=True, help="CSV de validation (features alignées)")
    ap.add_argument("--y_val", type=str, required=True, help="CSV/NPY/Parquet de cibles (colonne unique 'target')")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--model", type=str, default="artifacts/model_latest.joblib")
    ap.add_argument("--n_repeats", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    MODEL = joblib.load(args.model)
    # charge X
    X_val = pd.read_csv(args.X_val)
    # charge y
    if args.y_val.endswith(".npy"):
        y_val = np.load(args.y_val)
    else:
        yv = pd.read_csv(args.y_val)
        y_col = yv.columns[0] if yv.shape[1] == 1 else "target"
        y_val = yv[y_col].values

    # baseline AUC
    proba = MODEL.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)

    # permutation importance
    pi = permutation_importance(MODEL, X_val, y_val, n_repeats=args.n_repeats, scoring="roc_auc", random_state=42)
    imp = pd.DataFrame({
        "feature": X_val.columns,
        "importance": pi.importances_mean,
        "importance_std": pi.importances_std,
        "method": "permutation"
    }).sort_values("importance", ascending=False)

    out_csv = os.path.join(args.artifacts, "global_importance.csv")
    imp.to_csv(out_csv, index=False)
    print(f"[OK] Écrit: {out_csv} | baseline AUC={auc:.4f}")

    # résumé interprétabilité facultatif
    explanations = {}
    for f in imp.head(15)["feature"]:
        explanations[f] = f"Des valeurs élevées/faibles de **{f}** influencent significativement le risque (voir analyse)."

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "val_auc": float(auc),
        "top_features": imp.head(30)["feature"].tolist(),
        "explanations": explanations
    }
    out_json = os.path.join(args.artifacts, "interpretability_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Écrit: {out_json}")

if __name__ == "__main__":
    main()
