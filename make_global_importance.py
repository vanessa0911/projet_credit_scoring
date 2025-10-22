#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Produit les importances globales des features à partir du modèle entraîné.
- Cherche un modèle dans artifacts/ (ex: model_*.joblib)
- Cherche feature_names.npy (ou essaie de les déduire)
- Calcule l'importance selon le type de modèle :
    * Tree-based (feature_importances_)
    * Linéaire (coefficients absolus)
- Écrit :
    - artifacts/global_importance.csv (colonnes: feature, importance)
    - artifacts/interpretability_summary.json (top-k)

Exécution :
    python make_global_importance.py
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Optional

import joblib

ARTIFACTS_DIR = Path("artifacts")
GLOBAL_CSV = ARTIFACTS_DIR / "global_importance.csv"
SUMMARY_JSON = ARTIFACTS_DIR / "interpretability_summary.json"

# Option : si aucune liste de features n'est trouvée, on peut en donner une par défaut
FALLBACK_FEATURES: Optional[List[str]] = None  # ex: ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE", ...]

def _find_model_file() -> Path:
    """
    On prend le *premier* fichier model_*.joblib trouvé dans artifacts/.
    Adapte au besoin si tu as plusieurs modèles (XGB, LGBM, …).
    """
    candidates = sorted(ARTIFACTS_DIR.glob("model_*.joblib"))
    if not candidates:
        # fallback possible : "model.joblib"
        default = ARTIFACTS_DIR / "model.joblib"
        if default.exists():
            return default
        raise FileNotFoundError(
            "Aucun modèle trouvé dans artifacts/ (attendu: model_*.joblib ou model.joblib). "
            "Vérifie tes artefacts (Git LFS ?)."
        )
    return candidates[0]

def _load_feature_names() -> List[str]:
    """
    Priorité : artifacts/feature_names.npy (np.save) ; sinon essai depuis metadata.json ; sinon fallback.
    """
    npy = ARTIFACTS_DIR / "feature_names.npy"
    if npy.exists():
        arr = np.load(npy, allow_pickle=True)
        return [str(x) for x in list(arr)]

    meta = ARTIFACTS_DIR / "metadata.json"
    if meta.exists():
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            cols = meta_obj.get("expected_input_columns") or meta_obj.get("feature_names")
            if cols and isinstance(cols, list) and all(isinstance(c, (str, int)) for c in cols):
                return [str(c) for c in cols]
        except Exception:
            pass

    if FALLBACK_FEATURES is not None:
        return FALLBACK_FEATURES

    raise FileNotFoundError(
        "Impossible de déterminer la liste des features. "
        "Attendu: artifacts/feature_names.npy ou metadata.json(expected_input_columns)."
    )

def _coef_importance(model: Any) -> np.ndarray:
    """
    Importance pour modèles linéaires (LogisticRegression, LinearSVC…)
    - Valeurs absolues des coefficients (multi-classes -> moyenne).
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise AttributeError("Le modèle n'a pas d'attribut coef_.")
    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return np.mean(np.abs(coef), axis=0)

def _tree_importance(model: Any) -> np.ndarray:
    """
    Importance native pour modèles d'arbres (RandomForest, XGBoost/LightGBM sklearn API…)
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise AttributeError("Le modèle n'a pas d'attribut feature_importances_.")
    return np.asarray(imp)

def _norm(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return np.zeros_like(x)
    return x / s

def compute_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Détecte automatiquement la méthode d’importance à utiliser.
    """
    importance: np.ndarray
    try:
        importance = _tree_importance(model)
    except AttributeError:
        try:
            importance = _coef_importance(model)
        except AttributeError as e:
            raise RuntimeError(
                "Impossible de calculer l'importance : le modèle n'est ni 'tree-based' "
                "ni linéaire avec coef_."
            ) from e

    if importance.shape[0] != len(feature_names):
        raise ValueError(
            f"Incohérence tailles : {importance.shape[0]} importances vs {len(feature_names)} features."
        )

    imp_norm = _norm(importance)
    df = pd.DataFrame({"feature": feature_names, "importance": imp_norm})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = _find_model_file()
    model = joblib.load(model_path)

    feature_names = _load_feature_names()
    df_imp = compute_importance(model, feature_names)
    df_imp.to_csv(GLOBAL_CSV, index=False, encoding="utf-8")

    # Petit résumé JSON (top 15)
    top_k = 15 if df_imp.shape[0] >= 15 else df_imp.shape[0]
    summary = {
        "model_file": model_path.name,
        "n_features": int(df_imp.shape[0]),
        "top_k": top_k,
        "top_features": [
            {"feature": str(row["feature"]), "importance": float(row["importance"])}
            for _, row in df_imp.head(top_k).iterrows()
        ],
        "note": (
            "Importance normalisée (somme=1). Pour modèles linéaires : |coef| moyen. "
            "Pour modèles d’arbres : feature_importances_."
        ),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Importances écrites → {GLOBAL_CSV.as_posix()}")
    print(f"[OK] Résumé écrit → {SUMMARY_JSON.as_posi
