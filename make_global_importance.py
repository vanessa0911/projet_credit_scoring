# make_global_importance.py
# Génère:
#   - artifacts/global_importance.csv  (colonnes: raw_feature, abs_importance)
#   - artifacts/interpretability_summary.json (top 20)
#
# Points clés:
# - Charge le modèle "champion" à partir de artifacts/metadata.json (fallback sur baseline)
# - Trouve le ColumnTransformer "preprocess" de façon robuste
# - Mappe correctement les noms transformés (catégoriels) -> noms de features d'origine
# - Gère arbres (feature_importances_) et linéaires (coef_). Sinon met 0.

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

ART_DIR   = os.path.join(".", "artifacts")
META_PATH = os.path.join(ART_DIR, "metadata.json")

CANDIDATES = {
    "isotonic": "model_calibrated_isotonic.joblib",
    "sigmoid":  "model_calibrated_sigmoid.joblib",
    "baseline": "model_baseline_logreg.joblib",
}

OUT_CSV  = os.path.join(ART_DIR, "global_importance.csv")
OUT_JSON = os.path.join(ART_DIR, "interpretability_summary.json")


# ---------- Utils ----------
def read_json_tolerant(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        import re
        txt2 = re.sub(r',\s*([}\]])', r'\1', txt)
        return json.loads(txt2)


def pick_model_path() -> str:
    meta = read_json_tolerant(META_PATH)
    key = str(meta.get("chosen_model", "") or "").lower()
    ordered = []
    if key in CANDIDATES:
        ordered.append(os.path.join(ART_DIR, CANDIDATES[key]))
    # fallbacks
    for k, fname in CANDIDATES.items():
        p = os.path.join(ART_DIR, fname)
        if p not in ordered:
            ordered.append(p)
    for p in ordered:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Aucun artefact modèle trouvé dans ./artifacts")


def find_preprocess(pipe: Any) -> Optional[Any]:
    """Retourne l'objet ColumnTransformer (ou compatible) qui expose get_feature_names_out()."""
    # 1) step 'preprocess' attendu
    try:
        if hasattr(pipe, "named_steps") and "preprocess" in pipe.named_steps:
            return pipe.named_steps["preprocess"]
    except Exception:
        pass
    # 2) chercher dans steps/named_steps le premier qui a get_feature_names_out
    for attr in ("named_steps", "steps"):
        if hasattr(pipe, attr):
            steps = getattr(pipe, attr)
            if isinstance(steps, dict):
                for step in steps.values():
                    if hasattr(step, "get_feature_names_out"):
                        return step
            elif isinstance(steps, list):
                for _, step in steps:
                    if hasattr(step, "get_feature_names_out"):
                        return step
    return None


def unwrap_estimator(obj: Any) -> Any:
    """Déballe un calibrateur/wrapper pour atteindre l'estimateur avec coef_/feature_importances_."""
    # sklearn CalibratedClassifierCV: base_estimator / estimator selon versions
    for attr in ("base_estimator", "estimator", "classifier"):
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            return getattr(obj, attr)
    return obj


def get_group_columns_from_ct(ct: Any) -> Tuple[List[str], List[str]]:
    """
    Essaie d’extraire les listes de colonnes numériques/catégorielles depuis un ColumnTransformer.
    Retourne (num_cols, cat_cols). Si inconnu, listes vides.
    """
    num_cols, cat_cols = [], []
    if hasattr(ct, "transformers_"):
        for name, transformer, cols in ct.transformers_:
            # cols peut être array-like ou slice
            if cols is None:
                continue
            try:
                cols_list = list(cols)
            except Exception:
                # slice ou autre → on abandonne proprement
                cols_list = []
            if isinstance(name, str) and name.startswith("num"):
                num_cols.extend(map(str, cols_list))
            elif isinstance(name, str) and name.startswith("cat"):
                cat_cols.extend(map(str, cols_list))
    return num_cols, cat_cols


def raw_feature_from_transformed(name: str, num_cols: List[str], cat_cols: List[str]) -> str:
    """
    Mappe un nom transformé (p.ex. 'cat__NAME_CONTRACT_TYPE_Cash loans')
    vers la feature d'origine ('NAME_CONTRACT_TYPE').

    Règles:
    - 'num__<col>' -> '<col>'
    - 'cat__<col>_<valeurOneHot>' -> '<col>'  (on matche sur le préfixe le plus long de cat_cols)
    - 'remainder__<col>' -> '<col>'
    - Sinon: retourne 'name' tel quel.
    """
    s = str(name)

    def strip_prefix(s: str, prefix: str) -> str:
        return s[len(prefix):] if s.startswith(prefix) else s

    # num__
    if s.startswith("num__"):
        return strip_prefix(s, "num__")
    # cat__
    if s.startswith("cat__"):
        rest = strip_prefix(s, "cat__")
        # Trouver le préfixe de 'rest' qui correspond à une colonne catégorielle connue
        # On prend le plus long match pour gérer les underscores dans les noms
        candidates = [c for c in cat_cols if rest.startswith(c)]
        if candidates:
            return max(candidates, key=len)
        # fallback: couper sur la DERNIÈRE occurrence de '_'
        if "_" in rest:
            return rest[:rest.rfind("_")]
        return rest
    # remainder__
    if s.startswith("remainder__"):
        return strip_prefix(s, "remainder__")
    return s


def compute_importance(pipe: Any) -> pd.DataFrame:
    """Retourne un DataFrame [raw_feature, abs_importance] trié décroissant."""
    ct = find_preprocess(pipe)
    if ct is None or not hasattr(ct, "get_feature_names_out"):
        raise RuntimeError("Impossible de trouver le preprocess (ColumnTransformer) avec get_feature_names_out().")

    try:
        feat_out = ct.get_feature_names_out()
    except Exception as e:
        raise RuntimeError(f"get_feature_names_out a échoué : {e}")

    # Estimateur final
    clf = None
    try:
        clf = pipe.named_steps.get("clf")  # type: ignore[attr-defined]
    except Exception:
        clf = None
    if clf is None:
        # fallback: chercher un step qui a coef_ ou feature_importances_
        for attr in ("named_steps", "steps"):
            if hasattr(pipe, attr):
                steps = getattr(pipe, attr)
                if isinstance(steps, dict):
                    for step in steps.values():
                        if hasattr(step, "coef_") or hasattr(step, "feature_importances_"):
                            clf = step
                            break
                elif isinstance(steps, list):
                    for _, step in steps:
                        if hasattr(step, "coef_") or hasattr(step, "feature_importances_"):
                            clf = step
                            break
    if clf is None:
        raise RuntimeError("Estimateur final introuvable (step 'clf' absent).")

    base_est = unwrap_estimator(clf)

    # Importance brute par feature transformée
    if hasattr(base_est, "feature_importances_"):  # arbres, boosting
        raw_importance = np.abs(np.asarray(base_est.feature_importances_, dtype=float))
    elif hasattr(base_est, "coef_"):               # modèles linéaires (logreg)
        raw_importance = np.abs(np.asarray(base_est.coef_, dtype=float).ravel())
    else:
        # Pas d'importance dispo -> suite en zéros
        raw_importance = np.zeros(len(feat_out), dtype=float)

    # Sécurité: aligner les longueurs
    m = min(len(feat_out), len(raw_importance))
    feat_out = np.asarray(feat_out, dtype=object)[:m]
    raw_importance = raw_importance[:m]

    # Mapping transformé -> brut (utilise les listes de colonnes du CT pour fiabiliser)
    num_cols, cat_cols = get_group_columns_from_ct(ct)
    rows = []
    for n, imp in zip(feat_out, raw_importance):
        base = raw_feature_from_transformed(str(n), num_cols, cat_cols)
        rows.append((base, float(imp)))

    gi = (
        pd.DataFrame(rows, columns=["raw_feature", "abs_importance"])
        .groupby("raw_feature", as_index=False)["abs_importance"].mean()
        .sort_values("abs_importance", ascending=False, ignore_index=True)
    )
    return gi


# ---------- Script principal ----------
def main():
    model_path = pick_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    pipe = load(model_path)

    gi = compute_importance(pipe)

    # Sauvegardes
    os.makedirs(ART_DIR, exist_ok=True)
    gi.to_csv(OUT_CSV, index=False, encoding="utf-8")

    summary = {
        "interpretability_method": "feature_importances_or_coef",
        "model_artifact": os.path.basename(model_path),
        "top_global_features": gi.head(20).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Importances globales générées :")
    print(f" - {OUT_CSV}")
    print(f" - {OUT_JSON}")


if __name__ == "__main__":
    main()
