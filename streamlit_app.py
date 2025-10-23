# streamlit_app.py
# =========================================================
# Prêt à dépenser — Scoring & Explicabilité (compat API v0)
# =========================================================

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# (Optionnels) — l'app tourne même si ces libs ne sont pas installées
try:
    import plotly.graph_objects as go  # type: ignore
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from sklearn.metrics import confusion_matrix  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# -----------------------------
# Config UI
# -----------------------------
st.set_page_config(page_title="Prêt à dépenser — Scoring", page_icon="💳", layout="wide")
st.title("💳 Prêt à dépenser — Scoring & Explicabilité")


# -----------------------------
# Libellés FR + aides
# -----------------------------
VAR_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu annuel",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Annuité",
    "DAYS_BIRTH": "Âge (jours négatifs Kaggle)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours négatifs Kaggle)",
    "CNT_CHILDREN": "Nombre d’enfants",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "CODE_GENDER": "Sexe",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "AGE_YEARS": "Âge (années)",
    "PAYMENT_RATE": "Taux de paiement (annuité/crédit)",
    "CREDIT_INCOME_RATIO": "Ratio crédit / revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenu",
    "CREDIT_TERM_MONTHS": "Durée du crédit (mois)",
    "INCOME_PER_PERSON": "Revenu par personne au foyer",
    "CHILDREN_RATIO": "Ratio enfants / foyer",
    "AMT_GOODS_PRICE": "Prix du bien",
}

TOKEN_FR = {
    "AMT": "Montant", "CREDIT": "Crédit", "INCOME": "Revenu", "ANNUITY": "Annuité",
    "DAYS": "Jours", "BIRTH": "Naissance", "EMPLOYED": "Emploi", "EMPLOY": "Emploi",
    "YEARS": "Années", "REGISTRATION": "Enregistrement", "REG": "Enreg.",
    "NAME": "Libellé", "CODE": "Code", "GENDER": "Sexe", "FAMILY": "Famille",
    "STATUS": "Statut", "OWN": "Possession", "REALTY": "Immobilier",
    "FLAG": "Indicateur", "DOCUMENT": "Document", "CNT": "Nombre",
    "FAM": "Famille", "MEMBERS": "Membres", "CHILDREN": "Enfants",
    "GOODS": "Bien", "PRICE": "Prix", "EXT": "Externe", "SOURCE": "Source",
    "RATIO": "Ratio", "RATE": "Taux", "TERM": "Durée", "MISSING": "Manquantes"
}

def label_fr(col: str) -> str:
    if col in VAR_LABELS:
        return VAR_LABELS[col]
    parts = col.split("_")
    fr_parts = []
    for p in parts:
        up = p.upper()
        fr_parts.append(TOKEN_FR.get(up, p.capitalize()))
    out = " ".join(fr_parts)
    out = out.replace("Libellé Contract Type", "Type de contrat")
    out = out.replace("Code Gender", "Sexe")
    out = out.replace("Name Family Status", "Situation familiale")
    return out


# -----------------------------
# API URL & HTTP utils
# -----------------------------
def get_api_url() -> str:
    default = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        # secrets optionnels
        home_secrets = Path.home() / ".streamlit" / "secrets.toml"
        proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
        if home_secrets.exists() or proj_secrets.exists():
            try:
                return st.secrets.get("API_URL", default)  # type: ignore[attr-defined]
            except Exception:
                return default
        return default
    except Exception:
        return default

API_URL = get_api_url()

def http_get_json(url: str, timeout: int = 12) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        return False, {}, str(e)

def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        try:
            return False, r.json(), str(e)  # type: ignore[name-defined]
        except Exception:
            return False, {}, str(e)


# -----------------------------
# Sidebar: Technique & état API
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Technique (API)")
    api_url_input = st.text_input("API URL", API_URL, help="En Codespaces, colle l’URL publique du port 8000.")
    if api_url_input != API_URL:
        API_URL = api_url_input

    # Santé API (v0 : / ou /health)
    ok_root, root_json, err_root = http_get_json(f"{API_URL}/health")
    if not ok_root:
        ok_root, root_json, err_root = http_get_json(f"{API_URL}/")
    if not ok_root:
        st.error(f"API non joignable\n{err_root}")
    else:
        st.success("API: ✅ joignable")
        st.caption(json.dumps(root_json, ensure_ascii=False))

    @st.cache_data(ttl=30, show_spinner=False)
    def get_expected_cols(api_url: str) -> Optional[List[str]]:
        okc, js, _ = http_get_json(f"{api_url}/expected_columns")
        # v0 → liste brute
        if okc and isinstance(js, list):
            return list(js)
        # compat anciens formats
        if okc and isinstance(js, dict) and "expected_columns" in js:
            return list(js["expected_columns"])
        return None

    expected_cols = get_expected_cols(API_URL) or []
    if expected_cols:
        st.info(f"🧾 Colonnes attendues: {len(expected_cols)}", icon="ℹ️")
    else:
        st.warning("Aucune colonne attendue retournée.", icon="⚠️")

    # DS avancées masquées par défaut (cohortes, optimisation seuil, importances)
    SHOW_DS = st.toggle("Afficher sections DS avancées", value=False, help="Cohortes, optimisation de seuil, importances globales…")


# -----------------------------
# Données de référence locales (artifacts) optionnelles
# -----------------------------
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

@st.cache_data(show_spinner=False)
def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

REF_STATS = load_json_if_exists(ARTIFACT_DIR / "ref_stats.json")  # pour quelques comparaisons simples


# -----------------------------
# Saisie — Variables essentielles & avancées
# -----------------------------
st.markdown("### 🧾 Informations client (saisie)")

# Essentiels (cohérents avec le fallback API v0)
c1, c2, c3 = st.columns(3)
with c1:
    amt_income_total = st.number_input("AMT_INCOME_TOTAL (Revenu annuel)", min_value=0.0, step=1000.0, value=120000.0, help="Revenu annuel brut déclaré.")
with c2:
    amt_credit = st.number_input("AMT_CREDIT (Montant du crédit)", min_value=0.0, step=1000.0, value=200000.0, help="Montant total du crédit demandé.")
with c3:
    age_years = st.number_input("AGE_YEARS (Âge en années)", min_value=0, max_value=120, step=1, value=35, help="Âge en années (équivalent DAYS_BIRTH/365).")

st.caption("Les variables ci-dessous sont clés pour affiner la prédiction.")

# Top features pour “variables avancées”
DEFAULT_TOP_FEATURES = [
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

@st.cache_data(ttl=30, show_spinner=False)
def get_top_features(api_url: str) -> List[str]:
    ok, js, _ = http_get_json(f"{api_url}/top_features")
    if ok and isinstance(js, list) and js:
        return [str(x) for x in js][:10]
    return DEFAULT_TOP_FEATURES[:]

top_features = get_top_features(API_URL)
advanced_values: Dict[str, Any] = {}

with st.expander("⚙️ Variables avancées (Top 10 impactantes)", expanded=False):
    base_keys = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"}
    cols = st.columns(3)
    i = 0
    for feat in top_features:
        if feat in base_keys:
            continue
        ui = cols[i % 3]
        with ui:
            val = st.number_input(f"{label_fr(feat)} ({feat})", value=0.0, step=100.0, format="%.4f", key=f"adv_{feat}")
            advanced_values[feat] = val
        i += 1


# -----------------------------
# Évaluation — Appel /predict (API v0)
# -----------------------------
def ratio_to_prob(ratio: float) -> float:
    # même formule que l’API fallback v0: p = 0.2 + 0.6 * r/(r+1)
    if ratio <= 0:
        return 0.2
    return max(0.01, min(0.99, 0.2 + 0.6 * (ratio / (ratio + 1.0))))

def prob_target_to_ratio(p_target: float) -> float:
    # inverse p = 0.2 + 0.6 * x ; x = r/(r+1) = (p-0.2)/0.6 ; r = x/(1-x)
    x = (p_target - 0.2) / 0.6
    x = max(0.0, min(0.99, x))
    if x <= 0.0:
        return 0.0
    return x / (1.0 - x)

st.markdown("---")
st.subheader("🔮 Prédiction unitaire")

btn_predict = st.button("Évaluer ce dossier", type="primary", use_container_width=False)
if btn_predict:
    # payload plat (aligné sur /expected_columns si dispo)
    payload: Dict[str, Any] = {
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AGE_YEARS": age_years,
        **advanced_values,
    }
    if expected_cols:
        # on aligne la forme sur l’API : toutes les colonnes connues, sinon None
        pl2 = {c: payload.get(c, None) for c in expected_cols}
    else:
        pl2 = payload

    ok, pred_json, err = http_post_json(f"{API_URL}/predict", pl2)
    if not ok:
        st.error(f"Erreur /predict : {err}\nRéponse: {pred_json}")
    else:
        prob = float(pred_json.get("probability", 0.0))
        decision = int(pred_json.get("decision", 0))  # 1 = Refuser, 0 = Accorder
        threshold = float(pred_json.get("threshold", 0.5))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Probabilité défaut", f"{prob:.3f}")
        with c2:
            st.metric("Seuil décision", f"{threshold:.2f}")
        with c3:
            st.metric("Décision", "❌ Refus" if decision == 1 else "✅ Acceptation")
        with c4:
            ratio = (amt_credit / amt_income_total) if amt_income_total > 0 else float("inf")
            st.metric("Ratio crédit/revenu", f"{ratio:.3f}" if np.isfinite(ratio) else "∞")

        # Conseils si refusé (pistes concrètes)
        if decision == 1:
            st.warning("❗ Dossier refusé : pistes d’amélioration", icon="⚠️")
            p_target = min(0.49, threshold - 0.01)  # on vise un poil sous le seuil
            r_target = prob_target_to_ratio(p_target)
            if amt_income_total > 0 and np.isfinite(r_target):
                income_needed = amt_credit / r_target if r_target > 0 else float("inf")
                credit_needed = amt_income_total * r_target

                delta_income_pct = None if amt_income_total == 0 else ((income_needed / amt_income_total) - 1.0) * 100.0
                delta_credit_pct = None if amt_credit == 0 else (1.0 - (credit_needed / amt_credit)) * 100.0

                st.markdown(
                    f"""
                    Pour viser une décision **Acceptation** (p ≲ {p_target:.2f}) :
                    - 💸 Augmenter le **revenu** à **≥ {income_needed:,.0f}** ({(delta_income_pct or 0):.1f}% de plus), *ou*
                    - ✂️ Réduire le **crédit** à **≤ {credit_needed:,.0f}** ({(delta_credit_pct or 0):.1f}% de moins).
                    """.replace(",", " ")
                )
            else:
                st.info("Renseigne un revenu strictement positif pour obtenir des recommandations chiffrées.")

        with st.expander("Détails de la réponse"):
            st.json(pred_json)


# -----------------------------
# 🧪 Sections DS avancées (masquées par défaut)
# -----------------------------
if SHOW_DS:
    st.markdown("---")
    st.header("🧪 Sections Data Science (avancées)")

    # -------- Optimisation de seuil via /predict_proba_batch (compat v0) --------
    st.subheader("Optimiser le seuil (fonction coût) depuis un TRAIN local")
    st.caption("Lit un CSV local (avec TARGET), prédit par lots via /predict_proba_batch, puis sélectionne t* qui minimise une fonction coût.")
    train_csv_path_thr = st.text_input("Chemin CSV train (application_train.csv)", value="", key="thr_train_path")
    sample_n = st.number_input("Taille échantillon", min_value=100, max_value=200000, value=5000, step=500)
    batch_size = st.number_input("Taille des lots envoyés à l’API", min_value=200, max_value=5000, value=1000, step=100)
    cost_fn = st.number_input("Coût Faux Négatif (prêter à défaut)", min_value=0.0, value=5.0, step=0.5)
    cost_fp = st.number_input("Coût Faux Positif (refuser un bon client)", min_value=0.0, value=1.0, step=0.5)
    btn_thr = st.button("Calculer p sur l’échantillon & choisir t*")

    if btn_thr and train_csv_path_thr.strip():
        # Récup colonnes
        expected = expected_cols or []
        usecols = list(set(expected) | {"TARGET"}) if expected else None
        try:
            df_all = pd.read_csv(train_csv_path_thr.strip(), usecols=usecols, low_memory=False)
        except Exception as e:
            st.error(f"Lecture CSV impossible: {e}")
            df_all = None

        if df_all is not None:
            if "TARGET" not in df_all.columns:
                st.error("La colonne TARGET est absente du train.")
            else:
                df_all = df_all.dropna(subset=["TARGET"])
                df_all["TARGET"] = pd.to_numeric(df_all["TARGET"], errors="coerce")
                df_all = df_all[df_all["TARGET"].isin([0, 1])]
                if df_all.empty:
                    st.error("Aucune ligne avec TARGET valide.")
                else:
                    n = min(int(sample_n), len(df_all))
                    df_s = df_all.sample(n=n, random_state=42).reset_index(drop=True)

                    # builder records alignés sur expected
                    def build_records(df_chunk: pd.DataFrame, cols_exp: List[str]) -> List[dict]:
                        if not cols_exp:
                            # sinon on envoie les colonnes présentes
                            return df_chunk.to_dict(orient="records")
                        recs = []
                        for _, row in df_chunk.iterrows():
                            rec = {c: (None if pd.isna(row[c]) else row[c]) if c in df_chunk.columns else None for c in cols_exp}
                            recs.append(rec)
                        return recs

                    y_true = df_s["TARGET"].astype(int).tolist()
                    probas: List[float] = []
                    total = len(df_s)
                    for i in range(0, total, int(batch_size)):
                        sub = df_s.iloc[i:i+int(batch_size)]
                        records = build_records(sub, expected)
                        okp, js_p, errp = http_post_json(f"{API_URL}/predict_proba_batch", {"instances": records}, timeout=120)
                        if not okp:
                            st.error(f"/predict_proba_batch erreur: {errp}\nRéponse: {js_p}")
                            st.stop()
                        probas.extend([float(x) for x in js_p.get("probabilities", [])])

                    y = np.asarray(y_true, dtype=int)
                    p = np.asarray(probas, dtype=float)
                    mask = np.isfinite(p)
                    y = y[mask]; p = p[mask]
                    if y.size == 0:
                        st.error("Aucune probabilité valide reçue.")
                    else:
                        # Grid simple de t
                        def compute_confmat(y_true_a: np.ndarray, y_hat_a: np.ndarray):
                            if HAS_SKLEARN:
                                tn, fp, fn, tp = confusion_matrix(y_true_a, y_hat_a, labels=[0, 1]).ravel()
                                return int(tn), int(fp), int(fn), int(tp)
                            # fallback simple
                            tn = int(((y_true_a == 0) & (y_hat_a == 0)).sum())
                            fp = int(((y_true_a == 0) & (y_hat_a == 1)).sum())
                            fn = int(((y_true_a == 1) & (y_hat_a == 0)).sum())
                            tp = int(((y_true_a == 1) & (y_hat_a == 1)).sum())
                            return tn, fp, fn, tp

                        best = {"t": 0.5, "cost": float("inf"), "cm": (0, 0, 0, 0)}
                        for t in np.linspace(0.01, 0.99, 99):
                            y_hat = (p >= t).astype(int)  # 1 = refuser
                            tn, fp, fn, tp = compute_confmat(y, y_hat)
                            cost = fp * float(cost_fp) + fn * float(cost_fn)
                            if cost < best["cost"]:
                                best = {"t": float(t), "cost": float(cost), "cm": (tn, fp, fn, tp)}

                        tn, fp, fn, tp = best["cm"]
                        st.success(f"t* = {best['t']:.3f} (coût={best['cost']:.3f}) — N={len(y)}  •  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

                        # Écriture metadata.json (optionnelle)
                        meta_path = ARTIFACT_DIR / "metadata.json"
                        meta = load_json_if_exists(meta_path) or {}
                        meta["decision_threshold"] = {
                            "t_selected": best["t"],
                            "policy": {"cost_FN": float(cost_fn), "cost_FP": float(cost_fp)},
                            "selected_on": "train_sample",
                            "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                            "n": int(len(y)),
                            "timestamp": int(time.time())
                        }
                        try:
                            meta_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                            st.success("metadata.json mis à jour ✅ (redémarre l’API pour prise en compte si utilisée).")
                        except Exception as e:
                            st.warning(f"Impossible d’écrire metadata.json : {e}")

    # -------- Importances globales (artifacts/global_importance.csv) --------
    st.subheader("Variables les plus influentes (globales)")
    gi_path = ARTIFACT_DIR / "global_importance.csv"
    if gi_path.exists():
        try:
            gi = pd.read_csv(gi_path)
            # colonnes possibles : raw_feature / feature ; abs_importance / importance
            cols = [c.lower() for c in gi.columns]
            feat_col = "raw_feature" if "raw_feature" in cols else ("feature" if "feature" in cols else gi.columns[0])
            imp_col = "abs_importance" if "abs_importance" in cols else ("importance" if "importance" in cols else None)

            topk = st.slider("Top variables à afficher", 5, 30, 15, 1)
            sub = gi.copy()
            sub = sub[[feat_col] + ([imp_col] if imp_col else [])].rename(columns={feat_col: "feature", imp_col or "": "importance"})
            if imp_col:
                # tri décroissant
                sub = sub.sort_values(by="importance", key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False)
            sub = sub.head(topk)

            if HAS_PLOTLY and imp_col:
                fig = go.Figure(go.Bar(x=sub["importance"], y=sub["feature"], orientation="h"))
                fig.update_layout(height=400 + 18 * len(sub), margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(sub, use_container_width=True)
        except Exception as e:
            st.caption(f"Impossible de lire global_importance.csv : {e}")
    else:
        st.caption("Place `artifacts/global_importance.csv` pour visualiser.")
