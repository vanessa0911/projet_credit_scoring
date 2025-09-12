# streamlit_app.py
# =========================================================
# Tableau de bord scoring prês-metier
# - API_URL robuste (sans secrets.toml obligatoire)
# - Entête API: affiche chosen_model OU model_class
# - Inputs client -> payload -> /predict
# - Batch CSV -> /predict_proba_batch
# - Comparaison population & cohortes si artifacts présents
# =========================================================

# 0) Page config: doit être le PREMIER appel Streamlit
import streamlit as st
st.set_page_config(
    page_title="Prêt à dépenser — Scoring",
    page_icon="💳",
    layout="wide",
)

# 1) Imports standard
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

# 2) API_URL robuste (pas d'erreur si secrets.toml absent)
def get_api_url() -> str:
    default = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        home_secrets = Path.home() / ".streamlit" / "secrets.toml"
        proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
        # On ne lit st.secrets QUE si un fichier existe réellement
        if home_secrets.exists() or proj_secrets.exists():
            try:
                return st.secrets.get("API_URL", default)  # type: ignore[attr-defined]
            except Exception:
                return default
        return default
    except Exception:
        return default

API_URL = get_api_url()

# 3) Utils HTTP
def http_get_json(url: str, timeout: int = 6) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        return False, {}, str(e)

def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        # Essayer de lire réponse JSON d’erreur pour debug
        try:
            return False, r.json(), str(e)  # type: ignore[name-defined]
        except Exception:
            return False, {}, str(e)

# 4) Chargement d’artefacts locaux (optionnels)
@st.cache_data(show_spinner=False)
def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

ARTIFACT_DIR = Path("artifacts")
REF_STATS = load_json_if_exists(ARTIFACT_DIR / "ref_stats.json")   # stats population
SEG_STATS = load_json_if_exists(ARTIFACT_DIR / "seg_stats.json")   # stats cohortes
GLOBAL_IMP = None
try:
    gi_path = ARTIFACT_DIR / "global_importance.csv"
    if gi_path.exists():
        GLOBAL_IMP = pd.read_csv(gi_path)
except Exception:
    GLOBAL_IMP = None

# 5) En-tête chaleureux & état API
st.markdown("### 💬 Notre engagement")
st.info(
    "Derrière chaque dossier, il y a une personne. Notre rôle est d’accompagner au mieux, "
    "avec des indicateurs **statistiques** conçus pour protéger à la fois nos clients et la banque. "
    "Une décision négative n’est jamais punitive : elle évite un risque de surendettement et oriente vers des solutions adaptées."
)

with st.container():
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("État de l’API")
        ok, root_json, err = http_get_json(f"{API_URL}/")
        if not ok:
            st.error(f"API non joignable. Lance `python -m uvicorn api:app --reload`.\n\nEndpoint : {API_URL}\n\n{err}")
            status = {}
        else:
            status = root_json
            # label modèle (chosen_model sinon model_class sinon "?")
            model_label = status.get("chosen_model") or status.get("model_class") or "?"
            dt = status.get("decision_threshold")
            st.success(f"API ✅ • **Modèle** : {model_label} • **Seuil API** : {dt}")

    with c2:
        st.subheader("Fiche décision (aperçu)")
        st.caption("Renseigne les informations client ci-dessous, puis clique sur **Évaluer ce dossier**.")

st.markdown("---")

# 6) Sidebar — bloc technique + payload + batch
with st.sidebar:
    st.markdown("### ⚙️ Technique (Data/DS)")
    st.caption("Ces informations sont utiles au suivi technique, pas indispensables au métier.")

    # API URL paramétrable (optionnel)
    api_url_input = st.text_input("API URL", API_URL, help="Modifie l’URL si l’API est déployée ailleurs.")
    if api_url_input != API_URL:
        API_URL = api_url_input  # mise à jour live

    # Récup colonnes attendues pour assembler un payload complet (clé → None par défaut)
    @st.cache_data(ttl=30, show_spinner=False)
    def get_expected_cols() -> Optional[list[str]]:
        okc, js, _ = http_get_json(f"{API_URL}/expected_columns")
        if okc and isinstance(js, dict) and "columns" in js:
            return list(js["columns"])
        return None

    exp_cols = get_expected_cols()

    st.markdown("#### 🧩 Payload → API")
    st.caption("Aperçu du JSON envoyé à l’API (affiché quand les champs principaux sont saisis).")

# 7) Zone informations client (inputs)
st.markdown("### 🧾 Informations client")

# Aides : convertir les jours négatifs → années positives (métier)
def days_to_years_positive(days: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    try:
        return round(abs(float(days)) / 365.25, 2)
    except Exception:
        return None

# Champs essentiels (tu peux en ajouter selon ton modèle)
c1, c2, c3 = st.columns(3)
with c1:
    amt_income = st.number_input("Revenu annuel (AMT_INCOME_TOTAL)", min_value=0.0, step=1000.0, format="%.0f")
    amt_credit = st.number_input("Montant du crédit (AMT_CREDIT)", min_value=0.0, step=1000.0, format="%.0f")
with c2:
    days_birth = st.number_input("Âge (en jours, négatif dans les données)", value=-14000.0, step=100.0, help="Valeur Kaggle d’origine (jours négatifs).", format="%.0f")
    age_years   = days_to_years_positive(days_birth)
    st.caption(f"Âge estimé ≈ **{age_years if age_years is not None else '?'}** ans")
with c3:
    days_employed = st.number_input("Ancienneté emploi (DAYS_EMPLOYED, jours négatifs)", value=-2000.0, step=50.0, format="%.0f")
    emp_years     = days_to_years_positive(days_employed)
    st.caption(f"Ancienneté estimée ≈ **{emp_years if emp_years is not None else '?'}** ans")

# Quelques compléments utiles
cc1, cc2, cc3 = st.columns(3)
with cc1:
    children = st.number_input("Nombre d’enfants (CNT_CHILDREN)", min_value=0, step=1)
with cc2:
    annuity  = st.number_input("Annuité (AMT_ANNUITY)", min_value=0.0, step=500.0, format="%.0f")
with cc3:
    contract = st.selectbox("Type de contrat (NAME_CONTRACT_TYPE)", ["", "Cash loans", "Revolving loans"])

# 8) Construire le payload harmonisé aux colonnes attendues (si dispos)
def build_payload() -> Dict[str, Any]:
    # Champs “métier” utilisés souvent par les modèles
    base = {
        "AMT_INCOME_TOTAL": None if amt_income == 0 else amt_income,
        "AMT_CREDIT":       None if amt_credit == 0 else amt_credit,
        "DAYS_BIRTH":       None if days_birth == 0 else days_birth,           # on envoie la VRAIE valeur (jours négatifs)
        "DAYS_EMPLOYED":    None if days_employed == 0 else days_employed,     # idem
        "CNT_CHILDREN":     None if children == 0 else int(children),
        "AMT_ANNUITY":      None if annuity == 0 else annuity,
        "NAME_CONTRACT_TYPE": contract if contract else None,
        # Ajoute ici d’autres colonnes si ton modèle les attend…
    }
    # Si l’API fournit la liste exacte des colonnes attendues, on la respecte
    if exp_cols:
        payload = {col: base.get(col, None) for col in exp_cols}
        return payload
    return base

payload = build_payload()

# Afficher payload dans la sidebar UNIQUEMENT si valeurs clés sont remplies
with st.sidebar:
    show_preview = any([
        payload.get("AMT_INCOME_TOTAL") not in (None, 0, ""),
        payload.get("AMT_CREDIT") not in (None, 0, ""),
        payload.get("DAYS_BIRTH") not in (None, 0, ""),
    ])
    if show_preview:
        st.json({"data": {k: v for k, v in payload.items() if v is not None}})
    else:
        st.caption("Renseigne au moins revenu, crédit et âge pour voir l’aperçu du payload.")

# 9) Bouton Évaluer ce dossier → /predict
st.markdown("### 🧮 Évaluation")
left, right = st.columns([1, 3])
with left:
    do_predict = st.button("Évaluer ce dossier", type="primary")
with right:
    st.caption("Compare automatiquement la probabilité de défaut au **seuil API** pour proposer une décision.")

if do_predict:
    ok, pred_json, err = http_post_json(f"{API_URL}/predict", {"data": payload})
    if not ok:
        st.error(f"Erreur /predict : {err}\n\nRéponse: {pred_json}")
    else:
        proba = pred_json.get("probability_default")
        decision = pred_json.get("decision")
        thr = pred_json.get("threshold_used")
        if proba is None:
            st.warning(f"Réponse inattendue: {pred_json}")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilité de défaut", f"{proba:.2%}")
                st.metric("Décision", "❌ Refus" if decision == "reject" else "✅ Acceptation")
            with c2:
                st.write("**Seuil utilisé (API)**")
                st.json(thr)

# 10) Comparaison population (REF_STATS)
st.markdown("---")
st.markdown("### 📊 Comparaison avec la population (référence = jeu d’entraînement)")
if REF_STATS is None:
    st.info("Stats de population indisponibles (`artifacts/ref_stats.json`). Génère-les depuis ton notebook et place le fichier dans `artifacts/`.")
else:
    # Helper pour affichage percentile + interprétation
    def show_population_position(feature: str, label: str, value: Optional[float], fmt: str = ".0f", unit: str = ""):
        stats = REF_STATS.get(feature)
        if not stats or value is None:
            st.write(f"**{label}** : information manquante.")
            return
        mu = stats.get("mean")
        sigma = stats.get("std")
        q = stats.get("quantiles", {})  # dict "pXX" -> val
        # position percentile approx par histogramme (si dispo) — sinon via quantiles
        # Ici on tente une interpolation simple avec quantiles si p5/p50/p95
        p_est = None
        # on fabrique une phrase simple & utile
        if mu is not None and sigma not in (None, 0):
            z = (value - mu) / sigma
            side = "au-dessus" if value > mu else "au-dessous"
            st.write(f"**{label}** : {value:{fmt}}{unit} — {side} de la moyenne (μ={mu:{fmt}}{unit}, σ={sigma:{fmt}}).")
            # petit graphique repère
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": label},
                gauge={"axis": {"range": [max(0, mu - 3*sigma), mu + 3*sigma]},
                       "bar": {"thickness": 0.5}},
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Lecture : plus la valeur s’éloigne de la moyenne, plus le profil diffère du client moyen."
            )
        else:
            st.write(f"**{label}** : {value:{fmt}}{unit} (μ/σ indisponibles).")

    c1, c2, c3 = st.columns(3)
    with c1:
        show_population_position("AMT_CREDIT", "Montant du crédit", payload.get("AMT_CREDIT"), fmt=".0f", unit=" €")
    with c2:
        # Convertir âge (jours) → années pour LECTURE (on a envoyé jours à l’API)
        age_disp = age_years
        show_population_position("AGE_YEARS", "Âge (années)", age_disp, fmt=".1f", unit=" ans")
    with c3:
        inc = payload.get("AMT_INCOME_TOTAL")
        show_population_position("AMT_INCOME_TOTAL", "Revenu annuel", inc, fmt=".0f", unit=" €")

# 11) Cohorte similaire (si SEG_STATS dispo)
st.markdown("---")
st.markdown("### 👥 Cohorte similaire (âge & type de contrat)")
if SEG_STATS is None:
    st.info("Stats cohortes indisponibles (`artifacts/seg_stats.json`).")
else:
    # Trouver une tranche d’âge simple
    age_bin = None
    if age_years is not None:
        # Exemple de tranches : [0-30], (30-40], (40-50], (50+]
        bins = [0, 30, 40, 50, 120]
        labels = ["0-30", "30-40", "40-50", "50+"]
        for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
            if lo <= age_years <= hi:
                age_bin = lab
                break
    cohort_key = None
    if age_bin and contract:
        cohort_key = f"{age_bin}|{contract}"
    elif age_bin:
        cohort_key = f"{age_bin}|*"

    if not cohort_key or cohort_key not in SEG_STATS:
        st.warning("Aucune statistique disponible pour cette cohorte (vérifie les clés du JSON).")
    else:
        seg = SEG_STATS[cohort_key]
        st.write(f"**Cohorte**: `{cohort_key}` — taille ~ {seg.get('count', '?')}")
        # Exemple d’agrégats
        st.write("Aperçu indicateurs (moyennes) :")
        st.json(seg.get("means", {}))

# 12) Prédictions en lot (CSV) → sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📦 Prédictions en lot (CSV)")
    st.caption("Envoie un CSV avec **les mêmes colonnes** que l’API attend. Retour: probas par ligne.")
    up = st.file_uploader("Fichier CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            # Remplir les colonnes manquantes si exp_cols dispo
            if exp_cols:
                for col in exp_cols:
                    if col not in df.columns:
                        df[col] = None
                df = df[exp_cols]
            data = {"rows": df.to_dict(orient="records")}
            ok, js, err = http_post_json(f"{API_URL}/predict_proba_batch", data, timeout=30)
            if not ok:
                st.error(f"Erreur /predict_proba_batch : {err}\n\nRéponse: {js}")
            else:
                out = js.get("predictions", [])
                st.success(f"{len(out)} prédictions reçues")
                st.dataframe(pd.DataFrame({"proba_default": out}))
        except Exception as e:
            st.error(f"Lecture CSV impossible: {e}")

# 13) Importance globale (si fournie)
st.markdown("---")
st.markdown("### 🧠 Variables les plus influentes (globales)")
if GLOBAL_IMP is None or GLOBAL_IMP.empty:
    st.caption("Aucune importance globale fournie (place `artifacts/global_importance.csv`).")
else:
    topk = st.slider("Top variables à afficher", 5, 30, 15, 1)
    sub = GLOBAL_IMP.head(topk)
    fig = go.Figure(go.Bar(x=sub["abs_importance"], y=sub["raw_feature"], orientation="h"))
    fig.update_layout(height=400 + 18 * len(sub), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Ces importances proviennent du notebook d’entraînement (moyenne absolue de l’impact).")
