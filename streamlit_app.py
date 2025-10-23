# streamlit_app.py
import os
import io
import json
from typing import Any, Dict, List

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Config UI
# -----------------------------
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("📊 Credit Scoring — Dashboard")

# -----------------------------
# Sidebar: API URL + helpers
# -----------------------------
DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("URL de l'API", st.session_state.get("api_url", DEFAULT_API))
api_url = api_url.rstrip("/")  # normaliser
if "api_url" not in st.session_state or st.session_state["api_url"] != api_url:
    st.session_state["api_url"] = api_url

def api_get(path: str, **kwargs):
    url = f"{st.session_state['api_url']}{path}"
    r = requests.get(url, timeout=15, **kwargs)
    r.raise_for_status()
    return r

def api_post(path: str, json: Dict[str, Any] | None = None, **kwargs):
    url = f"{st.session_state['api_url']}{path}"
    r = requests.post(url, json=json, timeout=30, **kwargs)
    r.raise_for_status()
    return r

# -----------------------------
# Health + Expected Columns + Top Features
# -----------------------------
with st.sidebar:
    st.subheader("État de l'API")
    api_ok = False
    try:
        health_payload = api_get("/health").json()
        api_ok = True
        st.success("API: ✅ joignable", icon="✅")
        st.caption(f"Health: {json.dumps(health_payload, ensure_ascii=False)}")
    except Exception as e:
        st.error("API: ❌ non joignable", icon="🚫")
        st.caption(str(e))

    st.markdown("---")
    st.caption("Astuce: en Codespaces, utilise l’URL *githubpreview.dev* du port 8000.")

expected_cols: List[str] = []
top_features: List[str] = []
cols_msg = st.empty()

if api_ok:
    # Colonnes de référence (informative)
    try:
        expected_cols = api_get("/expected_columns").json()
        with cols_msg.container():
            if expected_cols:
                st.info(f"🧾 Colonnes attendues: {len(expected_cols)}", icon="ℹ️")
                st.code(", ".join(expected_cols[:50]) + (" ..." if len(expected_cols) > 50 else ""), language="text")
            else:
                st.warning("Aucune colonne attendue retournée (dataset non présent côté API ?).", icon="⚠️")
    except Exception as e:
        with cols_msg.container():
            st.error(f"Impossible de récupérer /expected_columns : {e}", icon="🚫")

    # Top features (pour la saisie avancée)
    try:
        top_features = api_get("/top_features").json()
    except Exception as e:
        top_features = []
        st.sidebar.warning(f"Impossible de récupérer /top_features : {e}")

st.markdown("---")

# -----------------------------
# Colonne gauche: Prédiction unitaire
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("🔮 Prédiction unitaire")
    st.caption("Renseigne au minimum le **revenu** et le **montant de crédit**. "
               "Les noms correspondent au dataset Home Credit.")

    # Champs essentiels (cohérents avec l'API fallback)
    amt_income_total = st.number_input("AMT_INCOME_TOTAL (revenu total)", min_value=0.0, step=100.0, value=120000.0)
    amt_credit = st.number_input("AMT_CREDIT (montant du crédit)", min_value=0.0, step=100.0, value=200000.0)
    age_years = st.number_input("AGE_YEARS (années)", min_value=0, max_value=120, step=1, value=35)

    # Variables avancées (top 10 impactantes)
    advanced_values: Dict[str, Any] = {}
    with st.expander("⚙️ Variables avancées (Top 10 impactantes)", expanded=False):
        if not top_features:
            st.info("Aucune liste de variables impactantes disponible. "
                    "Ajoute des artefacts dans `artifacts/` ou utilise les 3 champs essentiels.")
        else:
            # éviter les doublons avec les 3 champs déjà saisis
            base_keys = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"}
            for feat in top_features:
                if feat in base_keys:
                    continue
                label = feat
                # choix simple : number_input par défaut (features Home Credit sont majoritairement numériques)
                val = st.number_input(label, value=0.0, step=100.0, format="%.4f", key=f"adv_{feat}")
                advanced_values[feat] = val

    if st.button("Évaluer ce dossier", use_container_width=True, type="primary", disabled=not api_ok):
        if not api_ok:
            st.error("API indisponible. Vérifie l’URL dans la sidebar.")
        else:
            # Payload minimal + variables avancées
            payload: Dict[str, Any] = {
                "AMT_INCOME_TOTAL": amt_income_total,
                "AMT_CREDIT": amt_credit,
                "AGE_YEARS": age_years,
                **advanced_values,
            }
            # Optionnel : compléter les expected_cols à None (quand elles existent)
            if expected_cols:
                for c in expected_cols:
                    if c not in payload:
                        payload[c] = None

            try:
                resp = api_post("/predict", json=payload).json()
                prob = float(resp.get("probability", 0.0))
                decision = int(resp.get("decision", 0))
                threshold = float(resp.get("threshold", 0.5))

                st.success("Prédiction obtenue ✅")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Probabilité défaut", f"{prob:.3f}")
                mcol2.metric("Seuil décision", f"{threshold:.2f}")
                mcol3.metric("Décision", "Refuser" if decision == 1 else "Accorder")

                with st.expander("Détails de la réponse"):
                    st.json(resp)
            except Exception as e:
                st.error(f"Erreur lors de l'appel à /predict : {e}")

# -----------------------------
# Colonne droite: Informations
# -----------------------------
with right:
    st.subheader("ℹ️ Informations")
    st.write(
        "- Les 3 champs essentiels pilotent un fallback côté API (aucun modèle requis).\n"
        "- Les **variables avancées** alimentent la requête et seront prises en compte si ton modèle les utilise.\n"
        "- Quand tes artefacts (`artifacts/global_importance.csv` ou `interpretability_summary.json`) sont présents, la liste Top 10 est lue automatiquement."
    )

st.markdown("---")
st.caption(
    "Conseil : en Codespaces, mets l’URL du port 8000 (githubpreview.dev) dans la sidebar. "
    "Si le dataset n'est pas disponible côté API, les 3 colonnes par défaut restent actives."
)
