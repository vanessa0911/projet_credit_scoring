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
top_features: List[str] = []

if api_ok:
    # Colonnes de référence (informative)
    try:
        expected_cols = api_get("/expected_columns").json()
        if expected_cols:
            st.sidebar.info(f"🧾 Colonnes attendues: {len(expected_cols)}", icon="ℹ️")
        else:
            st.sidebar.warning("Aucune colonne attendue retournée (dataset non présent côté API ?).", icon="⚠️")
    except Exception as e:
        st.sidebar.error(f"Impossible de récupérer /expected_columns : {e}", icon="🚫")

    # Top features (pour la saisie avancée) — robuste: fallback local si API KO
    try:
        top_features = api_get("/top_features").json()
        if not isinstance(top_features, list) or not top_features:
            top_features = DEFAULT_TOP_FEATURES[:]
    except Exception:
        top_features = DEFAULT_TOP_FEATURES[:]
else:
    top_features = DEFAULT_TOP_FEATURES[:]

st.markdown("---")

# -----------------------------
# Prédiction unitaire (UI unique)
# -----------------------------
st.subheader("🔮 Prédiction unitaire")
st.caption("Les variables ci-dessous sont clés pour affiner la prédiction.")

# Champs essentiels (cohérents avec le fallback côté API)
amt_income_total = st.number_input("AMT_INCOME_TOTAL (revenu total)", min_value=0.0, step=100.0, value=120000.0)
amt_credit = st.number_input("AMT_CREDIT (montant du crédit)", min_value=0.0, step=100.0, value=200000.0)
age_years = st.number_input("AGE_YEARS (années)", min_value=0, max_value=120, step=1, value=35)

# Variables avancées (Top 10 impactantes)
advanced_values: Dict[str, Any] = {}
with st.expander("⚙️ Variables avancées (Top 10 impactantes)", expanded=False):
    base_keys = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"}
    for feat in top_features:
        if feat in base_keys:
            continue
        # Input numérique par défaut; adaptera quand un vrai modèle sera branché
        val = st.number_input(feat, value=0.0, step=100.0, format="%.4f", key=f"adv_{feat}")
        advanced_values[feat] = val

# -----------------------------
# Soumission et résultat
# -----------------------------
def ratio_to_prob(ratio: float) -> float:
    # même formule que l'API (fallback): p = 0.2 + 0.6 * r/(r+1)
    if ratio <= 0:
        return 0.2
    return max(0.01, min(0.99, 0.2 + 0.6 * (ratio / (ratio + 1.0))))

def prob_target_to_ratio(p_target: float) -> float:
    # inverse de p = 0.2 + 0.6 * x ; x = r/(r+1) = (p-0.2)/0.6 ; r = x/(1-x)
    x = (p_target - 0.2) / 0.6
    x = max(0.0, min(0.99, x))  # borne pour éviter divisions suspectes
    if x >= 0.99:
        x = 0.99
    if x <= 0.0:
        return 0.0
    return x / (1.0 - x)

col_btn, col_blank = st.columns([1, 3])
with col_btn:
    submit = st.button("Évaluer ce dossier", type="primary", disabled=not api_ok, use_container_width=True)

if submit:
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
        if expected_cols:
            for c in expected_cols:
                if c not in payload:
                    payload[c] = None

        try:
            resp = api_post("/predict", json=payload).json()
            prob = float(resp.get("probability", 0.0))
            decision = int(resp.get("decision", 0))  # 1 = Refuser (défaut élevé), 0 = Accorder
            threshold = float(resp.get("threshold", 0.5))

            # Affichage principal
            st.success("Prédiction obtenue ✅")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Probabilité défaut", f"{prob:.3f}")
            m2.metric("Seuil décision", f"{threshold:.2f}")
            decision_txt = "Refuser" if decision == 1 else "Accorder"
            m3.metric("Décision", decision_txt)
            ratio = (amt_credit / amt_income_total) if amt_income_total > 0 else float("inf")
            m4.metric("Ratio crédit/revenu", f"{ratio:.3f}" if ratio != float("inf") else "∞")

            # Conseils si refusé : comment basculer en “Accorder”
            if decision == 1:
                st.warning("❗ Dossier refusé : pistes d’amélioration", icon="⚠️")

                # On vise une proba légèrement < seuil (ex: 0.49) pour éviter l'égalité
                p_target = min(0.49, threshold - 0.01)
                r_target = prob_target_to_ratio(p_target)  # ratio visé

                if amt_income_total > 0:
                    income_needed = amt_credit / r_target if r_target > 0 else float("inf")
                    credit_needed = amt_income_total * r_target

                    # % d'effort
                    delta_income_pct = None if amt_income_total == 0 else ((income_needed / amt_income_total) - 1.0) * 100.0
                    delta_credit_pct = None if amt_credit == 0 else (1.0 - (credit_needed / amt_credit)) * 100.0

                    st.markdown(
                        f"""
                        Pour passer en **Accordé** (proba ≲ {p_target:.2f}) avec notre heuristique :
                        - 💸 **Augmenter le revenu** à **≥ {income_needed:,.0f}** ({(delta_income_pct or 0):.1f}% de plus), *ou*
                        - ✂️ **Réduire le crédit** à **≤ {credit_needed:,.0f}** ({(delta_credit_pct or 0):.1f}% de moins).
                        """.replace(",", " ")
                    )

                else:
                    st.info(
                        "Le revenu est 0 — impossible de calculer un ratio pertinent. "
                        "Renseigne un revenu strictement positif pour obtenir des recommandations chiffrées."
                    )

            with st.expander("Détails de la réponse"):
                st.json(resp)

        except Exception as e:
            st.error(f"Erreur lors de l'appel à /predict : {e}")
