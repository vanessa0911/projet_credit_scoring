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
api_url = api_url.rstrip("/")  # normaliser (pas de / final)
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
# Health + Expected Columns
# -----------------------------
with st.sidebar:
    st.subheader("État de l'API")

    api_ok = False
    health_payload = None

    try:
        # On cible explicitement /health (existe dans l'API)
        health_payload = api_get("/health").json()
        api_ok = True
        st.success("API: ✅ joignable", icon="✅")
        st.caption(f"Health: {json.dumps(health_payload, ensure_ascii=False)}")
    except Exception as e:
        st.error("API: ❌ non joignable", icon="🚫")
        st.caption(str(e))

    st.markdown("---")
    st.caption("Astuce: en Codespaces, utilise l’URL *githubpreview.dev* du port 8000.")

# Récupérer les colonnes attendues (si possible)
expected_cols: List[str] = []
cols_msg_placeholder = st.empty()
try:
    if api_ok:
        expected_cols = api_get("/expected_columns").json()
        with cols_msg_placeholder.container():
            if expected_cols:
                st.info(f"🧾 Colonnes attendues: {len(expected_cols)}", icon="ℹ️")
                st.code(", ".join(expected_cols[:50]) + (" ..." if len(expected_cols) > 50 else ""), language="text")
            else:
                st.warning("Aucune colonne attendue retournée (dataset non présent côté API ?).", icon="⚠️")
except Exception as e:
    with cols_msg_placeholder.container():
        st.error(f"Impossible de récupérer /expected_columns : {e}", icon="🚫")

st.markdown("---")

# -----------------------------
# Colonne gauche: Prédiction unitaire
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("🔮 Prédiction unitaire")

    st.caption("Renseigne au minimum le **revenu** et le **montant de crédit**. "
               "Les noms correspondent au dataset Home Credit.")

    # Champs les plus utiles pour le fallback de l'API:
    amt_income_total = st.number_input("AMT_INCOME_TOTAL (revenu total)", min_value=0.0, step=100.0, value=120000.0)
    amt_credit = st.number_input("AMT_CREDIT (montant du crédit)", min_value=0.0, step=100.0, value=200000.0)
    # Champs additionnels optionnels (le fallback ne les utilise pas mais on les expose à titre d’exemple)
    age_years = st.number_input("AGE (années) [optionnel]", min_value=0, max_value=120, step=1, value=35)

    if st.button("Évaluer ce dossier", use_container_width=True, type="primary", disabled=not api_ok):
        if not api_ok:
            st.error("API indisponible. Vérifie l’URL dans la sidebar.")
        else:
            # Crée un payload minimal compatible avec l'heuristique fallback (income/credit)
            payload: Dict[str, Any] = {
                "AMT_INCOME_TOTAL": amt_income_total,
                "AMT_CREDIT": amt_credit,
                "AGE_YEARS": age_years,  # illustratif
            }

            # Si l'API a fourni une liste de colonnes, on peut remplir à None celles non fournies
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
# Colonne droite: Prédictions en lot (CSV)
# -----------------------------
with right:
    st.subheader("📦 Prédictions en lot (CSV)")
    st.caption("Charge un CSV avec une ligne par dossier. "
               "Si des colonnes manquent par rapport à /expected_columns, elles seront complétées à None.")

    file = st.file_uploader("Fichier CSV", type=["csv"], accept_multiple_files=False, help="Dépose ton fichier ici.")
    if file is not None and api_ok:
        try:
            # Lecture CSV
            content = file.read()
            df = pd.read_csv(io.BytesIO(content))

            st.write("Aperçu CSV chargé :")
            st.dataframe(df.head(20), use_container_width=True)

            # Harmoniser avec expected_cols si connues
            instances: List[Dict[str, Any]] = df.to_dict(orient="records")
            if expected_cols:
                normed_instances = []
                for row in instances:
                    normed_row = {c: row.get(c, None) for c in expected_cols}
                    normed_instances.append(normed_row)
                instances = normed_instances

            # Envoi à l'API
            if st.button("Lancer les prédictions en lot", use_container_width=True):
                try:
                    resp = api_post("/predict_proba_batch", json={"instances": instances}).json()
                    probs = resp.get("probabilities", [])
                    out = df.copy()
                    out["probability"] = probs
                    out["decision"] = (out["probability"] >= 0.5).astype(int)

                    st.success(f"Prédictions OK ({len(probs)} lignes)")
                    st.dataframe(out.head(50), use_container_width=True)

                    # Téléchargement
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Télécharger les résultats (CSV)",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à /predict_proba_batch : {e}")
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")

# -----------------------------
# Footer infos
# -----------------------------
st.markdown("---")
st.caption(
    "Conseil : en Codespaces, mets l’URL du port 8000 (githubpreview.dev) dans la sidebar. "
    "Si /expected_columns est vide, le dashboard fonctionne quand même (fallback côté API)."
)
