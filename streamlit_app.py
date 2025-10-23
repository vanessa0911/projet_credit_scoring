# ================================================================
#  streamlit_app.py — Dashboard Credit Scoring (complet)
#  Layout + Variables clés en vert + Explicabilité (global/local)
#  + Zone Data Science avancée (PD, seuil, ratio crédit/revenu)
# ================================================================

from __future__ import annotations
import io
import json
import math
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Graphes interactifs
import plotly.express as px
import plotly.graph_objects as go

# SHAP (facultatif côté UI si l'API ne le fournit pas)
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ------------------------------
# Configuration générale
# ------------------------------
st.set_page_config(
    page_title="Credit Scoring – Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DEFAULT_API_URL: fallback si secrets.toml absent
try:
    DEFAULT_API_URL = st.secrets["API_URL"]  # nécessite .streamlit/secrets.toml
except Exception:
    DEFAULT_API_URL = "http://127.0.0.1:8000"

# Couleurs
GREEN = "#059669"  # vert (variables clés, décisions favorables)
RED = "#DC2626"    # rouge (refus)
GRAY = "#374151"

# ------------------------------
# Client API
# ------------------------------
class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, **params):
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        if r.text.strip() == "":
            return {}
        return r.json()

    def _post_json(self, path: str, payload: Dict[str, Any]):
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        if r.text.strip() == "":
            return {}
        return r.json()

    # Endpoints
    def health(self):
        try:
            return self._get("/")
        except Exception as e:
            return {"status": "down", "error": str(e)}

    def expected_columns(self) -> List[str]:
        data = self._get("/expected_columns")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("columns", []))
        return []

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self._post_json("/predict", {"features": features})

    def predict_proba_batch(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        data = self._post_json("/predict_proba_batch", {"rows": rows})
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def global_importance(self) -> Optional[pd.DataFrame]:
        try:
            data = self._get("/global_importance")
            return pd.DataFrame(data)
        except Exception:
            return None

    def shap_local(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return self._post_json("/shap_local", {"features": features})
        except Exception:
            return None

    def key_features(self) -> Optional[List[str]]:
        try:
            data = self._get("/key_features")
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            if isinstance(data, list):
                return list(data)
        except Exception:
            pass
        return None

# ------------------------------
# Utilitaires
# ------------------------------
def coerce_value(val: str) -> Any:
    """
    Convertit une chaîne en nombre si possible, sinon renvoie la chaîne.
    Gère '', 'nan', 'none', 'null' -> None.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        return float(s)
    except Exception:
        return s

def is_gender_feature(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["sex", "sexe", "gender", "genre"])

def is_ratio_feature(name: str) -> bool:
    n = name.lower()
    return ("ratio" in n) or ("/" in name)

def derive_key_features(gi: Optional[pd.DataFrame], fallback: Optional[List[str]], top_k: int = 20) -> List[str]:
    """
    Déduit les variables clés :
    - priorité à l'endpoint /key_features
    - sinon top-k de l'importance globale
    """
    if fallback:
        return list(fallback)
    if gi is not None and not gi.empty and {"feature", "importance"}.issubset(gi.columns):
        gi = gi.dropna(subset=["importance"]).sort_values("importance", ascending=False)
        return list(gi["feature"].head(top_k))
    return []

# ------------------------------
# Barre latérale
# ------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL, help="Ex: http://127.0.0.1:8000")
    client = ApiClient(api_url)

    threshold = st.slider(
        "Seuil d'acceptation (proba de défaut)",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Convention : PD ≥ seuil ⇒ Refus (adaptez selon votre usage)."
    )

    st.markdown("---")
    st.caption("Astuce : si l'état API est rouge, vérifiez que FastAPI tourne bien et que l'URL est correcte.")

# ------------------------------
# Header & KPIs
# ------------------------------
col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
with col1:
    st.title("📊 Credit Scoring – Dashboard")
with col2:
    health = client.health()
    status = "🟢 UP" if isinstance(health, dict) and health.get("status", "up").lower() in ("ok", "up") else "🔴 DOWN"
    st.metric("API", status)
with col3:
    st.metric("Seuil", f"{threshold:.2f}")
with col4:
    st.metric("Version UI", "v1.2")

st.markdown("---")

# ------------------------------
# Préchargements communs
# ------------------------------
try:
    expected_cols: List[str] = client.expected_columns()
except Exception:
    expected_cols = []

_gi = client.global_importance()
_kf = client.key_features()
key_feats = set(derive_key_features(_gi, _kf))

# ------------------------------
# Tabs
# ------------------------------
TAB_APERCU, TAB_DOSSIER, TAB_BATCH, TAB_EXPLAIN, TAB_DS = st.tabs(
    ["Aperçu", "Dossier individuel", "Batch CSV", "Explicabilité", "DS avancée"]
)

# ------------------------------
# 1) APERÇU
# ------------------------------
with TAB_APERCU:
    st.subheader("Vue d'ensemble")

    if expected_cols:
        st.success(f"Colonnes attendues ({len(expected_cols)})")
        df_cols = pd.DataFrame({
            "feature": expected_cols,
            "clé": ["oui" if c in key_feats else "" for c in expected_cols],
        })
        st.dataframe(df_cols, use_container_width=True, hide_index=True)
        st.caption("Les variables marquées « oui » sont identifiées comme **clés** (le formulaire les affiche en vert).")
    else:
        st.warning("Impossible de récupérer les colonnes attendues depuis l'API.")

    st.markdown("### Distributions de variables (exemple)")
    st.caption("Chargez un CSV dans l'onglet **Batch** pour explorer des distributions réelles.")

# ------------------------------
# 2) DOSSIER INDIVIDUEL
# ------------------------------
with TAB_DOSSIER:
    st.subheader("Prédiction dossier")

    features_input: Dict[str, Any] = {}

    # Formulaire dynamique (toutes variables de la base)
    if expected_cols:
        with st.form("form_dossier"):
            st.caption("Saisissez les caractéristiques du client (numériques/catégorielles)")
            for c in expected_cols:
                # Label coloré
                label_color = GREEN if c in key_feats else GRAY
                st.markdown(f"<span style='color:{label_color};font-weight:600'>{c}</span>", unsafe_allow_html=True)

                # Champ adapté
                if is_gender_feature(c):
                    val = st.selectbox(" ", ["", "F", "M", "Other"], index=0, key=f"sel_{c}", label_visibility="collapsed")
                    features_input[c] = None if val == "" else val
                elif is_ratio_feature(c):
                    v = st.number_input(" ", value=0.0, step=0.01, key=f"num_{c}", label_visibility="collapsed")
                    features_input[c] = v
                else:
                    txt = st.text_input(" ", value="", key=f"txt_{c}", label_visibility="collapsed")
                    features_input[c] = coerce_value(txt)

                st.divider()

            submitted = st.form_submit_button("Prédire")
    else:
        st.info("Colonnes inconnues. Saisissez un JSON libre ci-dessous.")
        features_json = st.text_area("Features (JSON)", value="{}", height=200)
        submitted = st.button("Prédire")
        try:
            features_input = json.loads(features_json)
        except Exception:
            features_input = {}

    if submitted:
        if not features_input:
            st.error("Aucune feature saisie.")
        else:
            try:
                pred = client.predict(features_input)
                if "error" in pred and pred["error"]:
                    st.error(f"Erreur API: {pred['error']}")
                else:
                    proba = float(pred.get("proba", np.nan))
                    y_hat = pred.get("y_hat", None)

                    colp1, colp2, colp3 = st.columns(3)
                    with colp1:
                        st.metric("Probabilité de défaut", f"{proba:.3f}" if np.isfinite(proba) else "NA")
                    with colp2:
                        decision = "Refus" if (np.isfinite(proba) and proba >= threshold) else "Accord"
                        st.metric("Décision", decision)
                    with colp3:
                        st.metric("Classe prédite", str(y_hat))

                    # Explication locale
                    with st.expander("🔍 Explication locale (SHAP)"):
                        shap_resp = client.shap_local(features_input)
                        if shap_resp is not None and isinstance(shap_resp, dict) and "shap_values" in shap_resp:
                            shap_values = np.array(shap_resp["shap_values"]).astype(float)
                            feature_names = shap_resp.get("feature_names", list(features_input.keys()))
                            base_value = float(shap_resp.get("base_value", np.nan))

                            df_local = pd.DataFrame({
                                "feature": feature_names,
                                "shap_value": shap_values,
                                "abs_value": np.abs(shap_values),
                            }).sort_values("abs_value", ascending=False)

                            # Highlight des features clés (utilisé juste pour info)
                            df_local["is_key"] = df_local["feature"].apply(lambda x: x in key_feats)

                            fig = px.bar(
                                df_local.head(20),
                                x="abs_value",
                                y="feature",
                                orientation="h",
                                title="Top contributeurs absolus (|SHAP|)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            if np.isfinite(base_value):
                                st.caption(f"Base value: {base_value:.4f} — (base + contributions) explique la sortie.")
                        else:
                            st.info("Pas d'endpoint /shap_local ou réponse vide. Activez-le côté API pour obtenir l'explication locale.")
            except Exception as e:
                st.exception(e)

# ------------------------------
# 3) BATCH CSV
# ------------------------------
with TAB_BATCH:
    st.subheader("Scoring en lot (CSV)")
    up = st.file_uploader("Charger un CSV (colonnes compatibles avec le modèle)", type=["csv"])

    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write(f"Aperçu ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
            st.dataframe(df.head(20))

            if st.button("Scorer le batch"):
                rows = df.to_dict(orient="records")
                out = client.predict_proba_batch(rows)

                out_df = df.copy()
                for col in out.columns:
                    out_df[col] = out[col]

                # Colonne probabilité
                proba_col = None
                for candidate in ["pred_proba", "proba", "probability"]:
                    if candidate in out_df.columns:
                        proba_col = candidate
                        break
                if proba_col is None:
                    numeric_cols = out_df.select_dtypes(include=[float, int]).columns.tolist()
                    proba_col = numeric_cols[-1] if numeric_cols else None

                if proba_col:
                    refus_rate = float((out_df[proba_col] >= threshold).mean())
                    st.metric("Taux de refus (selon seuil)", f"{refus_rate*100:.1f}%")

                    fig = px.histogram(out_df, x=proba_col, nbins=30, marginal="box",
                                       title="Distribution des probabilités de défaut (batch)")
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                tosave = io.BytesIO()
                out_df.to_csv(tosave, index=False)
                st.download_button(
                    "⬇️ Télécharger les scores (CSV)",
                    data=tosave.getvalue(),
                    file_name="batch_scored.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.exception(e)

# ------------------------------
# 4) EXPLICABILITÉ
# ------------------------------
with TAB_EXPLAIN:
    st.subheader("Explicabilité du modèle")

    st.markdown("#### Importance globale des variables")
    gi = _gi
    if gi is not None and not gi.empty and set(["feature", "importance"]).issubset(gi.columns):
        gi_sorted = gi.sort_values("importance", ascending=True)
        fig = px.bar(
            gi_sorted, x="importance", y="feature", orientation="h",
            title="Importance globale (ex. mean(|SHAP|) ou feature_importances_)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas d'endpoint /global_importance ou réponse vide. Exposez l'importance côté API ou chargez un artefact.")

    st.markdown("#### Guide d'interprétation (rappel)")
    with st.expander("Comment lire les graphiques ?"):
        st.write(
            """
            - **Importance globale** : impact moyen des features (ex. mean(|SHAP|)).
            - **Explication locale** : pour un dossier, quelles features poussent la proba vers Refus vs Accord.
            - **Seuil** : à calibrer selon vos contraintes métier (taux de défaut cible, coûts FP/FN, régulation).
            """
        )

# ------------------------------
# 5) ZONE DATA SCIENCE AVANCÉE
# ------------------------------
with TAB_DS:
    st.subheader("🔬 Data Science avancée")

    # 5.1 Probabilité de défaut & seuil
    st.markdown("### Probabilité de défaut & seuil de décision")
    colA, colB, colC = st.columns(3)
    with colA:
        pd_input = st.number_input("Probabilité de défaut (PD)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    with colB:
        th_input = st.number_input("Seuil de décision", min_value=0.0, max_value=1.0, value=float(threshold), step=0.01)
    with colC:
        decision_live = "Refus" if pd_input >= th_input else "Accord"
        color = RED if decision_live == "Refus" else GREEN
        st.markdown(f"**Décision** : <span style='color:{color}'>{decision_live}</span>", unsafe_allow_html=True)
    st.caption("Règle par défaut : PD ≥ seuil ⇒ Refus (inversez si votre modèle sort P(acceptation)).")

    # 5.2 Ratio crédit / revenu
    st.markdown("### Ratio crédit / revenu")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        credit_amount = st.number_input("Montant du crédit", min_value=0.0, value=10000.0, step=100.0)
    with col2:
        income_amount = st.number_input("Revenu mensuel net", min_value=0.0, value=3000.0, step=50.0)
    with col3:
        ratio = (credit_amount / income_amount) if income_amount > 0 else float("inf")
        ratio_display = ratio if np.isfinite(ratio) else float("nan")
        st.metric("Crédit / Revenu", f"{ratio_display:.2f}" if np.isfinite(ratio_display) else "∞")

    bands = pd.DataFrame({
        "Band": ["Faible", "Modéré", "Élevé", "Très élevé"],
        "Min": [0.0, 0.2, 0.4, 0.6],
        "Max": [0.2, 0.4, 0.6, 1.0],
    })
    fig_ratio = px.bar(bands, x="Band", y=["Max"], title="Bandes indicatives du ratio crédit/revenu (exemple)")
    st.plotly_chart(fig_ratio, use_container_width=True)

    # 5.3 Rappels d'explicabilité
    st.markdown("### Explication & interprétabilité — rappels")
    st.markdown(
        """
        - **PD (probabilité de défaut)** : estimation de la probabilité qu'un emprunteur fasse défaut.
        - **Seuil** : point de coupure convertissant une proba en décision, à définir selon **coûts métier** et **cadre réglementaire**.
        - **Ratio crédit/revenu** : indicateur simple de solvabilité, à compléter par d'autres signaux (stabilité emploi, historique, score global).
        - **Interprétabilité locale (SHAP)** : explique **pourquoi** un dossier est évalué ainsi (features pro/anti octroi).
        - **Interprétabilité globale** : hiérarchie moyenne des facteurs conduisant le modèle.
        """
    )
