# streamlit_app.py
import os
import json
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# 1) Page config — DOIT être la première commande Streamlit
# =========================
st.set_page_config(page_title="Prêt à dépenser — Scoring", page_icon="💳", layout="wide")

# =========================
# 2) Config API (ne lit st.secrets que si un secrets.toml existe)
# =========================
DEFAULT_API = "http://127.0.0.1:8000"  # API locale FastAPI
API_URL = os.getenv("API_URL", DEFAULT_API)

# chemins possibles du secrets.toml
_SECRETS_PATHS = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path.cwd() / ".streamlit" / "secrets.toml",
]

# on NE lit st.secrets que si le fichier existe (évite l'erreur rouge "No secrets files found")
if any(p.exists() for p in _SECRETS_PATHS):
    try:
        API_URL = st.secrets.get("API_URL", API_URL)
    except Exception:
        pass

st.title("💳 Prêt à dépenser — Dashboard de scoring")

# =========================
# 3) Utils
# =========================
def coerce_value(v: str):
    """Convertit une saisie texte en int/float si possible, sinon string/None."""
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    # int ?
    try:
        if v.isdigit() or (v[0] == "-" and v[1:].isdigit()):
            return int(v)
    except Exception:
        pass
    # float ?
    try:
        return float(v.replace(",", "."))
    except Exception:
        pass
    return v

def decision_badge(text: str):
    """Affiche un badge coloré (HTML simple)."""
    text = (text or "").lower().strip()
    if text == "accordé":
        color_bg, color_text = "#16a34a", "white"   # vert
        label = "ACCORDÉ"
    elif text == "refusé":
        color_bg, color_text = "#dc2626", "white"   # rouge
        label = "REFUSÉ"
    else:
        color_bg, color_text = "#6b7280", "white"   # gris
        label = text.upper() if text else "?"
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:14px;"
        f"background:{color_bg};color:{color_text};font-weight:700;letter-spacing:.3px;'>{label}</span>",
        unsafe_allow_html=True
    )

def gauge_prob(prob: float, title: str = "Probabilité défaut (%)"):
    """Jauge Plotly 0-100%."""
    val = max(0.0, min(1.0, float(prob))) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={'suffix': "%", 'valueformat': ".1f"},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.35},
                'steps': [
                    {'range': [0, 20], 'color': '#dcfce7'},
                    {'range': [20, 40], 'color': '#bbf7d0'},
                    {'range': [40, 60], 'color': '#fef3c7'},
                    {'range': [60, 80], 'color': '#fdba74'},
                    {'range': [80, 100], 'color': '#fecaca'},
                ],
            }
        )
    )
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=10), height=260)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 4) Appels API
# =========================
@st.cache_data(show_spinner=False, ttl=60)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "ko", "error": str(e)}

@st.cache_data(show_spinner=False, ttl=300)
def get_expected_features():
    try:
        r = requests.get(f"{API_URL}/expected_features", timeout=10)
        r.raise_for_status()
        return r.json().get("expected_features", [])
    except Exception:
        return []

def call_predict(features: dict):
    r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=20)
    r.raise_for_status()
    return r.json()

def call_explain(features: dict, top_k: int = 8):
    r = requests.post(f"{API_URL}/explain", params={"top_k": top_k}, json={"features": features}, timeout=20)
    r.raise_for_status()
    return r.json()

# =========================
# 5) Bandeau statut + paramètres (avec explications)
# =========================
CUSTOM_T = None  # valeur par défaut si le slider n'est pas rendu (sécurité)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🛰️ État de l’API")
    health = get_health()
    if health.get("status") == "ok":
        st.success(f"API OK • modèle = {health.get('used_model')} • seuil API = {health.get('threshold')}")
    else:
        st.error("API non joignable. Vérifie que `uvicorn api:app --reload` tourne et que l'URL est correcte.")
        st.stop()

with col2:
    st.subheader("⚙️ Paramètres")
    st.write(f"Endpoint API: `{API_URL}`")
    expected_cols = get_expected_features()
    with st.expander("Voir les colonnes attendues (avant encodage)"):
        st.caption(
            "Ce sont les **variables brutes** que l’API attend **avant imputation et encodage** "
            "(ex. One-Hot). Tu peux laisser des champs vides : ils seront imputés. "
            "Les colonnes en trop sont ignorées."
        )
        st.write(expected_cols if expected_cols else "—")

    # Slider de seuil UI (visualisation locale, ne change pas l'API)
    try:
        DEFAULT_T = float(health.get("threshold", 0.5))
    except Exception:
        DEFAULT_T = 0.5
    CUSTOM_T = st.slider(
        "Seuil décision (UI)",
        min_value=0.0, max_value=1.0, value=DEFAULT_T, step=0.01,
        help="Utilisé seulement pour l'affichage local. L'API garde son propre seuil."
    )

# —— Explication synthétique des indicateurs en haut ——
model_name = str(health.get("used_model", "?"))
thr = float(health.get("threshold", 0.5))
st.info(
    f"""
**Que signifient ces indicateurs ?**

- **API OK** : le serveur FastAPI est démarré et a chargé le modèle. L’endpoint `/health` répond correctement.
- **Modèle** : `{model_name}`.  
  *Exemple :* `isotonic` = régression logistique calibrée par régression isotone, pour obtenir des **probabilités mieux calibrées**.
- **Seuil API** : **{thr:.2f}**.  
  Règle de décision **côté serveur** : si `probabilité de défaut ≥ {thr:.2f}` ⇒ **refusé**, sinon ⇒ **accordé**.

> 💡 Le **slider “Seuil décision (UI)”** à droite sert juste à **simuler** un seuil différent côté interface (il **ne change pas** la logique sur le serveur).
"""
)

st.markdown("---")

# =========================
# 6) Formulaire — mode simple + mode avancé (JSON)
# =========================
st.subheader("🧾 Données client")

with st.expander("Mode simple (tu peux laisser la plupart vides) ✅", expanded=True):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        CODE_GENDER = st.selectbox("CODE_GENDER", ["", "F", "M"])
        NAME_EDUCATION_TYPE = st.selectbox(
            "NAME_EDUCATION_TYPE",
            ["", "Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
        )
        NAME_FAMILY_STATUS = st.selectbox(
            "NAME_FAMILY_STATUS",
            ["", "Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
        )
    with colB:
        AMT_CREDIT = st.text_input("AMT_CREDIT (montant crédit)", value="150000")
        AMT_INCOME_TOTAL = st.text_input("AMT_INCOME_TOTAL (revenu)", value="")
    with colC:
        DAYS_BIRTH = st.text_input("DAYS_BIRTH (jours négatifs)", value="-14000")
        DAYS_EMPLOYED = st.text_input("DAYS_EMPLOYED (jours négatifs)", value="")
    with colD:
        CNT_CHILDREN = st.text_input("CNT_CHILDREN", value="")
        NAME_CONTRACT_TYPE = st.selectbox("NAME_CONTRACT_TYPE", ["", "Cash loans", "Revolving loans"])

    simple_payload = {
        "CODE_GENDER": CODE_GENDER or None,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE or None,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS or None,
        "AMT_CREDIT": coerce_value(AMT_CREDIT),
        "AMT_INCOME_TOTAL": coerce_value(AMT_INCOME_TOTAL),
        "DAYS_BIRTH": coerce_value(DAYS_BIRTH),
        "DAYS_EMPLOYED": coerce_value(DAYS_EMPLOYED),
        "CNT_CHILDREN": coerce_value(CNT_CHILDREN),
        "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE or None,
    }

with st.expander("Mode avancé (coller/éditer JSON manuellement) 🧪"):
    default_keys = expected_cols[:10] or ["CODE_GENDER", "AMT_CREDIT", "DAYS_BIRTH"]
    default_json = json.dumps({k: None for k in default_keys}, indent=2)
    raw_json = st.text_area("JSON features (optionnel)", value=default_json, height=220)
    use_raw = st.checkbox("Utiliser le JSON avancé (sinon le mode simple au-dessus)", value=False)
    advanced_payload = None
    if use_raw:
        try:
            advanced_payload = json.loads(raw_json)
            st.info("JSON valide ✅")
        except Exception as e:
            st.error(f"JSON invalide : {e}")

# Choix du payload
features = advanced_payload if (use_raw and advanced_payload is not None) else simple_payload

with st.expander("Payload envoyé à l'API (aperçu)"):
    st.code(json.dumps(features, indent=2, ensure_ascii=False), language="json")

# =========================
# 7) Actions
# =========================
colL, colR = st.columns([1, 1])
with colL:
    do_predict = st.button("🔮 Prédire")
with colR:
    do_explain = st.button("🪄 Expliquer (top contributions)")

# =========================
# 8) Résultats - PREDICT
# =========================
if do_predict:
    try:
        pred = call_predict(features)
        decision_api = pred.get("decision", "?")
        proba = float(pred.get("probability_default", 0.0))
        thresh_api = float(pred.get("threshold", 0.5))

        cA, cB = st.columns([1, 1])
        with cA:
            st.subheader("Décision API")
            decision_badge(decision_api)
            st.caption(f"Seuil API : {thresh_api:.2f}")
        with cB:
            st.subheader("Décision (UI)")
            t_ui = CUSTOM_T if CUSTOM_T is not None else thresh_api
            decision_ui = "refusé" if proba >= t_ui else "accordé"
            decision_badge(decision_ui)
            st.caption(f"Seuil UI : {t_ui:.2f}")

        gauge_prob(proba, title="Probabilité de défaut (%)")

        if pred.get("missing_features"):
            with st.expander("Variables manquantes imputées (info)"):
                st.write(pred["missing_features"])

        # Bouton téléchargement JSON
        result_json = json.dumps(pred, indent=2, ensure_ascii=False)
        st.download_button("💾 Télécharger le résultat (JSON)", result_json, file_name="prediction.json", mime="application/json")

    except Exception as e:
        st.error(f"Erreur /predict : {e}")

# =========================
# 9) Résultats - EXPLAIN
# =========================
if do_explain:
    try:
        expl = call_explain(features, top_k=8)
        decision_api = expl.get("decision", "?")
        proba = float(expl.get("probability_default", 0.0))
        thresh_api = float(expl.get("threshold", 0.5))

        cA, cB = st.columns([1, 1])
        with cA:
            st.subheader("Décision API")
            decision_badge(decision_api)
            st.caption(f"Seuil API : {thresh_api:.2f}")
        with cB:
            st.subheader("Décision (UI)")
            t_ui = CUSTOM_T if CUSTOM_T is not None else thresh_api
            decision_ui = "refusé" if proba >= t_ui else "accordé"
            decision_badge(decision_ui)
            st.caption(f"Seuil UI : {t_ui:.2f}")

        gauge_prob(proba, title="Probabilité de défaut (%)")
        st.caption(f"Biais (intercept, log-odds) : {float(expl.get('bias', 0.0)):.4f}")

        contrib_df = pd.DataFrame(expl.get("top_contributions", []))
        if not contrib_df.empty:
            contrib_df["abs"] = contrib_df["contribution"].abs()
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Top contributions (log-odds)")
                st.dataframe(contrib_df[["feature", "contribution"]])
            with c2:
                st.subheader("Barres (positives = augmentent le risque)")
                st.bar_chart(contrib_df.set_index("feature")[["contribution"]])
        else:
            st.info("Pas de contributions retournées.")

        # Bouton téléchargement JSON
        explain_json = json.dumps(expl, indent=2, ensure_ascii=False)
        st.download_button("💾 Télécharger l'explication (JSON)", explain_json, file_name="explain.json", mime="application/json")

    except Exception as e:
        st.error(f"Erreur /explain : {e}")

# =========================
# 10) Prédictions en lot (CSV)
# =========================
st.markdown("---")
st.subheader("📦 Prédictions en lot (CSV)")
uploaded = st.file_uploader("Déposer un CSV (colonnes brutes comme attendues par l'API)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Aperçu CSV :", df.head())
        max_rows = st.number_input(
            "Nombre max. de lignes à traiter",
            min_value=1, max_value=int(df.shape[0]), value=min(100, df.shape[0])
        )
        rows = df.head(int(max_rows))

        results = []
        with st.spinner("Prédictions en cours..."):
            for _, row in rows.iterrows():
                # NaN -> None pour l'API
                feat = {col: (None if pd.isna(row[col]) else row[col]) for col in rows.columns}
                try:
                    pred = call_predict(feat)
                    results.append({
                        **{k: feat.get(k) for k in feat},  # rejouer les features pour la trace
                        "probability_default": pred.get("probability_default"),
                        "decision_api": pred.get("decision"),
                        "threshold_api": pred.get("threshold"),
                    })
                except Exception as e:
                    results.append({**feat, "error": str(e)})

        out_df = pd.DataFrame(results)
        st.success(f"Terminé. {len(out_df)} lignes.")
        st.dataframe(out_df)

        # Téléchargement CSV
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Télécharger les résultats CSV", csv_bytes, file_name="predictions_batch.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Lecture CSV impossible : {e}")

st.markdown("---")
st.caption("💡 Conseil : si tu déploies l’API en ligne, définis API_URL dans `.streamlit/secrets.toml` ou comme variable d’environnement.")
