# streamlit_app.py
from __future__ import annotations
import os, io, json, requests
import pandas as pd
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
REF_STATS_PATH = os.path.join(ARTIFACTS_DIR, "ref_stats.json")
GLOBAL_IMP_PATH = os.path.join(ARTIFACTS_DIR, "global_importance.csv")
INTERP_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "interpretability_summary.json")

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

# --------- Helpers cached --------- #
@st.cache_data(ttl=300)
def api_get(url: str, endpoint: str):
    r = requests.get(f"{url.rstrip('/')}/{endpoint.lstrip('/')}", timeout=5)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def api_post(url: str, endpoint: str, payload: dict):
    r = requests.post(f"{url.rstrip('/')}/{endpoint.lstrip('/')}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def load_ref_stats():
    if os.path.exists(REF_STATS_PATH):
        with open(REF_STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=600)
def load_global_importance():
    if os.path.exists(GLOBAL_IMP_PATH):
        return pd.read_csv(GLOBAL_IMP_PATH)
    return pd.DataFrame(columns=["feature", "importance", "method"])

@st.cache_data(ttl=600)
def load_interp_summary():
    if os.path.exists(INTERP_SUMMARY_PATH):
        with open(INTERP_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --------- Sidebar --------- #
st.sidebar.title("Configuration")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API)
ref_stats = load_ref_stats()
global_imp = load_global_importance()
interp_summary = load_interp_summary()

# Etat API
api_ok = False
try:
    status = api_get(api_url, "/")
    api_ok = True
    st.sidebar.success(f"API OK – modèle {status.get('model_version')}")
    expected = api_get(api_url, "/expected_columns")["expected_columns"]
except Exception as e:
    st.sidebar.error("API non joignable. Mode partiel (lecture artefacts).")
    expected = list(ref_stats.get("columns", {}).keys()) if ref_stats else []

# Template CSV
if expected:
    df_empty = pd.DataFrame(columns=expected)
    buf = io.StringIO()
    df_empty.to_csv(buf, index=False)
    st.sidebar.download_button("Télécharger template CSV", buf.getvalue(), "template_scoring.csv", "text/csv")

# --------- Layout principal --------- #
st.title("Credit Scoring — Dashboard")

col1, col2 = st.columns(2)
with col1:
    if api_ok:
        st.metric("Seuil (API)", f"{status.get('threshold', 0.5):.2f}")
        st.caption(f"N features: {status.get('n_features')} | Schema: {status.get('schema_hash')}")
    else:
        st.info("Seuil inconnu (API down)")

with col2:
    if ref_stats:
        st.metric("Population de référence", f"{ref_stats.get('n_rows', '—')} observations")
    else:
        st.info("ref_stats.json introuvable")

tabs = st.tabs(["Évaluer un dossier", "Comparaison population", "Variables influentes", "Batch CSV"])

# --------- Tab 1: Évaluer un dossier --------- #
with tabs[0]:
    st.subheader("Évaluer un dossier (inférence unitaire)")
    if not expected:
        st.warning("Colonnes attendues introuvables. Assure-toi que l'API tourne ou que ref_stats.json est présent.")
    else:
        # form dynamique générée à partir des ref_stats si dispo
        form_values = {}
        colA, colB = st.columns(2)
        for i, col in enumerate(expected):
            meta = (ref_stats.get("columns", {}).get(col) if ref_stats else {}) or {}
            role = (meta.get("role") or "").lower()
            if role == "numeric":
                ui_min = meta.get("ui_min", None)
                ui_max = meta.get("ui_max", None)
                default = meta.get("mean", 0.0)
                container = colA if i % 2 == 0 else colB
                form_values[col] = container.number_input(col, value=float(default), step=1.0, format="%.6f")
            elif role == "categorical":
                top_vals = meta.get("top_values", [])[:20]
                container = colA if i % 2 == 0 else colB
                form_values[col] = container.selectbox(col, options=[""] + top_vals, index=0)
            elif role == "boolean":
                container = colA if i % 2 == 0 else colB
                form_values[col] = container.selectbox(col, options=["", "Y", "N"], index=0)
            else:
                container = colA if i % 2 == 0 else colB
                form_values[col] = container.text_input(col, value="")

        c1, c2 = st.columns([1,2])
        with c1:
            run = st.button("Évaluer")
        with c2:
            threshold_ui = st.slider("Seuil local (simulation UI)", 0.0, 1.0, float(status.get("threshold", 0.5)) if api_ok else 0.5, 0.01)

        if run:
            if not api_ok:
                st.error("API indisponible, impossible de faire l'inférence.")
            else:
                try:
                    js = api_post(api_url, "/predict", {"data": form_values})
                    proba = js["probability"]
                    decision_api = js["decision"]
                    decision_ui = int(proba >= threshold_ui)
                    st.success(f"Probabilité de défaut: {proba:.4f}")
                    st.write(f"Décision API (seuil officiel {status.get('threshold', 0.5):.2f}) : **{['ACCEPTÉ','REFUSÉ'][decision_api]}**")
                    st.write(f"Décision (seuil UI {threshold_ui:.2f}) : **{['ACCEPTÉ','REFUSÉ'][decision_ui]}**")
                except Exception as e:
                    st.error(f"Erreur API: {e}")

# --------- Tab 2: Comparaison population --------- #
with tabs[1]:
    st.subheader("Comparaison vs population de référence")
    if not ref_stats:
        st.info("ref_stats.json manquant.")
    else:
        # affiche quelques features numériques avec stats
        numeric_cols = [c for c, meta in ref_stats["columns"].items() if (meta.get("role") or "").lower() == "numeric"]
        sel = st.multiselect("Choisir des variables numériques", numeric_cols[:10], default=numeric_cols[:5] if len(numeric_cols)>5 else numeric_cols)
        if sel:
            stats = []
            for c in sel:
                meta = ref_stats["columns"][c]
                stats.append({
                    "feature": c,
                    "mean": meta.get("mean"),
                    "std": meta.get("std"),
                    "p1": meta.get("p1"),
                    "p50": meta.get("p50"),
                    "p99": meta.get("p99"),
                })
            st.dataframe(pd.DataFrame(stats))

# --------- Tab 3: Variables influentes --------- #
with tabs[2]:
    st.subheader("Importances globales")
    if global_imp.empty:
        st.info("global_importance.csv manquant.")
    else:
        topn = st.slider("Top N", 5, min(30, len(global_imp)), 15)
        top_df = global_imp.sort_values("importance", ascending=False).head(topn)
        st.bar_chart(data=top_df.set_index("feature")["importance"])
    if interp_summary:
        st.caption("Résumé interprétabilité (phrases clés)")
        for f, txt in (interp_summary.get("explanations", {}) or {}).items():
            st.write(f"- **{f}** : {txt}")

# --------- Tab 4: Batch CSV --------- #
with tabs[3]:
    st.subheader("Prédiction par lot (CSV)")
    if not api_ok:
        st.warning("API indisponible.")
    else:
        up = st.file_uploader("Charger un CSV avec les colonnes attendues", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                st.write("Aperçu :", df.head())
                payload = {"records": df.to_dict(orient="records")}
                js = api_post(api_url, "/predict_proba_batch", payload)
                out = df.copy()
                out["probability"] = js["probabilities"]
                out["decision"] = js["decisions"]
                st.success(f"Inférence OK ({js['count']} lignes)")
                st.dataframe(out.head(50))
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Télécharger les résultats", buf.getvalue(), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Erreur: {e}")
