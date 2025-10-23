# streamlit_app.py
base_value = float(shap_resp.get("base_value", np.nan))
feature_names = shap_resp.get("feature_names", list(features_input.keys()))


df_local = pd.DataFrame({
"feature": feature_names,
"shap_value": shap_values,
"abs_value": np.abs(shap_values),
}).sort_values("abs_value", ascending=False)


st.write("Top contributeurs (absolu)")
fig = px.bar(df_local.head(15), x="abs_value", y="feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)


st.caption(f"Base value: {base_value:.4f} – La somme (base + contributions) explique la sortie (selon le lien du modèle).")
else:
st.info("Pas d'endpoint /shap_local. Vous pouvez l'ajouter côté API (voir suggestions) ou calculer côté UI si le modèle et ses artefacts sont disponibles.")


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
st.write(f"Aperçu ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
st.dataframe(df.head(20))


if st.button("Scorer le batch"):
rows = df.to_dict(orient="records")
out = client.predict_proba_batch(rows)


# Fusion si l'API renvoie seulement proba/y_hat
out_df = df.copy()
for col in out.columns:
out_df[col] = out[col]


# KPIs batch
proba_col = "pred_proba" if "pred_proba" in out_df.columns else out_df.select_dtypes(float).columns[-1]
refus_rate = float((out_df[proba_col] >= threshold).mean()) if proba_col in out_df else np.nan
st.metric("Taux de refus (selon seuil)", f"{refus_rate*100:.1f}%")


# Histogramme des probabilités
if proba_col in out_df:
fig = px.histogram(out_df, x=proba_col, nbins=30, marginal="box")
st.plotly_chart(fig, use_container_width=True)


# Téléchargement
tosave = io.BytesIO()
out_df.to_csv(tosave, index=False)
st.download_button("⬇️ Télécharger les scores (CSV)", data=tosave.getvalue(), file_name="batch_scored.csv", mime="text/csv")


except Exception as e:
st.exception(e)


# ------------------------------
# 4) EXPLICABILITÉ
# ------------------------------
with TAB_EXPLAIN:
st.subheader("Explicabilité du modèle")


st.markdown("#### Importance globale des variables")
gi = client.global_importance()
if gi is not None and not gi.empty and set(["feature", "importance"]).issubset(gi.columns):
gi_sorted = gi.sort_values("importance", ascending=True)
fig = px.bar(gi_sorted, x="importance", y="feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)
else:
st.info("Pas d'endpoint /global_importance. Vous pouvez exposer les importances (SHAP global moyen | importance modèle) côté API ou charger un CSV d'artefacts.")


st.markdown("#### Guide d'interprétation (rappel)")
with st.expander("Comment lire les graphiques ?"):
st.write(
"""
- **Importance globale** : mesures moyennes d’impact des features sur les prédictions (ex. mean(|SHAP|)).
- **Explication locale** : pour un dossier, quelles features poussent la prédiction vers le refus vs l’accord.
- **Seuil** : à calibrer selon vos contraintes métier (taux de défaut cible, coût FP/FN, etc.).
"""
)


st.markdown("---")
st.caption("💡 Pour activer les explications SHAP côté API, voir le snippet FastAPI ci-dessous.")
