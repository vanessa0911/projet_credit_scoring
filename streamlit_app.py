# streamlit_app.py — version complète mise à jour
gi_sorted = gi.sort_values("importance", ascending=True)
fig = px.bar(gi_sorted, x="importance", y="feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)
else:
st.info("Pas d'endpoint /global_importance. Exposez l'importance globale (SHAP moyen | importance modèle) côté API ou chargez un CSV d'artefacts.")


st.markdown("#### Guide d'interprétation (rappel)")
with st.expander("Comment lire les graphiques ?"):
st.write(
"""
- **Importance globale** : impact moyen des features (ex. mean(|SHAP|)).
- **Explication locale** : pour un dossier, quelles features poussent la proba vers le refus vs l'accord.
- **Seuil** : à calibrer selon vos contraintes métier (taux de défaut cible, coûts FP/FN, régulation).
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
st.markdown(f"**Décision** : <span style='color:{color}'>{decision_live}</span>", unsafe_allow_html=True)


st.caption("Règle par défaut : PD ≥ seuil ⇒ Refus. Adaptez la convention si votre modèle sort une proba d'acceptation.")


# 5.2 Ratio crédit / revenu
st.markdown("### Ratio crédit / revenu")
col1, col2, col3 = st.columns([1,1,1])
with col1:
credit_amount = st.number_input("Montant du crédit", min_value=0.0, value=10000.0, step=100.0)
with col2:
income_amount = st.number_input("Revenu mensuel net", min_value=0.0, value=3000.0, step=50.0)
with col3:
ratio = (credit_amount / income_amount) if income_amount > 0 else math.inf
ratio_display = np.nan if math.isinf(ratio) else ratio
st.metric("Crédit / Revenu", f"{ratio_display:.2f}" if not np.isnan(ratio_display) else "∞")


# Indication de bandes (exemple, à adapter aux politiques internes)
bands = pd.DataFrame({
"Band": ["Faible", "Modéré", "Élevé", "Très élevé"],
"Min": [0.0, 0.2, 0.4, 0.6],
"Max": [0.2, 0.4, 0.6, 1.0],
})
fig_ratio = px.bar(bands, x="Band", y=["Max"], title="Bandes indicatives du ratio crédit/revenu (exemple)")
st.plotly_chart(fig_ratio, use_container_width=True)


# 5.3 Explicabilité – rappel
st.markdown("### Explication & interprétabilité – rappels utiles")
st.markdown(
"""
- **PD (probabilité de défaut)** : estimation de la probabilité qu'un emprunteur fasse défaut.
- **Seuil** : point de coupure convertissant une proba en décision. Il dépend du **coût métier** (perte en cas de défaut vs manque à gagner si refus injustifié) et des contraintes **réglementaires**.
- **Ratio crédit/revenu** : indicateur simple de solvabilité – à compléter avec d'autres signaux (ancienneté, scoring, historique).
- **Interprétabilité locale (SHAP)** : explique **pourquoi** un dossier est évalué ainsi (features pro/anti octroi).
- **Interprétabilité globale** : hiérarchie moyenne des facteurs conduisant le modèle.
"""
)
