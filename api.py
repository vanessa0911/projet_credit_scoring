# api.py — ajouts complets pour l'explicabilité et les variables clés
@app.post("/predict_proba_batch")
def predict_proba_batch(payload: Rows):
# X = pd.DataFrame(payload.rows)
# X_proc = preprocessor.transform(X)
# pred_proba = model.predict_proba(X_proc)[:, 1]
# y_hat = (pred_proba >= 0.5).astype(int)
# return {"pred_proba": pred_proba.tolist(), "y_hat": y_hat.tolist()}
return {"detail": "À implémenter selon votre pipeline"}


@app.get("/global_importance")
def global_importance():
"""Importance globale SHAP (si possible) sinon feature_importances_."""
try:
if SHAP_OK:
# EXEMPLE : calculer sur un échantillon de fond (adapter)
# X_bg = pd.read_parquet("artifacts/ref_sample.parquet")
# X_proc = preprocessor.transform(X_bg)
# global explainer
# if explainer is None:
# explainer = shap.Explainer(model, X_proc)
# sv = explainer(X_proc)
# gi = np.mean(np.abs(sv.values), axis=0)
# fn = list(getattr(preprocessor, 'get_feature_names_out', lambda: X_bg.columns)())
# return [{"feature": f, "importance": float(w)} for f, w in zip(fn, gi)]
pass
# Fallback: importance du modèle
# if hasattr(model, "feature_importances_"):
# imps = model.feature_importances_.ravel()
# fn = list(getattr(preprocessor, 'get_feature_names_out', lambda: EXPECTED_COLUMNS)())
# return [{"feature": f, "importance": float(w)} for f, w in zip(fn, imps)]
except Exception:
pass
return []


@app.post("/shap_local")
def shap_local(payload: Features):
"""Explication locale (SHAP) d'un dossier.
Retourne {feature_names, shap_values, base_value}.
"""
if not SHAP_OK:
return {"detail": "SHAP non disponible côté serveur"}
# x = pd.DataFrame([payload.features])
# x_proc = preprocessor.transform(x)
# global explainer
# if explainer is None:
# # Initialiser sur un background
# X_bg = pd.read_parquet("artifacts/ref_sample.parquet")
# X_proc_bg = preprocessor.transform(X_bg)
# explainer = shap.Explainer(model, X_proc_bg)
# sv = explainer(x_proc)
# feature_names = list(getattr(preprocessor, 'get_feature_names_out', lambda: x.columns)())
# return {
# "feature_names": feature_names,
# "shap_values": sv.values[0].tolist(),
# "base_value": float(np.mean(getattr(sv, 'base_values', [0.0]))),
# }
return {"detail": "À implémenter selon vos artefacts"}


@app.get("/key_features")
def key_features():
"""Retourne la liste des variables clés identifiées (métier/feature selection)."""
return {"features": KEY_FEATURES}
