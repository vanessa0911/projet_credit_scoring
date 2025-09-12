# streamlit_app.py
import os
import json
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import streamlit as st

# --- Plotly (jauge / graphs) avec fallback si non disponible ---
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# =========================
# 1) Page config — DOIT être la première commande Streamlit
# =========================
st.set_page_config(page_title="Prêt à dépenser — Scoring", page_icon="💳", layout="wide")

# =========================
# 2) Config API (ne lit st.secrets que si un secrets.toml existe)
# =========================
DEFAULT_API = "http://127.0.0.1:8000"  # API locale FastAPI
API_URL = os.getenv("API_URL", DEFAULT_API)

_SECRETS_PATHS = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path.cwd() / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _SECRETS_PATHS):
    try:
        API_URL = st.secrets.get("API_URL", API_URL)
    except Exception:
        pass

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
    try:
        if v.isdigit() or (v[0] == "-" and v[1:].isdigit()):
            return int(v)
    except Exception:
        pass
    try:
        return float(v.replace(",", "."))
    except Exception:
        pass
    return v

def years_to_days_negative(years) -> int | None:
    """Convertit des années positives en jours négatifs (convention Home Credit)."""
    if years is None or years == "":
        return None
    try:
        y = float(years)
    except Exception:
        return None
    if y < 0:
        y = 0.0
    if y > 120:
        y = 120.0
    return int(round(-y * 365.25))

def decision_badge(text: str):
    """Badge coloré pour décision (métier-friendly)."""
    text = (text or "").lower().strip()
    if text == "accordé":
        color_bg, color_text = "#16a34a", "white"
        label = "ACCORDÉ"
    elif text == "refusé":
        color_bg, color_text = "#dc2626", "white"
        label = "REFUSÉ"
    else:
        color_bg, color_text = "#6b7280", "white"
        label = text.upper() if text else "?"
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:14px;"
        f"background:{color_bg};color:{color_text};font-weight:700;letter-spacing:.3px;'>{label}</span>",
        unsafe_allow_html=True
    )

def risk_chip(prob: float):
    """Catégorie de risque simple pour non-data (faible/modéré/élevé)."""
    p = float(prob)
    if p < 0.10:
        label, bg = "Risque FAIBLE", "#16a34a"
    elif p < 0.30:
        label, bg = "Risque MODÉRÉ", "#f59e0b"
    else:
        label, bg = "Risque ÉLEVÉ", "#dc2626"
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:14px;"
        f"background:{bg};color:white;font-weight:700;'>{label}</span>",
        unsafe_allow_html=True
    )

def gauge_prob(prob: float, title: str = "Probabilité défaut (%)"):
    """Jauge Plotly 0-100% avec fallback Streamlit si Plotly indisponible."""
    prob = max(0.0, min(1.0, float(prob)))
    if not HAS_PLOTLY:
        st.write(title)
        st.progress(int(round(prob * 100)))
        st.caption("⚠️ Plotly non disponible — affichage simplifié.")
        return

    val = prob * 100.0
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

def nice_feature_name(raw: str) -> str:
    mapping = {
        "AMT_CREDIT": "Montant du crédit",
        "AMT_INCOME_TOTAL": "Revenu total",
        "DAYS_BIRTH": "Âge (jours négatifs)",
        "DAYS_EMPLOYED": "Ancienneté emploi (jours négatifs)",
        "DAYS_REGISTRATION": "Ancienneté d’enregistrement (jours négatifs)",
        "CNT_CHILDREN": "Nombre d’enfants",
        "CNT_FAM_MEMBERS": "Membres du foyer",
        "CODE_GENDER": "Sexe",
        "NAME_EDUCATION_TYPE": "Niveau d’études",
        "NAME_FAMILY_STATUS": "Situation familiale",
        "NAME_CONTRACT_TYPE": "Type de contrat",
        "NAME_INCOME_TYPE": "Type de revenu",
        "NAME_HOUSING_TYPE": "Logement",
        "REGION_RATING_CLIENT": "Indice région client (1=meilleur)",
        "FLAG_OWN_CAR": "Possède une voiture",
        "FLAG_OWN_REALTY": "Possède un bien immobilier",
        "OCCUPATION_TYPE": "Métier",
    }
    return mapping.get(raw, raw)

def percentile_from_hist(x: float, edges: list, counts: list) -> float:
    edges = np.array(edges, dtype=float)
    counts = np.array(counts, dtype=float)
    if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
        return float("nan")
    total = counts.sum()
    if total <= 0:
        return float("nan")
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx = int(np.clip(bin_idx, 0, counts.size - 1))
    cum_before = counts[:bin_idx].sum()
    left, right = edges[bin_idx], edges[bin_idx + 1]
    width = max(1e-12, right - left)
    frac_in_bin = np.clip((x - left) / width, 0.0, 1.0)
    num_in_bin = counts[bin_idx] * frac_in_bin
    cdf = (cum_before + num_in_bin) / total
    return float(cdf * 100.0)

def z_from_mean_std(x: float, mean: float, std: float) -> float:
    if std and std > 0:
        return float((x - mean) / std)
    return 0.0

def format_number_fr(x: float, decimals: int = 0) -> str:
    try:
        if decimals == 0:
            return f"{x:,.0f}".replace(",", " ").replace("\xa0", " ")
        return f"{x:,.{decimals}f}".replace(",", " ").replace("\xa0", " ")
    except Exception:
        return str(x)

def days_to_years_positive(days: float | int) -> float:
    try:
        d = float(days)
    except Exception:
        return float("nan")
    return round(abs(d) / 365.25, 1)

def interpret_sentence(feat: str, value: float, perc: float, z: float, mean: float, std: float) -> str:
    f_nice = nice_feature_name(feat)
    if np.isnan(perc):
        pos_txt = "Position par rapport à la population : non disponible."
    else:
        if perc < 45:
            pos_txt = f"{f_nice} plus bas que ~{int(round(100 - perc))}% de la population."
        elif perc > 55:
            pos_txt = f"{f_nice} plus élevé que ~{int(round(perc))}% de la population."
        else:
            pos_txt = f"{f_nice} proche de la moyenne de la population."
    if z >= 0.2:
        z_txt = f"{abs(z):.2f} écart-type au-dessus de la moyenne."
    elif z <= -0.2:
        z_txt = f"{abs(z):.2f} écart-type en dessous de la moyenne."
    else:
        z_txt = "très proche de la moyenne."
    val_txt = None
    if feat in ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION"):
        yrs = days_to_years_positive(value)
        label = {
            "DAYS_BIRTH": "≈ âge",
            "DAYS_EMPLOYED": "≈ ancienneté emploi",
            "DAYS_REGISTRATION": "≈ ancienneté d’enregistrement"
        }[feat]
        if not np.isnan(yrs):
            val_txt = f"({label} ≈ {yrs:.1f} an(s))."
    sent = f"{pos_txt} Écart à la moyenne : {z_txt}"
    if val_txt:
        sent += f" {val_txt}"
    return sent

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

def call_predict(features: dict):
    r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def get_ref_stats():
    try:
        r = requests.get(f"{API_URL}/ref_stats", timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("available"):
            return None
        return data["stats"]
    except Exception:
        return None

# =========================
# 5) En-tête (métier-friendly)
# =========================
st.title("💳 Prêt à dépenser — Dashboard de scoring")
st.markdown(
    """
    <div style="
        padding:14px 16px;
        background:#F3F4F6;
        border-left:6px solid #10B981;
        border-radius:8px;
        line-height:1.55;
    ">
      <div style="font-weight:700; font-size:17px; margin-bottom:6px;">
        🤝 Notre engagement
      </div>
      <div style="font-size:15px;">
        Derrière chaque demande de crédit, il y a une personne et un projet de vie. Notre rôle est d’<strong>accompagner</strong> avec bienveillance et clarté.
        Ce tableau de bord s’appuie sur des <strong>modèles statistiques</strong> qui estiment une probabilité de défaut à partir d’informations objectives : 
        ils ne jugent pas, ils aident à <strong>protéger</strong> à la fois nos clients et la banque contre un endettement excessif.<br><br>
        Un <strong>refus automatique</strong> n’est jamais une fin : il ouvre un échange pour vérifier les données, comprendre la situation, 
        ajuster le montant ou la durée, ou proposer des alternatives. La <strong>décision finale</strong> appartient au conseiller, pas à l’algorithme.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
# 6) SIDEBAR — Panneau technique, seuil UI, batch & payload
# =========================
with st.sidebar:
    st.header("🔧 Panneau technique")
    health = get_health()
    if health.get("status") == "ok":
        st.success("API OK", icon="✅")
        st.caption(f"Modèle : {health.get('used_model')} • Seuil API : {float(health.get('threshold', 0.5)):.2f}")
    else:
        st.error("API non joignable.", icon="⚠️")
        st.caption("Vérifie que `uvicorn api:app --reload` tourne et que l'URL est correcte.")
    st.caption(f"Endpoint : `{API_URL}`")

    # Seuil UI (explication claire)
    try:
        DEFAULT_T = float(health.get("threshold", 0.5)) if health.get("status") == "ok" else 0.5
    except Exception:
        DEFAULT_T = 0.5
    ui_threshold = st.slider(
        "Seuil décision (UI)",
        min_value=0.0, max_value=1.0, value=DEFAULT_T, step=0.01,
        help=(
            "Ce curseur **ne modifie pas** l’API. Il sert uniquement à **simuler** une politique locale : "
            "si la probabilité de défaut affichée est **au-dessus** de ce seuil, la décision simulée serait un refus. "
            "Le **seuil API** affiché plus haut est celui réellement utilisé par le service de scoring."
        )
    )

    st.divider()

    # 📦 Prédictions en lot (CSV) — maintenant dans la sidebar
    st.subheader("📦 Prédictions en lot (CSV)")
    st.caption("Dépose un CSV (colonnes brutes attendues par l’API). Les valeurs manquantes seront imputées côté modèle.")
    uploaded = st.file_uploader("CSV", type=["csv"], key="batch_csv_sidebar")
    max_rows_batch = st.number_input("Max lignes à traiter", min_value=1, max_value=5000, value=100, step=10)
    if st.button("Lancer les prédictions (lot)"):
        if uploaded is None:
            st.warning("Dépose d’abord un fichier CSV.")
        else:
            try:
                df_batch = pd.read_csv(uploaded)
                rows = df_batch.head(int(max_rows_batch))
                results = []
                with st.spinner("Prédictions en cours..."):
                    for _, row in rows.iterrows():
                        feat = {col: (None if pd.isna(row[col]) else row[col]) for col in rows.columns}
                        try:
                            pred = call_predict(feat)
                            results.append({
                                **{k: feat.get(k) for k in feat},
                                "probability_default": pred.get("probability_default"),
                                "decision_api": pred.get("decision"),
                                "threshold_api": pred.get("threshold"),
                            })
                        except Exception as e:
                            results.append({**feat, "error": str(e)})
                out_df = pd.DataFrame(results)
                st.success(f"Terminé. {len(out_df)} lignes.")
                # Petit aperçu
                st.dataframe(out_df.head(10), use_container_width=True, height=250)
                # Téléchargement
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Télécharger résultats (CSV)", csv_bytes, file_name="predictions_batch.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lecture/traitement impossible : {e}")

    st.divider()

    # 🧩 Payload → API — affiché seulement si noyau de variables complété
    st.subheader("🧩 Payload → API")
    st.caption("Aperçu des données réellement envoyées à l’API (affiché quand les variables clés sont renseignées).")
    payload_container = st.container()

# =========================
# 7) Formulaire — Informations client (années positives)
# =========================
st.subheader("🧾 Informations client")

with st.expander("Informations client", expanded=True):
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
        AMT_CREDIT = st.text_input("AMT_CREDIT (montant crédit)", value="150000", help="Montant du prêt demandé (ex: 150000)")
        AMT_INCOME_TOTAL = st.text_input("AMT_INCOME_TOTAL (revenu)", value="", help="Revenu annuel brut")
    with colC:
        AGE_YEARS = st.number_input("Âge (années, positif)", min_value=0.0, max_value=120.0, value=38.0, step=0.5, help="Saisi en années positives (sera converti en jours négatifs).")
        EMPLOY_YEARS = st.number_input("Ancienneté emploi (années, positif)", min_value=0.0, max_value=60.0, value=5.0, step=0.5, help="Saisi en années positives (converti en jours négatifs).")
    with colD:
        CNT_CHILDREN = st.text_input("CNT_CHILDREN", value="", help="Nombre d'enfants à charge")
        NAME_CONTRACT_TYPE = st.selectbox("NAME_CONTRACT_TYPE", ["", "Cash loans", "Revolving loans"], help="Type de produit")

with st.expander("Informations complémentaires (optionnelles)", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        NAME_INCOME_TYPE = st.selectbox(
            "NAME_INCOME_TYPE",
            ["", "Working", "Commercial associate", "State servant", "Pensioner", "Unemployed", "Student", "Maternity leave"]
        )
        NAME_HOUSING_TYPE = st.selectbox(
            "NAME_HOUSING_TYPE",
            ["", "House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"]
        )
    with col2:
        FLAG_OWN_CAR_BOOL = st.selectbox("FLAG_OWN_CAR", ["", "Oui", "Non"])
        FLAG_OWN_REALTY_BOOL = st.selectbox("FLAG_OWN_REALTY", ["", "Oui", "Non"])
        CNT_FAM_MEMBERS = st.text_input("CNT_FAM_MEMBERS", value="")
    with col3:
        REG_YEARS = st.number_input("Ancienneté d’enregistrement (années, positif)", min_value=0.0, max_value=60.0, value=2.0, step=0.5, help="Saisi en années positives (converti en jours négatifs).")
        REGION_RATING_CLIENT = st.selectbox("REGION_RATING_CLIENT", ["", 1, 2, 3])
    with col4:
        OCCUPATION_TYPE = st.selectbox(
            "OCCUPATION_TYPE",
            ["", "Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Security staff", "Cooking staff", "Cleaning staff", "Private service staff", "Medicine staff", "HR staff", "IT staff"]
        )

def yn(val):
    if val == "Oui":
        return "Y"
    if val == "Non":
        return "N"
    return None

DAYS_BIRTH = years_to_days_negative(AGE_YEARS)
DAYS_EMPLOYED = years_to_days_negative(EMPLOY_YEARS)
DAYS_REGISTRATION = years_to_days_negative(REG_YEARS)

features = {
    "CODE_GENDER": CODE_GENDER or None,
    "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE or None,
    "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS or None,
    "AMT_CREDIT": coerce_value(AMT_CREDIT),
    "AMT_INCOME_TOTAL": coerce_value(AMT_INCOME_TOTAL),
    "DAYS_BIRTH": DAYS_BIRTH,
    "DAYS_EMPLOYED": DAYS_EMPLOYED,
    "CNT_CHILDREN": coerce_value(CNT_CHILDREN),
    "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE or None,
    "NAME_INCOME_TYPE": NAME_INCOME_TYPE or None,
    "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE or None,
    "FLAG_OWN_CAR": yn(FLAG_OWN_CAR_BOOL),
    "FLAG_OWN_REALTY": yn(FLAG_OWN_REALTY_BOOL),
    "CNT_FAM_MEMBERS": coerce_value(CNT_FAM_MEMBERS),
    "DAYS_REGISTRATION": DAYS_REGISTRATION,
    "REGION_RATING_CLIENT": coerce_value(str(REGION_RATING_CLIENT) if REGION_RATING_CLIENT != "" else ""),
    "OCCUPATION_TYPE": OCCUPATION_TYPE or None,
}

# Affichage conditionnel du payload (variables clés complétées)
required_core = {
    "AMT_CREDIT": features.get("AMT_CREDIT"),
    "AMT_INCOME_TOTAL": features.get("AMT_INCOME_TOTAL"),
    "DAYS_BIRTH": features.get("DAYS_BIRTH"),
}
core_ok = all(v is not None and v != "" for v in required_core.values())

with st.sidebar:
    with payload_container:
        if core_ok:
            st.code(json.dumps(features, indent=2, ensure_ascii=False), language="json")
        else:
            st.info("Complète au moins : **Montant du crédit**, **Revenu total**, **Âge** (années) pour afficher le payload.")

# =========================
# 8) Actions
# =========================
colL, colR = st.columns([1, 1])
with colL:
    do_predict = st.button("🔮 Prédire")
with colR:
    do_clear = st.button("🧹 Réinitialiser la saisie")

if do_clear:
    st.experimental_rerun()

# =========================
# 9) Résultats - PREDICT + recommandations
# =========================
ref_stats = get_ref_stats()

def feature_percentile(ref_stats: dict, feat: str, value) -> float | None:
    if ref_stats is None or value is None:
        return None
    feats = ref_stats.get("features", {})
    fs = feats.get(feat)
    if not fs:
        return None
    hist = fs.get("hist", {})
    edges, counts = hist.get("edges"), hist.get("counts")
    if not edges or not counts:
        return None
    try:
        return percentile_from_hist(float(value), edges, counts)
    except Exception:
        return None

def z_from_stats(ref_stats: dict, feat: str, value) -> float | None:
    if ref_stats is None or value is None:
        return None
    fs = ref_stats.get("features", {}).get(feat, {})
    mean, std = fs.get("mean"), fs.get("std")
    try:
        return z_from_mean_std(float(value), float(mean), float(std))
    except Exception:
        return None

def build_recommendations(pred_proba: float, thr: float, feats: dict, ref_stats: dict, missing_count: int) -> list[str]:
    recs = []
    margin = pred_proba - thr
    if pred_proba < thr - 0.10:
        recs.append("Le dossier est **confortablement sous le seuil** : un accord est probable sous réserve des vérifications usuelles.")
    elif abs(margin) <= 0.03:
        recs.append("Le score est **proche du seuil** : un examen manuel est recommandé (justificatifs, cohérence des montants, stabilité).")
    else:
        recs.append("Le score est **au-dessus du seuil de risque** : privilégier un ajustement du dossier avant re-soumission.")
    if missing_count > 0:
        recs.append("Certaines informations sont manquantes : **compléter les données** ou fournir des justificatifs pour améliorer l’évaluation.")
    if ref_stats:
        p_income = feature_percentile(ref_stats, "AMT_INCOME_TOTAL", feats.get("AMT_INCOME_TOTAL"))
        p_credit = feature_percentile(ref_stats, "AMT_CREDIT", feats.get("AMT_CREDIT"))
        if p_credit is not None and p_credit >= 70:
            recs.append("**Réduire le montant du crédit** (par ex. −10 à −20%) ou **allonger la durée** pour diminuer la charge mensuelle.")
        if p_income is not None and p_income <= 40:
            recs.append("**Renforcer la preuve de revenus** (bulletins, avis d’imposition) ou **ajouter un co-emprunteur**.")
        de = feats.get("DAYS_EMPLOYED")
        try:
            de = float(de) if de is not None else None
        except Exception:
            de = None
        if de is not None and de > -365:
            recs.append("Ancienneté professionnelle récente : **attendre quelques mois** ou **fournir un CDI/contrat** peut aider.")
    if len(recs) < 2:
        recs.append("Vérifier l’exactitude des informations (dates, montants) et **fournir les derniers justificatifs**.")
        recs.append("Envisager **co-emprunteur** ou **garant** si disponible.")
    return recs

if do_predict:
    try:
        pred = call_predict(features)
        decision_api = pred.get("decision", "?")
        proba = float(pred.get("probability_default", 0.0))
        thresh_api = float(pred.get("threshold", 0.5))
        missing = pred.get("missing_features", [])

        cA, cB, cC = st.columns([1, 1, 1.2])
        with cA:
            st.subheader("Décision API")
            decision_badge(decision_api)
            st.caption(f"Seuil API : {thresh_api:.2f}")
        with cB:
            st.subheader("Catégorie de risque")
            risk_chip(proba)
            st.caption(f"Probabilité défaut : {proba:.3f}")
        with cC:
            st.subheader("Qualité des données")
            # On n'a plus expected_features ici — on utilise les manquants retournés par l’API
            total_keys = len([k for k, v in features.items() if v is not None])
            # Approximation : ratio de champs fournis parmi ceux envoyés
            ratio = 1.0 if total_keys == 0 else 1.0 - (len(missing) / max(1, total_keys))
            st.progress(int(round(max(0.0, min(1.0, ratio)) * 100)))
            st.caption(f"Champs fournis ou imputés : {int(round(max(0.0, min(1.0, ratio))*100))}%")

        gauge_prob(proba, title="Probabilité de défaut (%)")

        # Décision simulée locale (UI)
        st.info(
            f"**Décision simulée (UI)** avec seuil {ui_threshold:.2f} : "
            + ("REFUSÉ" if proba >= ui_threshold else "ACCORDÉ")
            + " • Cette simulation n’affecte pas la décision de l’API.",
            icon="🧪"
        )

        st.markdown("### 🧭 Explication de la décision & recommandations")
        ref_stats_local = ref_stats or get_ref_stats()
        if ref_stats_local:
            feats_stats = ref_stats_local.get("features", {})
            default_feats = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "CNT_CHILDREN"]
            available_feats = sorted(list(feats_stats.keys()))
            preselect = [f for f in default_feats if f in available_feats][:6]
            st.caption("Affichage d’un sous-ensemble de variables. Utilise le sélecteur ci-dessous pour en ajouter.")
            selected_feats = st.multiselect(
                "Variables à afficher (comparaison à la population)",
                options=available_feats,
                default=preselect,
                help="Ajoute/retire des variables à comparer à la population de référence."
            )
            st.caption(f"Affichées : {len(selected_feats)} / {len(available_feats)} variables disponibles.")
            if len(available_feats) > len(selected_feats):
                with st.expander("Voir les autres variables disponibles"):
                    others = [nice_feature_name(f) for f in available_feats if f not in selected_feats]
                    st.write(", ".join(others) if others else "—")

            if selected_feats:
                ccols = st.columns(min(6, len(selected_feats)))
                for idx, f in enumerate(selected_feats):
                    val = features.get(f)
                    if val is None:
                        continue
                    fs = feats_stats[f]
                    mean, std = float(fs.get("mean", 0.0)), float(fs.get("std", 0.0))
                    hist = fs.get("hist", {})
                    edges, counts = hist.get("edges", []), hist.get("counts", [])
                    perc = percentile_from_hist(float(val), edges, counts) if (edges and counts) else float("nan")
                    z = z_from_mean_std(float(val), mean, std)
                    with ccols[idx % len(ccols)]:
                        # Affichage en années positives pour DAYS_*
                        if f in ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION"):
                            yrs = days_to_years_positive(val)
                            st.metric(label=nice_feature_name(f) + " (≈ années)", value=f"{yrs:.1f} an(s)" if not np.isnan(yrs) else "—")
                            st.caption(f"Valeur brute : {int(val)} jours (négatifs = passé)")
                        else:
                            st.metric(label=nice_feature_name(f), value=f"{val}")
                        st.write(interpret_sentence(f, float(val), perc, z, mean, std))

        recs = build_recommendations(proba, thresh_api, features, ref_stats, len(missing))
        st.write("\n".join([f"• {r}" for r in recs]))

        result_json = json.dumps(pred, indent=2, ensure_ascii=False)
        st.download_button("💾 Télécharger le résultat (JSON)", result_json, file_name="prediction.json", mime="application/json")

    except Exception as e:
        st.error(f"Erreur /predict : {e}")

# =========================
# 10) 📊 Comparaison avec la population (référence = jeu d’entraînement)
# =========================
st.markdown("---")
st.subheader("📊 Comparaison avec la population (référence = jeu d’entraînement)")

ref_stats = ref_stats or get_ref_stats()
if not ref_stats:
    st.warning("Stats de référence indisponibles. Lance `make_ref_stats.py` puis redémarre l’API.")
else:
    feats_stats = ref_stats.get("features", {})
    available_feats = sorted(list(feats_stats.keys()))
    default_feats = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "CNT_CHILDREN"]
    preselect = [f for f in default_feats if f in available_feats][:6]

    st.caption("Affichage d’un sous-ensemble de variables. Utilise le sélecteur ci-dessous pour en ajouter/retirer.")
    selected_feats = st.multiselect(
        "Variables à afficher (section récapitulée)",
        options=available_feats,
        default=preselect,
        help="Choisis les variables à illustrer dans cette section."
    )
    st.caption(f"Affichées : {len(selected_feats)} / {len(available_feats)} variables disponibles.")
    if len(available_feats) > len(selected_feats):
        with st.expander("Voir les autres variables disponibles"):
            others = [nice_feature_name(f) for f in available_feats if f not in selected_feats]
            st.write(", ".join(others) if others else "—")

    client_vals = features if isinstance(features, dict) else {}
    if not selected_feats:
        st.info("Aucune variable sélectionnée pour l’affichage.")
    else:
        for f in selected_feats:
            st.markdown(f"**{nice_feature_name(f)}**")
            if client_vals.get(f) is None:
                st.caption("Valeur client inconnue → positionnement impossible.")
                continue

            try:
                val = float(client_vals[f])
            except Exception:
                st.caption("Valeur non numérique — positionnement impossible.")
                continue

            fs = feats_stats[f]
            mean, std = float(fs.get("mean", 0.0)), float(fs.get("std", 0.0))
            hist = fs.get("hist", {})
            edges, counts = hist.get("edges", []), hist.get("counts", [])

            perc = percentile_from_hist(val, edges, counts) if (edges and counts) else float("nan")
            z = z_from_mean_std(val, mean, std)

            trend = "au-dessus de la moyenne" if (not np.isnan(z) and z > 0) else "en dessous de la moyenne"
            st.caption(f"Position : **p{0 if np.isnan(perc) else int(round(perc))}** ({trend}, z = {z:.2f}) • μ={mean:.2f}, σ={std:.2f}")

            # Phrase d’interprétation (et conversion années pour DAYS_*)
            st.write(interpret_sentence(f, val, perc, z, mean, std))

            # Graph histo
            if HAS_PLOTLY and edges and counts:
                centers = (np.array(edges[:-1]) + np.array(edges[1:])) / 2.0
                fig = go.Figure()
                fig.add_trace(go.Bar(x=centers, y=counts, name="Population", opacity=0.8))
                fig.add_vline(x=val, line_color="#ef4444", line_width=3)
                x_title = nice_feature_name(f)
                if f in ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION"):
                    x_title += " (jours négatifs • passé) — note : conversion affichée en années dans le texte"
                fig.update_layout(
                    xaxis_title=x_title,
                    yaxis_title="Effectif",
                    showlegend=False,
                    height=280,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Histogramme indisponible (Plotly ou stats manquantes).")

# =========================
# 11) Pied de page
# =========================
st.markdown("---")
st.caption("💡 Conseil : si tu déploies l’API en ligne, définis API_URL dans `.streamlit/secrets.toml` ou comme variable d’environnement.")
