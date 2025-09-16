# streamlit_app.py
# =========================================================
# Tableau de bord "Prêt à dépenser" — Scoring & Explicabilité
# - État API -> sidebar (DS only)
# - Calculette années/mois -> jours négatifs (pleine largeur)
# - Informations client (inclut toutes les variables des ratios)
# - Variables complémentaires (toutes les colonnes du modèle)
# - Évaluation: seuil (UI) ajustable + comparatif p vs t + jauge bullet
#   + Ratios (calculés) intégrés ici avec impact & importance
# - Comparaison population: explication par visuel + impact attendu
# - Cohorte: KPIs + comparatif cohorte vs population + interprétation
# - Importances globales
# =========================================================

import streamlit as st
st.set_page_config(page_title="Prêt à dépenser — Scoring", page_icon="💳", layout="wide")

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

# -----------------------------
# Libellés FR + aides contextuelles
# -----------------------------
VAR_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu annuel",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Annuité",
    "DAYS_BIRTH": "Âge (jours négatifs Kaggle)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours négatifs Kaggle)",
    "CNT_CHILDREN": "Nombre d’enfants",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "CODE_GENDER": "Sexe",
    "NAME_FAMILY_STATUS": "Situation familiale",
    # dérivées fréquentes
    "AGE_YEARS": "Âge (années)",
    "PAYMENT_RATE": "Taux de paiement (annuité/crédit)",
    "CREDIT_INCOME_RATIO": "Ratio crédit / revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenu",
    "CREDIT_TERM_MONTHS": "Durée du crédit (mois)",
    "INCOME_PER_PERSON": "Revenu par personne au foyer",
    "CHILDREN_RATIO": "Ratio enfants / foyer",
    "OWN_CAR_BOOL": "Possède une voiture (booléen)",
    "OWN_REALTY_BOOL": "Possède un bien immobilier (booléen)",
    "MISSING_COUNT_ROW": "Nb de valeurs manquantes (ligne)",
    "DOC_COUNT": "Nb de documents fournis",
    "EXT_SOURCES_MEAN": "Score externe (moyenne)",
    "EXT_SOURCES_SUM": "Score externe (somme)",
    "EMPLOY_YEARS": "Ancienneté emploi (années)",
    "REG_YEARS": "Ancienneté enregistrement (années)",
    "AMT_GOODS_PRICE": "Prix du bien",
}

VAR_HELP = {
    "AMT_INCOME_TOTAL": "Revenu annuel brut déclaré.",
    "AMT_CREDIT": "Montant total du crédit demandé.",
    "AMT_ANNUITY": "Annuité de remboursement estimée.",
    "DAYS_BIRTH": "Âge exprimé en jours négatifs (convention Home Credit). Utilise la calculette ci-dessous.",
    "DAYS_EMPLOYED": "Ancienneté exprimée en jours négatifs (convention Home Credit). Utilise la calculette.",
    "NAME_CONTRACT_TYPE": "Type de contrat (ex: Cash loans, Revolving loans).",
    "CODE_GENDER": "Sexe (M/F ; selon la donnée source).",
    "NAME_FAMILY_STATUS": "Statut familial (Married, Single, Separated, etc.).",
}

TOKEN_FR = {
    "AMT": "Montant", "CREDIT": "Crédit", "INCOME": "Revenu", "ANNUITY": "Annuité",
    "DAYS": "Jours", "BIRTH": "Naissance", "EMPLOYED": "Emploi", "EMPLOY": "Emploi",
    "YEARS": "Années", "REGISTRATION": "Enregistrement", "REG": "Enreg.",
    "NAME": "Libellé", "CODE": "Code", "GENDER": "Sexe", "FAMILY": "Famille",
    "STATUS": "Statut", "OWN": "Possession", "REALTY": "Immobilier",
    "FLAG": "Indicateur", "DOCUMENT": "Document", "CNT": "Nombre",
    "FAM": "Famille", "MEMBERS": "Membres", "CHILDREN": "Enfants",
    "GOODS": "Bien", "PRICE": "Prix", "EXT": "Externe", "SOURCE": "Source",
    "RATIO": "Ratio", "RATE": "Taux", "TERM": "Durée", "MISSING": "Manquantes"
}

def label_fr(col: str) -> str:
    if col in VAR_LABELS:
        return VAR_LABELS[col]
    parts = col.split("_")
    fr_parts = []
    for p in parts:
        up = p.upper()
        fr_parts.append(TOKEN_FR.get(up, p.capitalize()))
    out = " ".join(fr_parts)
    out = out.replace("Libellé Contract Type", "Type de contrat")
    out = out.replace("Code Gender", "Sexe")
    out = out.replace("Name Family Status", "Situation familiale")
    return out

def help_fr(col: str) -> Optional[str]:
    return VAR_HELP.get(col)

# Heuristiques "sens du risque" utilisables partout (ratios & cohortes)
RISK_UP = {
    # plus élevé → plutôt plus risqué
    "AMT_CREDIT", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "PAYMENT_RATE",
    "CREDIT_TERM_MONTHS",
}
PROTECTIVE_UP = {
    # plus élevé → plutôt protecteur
    "AMT_INCOME_TOTAL", "AGE_YEARS", "INCOME_PER_PERSON", "EMPLOY_YEARS",
}

# -------- API URL robuste --------
def get_api_url() -> str:
    default = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        home_secrets = Path.home() / ".streamlit" / "secrets.toml"
        proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
        if home_secrets.exists() or proj_secrets.exists():
            try:
                return st.secrets.get("API_URL", default)  # type: ignore[attr-defined]
            except Exception:
                return default
        return default
    except Exception:
        return default

API_URL = get_api_url()

# -------- Utils HTTP --------
def http_get_json(url: str, timeout: int = 6) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        return False, {}, str(e)

def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        try:
            return False, r.json(), str(e)  # type: ignore[name-defined]
        except Exception:
            return False, {}, str(e)

# -------- Chargement d’artefacts locaux --------
@st.cache_data(show_spinner=False)
def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

ARTIFACT_DIR = Path("artifacts")
REF_STATS = load_json_if_exists(ARTIFACT_DIR / "ref_stats.json")
SEG_STATS = load_json_if_exists(ARTIFACT_DIR / "seg_stats.json")
GLOBAL_IMP = None
try:
    gi_path = ARTIFACT_DIR / "global_importance.csv"
    if gi_path.exists():
        GLOBAL_IMP = pd.read_csv(gi_path)
except Exception:
    GLOBAL_IMP = None

# -------- Sidebar (technique) --------
with st.sidebar:
    st.markdown("### ⚙️ Technique (Data/DS)")
    api_url_input = st.text_input("API URL", API_URL, help="Modifie l’URL si l’API est déployée ailleurs.")
    if api_url_input != API_URL:
        API_URL = api_url_input

    @st.cache_data(ttl=30, show_spinner=False)
    def get_expected_cols(api_url: str) -> Optional[list[str]]:
        okc, js, _ = http_get_json(f"{api_url}/expected_columns")
        if okc and isinstance(js, dict) and "expected_columns" in js:
            return list(js["expected_columns"])
        return None

    @st.cache_data(ttl=60, show_spinner=False)
    def get_value_domains(api_url: str) -> Dict[str, list]:
        okd, jsd, _ = http_get_json(f"{api_url}/value_domains")
        if okd and isinstance(jsd, dict) and "domains" in jsd:
            out = {}
            for k, v in jsd["domains"].items():
                if isinstance(v, list):
                    out[k] = [None if x in (None, "None") else str(x) for x in v]
            return out
        return {}

    # État de l’API (déplacé ici)
    st.markdown("#### 🛰️ État de l’API")
    ok_root, root_json, err_root = http_get_json(f"{API_URL}/")
    if not ok_root:
        st.error(f"API non joignable\n{err_root}")
    else:
        model_label = root_json.get("chosen_model") or root_json.get("model_class") or "?"
        dt = root_json.get("decision_threshold")
        st.success(f"Modèle: **{model_label}**\n\nSeuil API: `{dt}`")
        st.caption("Infos techniques destinées aux Data Scientists.")

    exp_cols = get_expected_cols(API_URL) or []
    VALUE_DOMAINS = get_value_domains(API_URL)

    st.markdown("#### 🧩 Payload → API (aperçu)")
    st.caption("Aperçu des champs non vides envoyés à l’API.")

# -------- Bandeau "engagement" --------
st.markdown("### 💬 Notre engagement")
st.info(
    "Derrière chaque dossier, il y a une personne. Notre rôle est d’accompagner au mieux, "
    "avec des indicateurs **statistiques** conçus pour protéger à la fois nos clients et la banque. "
    "Une décision négative n’est jamais punitive : elle évite un risque de surendettement et oriente vers des solutions adaptées."
)

# -------- Calculette (pleine largeur) --------
st.markdown("### 🧮 Calculette (années/mois → jours négatifs Kaggle)")
for key, default in [("calc_years", 30.0), ("calc_months", 0.0),
                     ("DAYS_BIRTH_val", -14000.0), ("DAYS_EMPLOYED_val", -2000.0)]:
    if key not in st.session_state:
        st.session_state[key] = default

cY, cM, cRes, cBtns1, cBtns2 = st.columns([1,1,1,2,2])
with cY:
    years = st.number_input("Années", min_value=0.0, step=0.5, value=st.session_state["calc_years"])
    st.session_state["calc_years"] = years
with cM:
    months = st.number_input("Mois", min_value=0.0, max_value=11.0, step=1.0, value=st.session_state["calc_months"])
    st.session_state["calc_months"] = months
with cRes:
    days = -(years * 365.25 + months * 30.4375)
    st.metric("→ Jours (Kaggle)", f"{days:.0f}")
    st.caption("Négatif attendu par le dataset (Home Credit).")
with cBtns1:
    if st.button("Renseigner Âge (DAYS_BIRTH)"):
        st.session_state["DAYS_BIRTH_val"] = float(days)
        st.success("Âge (jours) mis à jour pour le formulaire.")
with cBtns2:
    if st.button("Renseigner Ancienneté (DAYS_EMPLOYED)"):
        st.session_state["DAYS_EMPLOYED_val"] = float(days)
        st.success("Ancienneté (jours) mise à jour pour le formulaire.")

st.markdown("---")

# -------- Informations client --------
st.markdown("### 🧾 Informations client")

def days_to_years_positive(days: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    try:
        return round(abs(float(days)) / 365.25, 2)
    except Exception:
        return None

# Domaines (fallback si endpoint absent)
default_gender = ["", "F", "M"]
default_family = ["", "Married", "Single / not married", "Separated", "Widow", "Civil marriage"]

domain_contract = VALUE_DOMAINS.get("NAME_CONTRACT_TYPE", ["", "Cash loans", "Revolving loans"])
domain_gender   = VALUE_DOMAINS.get("CODE_GENDER", default_gender)
domain_family   = VALUE_DOMAINS.get("NAME_FAMILY_STATUS", default_family)

c1, c2, c3 = st.columns(3)
with c1:
    amt_income = st.number_input(f"{label_fr('AMT_INCOME_TOTAL')} ({'AMT_INCOME_TOTAL'})",
                                 min_value=0.0, step=1000.0, format="%.0f",
                                 help=help_fr("AMT_INCOME_TOTAL"))
    amt_credit = st.number_input(f"{label_fr('AMT_CREDIT')} ({'AMT_CREDIT'})",
                                 min_value=0.0, step=1000.0, format="%.0f",
                                 help=help_fr("AMT_CREDIT"))
with c2:
    days_birth = st.number_input(f"{label_fr('DAYS_BIRTH')} ({'DAYS_BIRTH'})",
                                 value=float(st.session_state.get("DAYS_BIRTH_val", -14000.0)),
                                 step=100.0, format="%.0f",
                                 help=help_fr("DAYS_BIRTH"))
    age_years_disp = days_to_years_positive(days_birth)
    st.caption(f"Âge estimé ≈ **{age_years_disp if age_years_disp is not None else '?'}** ans")
with c3:
    days_employed = st.number_input(f"{label_fr('DAYS_EMPLOYED')} ({'DAYS_EMPLOYED'})",
                                    value=float(st.session_state.get("DAYS_EMPLOYED_val", -2000.0)),
                                    step=50.0, format="%.0f",
                                    help=help_fr("DAYS_EMPLOYED"))
    emp_years     = days_to_years_positive(days_employed)
    st.caption(f"Ancienneté estimée ≈ **{emp_years if emp_years is not None else '?'}** ans")

cc1, cc2, cc3 = st.columns(3)
with cc1:
    children = st.number_input(f"{label_fr('CNT_CHILDREN')} ({'CNT_CHILDREN'})",
                               min_value=0, step=1, help=help_fr("CNT_CHILDREN"))
with cc2:
    annuity  = st.number_input(f"{label_fr('AMT_ANNUITY')} ({'AMT_ANNUITY'})",
                               min_value=0.0, step=500.0, format="%.0f", help=help_fr("AMT_ANNUITY"))
with cc3:
    contract = st.selectbox(f"{label_fr('NAME_CONTRACT_TYPE')} ({'NAME_CONTRACT_TYPE'})",
                            [""] + [x for x in domain_contract if x], index=0, help=help_fr("NAME_CONTRACT_TYPE"))

cc4, cc5 = st.columns(2)
with cc4:
    code_gender = st.selectbox(f"{label_fr('CODE_GENDER')} ({'CODE_GENDER'})",
                               [x for x in domain_gender], index=0, help=help_fr("CODE_GENDER"))
with cc5:
    family_status = st.selectbox(f"{label_fr('NAME_FAMILY_STATUS')} ({'NAME_FAMILY_STATUS'})",
                                 [x for x in domain_family], index=0, help=help_fr("NAME_FAMILY_STATUS"))

def build_payload() -> Dict[str, Any]:
    base = {
        # Toutes les variables sources des ratios sont ici
        "AMT_INCOME_TOTAL": None if amt_income == 0 else amt_income,
        "AMT_CREDIT":       None if amt_credit == 0 else amt_credit,
        "AMT_ANNUITY":      None if annuity == 0 else annuity,
        "DAYS_BIRTH":       None if days_birth == 0 else days_birth,
        "DAYS_EMPLOYED":    None if days_employed == 0 else days_employed,
        # Autres infos de base
        "CNT_CHILDREN":     None if children == 0 else int(children),
        "NAME_CONTRACT_TYPE": contract if contract else None,
        "CODE_GENDER":       code_gender if code_gender else None,
        "NAME_FAMILY_STATUS": family_status if family_status else None,
    }
    if exp_cols:
        return {col: base.get(col, None) for col in exp_cols}
    return base

payload = build_payload()

# Aperçu payload (sidebar)
with st.sidebar:
    show_preview = any([
        payload.get("AMT_INCOME_TOTAL") not in (None, 0, ""),
        payload.get("AMT_CREDIT") not in (None, 0, ""),
        payload.get("DAYS_BIRTH") not in (None, 0, ""),
    ])
    if show_preview:
        st.json({"data": {k: v for k, v in payload.items() if v is not None}})
    else:
        st.caption("Renseigne au moins revenu, crédit et âge pour voir l’aperçu.")

# -------- Variables complémentaires (toutes les colonnes du modèle) --------
st.markdown("---")
with st.expander("🔧 Variables complémentaires (toutes les variables du modèle)", expanded=False):
    if not exp_cols:
        st.caption("La liste des colonnes attendues est indisponible (endpoint /expected_columns).")
    else:
        already = {
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED",
            "CNT_CHILDREN", "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_FAMILY_STATUS",
        }

        def get_features_dict(ref):
            if isinstance(ref, dict):
                if "features" in ref and isinstance(ref["features"], dict):
                    return ref["features"]
                if "stats" in ref and isinstance(ref["stats"], dict):
                    return ref["stats"]
            return {}
        features_dict = get_features_dict(REF_STATS)
        numeric_hint = {k for k, v in features_dict.items() if isinstance(v, dict) and v.get("type") == "numeric"}

        cols_adv = st.columns(3)
        i = 0
        for col in exp_cols:
            if col in already:
                continue
            ui_col = cols_adv[i % len(cols_adv)]
            with ui_col:
                domain = VALUE_DOMAINS.get(col)
                lab = f"{label_fr(col)} ({col})"
                hint  = help_fr(col)
                if domain and isinstance(domain, list) and len(domain) > 0:
                    choices = [""] + [x for x in domain if x is not None and x != ""]
                    choice = st.selectbox(lab, choices, index=0, help=hint)
                    if choice != "":
                        payload[col] = choice
                elif col in numeric_hint:
                    val = st.number_input(lab, value=0.0, step=1.0, format="%.4f", help=hint)
                    if val != 0.0:
                        payload[col] = float(val)
                else:
                    txt = st.text_input(lab, value="", help=hint or "Laisse vide si non renseigné")
                    if txt.strip() != "":
                        payload[col] = txt.strip()
            i += 1
        st.caption("Astuce : ne renseigne que ce qui est utile. Les champs vides sont envoyés en `None` et seront gérés par le prétraitement.")

# -------- Évaluation (/predict) --------
st.markdown("### 🧮 Évaluation")

# Utilitaires ratios + comparaison population (pour impact)
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _div(a, b):
    a = _safe_float(a); b = _safe_float(b)
    if a is None or b in (None, 0.0):
        return None
    return a / b

@st.cache_data(ttl=30, show_spinner=False)
def _get_api_threshold() -> Optional[float]:
    ok, js, _ = http_get_json(f"{API_URL}/")
    if ok and isinstance(js, dict):
        thr = js.get("decision_threshold")
        if isinstance(thr, dict) and "t_selected" in thr:
            try:
                return float(thr["t_selected"])
            except Exception:
                return None
        if isinstance(thr, (int, float)):
            return float(thr)
    return None

def _get_pop_stats(feature: str) -> tuple[Optional[float], Optional[float]]:
    """Retourne (mu, sigma) si dispos dans REF_STATS."""
    if REF_STATS is None:
        return None, None
    fs = None
    if "features" in REF_STATS and isinstance(REF_STATS["features"], dict):
        fs = REF_STATS["features"].get(feature)
    elif "stats" in REF_STATS and isinstance(REF_STATS["stats"], dict):
        fs = REF_STATS["stats"].get(feature)
    if isinstance(fs, dict):
        mu = fs.get("mean")
        sig = fs.get("std")
        try:
            mu = float(mu) if mu is not None else None
            sig = float(sig) if sig is not None else None
        except Exception:
            mu, sig = None, None
        return mu, sig
    return None, None

def _explain_ratio(name: str, val: Optional[float]) -> tuple[str, str]:
    """
    Retourne (badge_impact, phrase_explication).
    badge_impact: '🟩 plutôt favorable', '🟥 plus risqué', '🔸 impact à confirmer'
    """
    if val is None:
        return "—", "Valeur manquante : impossible d’évaluer l’impact."

    mu, sig = _get_pop_stats(name)
    if mu is None or sig in (None, 0):
        # Pas de repère population → explication générique
        if name in RISK_UP:
            return "🔸", "Plus la valeur est élevée, plus l’effet tend à **augmenter la probabilité de défaut** (à confirmer localement)."
        if name in PROTECTIVE_UP:
            return "🔸", "Une valeur plus élevée est **plutôt protectrice** (à confirmer localement)."
        return "🔸", "Impact dépendant du modèle et du contexte (référez-vous à l’explication locale)."

    direction = "au-dessus" if val > mu else "au-dessous"
    z = None
    try:
        z = (val - mu) / sig if sig else None
    except Exception:
        pass

    severite = ""
    if isinstance(z, (int, float)):
        if abs(z) >= 2:   severite = " (écart **important** vs population)"
        elif abs(z) >= 1: severite = " (écart **modéré** vs population)"

    if name in RISK_UP:
        impact_badge = "🟥 plus risqué" if val > mu else "🟩 plutôt favorable"
        cause = "valeur élevée → charge relative accrue" if val > mu else "valeur plus basse → charge relative réduite"
    elif name in PROTECTIVE_UP:
        impact_badge = "🟩 plutôt favorable" if val > mu else "🟥 plus risqué"
        cause = "valeur élevée corrélée à un meilleur profil" if val > mu else "valeur plus basse moins protectrice"
    else:
        impact_badge = "🔸 impact à confirmer"
        cause = "l’effet dépend du modèle et des interactions"

    exp = f"{impact_badge} — {direction} de la moyenne{severite} ; {cause}."
    return impact_badge, exp

# Prépare importances globales -> rang
IMP_RANK: dict[str, int] = {}
if GLOBAL_IMP is not None and not GLOBAL_IMP.empty and "raw_feature" in GLOBAL_IMP.columns:
    for i, row in GLOBAL_IMP.reset_index(drop=True).iterrows():
        try:
            IMP_RANK[str(row["raw_feature"])] = int(i) + 1  # 1 = plus important
        except Exception:
            pass

# Seuil par défaut (depuis API si possible)
t_api = _get_api_threshold()
if "t_ui" not in st.session_state:
    st.session_state["t_ui"] = t_api if isinstance(t_api, (int, float)) else 0.50

# Ligne de contrôles : bouton + slider
ctrl_l, ctrl_r = st.columns([1, 3])
with ctrl_l:
    do_predict = st.button("Évaluer ce dossier", type="primary")
with ctrl_r:
    st.caption("Ajustez le **seuil (UI)** pour tester l’impact sur la décision (l’API conserve son seuil interne).")
    st.session_state["t_ui"] = st.slider(
        "Seuil (UI) — transformera p en décision (refus si p ≥ t)",
        min_value=0.00, max_value=1.00, value=float(st.session_state["t_ui"]), step=0.01
    )
    if t_api is not None:
        st.caption(f"Seuil **API** (référence) : {t_api:.3f} • {t_api:.1%}")

# Bloc résultats (si on clique) — sinon on affiche déjà la partie ratios en-dessous
proba = None
thr_api_raw = None
if do_predict:
    ok, pred_json, err = http_post_json(f"{API_URL}/predict", {"data": payload})
    if not ok:
        st.error(f"Erreur /predict : {err}\n\nRéponse: {pred_json}")
    else:
        proba = pred_json.get("probability_default")
        thr_api_raw = pred_json.get("threshold")

        # Normalisation du seuil API en nombre (pour affichage)
        t_api_num = None
        if isinstance(thr_api_raw, dict) and "t_selected" in thr_api_raw:
            try:
                t_api_num = float(thr_api_raw["t_selected"])
            except Exception:
                t_api_num = None
        elif isinstance(thr_api_raw, (int, float)):
            t_api_num = float(thr_api_raw)
        t_ui = float(st.session_state.get("t_ui", 0.5))

        if proba is None:
            st.warning(f"Réponse inattendue: {pred_json}")
        else:
            decision_ui = 1 if float(proba) >= t_ui else 0

            # --- Synthèse + comparatif ---
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilité de défaut", f"{proba:.2%}")
                st.metric("Décision", "❌ Refus" if decision_ui == 1 else "✅ Acceptation")
                st.caption("Calculée avec le **seuil (UI)** ci-contre.")
            with c2:
                st.metric("Seuil (UI)", f"{t_ui:.3f}  •  {t_ui:.1%}")
                if t_api_num is not None:
                    st.caption(f"Seuil **API** (info) : {t_api_num:.3f}  •  {t_api_num:.1%}")
                cmp_sym = ">=" if float(proba) >= t_ui else "<"
                final = "❌ Refus" if float(proba) >= t_ui else "✅ Acceptation"
                st.markdown(f"**Comparatif** : `p = {float(proba):.3f}` {cmp_sym} `t(UI) = {t_ui:.3f}` → **{final}**")
                st.caption(
                    "ℹ️ **Le seuil** transforme une **probabilité** en **décision** : "
                    "si `p ≥ t` → **refus**, sinon **acceptation**. "
                    "Il matérialise l’**appétence au risque** (plus le seuil est bas, plus on refuse)."
                )

            # --- Visuel bullet p vs t ---
            st.markdown("#### Repères visuels p vs t")
            fig_bullet = go.Figure(go.Indicator(
                mode="number+gauge",
                value=float(proba),
                number={"valueformat": ".3f"},
                title={"text": "p (barre) vs t(UI) (ligne rouge)"},
                gauge={
                    "shape": "bullet",
                    "axis": {"range": [0, 1]},
                    "bar": {"thickness": 0.6},
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.85,
                        "value": float(t_ui),
                    },
                },
                domain={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            ))
            fig_bullet.update_layout(height=170, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bullet, use_container_width=True)
            st.caption("Repères : la **barre** représente `p`, la **ligne rouge** le seuil `t` (UI). Si la barre dépasse la ligne → **refus**.")

# --- Ratios (calculés) — intégrés dans Évaluation, lisibles en grille ---
st.markdown("#### 📐 Ratios clés (calculés)")

ratios = {
    "AGE_YEARS": (abs(_safe_float(payload.get("DAYS_BIRTH"))) / 365.25 if payload.get("DAYS_BIRTH") not in (None, "", 0) else None,
                  "Âge (années) estimé"),
    "EMPLOY_YEARS": (abs(_safe_float(payload.get("DAYS_EMPLOYED"))) / 365.25 if payload.get("DAYS_EMPLOYED") not in (None, "", 0) else None,
                     "Ancienneté emploi (années)"),
    "PAYMENT_RATE": (_div(payload.get("AMT_ANNUITY"), payload.get("AMT_CREDIT")),
                     "Taux de paiement = Annuité / Crédit"),
    "CREDIT_INCOME_RATIO": (_div(payload.get("AMT_CREDIT"), payload.get("AMT_INCOME_TOTAL")),
                            "Ratio crédit / revenu"),
    "ANNUITY_INCOME_RATIO": (_div(payload.get("AMT_ANNUITY"), payload.get("AMT_INCOME_TOTAL")),
                             "Ratio annuité / revenu"),
    "CREDIT_TERM_MONTHS": (_div(payload.get("AMT_CREDIT"), payload.get("AMT_ANNUITY")),
                           "Durée (mois) ≈ Crédit / Annuité"),
    "EMPLOY_TO_AGE_RATIO": (_div(
        (abs(_safe_float(payload.get("DAYS_EMPLOYED"))) / 365.25) if payload.get("DAYS_EMPLOYED") not in (None, "", 0) else None,
        (abs(_safe_float(payload.get("DAYS_BIRTH"))) / 365.25) if payload.get("DAYS_BIRTH") not in (None, "", 0) else None
    ), "Ancienneté / Âge"),
}

cols_rat = st.columns(3)
order_rat = [
    "PAYMENT_RATE",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM_MONTHS",
    "AGE_YEARS",
    "EMPLOY_YEARS",
    "EMPLOY_TO_AGE_RATIO",
]

for i, key in enumerate(order_rat):
    val, desc = ratios.get(key, (None, ""))  # type: ignore
    with cols_rat[i % 3]:
        label = f"{label_fr(key)} ({key})"
        # Valeur numérique affichée joliment
        if val is None:
            st.metric(label, "—")
        else:
            if key in ("AGE_YEARS", "EMPLOY_YEARS"):
                st.metric(label, f"{val:.2f}")
            elif key == "CREDIT_TERM_MONTHS":
                st.metric(label, f"{val:.1f} mois")
            else:
                st.metric(label, f"{val:.3f}")

        # Badge d'impact + explication courte
        badge, exp = _explain_ratio(key, val)
        st.caption(desc)
        st.markdown(f"**Impact estimé** : {badge}")
        st.caption(exp)

        # Importance globale du modèle (si connue)
        if GLOBAL_IMP is not None and not GLOBAL_IMP.empty and "raw_feature" in GLOBAL_IMP.columns:
            r = None
            try:
                r = int(GLOBAL_IMP.reset_index(drop=True).query("raw_feature == @key").index.min()) + 1
            except Exception:
                r = None
            if isinstance(r, int):
                tag = "🔝 variable importante" if r <= 10 else ("⬆️ importance notable" if r <= 20 else None)
                if tag:
                    st.caption(f"{tag} (rang global ≈ {r}).")

st.caption(
    "ℹ️ Lecture : l’**impact estimé** s’appuie sur une **comparaison à la population** et des **heuristiques** "
    "(sens attendu du ratio). Pour connaître l’effet **exact** sur **ce dossier**, utilisez l’explication **locale**."
)

# -------- Comparaison population --------
st.markdown("---")
st.markdown("### 📊 Comparaison avec la population (référence = jeu d’entraînement)")

def _get_feature_stats(ref_stats: dict, feature: str) -> Optional[dict]:
    if not isinstance(ref_stats, dict):
        return None
    if "features" in ref_stats and isinstance(ref_stats["features"], dict):
        return ref_stats["features"].get(feature)
    if "stats" in ref_stats and isinstance(ref_stats["stats"], dict):
        return ref_stats["stats"].get(feature)
    return None

def _impact_note(feature: str, above_mean: Optional[bool]) -> str:
    if above_mean is None:
        return "Impact : valeur non renseignée."
    if feature == "AMT_CREDIT":
        return ("Dans ce dossier, la valeur est **"
                + ("au-dessus" if above_mean else "au-dessous")
                + "** de la moyenne : pour le **montant de crédit**, être au-dessus tend à **augmenter le risque** ; "
                  "au-dessous tend à être **plutôt favorable**.")
    if feature == "AMT_INCOME_TOTAL":
        return ("Dans ce dossier, la valeur est **"
                + ("au-dessus" if above_mean else "au-dessous")
                + "** de la moyenne : pour le **revenu**, être au-dessus est **plutôt favorable** ; "
                  "au-dessous peut **augmenter le risque**.")
    if feature == "AGE_YEARS":
        return ("Dans ce dossier, la valeur est **"
                + ("au-dessus" if above_mean else "au-dessous")
                + "** de la moyenne : en général, un **âge plus élevé** est **plutôt favorable** ; "
                  "un âge très bas peut **augmenter le risque** (relation parfois non linéaire).")
    return ("Interprétation indicative : l’influence exacte dépend du modèle et du contexte. "
            "Référez-vous à l’explication locale pour confirmation.")

def show_population_position(feature: str, label_txt: str, value: Optional[float], fmt: str = ".0f", unit: str = ""):
    stats = _get_feature_stats(REF_STATS, feature) if REF_STATS else None
    if not stats or value is None:
        st.write(f"**{label_txt}** : information manquante.")
        return

    mu  = stats.get("mean")
    sig = stats.get("std")
    above_mean: Optional[bool] = None
    if mu is not None:
        try:
            above_mean = float(value) > float(mu)
        except Exception:
            above_mean = None

    if mu is not None and sig not in (None, 0):
        side = "au-dessus" if above_mean else "au-dessous"
        st.write(
            f"**{label_txt}** : {value:{fmt}}{unit} — {side} de la moyenne (μ={mu:{fmt}}{unit}, σ={sig:{fmt}}). "
            f"{_impact_note(feature, above_mean)}"
        )
        lo = mu - 3*sig
        hi = mu + 3*sig
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo < hi:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": label_txt},
                gauge={"axis": {"range": [lo, hi]}, "bar": {"thickness": 0.5}},
            ))
            st.plotly_chart(fig, use_container_width=True)
            return

    hist = stats.get("hist")
    if hist:
        edges = hist.get("edges", [])
        counts = hist.get("counts", [])
        if edges and counts and len(edges) == len(counts) + 1:
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            st.write(
                f"**{label_txt}** : {value:{fmt}}{unit} (comparé à la distribution). "
                f"{_impact_note(feature, above_mean)}"
            )
            fig = go.Figure(go.Bar(x=centers, y=counts))
            st.plotly_chart(fig, use_container_width=True)
            return

    st.write(
        f"**{label_txt}** : {value:{fmt}}{unit} (μ/σ indisponibles). "
        f"{_impact_note(feature, above_mean)}"
    )

if REF_STATS is None:
    st.info("Stats de population indisponibles (`artifacts/ref_stats.json`). Génère-les et place le fichier dans `artifacts/`.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        show_population_position("AMT_CREDIT", label_fr("AMT_CREDIT"), payload.get("AMT_CREDIT"), fmt=".0f", unit=" €")
    with c2:
        age_disp = (abs(payload.get("DAYS_BIRTH"))/365.25) if payload.get("DAYS_BIRTH") not in (None, 0, "") else None
        if isinstance(age_disp, (int, float)):
            age_disp = round(age_disp, 2)
        show_population_position("AGE_YEARS", label_fr("AGE_YEARS"), age_disp, fmt=".1f", unit=" ans")
    with c3:
        inc = payload.get("AMT_INCOME_TOTAL")
        show_population_position("AMT_INCOME_TOTAL", label_fr("AMT_INCOME_TOTAL"), inc, fmt=".0f", unit=" €")

# -------- Cohorte similaire --------
st.markdown("---")
st.markdown("### 👥 Cohorte similaire (âge & type de contrat)")

with st.expander("ℹ️ Comment lire cette section (mode d'emploi)", expanded=False):
    st.markdown(
        "- **Cohorte sélectionnée** : clients **similaires** par tranche d’**âge** et **type de contrat**.\n"
        "- **KPIs** (si présents) : ex. `default_rate` (taux de défaut moyen de la cohorte), `avg_probability_default`…\n"
        "- **Tableau Cohorte vs Population** : compare la **valeur typique** (moyenne/médiane) par variable.\n"
        "  - Si une variable **associée au risque** est **plus élevée** dans la cohorte → *plus risqué*.\n"
        "  - Si une variable **protectrice** est **plus basse** → *plus risqué*.\n"
        "  - À croiser avec les **importances** et l’**explication locale** du dossier.\n"
    )

def _label_fr_fallback(col: str) -> str:
    try:
        return label_fr(col)  # type: ignore[name-defined]
    except Exception:
        return col

def _get_seg_mapping(raw):
    if not isinstance(raw, dict):
        return {}
    if any(isinstance(k, str) and "|" in k for k in raw.keys()):
        return raw
    for v in raw.values():
        if isinstance(v, dict) and any(isinstance(k, str) and "|" in k for k in v.keys()):
            return v
    return raw

SEG_MAP = _get_seg_mapping(SEG_STATS) if SEG_STATS else {}

def _parse_interval_key(k: str) -> tuple[float | None, float | None, str | None]:
    try:
        interval, contract_k = k.split("|", 1)
        interval = interval.strip()
        contract_k = contract_k.strip()
        interval = interval.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        lo_s, hi_s = [x.strip() for x in interval.split(",")]
        lo = float(lo_s) if lo_s not in ("-inf", "", None) else None
        hi = float(hi_s) if hi_s not in ("inf", "", None) else None
        return lo, hi, contract_k
    except Exception:
        return None, None, None

def _find_segment_key(age_years: float | None, contract_val: str | None, seg_stats: dict) -> str | None:
    if age_years is None or not seg_stats:
        return None
    candidates = []
    for k in seg_stats.keys():
        lo, hi, c = _parse_interval_key(k)
        if c is None:
            continue
        if contract_val and c not in (contract_val, "*"):
            continue
        inside = True
        if lo is not None and age_years <= lo: inside = False
        if hi is not None and age_years >  hi: inside = False
        if inside:
            width = (hi - lo) if (lo is not None and hi is not None) else float("inf")
            candidates.append((width, k))
    if not candidates:
        for k in seg_stats.keys():
            lo, hi, _ = _parse_interval_key(k)
            if lo is None and hi is None:
                continue
            inside = True
            if lo is not None and age_years <= lo: inside = False
            if hi is not None and age_years >  hi: inside = False
            if inside:
                width = (hi - lo) if (lo is not None and hi is not None) else float("inf")
                candidates.append((width, k))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _is_num(x: Any) -> bool:
    try:
        return x is not None and not isinstance(x, str) and np.isfinite(float(x))
    except Exception:
        return False

def _mean_from_hist(edges: List[float], counts: List[int]) -> Optional[float]:
    if not edges or not counts or len(edges) != len(counts) + 1:
        return None
    mids = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    wsum = sum(c * m for c, m in zip(counts, mids))
    csum = sum(counts)
    if csum == 0:
        return None
    return wsum / csum

def _try_numeric_typical(obj: Any) -> Optional[float]:
    if _is_num(obj):
        return float(obj)
    if isinstance(obj, list):
        nums = [float(x) for x in obj if _is_num(x)]
        return float(np.mean(nums)) if nums else None
    if isinstance(obj, dict):
        for k in ["mean", "median"]:
            if k in obj and _is_num(obj[k]):
                return float(obj[k])
        if "quantiles" in obj and isinstance(obj["quantiles"], dict):
            for key in ["p50", "50", "P50", "median"]:
                v = obj["quantiles"].get(key)
                if _is_num(v): return float(v)
        if "values" in obj and isinstance(obj["values"], list):
            nums = [float(x) for x in obj["values"] if _is_num(x)]
            if nums: return float(np.mean(nums))
        if "hist" in obj and isinstance(obj["hist"], dict):
            edges = obj["hist"].get("edges") or []
            counts = obj["hist"].get("counts") or []
            m = _mean_from_hist(edges, counts)
            if m is not None: return float(m)
    return None

def _extract_kpis_and_measures(seg: Dict[str, Any], global_feats: set[str]) -> tuple[Dict[str, Any], Dict[str, float]]:
    kpis: Dict[str, Any] = {}
    measures: Dict[str, float] = {}

    if not isinstance(seg, dict):
        return kpis, measures

    for k, v in seg.items():
        if _is_num(v):
            kl = str(k).lower()
            if any(s in kl for s in ["rate", "prob", "default", "accept", "reject"]) or k in ("count", "size", "n"):
                kpis[k] = v

    for key in ["means", "features_means"]:
        if isinstance(seg.get(key), dict):
            for f, val in seg[key].items():
                m = _try_numeric_typical(val)
                if m is not None and (not global_feats or f in global_feats):
                    measures[f] = m

    if isinstance(seg.get("stats"), dict):
        for f, d in seg["stats"].items():
            m = _try_numeric_typical(d)
            if m is not None and (not global_feats or f in global_feats):
                measures[f] = m

    if not measures:
        for f, v in seg.items():
            if f in ("count", "size", "n"):
                continue
            if str(f).lower() in ("means", "features_means", "stats", "hist", "values"):
                continue
            m = _try_numeric_typical(v)
            if m is not None and (not global_feats or f in global_feats):
                measures[f] = m

    return kpis, measures

def _get_global_typical(feature: str) -> Optional[float]:
    fs = None
    if REF_STATS:
        if "features" in REF_STATS and feature in REF_STATS["features"]:
            fs = REF_STATS["features"][feature]
        elif "stats" in REF_STATS and feature in REF_STATS["stats"]:
            fs = REF_STATS["stats"][feature]
    if isinstance(fs, dict):
        for k in ["mean", "median"]:
            if _is_num(fs.get(k)): return float(fs[k])
        if "quantiles" in fs and isinstance(fs["quantiles"], dict):
            for key in ["p50", "50", "P50"]:
                v = fs["quantiles"].get(key)
                if _is_num(v): return float(v)
    return None

GLOBAL_FEATS = set()
if REF_STATS:
    if "features" in REF_STATS and isinstance(REF_STATS["features"], dict):
        GLOBAL_FEATS = set(REF_STATS["features"].keys())
    elif "stats" in REF_STATS and isinstance(REF_STATS["stats"], dict):
        GLOBAL_FEATS = set(REF_STATS["stats"].keys())

if not SEG_MAP:
    st.info("Stats cohortes indisponibles (`artifacts/seg_stats.json`).")
else:
    age_years = (abs(payload.get("DAYS_BIRTH"))/365.25) if payload.get("DAYS_BIRTH") not in (None, 0, "") else None
    seg_key = _find_segment_key(age_years, payload.get("NAME_CONTRACT_TYPE"), SEG_MAP)
    if seg_key is None:
        st.warning("Aucune statistique disponible pour cette cohorte (vérifie le format des clés dans `artifacts/seg_stats.json`).")
    else:
        seg = SEG_MAP.get(seg_key, {})

        count_val = seg.get("count")
        if isinstance(count_val, (int, float)) and np.isfinite(count_val):
            st.write(f"**Cohorte sélectionnée** : `{seg_key}` • **Taille** ≈ {int(count_val)}")
        else:
            st.write(f"**Cohorte sélectionnée** : `{seg_key}`")

        kpis, measures = _extract_kpis_and_measures(seg, GLOBAL_FEATS)
        if kpis:
            cols_k = st.columns(min(4, len(kpis)))
            for (i, (k, v)) in enumerate(kpis.items()):
                with cols_k[i % len(cols_k)]:
                    try:
                        if any(s in k.lower() for s in ["rate", "prob", "default"]):
                            st.metric(k.replace("_", " ").title(), f"{float(v)*100:.1f}%")
                        else:
                            st.metric(k.replace("_", " ").title(), f"{v}")
                    except Exception:
                        st.metric(k.replace("_", " ").title(), str(v))
        else:
            st.caption("Aucun KPI explicite détecté (ex.: `default_rate`, `avg_probability_default`).")

        rows = []
        if measures:
            priority = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS",
                        "PAYMENT_RATE", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
            ordered_feats = [f for f in priority if f in measures] + [f for f in measures if f not in priority]

            for feat in ordered_feats[:12]:
                m_cohort = measures.get(feat)
                m_global = _get_global_typical(feat)
                if m_cohort is None or m_global is None:
                    continue
                diff = m_cohort - m_global

                interp = "—"
                if feat in RISK_UP:
                    if diff > 0:  interp = "🟥 plus risqué"
                    elif diff < 0: interp = "🟩 plutôt favorable"
                elif feat in PROTECTIVE_UP:
                    if diff > 0:  interp = "🟩 plutôt favorable"
                    elif diff < 0: interp = "🟥 plus risqué"

                rows.append({
                    "Variable": f"{_label_fr_fallback(feat)} ({feat})",
                    "Cohorte (valeur typique)": round(m_cohort, 4),
                    "Population (valeur typique)": round(m_global, 4),
                    "Écart (cohorte - pop)": round(diff, 4),
                    "Interprétation": interp,
                })

        if rows:
            df_comp = pd.DataFrame(rows)
            st.write("**Comparaison cohorte vs population (valeur typique)**")
            st.dataframe(df_comp, use_container_width=True)
            st.caption(
                "💡 Repère les lignes avec un **écart** important et une **interprétation** en rouge : "
                "elles pointent les facteurs susceptibles d’**augmenter le risque** pour des profils similaires."
            )
        else:
            if isinstance(seg, dict) and seg:
                preview = {}
                for k, v in list(seg.items())[:12]:
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        preview[k] = float(v)
                    elif isinstance(v, list):
                        preview[k] = f"Liste ({len(v)} éléments)"
                    elif isinstance(v, dict):
                        preview[k] = f"Dictionnaire (clés: {', '.join(list(v.keys())[:5])}...)"
                    else:
                        preview[k] = str(type(v)).replace("<class '", "").replace("'>", "")
                st.write("**Contenu disponible (aperçu)**")
                st.json(preview)
                st.caption(
                    "Pour enrichir le comparatif, ajoute par cohorte un bloc `means` (ou `stats.mean` / `quantiles.p50`) "
                    "et un `count` dans `artifacts/seg_stats.json`."
                )
            else:
                st.info("Cohorte trouvée mais sans métriques exploitables pour comparaison.")

# -------- Importance globale --------
st.markdown("---")
st.markdown("### 🧠 Variables les plus influentes (globales)")
if GLOBAL_IMP is None or GLOBAL_IMP.empty:
    st.caption("Aucune importance globale fournie (place `artifacts/global_importance.csv`).")
else:
    topk = st.slider("Top variables à afficher", 5, 30, 15, 1)
    sub = GLOBAL_IMP.head(topk)
    fig = go.Figure(go.Bar(x=sub["abs_importance"], y=sub["raw_feature"], orientation="h"))
    fig.update_layout(height=400 + 18 * len(sub), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
