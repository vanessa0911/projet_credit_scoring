# streamlit_app.py
# =========================================================
# Prêt à dépenser — Scoring & Explicabilité (V0 stable+ UX)
# - Affichage nombres lisibles (séparateur ' suisse)
# - Libellés FR + définitions + unités (VAR_LABELS/VAR_HELP/VAR_UNITS)
# - Sidebar: état API + URL configurable
# - Champs essentiels + variables avancées (si /expected_columns)
# - Évaluation unitaire: p, t(UI), explications, position vs population
# - Graphiques: bullet p vs t, barres ratios clés
# =========================================================

import math
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Prêt à dépenser — Scoring", page_icon="💳", layout="wide")

# ---------- Libellés FR + définitions + unités ----------
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
    "AGE_YEARS": "Âge (années)",
    "PAYMENT_RATE": "Taux de paiement (annuité/crédit)",
    "CREDIT_INCOME_RATIO": "Ratio crédit / revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenu",
    "CREDIT_TERM_MONTHS": "Durée du crédit (mois)",
    "INCOME_PER_PERSON": "Revenu par personne au foyer",
    "CHILDREN_RATIO": "Ratio enfants / foyer",
    "AMT_GOODS_PRICE": "Prix du bien",
}

VAR_HELP = {
    "AMT_INCOME_TOTAL": "Revenu annuel brut déclaré (en € ou CHF selon le contexte).",
    "AMT_CREDIT": "Montant total du crédit demandé.",
    "AMT_ANNUITY": "Montant de l’échéance périodique (annuité).",
    "DAYS_BIRTH": "Âge exprimé en jours négatifs (convention Home Credit).",
    "DAYS_EMPLOYED": "Ancienneté d’emploi exprimée en jours négatifs.",
    "CNT_CHILDREN": "Nombre total d’enfants à charge.",
    "NAME_CONTRACT_TYPE": "Type de contrat de crédit (Cash, Revolving…).",
    "CODE_GENDER": "Sexe (M/F).",
    "NAME_FAMILY_STATUS": "Situation familiale (Married, Single, etc.).",
    "PAYMENT_RATE": "Annuité / Crédit. Plus c’est élevé, plus l’effort de remboursement est fort.",
    "CREDIT_INCOME_RATIO": "Crédit / Revenu annuel. Indique l’endettement relatif.",
    "ANNUITY_INCOME_RATIO": "Annuité / Revenu annuel. Indique l’effort mensuel relatif.",
    "CREDIT_TERM_MONTHS": "Durée implicite estimée du crédit (en mois).",
    "INCOME_PER_PERSON": "Revenu par personne au foyer.",
    "AGE_YEARS": "Âge en années (valeur dérivée de DAYS_BIRTH).",
}

VAR_UNITS = {
    "AMT_INCOME_TOTAL": "€",
    "AMT_CREDIT": "€",
    "AMT_ANNUITY": "€",
    "AGE_YEARS": "ans",
    "CREDIT_TERM_MONTHS": "mois",
    "PAYMENT_RATE": "",  # ratio
    "CREDIT_INCOME_RATIO": "",  # ratio
    "ANNUITY_INCOME_RATIO": "",  # ratio
    "INCOME_PER_PERSON": "€",
}

# ---------- Formatage nombres (séparateur suisse ' ) ----------
def fmt_thousands(x: Optional[float], decimals: int = 0, sep: str = "'") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        s = f"{float(x):,.{decimals}f}"  # 200,000.00
        s = s.replace(",", "§").replace(".", ",").replace("§", sep)  # 200'000,00 (style CH/FR)
        return s
    except Exception:
        return str(x)

def fmt_money(x: Optional[float], currency: str = "€", decimals: int = 0) -> str:
    v = fmt_thousands(x, decimals=decimals)
    return f"{v} {currency}" if v != "—" else v

def fmt_ratio(x: Optional[float], pct: bool = False, decimals: int = 2) -> str:
    if x is None: return "—"
    try:
        if pct:
            return f"{100*float(x):.{decimals}f}%"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"

def label_fr(col: str) -> str:
    return VAR_LABELS.get(col, col)

def help_fr(col: str) -> Optional[str]:
    return VAR_HELP.get(col)

# ---------- API URL ----------
def get_api_url() -> str:
    default = "http://127.0.0.1:8000"
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

# ---------- HTTP helpers ----------
def http_get_json(url: str, timeout: int = 12) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        try:
            return False, r.json(), str(e)  # type: ignore[name-defined]
        except Exception:
            return False, {}, str(e)

def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[bool, Dict[str, Any], str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        try:
            return False, r.json(), str(e)  # type: ignore[name-defined]
        except Exception:
            return False, {}, str(e)

# ---------- Artifacts optionnels ----------
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
ARTIFACT_DIR.mkdir(exist_ok=True)
REF_STATS = load_json_if_exists(ARTIFACT_DIR / "ref_stats.json")

GLOBAL_IMP = None
try:
    gi_path = ARTIFACT_DIR / "global_importance.csv"
    if gi_path.exists():
        GLOBAL_IMP = pd.read_csv(gi_path)
except Exception:
    GLOBAL_IMP = None

# =========================================================
# Sidebar — État API + config URL
# =========================================================
with st.sidebar:
    st.markdown("### 🛰️ État de l’API")
    api_url_input = st.text_input("API URL", API_URL, help="Si l’API est exposée via le globe du port 8000, colle l’URL publique ici.")
    if api_url_input != API_URL:
        API_URL = api_url_input

    ok_root, root_json, err_root = http_get_json(f"{API_URL}/")
    if not ok_root:
        st.error(f"API non joignable.\n{err_root}")
    else:
        endpoints = root_json.get("endpoints", [])
        model_label = root_json.get("chosen_model") or root_json.get("model_class") or "?"
        dt = root_json.get("decision_threshold")
        st.success(f"Modèle: **{model_label}**\n\nSeuil API: `{dt}`\n\nEndpoints: {endpoints}")

    @st.cache_data(ttl=30, show_spinner=False)
    def get_expected_cols(api_url: str) -> list[str]:
        ok, js, _ = http_get_json(f"{api_url}/expected_columns")
        if ok and isinstance(js, dict) and "expected_columns" in js:
            return list(js["expected_columns"])
        return []

    @st.cache_data(ttl=30, show_spinner=False)
    def get_value_domains(api_url: str) -> Dict[str, list]:
        ok, js, _ = http_get_json(f"{api_url}/value_domains")
        if ok and isinstance(js, dict) and "domains" in js:
            # normalise en str/None
            out = {}
            for k, v in js["domains"].items():
                if isinstance(v, list):
                    out[k] = [None if x in (None, "None") else str(x) for x in v]
            return out
        return {}

    EXPECTED_COLS = get_expected_cols(API_URL)
    VALUE_DOMAINS = get_value_domains(API_URL)

    st.markdown("#### 🧩 Payload → API (aperçu)")
    st.caption("Aperçu des champs non vides envoyés à l’API (après saisie).")

# =========================================================
# Entrées utilisateur (champs essentiels + avancés)
# =========================================================
st.markdown("### 🧾 Informations client")

def days_to_years_positive(days: Optional[float]) -> Optional[float]:
    if days is None: return None
    try: return round(abs(float(days)) / 365.25, 2)
    except Exception: return None

# Domains par défaut si /value_domains absent
default_contract = ["", "Cash loans", "Revolving loans"]
default_gender = ["", "F", "M"]
default_family = ["", "Married", "Single / not married", "Separated", "Widow", "Civil marriage"]
okd, jsd, _ = http_get_json(f"{API_URL}/value_domains")
VALUE_DOMAINS = jsd.get("domains", {}) if okd else {}

domain_contract = VALUE_DOMAINS.get("NAME_CONTRACT_TYPE", default_contract)
domain_gender   = VALUE_DOMAINS.get("CODE_GENDER", default_gender)
domain_family   = VALUE_DOMAINS.get("NAME_FAMILY_STATUS", default_family)

# Champs essentiels
c1, c2, c3 = st.columns(3)
with c1:
    amt_income = st.number_input(
        f"{label_fr('AMT_INCOME_TOTAL')} (AMT_INCOME_TOTAL)",
        min_value=0.0, step=1000.0, format="%.0f", help=help_fr("AMT_INCOME_TOTAL")
    )
    st.caption(f"Valeur: **{fmt_money(amt_income)}**")

with c2:
    amt_credit = st.number_input(
        f"{label_fr('AMT_CREDIT')} (AMT_CREDIT)",
        min_value=0.0, step=1000.0, format="%.0f", help=help_fr("AMT_CREDIT")
    )
    st.caption(f"Valeur: **{fmt_money(amt_credit)}**")

with c3:
    annuity = st.number_input(
        f"{label_fr('AMT_ANNUITY')} (AMT_ANNUITY)",
        min_value=0.0, step=100.0, format="%.0f", help=help_fr("AMT_ANNUITY")
    )
    st.caption(f"Valeur: **{fmt_money(annuity)}**")

cc1, cc2, cc3 = st.columns(3)
with cc1:
    days_birth = st.number_input(
        f"{label_fr('DAYS_BIRTH')} (DAYS_BIRTH)",
        value=float(-14000.0), step=100.0, format="%.0f", help=help_fr("DAYS_BIRTH")
    )
    st.caption(f"Âge estimé ≈ **{days_to_years_positive(days_birth)}** ans")
with cc2:
    days_employed = st.number_input(
        f"{label_fr('DAYS_EMPLOYED')} (DAYS_EMPLOYED)",
        value=float(-2000.0), step=50.0, format="%.0f", help=help_fr("DAYS_EMPLOYED")
    )
    st.caption(f"Ancienneté estimée ≈ **{days_to_years_positive(days_employed)}** ans")
with cc3:
    children = st.number_input(
        f"{label_fr('CNT_CHILDREN')} (CNT_CHILDREN)",
        min_value=0, step=1, help=help_fr("CNT_CHILDREN")
    )

cc4, cc5, cc6 = st.columns(3)
with cc4:
    contract = st.selectbox(
        f"{label_fr('NAME_CONTRACT_TYPE')} (NAME_CONTRACT_TYPE)",
        domain_contract, index=0, help=help_fr("NAME_CONTRACT_TYPE")
    )
with cc5:
    code_gender = st.selectbox(
        f"{label_fr('CODE_GENDER')} (CODE_GENDER)",
        domain_gender, index=0, help=help_fr("CODE_GENDER")
    )
with cc6:
    family_status = st.selectbox(
        f"{label_fr('NAME_FAMILY_STATUS')} (NAME_FAMILY_STATUS)",
        domain_family, index=0, help=help_fr("NAME_FAMILY_STATUS")
    )

def build_payload(base_prefill: bool = True) -> Dict[str, Any]:
    base = {
        "AMT_INCOME_TOTAL": None if amt_income == 0 else amt_income,
        "AMT_CREDIT":       None if amt_credit == 0 else amt_credit,
        "AMT_ANNUITY":      None if annuity == 0 else annuity,
        "DAYS_BIRTH":       None if days_birth == 0 else days_birth,
        "DAYS_EMPLOYED":    None if days_employed == 0 else days_employed,
        "CNT_CHILDREN":     None if children == 0 else int(children),
        "NAME_CONTRACT_TYPE": contract if contract else None,
        "CODE_GENDER":       code_gender if code_gender else None,
        "NAME_FAMILY_STATUS": family_status if family_status else None,
    }
    if not EXPECTED_COLS:
        okc, js, _ = http_get_json(f"{API_URL}/expected_columns")
        cols = js.get("expected_columns", []) if okc else []
    else:
        cols = EXPECTED_COLS

    if cols:
        # Aligne sur les colonnes attendues
        return {col: base.get(col, None) for col in cols}
    return base

payload = build_payload()

with st.sidebar:
    non_empty = {k: v for k, v in payload.items() if v not in (None, "", 0)}
    if non_empty:
        st.json({"data": non_empty})
    else:
        st.caption("Renseigne revenu, crédit et âge pour voir l’aperçu.")

# Variables avancées (si présentes dans /expected_columns)
st.markdown("---")
with st.expander("🔧 Variables avancées du modèle (facultatif)", expanded=False):
    if not EXPECTED_COLS:
        st.caption("Endpoint /expected_columns indisponible (ou modèle minimal côté API).")
    else:
        already = {
            "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED",
            "CNT_CHILDREN","NAME_CONTRACT_TYPE","CODE_GENDER","NAME_FAMILY_STATUS",
        }
        cols_adv = st.columns(3)
        i = 0
        for col in EXPECTED_COLS:
            if col in already:
                continue
            ui = cols_adv[i % len(cols_adv)]
            with ui:
                domain = VALUE_DOMAINS.get(col)
                lab = f"{label_fr(col)} ({col})"
                hint = help_fr(col)
                if domain and isinstance(domain, list) and len(domain) > 0:
                    choice = st.selectbox(lab, [""] + [x for x in domain if x], index=0, help=hint)
                    if choice != "":
                        payload[col] = choice
                else:
                    # par défaut: champ numérique/txt libre
                    txt = st.text_input(lab, value="", help=hint or "Laisse vide si non renseigné")
                    if txt.strip() != "":
                        # essaie numérique
                        try:
                            payload[col] = float(txt.replace(",", "."))
                        except Exception:
                            payload[col] = txt.strip()
            i += 1
        st.caption("Laisse vide ce qui n’est pas utile. Les champs vides sont envoyés en `None`.")

# =========================================================
# Évaluation unitaire
# =========================================================
st.markdown("### 🧮 Évaluation")

@st.cache_data(ttl=30, show_spinner=False)
def _get_api_threshold(api_url: str) -> Optional[float]:
    ok, js, _ = http_get_json(f"{api_url}/")
    if ok and isinstance(js, dict):
        thr = js.get("decision_threshold")
        if isinstance(thr, dict) and "t_selected" in thr:
            try: return float(thr["t_selected"])
            except Exception: return None
        if isinstance(thr, (int, float)): return float(thr)
    return None

t_api = _get_api_threshold(API_URL)
if "t_ui" not in st.session_state:
    st.session_state["t_ui"] = t_api if isinstance(t_api, (int, float)) else 0.50

ctrl_l, ctrl_r = st.columns([1, 3])
with ctrl_l:
    do_predict = st.button("Évaluer ce dossier", type="primary")
with ctrl_r:
    st.caption("Ajuste le **seuil (UI)** pour voir l’impact sur la décision (l’API garde son seuil interne).")
    st.session_state["t_ui"] = st.slider("Seuil (UI) — refus si p ≥ t", 0.00, 1.00, float(st.session_state["t_ui"]), 0.01)
    if t_api is not None: st.caption(f"Seuil **API** (info) : {t_api:.3f} • {t_api:.1%}")

proba = None
thr_api_raw = None
if do_predict:
    ok, pred_json, err = http_post_json(f"{API_URL}/predict", {"data": payload})
    if not ok:
        st.error(f"Erreur /predict : {err}\n\nRéponse: {pred_json}")
    else:
        proba = pred_json.get("probability_default")
        thr_api_raw = pred_json.get("threshold")

        t_api_num = None
        if isinstance(thr_api_raw, dict) and "t_selected" in thr_api_raw:
            try: t_api_num = float(thr_api_raw["t_selected"])
            except Exception: t_api_num = None
        elif isinstance(thr_api_raw, (int, float)):
            t_api_num = float(thr_api_raw)

        t_ui = float(st.session_state.get("t_ui", 0.5))
        if proba is None:
            st.warning(f"Réponse inattendue: {pred_json}")
        else:
            decision_ui = 1 if float(proba) >= t_ui else 0
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilité de défaut", f"{float(proba):.2%}")
                st.metric("Décision", "❌ Refus" if decision_ui == 1 else "✅ Acceptation")
            with c2:
                st.metric("Seuil (UI)", f"{t_ui:.3f}  •  {t_ui:.1%}")
                if t_api_num is not None: st.caption(f"Seuil **API** (référence) : {t_api_num:.3f}  •  {t_api_num:.1%}")
                cmp_sym = ">=" if float(proba) >= t_ui else "<"
                final = "❌ Refus" if float(proba) >= t_ui else "✅ Acceptation"
                st.markdown(f"**Comparatif** : `p = {float(proba):.3f}` {cmp_sym} `t(UI) = {t_ui:.3f}` → **{final}**")
                st.caption("ℹ️ Le seuil transforme `p` en décision : si `p ≥ t` → **refus**, sinon **acceptation**.")

            # Bullet p vs t
            st.markdown("#### Repère visuel p vs t")
            fig_bullet = go.Figure(go.Indicator(
                mode="number+gauge",
                value=float(proba),
                number={"valueformat": ".3f"},
                title={"text": "p (barre) vs t(UI) (ligne rouge)"},
                gauge={"shape": "bullet", "axis": {"range": [0, 1]},
                       "bar": {"thickness": 0.6},
                       "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.85,
                                     "value": float(t_ui)}},
                domain={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            ))
            fig_bullet.update_layout(height=170, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bullet, use_container_width=True)

            # ---------------------------------------------------------
            # Explications des indicateurs + position vs population
            # ---------------------------------------------------------
            st.markdown("### 📘 Comprendre les indicateurs")

            def get_pop_stat(ref: dict, feature: str) -> Tuple[Optional[float], Optional[float]]:
                if not ref: return None, None
                d = None
                if isinstance(ref, dict):
                    if "features" in ref: d = ref["features"].get(feature)
                    elif "stats" in ref: d = ref["stats"].get(feature)
                if isinstance(d, dict):
                    mu = d.get("mean")
                    sig = d.get("std")
                    try:
                        return (float(mu) if mu is not None else None,
                                float(sig) if sig is not None else None)
                    except Exception:
                        return None, None
                return None, None

            def _safe(p: dict, k: str) -> Optional[float]:
                v = p.get(k)
                try:
                    return float(v) if v not in (None, "") else None
                except Exception:
                    return None

            rows = []
            # 1) probabilité de défaut
            rows.append({
                "Indicateur": "Probabilité de défaut (p)",
                "Définition": "Probabilité estimée que le client fasse défaut. Décision: refus si p ≥ seuil.",
                "Valeur": fmt_ratio(proba, pct=True, decimals=2) if proba is not None else "—",
                "Référence": "—",
                "Interprétation": "Plus p est élevé, plus le dossier est risqué."
            })

            # 2) seuil de décision
            rows.append({
                "Indicateur": "Seuil de décision (t)",
                "Définition": "Frontière qui sépare acceptation (p < t) et refus (p ≥ t).",
                "Valeur": fmt_ratio(t_ui, pct=True, decimals=2),
                "Référence": "Seuil API si défini côté serveur",
                "Interprétation": "Ajuster t permet de moduler la politique risque."
            })

            # 3) ratios explicatifs clés si présents dans payload
            for feat, lab in [
                ("CREDIT_INCOME_RATIO", "Ratio crédit/revenu"),
                ("ANNUITY_INCOME_RATIO", "Ratio annuité/revenu"),
                ("PAYMENT_RATE", "Taux de paiement (annuité/crédit)"),
            ]:
                val = _safe(payload, feat)
                mu, _sig = get_pop_stat(REF_STATS, feat) if REF_STATS else (None, None)
                if val is None and mu is None:
                    continue
                ref_txt = f"μ≈ {fmt_ratio(mu)}" if mu is not None else "—"
                interp = "—"
                if val is not None and mu is not None:
                    interp = "🟥 plutôt défavorable (effort élevé)" if val > mu else "🟩 plutôt favorable (effort contenu)"
                rows.append({
                    "Indicateur": lab,
                    "Définition": VAR_HELP.get(feat, "—"),
                    "Valeur": fmt_ratio(val),
                    "Référence": ref_txt,
                    "Interprétation": interp
                })

            df_explain = pd.DataFrame(rows)
            st.dataframe(df_explain, use_container_width=True)

            # Barres horizontales pour 3 ratios clés (si présents)
            bars = []
            cats = []
            for feat, lab in [("CREDIT_INCOME_RATIO","Crédit / revenu"),
                              ("ANNUITY_INCOME_RATIO","Annuité / revenu"),
                              ("PAYMENT_RATE","Annuité / crédit")]:
                v = _safe(payload, feat)
                if v is not None:
                    cats.append(lab)
                    bars.append(v)

            if bars:
                fig = go.Figure(go.Bar(x=bars, y=cats, orientation="h"))
                fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10),
                                  xaxis=dict(range=[0, max(1.0, max(bars)*1.2)]))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Lecture : plus le ratio est élevé, plus l’effort relatif est important (souvent moins favorable).")

# =========================================================
# Bas de page — Aide
# =========================================================
st.markdown("---")
st.info(
    "Astuce : dans la **sidebar**, colle l’URL publique du port 8000 (globe 🌍) pour connecter le dashboard à l’API du Codespace. "
    "Les libellés sont en français et certaines variables affichent leur **unité** et **définition** au survol."
)
