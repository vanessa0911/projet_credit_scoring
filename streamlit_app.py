# streamlit_app.py
# --------------------------------------------------------
# Dashboard Streamlit pour le projet credit_scoring
# - API URL configurable (sidebar)
# - Healthcheck + statut clair
# - Récupération des colonnes attendues (cache)
# - Formulaire dynamique pour une prédiction unitaire
# - Upload CSV pour les prédictions en lot (batch)
# - Lecture optionnelle des artefacts locaux (artifacts/)
# - Gestion d'erreurs robuste (try/except + messages clairs)
# --------------------------------------------------------

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# --------- Utils ---------
# =========================

def _default_api_url() -> str:
    # Permet aussi de surcharger via une variable d'environnement
    return os.environ.get("API_URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=10)
def check_api_health(api_url: str) -> Tuple[bool, Dict[str, Any] | None, str | None]:
    """Ping l'API (GET /) pour vérifier qu'elle répond."""
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        r.raise_for_status()
        return True, r.json() if r.headers.get("content-type", "").startswith("application/json") else {"status": r.text}, None
    except Exception as e:
        return False, None, str(e)


@st.cache_data(ttl=300)
def get_expected_columns(api_url: str) -> Tuple[List[str] | None, str | None]:
    """Récupère la liste des colonnes attendues (GET /expected_columns)."""
    try:
        r = requests.get(f"{api_url}/expected_columns", timeout=10)
        r.raise_for_status()
        data = r.json()
        # On accepte plusieurs formats: {"columns": [...]}, ["..."], {"expected_columns":[...]}
        if isinstance(data, dict):
            cols = data.get("columns") or data.get("expected_columns")
        elif isinstance(data, list):
            cols = data
        else:
            cols = None
        if cols and all(isinstance(c, str) for c in cols):
            return cols, None
        return None, "Réponse inattendue du endpoint /expected_columns."
    except Exception as e:
        return None, str(e)


def _try_predict(api_url: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str | None]:
    """POST /predict avec différentes formes de payload pour s'adapter à l'API."""
    urls = [f"{api_url}/predict", f"{api_url}/predict_proba", f"{api_url}/predict_proba_single"]
    # Essaye plusieurs schémas classiques : {"features": ...}, {"data": ...}, {...}
    candidate_bodies = [
        {"features": payload},
        {"data": payload},
        payload,
    ]
    for url in urls:
        for body in candidate_bodies:
            try:
                r = requests.post(url, json=body, timeout=20)
                if r.status_code == 404:
                    # endpoint absent -> essaie le suivant
                    continue
                r.raise_for_status()
                # Retourne le JSON tel quel (plus flexible)
                return r.json(), None
            except requests.exceptions.HTTPError as he:
                # 422 = payload schema mismatch -> essaie une autre forme
                if r is not None and r.status_code in (400, 404, 415, 422):
                    continue
                return None, f"Erreur HTTP depuis {url}: {he}"
            except Exception as e:
                # Timeout/connexion -> ne casse pas tout de suite, on essaie l'autre forme
                continue
    return None, "Impossible d'appeler /predict avec les schémas standards (features/data). Vérifie l'API."


def _try_predict_batch(api_url: str, df: pd.DataFrame) -> Tuple[pd.DataFrame | None, str | None]:
    """
    Tente /predict_proba_batch de trois façons :
    A) multipart (fichier CSV)
    B) JSON {"rows":[{...}, ...]}
    C) Fallback : boucle sur /predict ligne à ligne
    """
    # A) multipart CSV
    try:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        files = {"file": ("input.csv", buf.getvalue(), "text/csv")}
        url = f"{api_url}/predict_proba_batch"
        r = requests.post(url, files=files, timeout=60)
        if r.status_code not in (404, 415):
            r.raise_for_status()
            data = r.json()
            # Formats acceptés : {"predictions":[...]} ou {"proba":[...]} ou direct [...]
            if isinstance(data, dict):
                preds = data.get("predictions") or data.get("proba") or data.get("result")
            else:
                preds = data
            if isinstance(preds, list) and len(preds) == len(df):
                out = df.copy()
                # nom de colonne robuste
                out["proba_default"] = [float(x) if x is not None else None for x in preds]
                return out, None
    except Exception:
        pass

    # B) JSON rows
    try:
        url = f"{api_url}/predict_proba_batch"
        body = {"rows": df.to_dict(orient="records")}
        r = requests.post(url, json=body, timeout=60)
        if r.status_code != 404:
            r.raise_for_status()
            data = r.json()
            preds = None
            if isinstance(data, dict):
                preds = data.get("predictions") or data.get("proba") or data.get("result")
            else:
                preds = data
            if isinstance(preds, list) and len(preds) == len(df):
                out = df.copy()
                out["proba_default"] = [float(x) if x is not None else None for x in preds]
                return out, None
    except Exception:
        pass

    # C) Fallback ligne à ligne
    try:
        results = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            j, err = _try_predict(api_url, payload)
            if j is None:
                results.append(None)
                continue
            # Cherche une proba dans des clés usuelles
            proba = None
            for k in ("proba", "probability", "score", "proba_default", "p_default", "y_proba"):
                if k in j:
                    proba = j[k]
                    break
            # Sinon, si dict avec "prediction": 0/1, garde quand même quelque chose
            if proba is None and "prediction" in j:
                proba = float(j["prediction"])
            results.append(proba)
        out = df.copy()
        out["proba_default"] = results
        return out, None
    except Exception as e:
        return None, f"Échec du fallback batch: {e}"


def _coerce_value(s: str) -> Any:
    """Convertit proprement une saisie texte en nombre si possible, sinon garde la chaîne."""
    s = s.strip()
    if s == "":
        return None
    # Int ?
    try:
        if s.isdigit() or (s[0] in "+-" and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    # Float ?
    try:
        return float(s.replace(",", "."))  # tolère les virgules
    except Exception:
        return s


# =========================
# ---------  UI  ----------
# =========================

st.set_page_config(page_title="Credit Scoring – Dashboard", layout="wide")

st.title("📊 Credit Scoring – Dashboard")

with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input(
        "URL de l'API",
        value=_default_api_url(),
        help="Exemple: http://127.0.0.1:8000 (en Codespaces, c'est correct par défaut).",
    )

    ok, health, err = check_api_health(api_url)
    if ok:
        st.success("API: ✅ joignable")
        if health:
            st.caption(f"Health: {json.dumps(health)[:200]}{'...' if len(json.dumps(health))>200 else ''}")
    else:
        st.error("API: ❌ non joignable")
        if err:
            st.caption(err)

    cols, cols_err = get_expected_columns(api_url)
    if cols:
        st.success(f"Colonnes attendues: {len(cols)} trouvées")
        with st.expander("Voir la liste des colonnes"):
            st.code("\n".join(cols))
    else:
        st.warning("Impossible de récupérer /expected_columns.")
        if cols_err:
            st.caption(cols_err)

# --- Tabs principales
tab_single, tab_batch, tab_artifacts = st.tabs(["🔎 Évaluation unitaire", "📁 Évaluation en lot (CSV)", "📚 Artefacts & interprétabilité"])

# --------------------------------------------------------
#                   Tab 1 : Unitaire
# --------------------------------------------------------
with tab_single:
    st.subheader("Évaluation d'un dossier (unitaire)")
    st.write("Renseigne les caractéristiques du client/dossier selon les colonnes attendues par l'API.")

    # Formulaire dynamique :
    # - Si on connaît les colonnes, on les utilise
    # - Sinon, proposer 3 champs courants en fallback
    input_data: Dict[str, Any] = {}

    if cols:
        st.info("Champs dynamiques construits à partir de /expected_columns.")
        # On affiche en 2 colonnes pour compacité
        left, right = st.columns(2)
        inputs = {}
        for i, c in enumerate(cols):
            placeholder = "Valeur (numérique ou texte)"
            # Heuristiques légères pour proposer un placeholder pertinent
            if any(k in c.lower() for k in ("age", "days", "annee", "year")):
                placeholder = "Ex: 35"
            elif any(k in c.lower() for k in ("amount", "amt", "credit", "loan", "revenue", "income", "montant")):
                placeholder = "Ex: 12000"
            container = left if i % 2 == 0 else right
            val = container.text_input(c, value="", placeholder=placeholder)
            inputs[c] = _coerce_value(val)
        input_data = inputs
    else:
        st.info("Fallback minimal (liste des colonnes inconnue).")
        c1, c2, c3 = st.columns(3)
        age = c1.text_input("age", value="35")
        amt = c2.text_input("credit_amount", value="12000")
        income = c3.text_input("annual_income", value="48000")
        input_data = {
            "age": _coerce_value(age),
            "credit_amount": _coerce_value(amt),
            "annual_income": _coerce_value(income),
        }

    if st.button("🚀 Évaluer ce dossier", type="primary"):
        with st.spinner("Appel de l'API…"):
            res, err = _try_predict(api_url, input_data)
        if err:
            st.error(err)
        elif res is None:
            st.error("Réponse vide de l'API.")
        else:
            st.success("Prédiction reçue")
            st.json(res)

            # Affichage pratique si quelques clés usuelles existent
            proba = None
            for k in ("proba", "probability", "score", "proba_default", "p_default", "y_proba"):
                if k in res:
                    proba = res[k]
                    break
            pred = res.get("prediction") if isinstance(res, dict) else None

            if proba is not None:
                st.metric("Probabilité de défaut", f"{float(proba):.3f}")
            if pred is not None:
                st.metric("Décision / Classe", str(pred))

# --------------------------------------------------------
#                   Tab 2 : Batch CSV
# --------------------------------------------------------
with tab_batch:
    st.subheader("Prédictions en lot (CSV)")
    st.write("Charge un fichier CSV dont les **colonnes correspondent** à `/expected_columns`.")
    file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            df = None

        if df is not None:
            st.write("Aperçu des données :")
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("🚀 Lancer la prédiction en lot"):
                with st.spinner("Appel batch en cours…"):
                    out, err = _try_predict_batch(api_url, df)
                if err:
                    st.error(err)
                elif out is None:
                    st.error("Réponse vide en batch.")
                else:
                    st.success("Prédictions lot reçues")
                    st.dataframe(out.head(50), use_container_width=True)
                    # Proposition de téléchargement
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Télécharger résultats (CSV)",
                        data=csv_bytes,
                        file_name="predictions_batch.csv",
                        mime="text/csv",
                    )

# --------------------------------------------------------
#         Tab 3 : Artefacts & Interprétabilité
# --------------------------------------------------------
with tab_artifacts:
    st.subheader("Artefacts (optionnel)")
    st.caption("Si tu as généré les artefacts en local (scripts `make_*.py`), ils seront affichés ici.")

    artifacts_dir = Path("artifacts")
    ref_stats_path = artifacts_dir / "ref_stats.json"
    glob_imp_path = artifacts_dir / "global_importance.csv"
    interp_summary_path = artifacts_dir / "interpretability_summary.json"

    if ref_stats_path.exists():
        st.success("✅ ref_stats.json trouvé")
        try:
            ref_stats = json.loads(ref_stats_path.read_text(encoding="utf-8"))
            st.write("**Statistiques de référence (extrait)**")
            # On affiche uniquement les 3000 premiers caractères pour éviter l'inondation
            dump = json.dumps(ref_stats, ensure_ascii=False)[:3000]
            st.code(dump + ("..." if len(json.dumps(ref_stats)) > 3000 else ""))
        except Exception as e:
            st.error(f"Lecture de ref_stats.json impossible: {e}")
    else:
        st.info("ref_stats.json non trouvé. Lance `python make_ref_stats.py` si besoin.")

    if glob_imp_path.exists():
        st.success("✅ global_importance.csv trouvé")
        try:
            gi = pd.read_csv(glob_imp_path)
            st.write("**Importance globale des variables (top 30)**")
            # On essaie d'interpréter des colonnes typiques: feature / importance
            cols_gi = [c.lower() for c in gi.columns]
            # Recherche de noms usuels
            feat_col = next((c for c in gi.columns if c.lower() in ("feature", "variable", "name")), gi.columns[0])
            imp_col = next((c for c in gi.columns if "imp" in c.lower() or "gain" in c.lower() or "weight" in c.lower()), gi.columns[-1])
            gi_show = gi[[feat_col, imp_col]].rename(columns={feat_col: "feature", imp_col: "importance"}).copy()
            gi_show = gi_show.sort_values("importance", ascending=False).head(30)
            st.dataframe(gi_show, use_container_width=True)
            st.bar_chart(gi_show.set_index("feature"))
        except Exception as e:
            st.error(f"Lecture de global_importance.csv impossible: {e}")
    else:
        st.info("global_importance.csv non trouvé. Lance `python make_global_importance.py` si besoin.")

    if interp_summary_path.exists():
        st.success("✅ interpretability_summary.json trouvé")
        try:
            inter = json.loads(interp_summary_path.read_text(encoding="utf-8"))
            st.write("**Interprétabilité – résumé (extrait)**")
            dump = json.dumps(inter, ensure_ascii=False)[:3000]
            st.code(dump + ("..." if len(json.dumps(inter)) > 3000 else ""))
        except Exception as e:
            st.error(f"Lecture d'interpretability_summary.json impossible: {e}")
    else:
        st.info("interpretability_summary.json non trouvé (optionnel).")
