# 🧩 Projet Credit Scoring — Version 0 (Stable)

## ✅ État fonctionnel validé
Cette version correspond à la configuration **stable et fonctionnelle** du projet.  
Elle sert de **point de restauration officiel** (« version 0 ») en cas de régression.

---

## 🚀 Fonctionnalités opérationnelles
- **API (FastAPI)** totalement fonctionnelle :
  - Endpoint `/health` → indique l’état de santé de l’API.
  - Endpoint `/expected_columns` → liste des colonnes du dataset.
  - Endpoint `/top_features` → détecte automatiquement les 10 variables les plus importantes à partir des artefacts.
  - Endpoint `/predict` → calcule une probabilité de défaut et une décision.
- **Dashboard (Streamlit)** :
  - Connexion automatique à l’API via URL (sidebar).
  - Saisie intuitive des variables principales et avancées.
  - Affichage des résultats de prédiction (probabilité, seuil, décision).
  - En cas de refus : conseils personnalisés pour passer à une décision favorable.
- **Fallback complet** : si le dataset ou les artefacts ne sont pas trouvés, l’application continue à fonctionner avec des valeurs par défaut.

---

## 🧠 Fichiers clés

| Fichier | Description |
|----------|--------------|
| `api.py` | API FastAPI robuste avec détection automatique des artefacts et fallback intégré. |
| `streamlit_app.py` | Interface utilisateur Streamlit simplifiée et ergonomique. |
| `artifacts/` | Contient les fichiers d’interprétabilité (`*.csv`, `*.json`) pour la détection des variables clés. |
| `data/` | Contient les datasets de base (`application_train.csv`, etc.). |
| `requirements.txt` | Liste des dépendances Python. |

---

## ⚙️ Commandes de lancement

### Terminal A → Lancer l’API
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Terminal B → Lancer le Dashboard
python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

Ouvrir le Dashboard

Dans GitHub Codespaces, onglet Ports → ouvrir le port 8501 (globe 🌍)

Dans la sidebar Streamlit, définir l’URL de l’API sur l’adresse publique du port 8000 :

https://<ton-codespace>-8000.app.github.dev



