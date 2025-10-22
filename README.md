🧠 Projet Credit Scoring — Dashboard & API
🎯 Objectif du projet

Ce projet illustre une approche complète de scoring de crédit, depuis la génération des données de référence jusqu’à la visualisation des résultats dans un dashboard Streamlit connecté à une API FastAPI.
L’objectif est de prédire la probabilité qu’un client rembourse un prêt, et de rendre les décisions explicables et explorables.

🗂️ Structure du dépôt
projet_credit_scoring/
├── api.py                         # API FastAPI pour la prédiction et l'explicabilité
├── streamlit_app.py               # Dashboard Streamlit (interface utilisateur)
├── make_ref_stats.py              # Script pour générer les stats de référence
├── make_global_importance.py      # Script pour générer les importances globales
│
├── data/                          # Données sources (ex: application_train.csv)
├── artifacts/                     # Fichiers générés (ref_stats.json, global_importance.csv…)
│
├── requirements.txt               # Dépendances Python
├── .devcontainer/devcontainer.json # Configuration Codespaces (Python, ports, etc.)
├── .streamlit/config.toml         # Config serveur Streamlit
└── README.md                      # Ce document

🚀 Lancer le projet dans GitHub Codespaces

🧩 Ces instructions sont destinées aux débutants : tu peux tout faire directement dans GitHub Codespaces, sans rien installer sur ton ordinateur.

1️⃣ Ouvrir le projet

Sur la page GitHub du dépôt → clique sur le bouton vert Code

Sélectionne Create codespace on main

Attends que VS Code (version web) s’ouvre avec le projet

2️⃣ Installation automatique (via devcontainer)

Si ton fichier .devcontainer/devcontainer.json contient :

"postCreateCommand": "git lfs install && git lfs pull || true && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt --no-cache-dir"


➡️ L’environnement s’installe automatiquement à la création du Codespace.

Sinon, tu peux le faire manuellement dans le terminal intégré :

git lfs install && git lfs pull || true
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

3️⃣ Générer les artefacts (si besoin)

Si les fichiers ref_stats.json et global_importance.csv n’existent pas encore :

mkdir -p data artifacts
python make_ref_stats.py
python make_global_importance.py


➡️ Les scripts lisent les données du dossier data/ (ex: application_train.csv) et enregistrent les résultats dans artifacts/.

4️⃣ Démarrer l’API FastAPI

L’API fournit les endpoints /predict, /predict_proba_batch, etc.

Lance-la avec :

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload


💡 Codespaces détecte le port 8000 : choisis Open in Browser pour voir l’API en action.

Endpoints principaux :

Endpoint	Description
/	Page d’accueil de l’API
/expected_columns	Liste les variables attendues par le modèle
/predict	Retourne la prédiction (accordé/refusé) pour un dossier
/predict_proba_batch	Retourne les probabilités pour un fichier CSV
5️⃣ Lancer le dashboard Streamlit

Dans un nouveau terminal (laisse l’API tourner dans le premier) :

python -m streamlit run streamlit_app.py


Codespaces détecte le port 8501 → choisis Open in Browser.

Le fichier .streamlit/config.toml garantit que Streamlit tourne sur le bon hôte :

[server]
address = "0.0.0.0"
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

🧭 Utilisation du dashboard
🔌 Connexion à l’API

L’URL par défaut est :
http://127.0.0.1:8000

Si l’état de l’API dans l’encart supérieur affiche ❌, vérifie :

que FastAPI tourne toujours dans le terminal,

que tu as bien ouvert le bon port.

🧍 Mode dossier unique

Renseigne les valeurs demandées (revenu, montant du crédit, âge, etc.)

Clique sur Évaluer ce dossier

Le résultat s’affiche : probabilité de remboursement + décision automatique

📈 Mode batch (fichier CSV)

Téléverse un fichier .csv via la barre latérale
→ Il doit contenir les colonnes attendues (/expected_columns)

Le tableau de résultats s’affiche automatiquement.

🧩 Visualisations

Comparaison population : place le client dans la distribution des emprunteurs similaires

Variables influentes : affiche les features les plus déterminantes dans la décision

🧰 Structure logique du projet
Élément	Rôle
api.py	Sert les prédictions via FastAPI
streamlit_app.py	Interface interactive Streamlit
make_ref_stats.py	Calcule les statistiques de référence
make_global_importance.py	Calcule les importances globales des features
artifacts/	Contient les fichiers utilisés par le dashboard
data/	Données d’entraînement (non versionnées en production)
🧑‍💻 Dépannage express
Problème	Solution
🔴 Erreur multiarray ou _ARRAY_API	Réinstalle ces versions : pandas==2.2.2, numpy==2.1.1, pyarrow==17.0.0
🔴 L’API n’est pas accessible	Vérifie que FastAPI tourne (uvicorn) et que l’URL de la sidebar est correcte
🔴 Port occupé	Change le port : --port 8001 ou --port 8502
🔴 Dashboard vide	Supprime le cache : streamlit cache clear
🔴 Artefacts manquants	Relance make_ref_stats.py et make_global_importance.py
🧪 Tests rapides (API)
# Test de l’endpoint racine
curl -s http://127.0.0.1:8000/ | jq .

# Vérifier les colonnes attendues
curl -s http://127.0.0.1:8000/expected_columns | jq .

📦 Requirements (versions stables recommandées)
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2

streamlit==1.38.0
pandas==2.2.2
numpy==2.1.1
pyarrow==17.0.0
scikit-learn==1.5.2

joblib==1.4.2
matplotlib==3.9.2
plotly==5.24.1
requests==2.32.3

🧩 Pour aller plus loin

Ajouter un modèle plus robuste (XGBoost, LightGBM)

Connecter à une base SQL pour les prédictions en production

Mettre en place une CI/CD pour déploiement automatique sur Render, Deta ou HuggingFace Spaces

🧾 Licence

Projet open source sous licence MIT.
Tu peux l’utiliser librement à des fins pédagogiques ou démonstratives.

💡 Auteurs & Crédits

Projet développé par Vanessa
Inspiré par les bonnes pratiques MLOps & DataViz.
Documentation et pipeline adaptés pour un usage pédagogique complet.
