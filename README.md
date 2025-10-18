Cheatsheet — Lancer, régénérer, dépanner
1) Préparer l’environnement
# Se placer dans le projet + activer le venv
cd "C:\Users\Maintenant Prêt\Desktop\Projet_credit_scoring"
.\.venv\Scripts\Activate.ps1

# Mettre à jour l’outillage + installer les deps figées
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# (optionnel) Variables d’env – utile si l’API n’est pas en local
$env:API_URL = "http://127.0.0.1:8000"


👉 Vérifier Python : python -V doit renvoyer 3.12.x.

2) (Re)générer les artefacts du dashboard
# Stats population + features dérivées (écrit artifacts/ref_stats.json)
python make_ref_stats.py

# Importances globales (écrit artifacts/global_importance.csv + interpretability_summary.json)
python make_global_importance.py


make_ref_stats.py lit data\application_train.csv (chemin déjà configuré dans le script).
Si tu changes d’emplacement, adapte TRAIN_CSV dans le script.

3) Démarrer l’API de scoring (FastAPI)
python -m uvicorn api:app --reload


Smoke tests rapides :

Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing        # status 200, montre chosen_model & threshold
Invoke-WebRequest http://127.0.0.1:8000/expected_columns -UseBasicParsing  # doit renvoyer count > 0


🧠 Endpoints principaux :

GET / : statut & méta (modèle, seuil…)

GET /expected_columns : colonnes d’entrée requises

POST /predict : {"data": {...}} → proba & décision

POST /predict_proba_batch : {"records":[{...},{...}]} → liste de résultats

4) Lancer le dashboard (Streamlit)
# Toujours avec le Python du venv :
python -m streamlit run streamlit_app.py


Dans l’UI :

Vérifie l’encart État de l’API (doit être ✅).

Renseigne revenu/crédit/âge → clique Évaluer ce dossier.

Sections Comparaison population et Variables influentes s’activent si les artefacts sont présents.

Changer l’URL de l’API : dans la sidebar, champ “API URL” (le cache est géré, mise à jour immédiate).

5) Mode batch (CSV)

Dans la sidebar du dashboard → Prédictions en lot (CSV) :

envoie un CSV avec les mêmes colonnes que /expected_columns (les colonnes manquantes sont complétées à None).

la réponse s’affiche dans un tableau (proba + décision).

6) Git — sauvegarder tes modifs
# Toujours à la racine du projet & venv activé
git checkout -b feat/dashboard-v1

# Ajouter les fichiers utiles (pas .venv/ ni data/)
git add api.py streamlit_app.py make_ref_stats.py make_global_importance.py `
        requirements.txt .gitignore .gitattributes `
        artifacts/metadata.json artifacts/ref_stats.json artifacts/global_importance.csv

git commit -m "feat: dashboard Streamlit + API robustifiée + artefacts utiles"
git push -u origin feat/dashboard-v1


Ouvre la PR : https://github.com/vanessa0911/projet_credit_scoring/compare

Gros fichiers : les modèles *.joblib et *.npy sont suivis via Git-LFS (déjà réglé par .gitattributes).
Si GitHub refuse un push (>100 MB), exécute :

git lfs install
git lfs migrate import --include="artifacts/*.joblib,artifacts/*.npy"
git push --force-with-lease

7) Dépannage express

Streamlit/Pandas/NumPy/Arrow : si tu vois _ARRAY_API ou multiarray → tes versions sont maintenant pinnées dans requirements.txt. Réinstalle :

pip uninstall -y pyarrow numpy pandas
pip install -r requirements.txt --no-cache-dir


Port déjà utilisé :
netstat -ano | findstr :8000 → récupère le PID → taskkill /PID <PID> /F

Cache Streamlit :
streamlit cache clear

API non joignable : assure-toi que uvicorn tourne, et que API_URL dans Streamlit pointe bien vers http://127.0.0.1:8000.

expected_columns vide : vérifie artifacts/metadata.json (clé expected_input_columns) ou laisse l’API déduire depuis le modèle; relance uvicorn.

8) Structure du repo (rappel)
Projet_credit_scoring/
├─ api.py                     # API FastAPI
├─ streamlit_app.py           # UI Streamlit
├─ make_ref_stats.py          # Stats population (ref_stats.json)
├─ make_global_importance.py  # Importances (global_importance.csv)
├─ requirements.txt
├─ .gitignore / .gitattributes (LFS)
├─ data/                      # (ignoré par Git)
└─ artifacts/
   ├─ metadata.json
   ├─ model_*.joblib         # (via Git-LFS)
   ├─ feature_names.npy      # (via Git-LFS)
   ├─ ref_stats.json
   ├─ global_importance.csv
   └─ interpretability_summary.json
