#!/usr/bin/env bash
set -euo pipefail

echo "🧩 Création de requirements.txt…"
cat > requirements.txt << 'EOF'
# --- runtime ---
fastapi>=0.110,<0.116
uvicorn[standard]>=0.23,<0.33
pydantic>=2.5,<3
requests>=2.31,<3

# data utils (needed by your app)
pandas>=2.1,<3
numpy>=1.26,<3

# dashboard
streamlit>=1.32,<2

# optional but supported by your app (won't break if unused)
scikit-learn>=1.3,<2
plotly>=5.18,<6
EOF

echo "⚙️  Upgrade pip/setuptools/wheel…"
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installation des dépendances du projet…"
pip install -r requirements.txt --no-cache-dir

echo "✅ Environnement prêt."
