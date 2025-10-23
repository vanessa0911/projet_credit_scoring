#!/usr/bin/env bash
set -euo pipefail

echo "🧩 (1/3) Création/MAJ du requirements.txt…"
cat > requirements.txt << 'EOF'
# --- runtime ---
fastapi>=0.110,<0.116
uvicorn[standard]>=0.23,<0.33
pydantic>=2.5,<3
requests>=2.31,<3

# data utils
pandas>=2.1,<3
numpy>=1.26,<3

# dashboard
streamlit>=1.32,<2

# optional (graphiques / ML)
plotly>=5.18,<6
scikit-learn>=1.3,<2
EOF

echo "⚙️  (2/3) Upgrade pip/setuptools/wheel…"
python -m pip install --upgrade pip setuptools wheel

echo "📦 (3/3) Installation des dépendances…"
pip install -r requirements.txt --no-cache-dir

echo "✅ Environnement prêt."
