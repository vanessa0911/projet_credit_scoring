#!/usr/bin/env bash
set -euo pipefail
python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
