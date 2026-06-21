#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv_wsl/bin/python" ]]; then
  PYTHON=".venv_wsl/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

"$PYTHON" -m streamlit run scripts/final_demo_app.py
