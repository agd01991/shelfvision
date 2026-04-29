#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting ShelfVision interface..."
python -m streamlit run scripts/interface_app.py
