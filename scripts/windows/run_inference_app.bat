@echo off
setlocal

echo ========================================
echo ShelfVision: inference_app.py
echo ========================================

cd /d %~dp0\..\..
python -m streamlit run scripts/inference_app.py

pause
