@echo off
setlocal

echo ========================================
echo ShelfVision: interface_app.py
echo ========================================

cd /d %~dp0\..\..
python -m streamlit run scripts/interface_app.py

pause
