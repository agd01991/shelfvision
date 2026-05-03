@echo off
setlocal

echo ========================================
echo ShelfVision: video_app.py
echo ========================================

cd /d %~dp0\..\..
python scripts\wsl_runtime.py --venv-dir .venv_wsl -m streamlit run scripts/video_app.py

pause
