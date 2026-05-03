@echo off
setlocal

echo ========================================
echo ShelfVision: Control Panel
echo ========================================

cd /d %~dp0\..\..

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m streamlit run scripts\control_panel.py
) else (
  python -m streamlit run scripts\control_panel.py
)

pause
