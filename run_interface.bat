@echo off
chcp 65001 >nul
cd /d %~dp0

echo Starting ShelfVision experiment interface on http://localhost:8502 ...

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run scripts/interface_app.py --server.port 8502 --server.headless true
) else (
  python -m streamlit run scripts/interface_app.py --server.port 8502 --server.headless true
)

pause
