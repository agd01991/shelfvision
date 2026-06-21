@echo off
setlocal
cd /d "%~dp0\..\.."

if exist ".venv\Scripts\python.exe" (
  set PYTHON=.venv\Scripts\python.exe
) else (
  set PYTHON=python
)

%PYTHON% -m streamlit run scripts\final_demo_history_app.py
endlocal
