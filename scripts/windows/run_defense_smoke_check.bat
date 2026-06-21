@echo off
setlocal
cd /d "%~dp0\..\.."

if exist ".venv_wsl\Scripts\python.exe" (
  set PYTHON=.venv_wsl\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
  set PYTHON=.venv\Scripts\python.exe
) else (
  set PYTHON=python
)

%PYTHON% scripts\defense_demo_smoke_check.py %*
endlocal
