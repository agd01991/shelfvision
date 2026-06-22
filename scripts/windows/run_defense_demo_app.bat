@echo off
setlocal
cd /d "%~dp0\..\.."

rem Prefer the WSL environment used for the final experiment.
where wsl.exe >nul 2>nul
if %errorlevel%==0 if exist ".venv_wsl\bin\python" (
  for /f "usebackq delims=" %%I in (`wsl.exe wslpath -a "%CD%"`) do set "WSL_ROOT=%%I"
  echo Starting demo in WSL environment: .venv_wsl
  wsl.exe bash -lc "cd \"$WSL_ROOT\" && .venv_wsl/bin/python -m streamlit run scripts/final_demo_history_app.py"
  goto :end
)

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo WSL environment not found. Starting with: %PYTHON%
%PYTHON% -m streamlit run scripts\final_demo_history_app.py

:end
endlocal
