@echo off
setlocal

echo ========================================
echo ShelfVision: reset WSL .venv_wsl
echo ========================================

cd /d %~dp0\..\..

set WSL_VENV=.venv_wsl
set REQUIREMENTS=requirements.txt

echo Removing WSL venv if exists ...
wsl bash -lc "cd \"$(wslpath '%cd%')\" && rm -rf '%WSL_VENV%'"
if errorlevel 1 (
  echo Failed to remove .venv_wsl through WSL.
  pause
  exit /b 1
)

echo Recreating WSL environment and installing requirements ...
python scripts\setup_wsl_env.py --venv-dir "%WSL_VENV%" --requirements "%REQUIREMENTS%"
if errorlevel 1 (
  echo Failed to setup WSL environment.
  pause
  exit /b 1
)

echo.
echo WSL .venv_wsl is ready.
pause
