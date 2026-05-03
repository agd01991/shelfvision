@echo off
setlocal

echo ========================================
echo ShelfVision: setup dependencies through WSL
echo ========================================

cd /d %~dp0\..\..

set WSL_VENV=.venv_wsl
set REQUIREMENTS=requirements.txt

echo WSL_VENV=%WSL_VENV%
echo REQUIREMENTS=%REQUIREMENTS%
echo.

python scripts\setup_wsl_env.py --venv-dir "%WSL_VENV%" --requirements "%REQUIREMENTS%"

pause
