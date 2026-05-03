@echo off
setlocal

echo ========================================
echo ShelfVision: first start control panel
echo ========================================
echo.

cd /d %~dp0\..\..

set WIN_VENV=.venv
set WIN_PY=%WIN_VENV%\Scripts\python.exe

if exist "%WIN_PY%" (
  "%WIN_PY%" -c "import sys; print(sys.executable)" >nul 2>nul
  if errorlevel 1 (
    echo [1/4] Existing .venv is broken or was created inside WSL. Recreating Windows .venv ...
    rmdir /s /q "%WIN_VENV%"
  ) else (
    echo [1/4] Windows virtual environment already exists.
  )
)

if not exist "%WIN_PY%" (
  echo [1/4] Creating Windows virtual environment .venv ...
  python -m venv "%WIN_VENV%"
  if errorlevel 1 (
    echo Failed to create .venv. Check that Python for Windows is installed and added to PATH.
    echo You can test it with: python --version
    pause
    exit /b 1
  )
)

echo [2/4] Upgrading pip in Windows .venv ...
"%WIN_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip in Windows .venv.
  pause
  exit /b 1
)

echo [3/4] Installing minimal packages for control panel ...
"%WIN_PY%" -m pip install streamlit PyYAML pandas
if errorlevel 1 (
  echo Failed to install minimal packages.
  pause
  exit /b 1
)

if not exist config\shelfvision.yaml (
  echo [4/4] Creating config\shelfvision.yaml from example ...
  if not exist config mkdir config
  copy config\shelfvision.example.yaml config\shelfvision.yaml >nul
) else (
  echo [4/4] Config already exists.
)

echo.
echo Starting ShelfVision Control Panel with WSL runtime support ...
echo Windows .venv is used only for the panel.
echo Work tasks can run through WSL .venv_wsl after scripts\windows\setup_wsl_env.bat.
"%WIN_PY%" -m streamlit run scripts\control_panel_wsl.py

pause
