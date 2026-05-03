@echo off
setlocal

echo ========================================
echo ShelfVision: first start control panel
echo ========================================
echo.

cd /d %~dp0\..\..

if not exist .venv\Scripts\python.exe (
  echo [1/4] Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Check that Python is installed and added to PATH.
    pause
    exit /b 1
  )
) else (
  echo [1/4] Virtual environment already exists.
)

echo [2/4] Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo [3/4] Installing minimal packages for control panel ...
.venv\Scripts\python.exe -m pip install streamlit PyYAML pandas
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
echo Starting ShelfVision Control Panel ...
.venv\Scripts\python.exe -m streamlit run scripts\control_panel.py

pause
