@echo off
setlocal

echo ========================================
echo ShelfVision: reset Windows .venv
echo ========================================

cd /d %~dp0\..\..

if exist .venv (
  echo Removing .venv ...
  rmdir /s /q .venv
) else (
  echo .venv does not exist.
)

echo Creating Windows .venv ...
python -m venv .venv
if errorlevel 1 (
  echo Failed to create .venv. Check Windows Python installation.
  pause
  exit /b 1
)

echo Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo Installing minimal Control Panel packages ...
.venv\Scripts\python.exe -m pip install streamlit PyYAML pandas
if errorlevel 1 (
  echo Failed to install minimal packages.
  pause
  exit /b 1
)

echo.
echo Windows .venv is ready. You can run scripts\windows\start_control_panel.bat
pause
