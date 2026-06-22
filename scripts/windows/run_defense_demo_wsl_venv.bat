@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem ShelfVision defense demo launcher.
rem Runs Streamlit strictly through WSL and .venv_wsl.

set "PORT=8508"
if not "%DEMO_PORT%"=="" set "PORT=%DEMO_PORT%"

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "APP=scripts/final_demo_history_app.py"
set "URL=http://localhost:%PORT%"

where wsl.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: wsl.exe was not found. Install or enable Windows Subsystem for Linux.
    pause
    exit /b 1
)

for /f "usebackq delims=" %%I in (`wsl.exe wslpath -a "%PROJECT_ROOT%"`) do set "WSL_PROJECT_ROOT=%%I"

if "%WSL_PROJECT_ROOT%"=="" (
    echo ERROR: failed to convert project path to WSL path.
    echo Project root: %PROJECT_ROOT%
    pause
    exit /b 1
)

echo Project root: %PROJECT_ROOT%
echo WSL root:     %WSL_PROJECT_ROOT%
echo URL:          %URL%
echo.

wsl.exe bash -lc "cd '%WSL_PROJECT_ROOT%' && test -x .venv_wsl/bin/python"
if errorlevel 1 (
    echo ERROR: .venv_wsl/bin/python was not found in the project.
    echo Run in WSL from the project root:
    echo   python3 -m venv .venv_wsl
    echo   .venv_wsl/bin/python -m pip install --upgrade pip
    echo   .venv_wsl/bin/python -m pip install -r requirements.txt
    pause
    exit /b 1
)

wsl.exe bash -lc "cd '%WSL_PROJECT_ROOT%' && test -f '%APP%'"
if errorlevel 1 (
    echo ERROR: app file was not found: %APP%
    pause
    exit /b 1
)

echo Starting Streamlit demo through WSL .venv_wsl...
echo Press Ctrl+C in this window to stop the demo.
echo.

start "" "%URL%"

wsl.exe bash -lc "cd '%WSL_PROJECT_ROOT%' && export PYTHONPATH=. && .venv_wsl/bin/python -m streamlit run %APP% --server.address 127.0.0.1 --server.port %PORT% --browser.gatherUsageStats false"

echo.
echo Demo server stopped.
pause
endlocal
