@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo_app.ps1"
set EXIT_CODE=%ERRORLEVEL%
endlocal & exit /b %EXIT_CODE%
