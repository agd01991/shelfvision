@echo off
setlocal

echo ========================================
echo ShelfVision: redirect to WSL Control Panel
echo ========================================
echo.

cd /d %~dp0

call scripts\windows\start_control_panel.bat
