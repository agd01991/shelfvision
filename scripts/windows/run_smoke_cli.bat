@echo off
setlocal

echo ========================================
echo ShelfVision: smoke CLI checks
echo ========================================

cd /d %~dp0\..\..
python scripts\smoke_cli.py

pause
