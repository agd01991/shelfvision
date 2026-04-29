@echo off
chcp 65001 >nul
echo Starting ShelfVision interface...
python -m streamlit run scripts/interface_app.py
pause
