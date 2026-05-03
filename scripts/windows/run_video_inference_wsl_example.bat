@echo off
setlocal

echo ========================================
echo ShelfVision: video inference through WSL .venv_wsl
echo ========================================

cd /d %~dp0\..\..

set WSL_VENV=.venv_wsl
set WEIGHTS=models/yolo/best.pt
set VIDEO=data/video/test.mp4
set OUT_DIR=results/video/yolo_wsl

python scripts\wsl_runtime.py --venv-dir "%WSL_VENV%" run_video_inference.py --model yolo --weights "%WEIGHTS%" --video "%VIDEO%" --out-dir "%OUT_DIR%" --conf 0.25 --imgsz 640 --frame-skip 3 --sample-frames 8

pause
