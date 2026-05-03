@echo off
setlocal

echo ========================================
echo ShelfVision: full pipeline through WSL .venv_wsl
echo ========================================

cd /d %~dp0\..\..

set WSL_VENV=.venv_wsl
set IMAGES_DIR=data/test/images
set LABELS_DIR=data/test/labels
set YOLO_WEIGHTS=models/yolo/best.pt
set RTDETR_WEIGHTS=models/rtdetr/best.pt
set OUT_DIR=results/full_pipeline_wsl

python scripts\wsl_runtime.py --venv-dir "%WSL_VENV%" run_full_pipeline.py --images-dir "%IMAGES_DIR%" --gt-yolo-labels "%LABELS_DIR%" --yolo-weights "%YOLO_WEIGHTS%" --rtdetr-weights "%RTDETR_WEIGHTS%" --models yolo rtdetr wbf --out-dir "%OUT_DIR%" --conf 0.25 --imgsz 640

pause
