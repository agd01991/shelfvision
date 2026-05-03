@echo off
setlocal

echo ========================================
echo ShelfVision: full pipeline example
echo ========================================

cd /d %~dp0\..\..

set IMAGES_DIR=data\test\images
set LABELS_DIR=data\test\labels
set YOLO_WEIGHTS=models\yolo\best.pt
set RTDETR_WEIGHTS=models\rtdetr\best.pt
set OUT_DIR=results\full_pipeline

echo Edit this file if paths are different:
echo IMAGES_DIR=%IMAGES_DIR%
echo LABELS_DIR=%LABELS_DIR%
echo YOLO_WEIGHTS=%YOLO_WEIGHTS%
echo RTDETR_WEIGHTS=%RTDETR_WEIGHTS%
echo OUT_DIR=%OUT_DIR%
echo.

python run_full_pipeline.py ^
  --images-dir "%IMAGES_DIR%" ^
  --gt-yolo-labels "%LABELS_DIR%" ^
  --yolo-weights "%YOLO_WEIGHTS%" ^
  --rtdetr-weights "%RTDETR_WEIGHTS%" ^
  --models yolo rtdetr wbf ^
  --out-dir "%OUT_DIR%" ^
  --conf 0.25 ^
  --imgsz 640

pause
