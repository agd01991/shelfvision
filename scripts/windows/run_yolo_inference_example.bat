@echo off
setlocal

echo ========================================
echo ShelfVision: YOLO inference example
echo ========================================

cd /d %~dp0\..\..

set WEIGHTS=models\yolo\best.pt
set IMAGE=data\test\image_001.jpg
set OUT_DIR=results\inference\yolo_example

echo Edit this file if paths are different:
echo WEIGHTS=%WEIGHTS%
echo IMAGE=%IMAGE%
echo OUT_DIR=%OUT_DIR%
echo.

python run_inference.py --model yolo --weights "%WEIGHTS%" --image "%IMAGE%" --out-dir "%OUT_DIR%" --conf 0.25 --imgsz 640

pause
