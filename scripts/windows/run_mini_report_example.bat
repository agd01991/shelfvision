@echo off
setlocal

echo ========================================
echo ShelfVision: mini report example
echo ========================================

cd /d %~dp0\..\..

set COMPARISON_JSON=results\model_comparison\model_comparison.json
set COMPARISON_CSV=results\model_comparison\model_comparison.csv
set RECOMMENDATION_JSON=results\recommendation\recommendation.json
set DENSITY_JSON=results\density\yolo\density_report.json
set DENSITY_CSV=results\density\yolo\density_summary.csv
set IMAGES_DIR=results\density\yolo\visualized
set OUT_DIR=results\mini_report

echo Edit this file if paths are different.
echo.

python run_mini_report.py ^
  --comparison-json "%COMPARISON_JSON%" ^
  --comparison-csv "%COMPARISON_CSV%" ^
  --recommendation-json "%RECOMMENDATION_JSON%" ^
  --density-json "%DENSITY_JSON%" ^
  --density-csv "%DENSITY_CSV%" ^
  --images-dir "%IMAGES_DIR%" ^
  --out-dir "%OUT_DIR%"

pause
