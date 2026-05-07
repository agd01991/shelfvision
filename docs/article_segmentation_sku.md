# Эксперимент для статьи: инстанс-сегментация как подготовка к SKU-идентификации

Тема статьи:

> Инстанс-сегментация товарных объектов на полочных изображениях как этап подготовки к SKU-идентификации

## Цель эксперимента

Проверить, что масочное выделение товарного объекта позволяет получить более чистый crop для последующей SKU-идентификации по сравнению с обычным вырезанием объекта по bounding box.

Основная исследовательская цепочка:

```text
полочное изображение → YOLO-Seg → mask-метрики → bbox crop / mask crop → оценка чистоты crop → подготовка к SKU matching
```

## Уже реализованные компоненты

В проекте уже есть:

- обучение YOLO-Seg: `scripts/train_yolo_seg.py`;
- инференс YOLO-Seg: `run_inference.py --model yolo_seg`;
- оценка mask-метрик: `run_segmentation_evaluation.py`;
- сравнение bbox crop и mask crop: `scripts/compare_bbox_mask_crops.py`;
- сбор таблиц для статьи: `scripts/export_segmentation_article_metrics.py`;
- сбор рисунков для статьи: `scripts/make_segmentation_article_figures.py`;
- опциональный модуль SKU matching: `run_identification.py`.

## 1. Сбор базовых таблиц

```bash
python3 scripts/export_segmentation_article_metrics.py
```

Результаты:

```text
reports/article_segmentation/article_segmentation_tables.md
reports/article_segmentation/article_segmentation_metrics.xlsx
reports/article_segmentation/missing_checklist.csv
```

## 2. YOLO-Seg inference на test split

```bash
python3 run_inference.py \
  --model yolo_seg \
  --weights runs/d2s_seg/d2s_small_yolov8s_seg_img6402/weights/best.pt \
  --images-dir data/yolo_cache/d2s_small_seg/images/test \
  --out-dir results/article_segmentation/yolo_seg_inference \
  --conf 0.25 \
  --imgsz 640 \
  --device 0
```

Результаты:

```text
results/article_segmentation/yolo_seg_inference/predictions.json
results/article_segmentation/yolo_seg_inference/summary.csv
results/article_segmentation/yolo_seg_inference/visualized/
```

## 3. Расчёт mask-метрик

```bash
python3 run_segmentation_evaluation.py \
  --predictions results/article_segmentation/yolo_seg_inference/predictions.json \
  --gt-coco data/coco_splits/d2s_small/test_fix.json \
  --out-dir results/article_segmentation/yolo_seg_masks \
  --iou 0.5
```

Результаты:

```text
results/article_segmentation/yolo_seg_masks/segmentation_metrics_summary.csv
results/article_segmentation/yolo_seg_masks/segmentation_metrics_per_image.csv
results/article_segmentation/yolo_seg_masks/mask_ap_by_threshold.csv
results/article_segmentation/yolo_seg_masks/segmentation_metrics.json
```

## 4. Сравнение bbox crop и mask crop

```bash
python3 scripts/compare_bbox_mask_crops.py \
  --predictions results/article_segmentation/yolo_seg_inference/predictions.json \
  --images-dir data/yolo_cache/d2s_small_seg/images/test \
  --out-dir results/article_segmentation/crop_comparison \
  --min-confidence 0.25 \
  --min-mask-area 50 \
  --padding 0.05 \
  --background 255 \
  --examples-limit 30
```

Результаты:

```text
results/article_segmentation/crop_comparison/crop_quality_per_object.csv
results/article_segmentation/crop_comparison/crop_quality_summary.csv
results/article_segmentation/crop_comparison/crop_quality_manifest.json
results/article_segmentation/crop_comparison/examples/
```

Главная таблица для статьи:

```text
results/article_segmentation/crop_comparison/crop_quality_summary.csv
```

Интерпретация:

- `bbox_crop` показывает долю объекта и лишней области внутри прямоугольного crop;
- `mask_crop_white_bg` показывает crop, где фон вне маски заменяется нейтральным белым фоном;
- `avg_removed_visual_background_ratio` показывает среднюю долю области внутри bbox, удаляемую при использовании маски.

## 5. Сбор рисунков для статьи

```bash
python3 scripts/make_segmentation_article_figures.py \
  --visualized-dir results/article_segmentation/yolo_seg_inference/visualized \
  --crop-examples-dir results/article_segmentation/crop_comparison/examples \
  --out-dir reports/article_segmentation/figures \
  --limit 5
```

Результаты:

```text
reports/article_segmentation/figures/yolo_seg_predictions/
reports/article_segmentation/figures/bbox_vs_mask_crop/
reports/article_segmentation/figures/figures_manifest.md
```

## 6. Повторная сборка таблиц после всех запусков

```bash
python3 scripts/export_segmentation_article_metrics.py
cat reports/article_segmentation/missing_checklist.csv
```

Минимально достаточное состояние для статьи:

```text
dataset_stats,ok
yolo_seg_training_metrics,ok
mask_evaluation_metrics,ok
bbox_vs_mask_crop_quality,ok
sku_bbox_preparation,optional_missing
sku_mask_preparation,optional_missing
```

## 7. Опциональный SKU matching

SKU matching запускается только при наличии реальной SKU-галереи.

Поддерживаемые форматы:

1. CSV с колонками:

```text
sku_id,sku_name,category,image_path
```

2. Папка вида:

```text
data/sku_gallery/<sku_id>/*.jpg
```

Если есть реальная галерея, можно запустить два режима:

```bash
python3 run_identification.py \
  --predictions results/article_segmentation/yolo_seg_inference/predictions.json \
  --images-dir data/yolo_cache/d2s_small_seg/images/test \
  --gallery-dir data/sku_gallery \
  --out-dir results/article_segmentation/sku_bbox \
  --threshold 0.65 \
  --top-k 3 \
  --padding 0.05
```

```bash
python3 run_identification.py \
  --predictions results/article_segmentation/yolo_seg_inference/predictions.json \
  --images-dir data/yolo_cache/d2s_small_seg/images/test \
  --gallery-dir data/sku_gallery \
  --out-dir results/article_segmentation/sku_mask \
  --threshold 0.65 \
  --top-k 3 \
  --padding 0.05 \
  --use-masks
```

Если реальной SKU-галереи нет, этот этап не является обязательным. В статье в этом случае формулируется не полноценная SKU-классификация, а подготовка очищенных crop-изображений для последующей SKU-идентификации.
