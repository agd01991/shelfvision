# Программный комплекс анализа полочных сцен

Программный комплекс предназначен для подготовки данных, локализации товарных объектов на изображениях полок, формирования вырезанных фрагментов, построения демонстрационной SKU-галереи, визуального SKU-сопоставления, ручной проверки результатов и формирования отчётных файлов.

Ранее в отдельных рабочих материалах использовалось техническое название `ShelfVision`. Оно не является предметом научной новизны и не используется как заявка на уникальный программный продукт.

## Основные возможности

- подготовка COCO-, SKU-110K- и D2S-данных;
- проверка BBox и разбиение на train/val/test;
- преобразование COCO segmentation в YOLO-Seg;
- обучение YOLO и YOLO-Seg;
- инференс YOLO, YOLO-Seg, RT-DETR и Faster R-CNN;
- отдельное объединение YOLO и RT-DETR через WBF;
- полный gallery/query-контур фотоидентификации;
- извлечение фрагментов по BBox или маске;
- автоматическое формирование демонстрационной SKU-галереи;
- визуальное сопоставление по HSV-гистограмме и усреднённому ORB-дескриптору;
- расчёт top-k кандидатов и margin между двумя лучшими различными SKU;
- статусы `matched`, `matched_uncertain`, `unknown`;
- ручная проверка назначений без изменения исходного CSV;
- отдельная корректировка SKU-галереи через merge/split;
- история событий и контрольные точки;
- выбор отдельных SKU для анализа;
- экспорт CSV, JSON, Markdown, изображений и ZIP-архива.

## Границы текущей реализации

- итоговый полный gallery/query-контур рассчитан на одну выбранную модельную ветку за запуск;
- финальный профиль использует YOLO;
- YOLO-Seg, RT-DETR и Faster R-CNN доступны как отдельные ветки инференса, но требуют самостоятельного сравнительного запуска;
- WBF доступен в отдельном сценарии инференса и не встроен в полный gallery/query-конвейер;
- автоматически создаваемые `sku_demo_*` являются визуальными группами, а не реальными товарными артикулами;
- `assigned_rate` показывает долю объектов с назначенным кандидатом и не является top-1 accuracy без эталонной SKU-разметки;
- контрольная точка истории сохраняет конфигурацию и путь к каталогу результатов, но не выполняет полный файловый rollback;
- промышленная проверка планограмм и интеграция с товарным справочником в основной сценарий не входят.

## Финальный пользовательский интерфейс

Основная точка входа:

```bash
.venv_wsl/bin/python -m streamlit run scripts/final_demo_history_app.py
```

В WSL/Linux можно использовать запускатель:

```bash
bash scripts/run_defense_demo_app.sh
```

В Windows:

```bat
scripts\windows\run_defense_demo_app.bat
```

Windows-запускатель сначала пытается использовать `.venv_wsl`, затем локальную `.venv`, затем Python из `PATH`.

Интерфейс содержит вкладки:

```text
Старт
Обзор
Параметры
Фрагменты
Идентификация
Ручная проверка
История
До/после
Выбор SKU
Экспорт
FAQ
```

Интерфейс читает результаты уже выполненного эксперимента. Обучение и полный вычислительный конвейер запускаются отдельными скриптами.

## Итоговый профиль

Параметры демонстрационного запуска находятся в:

```text
config/vkr_final.yaml
```

Ключевые значения:

| Параметр | Значение |
|---|---:|
| Модель | YOLO |
| gallery_count | 160 |
| query_count | 140 |
| max_sku | 200 |
| confidence детектора | 0,25 |
| imgsz | 640 |
| threshold τ | 0,65 |
| ambiguity margin δ | 0,03 |
| top-k | 5 |
| dedup_threshold | 0,82 |
| max_refs_per_sku | 15 |
| min_score | 0,35 |
| min_width / min_height | 20 / 20 |
| padding | 0,05 |
| seed | 42 |

Путь `images_dir` необходимо сверить с фактическим каталогом данных перед запуском. Источник конкретного эксперимента подтверждается файлами:

```text
00_manifest/all_images.csv
00_manifest/run_environment.json
00_manifest/split_params.json
```

Проверка согласованности:

```bash
.venv_wsl/bin/python scripts/verify_experiment_source.py \
  --experiment-dir /mnt/d/1Diplom/shelfvision_results/full_photo_identification_vkr_final \
  --config config/vkr_final.yaml \
  --strict
```

## Полный gallery/query-конвейер

Пример запуска:

```bash
.venv_wsl/bin/python run_full_photo_identification_pipeline.py \
  --model yolo \
  --weights models/yolo/best.pt \
  --images-dir /mnt/d/1Diplom/data/raw/d2s_full/images \
  --out-dir /mnt/d/1Diplom/shelfvision_results/full_photo_identification_vkr_final \
  --gallery-dir /mnt/d/1Diplom/sku_gallery_full_vkr_final \
  --gallery-csv /mnt/d/1Diplom/sku_gallery_full_vkr_final/gallery.csv \
  --gallery-count 160 \
  --query-count 140 \
  --max-sku 200 \
  --conf 0.25 \
  --imgsz 640 \
  --threshold 0.65 \
  --ambiguity-margin 0.03 \
  --top-k 5 \
  --dedup-threshold 0.82 \
  --max-refs-per-sku 15 \
  --min-score 0.35 \
  --min-width 20 \
  --min-height 20 \
  --padding 0.05 \
  --shuffle \
  --seed 42 \
  --enable-uncertain-status \
  --resume \
  --skip-existing
```

Замените `--images-dir` фактическим каталогом выбранного набора данных.

## Основные выходные файлы

```text
00_manifest/all_images.csv
00_manifest/gallery_images.csv
00_manifest/query_images.csv
00_manifest/split_params.json
00_manifest/run_environment.json
01_gallery_inference/predictions.json
01_gallery_inference/summary.csv
02_demo_gallery/demo_sku_gallery_summary.json
02_demo_gallery/demo_sku_gallery_items.csv
03_query_inference/predictions.json
03_query_inference/summary.csv
04_identification/crops_manifest.csv
04_identification/identification_results.csv
04_identification/identification_metrics.csv
05_reports/full_experiment_summary.json
05_reports/full_experiment_summary.md
05_reports/threshold_analysis.csv
06_manual_identification/manual_identification_edits.csv
06_manual_identification/identification_results_corrected.csv
history/events.csv
selected_sku_demo/selected_sku_report.md
export/data_source_check.json
export/demo_smoke_report.json
export/demo_artifacts.zip
```

## Smoke-проверка

```bash
.venv_wsl/bin/python scripts/defense_demo_smoke_check.py \
  --experiment-dir /mnt/d/1Diplom/shelfvision_results/full_photo_identification_vkr_final \
  --strict
```

Отчёты сохраняются в:

```text
export/demo_smoke_report.json
export/demo_smoke_report.md
```

## Тесты

```bash
.venv_wsl/bin/python -m unittest tests/test_demo_core.py -v
```

Тесты проверяют:

- назначение статусов;
- margin между различными SKU;
- размер визуального вектора;
- применение последней ручной правки;
- сохранение событий и контрольных точек;
- формирование минимального ZIP-архива.

## Фиксация окружения

```bash
python3 -m venv .venv_wsl
.venv_wsl/bin/python -m pip install --upgrade pip
.venv_wsl/bin/python -m pip install -r requirements.txt
.venv_wsl/bin/python -m pip freeze > requirements-wsl-freeze.txt
```

## Дополнительная документация

```text
docs/DEFENSE_FAQ.md
docs/SIMILAR_PROJECTS.md
docs/REPRODUCIBILITY.md
docs/DEMO_SCRIPT_5_MIN.md
data/README.md
```
