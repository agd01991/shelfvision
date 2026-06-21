# Демонстрационный интерфейс программного комплекса анализа полочных сцен

Ранее в рабочих материалах проекта использовалось техническое название `ShelfVision`. Название не является предметом научной новизны и не используется как заявка на уникальный программный продукт. В рамках ВКР проект рассматривается как программный комплекс для анализа изображений полочных сцен, локализации товарных объектов и экспериментального SKU-сопоставления.

Проект не является форком и не является новой версией сторонних открытых репозиториев с похожим названием. Совпадение названия объясняется общей предметной областью: `shelf`, `vision`, `retail shelf analysis`. Отличие данной реализации заключается в полном контуре обработки: локализация объектов, извлечение товарных фрагментов, автоматическое формирование демонстрационной SKU-галереи, расчет визуальных признаков HSV + ORB, косинусное сопоставление, top-k кандидаты, margin-анализ, статусы `matched` / `matched_uncertain` / `unknown` и формирование отчетных артефактов.

## Что входит в проект

- подготовка датасета в COCO/YOLO-подобных форматах;
- конвертация COCO segmentation в YOLO-Seg labels;
- обучение и сравнение моделей;
- инференс YOLO, YOLO-Seg, RT-DETR, Faster R-CNN и WBF через единый формат предсказаний;
- расчет bbox-метрик и визуализация ошибок модели;
- расчет mask-метрик для YOLO-Seg при наличии масочной разметки;
- полный контур gallery/query для фото-идентификации;
- извлечение вырезанных фрагментов товаров;
- автоматическое формирование демонстрационной SKU-галереи;
- SKU-сопоставление по HSV + ORB признакам и косинусной мере сходства;
- аудит похожих SKU и проверка смешанных SKU;
- ручная корректировка SKU-галереи;
- ручная проверка конкретных результатов идентификации;
- отчетные таблицы CSV, JSON, MD и визуализации;
- Streamlit-интерфейс для демонстрации защиты.

## Границы экспериментальной проверки

В рамках ВКР итоговая количественная проверка полного визуального контура выполнена на выбранной модельной ветке YOLO. YOLO-Seg, RT-DETR, Faster R-CNN и WBF предусмотрены архитектурно и частично реализованы через единый формат предсказаний, однако их полноценное сравнение требует отдельной серии экспериментов и не заявляется как завершённый результат текущей работы.

`matched_rate` / `assigned_rate` не являются top-1 accuracy реального SKU-распознавания без эталонной SKU-разметки каждого проверяемого объекта. Это доля объектов, которым демонстрационный контур назначил кандидата из автоматически сформированной SKU-галереи.

## Итоговый профиль ВКР

Итоговые параметры для воспроизводимой демонстрации вынесены в файл:

```text
config/vkr_final.yaml
```

Ключевые параметры:

| Параметр | Значение |
|---|---:|
| Модель итогового контура | YOLO |
| gallery_count | 160 |
| query_count | 140 |
| max_sku | 200 |
| confidence детектора | 0,25 |
| imgsz | 640 |
| threshold τ | 0,65 |
| ambiguity_margin δ | 0,03 |
| top-k | 5 |
| dedup_threshold | 0,82 |
| max_refs_per_sku | 15 |
| min_score | 0,35 |
| min_width / min_height | 20 / 20 |
| padding | 0,05 |
| seed | 42 |

## Демонстрационный интерфейс для защиты

Главный интерфейс запускается через WSL/локальное окружение:

```bash
streamlit run scripts/control_panel_wsl_app.py
```

На Windows можно запустить:

```bat
scripts\windows\run_defense_demo_app.bat
```

В WSL/Linux можно запустить:

```bash
bash scripts/run_defense_demo_app.sh
```

В интерфейсе добавлен раздел **«Демо защиты»**, который показывает полный сценарий:

1. выбор набора изображений;
2. разделение на gallery/query;
3. применение модели локализации;
4. просмотр BBox/масок и вырезанных фрагментов;
5. формирование демонстрационной SKU-галереи;
6. SKU-сопоставление по визуальным признакам;
7. просмотр top-k кандидатов, similarity и margin;
8. ручная проверка идентификации;
9. корректировка SKU-галереи;
10. сравнение результата до и после ручной проверки;
11. экспорт отчетных файлов.

Вкладка **«0. Сценарий защиты»** содержит чек-лист готовности и пошаговый сценарий демонстрации на 5 минут.

Интерфейс предназначен для демонстрации исследовательского прототипа и не заявляется как промышленная система контроля планограмм.

## Smoke-проверка перед защитой

Быстрая проверка наличия проектных файлов, импортов и основных артефактов эксперимента выполняется так:

```bash
.venv_wsl/bin/python scripts/defense_demo_smoke_check.py \
  --experiment-dir D:/1Diplom/shelfvision_results/full_photo_identification_vkr_final
```

На Windows можно запустить:

```bat
scripts\windows\run_defense_smoke_check.bat --experiment-dir D:/1Diplom/shelfvision_results/full_photo_identification_vkr_final
```

Проверка формирует:

```text
defense_export/defense_demo_smoke_report.json
defense_export/defense_demo_smoke_report.md
```

## Установка зависимостей через WSL

```bash
python3 -m venv .venv_wsl
.venv_wsl/bin/python -m pip install --upgrade pip
.venv_wsl/bin/python -m pip install -r requirements.txt
```

Для точной фиксации локального окружения защиты рекомендуется дополнительно сформировать freeze-файл:

```bash
.venv_wsl/bin/python -m pip freeze > requirements-wsl-freeze.txt
```

## Быстрый запуск полного контура

Пример запуска полного контура с итоговыми параметрами:

```bash
.venv_wsl/bin/python run_full_photo_identification_pipeline.py \
  --model yolo \
  --weights models/yolo/best.pt \
  --images-dir D:/1Diplom/data/raw/d2s_full/images \
  --out-dir D:/1Diplom/shelfvision_results/full_photo_identification_vkr_final \
  --gallery-dir D:/1Diplom/sku_gallery_full_vkr_final \
  --gallery-csv D:/1Diplom/sku_gallery_full_vkr_final/gallery.csv \
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

## Основные выходные файлы

```text
00_manifest/all_images.csv
00_manifest/gallery_images.csv
00_manifest/query_images.csv
01_gallery_inference/predictions.json
01_gallery_inference/summary.csv
02_demo_gallery/demo_sku_gallery_summary.json
02_demo_gallery/demo_sku_gallery_items.csv
03_query_inference/predictions.json
03_query_inference/summary.csv
04_identification/crops_manifest.csv
04_identification/identification_results.csv
04_identification/identification_report.md
05_reports/full_experiment_summary.json
05_reports/full_experiment_summary.md
05_reports/threshold_analysis.csv
06_manual_identification/manual_identification_edits.csv
06_manual_identification/manual_reference_suggestions.csv
06_manual_identification/identification_results_corrected.csv
selected_sku_demo/selected_sku_report.md
defense_export/vkr_defense_artifacts.zip
```

## Документы для защиты

- `docs/DEFENSE_FAQ.md` — ответы на вопросы комиссии;
- `docs/SIMILAR_PROJECTS.md` — отличие от открытых проектов с похожим названием;
- `docs/REPRODUCIBILITY.md` — воспроизводимость итогового запуска;
- `docs/DEMO_SCRIPT_5_MIN.md` — сценарий демонстрации на защите за 5 минут;
- `data/README.md` — описание локального размещения данных.
