# Воспроизведение результатов ВКР

Этот раздел описывает минимальный воспроизводимый сценарий запуска программного комплекса ShelfVision перед защитой ВКР.

## 1. Запуск панели управления

На Windows можно запустить панель управления через bat-файл:

```bat
scripts\windows\start_control_panel.bat
```

Или вручную из корня проекта:

```bash
streamlit run scripts/control_panel_wsl.py
```

Для запуска через WSL используется виртуальное окружение `.venv_wsl`.

## 2. Проверка окружения

В панели управления нужно выполнить проверку окружения и зависимостей. Без интерфейса можно использовать команды:

```bash
python -m compileall scripts
python scripts/validate_run_outputs.py --run-dir results/demo_defense
```

## 3. Демонстрационный запуск

Короткий сценарий для защиты запускает полный контур на небольшом числе изображений:

```bash
python scripts/run_demo_shelfvision.py \
  --images-dir data/raw/sku110k_small/images \
  --weights /mnt/d/1Diplom/runs/yolo_night/E03_imgsz_640/weights/best.pt \
  --model yolo \
  --limit 10 \
  --out-dir results/demo_defense \
  --conf 0.25 \
  --imgsz 640 \
  --threshold 0.65 \
  --top-k 5
```

После запуска в папке результата должны появиться предсказания модели, вырезанные фрагменты товаров, демонстрационная SKU-галерея, результаты SKU-сопоставления, визуализации, паспорт запуска и отчет проверки результата.

## 4. Проверка результата

Папку результата можно проверить отдельно:

```bash
python scripts/validate_run_outputs.py --run-dir results/demo_defense
```

Проверка создает:

```text
validation_report.md
validation_summary.json
```

## 5. Паспорт запуска

Паспорт запуска можно сформировать отдельно:

```bash
python scripts/run_metadata.py \
  --run-dir results/demo_defense \
  --model yolo \
  --weights /mnt/d/1Diplom/runs/yolo_night/E03_imgsz_640/weights/best.pt \
  --images-dir data/raw/sku110k_small/images \
  --conf 0.25 \
  --imgsz 640 \
  --threshold 0.65 \
  --top-k 5
```

Скрипт сохраняет:

```text
run_config.yaml
run_manifest.json
environment.txt
```

## 6. Отчет по спорным случаям

Для анализа объектов со статусом `matched_uncertain` используется команда:

```bash
python scripts/build_uncertain_report.py \
  --results-csv results/demo_defense/04_identification/identification_results.csv \
  --out-dir results/demo_defense/uncertain_report
```

На выходе создаются:

```text
matched_uncertain_top.csv
matched_uncertain_report.md
```

## 7. Экспорт материалов для защиты

Для подготовки папки со скриншотами, таблицами и ключевыми метриками используется команда:

```bash
python scripts/export_presentation_assets.py \
  --run-dir results/demo_defense \
  --out-dir results/demo_defense/presentation_assets \
  --limit 12
```

На выходе создается папка:

```text
presentation_assets/
  summary_for_presentation.md
  key_metrics.csv
  status_distribution.csv
  visualized_examples/
```

## 8. Экран итогового отчета запуска

Для красивого скриншота с карточками метрик можно открыть отдельную страницу:

```bash
streamlit run scripts/run_summary_panel.py
```

В поле `Папка результата` указывается путь к папке запуска, например:

```text
results/demo_defense
```

Экран показывает:

```text
Обработано изображений
Найдено объектов
Вырезанных фрагментов
SKU в галерее
Эталонов в галерее
matched
matched_uncertain
unknown
Доля с кандидатом
Средняя оценка сходства
Средний margin
```

Этот экран удобно использовать в главе 4 ВКР и в презентации как итоговую демонстрацию работы программного комплекса.

## Основные выходные файлы

- `predictions.json` - предсказания модели по изображениям.
- `summary.csv` - сводка по изображениям, числу объектов и уверенности модели.
- `crops_manifest.csv` - реестр вырезанных фрагментов товаров.
- `gallery.csv` - описание демонстрационной SKU-галереи.
- `identification_results.csv` - результаты SKU-сопоставления найденных объектов.
- `matched_uncertain_candidates.csv` - спорные случаи сопоставления, если они были найдены.
- `validation_report.md` - отчет проверки выходной папки.
- `run_manifest.json` - паспорт запуска с параметрами и сводными показателями.
- `environment.txt` - сведения о Python, платформе и ключевых пакетах.
- `04_identification/visualized/` - итоговые изображения с найденными и идентифицированными товарами.

## Что показывать на защите

Для демонстрации практического результата лучше использовать следующие материалы:

1. Главную панель ShelfVision.
2. Настройки короткого или полного запуска.
3. Итоговую визуализацию из `04_identification/visualized/`.
4. Экран `ShelfVision: итоговый отчёт запуска`.
5. `identification_results.csv` с колонками статусов.
6. `validation_report.md` как подтверждение корректности выходной папки.
7. `run_manifest.json` как паспорт воспроизводимости.
8. `matched_uncertain_report.md` как пример выделения спорных случаев.
