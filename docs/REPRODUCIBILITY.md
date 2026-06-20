# Воспроизводимость демонстрационного контура

## Основной сценарий

1. Подготовить локальные данные по структуре, описанной в `data/README.md`.
2. Создать окружение WSL и установить зависимости.
3. Использовать итоговый профиль `config/vkr_final.yaml`.
4. Запустить полный контур фото-идентификации.
5. Проверить результаты в демонстрационном интерфейсе.

## Основные параметры итогового профиля

| Параметр | Значение |
|---|---:|
| Модель | YOLO |
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
| shuffle seed | 42 |

## Выходные файлы

После полного запуска формируются:

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
```

## Ограничения воспроизводимости

Абсолютное время обучения и применения моделей зависит от версии драйверов, CUDA, PyTorch, Ultralytics, видеокарты и режима запуска. Поэтому время интерпретируется как сравнительное внутри одной программно-аппаратной среды.

Доля объектов с назначенным кандидатом не является top-1 accuracy реального SKU-распознавания без эталонной SKU-разметки каждого проверяемого объекта.
