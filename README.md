# ShelfVision

Проект для ВКР: система анализа изображений товарных полок с использованием моделей детекции и сегментации.

Сейчас проект включает:
- подготовку датасета в COCO/YOLO-подобных форматах;
- обучение и сравнение моделей;
- отчётные таблицы и графики;
- Streamlit-интерфейс экспериментов;
- Streamlit-интерфейс интерактивного инференса;
- единый слой инференса для подключения YOLO, RT-DETR, Faster R-CNN и WBF;
- расчёт bbox-метрик и визуализацию ошибок модели;
- автоматическую рекомендацию лучшего pipeline по набору метрик;
- единый отчёт сравнения нескольких моделей с таблицами и графиками;
- анализ плотности найденных товаров по зонам изображения;
- экспорт итогового мини-отчёта для презентации и защиты;
- запуск полного pipeline одной командой;
- `.bat`-файлы для удобного запуска на Windows.

## Быстрый старт

1) Установка зависимостей:
```bash
pip install -r requirements.txt
```

2) Подготовить демо-датасет (COCO bbox):
- data/raw/demo_coco/annotations.json
- data/raw/demo_coco/images/...

3) Запуск подготовки:
```bash
python scripts/prepare_dataset.py --dataset demo_coco --version v1
```

Результат:
- data/prepared/demo_coco/v1/annotations.json
- data/prepared/demo_coco/v1/splits.json
- data/prepared/demo_coco/v1/passport.json
- data/prepared/demo_coco/v1/issues.json
- data/prepared/demo_coco/v1/reports/samples_{train,val,test}/

## Запуск интерфейса экспериментов

```bash
streamlit run scripts/interface_app.py
```

Интерфейс показывает таблицы метрик, графики, визуальные примеры, устойчивость моделей и результаты YOLO-Seg.

## Запуск интерактивного инференса

```bash
streamlit run scripts/inference_app.py
```

В этом интерфейсе можно:
- загрузить изображение полки;
- выбрать модель: YOLO, RT-DETR-L, Faster R-CNN или WBF;
- настроить confidence threshold и размер изображения;
- указать пути к весам моделей;
- получить изображение с bbox/masks;
- посмотреть таблицу найденных объектов;
- сохранить JSON, CSV и визуальный результат.

## Запуск инференса YOLO на одном изображении

```bash
python run_inference.py --model yolo --weights models/yolo/best.pt --image data/test/image_001.jpg --out-dir results/inference/yolo
```

## Запуск инференса RT-DETR-L на одном изображении

```bash
python run_inference.py --model rtdetr --weights models/rtdetr/best.pt --image data/test/image_001.jpg --out-dir results/inference/rtdetr
```

## Запуск инференса Faster R-CNN на одном изображении

```bash
python run_inference.py --model frcnn --weights models/faster_rcnn/model_final.pth --image data/test/image_001.jpg --out-dir results/inference/frcnn
```

Faster R-CNN использует Detectron2. Если библиотека не установлена, её нужно поставить отдельно под свою версию PyTorch/CUDA.

## Запуск WBF-ансамбля на одном изображении

```bash
python run_inference.py --model wbf --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --image data/test/image_001.jpg --out-dir results/inference/wbf
```

WBF объединяет bbox-предсказания YOLO и RT-DETR-L. Дополнительно можно менять параметры:

```bash
python run_inference.py --model wbf --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --image data/test/image_001.jpg --out-dir results/inference/wbf --wbf-iou 0.55 --wbf-skip 0.001 --yolo-weight 1.0 --rtdetr-weight 1.0
```

После запуска будут сохранены:
- `prediction.json` — предсказания в едином формате;
- `summary.csv` — краткая аналитика;
- `visualized/` — изображение с bbox/masks.

## Пакетная обработка папки изображений

YOLO:
```bash
python run_inference.py --model yolo --weights models/yolo/best.pt --images-dir data/test --out-dir results/inference/yolo_batch
```

RT-DETR-L:
```bash
python run_inference.py --model rtdetr --weights models/rtdetr/best.pt --images-dir data/test --out-dir results/inference/rtdetr_batch
```

Faster R-CNN:
```bash
python run_inference.py --model frcnn --weights models/faster_rcnn/model_final.pth --images-dir data/test --out-dir results/inference/frcnn_batch
```

WBF:
```bash
python run_inference.py --model wbf --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --images-dir data/test --out-dir results/inference/wbf_batch
```

После запуска будут сохранены:
- `predictions.json`;
- `summary.csv`;
- папка `visualized/` с отрисованными результатами.

## Расчёт метрик

Для COCO-разметки:

```bash
python run_evaluation.py --predictions results/inference/yolo_batch/predictions.json --gt-coco data/test/annotations.json --out-dir results/evaluation/yolo
```

Для YOLO-разметки:

```bash
python run_evaluation.py --predictions results/inference/yolo_batch/predictions.json --gt-yolo-labels data/test/labels --images-dir data/test/images --out-dir results/evaluation/yolo
```

С визуализацией ошибок:

```bash
python run_evaluation.py --predictions results/inference/yolo_batch/predictions.json --gt-yolo-labels data/test/labels --images-dir data/test/images --out-dir results/evaluation/yolo --visualize-errors --limit 20
```

После запуска будут сохранены:
- `metrics.json`;
- `metrics_summary.csv`;
- `metrics_per_image.csv`;
- `ap_by_threshold.csv`;
- `errors/` с изображениями ошибок, если указан `--visualize-errors`.

Цвета ошибок:
- зелёный — правильное обнаружение TP;
- красный — ложное обнаружение FP;
- жёлтый — пропущенный объект FN.

## Автоматическая рекомендация лучшего pipeline

После расчёта метрик для нескольких моделей можно выбрать лучший pipeline:

```bash
python run_recommendation.py --metrics results/evaluation/yolo/metrics_summary.csv results/evaluation/rtdetr/metrics_summary.csv results/evaluation/frcnn/metrics_summary.csv results/evaluation/wbf/metrics_summary.csv --labels YOLO RT-DETR Faster-R-CNN WBF --out-dir results/recommendation
```

По умолчанию итоговый score считается по весам:
- AP50-95 — 0.40;
- AP50 — 0.20;
- Recall — 0.15;
- Precision — 0.15;
- F1 — 0.05;
- скорость — 0.05.

Веса можно менять через параметры:

```bash
python run_recommendation.py --metrics results/evaluation/yolo/metrics_summary.csv results/evaluation/rtdetr/metrics_summary.csv --labels YOLO RT-DETR --w-ap50-95 0.50 --w-recall 0.20 --out-dir results/recommendation
```

После запуска будут сохранены:
- `recommendation.json`;
- `recommendation_ranking.csv`;
- `recommendation.md`.

## Единый отчёт сравнения моделей

Для формирования отчёта по нескольким моделям:

```bash
python run_compare.py --metrics results/evaluation/yolo/metrics_summary.csv results/evaluation/rtdetr/metrics_summary.csv results/evaluation/frcnn/metrics_summary.csv results/evaluation/wbf/metrics_summary.csv --labels YOLO RT-DETR Faster-R-CNN WBF --out-dir results/model_comparison
```

После запуска будут сохранены:
- `model_comparison.json`;
- `model_comparison.csv`;
- `model_comparison.md`;
- `plots/` с графиками по AP50-95, AP50, precision, recall, F1 и recommendation score.

Этот отчёт удобно использовать в практической главе ВКР: он показывает рейтинг моделей, лучшую модель по каждой метрике и итоговую рекомендацию pipeline.

## Анализ плотности товаров

После инференса можно оценить, как найденные товары распределены по зонам изображения:

```bash
python run_density.py --predictions results/inference/yolo_batch/predictions.json --out-dir results/density/yolo --rows 3 --cols 3
```

Для одного изображения:

```bash
python run_density.py --predictions results/inference/yolo/prediction.json --out-dir results/density/yolo_single --rows 3 --cols 3
```

После запуска будут сохранены:
- `density_by_zone.csv` — статистика по каждой зоне каждого изображения;
- `density_summary.csv` — агрегированная статистика по зонам;
- `density_report.json` — краткий отчёт;
- `visualized/` — изображения с сеткой и тепловой заливкой плотности.

Этот блок можно описывать как аналитический модуль для ритейла: система не только находит товары, но и показывает, какие части полки заполнены сильнее.

## Итоговый мини-отчёт для презентации

После сравнения моделей и анализа плотности можно собрать короткий отчёт для презентации:

```bash
python run_mini_report.py --comparison-json results/model_comparison/model_comparison.json --comparison-csv results/model_comparison/model_comparison.csv --recommendation-json results/recommendation/recommendation.json --density-json results/density/yolo/density_report.json --density-csv results/density/yolo/density_summary.csv --images-dir results/density/yolo/visualized --out-dir results/mini_report
```

После запуска будут сохранены:
- `mini_report.md` — markdown-отчёт;
- `mini_report.html` — HTML-отчёт для просмотра в браузере;
- `mini_report_manifest.json` — список использованных входных и выходных файлов.

Мини-отчёт содержит:
- назначение системы;
- рекомендуемый pipeline;
- таблицу сравнения моделей;
- анализ плотности товаров;
- визуальные примеры;
- список пунктов, которые удобно показать на защите.

## Запуск полного pipeline одной командой

Полный pipeline запускает инференс, оценку, рекомендацию, сравнение моделей, анализ плотности и мини-отчёт:

```bash
python run_full_pipeline.py --images-dir data/test/images --gt-yolo-labels data/test/labels --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --models yolo rtdetr wbf --out-dir results/full_pipeline
```

Если используется COCO-разметка:

```bash
python run_full_pipeline.py --images-dir data/test/images --gt-coco data/test/annotations.json --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --models yolo rtdetr wbf --out-dir results/full_pipeline
```

Для добавления Faster R-CNN:

```bash
python run_full_pipeline.py --images-dir data/test/images --gt-yolo-labels data/test/labels --yolo-weights models/yolo/best.pt --rtdetr-weights models/rtdetr/best.pt --frcnn-weights models/faster_rcnn/model_final.pth --models yolo rtdetr frcnn wbf --out-dir results/full_pipeline
```

После запуска будут сформированы папки:
- `inference/` — предсказания и визуализации моделей;
- `evaluation/` — метрики и ошибки;
- `recommendation/` — выбор лучшего pipeline;
- `model_comparison/` — единый отчёт сравнения;
- `density/` — анализ плотности;
- `mini_report/` — итоговый HTML/Markdown-отчёт.

## Запуск на Windows через `.bat`

Готовые `.bat`-файлы лежат в папке:

```text
scripts/windows/
```

Доступные сценарии:

```text
run_interface.bat              — запуск интерфейса таблиц и графиков
run_inference_app.bat          — запуск интерфейса инференса
run_yolo_inference_example.bat — пример запуска YOLO на одном изображении
run_full_pipeline_example.bat  — пример запуска полного pipeline
run_mini_report_example.bat    — пример сборки мини-отчёта
```

Перед запуском example-файлов нужно открыть `.bat` и при необходимости изменить пути:

```bat
set WEIGHTS=models\yolo\best.pt
set IMAGE=data\test\image_001.jpg
set IMAGES_DIR=data\test\images
set LABELS_DIR=data\test\labels
```

## Новая структура инференса, оценки, аналитики и отчётов

```text
src/inference/
├── prediction.py             # единый формат результата
├── yolo_inference.py         # адаптер YOLO/YOLO-Seg
├── rtdetr_inference.py       # адаптер RT-DETR-L
├── faster_rcnn_inference.py  # адаптер Faster R-CNN
└── ensemble_wbf.py           # WBF-ансамбль YOLO + RT-DETR

src/visualization/
└── draw_boxes.py             # отрисовка bbox и masks

src/evaluation/
├── metrics.py                # IoU, Precision, Recall, F1, AP50, AP50-95
├── error_visualization.py    # отрисовка TP/FP/FN
├── recommend_model.py        # автоматический выбор лучшего pipeline
└── compare_models.py         # единый отчёт сравнения моделей

src/analytics/
└── density.py                # анализ плотности товаров по зонам

src/reporting/
└── mini_report.py            # итоговый мини-отчёт для презентации

scripts/
├── interface_app.py          # интерфейс таблиц и графиков экспериментов
├── inference_app.py          # интерактивный инференс по изображению
└── windows/                  # .bat-файлы для Windows

run_inference.py              # CLI-запуск инференса
run_evaluation.py             # CLI-запуск оценки качества
run_recommendation.py         # CLI-рекомендация лучшего pipeline
run_compare.py                # CLI-сравнение нескольких моделей
run_density.py                # CLI-анализ плотности товаров
run_mini_report.py            # CLI-сборка мини-отчёта
run_full_pipeline.py          # CLI-запуск полного pipeline
```

Единый формат нужен, чтобы результаты разных моделей можно было сравнивать одинаково:

```python
{
    "image_path": "data/test/image_001.jpg",
    "model_name": "YOLO",
    "boxes": [[x1, y1, x2, y2]],
    "scores": [0.91],
    "labels": ["product"],
    "masks": [],
    "objects_count": 1,
    "average_confidence": 0.91,
    "inference_time": 0.08
}
```

## Следующие этапы

1. Добавить smoke-тесты для проверки основных CLI-скриптов.
2. Провести ручную проверку pipeline на реальных весах и тестовой папке.
