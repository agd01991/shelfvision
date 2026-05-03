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
- автоматическую рекомендацию лучшего pipeline по набору метрик.

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

## Новая структура инференса и оценки

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
└── recommend_model.py        # автоматический выбор лучшего pipeline

scripts/
├── interface_app.py          # интерфейс таблиц и графиков экспериментов
└── inference_app.py          # интерактивный инференс по изображению

run_inference.py              # CLI-запуск инференса
run_evaluation.py             # CLI-запуск оценки качества
run_recommendation.py         # CLI-рекомендация лучшего pipeline
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

1. Добавить сравнение нескольких моделей в одном отчёте.
2. Добавить отдельный блок анализа плотности товаров.
