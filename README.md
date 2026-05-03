# ShelfVision

Проект для ВКР: система анализа изображений товарных полок с использованием моделей детекции и сегментации.

Сейчас проект включает:
- подготовку датасета в COCO/YOLO-подобных форматах;
- обучение и сравнение моделей;
- отчётные таблицы и графики;
- Streamlit-интерфейс экспериментов;
- единый слой инференса для подключения YOLO, RT-DETR, Faster R-CNN и WBF.

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

## Новая структура инференса

```text
src/inference/
├── prediction.py             # единый формат результата
├── yolo_inference.py         # адаптер YOLO/YOLO-Seg
├── rtdetr_inference.py       # адаптер RT-DETR-L
├── faster_rcnn_inference.py  # адаптер Faster R-CNN
└── ensemble_wbf.py           # WBF-ансамбль YOLO + RT-DETR

src/visualization/
└── draw_boxes.py             # отрисовка bbox и masks

run_inference.py              # CLI-запуск инференса
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

1. Добавить расчёт метрик и визуализацию ошибок.
2. Расширить интерфейс режимом загрузки изображения и выбора модели.
3. Добавить автоматическую рекомендацию лучшего pipeline.
