# ShelfVision (scaffold)

Минимальный каркас проекта для ВКР: подготовка датасета в единый COCO-подобный формат (bbox),
валидация разметки, групповое разбиение train/val/test, генерация отчётов с примерами.

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
