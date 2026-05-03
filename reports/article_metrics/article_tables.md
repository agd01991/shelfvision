# Таблицы для статей

Сгенерировано скриптом `scripts/export_article_metrics.py`.


## Статья 2. Clean-сравнение моделей

| model             | paradigm          |   AP50-95 |    AP50 |    AP75 |   AP_small |   AP_medium |   AP_large |   ms_per_image |
|:------------------|:------------------|----------:|--------:|--------:|-----------:|------------:|-----------:|---------------:|
| YOLOv8s           | one-stage         |   0.53835 | 0.88806 | 0.59693 |    0.09465 |     0.31529 |    0.59224 |        37.8841 |
| RT-DETR-L         | transformer-based |   0.53879 | 0.89088 | 0.58845 |    0.06    |     0.32058 |    0.59364 |        52.2822 |
| Faster R-CNN (D2) | two-stage         |   0.47307 | 0.85756 | 0.47615 |    0.02211 |     0.28145 |    0.51922 |        88.4393 |
| WBF(YOLO+RTDETR)  | ensemble          |   0.51891 | 0.87535 | 0.55422 |    0.08914 |     0.29182 |    0.58284 |       nan      |



## Текущая robustness-таблица с Δ

| status   | model                 | dataset_mode           | corruption   |   param |   AP50-95 |     AP50 |     AP75 |    AR100 | source_file                                                                                                     |   delta_AP50-95 |   delta_AP50 |   delta_AP75 |   delta_AR100 |
|:---------|:----------------------|:-----------------------|:-------------|--------:|----------:|---------:|---------:|---------:|:----------------------------------------------------------------------------------------------------------------|----------------:|-------------:|-------------:|--------------:|
| ok       | current_base_detector | current_or_unspecified | blur         |     2   |  0.525757 | 0.872695 | 0.572883 | 0.625411 | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_blur_2p0.json      |        0.01239  |     0.015511 |     0.022057 |      0.008971 |
| ok       | current_base_detector | current_or_unspecified | clean        |     0   |  0.538146 | 0.888206 | 0.59494  | 0.634382 | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_clean_0p0.json     |        0        |     0        |     0        |      0        |
| ok       | current_base_detector | current_or_unspecified | dark         |     0.6 |  0.536312 | 0.887636 | 0.589586 | 0.632594 | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_dark_0p6.json      |        0.001835 |     0.00057  |     0.005354 |      0.001787 |
| ok       | current_base_detector | current_or_unspecified | downscale    |     0.5 |  0.537121 | 0.887844 | 0.59367  | 0.633896 | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_downscale_0p5.json |        0.001026 |     0.000361 |     0.00127  |      0.000485 |
| ok       | current_base_detector | current_or_unspecified | jpeg         |    50   |  0.538339 | 0.887409 | 0.594908 | 0.634437 | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_jpeg_50p0.json     |       -0.000193 |     0.000797 |     3.2e-05  |     -5.5e-05  |
| ok       | current_base_detector | current_or_unspecified | noise        |    10   |  0.53179  | 0.875255 | 0.587869 | 0.62657  | /mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/artifacts/dir5_robustness/metrics_noise_10p0.json    |        0.006356 |     0.012951 |     0.007071 |      0.007812 |



## Чек-лист недостающих данных

| article   | item                                                  | status   | what_to_do                                                                                                                                     |
|:----------|:------------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| 1         | Характеристики non-tiled/tiled датасетов              | ok       | Проверь dataset_summary.csv; если counts пустые, открой passport.json/annotations.json вручную.                                                |
| 1         | Парное сравнение tiled vs non-tiled на clean/degraded | missing  | Нужен файл reports/article_eval/article1_tiling_robustness.csv с колонками model,dataset_mode,corruption,AP50-95,AP50,AP75,AR100,ms_per_image. |
| 1/2       | Текущая robustness-таблица по деградациям             | ok       | Используй current_robustness_with_delta.csv как временную таблицу; она не заменяет tiled/non-tiled и 3-model robustness.                       |
| 2         | Clean-сравнение YOLO / RT-DETR / Faster R-CNN         | ok       | Используй article2_clean_models.csv.                                                                                                           |
| 2         | Robustness YOLO / RT-DETR / Faster R-CNN              | missing  | Нужен файл reports/article_eval/article2_model_robustness.csv с колонками model,corruption,AP50-95,AP50,AP75,AR100,delta_AP50-95.              |
| 2         | AR100 в clean-сравнении моделей                       | missing  | Если AR100 нужен в статье 2, дополни DIR1_metrics_csv или отдельный article2_model_robustness.csv clean-строками.                              |

