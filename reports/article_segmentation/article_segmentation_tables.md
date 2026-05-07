# Таблицы для статьи по инстанс-сегментации

Сгенерировано скриптом `scripts/export_segmentation_article_metrics.py`.


## Таблица 1. Характеристики сегментационного набора

split,status,source_file,images_count,annotations_count,segmentation_annotations_count,categories_count,avg_objects_per_image,min_objects_per_image,max_objects_per_image,avg_annotation_area,small_area_annotations_lt_32x32,medium_area_annotations_32x32_96x96,large_area_annotations_gt_96x96
train,ok,data/coco_splits/d2s_small/train_fix.json,210,629,629,60,2.9952,1,13,143507.31,0,3,626
val,ok,data/coco_splits/d2s_small/val_fix.json,45,94,94,60,2.0889,1,6,170278.89,0,1,93
test,ok,data/coco_splits/d2s_small/test_fix.json,45,124,124,60,2.7556,1,12,145141.1,0,1,123
total,ok,data/coco_splits/d2s_small/train_fix.json + data/coco_splits/d2s_small/val_fix.json + data/coco_splits/d2s_small/test_fix.json,300,847,847,60,2.8233,,,,0,5,842




## Таблица 2. Итоговые метрики обучения YOLO-Seg

run_dir,status,epoch_last,P_box,R_box,mAP50_box,mAP5095_box,mAP50_mask,mAP5095_mask,source_file
/mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision/runs/d2s_seg/d2s_small_yolov8s_seg_img6402,ok,30,0.88011,0.58831,0.75064,0.71896,0.75064,0.71763,reports/all_stats/D2S_YOLO_SEG_last.csv




## Таблица 3. Дополнительная mask-оценка

status,images_count,tp,fp,fn,mask_precision,mask_recall,mask_f1,mean_mask_iou,APmask50,APmask75,APmask50-95,iou_threshold,source_file
ok,45,112,48,12,0.7,0.9032258064516128,0.7887323943661972,0.9336184382490818,0.7,0.7,0.6481250000000001,0.5,results/article_segmentation/yolo_seg_masks/segmentation_metrics_summary.csv




## Таблица 4. Сравнение bbox crop и mask crop

status,method,objects_count,avg_crop_area_px,avg_object_area_px,avg_object_purity,avg_visual_background_ratio,avg_removed_visual_background_ratio,total_removed_background_pixels,interpretation,source_file
ok,bbox_crop,160,288930.175,150251.7062,0.549965,0.450035,0.0,0,"Обычный crop по bounding box: внутри остаются фон, соседние товары, ценники и части полки.",results/article_segmentation/crop_comparison/crop_quality_summary.csv
ok,mask_crop_white_bg,160,288930.175,150251.7062,1.0,0.0,0.450035,22188555,"Crop по mask: визуальный фон внутри bbox заменяется нейтральным белым фоном, объект остаётся основным содержимым crop.",results/article_segmentation/crop_comparison/crop_quality_summary.csv




## Таблица 5a. SKU preparation: bbox crops

status,source_file,note
missing,results/article_segmentation/sku_bbox/identification_metrics.csv,Файл не найден. Этот блок ещё нужно получить отдельным запуском.




## Таблица 5b. SKU preparation: mask crops

status,source_file,note
missing,results/article_segmentation/sku_mask/identification_metrics.csv,Файл не найден. Этот блок ещё нужно получить отдельным запуском.




## Чек-лист готовности данных

block,status,what_to_do
dataset_stats,ok,Проверить пути к COCO segmentation JSON train/val/test.
yolo_seg_training_metrics,ok,Нужен reports/all_stats/D2S_YOLO_SEG_last.csv или новый запуск обучения YOLO-Seg.
mask_evaluation_metrics,ok,Запустить run_segmentation_evaluation.py после YOLO-Seg inference.
bbox_vs_mask_crop_quality,ok,Добавить/запустить scripts/compare_bbox_mask_crops.py.
sku_bbox_preparation,optional_missing,"Запустить run_identification.py без --use-masks, если есть SKU gallery."
sku_mask_preparation,optional_missing,"Запустить run_identification.py с --use-masks, если есть SKU gallery."


