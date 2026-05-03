from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .prediction import DetectionPrediction, ImagePrediction
from .rtdetr_inference import predict_rtdetr_folder, predict_rtdetr_image
from .yolo_inference import predict_yolo_image, prediction_summary


def _normalize_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, float(x1) / width)),
        max(0.0, min(1.0, float(y1) / height)),
        max(0.0, min(1.0, float(x2) / width)),
        max(0.0, min(1.0, float(y2) / height)),
    ]


def _denormalize_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        float(x1) * width,
        float(y1) * height,
        float(x2) * width,
        float(y2) * height,
    ]


def combine_predictions_wbf(
    predictions: Sequence[ImagePrediction],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    weights: Optional[Sequence[float]] = None,
    model_name: str = "WBF(YOLO + RT-DETR)",
) -> ImagePrediction:
    """Объединяет несколько ImagePrediction через Weighted Boxes Fusion.

    На вход подаются предсказания разных моделей для одного изображения.
    На выходе получается новое ImagePrediction в общем формате ShelfVision.
    """

    if not predictions:
        raise ValueError("Для WBF нужно передать хотя бы одно предсказание")

    base = predictions[0]
    width = base.image_width
    height = base.image_height
    if width is None or height is None:
        raise ValueError("Для WBF нужны image_width и image_height в предсказании")

    boxes_list: List[List[List[float]]] = []
    scores_list: List[List[float]] = []
    labels_list: List[List[int]] = []

    for prediction in predictions:
        boxes: List[List[float]] = []
        scores: List[float] = []
        labels: List[int] = []
        for detection in prediction.detections:
            boxes.append(_normalize_box(detection.box, width, height))
            scores.append(float(detection.score))
            labels.append(int(detection.class_id))
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if weights is None:
        weights = [1.0] * len(predictions)

    from ensemble_boxes import weighted_boxes_fusion

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=list(weights),
        iou_thr=float(iou_thr),
        skip_box_thr=float(skip_box_thr),
    )

    detections = [
        DetectionPrediction(
            box=_denormalize_box(box, width, height),
            score=float(score),
            label="product" if int(label) == 0 else f"class_{int(label)}",
            class_id=int(label),
            mask=None,
        )
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels)
    ]

    return ImagePrediction(
        image_path=base.image_path,
        model_name=model_name,
        detections=detections,
        inference_time=sum(item.inference_time for item in predictions),
        image_width=width,
        image_height=height,
        metadata={
            "adapter": "ensemble_wbf",
            "source_models": [item.model_name for item in predictions],
            "iou_thr": iou_thr,
            "skip_box_thr": skip_box_thr,
            "weights": list(weights),
        },
    )


def predict_wbf_image(
    yolo_model_path: str | Path,
    rtdetr_model_path: str | Path,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    yolo_weight: float = 1.0,
    rtdetr_weight: float = 1.0,
    model_name: str = "WBF(YOLO + RT-DETR)",
) -> ImagePrediction:
    """Запускает YOLO и RT-DETR на одном изображении и объединяет bbox через WBF."""

    yolo_prediction = predict_yolo_image(
        model_path=yolo_model_path,
        image_path=image_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        model_name="YOLO",
    )
    rtdetr_prediction = predict_rtdetr_image(
        model_path=rtdetr_model_path,
        image_path=image_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        model_name="RT-DETR-L",
    )

    return combine_predictions_wbf(
        [yolo_prediction, rtdetr_prediction],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        weights=[yolo_weight, rtdetr_weight],
        model_name=model_name,
    )


def predict_wbf_folder(
    yolo_model_path: str | Path,
    rtdetr_model_path: str | Path,
    images_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    yolo_weight: float = 1.0,
    rtdetr_weight: float = 1.0,
    model_name: str = "WBF(YOLO + RT-DETR)",
) -> List[ImagePrediction]:
    """Пакетный WBF-инференс по папке изображений."""

    images_dir = Path(images_dir)
    from .yolo_inference import IMAGE_EXTS

    image_paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    return [
        predict_wbf_image(
            yolo_model_path=yolo_model_path,
            rtdetr_model_path=rtdetr_model_path,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            yolo_weight=yolo_weight,
            rtdetr_weight=rtdetr_weight,
            model_name=model_name,
        )
        for image_path in image_paths
    ]


def wbf_summary(prediction: ImagePrediction) -> Dict[str, Any]:
    """Краткая аналитика WBF-предсказания."""

    return prediction_summary(prediction)
