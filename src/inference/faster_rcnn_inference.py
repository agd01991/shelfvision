from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .prediction import DetectionPrediction, ImagePrediction
from .yolo_inference import IMAGE_EXTS, prediction_summary


DEFAULT_DETECTRON_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def _build_predictor(
    model_path: str | Path,
    config_name: str = DEFAULT_DETECTRON_CONFIG,
    score_threshold: float = 0.25,
    num_classes: int = 1,
    device: Optional[str] = None,
    min_size_test: int = 640,
    max_size_test: int = 640,
) -> Any:
    """Создаёт Detectron2 DefaultPredictor для Faster R-CNN.

    Detectron2 импортируется внутри функции, чтобы весь проект не падал,
    если библиотека не установлена на машине пользователя.
    """

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найдены веса Faster R-CNN: {model_path}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_threshold)
    cfg.INPUT.MIN_SIZE_TEST = int(min_size_test)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size_test)

    if device:
        # Detectron2 обычно ожидает "cuda" или "cpu".
        cfg.MODEL.DEVICE = "cuda" if str(device) not in {"cpu", "CPU"} else "cpu"

    return DefaultPredictor(cfg)


def _instances_to_detections(instances: Any, label: str = "product") -> List[DetectionPrediction]:
    detections: List[DetectionPrediction] = []
    instances = instances.to("cpu")

    if not instances.has("pred_boxes"):
        return detections

    boxes = instances.pred_boxes.tensor.numpy().tolist()
    scores = instances.scores.numpy().tolist() if instances.has("scores") else [0.0] * len(boxes)
    classes = instances.pred_classes.numpy().tolist() if instances.has("pred_classes") else [0] * len(boxes)

    for box, score, class_id in zip(boxes, scores, classes):
        detections.append(
            DetectionPrediction(
                box=[float(v) for v in box],
                score=float(score),
                label=label if int(class_id) == 0 else f"class_{int(class_id)}",
                class_id=int(class_id),
                mask=None,
            )
        )

    return detections


def predict_faster_rcnn_image(
    model_path: str | Path,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "Faster R-CNN",
    config_name: str = DEFAULT_DETECTRON_CONFIG,
    num_classes: int = 1,
) -> ImagePrediction:
    """Запускает Faster R-CNN на одном изображении и возвращает общий формат."""

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Не найдено изображение: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    predictor = _build_predictor(
        model_path=model_path,
        config_name=config_name,
        score_threshold=conf,
        num_classes=num_classes,
        device=device,
        min_size_test=imgsz,
        max_size_test=imgsz,
    )

    start = time.perf_counter()
    outputs = predictor(image)
    inference_time = time.perf_counter() - start

    instances = outputs.get("instances")
    detections = _instances_to_detections(instances) if instances is not None else []

    height, width = image.shape[:2]
    return ImagePrediction(
        image_path=str(image_path),
        model_name=model_name,
        detections=detections,
        inference_time=inference_time,
        image_width=width,
        image_height=height,
        metadata={
            "weights": str(model_path),
            "conf": conf,
            "imgsz": imgsz,
            "adapter": "faster_rcnn_inference",
            "config_name": config_name,
            "num_classes": num_classes,
        },
    )


def predict_faster_rcnn_folder(
    model_path: str | Path,
    images_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "Faster R-CNN",
    config_name: str = DEFAULT_DETECTRON_CONFIG,
    num_classes: int = 1,
) -> List[ImagePrediction]:
    """Пакетный Faster R-CNN-инференс по папке изображений.

    Predictor создаётся один раз на всю папку, чтобы не перезагружать веса
    на каждом изображении.
    """

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с изображениями: {images_dir}")

    image_paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    predictor = _build_predictor(
        model_path=model_path,
        config_name=config_name,
        score_threshold=conf,
        num_classes=num_classes,
        device=device,
        min_size_test=imgsz,
        max_size_test=imgsz,
    )

    predictions: List[ImagePrediction] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        start = time.perf_counter()
        outputs = predictor(image)
        inference_time = time.perf_counter() - start

        instances = outputs.get("instances")
        detections = _instances_to_detections(instances) if instances is not None else []
        height, width = image.shape[:2]
        predictions.append(
            ImagePrediction(
                image_path=str(image_path),
                model_name=model_name,
                detections=detections,
                inference_time=inference_time,
                image_width=width,
                image_height=height,
                metadata={
                    "weights": str(model_path),
                    "conf": conf,
                    "imgsz": imgsz,
                    "adapter": "faster_rcnn_inference",
                    "config_name": config_name,
                    "num_classes": num_classes,
                },
            )
        )

    return predictions


def faster_rcnn_summary(prediction: ImagePrediction) -> Dict[str, Any]:
    """Краткая аналитика Faster R-CNN-предсказания."""

    return prediction_summary(prediction)
