from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prediction import DetectionPrediction, ImagePrediction


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _extract_masks(result: Any) -> List[Optional[List[List[float]]]]:
    """Возвращает polygon masks из Ultralytics-результата, если они есть."""

    boxes_count = 0
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "xyxy", None) is not None:
        boxes_count = len(result.boxes.xyxy)

    empty = [None for _ in range(boxes_count)]
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return empty

    polygons: List[Optional[List[List[float]]]] = []
    for polygon in masks.xy:
        try:
            polygons.append([[float(x), float(y)] for x, y in polygon.tolist()])
        except Exception:
            polygons.append(None)

    if len(polygons) < boxes_count:
        polygons.extend([None] * (boxes_count - len(polygons)))
    return polygons[:boxes_count]


def predict_yolo_image(
    model_path: str | Path,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "YOLO",
    **predict_kwargs: Any,
) -> ImagePrediction:
    """Запускает YOLO/YOLO-Seg на одном изображении и возвращает общий формат.

    Пример:
        prediction = predict_yolo_image("models/yolo/best.pt", "data/test/img.jpg")
    """

    from ultralytics import YOLO

    model_path = Path(model_path)
    image_path = Path(image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Не найдены веса модели: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Не найдено изображение: {image_path}")

    model = YOLO(str(model_path))

    start = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        **predict_kwargs,
    )
    inference_time = time.perf_counter() - start

    if not results:
        return ImagePrediction(
            image_path=str(image_path),
            model_name=model_name,
            inference_time=inference_time,
            metadata={"weights": str(model_path), "conf": conf, "imgsz": imgsz},
        )

    result = results[0]
    names = getattr(result, "names", getattr(model, "names", {}))
    detections: List[DetectionPrediction] = []

    boxes = getattr(result, "boxes", None)
    masks = _extract_masks(result)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy)
        classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else [0] * len(xyxy)

        for idx, box in enumerate(xyxy):
            class_id = int(classes[idx])
            detections.append(
                DetectionPrediction(
                    box=[float(v) for v in box.tolist()],
                    score=float(scores[idx]),
                    label=_class_name(names, class_id),
                    class_id=class_id,
                    mask=masks[idx] if idx < len(masks) else None,
                )
            )

    image_height = None
    image_width = None
    if getattr(result, "orig_shape", None):
        image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])

    return ImagePrediction(
        image_path=str(image_path),
        model_name=model_name,
        detections=detections,
        inference_time=inference_time,
        image_width=image_width,
        image_height=image_height,
        metadata={"weights": str(model_path), "conf": conf, "imgsz": imgsz},
    )


def predict_yolo_folder(
    model_path: str | Path,
    images_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "YOLO",
) -> List[ImagePrediction]:
    """Пакетный инференс по папке с изображениями."""

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с изображениями: {images_dir}")

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    return [
        predict_yolo_image(
            model_path=model_path,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            model_name=model_name,
        )
        for image_path in images
    ]


def prediction_summary(prediction: ImagePrediction) -> Dict[str, Any]:
    """Краткая аналитика для интерфейса и CSV/JSON отчёта."""

    scores = [item.score for item in prediction.detections]
    return {
        "image_path": prediction.image_path,
        "model_name": prediction.model_name,
        "objects_count": prediction.objects_count,
        "average_confidence": prediction.average_confidence,
        "min_confidence": min(scores) if scores else 0.0,
        "max_confidence": max(scores) if scores else 0.0,
        "inference_time": prediction.inference_time,
    }
