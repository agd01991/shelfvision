from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .prediction import ImagePrediction
from .yolo_inference import predict_yolo_folder, predict_yolo_image, prediction_summary


def predict_rtdetr_image(
    model_path: str | Path,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "RT-DETR-L",
    **predict_kwargs: Any,
) -> ImagePrediction:
    """Запускает RT-DETR/RT-DETR-L на одном изображении.

    В проекте RT-DETR используется через Ultralytics API, поэтому технически
    он обрабатывается тем же способом, что и YOLO. Отдельный адаптер нужен,
    чтобы в отчётах, CLI и будущей WBF-сборке модель была явно отделена от YOLO.
    """

    prediction = predict_yolo_image(
        model_path=model_path,
        image_path=image_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        model_name=model_name,
        **predict_kwargs,
    )
    prediction.metadata = {
        **prediction.metadata,
        "adapter": "rtdetr_inference",
        "note": "RT-DETR is executed through Ultralytics predict API",
    }
    return prediction


def predict_rtdetr_folder(
    model_path: str | Path,
    images_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    model_name: str = "RT-DETR-L",
) -> List[ImagePrediction]:
    """Пакетный RT-DETR-инференс по папке изображений."""

    predictions = predict_yolo_folder(
        model_path=model_path,
        images_dir=images_dir,
        conf=conf,
        imgsz=imgsz,
        device=device,
        model_name=model_name,
    )
    for prediction in predictions:
        prediction.metadata = {
            **prediction.metadata,
            "adapter": "rtdetr_inference",
            "note": "RT-DETR is executed through Ultralytics predict API",
        }
    return predictions


def rtdetr_summary(prediction: ImagePrediction) -> Dict[str, Any]:
    """Краткая аналитика RT-DETR-предсказания."""

    return prediction_summary(prediction)
