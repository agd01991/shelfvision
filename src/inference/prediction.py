from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


BBox = List[float]
Polygon = List[List[float]]


@dataclass
class DetectionPrediction:
    """Один найденный объект на изображении.

    Формат специально сделан общим, чтобы результаты YOLO, RT-DETR,
    Faster R-CNN и WBF можно было сравнивать и визуализировать одинаково.
    """

    box: BBox
    score: float
    label: str = "product"
    class_id: int = 0
    mask: Optional[Polygon] = None
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImagePrediction:
    """Предсказания модели для одного изображения."""

    image_path: str
    model_name: str
    detections: List[DetectionPrediction] = field(default_factory=list)
    inference_time: float = 0.0
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def objects_count(self) -> int:
        return len(self.detections)

    @property
    def average_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(item.score for item in self.detections) / len(self.detections)

    def filter_by_confidence(self, threshold: float) -> "ImagePrediction":
        return ImagePrediction(
            image_path=self.image_path,
            model_name=self.model_name,
            detections=[item for item in self.detections if item.score >= threshold],
            inference_time=self.inference_time,
            image_width=self.image_width,
            image_height=self.image_height,
            metadata={**self.metadata, "confidence_threshold": threshold},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "model_name": self.model_name,
            "boxes": [item.box for item in self.detections],
            "scores": [item.score for item in self.detections],
            "labels": [item.label for item in self.detections],
            "class_ids": [item.class_id for item in self.detections],
            "masks": [item.mask for item in self.detections],
            "track_ids": [item.track_id for item in self.detections],
            "objects_count": self.objects_count,
            "average_confidence": self.average_confidence,
            "inference_time": self.inference_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "metadata": self.metadata,
        }


def save_prediction_json(prediction: ImagePrediction, output_path: str | Path) -> Path:
    """Сохраняет предсказание в JSON для отчётов и последующего сравнения."""

    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prediction.to_dict(), f, ensure_ascii=False, indent=2)
    return output_path


def save_predictions_json(predictions: Sequence[ImagePrediction], output_path: str | Path) -> Path:
    """Сохраняет набор предсказаний для пакетной обработки."""

    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([item.to_dict() for item in predictions], f, ensure_ascii=False, indent=2)
    return output_path
