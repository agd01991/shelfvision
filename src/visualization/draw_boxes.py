from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.inference.prediction import ImagePrediction


Color = Tuple[int, int, int]


DEFAULT_BOX_COLOR: Color = (0, 200, 0)
DEFAULT_MASK_COLOR: Color = (0, 160, 255)
TEXT_COLOR: Color = (255, 255, 255)
TEXT_BG_COLOR: Color = (0, 0, 0)


def _draw_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, h + baseline + 4)
    cv2.rectangle(image, (x, y - h - baseline - 4), (x + w + 6, y + 2), TEXT_BG_COLOR, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 1), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_mask(image: np.ndarray, polygon: list[list[float]], color: Color, alpha: float) -> None:
    if not polygon or len(polygon) < 3:
        return

    points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


def draw_prediction(
    prediction: ImagePrediction,
    output_path: str | Path | None = None,
    show_masks: bool = True,
    box_color: Color = DEFAULT_BOX_COLOR,
    mask_color: Color = DEFAULT_MASK_COLOR,
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """Отрисовывает bbox/masks на изображении по общему формату ShelfVision.

    Возвращает изображение в формате BGR. Если указан output_path — сохраняет файл.
    """

    image_path = Path(prediction.image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    for detection in prediction.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.box]

        if show_masks and detection.mask:
            _draw_mask(image, detection.mask, mask_color, mask_alpha)

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        label = f"{detection.label} {detection.score:.2f}"
        _draw_label(image, label, x1, y1)

    footer = (
        f"{prediction.model_name}: objects={prediction.objects_count}, "
        f"avg_conf={prediction.average_confidence:.3f}, time={prediction.inference_time:.3f}s"
    )
    _draw_label(image, footer, 8, image.shape[0] - 10)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    return image
