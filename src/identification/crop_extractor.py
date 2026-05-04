from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from .common import load_predictions, resolve_image_path, safe_stem, save_json


@dataclass
class CropRecord:
    image_path: str
    image_name: str
    object_id: int
    crop_path: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    class_id: int
    source_type: str
    has_mask: bool


def _clip_box(box: Sequence[float], width: int, height: int, padding_ratio: float = 0.05) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _polygon_to_mask(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon or len(polygon) < 3:
        return mask
    points = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    if points.shape[0] < 3:
        return mask
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _crop_with_mask(image: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int], background: int = 255) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    bg = np.full_like(crop, background)
    return np.where(crop_mask[:, :, None] > 0, crop, bg)


def extract_crops_from_prediction(
    prediction: Dict[str, Any],
    images_dir: str | Path | None,
    crops_dir: str | Path,
    use_masks: bool = True,
    padding_ratio: float = 0.05,
) -> List[CropRecord]:
    image_path = resolve_image_path(str(prediction.get("image_path", "")), images_dir=images_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    height, width = image.shape[:2]
    boxes = prediction.get("boxes", []) or []
    scores = prediction.get("scores", []) or []
    labels = prediction.get("labels", []) or []
    class_ids = prediction.get("class_ids", []) or []
    masks = prediction.get("masks", []) or []

    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    records: List[CropRecord] = []
    image_stem = safe_stem(image_path)

    for idx, box in enumerate(boxes, start=1):
        if not box or len(box) != 4:
            continue
        clipped = _clip_box(box, width=width, height=height, padding_ratio=padding_ratio)
        mask_polygon = masks[idx - 1] if idx - 1 < len(masks) else None
        has_mask = bool(mask_polygon and len(mask_polygon) >= 3)

        if use_masks and has_mask:
            full_mask = _polygon_to_mask(mask_polygon, width=width, height=height)
            crop = _crop_with_mask(image, full_mask, clipped)
            source_type = "mask"
        else:
            x1, y1, x2, y2 = clipped
            crop = image[y1:y2, x1:x2].copy()
            source_type = "bbox"

        if crop.size == 0:
            continue

        crop_path = crops_dir / f"{image_stem}_obj_{idx:04d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        x1, y1, x2, y2 = clipped
        records.append(
            CropRecord(
                image_path=str(image_path),
                image_name=image_path.name,
                object_id=idx,
                crop_path=str(crop_path),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(scores[idx - 1]) if idx - 1 < len(scores) else 0.0,
                label=str(labels[idx - 1]) if idx - 1 < len(labels) else "product",
                class_id=int(class_ids[idx - 1]) if idx - 1 < len(class_ids) else 0,
                source_type=source_type,
                has_mask=has_mask,
            )
        )

    return records


def extract_crops_from_predictions_file(
    predictions_json: str | Path,
    images_dir: str | Path | None,
    out_dir: str | Path,
    use_masks: bool = True,
    padding_ratio: float = 0.05,
) -> List[CropRecord]:
    predictions = load_predictions(predictions_json)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    all_records: List[CropRecord] = []

    for prediction in predictions:
        all_records.extend(
            extract_crops_from_prediction(
                prediction=prediction,
                images_dir=images_dir,
                crops_dir=crops_dir,
                use_masks=use_masks,
                padding_ratio=padding_ratio,
            )
        )

    rows = [asdict(item) for item in all_records]
    pd.DataFrame(rows).to_csv(out_dir / "crops_manifest.csv", index=False)
    save_json(rows, out_dir / "crops_manifest.json")
    return all_records
