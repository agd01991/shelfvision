from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2

from .metrics import BBox, load_gt_coco, load_gt_yolo, load_predictions, match_boxes


GREEN = (0, 200, 0)      # true positive
RED = (0, 0, 255)        # false positive
YELLOW = (0, 220, 255)   # false negative
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _draw_label(image, text: str, x: int, y: int, color=BLACK) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, h + baseline + 4)
    cv2.rectangle(image, (x, y - h - baseline - 4), (x + w + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 1), font, scale, WHITE, thickness, cv2.LINE_AA)


def _draw_box(image, box: Sequence[float], color, label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    _draw_label(image, label, x1, y1, color=BLACK)


def _prediction_name(prediction: Dict[str, Any]) -> str:
    return Path(str(prediction.get("image_path", ""))).name


def visualize_prediction_errors(
    prediction: Dict[str, Any],
    gt_boxes: List[BBox],
    output_path: str | Path,
    iou_threshold: float = 0.5,
) -> Path:
    """Сохраняет изображение с цветовой разметкой ошибок.

    Зелёный — TP, красный — FP, жёлтый — FN.
    """

    image_path = Path(str(prediction.get("image_path", "")))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    pred_boxes = prediction.get("boxes", []) or []
    scores = prediction.get("scores", []) or [0.0] * len(pred_boxes)
    match = match_boxes(gt_boxes, pred_boxes, scores, iou_threshold=iou_threshold)

    matched_gt = {gt_idx for gt_idx, _, _ in match.matched_pairs}
    matched_pred = {pred_idx for _, pred_idx, _ in match.matched_pairs}

    for gt_idx, pred_idx, iou in match.matched_pairs:
        score = scores[pred_idx] if pred_idx < len(scores) else 0.0
        _draw_box(image, pred_boxes[pred_idx], GREEN, f"TP {score:.2f} IoU={iou:.2f}")

    for pred_idx in match.false_pred_indices:
        score = scores[pred_idx] if pred_idx < len(scores) else 0.0
        _draw_box(image, pred_boxes[pred_idx], RED, f"FP {score:.2f}")

    for gt_idx in match.missed_gt_indices:
        _draw_box(image, gt_boxes[gt_idx], YELLOW, "FN", thickness=2)

    footer = f"TP={match.true_positive} FP={match.false_positive} FN={match.false_negative} IoU_thr={iou_threshold}"
    _draw_label(image, footer, 8, image.shape[0] - 10, color=BLACK)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def visualize_errors_from_files(
    predictions_json: str | Path,
    out_dir: str | Path,
    gt_coco_json: str | Path | None = None,
    gt_yolo_labels_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    iou_threshold: float = 0.5,
    limit: int = 0,
) -> List[Path]:
    predictions = load_predictions(predictions_json)

    if gt_coco_json:
        gt_by_image = load_gt_coco(gt_coco_json)
    elif gt_yolo_labels_dir and images_dir:
        gt_by_image = load_gt_yolo(gt_yolo_labels_dir, images_dir)
    else:
        raise ValueError("Укажите gt_coco_json или пару gt_yolo_labels_dir + images_dir")

    out_dir = Path(out_dir)
    saved: List[Path] = []
    for prediction in predictions[: limit or None]:
        image_name = _prediction_name(prediction)
        gt_boxes = gt_by_image.get(image_name, [])
        output_path = out_dir / f"{Path(image_name).stem}__errors.jpg"
        saved.append(visualize_prediction_errors(prediction, gt_boxes, output_path, iou_threshold=iou_threshold))
    return saved
