from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


BBox = List[float]
IOU_THRESHOLDS_50_95 = [round(x / 100, 2) for x in range(50, 100, 5)]


@dataclass
class MatchResult:
    true_positive: int
    false_positive: int
    false_negative: int
    matched_pairs: List[Tuple[int, int, float]]
    missed_gt_indices: List[int]
    false_pred_indices: List[int]


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU для bbox формата [x1, y1, x2, y2]."""

    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def xywh_to_xyxy(box: Sequence[float]) -> BBox:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def yolo_to_xyxy(values: Sequence[float], width: int, height: int) -> BBox:
    xc, yc, bw, bh = [float(v) for v in values]
    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height
    return [x1, y1, x2, y2]


def match_boxes(gt_boxes: Sequence[BBox], pred_boxes: Sequence[BBox], scores: Sequence[float], iou_threshold: float) -> MatchResult:
    """Жадное сопоставление предсказаний и GT по IoU."""

    order = sorted(range(len(pred_boxes)), key=lambda idx: scores[idx] if idx < len(scores) else 0.0, reverse=True)
    used_gt: set[int] = set()
    matched_pairs: List[Tuple[int, int, float]] = []
    false_pred_indices: List[int] = []

    for pred_idx in order:
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            iou = bbox_iou(gt_box, pred_boxes[pred_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            matched_pairs.append((best_gt_idx, pred_idx, best_iou))
        else:
            false_pred_indices.append(pred_idx)

    missed_gt_indices = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    return MatchResult(
        true_positive=len(matched_pairs),
        false_positive=len(false_pred_indices),
        false_negative=len(missed_gt_indices),
        matched_pairs=matched_pairs,
        missed_gt_indices=missed_gt_indices,
        false_pred_indices=false_pred_indices,
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("predictions JSON должен быть объектом или списком объектов")


def load_gt_coco(path: str | Path) -> Dict[str, List[BBox]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    image_by_id = {item["id"]: item for item in data.get("images", [])}
    gt_by_image: Dict[str, List[BBox]] = {item["file_name"]: [] for item in data.get("images", [])}

    for ann in data.get("annotations", []):
        image = image_by_id.get(ann.get("image_id"))
        if not image:
            continue
        gt_by_image.setdefault(image["file_name"], []).append(xywh_to_xyxy(ann["bbox"]))
    return gt_by_image


def load_gt_yolo(labels_dir: str | Path, images_dir: str | Path) -> Dict[str, List[BBox]]:
    """Загружает YOLO txt-разметку. Размеры изображений читаются через OpenCV."""

    import cv2

    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    gt_by_image: Dict[str, List[BBox]] = {}

    for image_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rel = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel).with_suffix(".txt")
        boxes: List[BBox] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                boxes.append(yolo_to_xyxy([float(v) for v in parts[1:5]], width=width, height=height))
        gt_by_image[image_path.name] = boxes
    return gt_by_image


def _prediction_name(prediction: Dict[str, Any]) -> str:
    return Path(str(prediction.get("image_path", ""))).name


def evaluate_predictions(
    predictions: Sequence[Dict[str, Any]],
    gt_by_image: Dict[str, List[BBox]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0

    for prediction in predictions:
        image_name = _prediction_name(prediction)
        gt_boxes = gt_by_image.get(image_name, [])
        pred_boxes = prediction.get("boxes", []) or []
        scores = prediction.get("scores", []) or [0.0] * len(pred_boxes)

        match = match_boxes(gt_boxes, pred_boxes, scores, iou_threshold=iou_threshold)
        total_tp += match.true_positive
        total_fp += match.false_positive
        total_fn += match.false_negative

        per_image = precision_recall_f1(match.true_positive, match.false_positive, match.false_negative)
        rows.append(
            {
                "image_name": image_name,
                "model_name": prediction.get("model_name", "unknown"),
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": match.true_positive,
                "fp": match.false_positive,
                "fn": match.false_negative,
                **per_image,
            }
        )

    metrics_50 = precision_recall_f1(total_tp, total_fp, total_fn)

    ap_by_threshold: Dict[str, float] = {}
    for threshold in IOU_THRESHOLDS_50_95:
        tp = fp = fn = 0
        for prediction in predictions:
            image_name = _prediction_name(prediction)
            gt_boxes = gt_by_image.get(image_name, [])
            pred_boxes = prediction.get("boxes", []) or []
            scores = prediction.get("scores", []) or [0.0] * len(pred_boxes)
            match = match_boxes(gt_boxes, pred_boxes, scores, iou_threshold=threshold)
            tp += match.true_positive
            fp += match.false_positive
            fn += match.false_negative
        ap_by_threshold[f"AP{int(threshold * 100)}"] = precision_recall_f1(tp, fp, fn)["precision"]

    ap50 = ap_by_threshold.get("AP50", 0.0)
    ap50_95 = sum(ap_by_threshold.values()) / len(ap_by_threshold) if ap_by_threshold else 0.0

    return {
        "summary": {
            "images_count": len(predictions),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": metrics_50["precision"],
            "recall": metrics_50["recall"],
            "f1": metrics_50["f1"],
            "AP50": ap50,
            "AP50-95": ap50_95,
            "iou_threshold": iou_threshold,
        },
        "ap_by_threshold": ap_by_threshold,
        "per_image": rows,
    }


def evaluate_predictions_file(
    predictions_json: str | Path,
    gt_coco_json: str | Path | None = None,
    gt_yolo_labels_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    out_dir: str | Path = "results/evaluation",
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    predictions = load_predictions(predictions_json)

    if gt_coco_json:
        gt_by_image = load_gt_coco(gt_coco_json)
    elif gt_yolo_labels_dir and images_dir:
        gt_by_image = load_gt_yolo(gt_yolo_labels_dir, images_dir)
    else:
        raise ValueError("Укажите gt_coco_json или пару gt_yolo_labels_dir + images_dir")

    result = evaluate_predictions(predictions, gt_by_image, iou_threshold=iou_threshold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([result["summary"]]).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(result["per_image"]).to_csv(out_dir / "metrics_per_image.csv", index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result["ap_by_threshold"].items()]
    ).to_csv(out_dir / "ap_by_threshold.csv", index=False)

    return result
