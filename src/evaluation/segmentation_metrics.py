from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


Mask = np.ndarray
Polygon = List[List[float]]
IOU_THRESHOLDS_50_95 = [round(x / 100, 2) for x in range(50, 100, 5)]


@dataclass
class MaskObject:
    mask: Mask
    score: float = 1.0
    class_id: int = 0


@dataclass
class MaskMatchResult:
    true_positive: int
    false_positive: int
    false_negative: int
    matched_pairs: List[Tuple[int, int, float]]
    missed_gt_indices: List[int]
    false_pred_indices: List[int]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("predictions JSON должен быть объектом или списком объектов")


def _prediction_name(prediction: Dict[str, Any]) -> str:
    return Path(str(prediction.get("image_path", ""))).name


def _empty_mask(width: int, height: int) -> Mask:
    return np.zeros((height, width), dtype=np.uint8)


def polygon_to_mask(polygon: Sequence[Sequence[float]], width: int, height: int) -> Mask:
    """Преобразует один polygon [[x, y], ...] в бинарную маску."""

    mask = _empty_mask(width, height)
    if not polygon or len(polygon) < 3:
        return mask

    points = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    if points.shape[0] < 3:
        return mask

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [points], 1)
    return mask


def flat_polygon_to_mask(poly: Sequence[float], width: int, height: int) -> Mask:
    """Преобразует COCO polygon [x1, y1, x2, y2, ...] в бинарную маску."""

    if not poly or len(poly) < 6 or len(poly) % 2 != 0:
        return _empty_mask(width, height)

    points = [[float(poly[i]), float(poly[i + 1])] for i in range(0, len(poly), 2)]
    return polygon_to_mask(points, width=width, height=height)


def coco_segmentation_to_mask(segmentation: Any, width: int, height: int) -> Mask:
    """Преобразует COCO segmentation polygon/RLE в бинарную маску объекта."""

    mask = _empty_mask(width, height)

    # COCO polygon as flat list
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], (int, float)):
        return flat_polygon_to_mask(segmentation, width=width, height=height)

    # COCO polygon as list of polygons
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        for poly in segmentation:
            mask = np.maximum(mask, flat_polygon_to_mask(poly, width=width, height=height))
        return mask

    # COCO RLE
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return (decoded > 0).astype(np.uint8)
        except Exception:
            return mask

    return mask


def mask_iou(mask_a: Mask, mask_b: Mask) -> float:
    """IoU для двух бинарных масок одинакового размера."""

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Размеры масок не совпадают: {mask_a.shape} и {mask_b.shape}")

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def load_gt_coco_masks(path: str | Path) -> Tuple[Dict[str, List[MaskObject]], Dict[str, Tuple[int, int]]]:
    """Загружает GT-маски из COCO JSON.

    Возвращает:
    - словарь basename изображения -> список MaskObject;
    - словарь basename изображения -> (width, height).
    """

    data = load_json(path)
    images = data.get("images", [])
    anns = data.get("annotations", [])

    image_by_id = {int(item["id"]): item for item in images}
    sizes_by_image: Dict[str, Tuple[int, int]] = {}
    gt_by_image: Dict[str, List[MaskObject]] = {}

    for image in images:
        name = Path(str(image.get("file_name", ""))).name
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if name and width > 0 and height > 0:
            sizes_by_image[name] = (width, height)
            gt_by_image[name] = []

    for ann in anns:
        image = image_by_id.get(int(ann.get("image_id", -1)))
        if not image:
            continue
        name = Path(str(image.get("file_name", ""))).name
        width, height = sizes_by_image.get(name, (0, 0))
        if width <= 0 or height <= 0:
            continue

        segmentation = ann.get("segmentation")
        mask = coco_segmentation_to_mask(segmentation, width=width, height=height)
        if mask.sum() == 0:
            continue

        gt_by_image.setdefault(name, []).append(
            MaskObject(mask=mask, score=1.0, class_id=int(ann.get("category_id", 0)))
        )

    return gt_by_image, sizes_by_image


def load_pred_masks(
    predictions: Sequence[Dict[str, Any]],
    sizes_by_image: Dict[str, Tuple[int, int]],
) -> Dict[str, List[MaskObject]]:
    """Загружает предсказанные polygon masks из общего JSON ShelfVision."""

    pred_by_image: Dict[str, List[MaskObject]] = {}

    for prediction in predictions:
        name = _prediction_name(prediction)
        width, height = sizes_by_image.get(
            name,
            (
                int(prediction.get("image_width") or 0),
                int(prediction.get("image_height") or 0),
            ),
        )
        if width <= 0 or height <= 0:
            pred_by_image[name] = []
            continue

        masks = prediction.get("masks", []) or []
        scores = prediction.get("scores", []) or [0.0] * len(masks)
        class_ids = prediction.get("class_ids", []) or [0] * len(masks)

        objects: List[MaskObject] = []
        for idx, polygon in enumerate(masks):
            if not polygon:
                continue
            mask = polygon_to_mask(polygon, width=width, height=height)
            if mask.sum() == 0:
                continue
            objects.append(
                MaskObject(
                    mask=mask,
                    score=float(scores[idx]) if idx < len(scores) else 0.0,
                    class_id=int(class_ids[idx]) if idx < len(class_ids) else 0,
                )
            )
        pred_by_image[name] = objects

    return pred_by_image


def match_masks(
    gt_objects: Sequence[MaskObject],
    pred_objects: Sequence[MaskObject],
    iou_threshold: float,
) -> MaskMatchResult:
    """Жадное сопоставление GT и предсказанных масок по mask IoU."""

    order = sorted(range(len(pred_objects)), key=lambda idx: pred_objects[idx].score, reverse=True)
    used_gt: set[int] = set()
    matched_pairs: List[Tuple[int, int, float]] = []
    false_pred_indices: List[int] = []

    for pred_idx in order:
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_idx in used_gt:
                continue
            iou = mask_iou(gt_obj.mask, pred_objects[pred_idx].mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            matched_pairs.append((best_gt_idx, pred_idx, best_iou))
        else:
            false_pred_indices.append(pred_idx)

    missed_gt_indices = [idx for idx in range(len(gt_objects)) if idx not in used_gt]
    return MaskMatchResult(
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


def evaluate_segmentation_predictions(
    predictions: Sequence[Dict[str, Any]],
    gt_by_image: Dict[str, List[MaskObject]],
    sizes_by_image: Dict[str, Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    pred_by_image = load_pred_masks(predictions, sizes_by_image)
    image_names = sorted(set(gt_by_image) | set(pred_by_image))

    rows: List[Dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    matched_ious: List[float] = []

    for image_name in image_names:
        gt_objects = gt_by_image.get(image_name, [])
        pred_objects = pred_by_image.get(image_name, [])
        match = match_masks(gt_objects, pred_objects, iou_threshold=iou_threshold)

        total_tp += match.true_positive
        total_fp += match.false_positive
        total_fn += match.false_negative
        matched_ious.extend([pair[2] for pair in match.matched_pairs])

        per_image = precision_recall_f1(match.true_positive, match.false_positive, match.false_negative)
        rows.append(
            {
                "image_name": image_name,
                "gt_masks": len(gt_objects),
                "pred_masks": len(pred_objects),
                "tp": match.true_positive,
                "fp": match.false_positive,
                "fn": match.false_negative,
                "mean_mask_iou": sum(pair[2] for pair in match.matched_pairs) / len(match.matched_pairs)
                if match.matched_pairs
                else 0.0,
                **per_image,
            }
        )

    metrics_50 = precision_recall_f1(total_tp, total_fp, total_fn)

    ap_by_threshold: Dict[str, float] = {}
    for threshold in IOU_THRESHOLDS_50_95:
        tp = fp = fn = 0
        for image_name in image_names:
            match = match_masks(gt_by_image.get(image_name, []), pred_by_image.get(image_name, []), iou_threshold=threshold)
            tp += match.true_positive
            fp += match.false_positive
            fn += match.false_negative
        ap_by_threshold[f"APmask{int(threshold * 100)}"] = precision_recall_f1(tp, fp, fn)["precision"]

    apmask50 = ap_by_threshold.get("APmask50", 0.0)
    apmask75 = ap_by_threshold.get("APmask75", 0.0)
    apmask50_95 = sum(ap_by_threshold.values()) / len(ap_by_threshold) if ap_by_threshold else 0.0

    return {
        "summary": {
            "images_count": len(image_names),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "mask_precision": metrics_50["precision"],
            "mask_recall": metrics_50["recall"],
            "mask_f1": metrics_50["f1"],
            "mean_mask_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
            "APmask50": apmask50,
            "APmask75": apmask75,
            "APmask50-95": apmask50_95,
            "iou_threshold": iou_threshold,
        },
        "ap_by_threshold": ap_by_threshold,
        "per_image": rows,
    }


def evaluate_segmentation_predictions_file(
    predictions_json: str | Path,
    gt_coco_json: str | Path,
    out_dir: str | Path = "results/evaluation/yolo_seg_masks",
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    predictions = load_predictions(predictions_json)
    gt_by_image, sizes_by_image = load_gt_coco_masks(gt_coco_json)
    result = evaluate_segmentation_predictions(
        predictions=predictions,
        gt_by_image=gt_by_image,
        sizes_by_image=sizes_by_image,
        iou_threshold=iou_threshold,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "segmentation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([result["summary"]]).to_csv(out_dir / "segmentation_metrics_summary.csv", index=False)
    pd.DataFrame(result["per_image"]).to_csv(out_dir / "segmentation_metrics_per_image.csv", index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result["ap_by_threshold"].items()]
    ).to_csv(out_dir / "mask_ap_by_threshold.csv", index=False)

    return result
