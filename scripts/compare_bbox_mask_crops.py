from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("predictions JSON должен быть объектом или списком объектов")


def resolve_image_path(image_path: str, images_dir: str | Path | None = None) -> Path:
    p = Path(image_path)
    if p.exists():
        return p

    if images_dir:
        root = Path(images_dir)
        direct = root / p.name
        if direct.exists():
            return direct
        matches = list(root.rglob(p.name))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Не найдено изображение: {image_path}")


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


def clip_box(
    box: Sequence[float],
    width: int,
    height: int,
    padding_ratio: float = 0.05,
) -> tuple[int, int, int, int]:
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


def polygon_to_mask(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
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


def crop_with_mask(
    image: np.ndarray,
    full_mask: np.ndarray,
    box: tuple[int, int, int, int],
    background: int = 255,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = full_mask[y1:y2, x1:x2]
    bg = np.full_like(crop, background)
    return np.where(crop_mask[:, :, None] > 0, crop, bg)


def save_example_pair(
    image: np.ndarray,
    full_mask: np.ndarray,
    box: tuple[int, int, int, int],
    out_dir: Path,
    name: str,
    background: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box

    bbox_crop = image[y1:y2, x1:x2].copy()
    mask_crop = crop_with_mask(image, full_mask, box, background=background)

    bbox_path = out_dir / f"{name}_bbox.jpg"
    mask_path = out_dir / f"{name}_mask_white_bg.jpg"
    cv2.imwrite(str(bbox_path), bbox_crop)
    cv2.imwrite(str(mask_path), mask_crop)

    panel_h = max(bbox_crop.shape[0], mask_crop.shape[0])
    panel_w = bbox_crop.shape[1] + mask_crop.shape[1]
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    panel[: bbox_crop.shape[0], : bbox_crop.shape[1]] = bbox_crop
    panel[: mask_crop.shape[0], bbox_crop.shape[1] :] = mask_crop
    panel_path = out_dir / f"{name}_bbox_vs_mask.jpg"
    cv2.imwrite(str(panel_path), panel)

    return {
        "example_bbox_crop_path": str(bbox_path),
        "example_mask_crop_path": str(mask_path),
        "example_panel_path": str(panel_path),
    }


def summarize_methods(per_object: pd.DataFrame) -> pd.DataFrame:
    if per_object.empty:
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "note": "Нет объектов с валидными bbox и mask. Проверь YOLO-Seg predictions.json.",
                }
            ]
        )

    total = int(len(per_object))
    avg_mask_tightness = float(per_object["mask_tightness"].mean())
    avg_removed_background = float(per_object["removed_visual_background_ratio"].mean())
    avg_bbox_area = float(per_object["bbox_area"].mean())
    avg_mask_area = float(per_object["mask_area"].mean())
    total_removed_pixels = int(per_object["removed_background_pixels"].sum())

    return pd.DataFrame(
        [
            {
                "status": "ok",
                "method": "bbox_crop",
                "objects_count": total,
                "avg_crop_area_px": round(avg_bbox_area, 4),
                "avg_object_area_px": round(avg_mask_area, 4),
                "avg_object_purity": round(avg_mask_tightness, 6),
                "avg_visual_background_ratio": round(1.0 - avg_mask_tightness, 6),
                "avg_removed_visual_background_ratio": 0.0,
                "total_removed_background_pixels": 0,
                "interpretation": "Обычный crop по bounding box: внутри остаются фон, соседние товары, ценники и части полки.",
            },
            {
                "status": "ok",
                "method": "mask_crop_white_bg",
                "objects_count": total,
                "avg_crop_area_px": round(avg_bbox_area, 4),
                "avg_object_area_px": round(avg_mask_area, 4),
                "avg_object_purity": 1.0,
                "avg_visual_background_ratio": 0.0,
                "avg_removed_visual_background_ratio": round(avg_removed_background, 6),
                "total_removed_background_pixels": total_removed_pixels,
                "interpretation": "Crop по mask: визуальный фон внутри bbox заменяется нейтральным белым фоном, объект остаётся основным содержимым crop.",
            },
        ]
    )


def run_comparison(
    predictions_json: str | Path,
    images_dir: str | Path | None,
    out_dir: str | Path,
    min_confidence: float = 0.0,
    min_mask_area: int = 1,
    padding_ratio: float = 0.05,
    background: int = 255,
    examples_limit: int = 30,
) -> dict[str, Path]:
    predictions = load_predictions(predictions_json)
    out_dir = Path(out_dir)
    examples_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    examples_saved = 0

    for prediction in predictions:
        image_path_raw = str(prediction.get("image_path", ""))
        if not image_path_raw:
            continue
        image_path = resolve_image_path(image_path_raw, images_dir=images_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        boxes = prediction.get("boxes", []) or []
        scores = prediction.get("scores", []) or []
        labels = prediction.get("labels", []) or []
        class_ids = prediction.get("class_ids", []) or []
        masks = prediction.get("masks", []) or []
        image_stem = safe_stem(image_path)

        for object_idx, box in enumerate(boxes, start=1):
            if not box or len(box) != 4:
                continue
            score = float(scores[object_idx - 1]) if object_idx - 1 < len(scores) else 0.0
            if score < min_confidence:
                continue

            polygon = masks[object_idx - 1] if object_idx - 1 < len(masks) else None
            has_mask = bool(polygon and len(polygon) >= 3)
            if not has_mask:
                continue

            full_mask = polygon_to_mask(polygon, width=width, height=height)
            clipped = clip_box(box, width=width, height=height, padding_ratio=padding_ratio)
            x1, y1, x2, y2 = clipped
            crop_mask = full_mask[y1:y2, x1:x2]

            bbox_area = int(max(1, (x2 - x1) * (y2 - y1)))
            mask_area = int((crop_mask > 0).sum())
            if mask_area < min_mask_area:
                continue

            mask_tightness = mask_area / bbox_area if bbox_area else 0.0
            removed_background_pixels = max(0, bbox_area - mask_area)
            removed_visual_background_ratio = 1.0 - mask_tightness

            example_paths: dict[str, str] = {}
            if examples_saved < examples_limit:
                example_name = f"{image_stem}_obj_{object_idx:04d}"
                example_paths = save_example_pair(
                    image=image,
                    full_mask=full_mask,
                    box=clipped,
                    out_dir=examples_dir,
                    name=example_name,
                    background=background,
                )
                examples_saved += 1

            rows.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "object_id": object_idx,
                    "score": score,
                    "label": str(labels[object_idx - 1]) if object_idx - 1 < len(labels) else "product",
                    "class_id": int(class_ids[object_idx - 1]) if object_idx - 1 < len(class_ids) else 0,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "padding_ratio": padding_ratio,
                    "bbox_area": bbox_area,
                    "mask_area": mask_area,
                    "mask_tightness": round(mask_tightness, 8),
                    "bbox_crop_purity": round(mask_tightness, 8),
                    "bbox_visual_background_ratio": round(1.0 - mask_tightness, 8),
                    "mask_crop_purity_white_bg": 1.0,
                    "mask_crop_visual_background_ratio_white_bg": 0.0,
                    "removed_background_pixels": removed_background_pixels,
                    "removed_visual_background_ratio": round(removed_visual_background_ratio, 8),
                    **example_paths,
                }
            )

    per_object = pd.DataFrame(rows)
    summary = summarize_methods(per_object)

    per_object_path = out_dir / "crop_quality_per_object.csv"
    summary_path = out_dir / "crop_quality_summary.csv"
    manifest_path = out_dir / "crop_quality_manifest.json"

    per_object.to_csv(per_object_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = {
        "predictions_json": str(predictions_json),
        "images_dir": str(images_dir) if images_dir else None,
        "out_dir": str(out_dir),
        "min_confidence": min_confidence,
        "min_mask_area": min_mask_area,
        "padding_ratio": padding_ratio,
        "background": background,
        "examples_limit": examples_limit,
        "objects_with_masks": int(len(per_object)),
        "outputs": {
            "per_object": str(per_object_path),
            "summary": str(summary_path),
            "examples_dir": str(examples_dir),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "per_object": per_object_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "examples_dir": examples_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнивает bbox crop и mask crop для статьи про подготовку объектов к SKU-идентификации."
    )
    parser.add_argument("--predictions", required=True, help="predictions.json после YOLO-Seg inference")
    parser.add_argument("--images-dir", default=None, help="Папка изображений, если пути в predictions относительные")
    parser.add_argument("--out-dir", default="results/article_segmentation/crop_comparison")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=1)
    parser.add_argument("--padding", type=float, default=0.05, help="Padding вокруг bbox, как в SKU crop extraction")
    parser.add_argument("--background", type=int, default=255, help="Фон для mask crop: 255 белый, 0 чёрный")
    parser.add_argument("--examples-limit", type=int, default=30)
    args = parser.parse_args()

    outputs = run_comparison(
        predictions_json=args.predictions,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        min_confidence=args.min_confidence,
        min_mask_area=args.min_mask_area,
        padding_ratio=args.padding,
        background=args.background,
        examples_limit=args.examples_limit,
    )

    print("=== ShelfVision bbox vs mask crop comparison ===")
    print(f"Per-object: {outputs['per_object']}")
    print(f"Summary:    {outputs['summary']}")
    print(f"Examples:   {outputs['examples_dir']}")
    print(f"Manifest:   {outputs['manifest']}")


if __name__ == "__main__":
    main()
