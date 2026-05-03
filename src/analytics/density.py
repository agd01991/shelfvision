from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import load_predictions


BBox = Sequence[float]


def _prediction_name(prediction: Dict[str, Any]) -> str:
    return Path(str(prediction.get("image_path", ""))).name


def _box_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _zone_name(row: int, col: int, rows: int, cols: int) -> str:
    vertical = {
        0: "верхняя зона",
        1: "средняя зона",
        2: "нижняя зона",
    }.get(row, f"зона по вертикали {row + 1}")

    horizontal = {
        0: "левая часть",
        1: "центр",
        2: "правая часть",
    }.get(col, f"часть по горизонтали {col + 1}")

    if rows == 3 and cols == 3:
        return f"{vertical}, {horizontal}"
    return f"row_{row + 1}_col_{col + 1}"


def _load_image_size(prediction: Dict[str, Any]) -> Tuple[int, int]:
    width = prediction.get("image_width")
    height = prediction.get("image_height")
    if width and height:
        return int(width), int(height)

    image_path = Path(str(prediction.get("image_path", "")))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось определить размер изображения: {image_path}")
    h, w = image.shape[:2]
    return int(w), int(h)


def analyze_prediction_density(
    prediction: Dict[str, Any],
    rows: int = 3,
    cols: int = 3,
) -> pd.DataFrame:
    """Считает плотность найденных объектов по сетке изображения."""

    width, height = _load_image_size(prediction)
    boxes = prediction.get("boxes", []) or []
    scores = prediction.get("scores", []) or [0.0] * len(boxes)

    zone_rows: List[Dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            zone_rows.append(
                {
                    "image_name": _prediction_name(prediction),
                    "model_name": prediction.get("model_name", "unknown"),
                    "row": row + 1,
                    "col": col + 1,
                    "zone_name": _zone_name(row, col, rows, cols),
                    "objects_count": 0,
                    "avg_confidence": 0.0,
                    "zone_x1": round(col * width / cols, 2),
                    "zone_y1": round(row * height / rows, 2),
                    "zone_x2": round((col + 1) * width / cols, 2),
                    "zone_y2": round((row + 1) * height / rows, 2),
                }
            )

    score_buckets: Dict[Tuple[int, int], List[float]] = {}
    for box, score in zip(boxes, scores):
        cx, cy = _box_center(box)
        col = min(cols - 1, max(0, int(cx / max(1, width) * cols)))
        row = min(rows - 1, max(0, int(cy / max(1, height) * rows)))
        idx = row * cols + col
        zone_rows[idx]["objects_count"] += 1
        score_buckets.setdefault((row, col), []).append(float(score))

    for item in zone_rows:
        key = (int(item["row"]) - 1, int(item["col"]) - 1)
        values = score_buckets.get(key, [])
        item["avg_confidence"] = round(sum(values) / len(values), 4) if values else 0.0

    total = sum(item["objects_count"] for item in zone_rows)
    for item in zone_rows:
        item["share"] = round(item["objects_count"] / total, 4) if total else 0.0
    return pd.DataFrame(zone_rows)


def visualize_density(
    prediction: Dict[str, Any],
    density_df: pd.DataFrame,
    output_path: str | Path,
    alpha: float = 0.35,
) -> Path:
    """Сохраняет изображение с сеткой зон и интенсивностью заполнения."""

    image_path = Path(str(prediction.get("image_path", "")))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    overlay = image.copy()
    max_count = int(density_df["objects_count"].max()) if not density_df.empty else 0

    for _, row in density_df.iterrows():
        x1 = int(row["zone_x1"])
        y1 = int(row["zone_y1"])
        x2 = int(row["zone_x2"])
        y2 = int(row["zone_y2"])
        count = int(row["objects_count"])
        intensity = 0 if max_count == 0 else int(255 * count / max_count)
        color = (0, 255 - intensity // 2, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        text = f"{count} obj"
        cv2.putText(
            image,
            text,
            (x1 + 8, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    footer = f"Density: total={int(density_df['objects_count'].sum())}, max_zone={max_count}"
    cv2.putText(
        image,
        footer,
        (8, image.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def analyze_density_file(
    predictions_json: str | Path,
    out_dir: str | Path = "results/density",
    rows: int = 3,
    cols: int = 3,
    visualize: bool = True,
    limit: int = 0,
) -> Dict[str, Path]:
    predictions = load_predictions(predictions_json)
    selected = predictions[: limit or None]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    saved_images: List[str] = []

    for prediction in selected:
        density_df = analyze_prediction_density(prediction, rows=rows, cols=cols)
        all_rows.append(density_df)
        if visualize:
            image_name = Path(_prediction_name(prediction)).stem
            out_image = out_dir / "visualized" / f"{image_name}__density.jpg"
            saved_images.append(str(visualize_density(prediction, density_df, out_image)))

    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    zones_csv = out_dir / "density_by_zone.csv"
    summary_csv = out_dir / "density_summary.csv"
    report_json = out_dir / "density_report.json"

    full_df.to_csv(zones_csv, index=False)

    if not full_df.empty:
        summary_df = (
            full_df.groupby(["row", "col", "zone_name"], as_index=False)
            .agg(
                objects_count=("objects_count", "sum"),
                avg_confidence=("avg_confidence", "mean"),
                avg_share=("share", "mean"),
            )
            .sort_values("objects_count", ascending=False)
        )
    else:
        summary_df = pd.DataFrame()
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "predictions_json": str(predictions_json),
        "images_count": len(selected),
        "rows": rows,
        "cols": cols,
        "total_objects": int(full_df["objects_count"].sum()) if not full_df.empty else 0,
        "densest_zone": summary_df.iloc[0].to_dict() if not summary_df.empty else None,
        "saved_images": saved_images,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"zones_csv": zones_csv, "summary_csv": summary_csv, "report_json": report_json}
