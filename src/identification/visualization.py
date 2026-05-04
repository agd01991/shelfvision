from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2

from .common import resolve_image_path
from .matcher import IdentificationResult


BOX_COLOR_MATCHED = (0, 180, 0)
BOX_COLOR_UNKNOWN = (0, 0, 220)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)


def _draw_label(image, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, h + baseline + 6)
    cv2.rectangle(image, (x, y - h - baseline - 6), (x + w + 8, y + 3), TEXT_BG_COLOR, -1)
    cv2.putText(image, text, (x + 4, y - baseline - 2), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def visualize_identification_results(
    results: List[IdentificationResult],
    images_dir: str | Path | None,
    out_dir: str | Path,
    limit: int = 30,
) -> List[Path]:
    out_dir = Path(out_dir) / "visualized"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_image: Dict[str, List[IdentificationResult]] = {}
    for item in results:
        by_image.setdefault(item.image_path, []).append(item)

    saved: List[Path] = []
    for image_index, (image_path_str, image_results) in enumerate(by_image.items()):
        if limit and image_index >= limit:
            break
        image_path = resolve_image_path(image_path_str, images_dir=images_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for item in image_results:
            color = BOX_COLOR_MATCHED if item.sku_status == "matched" else BOX_COLOR_UNKNOWN
            x1, y1, x2, y2 = [int(round(v)) for v in (item.x1, item.y1, item.x2, item.y2)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{item.sku_name} | {item.sku_confidence:.2f}"
            _draw_label(image, label, x1, y1)

        output_path = out_dir / f"{Path(image_path).stem}_identified.jpg"
        cv2.imwrite(str(output_path), image)
        saved.append(output_path)
    return saved
