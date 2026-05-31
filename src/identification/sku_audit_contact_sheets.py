from __future__ import annotations

import math
from pathlib import Path
from typing import List

import cv2
import numpy as np


def _make_thumb(path: Path, label: str, size: int = 128) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None

    h, w = image.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size + 24, size, 3), 255, dtype=np.uint8)
    y = (size - new_h) // 2 + 24
    x = (size - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized

    cv2.putText(
        canvas,
        label[:22],
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
    )
    return canvas


def write_pair_contact_sheet(
    sku_a: str,
    refs_a: List[Path],
    sku_b: str,
    refs_b: List[Path],
    output_path: Path,
    max_refs: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbs: List[np.ndarray] = []

    for ref in refs_a[:max_refs]:
        thumb = _make_thumb(ref, f"{sku_a}/{ref.name}")
        if thumb is not None:
            thumbs.append(thumb)

    for ref in refs_b[:max_refs]:
        thumb = _make_thumb(ref, f"{sku_b}/{ref.name}")
        if thumb is not None:
            thumbs.append(thumb)

    if not thumbs:
        return

    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]

    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * th
        x = (idx % cols) * tw
        sheet[y : y + th, x : x + tw] = thumb

    cv2.imwrite(str(output_path), sheet)
