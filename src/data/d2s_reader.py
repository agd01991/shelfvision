# src/data/d2s_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from src.data.coco_schema import CocoImage, CocoAnnotation, CocoCategory


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    COCO RLE (uncompressed): counts as list.
    Порядок обхода COCO: Fortran order (column-major).
    """
    m = binary_mask.astype(np.uint8)
    h, w = m.shape
    pixels = m.T.flatten()  # Fortran order
    counts: List[int] = []
    count = 0
    prev = 0
    for p in pixels:
        if p == prev:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev = int(p)
    counts.append(count)
    return {"size": [h, w], "counts": counts}


def _bbox_from_mask(binary_mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def load_d2s_as_coco_from_instance_id_png(
    images_dir: Path,
    masks_dir: Path,
    encoding: str,
    categories: List[CocoCategory],
    mask_suffix: str = ".png",
    logger=None,
) -> Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
    """
    encoding:
      - "class_times_1000_plus_instance": label = class_id*1000 + inst_id, background=0
      - "class_only": label = class_id (без instance), тогда это не instance-сегментация (редко нужно)
    """
    img_paths = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )

    cat_ids = {c.id for c in categories}
    images: List[CocoImage] = []
    anns: List[CocoAnnotation] = []

    next_img_id = 1
    next_ann_id = 1

    for ip in img_paths:
        rel = ip.relative_to(images_dir).as_posix()

        mp = (masks_dir / ip.relative_to(images_dir)).with_suffix(mask_suffix)
        if not mp.exists():
            # иногда маски лежат плоско: masks_dir/<stem>.png
            mp2 = masks_dir / (ip.stem + mask_suffix)
            if mp2.exists():
                mp = mp2
            else:
                continue

        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue

        images.append(
            CocoImage(id=next_img_id, file_name=rel, width=int(W), height=int(H))
        )

        uniq = np.unique(m)
        uniq = uniq[uniq != 0]

        if encoding == "class_times_1000_plus_instance":
            # label = class*1000 + inst
            for v in uniq:
                v = int(v)
                class_id = v // 1000
                if class_id == 0:
                    continue
                if class_id not in cat_ids:
                    # неизвестный класс — можно пропустить или залогировать
                    if logger:
                        logger.warning(
                            f"[d2s] unknown class_id={class_id} value={v} file={mp.name}"
                        )
                    continue

                bin_mask = m == v
                bbox = _bbox_from_mask(bin_mask)
                if bbox is None:
                    continue
                area = float(bin_mask.sum())

                anns.append(
                    CocoAnnotation(
                        id=next_ann_id,
                        image_id=next_img_id,
                        category_id=class_id,
                        bbox=bbox,
                        area=area,
                        iscrowd=1,  # RLE в COCO обычно с iscrowd=1
                        segmentation=_mask_to_rle(bin_mask),
                    )
                )
                next_ann_id += 1

        elif encoding == "class_only":
            for class_id in uniq:
                class_id = int(class_id)
                if class_id not in cat_ids:
                    continue
                bin_mask = m == class_id
                bbox = _bbox_from_mask(bin_mask)
                if bbox is None:
                    continue
                area = float(bin_mask.sum())
                anns.append(
                    CocoAnnotation(
                        id=next_ann_id,
                        image_id=next_img_id,
                        category_id=class_id,
                        bbox=bbox,
                        area=area,
                        iscrowd=1,
                        segmentation=_mask_to_rle(bin_mask),
                    )
                )
                next_ann_id += 1
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        next_img_id += 1

    if logger:
        logger.info(f"[d2s] images={len(images)} anns={len(anns)} encoding={encoding}")
    return images, anns, categories
