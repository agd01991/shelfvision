# src/data/sku110k_reader.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2

from src.data.coco_schema import CocoImage, CocoAnnotation, CocoCategory


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(w), int(h)


def load_sku110k_as_coco(
    images_dir: Path,
    csv_path: Path,
    category_name: str = "product",
    category_id: int = 1,
    logger=None,
) -> Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
    """
    Ожидаемый формат CSV (типичный для SKU-110K):
    image,x1,y1,x2,y2,class,image_width,image_height
    class может игнорироваться (если делается просто "товар" как 1 класс).
    """
    img_map: Dict[str, int] = {}
    images: List[CocoImage] = []
    anns: List[CocoAnnotation] = []
    cats = [CocoCategory(id=category_id, name=category_name)]

    next_img_id = 1
    next_ann_id = 1

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image", "x1", "y1", "x2", "y2"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"SKU-110K CSV: expected columns {sorted(required)}; got {reader.fieldnames}")

        for row in reader:
            rel = row["image"].strip()
            if rel == "":
                continue

            if rel not in img_map:
                img_path = images_dir / rel
                w = int(float(row.get("image_width", 0) or 0))
                h = int(float(row.get("image_height", 0) or 0))
                if w <= 0 or h <= 0:
                    w2, h2 = _read_image_size(img_path)
                    w, h = w2, h2

                img_map[rel] = next_img_id
                images.append(CocoImage(id=next_img_id, file_name=rel, width=w, height=h))
                next_img_id += 1

            image_id = img_map[rel]
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = bw * bh

            anns.append(
                CocoAnnotation(
                    id=next_ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    bbox=[x1, y1, bw, bh],
                    area=area,
                    iscrowd=0,
                    segmentation=None,
                )
            )
            next_ann_id += 1

    if logger:
        logger.info(f"[sku110k] images={len(images)} anns={len(anns)} from {csv_path.name}")
    return images, anns, cats
