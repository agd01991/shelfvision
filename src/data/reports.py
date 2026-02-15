from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import random
import cv2

from src.data.coco_schema import CocoImage, CocoAnnotation


def draw_samples(
    image_root: str | Path,
    images: List[CocoImage],
    anns_by_image: Dict[int, List[CocoAnnotation]],
    out_dir: str | Path,
    sample_n: int,
    draw_max_boxes: int,
    seed: int = 42,
) -> None:
    image_root = Path(image_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(seed)
    imgs = images[:]
    rnd.shuffle(imgs)
    imgs = imgs[:sample_n]

    for im in imgs:
        src = image_root / im.file_name
        if not src.exists():
            alt = image_root / Path(im.file_name).name
            if alt.exists():
                src = alt
            else:
                continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        dets = anns_by_image.get(im.id, [])[:draw_max_boxes]
        for a in dets:
            x, y, w, h = a.bbox
            p1 = (int(round(x)), int(round(y)))
            p2 = (int(round(x + w)), int(round(y + h)))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

        out_path = out_dir / f"{im.id:07d}.jpg"
        cv2.imwrite(str(out_path), img)
