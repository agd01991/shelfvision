# src/data/tiling.py
from dataclasses import dataclass
from typing import List, Tuple
from src.data.validators import BBox, clip_bbox_to_image, is_valid_bbox

@dataclass(frozen=True)
class Tile:
    tile_id: str
    x0: int
    y0: int
    w: int
    h: int

def build_tiles(img_w: int, img_h: int, tile_w: int, tile_h: int, overlap: int) -> List[Tile]:
    tiles: List[Tile] = []
    step_x = max(1, tile_w - overlap)
    step_y = max(1, tile_h - overlap)
    tid = 0
    for y0 in range(0, img_h, step_y):
        for x0 in range(0, img_w, step_x):
            x1 = min(img_w, x0 + tile_w)
            y1 = min(img_h, y0 + tile_h)
            # корректировка тайла у границ
            x0c = max(0, x1 - tile_w)
            y0c = max(0, y1 - tile_h)
            tiles.append(Tile(tile_id=f"t{tid:05d}", x0=x0c, y0=y0c, w=min(tile_w, img_w), h=min(tile_h, img_h)))
            tid += 1
    return tiles

def bbox_to_tile_coords(b: BBox, tile: Tile) -> BBox:
    return BBox(x=b.x - tile.x0, y=b.y - tile.y0, w=b.w, h=b.h)

def filter_tile_bbox(b: BBox, tile_w: int, tile_h: int, min_visible: float = 0.4) -> Tuple[bool, BBox]:
    clipped, _ = clip_bbox_to_image(b, tile_w, tile_h)
    if not is_valid_bbox(clipped):
        return False, clipped
    # доля видимой площади: clipped_area / orig_area
    orig_area = max(1e-6, b.w * b.h)
    vis_area = clipped.w * clipped.h
    if (vis_area / orig_area) < min_visible:
        return False, clipped
    return True, clipped

# src/data/tiling.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional

import cv2
import numpy as np

from src.data.coco_schema import CocoImage, CocoAnnotation


@dataclass
class TileConfig:
    enabled: bool = False
    tile_size: int = 1024
    overlap: float = 0.2  # 0..0.9
    min_visible: float = 0.25  # доля площади bbox, которая должна попасть в тайл
    save_format: str = "jpg"
    jpeg_quality: int = 92


def _compute_steps(full: int, tile: int, overlap: float) -> List[int]:
    if full <= tile:
        return [0]
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, full - tile + 1, step))
    if xs[-1] != full - tile:
        xs.append(full - tile)
    return xs


def generate_tiles(w: int, h: int, tile_size: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    xs = _compute_steps(w, tile_size, overlap)
    ys = _compute_steps(h, tile_size, overlap)
    tiles = []
    for y in ys:
        for x in xs:
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            tiles.append((x, y, tw, th))
    return tiles


def _bbox_intersection_area(b1: List[float], b2: List[float]) -> float:
    # b = [x,y,w,h]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def _clip_bbox_to_tile(b: List[float], tx: int, ty: int, tw: int, th: int) -> Optional[List[float]]:
    # переводим в координаты тайла и клипаем
    x, y, w, h = b
    x1, y1 = x - tx, y - ty
    x2, y2 = x1 + w, y1 + h

    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(tw), x2), min(float(th), y2)
    nw, nh = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
    if nw <= 0 or nh <= 0:
        return None
    return [cx1, cy1, nw, nh]


def tile_coco(
    images: List[CocoImage],
    anns: List[CocoAnnotation],
    image_root: Path,
    out_images_dir: Path,
    split_map: Dict[int, str],
    cfg: TileConfig,
    logger,
) -> Tuple[List[CocoImage], List[CocoAnnotation], Dict[str, Any], Dict[int, str]]:
    """
    Делает новый COCO на тайлах.
    split тайла = split исходного изображения (чтобы не было утечки).
    Возвращает: (tile_images, tile_anns, tile_map, tile_split_map)
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img: Dict[int, List[CocoAnnotation]] = {}
    for a in anns:
        anns_by_img.setdefault(a.image_id, []).append(a)

    tile_images: List[CocoImage] = []
    tile_anns: List[CocoAnnotation] = []
    tile_map: Dict[int, Dict[str, Any]] = {}
    tile_split_map: Dict[int, str] = {}

    next_img_id = 1
    next_ann_id = 1

    for im in images:
        img_path = image_root / im.file_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"[tiling] cannot read image: {img_path}")
            continue

        H, W = img.shape[:2]
        tiles = generate_tiles(W, H, cfg.tile_size, cfg.overlap)
        src_split = split_map.get(im.id, "train")

        im_anns = anns_by_img.get(im.id, [])

        for (tx, ty, tw, th) in tiles:
            # выбираем аннотации, которые достаточно попали в тайл
            kept: List[Tuple[CocoAnnotation, List[float]]] = []
            tile_box = [float(tx), float(ty), float(tw), float(th)]

            for a in im_anns:
                inter = _bbox_intersection_area(a.bbox, tile_box)
                if a.area <= 0:
                    continue
                if (inter / a.area) < cfg.min_visible:
                    continue

                clipped = _clip_bbox_to_tile(a.bbox, tx, ty, tw, th)
                if clipped is None:
                    continue
                kept.append((a, clipped))

            # если в тайле нет объектов — можно пропускать (или сохранять как negative, но это отдельная логика)
            if not kept:
                continue

            crop = img[ty : ty + th, tx : tx + tw]
            tile_name = f"{Path(im.file_name).stem}__x{tx}_y{ty}_w{tw}_h{th}.{cfg.save_format}"
            out_path = out_images_dir / tile_name

            if cfg.save_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
            else:
                cv2.imwrite(str(out_path), crop)

            tile_im = CocoImage(id=next_img_id, file_name=tile_name, width=tw, height=th)
            tile_images.append(tile_im)
            tile_split_map[next_img_id] = src_split

            tile_map[next_img_id] = {
                "src_image_id": im.id,
                "src_file_name": im.file_name,
                "x": tx,
                "y": ty,
                "w": tw,
                "h": th,
            }

            for (a, clipped_bbox) in kept:
                na = CocoAnnotation(
                    id=next_ann_id,
                    image_id=next_img_id,
                    category_id=a.category_id,
                    bbox=clipped_bbox,
                    area=float(clipped_bbox[2] * clipped_bbox[3]),
                    iscrowd=getattr(a, "iscrowd", 0),
                    segmentation=getattr(a, "segmentation", None),
                )
                tile_anns.append(na)
                next_ann_id += 1

            next_img_id += 1

    meta = {
        "tile_size": cfg.tile_size,
        "overlap": cfg.overlap,
        "min_visible": cfg.min_visible,
        "tiles_total": len(tile_images),
        "annotations_total": len(tile_anns),
    }
    return tile_images, tile_anns, tile_map, tile_split_map
