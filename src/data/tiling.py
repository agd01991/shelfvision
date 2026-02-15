from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2

from src.data.coco_schema import CocoImage, CocoAnnotation


@dataclass
class TileConfig:
    enabled: bool = False
    tile_size: int = 1024
    overlap: float = 0.2
    min_visible: float = 0.25
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


def generate_tiles(
    w: int, h: int, tile_size: int, overlap: float
) -> List[Tuple[int, int, int, int]]:
    xs = _compute_steps(w, tile_size, overlap)
    ys = _compute_steps(h, tile_size, overlap)
    tiles: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            tiles.append((x, y, tw, th))
    return tiles


def _bbox_intersection_area(b1: List[float], b2: List[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def _clip_bbox_to_tile(
    b: List[float], tx: int, ty: int, tw: int, th: int
) -> Optional[List[float]]:
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
) -> Tuple[
    List[CocoImage], List[CocoAnnotation], Dict[int, Dict[str, Any]], Dict[int, str]
]:
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

        for tx, ty, tw, th in tiles:
            kept: List[Tuple[CocoAnnotation, List[float]]] = []
            tile_box = [float(tx), float(ty), float(tw), float(th)]

            for a in im_anns:
                a_area = (
                    float(a.area)
                    if (a.area is not None and a.area > 0)
                    else float(a.bbox[2] * a.bbox[3])
                )
                if a_area <= 0:
                    continue

                inter = _bbox_intersection_area(a.bbox, tile_box)
                if (inter / a_area) < cfg.min_visible:
                    continue

                clipped = _clip_bbox_to_tile(a.bbox, tx, ty, tw, th)
                if clipped is None:
                    continue

                kept.append((a, clipped))

            if not kept:
                continue

            crop = img[ty : ty + th, tx : tx + tw]
            tile_name = (
                f"{Path(im.file_name).stem}__x{tx}_y{ty}_w{tw}_h{th}.{cfg.save_format}"
            )
            out_path = out_images_dir / tile_name

            if cfg.save_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(
                    str(out_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)],
                )
            else:
                cv2.imwrite(str(out_path), crop)

            tile_im = CocoImage(
                id=next_img_id, file_name=tile_name, width=tw, height=th
            )
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

            for a, clipped_bbox in kept:
                na = CocoAnnotation(
                    id=next_ann_id,
                    image_id=next_img_id,
                    category_id=a.category_id,
                    bbox=clipped_bbox,
                    area=float(clipped_bbox[2] * clipped_bbox[3]),
                    iscrowd=int(getattr(a, "iscrowd", 0)),
                    segmentation=getattr(a, "segmentation", None),
                )
                tile_anns.append(na)
                next_ann_id += 1

            next_img_id += 1

    return tile_images, tile_anns, tile_map, tile_split_map
