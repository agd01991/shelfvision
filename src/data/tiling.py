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
