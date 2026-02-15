from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from src.data.coco_schema import CocoImage, CocoAnnotation


@dataclass
class ValidationStats:
    total_anns: int = 0
    invalid_area: int = 0
    clipped: int = 0
    dropped: int = 0


# def clip_bbox(bbox: List[float], w: int, h: int) -> Tuple[List[float], bool]:
#     x, y, bw, bh = bbox
#     x1 = max(0.0, x)
#     y1 = max(0.0, y)
#     x2 = min(float(w), x + max(0.0, bw))
#     y2 = min(float(h), y + max(0.0, bh))
#     nw = max(0.0, x2 - x1)
#     nh = max(0.0, y2 - y1)
#     changed = (x1 != x) or (y1 != y) or (nw != bw) or (nh != bh)
#     return [x1, y1, nw, nh], changed

def clip_bbox(bbox: List[float], w: int, h: int) -> Tuple[List[float], bool]:
    x, y, bw, bh = bbox
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(w), x + max(0.0, bw))
    y2 = min(float(h), y + max(0.0, bh))
    nw = max(0.0, x2 - x1)
    nh = max(0.0, y2 - y1)

    eps = 1e-6
    changed = (abs(x1 - x) > eps) or (abs(y1 - y) > eps) or (abs(nw - bw) > eps) or (abs(nh - bh) > eps)
    return [x1, y1, nw, nh], changed


def bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def validate_and_fix_bboxes(
    images: List[CocoImage],
    anns: List[CocoAnnotation],
    min_area: float,
    clip_to_image: bool,
) -> Tuple[List[CocoAnnotation], Dict[str, Any], ValidationStats]:
    img_map = {im.id: im for im in images}

    fixed: List[CocoAnnotation] = []
    issues: Dict[str, Any] = {
        "clipped": [],
        "dropped": [],
        "missing_image": [],
    }
    stats = ValidationStats(total_anns=len(anns))

    clipped_ratio_sum = 0.0        # сумма (after_area / before_area)
    clipped_ratio_n = 0            # сколько clipped вообще
    clipped_hard = 0               # сколько “жёстких” клиппов (например, < 0.7 площади осталось)

    for a in anns:
        im = img_map.get(a.image_id)
        if im is None:
            issues["missing_image"].append({"ann_id": a.id, "image_id": a.image_id})
            stats.dropped += 1
            continue

        bbox = a.bbox
        if clip_to_image:
            before_bbox = bbox
            before_area = bbox_area(before_bbox)

            bbox2, changed = clip_bbox(before_bbox, im.width, im.height)
            if changed:
                issues["clipped"].append(
                    {"ann_id": a.id, "image_id": a.image_id, "before": before_bbox, "after": bbox2}
                )
                stats.clipped += 1

                after_area = bbox_area(bbox2)
                ratio = after_area / max(1e-6, before_area)
                clipped_ratio_sum += ratio
                clipped_ratio_n += 1
                if ratio < 0.7:
                    clipped_hard += 1

            bbox = bbox2

        area = bbox_area(bbox)
        if area < min_area:
            issues["dropped"].append(
                {"ann_id": a.id, "image_id": a.image_id, "reason": "min_area", "bbox": bbox, "area": area}
            )
            stats.invalid_area += 1
            stats.dropped += 1
            continue

        a.bbox = bbox
        a.area = area
        fixed.append(a)

    issues["clipped_summary"] = {
        "count": clipped_ratio_n,
        "avg_after_to_before_area": (clipped_ratio_sum / clipped_ratio_n) if clipped_ratio_n else 1.0,
        "hard_clip_count_ratio_lt_0.7": clipped_hard,
    }

    return fixed, issues, stats





####################################################################################################

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# from src.data.coco_schema import CocoImage, CocoAnnotation


# @dataclass
# class ValidationStats:
#     total_anns: int = 0
#     invalid_area: int = 0
#     clipped: int = 0
#     dropped: int = 0


# def clip_bbox(bbox: List[float], w: int, h: int) -> Tuple[List[float], bool]:
#     x, y, bw, bh = bbox
#     x1 = max(0.0, x)
#     y1 = max(0.0, y)
#     x2 = min(float(w), x + max(0.0, bw))
#     y2 = min(float(h), y + max(0.0, bh))
#     nw = max(0.0, x2 - x1)
#     nh = max(0.0, y2 - y1)
#     changed = (x1 != x) or (y1 != y) or (nw != bw) or (nh != bh)
#     return [x1, y1, nw, nh], changed


# def bbox_area(bbox: List[float]) -> float:
#     return max(0.0, bbox[2]) * max(0.0, bbox[3])


# def validate_and_fix_bboxes(
#     images: List[CocoImage],
#     anns: List[CocoAnnotation],
#     min_area: float,
#     clip_to_image: bool,
# ) -> Tuple[List[CocoAnnotation], Dict, ValidationStats]:
#     img_map = {im.id: im for im in images}

#     fixed: List[CocoAnnotation] = []
#     issues: Dict[str, List[Dict]] = {
#         "clipped": [],
#         "dropped": [],
#         "missing_image": [],
#     }
#     stats = ValidationStats(total_anns=len(anns))

#     for a in anns:
#         im = img_map.get(a.image_id)
#         if im is None:
#             issues["missing_image"].append({"ann_id": a.id, "image_id": a.image_id})
#             stats.dropped += 1
#             continue

#         bbox = a.bbox
#         changed = False
#         if clip_to_image:
#             bbox2, changed = clip_bbox(bbox, im.width, im.height)
#             if changed:
#                 issues["clipped"].append({"ann_id": a.id, "image_id": a.image_id, "before": bbox, "after": bbox2})
#                 stats.clipped += 1
#             bbox = bbox2

#         area = bbox_area(bbox)
#         if area < min_area:
#             issues["dropped"].append(
#                 {"ann_id": a.id, "image_id": a.image_id, "reason": "min_area", "bbox": bbox, "area": area}
#             )
#             stats.invalid_area += 1
#             stats.dropped += 1
#             continue

#         a.bbox = bbox
#         a.area = area
#         fixed.append(a)

#     return fixed, issues, stats

#################################################################################################################################

# # src/data/validators.py
# from dataclasses import dataclass
# from typing import Tuple, List

# @dataclass
# class BBox:
#     x: float
#     y: float
#     w: float
#     h: float

# def clip_bbox_to_image(b: BBox, img_w: int, img_h: int) -> Tuple[BBox, bool]:
#     # returns (bbox, changed)
#     x1 = max(0.0, b.x)
#     y1 = max(0.0, b.y)
#     x2 = min(float(img_w), b.x + max(0.0, b.w))
#     y2 = min(float(img_h), b.y + max(0.0, b.h))
#     w = max(0.0, x2 - x1)
#     h = max(0.0, y2 - y1)
#     changed = (x1 != b.x) or (y1 != b.y) or (w != b.w) or (h != b.h)
#     return BBox(x1, y1, w, h), changed

# def is_valid_bbox(b: BBox, min_area: float = 4.0) -> bool:
#     if b.w <= 0 or b.h <= 0:
#         return False
#     return (b.w * b.h) >= min_area
