# src/infer/merge_tiles.py
from typing import List
from src.infer.engine import DetObject

def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union

def merge_detections(dets: List[DetObject], cfg: dict) -> List[DetObject]:
    thr = float(cfg.get("iou_thr", 0.5))
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: List[DetObject] = []
    for d in dets:
        dup = False
        for k in keep:
            if iou_xywh(d.bbox, k.bbox) >= thr:
                dup = True
                break
        if not dup:
            keep.append(d)
        if len(keep) >= int(cfg.get("max_det", 3000)):
            break
    return keep
