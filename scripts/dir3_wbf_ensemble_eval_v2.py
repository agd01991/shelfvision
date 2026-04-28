from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO


# ----------------------------
# Utils: path normalization (WSL-friendly)
# ----------------------------
def normalize_any_path(p: str) -> str:
    """Accepts Windows paths like D:\\... and converts to /mnt/d/... in WSL."""
    p = p.strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", p):
        # try wslpath (best)
        try:
            return subprocess.check_output(["wslpath", "-u", p]).decode().strip()
        except Exception:
            drive = p[0].lower()
            tail = p[2:].lstrip("\\/").replace("\\", "/")
            return f"/mnt/{drive}/{tail}"
    return p


def resolve_path(p: str) -> Path:
    return Path(normalize_any_path(p)).expanduser().resolve()


def chunked(items: List[Tuple[int, str, int, int]], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ----------------------------
# COCO eval
# ----------------------------
def coco_eval_bbox(gt_json: str, pred_json_path: str) -> Dict[str, float]:
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    return {
        "AP": float(ev.stats[0]),      # mAP50-95
        "AP50": float(ev.stats[1]),    # mAP50
        "AP75": float(ev.stats[2]),
        "AR1": float(ev.stats[6]),
        "AR10": float(ev.stats[7]),
        "AR100": float(ev.stats[8]),
    }


# ----------------------------
# Prediction helpers
# ----------------------------
def predict_ultralytics_on_items(
    weights: str,
    items: List[Tuple[int, str, int, int]],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    batch: int,
    half: bool,
    max_det: int = 300,
) -> Dict[int, Tuple[List[List[float]], List[float], List[int]]]:
    """
    Returns dict: image_id -> (boxes_xyxy, scores, labels)
    Uses stream inference + chunking and frees VRAM periodically.
    """
    model = YOLO(weights)

    out: Dict[int, Tuple[List[List[float]], List[float], List[int]]] = {}

    # Torch inference mode to reduce memory
    with torch.inference_mode():
        for pack in tqdm(list(chunked(items, max(1, batch))), desc=f"predict[{Path(weights).name}]"):
            paths = [p for _, p, _, _ in pack]

            # stream=True yields results one-by-one (lower peak RAM)
            results_iter = model.predict(
                source=paths,
                imgsz=imgsz,
                device=device,
                conf=conf,
                iou=iou,
                verbose=False,
                stream=True,
                half=half,
                max_det=max_det,
            )

            for (img_id, _path, _w, _h), r in zip(pack, results_iter):
                if r.boxes is None or len(r.boxes) == 0:
                    out[int(img_id)] = ([], [], [])
                    continue

                xyxy = r.boxes.xyxy.detach().cpu().numpy().tolist()
                scores = r.boxes.conf.detach().cpu().numpy().tolist()
                labels = r.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                out[int(img_id)] = (xyxy, scores, labels)

            # free cached blocks (helps on 8GB)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # unload model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def norm_xyxy(boxes: List[List[float]], w: int, h: int) -> List[List[float]]:
    nb = []
    for x1, y1, x2, y2 in boxes:
        nb.append([x1 / w, y1 / h, x2 / w, y2 / h])
    return nb


def denorm_to_xywh(box: List[float], w: int, h: int) -> List[float]:
    x1 = box[0] * w
    y1 = box[1] * h
    x2 = box[2] * w
    y2 = box[3] * h
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(bw), float(bh)]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True, help="COCO test.json (v1_tiled)")
    ap.add_argument("--images_dir", required=True, help="tiles dir (images)")
    ap.add_argument("--yolo_weights", required=True)
    ap.add_argument("--detr_weights", required=True)

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")

    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)

    # separate batches to avoid OOM on RT-DETR
    ap.add_argument("--yolo_batch", type=int, default=8)
    ap.add_argument("--detr_batch", type=int, default=1)

    ap.add_argument("--half", action="store_true", default=True, help="Use FP16 inference on GPU")

    ap.add_argument("--wbf_iou_thr", type=float, default=0.55)
    ap.add_argument("--wbf_skip_thr", type=float, default=0.001)
    ap.add_argument("--wbf_w1", type=float, default=1.0)
    ap.add_argument("--wbf_w2", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0, help="0=all, else first N images")
    ap.add_argument("--out_dir", default="artifacts/dir3_wbf")

    args = ap.parse_args()

    # normalize paths for WSL
    test_json = str(resolve_path(args.test_json))
    images_dir = resolve_path(args.images_dir)
    yolo_weights = str(resolve_path(args.yolo_weights))
    detr_weights = str(resolve_path(args.detr_weights))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional: reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # load COCO
    coco = COCO(test_json)
    img_ids = coco.getImgIds()
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    cat_ids = coco.getCatIds()
    if len(cat_ids) != 1:
        raise RuntimeError(f"Expected 1 category in COCO, got {cat_ids}")
    cat_id = int(cat_ids[0])

    # resolve file paths
    items: List[Tuple[int, str, int, int]] = []
    missing = 0
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        fn = info["file_name"]
        p = images_dir / fn
        if not p.exists():
            p2 = images_dir / Path(fn).name
            if not p2.exists():
                missing += 1
                continue
            p = p2

        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        if w <= 0 or h <= 0:
            im = cv2.imread(str(p))
            if im is None:
                missing += 1
                continue
            h, w = im.shape[:2]
        items.append((int(img_id), str(p), int(w), int(h)))

    if not items:
        raise RuntimeError("No test images resolved. Check images_dir vs file_name paths.")
    if missing:
        print(f"[WARN] missing images skipped: {missing}")

    # --- 1) predict YOLO then free VRAM ---
    t0 = time.time()
    yolo_preds = predict_ultralytics_on_items(
        weights=yolo_weights,
        items=items,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        batch=args.yolo_batch,
        half=args.half,
    )
    t_yolo = time.time() - t0

    # --- 2) predict RT-DETR then free VRAM ---
    t0 = time.time()
    detr_preds = predict_ultralytics_on_items(
        weights=detr_weights,
        items=items,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        batch=args.detr_batch,  # default 1 to avoid OOM
        half=args.half,
    )
    t_detr = time.time() - t0

    # --- 3) Build COCO predictions for YOLO, RT-DETR and WBF ensemble ---
    preds_yolo = []
    preds_detr = []
    preds_wbf = []

    for img_id, _path, w, h in tqdm(items, desc="WBF"):
        b1, s1, l1 = yolo_preds.get(img_id, ([], [], []))
        b2, s2, l2 = detr_preds.get(img_id, ([], [], []))

        def filt(boxes, scores, labels):
            fb, fs, fl = [], [], []
            for bb, sc, ll in zip(boxes, scores, labels):
                if int(ll) == 0:
                    fb.append(bb)
                    fs.append(float(sc))
                    fl.append(0)
            return fb, fs, fl

        b1, s1, l1 = filt(b1, s1, l1)
        b2, s2, l2 = filt(b2, s2, l2)

        # YOLO-only
        for bb, sc in zip(b1, s1):
            x1, y1, x2, y2 = bb
            preds_yolo.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(sc),
            })

        # RT-DETR-only
        for bb, sc in zip(b2, s2):
            x1, y1, x2, y2 = bb
            preds_detr.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(sc),
            })

        if not b1 and not b2:
            continue

        boxes_list = [norm_xyxy(b1, w, h), norm_xyxy(b2, w, h)]
        scores_list = [s1, s2]
        labels_list = [l1, l2]

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[args.wbf_w1, args.wbf_w2],
            iou_thr=args.wbf_iou_thr,
            skip_box_thr=args.wbf_skip_thr,
        )

        for bb, sc, ll in zip(boxes, scores, labels):
            if int(ll) != 0:
                continue
            preds_wbf.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": denorm_to_xywh(bb, w, h),
                "score": float(sc),
            })

    # --- 4) Save predictions ---
    p_yolo = out_dir / "pred_yolo.json"
    p_detr = out_dir / "pred_rtdetr.json"
    p_wbf = out_dir / "pred_wbf.json"

    p_yolo.write_text(json.dumps(preds_yolo, ensure_ascii=False), encoding="utf-8")
    p_detr.write_text(json.dumps(preds_detr, ensure_ascii=False), encoding="utf-8")
    p_wbf.write_text(json.dumps(preds_wbf, ensure_ascii=False), encoding="utf-8")

    # --- 5) COCO eval ---
    m_yolo = coco_eval_bbox(test_json, str(p_yolo))
    m_detr = coco_eval_bbox(test_json, str(p_detr))
    m_wbf = coco_eval_bbox(test_json, str(p_wbf))

    report = {
        "name": "dir3_wbf",
        "test_json": test_json,
        "images_dir": str(images_dir),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "batches": {"yolo_batch": args.yolo_batch, "detr_batch": args.detr_batch},
        "half": bool(args.half),
        "wbf": {
            "iou_thr": args.wbf_iou_thr,
            "skip_thr": args.wbf_skip_thr,
            "weights": [args.wbf_w1, args.wbf_w2],
        },
        "timing_seconds": {
            "yolo_predict_total": round(t_yolo, 2),
            "detr_predict_total": round(t_detr, 2),
            "images_count": len(items),
            "yolo_sec_per_image": round(t_yolo / max(1, len(items)), 4),
            "detr_sec_per_image": round(t_detr / max(1, len(items)), 4),
        },
        "metrics_test": {"yolo": m_yolo, "rtdetr": m_detr, "wbf": m_wbf},
        "predictions": {"yolo": str(p_yolo), "rtdetr": str(p_detr), "wbf": str(p_wbf)},
    }

    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()




