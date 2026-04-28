from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from tqdm import tqdm


def apply_dark(img: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

def apply_noise(img: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img

def apply_downscale(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def coco_eval(gt_json: str, pred_json: str) -> Dict[str, float]:
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AR100": float(ev.stats[8]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)          # COCO test.json (тайлы)
    ap.add_argument("--images_dir", required=True)         # tiles dir
    ap.add_argument("--weights", required=True)            # best.pt
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # distortions
    ap.add_argument("--mode", required=True, choices=["clean", "dark", "blur", "noise", "jpeg", "downscale"])
    ap.add_argument("--param", type=float, default=0.0)    # alpha/sigma/std/quality/scale

    ap.add_argument("--out_dir", default="artifacts/dir5_robustness")
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.test_json)
    img_ids = coco.getImgIds()
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    cat_ids = coco.getCatIds()
    if len(cat_ids) != 1:
        raise RuntimeError(f"Expected 1 category, got {cat_ids}")
    cat_id = int(cat_ids[0])

    model = YOLO(args.weights)

    preds = []
    for img_id in tqdm(img_ids, desc=f"predict {args.mode}({args.param})"):
        info = coco.loadImgs(img_id)[0]
        fn = info["file_name"]
        p = Path(args.images_dir) / fn
        if not p.exists():
            p2 = Path(args.images_dir) / Path(fn).name
            if not p2.exists():
                continue
            p = p2

        img = cv2.imread(str(p))
        if img is None:
            continue

        # apply distortion
        if args.mode == "clean":
            x = img
        elif args.mode == "dark":
            x = apply_dark(img, float(args.param))
        elif args.mode == "blur":
            x = apply_blur(img, float(args.param))
        elif args.mode == "noise":
            x = apply_noise(img, float(args.param))
        elif args.mode == "jpeg":
            x = apply_jpeg(img, int(args.param))
        elif args.mode == "downscale":
            x = apply_downscale(img, float(args.param))
        else:
            x = img

        r = model.predict(
            source=x,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), sc in zip(xyxy, scores):
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            preds.append({
                "image_id": int(img_id),
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(sc),
            })

    tag = f"{args.mode}_{str(args.param).replace('.', 'p')}"
    pred_path = out_dir / f"pred_{tag}.json"
    pred_path.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")

    metrics = coco_eval(args.test_json, str(pred_path))
    report = {
        "mode": args.mode,
        "param": args.param,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "metrics": metrics,
        "pred_json": str(pred_path),
    }
    rep_path = out_dir / f"metrics_{tag}.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", rep_path)


if __name__ == "__main__":
    main()