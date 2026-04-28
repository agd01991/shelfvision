from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# torch is optional for cache cleanup, but usually available
try:
    import torch
except Exception:
    torch = None

# detectron2 optional
D2_AVAILABLE = True
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
except Exception:
    D2_AVAILABLE = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -------------------------
# Path helpers
# -------------------------
def normalize_any_path(p: str) -> str:
    """Normalize Windows path to WSL if needed."""
    p = str(p).strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", p):
        # Windows path -> WSL path
        try:
            return subprocess.check_output(["wslpath", "-u", p]).decode().strip()
        except Exception:
            drive = p[0].lower()
            tail = p[2:].lstrip("\\/").replace("\\", "/")
            return f"/mnt/{drive}/{tail}"
    return p


def P(p: str | Path) -> Path:
    return Path(normalize_any_path(str(p))).expanduser().resolve()


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def get_found_path(paths: dict, key: str) -> Optional[Path]:
    item = paths.get(key, {})
    if not isinstance(item, dict):
        return None
    if item.get("status") != "FOUND":
        return None
    raw = item.get("path")
    if not raw:
        return None
    pp = P(raw)
    return pp if pp.exists() else None


# -------------------------
# Image helpers
# -------------------------
def list_images(folder: Path, limit: int = 0, seed: int = 42) -> List[Path]:
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    if limit and limit > 0 and len(imgs) > limit:
        rng = random.Random(seed)
        idx = list(range(len(imgs)))
        rng.shuffle(idx)
        imgs = [imgs[i] for i in idx[:limit]]
        imgs.sort()
    return imgs


def draw_boxes(img: np.ndarray, boxes_xyxy: List[List[float]], scores: List[float], label: str) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), sc in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{label}:{sc:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


# -------------------------
# Prediction helpers
# -------------------------
def _cuda_cleanup():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def ultralytics_predict(
    model: YOLO,
    img_path: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float = 0.7,
    half: bool = True,
    max_det: int = 300,
) -> Tuple[List[List[float]], List[float]]:
    """
    Single-image predict to avoid OOM.
    Returns xyxy and scores.
    """
    try:
        r = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
            stream=False,
            half=half,
            max_det=max_det,
        )[0]
    except Exception as e:
        # If CUDA OOM -> retry on CPU
        msg = str(e).lower()
        if ("out of memory" in msg or "cuda" in msg) and device != "cpu":
            _cuda_cleanup()
            r = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                device="cpu",
                conf=conf,
                iou=iou,
                verbose=False,
                stream=False,
                half=False,
                max_det=max_det,
            )[0]
        else:
            raise

    if r.boxes is None or len(r.boxes) == 0:
        return [], []
    xyxy = r.boxes.xyxy.detach().cpu().numpy().tolist()
    scores = r.boxes.conf.detach().cpu().numpy().tolist()
    return xyxy, scores


def wbf_merge(
    img_w: int,
    img_h: int,
    a_xyxy: List[List[float]],
    a_sc: List[float],
    b_xyxy: List[List[float]],
    b_sc: List[float],
    iou_thr: float,
    skip_thr: float,
    w1: float,
    w2: float,
) -> Tuple[List[List[float]], List[float]]:
    def norm(xyxy):
        out = []
        for x1, y1, x2, y2 in xyxy:
            out.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
        return out

    boxes_list = [norm(a_xyxy), norm(b_xyxy)]
    scores_list = [a_sc, b_sc]
    labels_list = [[0] * len(a_sc), [0] * len(b_sc)]

    boxes, scores, _labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=[w1, w2],
        iou_thr=iou_thr,
        skip_box_thr=skip_thr,
    )

    xyxy = []
    for x1, y1, x2, y2 in boxes:
        xyxy.append([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h])
    return xyxy, list(scores)


# -------------------------
# Best system selection (from metrics)
# -------------------------
def choose_best_system_from_paths(paths: dict) -> Tuple[str, dict]:
    """
    Priority:
      1) DIR3_metrics_json (AP from WBF eval on test) -> best among yolo/rtdetr/wbf
      2) DIR1_metrics_csv (best model AP50-95)
      3) fallback: "wbf"
    """
    # 1) dir3
    p_dir3 = get_found_path(paths, "DIR3_metrics_json")
    if p_dir3:
        d = read_json(p_dir3)
        mt = d.get("metrics_test", {})
        # metrics_test: {yolo:{AP..}, rtdetr:{AP..}, wbf:{AP..}}
        best_k = None
        best_ap = -1.0
        for k in ["wbf", "rtdetr", "yolo"]:
            if isinstance(mt.get(k), dict):
                ap = mt[k].get("AP", None)
                try:
                    ap = float(ap)
                except Exception:
                    ap = None
                if ap is not None and ap > best_ap:
                    best_ap = ap
                    best_k = k
        if best_k:
            return best_k, {"source": str(p_dir3), "AP50-95": best_ap}

    # 2) dir1 models.csv
    p_dir1 = get_found_path(paths, "DIR1_metrics_csv")
    if p_dir1:
        import pandas as pd

        df = pd.read_csv(p_dir1)
        # try common columns
        ap_col = "AP50-95" if "AP50-95" in df.columns else ("AP" if "AP" in df.columns else None)
        if ap_col and "model" in df.columns:
            df["_ap"] = pd.to_numeric(df[ap_col], errors="coerce")
            row = df.sort_values("_ap", ascending=False).iloc[0]
            return str(row["model"]), {"source": str(p_dir1), "AP50-95": float(row["_ap"])}

    return "wbf", {"source": "fallback", "reason": "no metrics found in paths.json"}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--paths_json", default="reports/path_scan/paths.json")
    ap.add_argument("--images_dir", default="", help="Folder with demo images. If empty, auto-pick from paths.json.")
    ap.add_argument("--out_dir", default="artifacts/final_showcase")

    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    # WBF params
    ap.add_argument("--wbf_iou", type=float, default=0.55)
    ap.add_argument("--wbf_skip", type=float, default=0.001)
    ap.add_argument("--wbf_w1", type=float, default=1.0)
    ap.add_argument("--wbf_w2", type=float, default=1.0)

    # allow manual override for demo
    ap.add_argument("--force_best", default="", help="yolo|rtdetr|wbf|frcnn or empty for auto")

    args = ap.parse_args()

    paths_json = P(args.paths_json)
    if not paths_json.exists():
        raise FileNotFoundError(f"paths_json not found: {paths_json}")

    paths = read_json(paths_json)

    repo_root = get_found_path(paths, "ROOT_repo")
    if not repo_root:
        raise RuntimeError("ROOT_repo missing in paths.json")

    # resolve weights
    yolo_pt = get_found_path(paths, "DIR1_yolo_best_pt")
    rtdetr_pt = get_found_path(paths, "DIR1_rtdetr_best_pt")
    frcnn_pt = get_found_path(paths, "DIR1_detectron_frcnn_model_final")

    if not yolo_pt or not rtdetr_pt:
        raise RuntimeError(
            "Missing DIR1_yolo_best_pt or DIR1_rtdetr_best_pt in paths.json. "
            "Re-scan paths or update reports/path_scan/paths.json."
        )

    # choose images dir
    if args.images_dir:
        images_dir = P(args.images_dir)
    else:
        # best default: tiled images
        prep_tiled = get_found_path(paths, "SKU_prepared_small_v1_tiled")
        if prep_tiled and (prep_tiled / "images/tiles").exists():
            images_dir = prep_tiled / "images/tiles"
        else:
            # fallback: any known folder
            prep = get_found_path(paths, "SKU_prepared_small_v1")
            if prep and (prep / "images").exists():
                images_dir = prep / "images"
            else:
                raise RuntimeError(
                    "images_dir not provided and cannot auto-detect from paths.json. "
                    "Pass --images_dir explicitly."
                )

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir, limit=args.limit, seed=args.seed)
    if not imgs:
        raise FileNotFoundError(f"No images in {images_dir}")

    # pick best system based on reports
    best_system, best_meta = choose_best_system_from_paths(paths)
    if args.force_best.strip():
        best_system = args.force_best.strip().lower()
        best_meta = {"source": "force_best", "forced": True}

    # Load ultralytics models sequentially to reduce VRAM usage
    # Also: reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 1) YOLO preds
    t0 = time.time()
    yolo_model = YOLO(str(yolo_pt))
    yolo_preds: Dict[str, Tuple[List[List[float]], List[float]]] = {}

    for p in tqdm(imgs, desc="YOLO predict"):
        xyxy, sc = ultralytics_predict(
            yolo_model, p, args.imgsz, args.device, args.conf, iou=args.iou, half=True
        )
        yolo_preds[p.name] = (xyxy, sc)

    del yolo_model
    gc.collect()
    _cuda_cleanup()
    t_yolo = time.time() - t0

    # 2) RT-DETR preds
    t0 = time.time()
    rtdetr_model = YOLO(str(rtdetr_pt))
    rtdetr_preds: Dict[str, Tuple[List[List[float]], List[float]]] = {}

    for p in tqdm(imgs, desc="RT-DETR predict"):
        xyxy, sc = ultralytics_predict(
            rtdetr_model, p, args.imgsz, args.device, args.conf, iou=args.iou, half=True
        )
        rtdetr_preds[p.name] = (xyxy, sc)

    del rtdetr_model
    gc.collect()
    _cuda_cleanup()
    t_rtdetr = time.time() - t0

    # 3) Detectron2 (optional)
    d2_preds: Dict[str, Tuple[List[List[float]], List[float]]] = {}
    d2_used = False
    d2_error = None

    if frcnn_pt and D2_AVAILABLE:
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = str(frcnn_pt)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            predictor = DefaultPredictor(cfg)

            for p in tqdm(imgs, desc="Detectron2 predict"):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                out = predictor(img)
                inst = out["instances"].to("cpu")
                boxes = inst.pred_boxes.tensor.numpy().tolist() if inst.has("pred_boxes") else []
                scores = inst.scores.numpy().tolist() if inst.has("scores") else []
                d2_preds[p.name] = (boxes, scores)

            del predictor
            gc.collect()
            _cuda_cleanup()
            d2_used = True
        except Exception as e:
            d2_error = str(e)
            d2_used = False
            d2_preds = {}

    # 4) Save visualizations
    all_dir = out_dir / "all_models"
    all_dir.mkdir(parents=True, exist_ok=True)

    best_dir = out_dir / "best_demo"
    best_dir.mkdir(parents=True, exist_ok=True)

    def pick_best_variant_name(bs: str) -> str:
        bs = bs.lower()
        if "wbf" in bs:
            return "wbf"
        if "rtd" in bs or "detr" in bs:
            return "rtdetr"
        if "faster" in bs or "frcnn" in bs:
            return "frcnn"
        if "yolo" in bs:
            return "yolo"
        return "wbf"

    best_key = pick_best_variant_name(best_system)

    for p in tqdm(imgs, desc="Save visuals"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        y_xyxy, y_sc = yolo_preds.get(p.name, ([], []))
        d_xyxy, d_sc = rtdetr_preds.get(p.name, ([], []))

        wbf_xyxy, wbf_sc = wbf_merge(
            w,
            h,
            y_xyxy,
            y_sc,
            d_xyxy,
            d_sc,
            iou_thr=args.wbf_iou,
            skip_thr=args.wbf_skip,
            w1=args.wbf_w1,
            w2=args.wbf_w2,
        )

        out_y = draw_boxes(img, y_xyxy, y_sc, "YOLO")
        out_d = draw_boxes(img, d_xyxy, d_sc, "RTD")
        out_w = draw_boxes(img, wbf_xyxy, wbf_sc, "WBF")

        p_y = all_dir / f"{p.stem}__yolo.jpg"
        p_d = all_dir / f"{p.stem}__rtdetr.jpg"
        p_w = all_dir / f"{p.stem}__wbf.jpg"

        cv2.imwrite(str(p_y), out_y)
        cv2.imwrite(str(p_d), out_d)
        cv2.imwrite(str(p_w), out_w)

        p_f = None
        if d2_preds:
            f_xyxy, f_sc = d2_preds.get(p.name, ([], []))
            out_f = draw_boxes(img, f_xyxy, f_sc, "FRCNN")
            p_f = all_dir / f"{p.stem}__frcnn.jpg"
            cv2.imwrite(str(p_f), out_f)

        # copy best to best_demo
        src = {"yolo": p_y, "rtdetr": p_d, "wbf": p_w}.get(best_key, p_w)
        if best_key == "frcnn" and p_f is not None:
            src = p_f

        if src and src.exists():
            (best_dir / src.name).write_bytes(src.read_bytes())

    summary = {
        "best_system": best_system,
        "best_meta": best_meta,
        "picked_variant": best_key,
        "paths_json": str(paths_json),
        "resolved": {
            "repo_root": str(repo_root),
            "images_dir": str(images_dir),
            "yolo_pt": str(yolo_pt),
            "rtdetr_pt": str(rtdetr_pt),
            "frcnn_pt": str(frcnn_pt) if frcnn_pt else None,
        },
        "timing_seconds": {
            "images_count": len(imgs),
            "yolo_total": round(t_yolo, 3),
            "rtdetr_total": round(t_rtdetr, 3),
            "yolo_sec_per_image": round(t_yolo / max(1, len(imgs)), 4),
            "rtdetr_sec_per_image": round(t_rtdetr / max(1, len(imgs)), 4),
        },
        "detectron2": {"available": D2_AVAILABLE, "used": d2_used, "error": d2_error},
        "params": {
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "limit": args.limit,
            "seed": args.seed,
            "wbf": {"iou": args.wbf_iou, "skip": args.wbf_skip, "w1": args.wbf_w1, "w2": args.wbf_w2},
        },
        "outputs": {
            "out_dir": str(out_dir),
            "all_models": str(all_dir),
            "best_demo": str(best_dir),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE:", out_dir / "summary.json")


if __name__ == "__main__":
    main()