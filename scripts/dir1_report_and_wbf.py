from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import gc
import torch

# ultralytics
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is not installed or broken. pip install ultralytics") from e

# ensemble
try:
    from ensemble_boxes import weighted_boxes_fusion
except Exception as e:
    raise RuntimeError("ensemble-boxes is not installed. pip install ensemble-boxes") from e

def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_path(p: str) -> Path:
    """
    Позволяет передавать пути и в стиле Windows (D:\\...) и в стиле Linux (/mnt/d/...).
    """
    p = p.strip().strip('"').strip("'")
    # D:\xxx or C:\xxx  -> /mnt/d/xxx
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        tail = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{tail}")
    # D:/xxx -> /mnt/d/xxx (часто бывает)
    m2 = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m2:
        drive = m2.group(1).lower()
        tail = m2.group(2)
        return Path(f"/mnt/{drive}/{tail}")
    return Path(p).expanduser()


def parse_dataset_yaml(yaml_path: Path) -> dict:
    """
    Мини-парсер ultralytics dataset.yaml:
      path: ...
      train: ...
      val: ...
      test: ...
    """
    txt = yaml_path.read_text(encoding="utf-8").splitlines()
    d = {}
    for line in txt:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k in {"path", "train", "val", "test", "names", "nc"}:
            d[k] = v
    return d


def list_images_from_split(root_path: Path, split_value: str) -> List[Path]:
    """
    split_value может быть:
    - папкой (images/test)
    - файлом со списком картинок (.txt)
    """
    p = split_value
    # абсолютный или относительный путь
    p_norm = normalize_path(str(split_value))
    split_path = p_norm.resolve() if p_norm.is_absolute() else (root_path / p_norm).resolve()

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        imgs = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ip = (root_path / line).resolve() if not Path(line).is_absolute() else Path(line).resolve()
            imgs.append(ip)
        return imgs

    if split_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [p for p in split_path.rglob("*") if p.suffix.lower() in exts]
        imgs.sort()
        return imgs

    raise FileNotFoundError(f"Split path not found: {split_path}")


def image_to_label_path(img_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    """
    Делает label path по относительному пути картинки относительно images_dir.
    """
    rel = img_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def build_coco_gt_from_yolo(
    images_dir: Path,
    labels_dir: Path,
    image_paths: List[Path],
    out_json: Path,
    category_id: int = 1,
    category_name: str = "product",
) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for img_id, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue
        h, w = img.shape[:2]

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,  # важно: будем использовать images_root = images_dir
                "width": int(w),
                "height": int(h),
            }
        )

        lbl_path = image_to_label_path(img_path, images_dir, labels_dir)
        if not lbl_path.exists():
            # иногда бывают пустые/пропущенные
            continue

        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            # YOLO: cls xc yc bw bh (normalized)
            try:
                _, xc, yc, bw, bh = map(float, parts[:5])
            except Exception:
                continue

            x = (xc - bw / 2.0) * w
            y = (yc - bh / 2.0) * h
            bw_abs = bw * w
            bh_abs = bh * h

            x = max(0.0, x)
            y = max(0.0, y)
            bw_abs = max(0.0, min(bw_abs, w - x))
            bh_abs = max(0.0, min(bh_abs, h - y))
            area = bw_abs * bh_abs

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(bw_abs), float(bh_abs)],
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {"description": "Generated COCO GT from YOLO labels"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": category_name}],
    }

    out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] COCO GT saved: {out_json} (images={len(images)}, ann={len(annotations)})")
    return out_json


def coco_eval(gt_json: Path, pred_json: Path) -> dict:
    cocoGt = COCO(str(gt_json))
    cocoDt = cocoGt.loadRes(str(pred_json))
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "AP50-95": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AP_small": float(ev.stats[3]),
        "AP_medium": float(ev.stats[4]),
        "AP_large": float(ev.stats[5]),
    }


def chunked(xs: List[Path], n: int) -> Iterable[List[Path]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@dataclass
class UltralyticsPred:
    name: str
    pred_path: Path
    speed_ms: float
    metrics: dict


def predict_ultralytics_to_coco(
    model_path: Path,
    model_name: str,
    image_paths: List[Path],
    image_id_by_name: dict,
    out_pred_json: Path,
    imgsz: int,
    device: str,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    batch_chunk: int = 4,     # было 16 — уменьшаем
    speed_n: int = 50,        # было 200 — на 8GB лучше меньше
) -> UltralyticsPred:
    out_pred_json.parent.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        model = YOLO(str(model_path))

        # --- speed benchmark (без OOM) ---
        bench_imgs = image_paths[: min(speed_n, len(image_paths))]
        t0 = time.perf_counter()

        for pack in chunked(bench_imgs, batch_chunk):
            for _ in model.predict(
                source=[str(p) for p in pack],
                imgsz=imgsz,
                conf=0.25,
                iou=iou,
                max_det=max_det,
                device=device,
                stream=True,
                verbose=False,
                half=True,
                batch=1,  # ключевой момент
            ):
                pass

        dt = time.perf_counter() - t0
        speed_ms = 1000.0 * dt / max(1, len(bench_imgs))

        # --- inference full (без OOM) ---
        preds = []
        for pack in chunked(image_paths, batch_chunk):
            results = model.predict(
                source=[str(p) for p in pack],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
                half=True,
                batch=1,  # ключевой момент
                stream=False,
            )

            for img_path, r in zip(pack, results):
                img_id = image_id_by_name[img_path.name]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), sc in zip(xyxy, scores):
                    w = float(max(0.0, x2 - x1))
                    h = float(max(0.0, y2 - y1))
                    preds.append(
                        {
                            "image_id": int(img_id),
                            "category_id": 1,
                            "bbox": [float(x1), float(y1), w, h],
                            "score": float(sc),
                        }
                    )

        out_pred_json.write_text(json.dumps(preds), encoding="utf-8")
        metrics = coco_eval(gt_json=ARGS.gt_json, pred_json=out_pred_json)  # type: ignore[name-defined]
        print(f"[OK] {model_name} preds: {out_pred_json} | speed={speed_ms:.1f} ms/img | AP={metrics['AP50-95']:.4f}")

        return UltralyticsPred(name=model_name, pred_path=out_pred_json, speed_ms=speed_ms, metrics=metrics)

    finally:
        # важно: выгрузить модель и очистить кэш
        try:
            del model
        except Exception:
            pass
        cuda_cleanup()

def eval_detectron2(
    d2_weights: Path,
    gt_json: Path,
    images_root: Path,
    out_dir: Path,
    speed_n: int = 100,
) -> Tuple[dict, float, Path]:
    """
    Запускает COCOEvaluator в detectron2 и возвращает:
    - metrics (AP etc, в диапазоне 0..1)
    - speed_ms (примерная, на predictor)
    - pred_json_path (куда evaluator пишет coco_instances_results.json)
    """
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
    except Exception as e:
        raise RuntimeError("Detectron2 is not installed in this environment.") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # register dataset
    ds_name = "shelf_test_eval"
    register_coco_instances(ds_name, {}, str(gt_json), str(images_root))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(d2_weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = str(out_dir)

    # eval
    evaluator = COCOEvaluator(ds_name, cfg, False, output_dir=str(out_dir))
    loader = build_detection_test_loader(cfg, ds_name)
    res = inference_on_dataset(DefaultPredictor(cfg).model, loader, evaluator)

    # detectron2 возвращает 0..100, приводим к 0..1
    ap = res.get("bbox", {}).get("AP", None)
    ap50 = res.get("bbox", {}).get("AP50", None)
    ap75 = res.get("bbox", {}).get("AP75", None)
    aps = res.get("bbox", {}).get("APs", None)
    apm = res.get("bbox", {}).get("APm", None)
    apl = res.get("bbox", {}).get("APl", None)

    def norm(v):
        if v is None:
            return float("nan")
        v = float(v)
        return v / 100.0 if v > 1.0 else v

    metrics = {
        "AP50-95": norm(ap),
        "AP50": norm(ap50),
        "AP75": norm(ap75),
        "AP_small": norm(aps),
        "AP_medium": norm(apm),
        "AP_large": norm(apl),
    }

    # speed benchmark (predictor)
    predictor = DefaultPredictor(cfg)
    img_paths = sorted([p for p in images_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    bench = img_paths[: min(speed_n, len(img_paths))]
    t0 = time.perf_counter()
    for p in bench:
        img = cv2.imread(str(p))
        _ = predictor(img)
    dt = time.perf_counter() - t0
    speed_ms = 1000.0 * dt / max(1, len(bench))

    pred_json_path = out_dir / "inference" / "coco_instances_results.json"
    print(f"[OK] Detectron2 eval | speed={speed_ms:.1f} ms/img | AP={metrics['AP50-95']:.4f}")
    return metrics, speed_ms, pred_json_path


def wbf_fuse(
    gt_json: Path,
    pred_a: Path,
    pred_b: Path,
    out_pred: Path,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> dict:
    coco = COCO(str(gt_json))
    imgs = {im["id"]: im for im in coco.dataset["images"]}

    pa = json.loads(pred_a.read_text(encoding="utf-8"))
    pb = json.loads(pred_b.read_text(encoding="utf-8"))

    by_img_a = defaultdict(list)
    by_img_b = defaultdict(list)
    for p in pa:
        by_img_a[p["image_id"]].append(p)
    for p in pb:
        by_img_b[p["image_id"]].append(p)

    fused = []
    for img_id, im in imgs.items():
        W, H = im["width"], im["height"]

        def to_norm_xyxy(plist):
            boxes, scores, labels = [], [], []
            for p in plist:
                x, y, w, h = p["bbox"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
                scores.append(p["score"])
                labels.append(0)
            return boxes, scores, labels

        b1, s1, l1 = to_norm_xyxy(by_img_a.get(img_id, []))
        b2, s2, l2 = to_norm_xyxy(by_img_b.get(img_id, []))
        if len(b1) + len(b2) == 0:
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            [b1, b2],
            [s1, s2],
            [l1, l2],
            weights=list(weights),
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        for bb, sc in zip(boxes, scores):
            x1, y1, x2, y2 = bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H
            fused.append(
                {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                }
            )

    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text(json.dumps(fused), encoding="utf-8")
    metrics = coco_eval(gt_json, out_pred)
    print(f"[OK] WBF saved: {out_pred} | AP={metrics['AP50-95']:.4f}")
    return metrics


def plot_reports(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar: AP50-95
    plt.figure()
    plt.bar(df["model"], df["AP50-95"])
    plt.title("COCO AP (AP50-95)")
    plt.ylabel("AP")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ap5095_bar.png", dpi=160)
    plt.close()

    # Bar: AP50
    plt.figure()
    plt.bar(df["model"], df["AP50"])
    plt.title("COCO AP50")
    plt.ylabel("AP50")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ap50_bar.png", dpi=160)
    plt.close()

    # Grouped: AP_small/medium/large
    plt.figure()
    x = range(len(df))
    w = 0.25
    plt.bar([i - w for i in x], df["AP_small"], width=w, label="AP_small")
    plt.bar([i for i in x], df["AP_medium"], width=w, label="AP_medium")
    plt.bar([i + w for i in x], df["AP_large"], width=w, label="AP_large")
    plt.xticks(list(x), df["model"], rotation=20, ha="right")
    plt.title("AP by object size")
    plt.ylabel("AP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ap_sizes.png", dpi=160)
    plt.close()

    # Scatter: quality vs speed
    plt.figure()
    plt.scatter(df["ms_per_image"], df["AP50-95"])
    for _, r in df.iterrows():
        plt.annotate(r["model"], (r["ms_per_image"], r["AP50-95"]))
    plt.title("Quality vs Speed")
    plt.xlabel("ms / image (approx)")
    plt.ylabel("AP50-95")
    plt.tight_layout()
    plt.savefig(out_dir / "quality_vs_speed.png", dpi=160)
    plt.close()


# argparse
ap = argparse.ArgumentParser()
ap.add_argument("--dataset_yaml", required=True, help="Ultralytics dataset.yaml")
ap.add_argument("--yolo_pt", required=True, help="YOLO best.pt")
ap.add_argument("--rtdetr_pt", required=True, help="RT-DETR best.pt")
ap.add_argument("--d2_weights", required=True, help="Detectron2 model_final.pth")
ap.add_argument("--out_dir", required=True, help="Where to save reports")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="0")
ap.add_argument("--speed_n", type=int, default=200)
ap.add_argument("--wbf_iou", type=float, default=0.55)
ap.add_argument("--wbf_skip", type=float, default=0.001)
ARGS = ap.parse_args()

# normalize paths
dataset_yaml = normalize_path(ARGS.dataset_yaml).resolve()
yolo_pt = normalize_path(ARGS.yolo_pt)
rtdetr_pt = normalize_path(ARGS.rtdetr_pt)
d2_weights = normalize_path(ARGS.d2_weights)
out_dir = normalize_path(ARGS.out_dir)

ds = parse_dataset_yaml(dataset_yaml)
root = normalize_path(ds.get("path", str(dataset_yaml.parent))).resolve()
split = ds.get("test") or ds.get("val")
if split is None:
    raise RuntimeError("dataset.yaml must have at least 'val:' (and ideally 'test:')")

images = list_images_from_split(root, split)
if not images:
    raise RuntimeError("No images found in split")

# define images_dir and labels_dir (типичная структура YOLO: images/<split>, labels/<split>)
# если split указывает на images/test, тогда labels лежат по аналогии
split_path = (root / split).resolve() if not Path(split).is_absolute() else Path(split).resolve()
if split_path.is_dir():
    images_dir = split_path
else:
    # если split = txt со списком, берём папку по первой картинке
    images_dir = images[0].parent

# попытка вывести labels_dir из images_dir
# images/... -> labels/...
parts = list(images_dir.parts)
if "images" in parts:
    idx = parts.index("images")
    labels_dir = Path(*parts[:idx], "labels", *parts[idx + 1 :])
else:
    # fallback: соседняя папка labels рядом с images_dir
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name

# build COCO GT
gt_json = out_dir / "gt_test_coco.json"
build_coco_gt_from_yolo(images_dir=images_dir, labels_dir=labels_dir, image_paths=images, out_json=gt_json)

# image_id mapping by filename
gt = json.loads(gt_json.read_text(encoding="utf-8"))
image_id_by_name = {im["file_name"]: im["id"] for im in gt["images"]}

# predict ultralytics models -> coco preds
pred_yolo = out_dir / "pred_yolov8s.json"
pred_rtdetr = out_dir / "pred_rtdetr_l.json"

# hack: make gt_json visible inside predict_ultralytics_to_coco
ARGS.gt_json = gt_json  # type: ignore[attr-defined]

yolo_res = predict_ultralytics_to_coco(
    model_path=yolo_pt,
    model_name="YOLOv8s",
    image_paths=images,
    image_id_by_name=image_id_by_name,
    out_pred_json=pred_yolo,
    imgsz=ARGS.imgsz,
    device=ARGS.device,
    speed_n=ARGS.speed_n,
)

cuda_cleanup()

rtdetr_res = predict_ultralytics_to_coco(
    model_path=rtdetr_pt,
    model_name="RT-DETR-L",
    image_paths=images,
    image_id_by_name=image_id_by_name,
    out_pred_json=pred_rtdetr,
    imgsz=ARGS.imgsz,
    device=ARGS.device,
    speed_n=ARGS.speed_n,
)

# detectron2 eval on the same GT/images_root
d2_out = out_dir / "detectron2_eval"
d2_metrics, d2_speed, d2_pred_path = eval_detectron2(
    d2_weights=d2_weights,
    gt_json=gt_json,
    images_root=images_dir,  # file_name в GT = basename, значит root = images_dir
    out_dir=d2_out,
    speed_n=min(100, ARGS.speed_n),
)

# WBF ensemble (YOLO + RTDETR)
pred_wbf = out_dir / "pred_wbf_yolo_rtdetr.json"
wbf_metrics = wbf_fuse(
    gt_json=gt_json,
    pred_a=pred_yolo,
    pred_b=pred_rtdetr,
    out_pred=pred_wbf,
    iou_thr=ARGS.wbf_iou,
    skip_box_thr=ARGS.wbf_skip,
)

# summary table
rows = [
    {"model": yolo_res.name, **yolo_res.metrics, "ms_per_image": yolo_res.speed_ms},
    {"model": rtdetr_res.name, **rtdetr_res.metrics, "ms_per_image": rtdetr_res.speed_ms},
    {"model": "Faster R-CNN (D2)", **d2_metrics, "ms_per_image": d2_speed},
    {"model": "WBF(YOLO+RTDETR)", **wbf_metrics, "ms_per_image": float("nan")},
]
df = pd.DataFrame(rows)

out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "metrics.csv", index=False)
df.to_markdown(out_dir / "metrics.md", index=False)

plot_reports(df, out_dir)

print("\n=== DONE ===")
print("Saved:", out_dir)
print("Table:", out_dir / "metrics.csv")
print("Plots:", out_dir / "ap5095_bar.png", out_dir / "quality_vs_speed.png")