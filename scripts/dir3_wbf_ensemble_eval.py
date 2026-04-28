from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _predict_ultralytics_batch(
    model: YOLO,
    paths: List[str],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    batch: int,
) -> List[Tuple[List[List[float]], List[float], List[int]]]:
    """
    Возвращает по каждому изображению:
      boxes_xyxy (list of [x1,y1,x2,y2]), scores, labels
    """
    results = model.predict(
        source=paths,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
        stream=False,
        half=True,
        batch=batch,
    )

    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(([], [], []))
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        boxes = xyxy.tolist()
        scores = confs.tolist()
        labels = clss.tolist()
        out.append((boxes, scores, labels))

    return out


def _norm_xyxy(boxes: List[List[float]], w: int, h: int) -> List[List[float]]:
    nb = []
    for x1, y1, x2, y2 in boxes:
        nb.append([x1 / w, y1 / h, x2 / w, y2 / h])
    return nb


def _denorm_to_xywh(box: List[float], w: int, h: int) -> List[float]:
    x1 = box[0] * w
    y1 = box[1] * h
    x2 = box[2] * w
    y2 = box[3] * h
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(bw), float(bh)]


def coco_eval_bbox(gt_json: str, pred_json_path: str) -> Dict[str, float]:
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    return {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AR1": float(ev.stats[6]),
        "AR10": float(ev.stats[7]),
        "AR100": float(ev.stats[8]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True, help="COCO test.json (v1_tiled)")
    ap.add_argument("--images_dir", required=True, help="tiles dir")
    ap.add_argument("--yolo_weights", required=True)
    ap.add_argument("--detr_weights", required=True)

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")

    ap.add_argument("--conf", type=float, default=0.001, help="predict conf threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="predict NMS iou for ultralytics models")
    ap.add_argument("--pred_batch", type=int, default=8)

    ap.add_argument("--wbf_iou_thr", type=float, default=0.55)
    ap.add_argument("--wbf_skip_thr", type=float, default=0.001)
    ap.add_argument("--wbf_w1", type=float, default=1.0)
    ap.add_argument("--wbf_w2", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0, help="0 = all test images, else first N")
    ap.add_argument("--out_dir", default="artifacts/dir3_wbf")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_json = str(Path(args.test_json).resolve())
    images_dir = Path(args.images_dir).resolve()

    coco = COCO(test_json)
    img_ids = coco.getImgIds()
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    # category id (в COCO у тебя обычно 1 категория product)
    cat_ids = coco.getCatIds()
    if len(cat_ids) != 1:
        raise RuntimeError(f"Expected 1 category in COCO, got {cat_ids}")
    cat_id = int(cat_ids[0])

    # Собираем список путей + размеров
    items = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        fn = info["file_name"]
        p = images_dir / fn
        if not p.exists():
            # fallback: только basename
            p2 = images_dir / Path(fn).name
            if not p2.exists():
                continue
            p = p2
        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        if w <= 0 or h <= 0:
            # последний fallback: читаем через cv2
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
        items.append((int(img_id), str(p), w, h))

    if not items:
        raise RuntimeError("No test images resolved. Check images_dir vs file_name paths.")

    # Модели
    yolo = YOLO(args.yolo_weights)
    detr = YOLO(args.detr_weights)

    # --- 1) Predictions YOLO and RT-DETR on TEST (batched) ---
    paths = [p for _, p, _, _ in items]

    t0 = time.time()
    yolo_out = _predict_ultralytics_batch(yolo, paths, args.imgsz, args.device, args.conf, args.iou, args.pred_batch)
    t_yolo = time.time() - t0

    t0 = time.time()
    detr_out = _predict_ultralytics_batch(detr, paths, args.imgsz, args.device, args.conf, args.iou, args.pred_batch)
    t_detr = time.time() - t0

    # --- 2) Build COCO predictions for YOLO, RT-DETR and WBF ensemble ---
    preds_yolo = []
    preds_detr = []
    preds_wbf = []

    for (img_id, _, w, h), (b1, s1, l1), (b2, s2, l2) in zip(items, yolo_out, detr_out):
        # оставляем только класс 0 (Ultralytics), т.к. у нас 1 класс
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

        # YOLO-only preds
        for bb, sc in zip(b1, s1):
            x1, y1, x2, y2 = bb
            preds_yolo.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(sc),
            })

        # RT-DETR-only preds
        for bb, sc in zip(b2, s2):
            x1, y1, x2, y2 = bb
            preds_detr.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(sc),
            })

        # WBF ensemble
        if not b1 and not b2:
            continue

        boxes_list = [_norm_xyxy(b1, w, h), _norm_xyxy(b2, w, h)]
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
                "bbox": _denorm_to_xywh(bb, w, h),
                "score": float(sc),
            })

    # --- 3) Save predictions ---
    p_yolo = out_dir / "pred_yolo.json"
    p_detr = out_dir / "pred_rtdetr.json"
    p_wbf = out_dir / "pred_wbf.json"

    p_yolo.write_text(json.dumps(preds_yolo, ensure_ascii=False), encoding="utf-8")
    p_detr.write_text(json.dumps(preds_detr, ensure_ascii=False), encoding="utf-8")
    p_wbf.write_text(json.dumps(preds_wbf, ensure_ascii=False), encoding="utf-8")

    # --- 4) COCO eval ---
    m_yolo = coco_eval_bbox(test_json, str(p_yolo))
    m_detr = coco_eval_bbox(test_json, str(p_detr))
    m_wbf = coco_eval_bbox(test_json, str(p_wbf))

    report = {
        "name": "dir3_wbf",
        "test_json": test_json,
        "images_dir": str(images_dir),
        "imgsz": args.imgsz,
        "device": args.device,
        "pred_batch": args.pred_batch,
        "conf": args.conf,
        "iou": args.iou,
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
        "metrics_test": {
            "yolo": m_yolo,
            "rtdetr": m_detr,
            "wbf": m_wbf,
        },
        "predictions": {
            "yolo": str(p_yolo),
            "rtdetr": str(p_detr),
            "wbf": str(p_wbf),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()