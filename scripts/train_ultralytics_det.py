from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


def read_last_metrics(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {"status": "no_results.csv"}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty_results.csv"}
    last = rows[-1]

    def pick(*keys):
        for k in keys:
            if k in last and last[k] != "":
                try:
                    return float(last[k])
                except Exception:
                    return last[k]
        return ""

    return {
        "status": "ok",
        "epoch_last": pick("epoch"),
        "P": pick("metrics/precision(B)", "metrics/precision"),
        "R": pick("metrics/recall(B)", "metrics/recall"),
        "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50-95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_path = cfg["model"]
    data_yaml = cfg["data"]
    name = cfg["name"]
    project = cfg.get("project", "runs/dir1")

    train_kwargs = dict(
        data=data_yaml,
        imgsz=int(cfg.get("imgsz", 640)),
        epochs=int(cfg.get("epochs", 30)),
        batch=int(cfg.get("batch", 4)),
        device=str(cfg.get("device", "0")),
        workers=int(cfg.get("workers", 2)),
        project=str(project),
        name=str(name),
        seed=int(cfg.get("seed", 42)),
        plots=bool(cfg.get("plots", False)),
        verbose=bool(cfg.get("verbose", False)),
        save_period=int(cfg.get("save_period", 1)),
    )

    # дополнительные параметры (если есть)
    for k in ("lr0", "weight_decay", "optimizer", "mosaic", "mixup", "close_mosaic"):
        if k in cfg:
            train_kwargs[k] = cfg[k]

    t0 = time.time()
    model = YOLO(model_path)
    res = model.train(**train_kwargs)
    dt = time.time() - t0

    save_dir = Path(res.save_dir)
    metrics = read_last_metrics(save_dir / "results.csv")
    metrics.update(
        {
            "model": model_path,
            "data": data_yaml,
            "seconds_total": round(dt, 1),
            "save_dir": str(save_dir),
            "config": str(Path(args.config).resolve()),
        }
    )
    (save_dir / "metrics_last.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE:", save_dir)


if __name__ == "__main__":
    main()
