from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

# FULL dataset (собран скриптом build_sku110k_full_yolo.py)
DATASET_DIR = (ROOT / "data" / "yolo_cache" / "sku110k_full").resolve()
DATA_YAML = DATASET_DIR / "dataset.yaml"

RUNS_LOCAL = (ROOT / "runs" / "yolo_full").resolve()
BACKUP_DIR = (ROOT / "runs_full_backup").resolve()
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = BACKUP_DIR / "state.json"
SUMMARY_PATH = BACKUP_DIR / "summary.csv"

DEVICE = "0"
SEED = 42

# FULL датасет очень тяжёлый. Для ночных прогонов ставь меньше эпох.
BASE = dict(
    data=str(DATA_YAML),
    imgsz=1024,
    epochs=30,  # <-- для полного лучше начать с 10, потом увеличивать
    batch=4,  # 8GB VRAM: безопасно 4 при 1024
    device=DEVICE,
    workers=2,
    project=str(RUNS_LOCAL),
    seed=SEED,
    plots=False,
    verbose=False,
    save_period=1,
)

BASE_MODEL = "yolov8s.pt"
BASE_NAME = "FULL_BASE_yolov8s_img1024_e10_b4"

EXPERIMENTS = [
    dict(name="FULL_E01_model_yolov8n", model="yolov8n.pt", override={}),
    dict(name="FULL_E02_model_yolov8m", model="yolov8m.pt", override={}),
    dict(name="FULL_E03_imgsz_640", model=BASE_MODEL, override={"imgsz": 640}),
    dict(name="FULL_E04_imgsz_1280", model=BASE_MODEL, override={"imgsz": 1280}),
    dict(name="FULL_E05_lr0_0p005", model=BASE_MODEL, override={"lr0": 0.005}),
    dict(name="FULL_E06_wd_0p001", model=BASE_MODEL, override={"weight_decay": 0.001}),
    dict(
        name="FULL_E07_optimizer_AdamW",
        model=BASE_MODEL,
        override={"optimizer": "AdamW"},
    ),
    dict(name="FULL_E08_mosaic_0", model=BASE_MODEL, override={"mosaic": 0.0}),
    dict(name="FULL_E09_mixup_0p2", model=BASE_MODEL, override={"mixup": 0.2}),
    dict(
        name="FULL_E10_close_mosaic_10", model=BASE_MODEL, override={"close_mosaic": 10}
    ),
]


def patch_dataset_yaml(path_yaml: Path, dataset_dir: Path) -> None:
    """Правит строку `path:` в dataset.yaml → абсолютный путь до dataset_dir."""
    if not path_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path_yaml}")
    lines = path_yaml.read_text(encoding="utf-8").splitlines()
    fixed = []
    for line in lines:
        if line.strip().startswith("path:"):
            fixed.append(f"path: {dataset_dir.as_posix()}")
        else:
            fixed.append(line)
    path_yaml.write_text("\n".join(fixed) + "\n", encoding="utf-8")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"done": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def safe_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


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


def append_summary(row: dict) -> None:
    write_header = not SUMMARY_PATH.exists()
    fieldnames = sorted(row.keys())
    with SUMMARY_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_one(exp_name: str, model_path: str, override: dict) -> None:
    print("\n" + "=" * 90)
    print("RUN:", exp_name, "| model:", model_path, "| override:", override)

    model = YOLO(model_path)
    kwargs = dict(BASE)
    kwargs.update(override)

    t0 = time.time()
    res = model.train(**kwargs, name=exp_name)
    dt = time.time() - t0

    save_dir = Path(res.save_dir)
    dst_full = BACKUP_DIR / f"{exp_name}__full"
    safe_copytree(save_dir, dst_full)

    met = read_last_metrics(dst_full / "results.csv")
    met.update(
        {
            "exp": exp_name,
            "model": model_path,
            "override": str(override),
            "seconds_total": round(dt, 1),
            "save_dir_full": str(dst_full),
        }
    )
    append_summary(met)

    print("SAVED:", dst_full)
    print("SUMMARY:", SUMMARY_PATH)


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"DATASET_DIR not found: {DATASET_DIR}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {DATA_YAML}")

    patch_dataset_yaml(DATA_YAML, DATASET_DIR)
    print("dataset.yaml patched:", DATA_YAML)

    state = load_state()
    done = set(state.get("done", []))

    # BASE (можешь комментировать как раньше)
    if BASE_NAME not in done:
        try:
            run_one(BASE_NAME, BASE_MODEL, {})
            done.add(BASE_NAME)
            state["done"] = sorted(done)
            save_state(state)
        except Exception as e:
            print("BASE FAILED:", repr(e))
            traceback.print_exc()
            append_summary(
                {
                    "exp": BASE_NAME,
                    "status": f"failed: {type(e).__name__}",
                    "error": repr(e),
                }
            )
            save_state({"done": sorted(done)})

    for exp in EXPERIMENTS:
        if exp["name"] in done:
            print("SKIP (already done):", exp["name"])
            continue
        try:
            run_one(exp["name"], exp["model"], exp["override"])
            done.add(exp["name"])
            state["done"] = sorted(done)
            save_state(state)
        except Exception as e:
            print("FAILED:", exp["name"], "->", repr(e))
            traceback.print_exc()
            append_summary(
                {
                    "exp": exp["name"],
                    "status": f"failed: {type(e).__name__}",
                    "error": repr(e),
                }
            )
            save_state({"done": sorted(done)})
            continue

    print("\nDONE")
    print("Backup dir:", BACKUP_DIR)
    print("State:", STATE_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
