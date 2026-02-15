# src/models/yolo/train.py
from pathlib import Path
from src.core.config import ensure_dir, load_yaml
from src.core.seed import seed_all

def train_yolo(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    seed_all(int(cfg.get("seed", 42)))

    # Импортируется внутри, чтобы проект не падал при чтении без ultralytics
    from ultralytics import YOLO

    run_dir = ensure_dir(cfg["run_dir"])
    model = YOLO(cfg["model"])
    results = model.train(
        data=cfg["data_yaml"],
        imgsz=int(cfg["imgsz"]),
        epochs=int(cfg["epochs"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        weight_decay=float(cfg["weight_decay"]),
        device=cfg.get("device", "0"),
        project=str(run_dir),
        name=cfg.get("name", "yolo_run"),
    )
    # ultralytics сам создаёт папку; возвращается путь к ней
    return Path(results.save_dir)
