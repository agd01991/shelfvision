from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-Seg model for ShelfVision")
    parser.add_argument("--data", required=True, help="Путь к YOLO-Seg dataset.yaml")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="Базовая YOLO-Seg модель или путь к весам")
    parser.add_argument("--project", default="runs/yolo_seg", help="Папка Ultralytics project")
    parser.add_argument("--name", default="d2s_small_yolov8s_seg", help="Название эксперимента")
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--workers", type=int, default=4, help="Количество workers DataLoader")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден dataset.yaml: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
    )

    save_dir: Optional[Path] = getattr(results, "save_dir", None)
    print("=== ShelfVision YOLO-Seg training ===")
    print(f"Dataset: {data_path}")
    print(f"Model:   {args.model}")
    if save_dir:
        print(f"Saved:   {save_dir}")
        print(f"Best:    {Path(save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
