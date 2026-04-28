from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# чтобы `from src...` работало при запуске скрипта
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ensure_dir, load_yaml, read_json, write_json
from src.core.logging import setup_logging
from src.core.seed import seed_all
from src.data.coco_schema import CocoAnnotation, CocoCategory, CocoImage, load_coco


def link_or_copy(src: Path, dst: Path) -> None:
    """Создать hardlink (быстро, без копий), если нельзя — копировать."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # работает, если один диск/том
    except OSError:
        shutil.copy2(src, dst)


def resolve_image_root(prepared_dir: Path) -> Path:
    """Определить, откуда брать изображения (raw или tiles)."""
    passport_path = prepared_dir / "passport.json"
    if not passport_path.exists():
        raise FileNotFoundError(f"passport.json not found: {passport_path}")

    passport = read_json(passport_path)

    # Если был тайлинг в prepared — брать тайлы
    tiling = passport.get("tiling", None)
    if isinstance(tiling, dict) and tiling.get("enabled", False):
        # В passport хранится строка пути, обычно вида "data/prepared/.../images/tiles"
        p = Path(tiling.get("images_dir", ""))
        if p.is_absolute():
            return p
        return (ROOT / p).resolve()

    # Иначе брать исходные изображения из source
    source = passport.get("source", {})
    img_root = source.get("image_root") or source.get("images_dir")
    if not img_root:
        raise KeyError("passport.source has no 'image_root' or 'images_dir'")
    p = Path(img_root)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def coco_to_yolo_dataset(
    prepared_dir: Path,
    yolo_dir: Path,
    rebuild: bool,
    logger,
) -> Path:
    """Собирает YOLO-датасет из prepared COCO (bbox) + splits."""
    prepared_dir = prepared_dir.resolve()
    yolo_dir = yolo_dir.resolve()

    ann_path = prepared_dir / "annotations.json"
    splits_path = prepared_dir / "splits.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {ann_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")

    if rebuild and yolo_dir.exists():
        shutil.rmtree(yolo_dir, ignore_errors=True)

    ensure_dir(yolo_dir)
    ensure_dir(yolo_dir / "images")
    ensure_dir(yolo_dir / "labels")

    image_root = resolve_image_root(prepared_dir)
    logger.info(f"[train_yolo] image_root = {image_root}")

    coco_obj = read_json(ann_path)
    images, anns, cats = load_coco(coco_obj)
    splits = read_json(splits_path)

    # category_id -> yolo_class_idx (0..nc-1)
    cats_sorted = sorted(cats, key=lambda c: c.id)
    cat_id_to_idx = {c.id: i for i, c in enumerate(cats_sorted)}
    names = [c.name for c in cats_sorted]
    nc = len(names) if names else 1

    # annotations by image_id
    anns_by_img: Dict[int, List[CocoAnnotation]] = {}
    for a in anns:
        anns_by_img.setdefault(a.image_id, []).append(a)

    img_map: Dict[int, CocoImage] = {im.id: im for im in images}

    def process_split(split_name: str) -> int:
        ids: List[int] = splits.get(split_name, [])
        count = 0
        for image_id in ids:
            image_id = int(image_id)
            im = img_map.get(image_id)
            if not im:
                continue

            rel = Path(im.file_name)
            src_img = image_root / rel
            if not src_img.exists():
                # fallback: иногда file_name только basename
                src_img = image_root / rel.name
                if not src_img.exists():
                    logger.warning(f"[train_yolo] missing image file: {im.file_name}")
                    continue

            dst_img = yolo_dir / "images" / split_name / rel
            dst_lbl = (yolo_dir / "labels" / split_name / rel).with_suffix(".txt")

            link_or_copy(src_img, dst_img)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

            # YOLO label lines
            lines: List[str] = []
            W = float(im.width)
            H = float(im.height)
            for a in anns_by_img.get(image_id, []):
                # bbox = [x,y,w,h]
                x, y, w, h = a.bbox
                if w <= 0 or h <= 0 or W <= 0 or H <= 0:
                    continue

                cls = cat_id_to_idx.get(int(a.category_id), None)
                if cls is None:
                    continue

                xc = (x + w / 2.0) / W
                yc = (y + h / 2.0) / H
                wn = w / W
                hn = h / H

                # чуть страхуемся
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    continue
                if wn <= 0 or hn <= 0:
                    continue

                lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            dst_lbl.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            count += 1
        return count

    n_train = process_split("train")
    n_val = process_split("val")
    n_test = process_split("test")

    logger.info(
        f"[train_yolo] yolo images: train={n_train}, val={n_val}, test={n_test}"
    )

    # dataset.yaml for Ultralytics
    dataset_yaml = yolo_dir / "dataset.yaml"
    data = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": int(nc),
        "names": names if names else ["object"],
    }
    write_json(
        yolo_dir / "dataset.meta.json",
        {"prepared_dir": str(prepared_dir), "image_root": str(image_root), **data},
    )
    # Ultralytics читает YAML, поэтому пишем YAML через простую строковую сборку
    yaml_text = (
        f"path: {data['path']}\n"
        f"train: {data['train']}\n"
        f"val: {data['val']}\n"
        f"test: {data['test']}\n"
        f"nc: {data['nc']}\n"
        f"names: {data['names']}\n"
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")

    return dataset_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo_det.yaml")
    ap.add_argument(
        "--only_build", action="store_true", help="Только собрать YOLO dataset и выйти"
    )
    args = ap.parse_args()

    logger = setup_logging()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    seed_all(seed)

    prepared_dir = Path(cfg["prepared_dir"])
    yolo_dir = Path(cfg["yolo_dataset_dir"])
    rebuild = bool(cfg.get("rebuild_yolo_dataset", True))

    dataset_yaml = coco_to_yolo_dataset(
        prepared_dir, yolo_dir, rebuild=rebuild, logger=logger
    )

    if args.only_build:
        logger.info(f"[train_yolo] only_build=1, dataset saved at: {dataset_yaml}")
        return

    # тренировка
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Не найден ultralytics. Установи: pip install ultralytics\n"
            "Если будут проблемы с torch на Windows, ставь torch отдельно."
        ) from e

    model_name = str(cfg.get("model", "yolov8n.pt"))
    model = YOLO(model_name)

    run_dir = ensure_dir(cfg.get("run_dir", "runs/yolo_det"))
    name = str(cfg.get("name", "yolo_run"))

    logger.info(f"[train_yolo] start training: model={model_name}, data={dataset_yaml}")

    results = model.train(
        data=str(dataset_yaml),
        imgsz=int(cfg.get("imgsz", 1024)),
        epochs=int(cfg.get("epochs", 20)),
        batch=int(cfg.get("batch", 8)),
        lr0=float(cfg.get("lr0", 0.01)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        device=str(cfg.get("device", "0")),
        project=str(run_dir),
        name=name,
    )

    logger.info(f"[train_yolo] done. save_dir={results.save_dir}")


if __name__ == "__main__":
    main()
