from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class BuildStats:
    images: int = 0
    labels_files: int = 0
    anns: int = 0
    clipped: int = 0
    dropped: int = 0
    missing_images: int = 0


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # быстро, если один диск
    except OSError:
        shutil.copy2(src, dst)


def clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[float, float, float, float, bool]:
    cx1 = max(0.0, min(float(W), x1))
    cy1 = max(0.0, min(float(H), y1))
    cx2 = max(0.0, min(float(W), x2))
    cy2 = max(0.0, min(float(H), y2))
    changed = (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2)
    return cx1, cy1, cx2, cy2, changed


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc / W, yc / H, w / W, h / H


def read_rows_no_header(csv_path: Path):
    # формат строк в SKU110K_fixed: image,x1,y1,x2,y2,class,W,H (без заголовка)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            yield row


def build_split(
    split_name: str,
    csv_path: Path,
    images_dir: Path,
    out_dir: Path,
    min_area: float,
    clip_to_image: bool,
    cls_id: int,
) -> BuildStats:
    stats = BuildStats()

    labels_dir = out_dir / "labels" / split_name
    imgs_out_dir = out_dir / "images" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs_out_dir.mkdir(parents=True, exist_ok=True)

    # собираем аннотации в память по картинке (для одного split это ок)
    by_img: Dict[str, List[Tuple[float, float, float, float, int, int]]] = {}

    for row in read_rows_no_header(csv_path):
        # row: [image, x1,y1,x2,y2, class, W, H]
        img = row[0].strip()
        x1 = float(row[1])
        y1 = float(row[2])
        x2 = float(row[3])
        y2 = float(row[4])
        W = int(float(row[6]))
        H = int(float(row[7]))
        by_img.setdefault(img, []).append((x1, y1, x2, y2, W, H))
        stats.anns += 1

    stats.images = len(by_img)

    for rel, boxes in by_img.items():
        src_img = images_dir / rel
        if not src_img.exists():
            # fallback: иногда в csv только имя
            src_img = images_dir / Path(rel).name
            if not src_img.exists():
                stats.missing_images += 1
                continue

        # копируем/линкуем картинку
        dst_img = imgs_out_dir / rel
        link_or_copy(src_img, dst_img)

        # пишем label file
        dst_lbl = (labels_dir / rel).with_suffix(".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for x1, y1, x2, y2, W, H in boxes:
            changed = False
            if clip_to_image:
                x1, y1, x2, y2, changed = clip_xyxy(x1, y1, x2, y2, W, H)
                if changed:
                    stats.clipped += 1

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_area:
                stats.dropped += 1
                continue

            xc, yc, wn, hn = xyxy_to_yolo(x1, y1, x2, y2, W, H)
            if wn <= 0 or hn <= 0:
                stats.dropped += 1
                continue
            # один класс "product"
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats.labels_files += 1

    return stats


def write_dataset_yaml(out_dir: Path, nc: int, names: List[str]) -> None:
    # Важно: path лучше делать "." — тогда не нужно патчить при переносе
    yaml_text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="data/raw/sku110k_full/sku110k")
    ap.add_argument(
        "--out_dir",
        default="data/yolo_cache/sku110k_full",
        help="куда собрать YOLO dataset",
    )
    ap.add_argument("--images_dir", default="images")
    ap.add_argument("--ann_dir", default="annotations")
    ap.add_argument("--train_csv", default="annotations_train.csv")
    ap.add_argument("--val_csv", default="annotations_val.csv")
    ap.add_argument("--test_csv", default="annotations_test.csv")
    ap.add_argument("--min_area", type=float, default=16.0)
    ap.add_argument("--clip_to_image", action="store_true", default=True)
    ap.add_argument("--rebuild", action="store_true", default=True)
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    images_dir = src_root / args.images_dir
    ann_dir = src_root / args.ann_dir

    train_csv = ann_dir / args.train_csv
    val_csv = ann_dir / args.val_csv
    test_csv = ann_dir / args.test_csv

    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [images_dir, train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    # 1 класс: product
    stats_train = build_split(
        "train",
        train_csv,
        images_dir,
        out_dir,
        args.min_area,
        args.clip_to_image,
        cls_id=0,
    )
    stats_val = build_split(
        "val", val_csv, images_dir, out_dir, args.min_area, args.clip_to_image, cls_id=0
    )
    stats_test = build_split(
        "test",
        test_csv,
        images_dir,
        out_dir,
        args.min_area,
        args.clip_to_image,
        cls_id=0,
    )

    write_dataset_yaml(out_dir, nc=1, names=["product"])

    report = {
        "src_root": str(src_root),
        "out_dir": str(out_dir),
        "min_area": args.min_area,
        "clip_to_image": args.clip_to_image,
        "train": stats_train.__dict__,
        "val": stats_val.__dict__,
        "test": stats_test.__dict__,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE. YOLO dataset:", out_dir)
    print("Report:", out_dir / "build_report.json")


if __name__ == "__main__":
    import json

    main()
