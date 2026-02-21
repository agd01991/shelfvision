import argparse
import csv
import random
import shutil
from pathlib import Path

DEFAULT_FIELDS_8 = [
    "image",
    "x1",
    "y1",
    "x2",
    "y2",
    "class",
    "image_width",
    "image_height",
]


def iter_rows(csv_path: Path, encoding: str = "utf-8"):
    """
    Возвращает (fieldnames, rows_dict_iterable).
    Поддерживает:
      - CSV с заголовком (DictReader)
      - CSV без заголовка (Reader + DEFAULT_FIELDS_8)
    """
    with csv_path.open("r", newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("CSV is empty")

        # Определяем: это заголовок или строка данных
        # Если первый элемент == "image" или среди полей есть x1/y1/x2/y2 — считаем это header
        first_lower = [c.strip().lower() for c in first]
        looks_like_header = (len(first_lower) > 0 and first_lower[0] == "image") or (
            "x1" in first_lower
            and "y1" in first_lower
            and "x2" in first_lower
            and "y2" in first_lower
        )

        if looks_like_header:
            fieldnames = first
            dict_reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in dict_reader:
                yield fieldnames, row
        else:
            # Нет заголовка: используем дефолтные имена полей по длине строки
            fieldnames = DEFAULT_FIELDS_8[: len(first)]
            # first — это строка данных
            yield fieldnames, dict(zip(fieldnames, first))
            for row_list in reader:
                if not row_list:
                    continue
                yield fieldnames, dict(zip(fieldnames, row_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_root",
        required=True,
        help="Корень sku110k (где есть images/ и annotations/)",
    )
    ap.add_argument(
        "--dst_root", default="data/raw/sku110k_small", help="Куда сохранить small"
    )
    ap.add_argument(
        "--csv", default="annotations.csv", help="Путь к CSV относительно src_root"
    )
    ap.add_argument(
        "--images_dir",
        default="images",
        help="Папка с изображениями относительно src_root",
    )
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--image_col", default="image", help="Имя колонки с путём к картинке"
    )
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    src_csv = src_root / args.csv
    src_images = src_root / args.images_dir

    if not src_csv.exists():
        raise FileNotFoundError(f"CSV not found: {src_csv}")
    if not src_images.exists():
        raise FileNotFoundError(f"images_dir not found: {src_images}")

    dst_root = Path(args.dst_root)
    dst_images = dst_root / "images"
    dst_root.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    # 1) читаем CSV → группируем строки по image
    rows_by_image = {}
    fieldnames = None

    for fns, row in iter_rows(src_csv, encoding=args.encoding):
        fieldnames = fns
        img = (row.get(args.image_col) or "").strip()
        if not img:
            continue
        rows_by_image.setdefault(img, []).append(row)

    if not rows_by_image:
        raise ValueError("No rows parsed from CSV (check encoding/format)")

    images_list = sorted(rows_by_image.keys())
    rnd = random.Random(args.seed)
    rnd.shuffle(images_list)
    picked = images_list[: min(args.n, len(images_list))]

    # 2) копируем изображения (с подпапками если есть)
    copied = 0
    for rel in picked:
        src_path = src_images / rel
        if not src_path.exists():
            # fallback: иногда в CSV лежит только имя файла
            src_path = src_images / Path(rel).name
            if not src_path.exists():
                continue

        dst_path = dst_images / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied += 1

    # 3) пишем новый CSV уже С ЗАГОЛОВКОМ (чтобы дальше sku110k_reader работал как есть)
    out_csv = dst_root / "annotations.csv"
    if fieldnames is None:
        fieldnames = DEFAULT_FIELDS_8

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in picked:
            for row in rows_by_image.get(rel, []):
                writer.writerow(row)

    print(f"Done. picked_images={len(picked)} copied_images={copied}")
    print(f"Saved: {dst_root}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
