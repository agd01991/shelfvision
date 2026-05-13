from __future__ import annotations

import argparse

from src.identification.gallery_manager import build_sku_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision SKU gallery manager")
    parser.add_argument("--gallery-dir", required=True, help="Папка вида sku_gallery/<sku_id>/*.jpg")
    parser.add_argument("--output-csv", required=True, help="Куда сохранить gallery.csv")
    parser.add_argument("--out-dir", default="results/sku_gallery", help="Папка для отчётов по галерее")
    parser.add_argument("--min-images-per-sku", type=int, default=3, help="Минимальное число эталонов на SKU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_sku_gallery(
        gallery_dir=args.gallery_dir,
        output_csv=args.output_csv,
        out_dir=args.out_dir,
        min_images_per_sku=max(1, args.min_images_per_sku),
    )

    print("=== ShelfVision SKU gallery manager ===", flush=True)
    for name, path in outputs.items():
        print(f"- {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
