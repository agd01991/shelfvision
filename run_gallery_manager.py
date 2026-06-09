from __future__ import annotations

import argparse

from src.identification.gallery_manager import build_sku_gallery


OUTPUT_LABELS_RU = {
    "gallery_csv": "CSV-файл галереи",
    "gallery_report_md": "Markdown-отчёт по галерее",
    "gallery_report_json": "JSON-отчёт по галерее",
    "sku_gallery_report_md": "Markdown-отчёт по SKU-галерее",
    "sku_gallery_report_json": "JSON-отчёт по SKU-галерее",
    "valid_items_csv": "CSV валидных эталонов",
    "invalid_items_csv": "CSV проблемных эталонов",
    "summary_json": "JSON-сводка",
}


def _label_output(name: str) -> str:
    return OUTPUT_LABELS_RU.get(str(name), str(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Менеджер SKU-галереи ShelfVision")
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

    print("=== ShelfVision: менеджер SKU-галереи ===", flush=True)
    for name, path in outputs.items():
        print(f"- {_label_output(name)}: {path}", flush=True)


if __name__ == "__main__":
    main()
