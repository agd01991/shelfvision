from __future__ import annotations

import argparse

from src.identification.demo_gallery_builder import build_demo_sku_gallery_from_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision automatic demo SKU gallery builder")
    parser.add_argument("--predictions", required=True, help="predictions.json or video_predictions.json")
    parser.add_argument("--images-dir", default=None, help="Source images directory for relative paths")
    parser.add_argument("--gallery-dir", default="D:/1Diplom/sku_gallery", help="Output gallery directory")
    parser.add_argument("--gallery-csv", default="D:/1Diplom/sku_gallery/gallery.csv", help="Output gallery.csv path")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/demo_sku_gallery", help="Output reports directory")
    parser.add_argument("--max-sku", type=int, default=30, help="Maximum demo SKU count")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum detection score")
    parser.add_argument("--min-width", type=int, default=20, help="Minimum crop width")
    parser.add_argument("--min-height", type=int, default=20, help="Minimum crop height")
    parser.add_argument("--padding", type=float, default=0.05, help="BBox padding ratio")
    parser.add_argument("--bbox-only", action="store_true", help="Use bbox crop even when masks exist")
    parser.add_argument("--prefix", default="sku_demo_", help="Demo SKU prefix")
    parser.add_argument("--keep-old-demo", action="store_true", help="Keep previous demo SKU folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_demo_sku_gallery_from_predictions(
        predictions_json=args.predictions,
        images_dir=args.images_dir,
        gallery_dir=args.gallery_dir,
        gallery_csv=args.gallery_csv,
        out_dir=args.out_dir,
        max_sku=max(1, args.max_sku),
        min_score=args.min_score,
        min_width=max(1, args.min_width),
        min_height=max(1, args.min_height),
        use_masks=not args.bbox_only,
        padding_ratio=args.padding,
        prefix=args.prefix,
        clear_old_demo=not args.keep_old_demo,
    )
    print("=== ShelfVision demo SKU gallery builder ===", flush=True)
    for name, path in outputs.items():
        print(f"- {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
