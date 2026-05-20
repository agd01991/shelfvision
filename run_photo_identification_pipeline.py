from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from run_inference import main as inference_main
from src.identification.demo_gallery_builder import build_demo_sku_gallery_from_predictions
from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.visualization import visualize_identification_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision photo identification pipeline")
    parser.add_argument("--model", choices=["yolo", "yolo_seg", "rtdetr", "frcnn"], default="yolo", help="Detection/segmentation model")
    parser.add_argument("--weights", required=True, help="Model weights path")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--images-dir", default=None, help="Images directory")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/photo_identification", help="Pipeline output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Model image size")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, cuda:0")
    parser.add_argument("--bbox-only", action="store_true", help="Use bbox crops even if masks exist")

    parser.add_argument("--gallery-dir", default="D:/1Diplom/sku_gallery", help="Demo SKU gallery directory")
    parser.add_argument("--gallery-csv", default="D:/1Diplom/sku_gallery/gallery.csv", help="Demo SKU gallery.csv path")
    parser.add_argument("--max-sku", type=int, default=30, help="Maximum demo SKU count")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum detection score for demo SKU reference")
    parser.add_argument("--min-width", type=int, default=20, help="Minimum reference crop width")
    parser.add_argument("--min-height", type=int, default=20, help="Minimum reference crop height")
    parser.add_argument("--padding", type=float, default=0.05, help="Crop padding ratio")
    parser.add_argument("--prefix", default="sku_demo_", help="Demo SKU prefix")
    parser.add_argument("--keep-old-demo", action="store_true", help="Keep previous demo SKU folders")

    parser.add_argument("--threshold", type=float, default=0.65, help="SKU matching threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k SKU candidates")
    parser.add_argument("--gt-csv", default=None, help="Optional GT CSV for identification metrics")
    parser.add_argument("--visualize-limit", type=int, default=50, help="Visualization image limit")
    return parser.parse_args()


def _run_inference(args: argparse.Namespace, inference_dir: Path) -> Path:
    import sys

    cli_args: List[str] = [
        "run_inference.py",
        "--model",
        args.model,
        "--weights",
        args.weights,
        "--out-dir",
        str(inference_dir),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
    ]
    if args.image:
        cli_args.extend(["--image", args.image])
    if args.images_dir:
        cli_args.extend(["--images-dir", args.images_dir])
    if args.device:
        cli_args.extend(["--device", args.device])
    if args.bbox_only:
        cli_args.append("--no-masks")

    old_argv = sys.argv
    try:
        sys.argv = cli_args
        inference_main()
    finally:
        sys.argv = old_argv

    prediction_file = inference_dir / ("prediction.json" if args.image else "predictions.json")
    if not prediction_file.exists():
        raise FileNotFoundError(f"Inference finished, but prediction file not found: {prediction_file}")
    return prediction_file


def main() -> None:
    args = parse_args()
    if not args.image and not args.images_dir:
        raise SystemExit("Specify --image or --images-dir")

    out_dir = Path(args.out_dir)
    inference_dir = out_dir / "01_inference"
    demo_gallery_out_dir = out_dir / "02_demo_gallery"
    identification_dir = out_dir / "03_identification"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ShelfVision photo identification pipeline ===", flush=True)
    print("Step 1/3: inference", flush=True)
    predictions_json = _run_inference(args, inference_dir=inference_dir)
    images_dir_for_matching = args.images_dir if args.images_dir else None

    print("Step 2/3: build demo SKU gallery", flush=True)
    demo_outputs = build_demo_sku_gallery_from_predictions(
        predictions_json=predictions_json,
        images_dir=images_dir_for_matching,
        gallery_dir=args.gallery_dir,
        gallery_csv=args.gallery_csv,
        out_dir=demo_gallery_out_dir,
        max_sku=max(1, args.max_sku),
        min_score=args.min_score,
        min_width=max(1, args.min_width),
        min_height=max(1, args.min_height),
        use_masks=not args.bbox_only,
        padding_ratio=args.padding,
        prefix=args.prefix,
        clear_old_demo=not args.keep_old_demo,
    )

    print("Step 3/3: identify photo objects", flush=True)
    results = run_sku_matching(
        predictions_json=predictions_json,
        images_dir=images_dir_for_matching,
        out_dir=identification_dir,
        gallery_csv=args.gallery_csv,
        gallery_dir=args.gallery_dir,
        use_masks=not args.bbox_only,
        threshold=args.threshold,
        top_k=args.top_k,
        padding_ratio=args.padding,
    )
    metrics = evaluate_with_ground_truth(results, gt_csv=args.gt_csv)
    save_identification_metrics(metrics, out_dir=identification_dir)
    save_identification_outputs(
        predictions_json=predictions_json,
        results=results,
        metrics=metrics,
        out_dir=identification_dir,
    )
    visualize_identification_results(
        results=results,
        images_dir=images_dir_for_matching,
        out_dir=identification_dir,
        limit=max(0, args.visualize_limit),
    )

    print("=== Done ===", flush=True)
    print(f"Pipeline output: {out_dir}", flush=True)
    print(f"Predictions: {predictions_json}", flush=True)
    print(f"Demo gallery: {args.gallery_dir}", flush=True)
    print(f"Gallery CSV: {args.gallery_csv}", flush=True)
    for name, path in demo_outputs.items():
        print(f"Demo {name}: {path}", flush=True)
    print(f"Identification results: {identification_dir}", flush=True)
    print(f"Objects: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Matched: {metrics.get('matched', 0)}", flush=True)
    print(f"Unknown: {metrics.get('unknown', 0)}", flush=True)


if __name__ == "__main__":
    main()
