from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.error_visualization import visualize_errors_from_files
from src.evaluation.metrics import evaluate_predictions_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision evaluation runner")
    parser.add_argument("--predictions", required=True, help="prediction.json или predictions.json из run_inference.py")
    parser.add_argument("--gt-coco", help="COCO JSON с эталонной bbox-разметкой")
    parser.add_argument("--gt-yolo-labels", help="Папка labels с YOLO txt-разметкой")
    parser.add_argument("--images-dir", help="Папка изображений для YOLO-разметки")
    parser.add_argument("--out-dir", default="results/evaluation", help="Папка для метрик и визуализаций")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold для TP/FP/FN")
    parser.add_argument("--visualize-errors", action="store_true", help="Сохранять изображения с TP/FP/FN")
    parser.add_argument("--limit", type=int, default=0, help="Сколько изображений визуализировать, 0 — все")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    result = evaluate_predictions_file(
        predictions_json=args.predictions,
        gt_coco_json=args.gt_coco,
        gt_yolo_labels_dir=args.gt_yolo_labels,
        images_dir=args.images_dir,
        out_dir=out_dir,
        iou_threshold=args.iou,
    )

    summary = result["summary"]
    print("=== ShelfVision evaluation ===")
    print(f"Images:    {summary['images_count']}")
    print(f"TP/FP/FN:  {summary['tp']} / {summary['fp']} / {summary['fn']}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"F1:        {summary['f1']:.4f}")
    print(f"AP50:      {summary['AP50']:.4f}")
    print(f"AP50-95:   {summary['AP50-95']:.4f}")
    print(f"Saved to:  {out_dir}")

    if args.visualize_errors:
        saved = visualize_errors_from_files(
            predictions_json=args.predictions,
            out_dir=out_dir / "errors",
            gt_coco_json=args.gt_coco,
            gt_yolo_labels_dir=args.gt_yolo_labels,
            images_dir=args.images_dir,
            iou_threshold=args.iou,
            limit=args.limit,
        )
        print(f"Error visualizations: {len(saved)} files saved to {out_dir / 'errors'}")


if __name__ == "__main__":
    main()
