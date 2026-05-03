from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.segmentation_metrics import evaluate_segmentation_predictions_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision segmentation evaluation runner")
    parser.add_argument("--predictions", required=True, help="prediction.json или predictions.json с masks из run_inference.py")
    parser.add_argument("--gt-coco", required=True, help="COCO JSON с segmentation-разметкой")
    parser.add_argument("--out-dir", default="results/evaluation/yolo_seg_masks", help="Папка для mask-метрик")
    parser.add_argument("--iou", type=float, default=0.5, help="Mask IoU threshold для TP/FP/FN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    result = evaluate_segmentation_predictions_file(
        predictions_json=args.predictions,
        gt_coco_json=args.gt_coco,
        out_dir=out_dir,
        iou_threshold=args.iou,
    )

    summary = result["summary"]
    print("=== ShelfVision segmentation evaluation ===")
    print(f"Images:       {summary['images_count']}")
    print(f"TP/FP/FN:     {summary['tp']} / {summary['fp']} / {summary['fn']}")
    print(f"Mask P:       {summary['mask_precision']:.4f}")
    print(f"Mask R:       {summary['mask_recall']:.4f}")
    print(f"Mask F1:      {summary['mask_f1']:.4f}")
    print(f"Mean mask IoU:{summary['mean_mask_iou']:.4f}")
    print(f"APmask50:     {summary['APmask50']:.4f}")
    print(f"APmask75:     {summary['APmask75']:.4f}")
    print(f"APmask50-95:  {summary['APmask50-95']:.4f}")
    print(f"Saved to:     {out_dir}")


if __name__ == "__main__":
    main()
