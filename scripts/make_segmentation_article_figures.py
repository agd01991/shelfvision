from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_first_images(src_dir: Path, dst_dir: Path, prefix: str, limit: int) -> int:
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

    copied = 0
    for idx, src in enumerate(files[:limit], start=1):
        dst = dst_dir / f"{prefix}_{idx:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Готовит рисунки для статьи по YOLO-Seg и SKU preparation.")
    parser.add_argument(
        "--visualized-dir",
        default="results/article_segmentation/yolo_seg_inference/visualized",
        help="Папка с визуализациями YOLO-Seg после run_inference.py",
    )
    parser.add_argument(
        "--crop-examples-dir",
        default="results/article_segmentation/crop_comparison/examples",
        help="Папка с bbox_vs_mask примерами после compare_bbox_mask_crops.py",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/article_segmentation/figures",
        help="Куда сохранить отобранные рисунки для статьи",
    )
    parser.add_argument("--limit", type=int, default=5, help="Сколько примеров взять из каждой папки")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    yolo_seg_dir = out_dir / "yolo_seg_predictions"
    crop_dir = out_dir / "bbox_vs_mask_crop"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_predictions = copy_first_images(
        src_dir=Path(args.visualized_dir),
        dst_dir=yolo_seg_dir,
        prefix="fig_yolo_seg_prediction",
        limit=args.limit,
    )
    copied_crops = copy_first_images(
        src_dir=Path(args.crop_examples_dir),
        dst_dir=crop_dir,
        prefix="fig_bbox_vs_mask_crop",
        limit=args.limit,
    )

    manifest = out_dir / "figures_manifest.md"
    manifest.write_text(
        "\n".join(
            [
                "# Рисунки для статьи по инстанс-сегментации",
                "",
                f"YOLO-Seg visualized source: `{args.visualized_dir}`",
                f"BBox/mask crop source: `{args.crop_examples_dir}`",
                "",
                f"Copied YOLO-Seg prediction figures: {copied_predictions}",
                f"Copied bbox/mask crop figures: {copied_crops}",
                "",
                "Рекомендуемое использование в статье:",
                "1. `yolo_seg_predictions/fig_yolo_seg_prediction_01.*` — пример предсказания масок на полочном изображении.",
                "2. `bbox_vs_mask_crop/fig_bbox_vs_mask_crop_01.*` — центральный рисунок сравнения bbox crop и mask crop.",
                "3. Остальные изображения можно использовать как дополнительные примеры или приложение.",
            ]
        ),
        encoding="utf-8",
    )

    print("=== ShelfVision article figures ===")
    print(f"Output: {out_dir}")
    print(f"YOLO-Seg prediction figures: {copied_predictions}")
    print(f"BBox/mask crop figures:     {copied_crops}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
