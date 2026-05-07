from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SPLITS = {
    "train": "data/coco_splits/d2s_small/train_fix.json",
    "val": "data/coco_splits/d2s_small/val_fix.json",
    "test": "data/coco_splits/d2s_small/test_fix.json",
}

SPLIT_CANDIDATES = {
    "train": [
        "data/coco_splits/d2s_small/train_fix.json",
        "data/coco_splits/d2s_small/train.json",
        "data/raw/d2s_small/annotations_train.json",
    ],
    "val": [
        "data/coco_splits/d2s_small/val_fix.json",
        "data/coco_splits/d2s_small/val.json",
        "data/raw/d2s_small/annotations_val.json",
    ],
    "test": [
        "data/coco_splits/d2s_small/test_fix.json",
        "data/coco_splits/d2s_small/test.json",
        "data/raw/d2s_small/annotations_test.json",
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing_path(path: str | Path, candidates: list[str] | None = None) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for candidate in candidates or []:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            print(f"[PATH] {p} не найден, использую {candidate_path}")
            return candidate_path
    return p


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "source_file": str(path),
                    "note": "Файл не найден. Этот блок ещё нужно получить отдельным запуском.",
                }
            ]
        )
    df = pd.read_csv(path)
    if "status" not in df.columns:
        df.insert(0, "status", "ok")
    else:
        df["status"] = df["status"].fillna("ok")
    df["source_file"] = str(path)
    return df


def segmentation_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return bool(value)
    return False


def summarize_coco_split(split_name: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "status": "missing",
        "source_file": str(path),
        "images_count": 0,
        "annotations_count": 0,
        "segmentation_annotations_count": 0,
        "categories_count": 0,
        "avg_objects_per_image": 0.0,
        "min_objects_per_image": 0,
        "max_objects_per_image": 0,
        "avg_annotation_area": 0.0,
        "small_area_annotations_lt_32x32": 0,
        "medium_area_annotations_32x32_96x96": 0,
        "large_area_annotations_gt_96x96": 0,
    }
    if not path.exists():
        return row

    data = read_json(path)
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []
    categories = data.get("categories", []) or []

    objects_by_image = {int(image.get("id")): 0 for image in images if image.get("id") is not None}
    areas: list[float] = []
    seg_count = 0
    small = medium = large = 0

    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is not None:
            objects_by_image[int(image_id)] = objects_by_image.get(int(image_id), 0) + 1
        if segmentation_is_present(ann.get("segmentation")):
            seg_count += 1
        area = ann.get("area")
        if area is None and isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
            area = float(ann["bbox"][2]) * float(ann["bbox"][3])
        if area is not None:
            area = float(area)
            areas.append(area)
            if area < 32 * 32:
                small += 1
            elif area <= 96 * 96:
                medium += 1
            else:
                large += 1

    counts = list(objects_by_image.values())
    row.update(
        {
            "status": "ok",
            "images_count": len(images),
            "annotations_count": len(annotations),
            "segmentation_annotations_count": seg_count,
            "categories_count": len(categories),
            "avg_objects_per_image": round(sum(counts) / len(counts), 4) if counts else 0.0,
            "min_objects_per_image": min(counts) if counts else 0,
            "max_objects_per_image": max(counts) if counts else 0,
            "avg_annotation_area": round(sum(areas) / len(areas), 2) if areas else 0.0,
            "small_area_annotations_lt_32x32": small,
            "medium_area_annotations_32x32_96x96": medium,
            "large_area_annotations_gt_96x96": large,
        }
    )
    return row


def make_dataset_stats(split_paths: dict[str, Path]) -> pd.DataFrame:
    rows = [summarize_coco_split(split, path) for split, path in split_paths.items()]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        total_images = sum(int(row["images_count"] or 0) for row in ok_rows)
        total_annotations = sum(int(row["annotations_count"] or 0) for row in ok_rows)
        total_seg_annotations = sum(int(row["segmentation_annotations_count"] or 0) for row in ok_rows)
        total_small = sum(int(row["small_area_annotations_lt_32x32"] or 0) for row in ok_rows)
        total_medium = sum(int(row["medium_area_annotations_32x32_96x96"] or 0) for row in ok_rows)
        total_large = sum(int(row["large_area_annotations_gt_96x96"] or 0) for row in ok_rows)
        rows.append(
            {
                "split": "total",
                "status": "ok",
                "source_file": " + ".join(str(path) for path in split_paths.values()),
                "images_count": total_images,
                "annotations_count": total_annotations,
                "segmentation_annotations_count": total_seg_annotations,
                "categories_count": max(int(row["categories_count"] or 0) for row in ok_rows),
                "avg_objects_per_image": round(total_annotations / total_images, 4) if total_images else 0.0,
                "min_objects_per_image": "",
                "max_objects_per_image": "",
                "avg_annotation_area": "",
                "small_area_annotations_lt_32x32": total_small,
                "medium_area_annotations_32x32_96x96": total_medium,
                "large_area_annotations_gt_96x96": total_large,
            }
        )
    return pd.DataFrame(rows)


def make_missing_checklist(
    dataset_stats: pd.DataFrame,
    yolo_seg_last: pd.DataFrame,
    mask_summary: pd.DataFrame,
    crop_quality: pd.DataFrame,
    sku_bbox: pd.DataFrame,
    sku_mask: pd.DataFrame,
) -> pd.DataFrame:
    def ok(df: pd.DataFrame) -> bool:
        return "status" in df.columns and (df["status"].astype(str) == "ok").any()

    dataset_ok = "status" in dataset_stats.columns and (dataset_stats["status"].astype(str) == "ok").any()
    return pd.DataFrame(
        [
            {
                "block": "dataset_stats",
                "status": "ok" if dataset_ok else "missing",
                "what_to_do": "Проверить пути к COCO segmentation JSON train/val/test.",
            },
            {
                "block": "yolo_seg_training_metrics",
                "status": "ok" if ok(yolo_seg_last) else "missing",
                "what_to_do": "Нужен reports/all_stats/D2S_YOLO_SEG_last.csv или новый запуск обучения YOLO-Seg.",
            },
            {
                "block": "mask_evaluation_metrics",
                "status": "ok" if ok(mask_summary) else "missing",
                "what_to_do": "Запустить run_segmentation_evaluation.py после YOLO-Seg inference.",
            },
            {
                "block": "bbox_vs_mask_crop_quality",
                "status": "ok" if ok(crop_quality) else "missing",
                "what_to_do": "Добавить/запустить scripts/compare_bbox_mask_crops.py.",
            },
            {
                "block": "sku_bbox_preparation",
                "status": "ok" if ok(sku_bbox) else "optional_missing",
                "what_to_do": "Запустить run_identification.py без --use-masks, если есть SKU gallery.",
            },
            {
                "block": "sku_mask_preparation",
                "status": "ok" if ok(sku_mask) else "optional_missing",
                "what_to_do": "Запустить run_identification.py с --use-masks, если есть SKU gallery.",
            },
        ]
    )


def write_markdown(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = [
        "# Таблицы для статьи по инстанс-сегментации\n",
        "Сгенерировано скриптом `scripts/export_segmentation_article_metrics.py`.\n",
    ]
    titles = {
        "table_1_dataset_stats": "Таблица 1. Характеристики сегментационного набора",
        "table_2_yolo_seg_training_metrics": "Таблица 2. Итоговые метрики обучения YOLO-Seg",
        "table_3_mask_evaluation_metrics": "Таблица 3. Дополнительная mask-оценка",
        "table_4_crop_quality": "Таблица 4. Сравнение bbox crop и mask crop",
        "table_5_sku_bbox_metrics": "Таблица 5a. SKU preparation: bbox crops",
        "table_6_sku_mask_metrics": "Таблица 5b. SKU preparation: mask crops",
        "missing_checklist": "Чек-лист готовности данных",
    }
    for key, title in titles.items():
        df = tables.get(key)
        if df is None or df.empty:
            continue
        lines.append(f"\n## {title}\n")
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append(df.to_csv(index=False))
        lines.append("\n")
    (out_dir / "article_segmentation_tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_excel(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    xlsx_path = out_dir / "article_segmentation_metrics.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception as exc:  # noqa: BLE001
        (out_dir / "excel_export_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает таблицы для статьи про instance segmentation и SKU preparation.")
    parser.add_argument("--train-coco", default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val-coco", default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test-coco", default=DEFAULT_SPLITS["test"])
    parser.add_argument("--yolo-seg-last", default="reports/all_stats/D2S_YOLO_SEG_last.csv")
    parser.add_argument("--mask-summary", default="results/article_segmentation/yolo_seg_masks/segmentation_metrics_summary.csv")
    parser.add_argument("--crop-quality-summary", default="results/article_segmentation/crop_comparison/crop_quality_summary.csv")
    parser.add_argument("--sku-bbox-metrics", default="results/article_segmentation/sku_bbox/identification_metrics.csv")
    parser.add_argument("--sku-mask-metrics", default="results/article_segmentation/sku_mask/identification_metrics.csv")
    parser.add_argument("--out-dir", default="reports/article_segmentation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": resolve_existing_path(args.train_coco, SPLIT_CANDIDATES["train"]),
        "val": resolve_existing_path(args.val_coco, SPLIT_CANDIDATES["val"]),
        "test": resolve_existing_path(args.test_coco, SPLIT_CANDIDATES["test"]),
    }
    dataset_stats = make_dataset_stats(split_paths)
    yolo_seg_last = safe_read_csv(Path(args.yolo_seg_last))
    mask_summary = safe_read_csv(Path(args.mask_summary))
    crop_quality = safe_read_csv(Path(args.crop_quality_summary))
    sku_bbox = safe_read_csv(Path(args.sku_bbox_metrics))
    sku_mask = safe_read_csv(Path(args.sku_mask_metrics))

    tables = {
        "table_1_dataset_stats": dataset_stats,
        "table_2_yolo_seg_training_metrics": yolo_seg_last,
        "table_3_mask_evaluation_metrics": mask_summary,
        "table_4_crop_quality": crop_quality,
        "table_5_sku_bbox_metrics": sku_bbox,
        "table_6_sku_mask_metrics": sku_mask,
    }
    tables["missing_checklist"] = make_missing_checklist(
        dataset_stats=dataset_stats,
        yolo_seg_last=yolo_seg_last,
        mask_summary=mask_summary,
        crop_quality=crop_quality,
        sku_bbox=sku_bbox,
        sku_mask=sku_mask,
    )

    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    write_markdown(out_dir, tables)
    write_excel(out_dir, tables)

    print("DONE")
    print(f"Output dir: {out_dir}")
    print(f"Markdown:   {out_dir / 'article_segmentation_tables.md'}")
    print(f"Checklist:  {out_dir / 'missing_checklist.csv'}")
    print(f"Excel:      {out_dir / 'article_segmentation_metrics.xlsx'}")


if __name__ == "__main__":
    main()
