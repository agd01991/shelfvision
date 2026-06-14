from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STATUS_COLUMNS = ["status", "sku_status", "assignment_status"]
SIMILARITY_COLUMNS = ["best_similarity", "similarity", "score", "sku_confidence"]
MARGIN_COLUMNS = ["margin", "distinct_margin"]


def _first_existing(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _mean_from_columns(df: pd.DataFrame, columns: Iterable[str]) -> float:
    for col in columns:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
    return 0.0


def _copy(src: Optional[Path], dst_dir: Path) -> Optional[Path]:
    if src is None or not src.exists() or not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return dst


def _find_visualized(run_dir: Path) -> list[Path]:
    roots = [
        run_dir / "04_identification" / "visualized",
        run_dir / "04_identification" / "visualized_selected",
        run_dir / "06_manual_gallery" / "manual_identification" / "visualized",
        run_dir / "03_identification" / "visualized",
    ]
    images: list[Path] = []
    for root in roots:
        if root.exists():
            images.extend(path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    return images


def _status_counts(results_df: pd.DataFrame) -> tuple[str, int, int, int]:
    status_col = _status_column(results_df)
    if not status_col:
        return "", 0, 0, 0
    statuses = results_df[status_col].astype(str)
    return (
        status_col,
        int(statuses.eq("matched").sum()),
        int(statuses.eq("matched_uncertain").sum()),
        int(statuses.eq("unknown").sum()),
    )


def export_assets(run_dir: str | Path, out_dir: str | Path, limit: int = 12) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = _first_existing(run_dir, ["04_identification/identification_results.csv", "03_identification/identification_results.csv", "identification_results.csv"])
    corrected_csv = _first_existing(run_dir, ["04_identification/identification_results_corrected.csv", "identification_results_corrected.csv"])
    gallery_csv = _first_existing(run_dir, ["02_demo_gallery/sku_gallery_final/gallery.csv", "02_demo_gallery/gallery.csv", "gallery.csv"])
    uncertain_csv = _first_existing(run_dir, ["04_identification/matched_uncertain_candidates.csv", "matched_uncertain_candidates.csv", "uncertain_report/matched_uncertain_top.csv"])
    validation_md = _first_existing(run_dir, ["validation_report.md"])
    manual_report_md = _first_existing(run_dir, ["manual_corrections_report.md"])
    manifest_json = _first_existing(run_dir, ["run_manifest.json"])

    results_df = _read_csv(results_csv)
    corrected_df = _read_csv(corrected_csv)
    gallery_df = _read_csv(gallery_csv)
    status_col, matched, matched_uncertain, unknown = _status_counts(results_df)

    key_metrics = {
        "objects_total": len(results_df),
        "sku_count": int(gallery_df["sku_id"].nunique()) if "sku_id" in gallery_df.columns else 0,
        "gallery_refs": len(gallery_df),
        "status_column": status_col,
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "avg_similarity": _mean_from_columns(results_df, SIMILARITY_COLUMNS),
        "avg_margin": _mean_from_columns(results_df, MARGIN_COLUMNS),
    }
    if not corrected_df.empty:
        corrected_status_col, corrected_matched, corrected_uncertain, corrected_unknown = _status_counts(corrected_df)
        key_metrics.update(
            {
                "corrected_status_column": corrected_status_col,
                "corrected_matched": corrected_matched,
                "corrected_matched_uncertain": corrected_uncertain,
                "corrected_unknown": corrected_unknown,
            }
        )
    key_metrics_csv = out_dir / "key_metrics.csv"
    pd.DataFrame([key_metrics]).to_csv(key_metrics_csv, index=False)

    status_distribution_csv = out_dir / "status_distribution.csv"
    if status_col:
        status_df = results_df[status_col].astype(str).value_counts().rename_axis("status").reset_index(name="count")
        status_df["share"] = status_df["count"] / max(1, int(status_df["count"].sum()))
        status_df.to_csv(status_distribution_csv, index=False)
    else:
        pd.DataFrame(columns=["status", "count", "share"]).to_csv(status_distribution_csv, index=False)

    visualized_dir = out_dir / "visualized_examples"
    visualized_dir.mkdir(parents=True, exist_ok=True)
    copied_visualized = 0
    for index, image in enumerate(_find_visualized(run_dir)[: max(0, limit)], start=1):
        shutil.copyfile(image, visualized_dir / f"{index:03d}_{image.name}")
        copied_visualized += 1

    _copy(results_csv, out_dir)
    _copy(corrected_csv, out_dir)
    _copy(gallery_csv, out_dir)
    _copy(uncertain_csv, out_dir)
    _copy(validation_md, out_dir)
    _copy(manual_report_md, out_dir)
    _copy(manifest_json, out_dir)

    summary_md = out_dir / "summary_for_presentation.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Материалы для защиты ShelfVision",
                "",
                f"Папка результата: `{run_dir}`",
                f"Ключевые метрики: `{key_metrics_csv}`",
                f"Распределение статусов: `{status_distribution_csv}`",
                f"Колонка статуса: `{status_col or 'не найдена'}`",
                f"Итоговых визуализаций скопировано: {copied_visualized}",
                "",
                "Папка `visualized_examples` содержит изображения для демонстрации результата: полка, найденные товарные объекты и подписи SKU-сопоставления.",
                "Если сформирован файл `identification_results_corrected.csv`, он также включён в материалы и отражает ручные решения пользователя.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "summary_for_presentation_md": summary_md,
        "key_metrics_csv": key_metrics_csv,
        "status_distribution_csv": status_distribution_csv,
        "visualized_examples_dir": visualized_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт материалов ShelfVision для защиты")
    parser.add_argument("--run-dir", required=True, help="Папка результата запуска")
    parser.add_argument("--out-dir", required=True, help="Папка для материалов защиты")
    parser.add_argument("--limit", type=int, default=12, help="Сколько итоговых визуализаций скопировать")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_assets(args.run_dir, args.out_dir, limit=args.limit)
    print("=== ShelfVision: материалы для защиты экспортированы ===")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
