from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def export_assets(run_dir: str | Path, out_dir: str | Path, limit: int = 12) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = _first_existing(run_dir, ["04_identification/identification_results.csv", "03_identification/identification_results.csv", "identification_results.csv"])
    gallery_csv = _first_existing(run_dir, ["02_demo_gallery/sku_gallery_final/gallery.csv", "02_demo_gallery/gallery.csv", "gallery.csv"])
    uncertain_csv = _first_existing(run_dir, ["04_identification/matched_uncertain_candidates.csv", "matched_uncertain_candidates.csv"])
    validation_md = _first_existing(run_dir, ["validation_report.md"])
    manifest_json = _first_existing(run_dir, ["run_manifest.json"])

    results_df = _read_csv(results_csv)
    gallery_df = _read_csv(gallery_csv)

    key_metrics = {
        "objects_total": len(results_df),
        "sku_count": int(gallery_df["sku_id"].nunique()) if "sku_id" in gallery_df.columns else 0,
        "gallery_refs": len(gallery_df),
    }
    if "status" in results_df.columns:
        statuses = results_df["status"].astype(str)
        key_metrics["matched"] = int(statuses.eq("matched").sum())
        key_metrics["matched_uncertain"] = int(statuses.eq("matched_uncertain").sum())
        key_metrics["unknown"] = int(statuses.eq("unknown").sum())
    key_metrics_csv = out_dir / "key_metrics.csv"
    pd.DataFrame([key_metrics]).to_csv(key_metrics_csv, index=False)

    status_distribution_csv = out_dir / "status_distribution.csv"
    if "status" in results_df.columns:
        status_df = results_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
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
    _copy(gallery_csv, out_dir)
    _copy(uncertain_csv, out_dir)
    _copy(validation_md, out_dir)
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
                f"Итоговых визуализаций скопировано: {copied_visualized}",
                "",
                "Папка `visualized_examples` содержит изображения для демонстрации результата: полка, найденные товарные объекты и подписи SKU-сопоставления.",
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
