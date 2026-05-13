from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import pandas as pd

from .sku_gallery import IMAGE_EXTS, SkuGalleryItem


@dataclass
class GalleryImageRecord:
    sku_id: str
    sku_name: str
    category: str
    image_path: str
    width: int
    height: int
    status: str
    error: str = ""


@dataclass
class GallerySkuStats:
    sku_id: str
    sku_name: str
    category: str
    images_count: int
    valid_images_count: int
    broken_images_count: int
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    status: str


@dataclass
class GalleryReport:
    gallery_dir: str
    output_csv: str
    out_dir: str
    sku_count: int
    valid_sku_count: int
    empty_sku_count: int
    weak_sku_count: int
    images_count: int
    valid_images_count: int
    broken_images_count: int
    min_images_per_sku: int
    status: str
    warnings: List[str]


def _iter_sku_dirs(gallery_dir: Path) -> Iterable[Path]:
    if not gallery_dir.exists():
        return []
    return sorted(path for path in gallery_dir.iterdir() if path.is_dir())


def _iter_image_paths(sku_dir: Path) -> Iterable[Path]:
    return sorted(path for path in sku_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _read_image_size(path: Path) -> tuple[int, int, str, str]:
    try:
        image = cv2.imread(str(path))
        if image is None:
            return 0, 0, "broken", "cv2.imread вернул None"
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return 0, 0, "broken", "нулевой размер изображения"
        return int(width), int(height), "valid", ""
    except Exception as exc:  # noqa: BLE001 - диагностический модуль не должен падать на одном файле
        return 0, 0, "broken", str(exc)


def _sku_name_from_dir(sku_dir: Path) -> str:
    return sku_dir.name.replace("_", " ")


def _category_from_sku_dir(sku_dir: Path) -> str:
    # Можно позже заменить на чтение metadata.json, но для минимального мастера категория необязательна.
    return ""


def _status_for_sku(valid_count: int, total_count: int, min_images_per_sku: int) -> str:
    if total_count == 0:
        return "empty"
    if valid_count == 0:
        return "broken"
    if valid_count < min_images_per_sku:
        return "weak"
    return "ok"


def scan_sku_gallery(
    gallery_dir: str | Path,
    min_images_per_sku: int = 3,
) -> tuple[List[GalleryImageRecord], List[GallerySkuStats]]:
    gallery_dir = Path(gallery_dir)
    image_records: List[GalleryImageRecord] = []
    sku_stats: List[GallerySkuStats] = []

    for sku_dir in _iter_sku_dirs(gallery_dir):
        sku_id = sku_dir.name
        sku_name = _sku_name_from_dir(sku_dir)
        category = _category_from_sku_dir(sku_dir)
        image_paths = list(_iter_image_paths(sku_dir))

        valid_sizes: List[tuple[int, int]] = []
        broken_count = 0
        for image_path in image_paths:
            width, height, status, error = _read_image_size(image_path)
            if status == "valid":
                valid_sizes.append((width, height))
            else:
                broken_count += 1
            image_records.append(
                GalleryImageRecord(
                    sku_id=sku_id,
                    sku_name=sku_name,
                    category=category,
                    image_path=str(image_path),
                    width=width,
                    height=height,
                    status=status,
                    error=error,
                )
            )

        valid_count = len(valid_sizes)
        sku_status = _status_for_sku(valid_count, len(image_paths), min_images_per_sku)
        widths = [item[0] for item in valid_sizes] or [0]
        heights = [item[1] for item in valid_sizes] or [0]
        sku_stats.append(
            GallerySkuStats(
                sku_id=sku_id,
                sku_name=sku_name,
                category=category,
                images_count=len(image_paths),
                valid_images_count=valid_count,
                broken_images_count=broken_count,
                min_width=min(widths),
                min_height=min(heights),
                max_width=max(widths),
                max_height=max(heights),
                status=sku_status,
            )
        )

    return image_records, sku_stats


def _write_gallery_csv(records: List[GalleryImageRecord], output_csv: str | Path, only_valid: bool = True) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in records:
        if only_valid and item.status != "valid":
            continue
        rows.append(
            {
                "sku_id": item.sku_id,
                "sku_name": item.sku_name,
                "category": item.category,
                "image_path": item.image_path,
            }
        )
    pd.DataFrame(rows, columns=["sku_id", "sku_name", "category", "image_path"]).to_csv(output_csv, index=False)
    return output_csv


def _write_csv(path: Path, rows: List[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown_report(report: GalleryReport, sku_stats: List[GallerySkuStats], out_dir: Path) -> Path:
    lines = [
        "# ShelfVision: отчёт по SKU-галерее",
        "",
        "## Сводка",
        "",
        f"- Папка галереи: `{report.gallery_dir}`",
        f"- CSV галереи: `{report.output_csv}`",
        f"- SKU всего: {report.sku_count}",
        f"- SKU в хорошем состоянии: {report.valid_sku_count}",
        f"- Пустых SKU: {report.empty_sku_count}",
        f"- Слабых SKU: {report.weak_sku_count}",
        f"- Изображений всего: {report.images_count}",
        f"- Валидных изображений: {report.valid_images_count}",
        f"- Битых изображений: {report.broken_images_count}",
        f"- Минимум эталонов на SKU: {report.min_images_per_sku}",
        f"- Статус: **{report.status}**",
        "",
    ]
    if report.warnings:
        lines.extend(["## Предупреждения", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend([
        "## SKU",
        "",
        "| sku_id | images | valid | broken | status |",
        "|---|---:|---:|---:|---|",
    ])
    for item in sku_stats[:100]:
        lines.append(
            f"| {item.sku_id} | {item.images_count} | {item.valid_images_count} | "
            f"{item.broken_images_count} | {item.status} |"
        )
    report_path = out_dir / "sku_gallery_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_sku_gallery(
    gallery_dir: str | Path,
    output_csv: str | Path,
    out_dir: str | Path,
    min_images_per_sku: int = 3,
) -> Dict[str, Path]:
    """Checks SKU gallery quality and creates gallery.csv for identification."""

    gallery_dir = Path(gallery_dir)
    output_csv = Path(output_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"SKU-галерея не найдена: {gallery_dir}")
    if not gallery_dir.is_dir():
        raise NotADirectoryError(f"SKU-галерея должна быть папкой: {gallery_dir}")

    image_records, sku_stats = scan_sku_gallery(gallery_dir, min_images_per_sku=min_images_per_sku)
    gallery_csv = _write_gallery_csv(image_records, output_csv=output_csv, only_valid=True)

    valid_sku_count = sum(1 for item in sku_stats if item.status == "ok")
    empty_sku_count = sum(1 for item in sku_stats if item.status == "empty")
    weak_sku_count = sum(1 for item in sku_stats if item.status == "weak")
    broken_sku_count = sum(1 for item in sku_stats if item.status == "broken")
    valid_images_count = sum(1 for item in image_records if item.status == "valid")
    broken_images_count = sum(1 for item in image_records if item.status == "broken")

    warnings: List[str] = []
    if empty_sku_count:
        warnings.append(f"Есть пустые папки SKU: {empty_sku_count}.")
    if weak_sku_count:
        warnings.append(f"Есть SKU с малым числом эталонов: {weak_sku_count}.")
    if broken_sku_count:
        warnings.append(f"Есть SKU только с битыми изображениями: {broken_sku_count}.")
    if broken_images_count:
        warnings.append(f"Найдены битые изображения: {broken_images_count}.")
    if not sku_stats:
        warnings.append("В папке галереи не найдено ни одной SKU-папки.")

    status = "ok" if sku_stats and not warnings else "warning"
    if not sku_stats or valid_images_count == 0:
        status = "error"

    report = GalleryReport(
        gallery_dir=str(gallery_dir),
        output_csv=str(gallery_csv),
        out_dir=str(out_dir),
        sku_count=len(sku_stats),
        valid_sku_count=valid_sku_count,
        empty_sku_count=empty_sku_count,
        weak_sku_count=weak_sku_count,
        images_count=len(image_records),
        valid_images_count=valid_images_count,
        broken_images_count=broken_images_count,
        min_images_per_sku=min_images_per_sku,
        status=status,
        warnings=warnings,
    )

    summary_json = out_dir / "sku_gallery_report.json"
    summary_json.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    sku_stats_csv = _write_csv(out_dir / "sku_gallery_sku_stats.csv", [asdict(item) for item in sku_stats])
    image_stats_csv = _write_csv(out_dir / "sku_gallery_images.csv", [asdict(item) for item in image_records])
    markdown_report = _write_markdown_report(report, sku_stats, out_dir=out_dir)

    return {
        "gallery_csv": gallery_csv,
        "summary_json": summary_json,
        "sku_stats_csv": sku_stats_csv,
        "image_stats_csv": image_stats_csv,
        "markdown_report": markdown_report,
    }
