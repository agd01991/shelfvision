from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


@dataclass
class SelectedSkuExportSummary:
    experiment_dir: str
    output_dir: str
    selected_sku_count: int
    selected_result_rows: int
    gallery_refs_copied: int
    query_crops_copied: int
    status: str


def _current_os_path(value: str | Path | None) -> Path:
    raw = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    if os.name == "nt":
        match = WSL_MOUNT_RE.match(raw)
        if match:
            return Path(f"{match.group(1).upper()}:/{match.group(2)}")
        return Path(raw)
    match = WINDOWS_DRIVE_RE.match(raw)
    if match:
        return Path(f"/mnt/{match.group(1).lower()}/{match.group(2)}")
    return Path(raw)


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _read_csv(path: Path) -> pd.DataFrame:
    path = _current_os_path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _copy_file(src: Path, dst_dir: Path, prefix: str = "") -> Path | None:
    src = _current_os_path(src)
    dst_dir = _current_os_path(dst_dir)
    if not src.exists() or not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}{src.name}" if prefix else src.name
    dst = dst_dir / name
    shutil.copy2(src, dst)
    return dst


def _iter_gallery_refs(gallery_dir: Path, sku_id: str) -> Iterable[Path]:
    gallery_dir = _current_os_path(gallery_dir)
    sku_dir = gallery_dir / sku_id
    if not sku_dir.exists() or not sku_dir.is_dir():
        return []
    return sorted(path for path in sku_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def export_selected_sku_demo(
    experiment_dir: str | Path,
    selected_skus: Sequence[str],
    output_dir: str | Path | None = None,
    results_csv: str | Path | None = None,
    gallery_dir: str | Path | None = None,
    max_rows_per_sku: int = 40,
    include_unknown_similar: bool = False,
) -> Dict[str, Path]:
    """Export a compact demo subset for selected SKU identifiers."""

    exp = _current_os_path(experiment_dir)
    selected = [sku.strip() for sku in selected_skus if str(sku).strip()]
    if output_dir is None:
        output_dir = exp / "selected_sku_demo"
    out = _current_os_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if results_csv is None:
        corrected = exp / "06_manual_identification" / "identification_results_corrected.csv"
        raw = exp / "04_identification" / "identification_results.csv"
        results_csv = corrected if corrected.exists() else raw
    results_path = _current_os_path(results_csv)

    if gallery_dir is None:
        # Prefer the final gallery directory saved by full experiment report.
        summary_path = exp / "05_reports" / "full_experiment_summary.json"
        gallery_from_summary = ""
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                gallery_from_summary = _safe_text(summary.get("gallery_csv"))
            except Exception:
                gallery_from_summary = ""
        if gallery_from_summary:
            gallery_dir = _current_os_path(gallery_from_summary).parent
        else:
            gallery_dir = exp / "02_demo_gallery"
    gallery_root = _current_os_path(gallery_dir)

    results = _read_csv(results_path)
    selected_set = set(selected)
    if results.empty:
        selected_results = pd.DataFrame()
    else:
        mask = results.get("sku_id", pd.Series(dtype=str)).astype(str).isin(selected_set)
        if include_unknown_similar and "top_k" in results.columns:
            for sku in selected:
                mask = mask | results["top_k"].astype(str).str.contains(sku, case=False, na=False)
        selected_results = results[mask].copy()

    if not selected_results.empty and max_rows_per_sku > 0 and "sku_id" in selected_results.columns:
        selected_results = (
            selected_results.groupby("sku_id", group_keys=False)
            .head(max_rows_per_sku)
            .reset_index(drop=True)
        )

    selected_skus_csv = out / "selected_skus.csv"
    pd.DataFrame({"sku_id": selected}).to_csv(selected_skus_csv, index=False)

    selected_results_csv = out / "selected_identification_results.csv"
    selected_results.to_csv(selected_results_csv, index=False)

    refs_copied = 0
    crops_copied = 0

    gallery_out = out / "gallery_refs"
    for sku in selected:
        for ref in _iter_gallery_refs(gallery_root, sku):
            copied = _copy_file(ref, gallery_out / sku)
            if copied is not None:
                refs_copied += 1

    crops_out = out / "query_matches"
    if not selected_results.empty and "crop_path" in selected_results.columns:
        for _, row in selected_results.iterrows():
            sku = _safe_text(row.get("sku_id")) or "unknown"
            object_id = _safe_text(row.get("object_id"))
            crop = _current_os_path(_safe_text(row.get("crop_path")))
            copied = _copy_file(crop, crops_out / sku, prefix=f"obj_{object_id}_")
            if copied is not None:
                crops_copied += 1

    summary = SelectedSkuExportSummary(
        experiment_dir=str(exp),
        output_dir=str(out),
        selected_sku_count=len(selected),
        selected_result_rows=len(selected_results),
        gallery_refs_copied=refs_copied,
        query_crops_copied=crops_copied,
        status="ok" if selected else "warning",
    )

    summary_json = out / "selected_sku_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = out / "selected_sku_report.md"
    lines: List[str] = [
        "# Демонстрационный набор выбранных SKU",
        "",
        f"- Папка эксперимента: `{summary.experiment_dir}`",
        f"- Выбранных SKU: {summary.selected_sku_count}",
        f"- Строк результатов идентификации: {summary.selected_result_rows}",
        f"- Скопировано эталонов галереи: {summary.gallery_refs_copied}",
        f"- Скопировано проверяемых фрагментов: {summary.query_crops_copied}",
        "",
        "## Выбранные SKU",
        "",
    ]
    lines.extend(f"- `{sku}`" for sku in selected)
    lines.extend(
        [
            "",
            "## Назначение",
            "",
            "Этот набор нужен для защиты: по выбранным SKU можно показать эталоны галереи, проверяемые фрагменты, top-k кандидатов и спорные случаи.",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "selected_skus_csv": selected_skus_csv,
        "selected_results_csv": selected_results_csv,
        "summary_json": summary_json,
        "report_md": report_md,
        "output_dir": out,
    }
