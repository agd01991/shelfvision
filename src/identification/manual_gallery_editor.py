from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .sku_gallery import IMAGE_EXTS


EDIT_COLUMNS = [
    "operation",
    "source_sku_id",
    "target_sku_id",
    "new_sku_id",
    "ref_files",
    "comment",
    "created_at",
]
WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


@dataclass
class ManualGalleryEdit:
    operation: str
    source_sku_id: str
    target_sku_id: str = ""
    new_sku_id: str = ""
    ref_files: str = ""
    comment: str = ""
    created_at: str = ""


@dataclass
class ManualGallerySummary:
    source_gallery_dir: str
    output_gallery_dir: str
    output_gallery_csv: str
    edits_csv: str
    original_sku_count: int
    manual_sku_count: int
    original_refs_count: int
    manual_refs_count: int
    edits_count: int
    merge_edits_count: int
    split_edits_count: int
    skipped_edits_count: int
    status: str
    warnings: List[str]


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


def _iter_sku_dirs(gallery_dir: Path) -> Iterable[Path]:
    if not gallery_dir.exists():
        return []
    return sorted(path for path in gallery_dir.iterdir() if path.is_dir())


def _iter_image_refs(sku_dir: Path) -> List[Path]:
    return sorted(path for path in sku_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def list_sku_refs(gallery_dir: str | Path) -> Dict[str, List[Path]]:
    gallery_dir = _current_os_path(gallery_dir)
    result: Dict[str, List[Path]] = {}
    for sku_dir in _iter_sku_dirs(gallery_dir):
        refs = _iter_image_refs(sku_dir)
        if refs:
            result[sku_dir.name] = refs
    return result


def infer_gallery_dir_from_experiment(experiment_dir: str | Path) -> Path | None:
    experiment_dir = _current_os_path(experiment_dir)
    demo_dir = experiment_dir / "02_demo_gallery"

    for json_path in [
        demo_dir / "demo_sku_gallery_summary.json",
        experiment_dir / "05_reports" / "full_experiment_summary.json",
    ]:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ["gallery_dir", "output_gallery_dir"]:
            value = data.get(key)
            if value:
                path = _current_os_path(str(value))
                if path.exists():
                    return path
        gallery_csv = data.get("gallery_csv")
        if gallery_csv:
            path = _current_os_path(str(gallery_csv)).parent
            if path.exists():
                return path

    for csv_path in [demo_dir / "demo_sku_gallery_items.csv", demo_dir / "sku_clusters.csv"]:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "gallery_image_path" not in df.columns or df.empty:
            continue
        first = _current_os_path(str(df["gallery_image_path"].dropna().astype(str).iloc[0]))
        if first.parent.parent.exists():
            return first.parent.parent

    return None


def read_manual_edits(edits_csv: str | Path) -> List[ManualGalleryEdit]:
    edits_csv = _current_os_path(edits_csv)
    if not edits_csv.exists():
        return []
    try:
        df = pd.read_csv(edits_csv).fillna("")
    except Exception:
        return []
    edits: List[ManualGalleryEdit] = []
    for _, row in df.iterrows():
        operation = str(row.get("operation", "")).strip().lower()
        source_sku_id = str(row.get("source_sku_id", "")).strip()
        if operation not in {"merge", "split"} or not source_sku_id:
            continue
        edits.append(
            ManualGalleryEdit(
                operation=operation,
                source_sku_id=source_sku_id,
                target_sku_id=str(row.get("target_sku_id", "")).strip(),
                new_sku_id=str(row.get("new_sku_id", "")).strip(),
                ref_files=str(row.get("ref_files", "")).strip(),
                comment=str(row.get("comment", "")).strip(),
                created_at=str(row.get("created_at", "")).strip(),
            )
        )
    return edits


def append_manual_edit(edits_csv: str | Path, edit: ManualGalleryEdit) -> Path:
    edits_csv = _current_os_path(edits_csv)
    edits_csv.parent.mkdir(parents=True, exist_ok=True)
    if not edit.created_at:
        edit.created_at = datetime.now().isoformat(timespec="seconds")
    exists = edits_csv.exists() and edits_csv.stat().st_size > 0
    with edits_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDIT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(edit))
    return edits_csv


def _write_edits_csv(edits_csv: Path, edits: List[ManualGalleryEdit]) -> Path:
    edits_csv.parent.mkdir(parents=True, exist_ok=True)
    with edits_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDIT_COLUMNS)
        writer.writeheader()
        for edit in edits:
            writer.writerow(asdict(edit))
    return edits_csv


def _parse_ref_files(value: str) -> List[str]:
    refs: List[str] = []
    for chunk in str(value).replace(",", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            refs.append(chunk)
    return refs


def _next_manual_sku_id(groups: Dict[str, List[Path]], prefix: str = "sku_demo_manual_") -> str:
    index = 1
    while f"{prefix}{index:03d}" in groups:
        index += 1
    return f"{prefix}{index:03d}"


def _copy_groups_to_gallery(groups: Dict[str, List[Path]], output_gallery_dir: Path) -> int:
    if output_gallery_dir.exists():
        shutil.rmtree(output_gallery_dir)
    output_gallery_dir.mkdir(parents=True, exist_ok=True)

    refs_count = 0
    for sku_id, refs in sorted(groups.items()):
        if not refs:
            continue
        sku_dir = output_gallery_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)
        for index, src in enumerate(refs, start=1):
            suffix = src.suffix.lower() if src.suffix else ".jpg"
            dst = sku_dir / f"ref_{index:03d}{suffix}"
            shutil.copy2(src, dst)
            refs_count += 1
    return refs_count


def _manual_gallery_rows(output_gallery_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for sku_id, refs in list_sku_refs(output_gallery_dir).items():
        for index, ref in enumerate(refs, start=1):
            rows.append(
                {
                    "sku_id": sku_id,
                    "sku_name": sku_id.replace("_", " "),
                    "category": "",
                    "ref_index": index,
                    "image_path": str(ref),
                }
            )
    return rows


def _write_manual_items(output_gallery_dir: Path, out_dir: Path) -> Path:
    path = out_dir / "manual_gallery_items.csv"
    pd.DataFrame(_manual_gallery_rows(output_gallery_dir)).to_csv(path, index=False)
    return path


def _write_simple_gallery_csv(output_gallery_dir: Path, output_gallery_csv: Path) -> Path:
    rows = _manual_gallery_rows(output_gallery_dir)
    output_gallery_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["sku_id", "sku_name", "category", "image_path"]).to_csv(output_gallery_csv, index=False)
    return output_gallery_csv


def _write_simple_gallery_check(output_gallery_dir: Path, output_gallery_csv: Path, out_dir: Path) -> Dict[str, Path]:
    check_dir = out_dir / "gallery_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    rows = _manual_gallery_rows(output_gallery_dir)
    stats = []
    for sku_id, refs in list_sku_refs(output_gallery_dir).items():
        stats.append({"sku_id": sku_id, "images_count": len(refs), "status": "ok" if refs else "empty"})
    summary = {
        "gallery_dir": str(output_gallery_dir),
        "output_csv": str(output_gallery_csv),
        "sku_count": len(stats),
        "images_count": len(rows),
        "status": "ok" if rows else "error",
        "note": "Manual gallery check does not require cv2; image integrity is not validated here.",
    }
    summary_json = check_dir / "sku_gallery_report.json"
    sku_stats_csv = check_dir / "sku_gallery_sku_stats.csv"
    image_stats_csv = check_dir / "sku_gallery_images.csv"
    markdown_report = check_dir / "sku_gallery_report.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(stats).to_csv(sku_stats_csv, index=False)
    pd.DataFrame(rows).to_csv(image_stats_csv, index=False)
    markdown_report.write_text(
        "\n".join(
            [
                "# ShelfVision: manual SKU gallery check",
                "",
                f"- Gallery dir: `{output_gallery_dir}`",
                f"- Gallery CSV: `{output_gallery_csv}`",
                f"- SKU count: {summary['sku_count']}",
                f"- Images count: {summary['images_count']}",
                f"- Status: **{summary['status']}**",
                "",
                "This lightweight check intentionally does not import OpenCV, so the Control Panel can run even when `cv2` is unavailable in the Windows environment.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "gallery_csv": output_gallery_csv,
        "summary_json": summary_json,
        "sku_stats_csv": sku_stats_csv,
        "image_stats_csv": image_stats_csv,
        "markdown_report": markdown_report,
    }


def _write_report(summary: ManualGallerySummary, out_dir: Path) -> Path:
    path = out_dir / "manual_gallery_report.md"
    lines = [
        "# ShelfVision: ручная корректировка SKU-галереи",
        "",
        "## Назначение",
        "",
        "Модуль ручной корректировки позволяет объединять похожие SKU-кластеры и разделять смешанные кластеры без изменения исходного результата эксперимента. Все операции сохраняются в CSV, после чего формируется новая manual SKU-галерея и новый `gallery.csv`.",
        "",
        "## Сводка",
        "",
        f"- Исходная gallery: `{summary.source_gallery_dir}`",
        f"- Manual gallery: `{summary.output_gallery_dir}`",
        f"- Manual gallery.csv: `{summary.output_gallery_csv}`",
        f"- Edits CSV: `{summary.edits_csv}`",
        f"- SKU до правок: {summary.original_sku_count}",
        f"- SKU после правок: {summary.manual_sku_count}",
        f"- Refs до правок: {summary.original_refs_count}",
        f"- Refs после правок: {summary.manual_refs_count}",
        f"- Операций всего: {summary.edits_count}",
        f"- Merge-операций: {summary.merge_edits_count}",
        f"- Split-операций: {summary.split_edits_count}",
        f"- Пропущено операций: {summary.skipped_edits_count}",
        f"- Статус: **{summary.status}**",
        "",
    ]
    if summary.warnings:
        lines.extend(["## Предупреждения", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
        lines.append("")
    lines.extend(
        [
            "## Формулировка для ВКР",
            "",
            "Для повышения интерпретируемости результата в систему добавлен механизм экспертной корректировки автоматически сформированной SKU-галереи. Пользователь может вручную объединять кластеры, относящиеся к одному товару, а также выделять ошибочно объединённые эталоны в отдельные SKU. Исходные результаты эксперимента при этом не изменяются: на их основе формируется отдельная manual-версия галереи, что позволяет сравнить качество идентификации до и после экспертной корректировки.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_manual_gallery_from_edits(
    source_gallery_dir: str | Path,
    output_gallery_dir: str | Path,
    edits_csv: str | Path,
    out_dir: str | Path | None = None,
    output_gallery_csv: str | Path | None = None,
) -> Dict[str, Path]:
    source_gallery_dir = _current_os_path(source_gallery_dir)
    output_gallery_dir = _current_os_path(output_gallery_dir)
    edits_csv = _current_os_path(edits_csv)
    out_dir = _current_os_path(out_dir) if out_dir is not None else output_gallery_dir.parent / "manual_gallery_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_gallery_csv = _current_os_path(output_gallery_csv) if output_gallery_csv is not None else output_gallery_dir / "gallery.csv"

    original_groups = list_sku_refs(source_gallery_dir)
    groups: Dict[str, List[Path]] = {sku_id: list(refs) for sku_id, refs in original_groups.items()}
    edits = read_manual_edits(edits_csv)
    warnings: List[str] = []
    skipped = 0

    for edit in edits:
        if edit.operation == "merge":
            if not edit.target_sku_id:
                warnings.append(f"merge {edit.source_sku_id}: target_sku_id пустой")
                skipped += 1
                continue
            if edit.source_sku_id not in groups:
                warnings.append(f"merge: source_sku_id не найден: {edit.source_sku_id}")
                skipped += 1
                continue
            if edit.target_sku_id not in groups:
                warnings.append(f"merge: target_sku_id не найден: {edit.target_sku_id}")
                skipped += 1
                continue
            if edit.source_sku_id == edit.target_sku_id:
                warnings.append(f"merge: source и target совпадают: {edit.source_sku_id}")
                skipped += 1
                continue
            groups[edit.target_sku_id].extend(groups.pop(edit.source_sku_id))

        elif edit.operation == "split":
            if edit.source_sku_id not in groups:
                warnings.append(f"split: source_sku_id не найден: {edit.source_sku_id}")
                skipped += 1
                continue
            ref_names = set(_parse_ref_files(edit.ref_files))
            if not ref_names:
                warnings.append(f"split {edit.source_sku_id}: ref_files пустой")
                skipped += 1
                continue
            source_refs = groups[edit.source_sku_id]
            selected = [ref for ref in source_refs if ref.name in ref_names]
            if not selected:
                warnings.append(f"split {edit.source_sku_id}: выбранные refs не найдены: {edit.ref_files}")
                skipped += 1
                continue
            new_sku_id = edit.new_sku_id or _next_manual_sku_id(groups)
            if new_sku_id in groups:
                groups[new_sku_id].extend(selected)
            else:
                groups[new_sku_id] = list(selected)
            selected_names = {ref.name for ref in selected}
            groups[edit.source_sku_id] = [ref for ref in source_refs if ref.name not in selected_names]

    groups = {sku_id: refs for sku_id, refs in groups.items() if refs}
    manual_refs_count = _copy_groups_to_gallery(groups, output_gallery_dir)
    manual_items_csv = _write_manual_items(output_gallery_dir, out_dir)
    _write_simple_gallery_csv(output_gallery_dir, output_gallery_csv)
    gallery_outputs = _write_simple_gallery_check(output_gallery_dir, output_gallery_csv, out_dir)

    applied_edits_csv = out_dir / "manual_cluster_edits_applied.csv"
    _write_edits_csv(applied_edits_csv, edits)

    status = "ok" if manual_refs_count > 0 else "error"
    if skipped:
        status = "warning"
    summary = ManualGallerySummary(
        source_gallery_dir=str(source_gallery_dir),
        output_gallery_dir=str(output_gallery_dir),
        output_gallery_csv=str(output_gallery_csv),
        edits_csv=str(edits_csv),
        original_sku_count=len(original_groups),
        manual_sku_count=len(groups),
        original_refs_count=sum(len(refs) for refs in original_groups.values()),
        manual_refs_count=manual_refs_count,
        edits_count=len(edits),
        merge_edits_count=sum(1 for edit in edits if edit.operation == "merge"),
        split_edits_count=sum(1 for edit in edits if edit.operation == "split"),
        skipped_edits_count=skipped,
        status=status,
        warnings=warnings,
    )
    summary_json = out_dir / "manual_gallery_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = _write_report(summary, out_dir)

    outputs = {
        "manual_gallery_dir": output_gallery_dir,
        "manual_gallery_csv": output_gallery_csv,
        "manual_items_csv": manual_items_csv,
        "applied_edits_csv": applied_edits_csv,
        "summary_json": summary_json,
        "report_md": report_md,
    }
    outputs.update(gallery_outputs)
    return outputs
