from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

EDIT_COLUMNS = [
    "edit_id",
    "image_name",
    "image_path",
    "object_id",
    "crop_path",
    "old_sku_id",
    "old_sku_name",
    "old_status",
    "old_score",
    "old_margin",
    "new_sku_id",
    "new_sku_name",
    "new_status",
    "edit_type",
    "comment",
    "created_at",
]

STATUS_VALUES = {"matched", "matched_uncertain", "unknown"}
EDIT_TYPES = {
    "confirm",
    "change_sku",
    "set_unknown",
    "create_new_sku",
    "add_as_reference",
    "reject_match",
}


@dataclass
class ManualIdentificationEdit:
    edit_id: str
    image_name: str
    image_path: str
    object_id: int
    crop_path: str
    old_sku_id: str
    old_sku_name: str
    old_status: str
    old_score: float
    old_margin: float
    new_sku_id: str
    new_sku_name: str
    new_status: str
    edit_type: str
    comment: str = ""
    created_at: str = ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_int(value: object) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _normalize_status(value: str) -> str:
    value = str(value or "").strip()
    return value if value in STATUS_VALUES else "unknown"


def _normalize_edit_type(value: str) -> str:
    value = str(value or "").strip()
    return value if value in EDIT_TYPES else "change_sku"


def read_manual_identification_edits(edits_csv: str | Path) -> List[ManualIdentificationEdit]:
    path = Path(edits_csv)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return []

    edits: List[ManualIdentificationEdit] = []
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", "")).strip()
        object_id = _safe_int(row.get("object_id"))
        if not image_name or object_id <= 0:
            continue
        edits.append(
            ManualIdentificationEdit(
                edit_id=str(row.get("edit_id", "")).strip() or f"manual_{len(edits) + 1:04d}",
                image_name=image_name,
                image_path=str(row.get("image_path", "")).strip(),
                object_id=object_id,
                crop_path=str(row.get("crop_path", "")).strip(),
                old_sku_id=str(row.get("old_sku_id", "")).strip(),
                old_sku_name=str(row.get("old_sku_name", "")).strip(),
                old_status=str(row.get("old_status", "")).strip(),
                old_score=_safe_float(row.get("old_score")),
                old_margin=_safe_float(row.get("old_margin")),
                new_sku_id=str(row.get("new_sku_id", "")).strip(),
                new_sku_name=str(row.get("new_sku_name", "")).strip(),
                new_status=_normalize_status(str(row.get("new_status", ""))),
                edit_type=_normalize_edit_type(str(row.get("edit_type", ""))),
                comment=str(row.get("comment", "")).strip(),
                created_at=str(row.get("created_at", "")).strip(),
            )
        )
    return edits


def append_manual_identification_edit(edits_csv: str | Path, edit: ManualIdentificationEdit) -> Path:
    path = Path(edits_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not edit.created_at:
        edit.created_at = _now()
    if not edit.edit_id:
        edit.edit_id = f"manual_{edit.created_at.replace(':', '').replace('-', '').replace('T', '_')}"

    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDIT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(edit))
    return path


def _latest_edits_by_object(edits: List[ManualIdentificationEdit]) -> Dict[tuple[str, int], ManualIdentificationEdit]:
    result: Dict[tuple[str, int], ManualIdentificationEdit] = {}
    for edit in edits:
        result[(edit.image_name, int(edit.object_id))] = edit
    return result


def apply_manual_identification_edits(
    identification_results_csv: str | Path,
    edits_csv: str | Path,
    output_csv: str | Path,
    report_dir: str | Path | None = None,
) -> Dict[str, Path]:
    source_path = Path(identification_results_csv)
    edits_path = Path(edits_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_root = Path(report_dir) if report_dir is not None else output_path.parent
    report_root.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Файл результатов идентификации не найден: {source_path}")

    df = pd.read_csv(source_path).fillna("")
    edits = read_manual_identification_edits(edits_path)
    latest = _latest_edits_by_object(edits)

    if "manual_edit_applied" not in df.columns:
        df["manual_edit_applied"] = False
    if "manual_edit_type" not in df.columns:
        df["manual_edit_type"] = ""
    if "manual_comment" not in df.columns:
        df["manual_comment"] = ""
    if "original_sku_id" not in df.columns:
        df["original_sku_id"] = df.get("sku_id", "")
    if "original_sku_status" not in df.columns:
        df["original_sku_status"] = df.get("sku_status", "")

    applied = 0
    for index, row in df.iterrows():
        key = (str(row.get("image_name", "")).strip(), _safe_int(row.get("object_id")))
        edit = latest.get(key)
        if edit is None:
            continue

        status = _normalize_status(edit.new_status)
        sku_id = edit.new_sku_id.strip()
        sku_name = edit.new_sku_name.strip() or sku_id.replace("_", " ")

        df.at[index, "manual_edit_applied"] = True
        df.at[index, "manual_edit_type"] = edit.edit_type
        df.at[index, "manual_comment"] = edit.comment
        df.at[index, "sku_status"] = status

        if status == "unknown" or edit.edit_type in {"set_unknown", "reject_match"}:
            df.at[index, "sku_id"] = ""
            df.at[index, "sku_name"] = "unknown"
            if "safe_sku_id" in df.columns:
                df.at[index, "safe_sku_id"] = ""
            if "safe_sku_name" in df.columns:
                df.at[index, "safe_sku_name"] = ""
        else:
            df.at[index, "sku_id"] = sku_id
            df.at[index, "sku_name"] = sku_name
            if "safe_sku_id" in df.columns:
                df.at[index, "safe_sku_id"] = sku_id if status == "matched" else ""
            if "safe_sku_name" in df.columns:
                df.at[index, "safe_sku_name"] = sku_name if status == "matched" else ""
        applied += 1

    df.to_csv(output_path, index=False)

    total = len(df)
    matched = int((df.get("sku_status", "") == "matched").sum()) if total else 0
    uncertain = int((df.get("sku_status", "") == "matched_uncertain").sum()) if total else 0
    unknown = int((df.get("sku_status", "") == "unknown").sum()) if total else 0
    assigned = matched + uncertain

    summary = {
        "source_results_csv": str(source_path),
        "edits_csv": str(edits_path),
        "output_csv": str(output_path),
        "total_objects": total,
        "manual_edits_count": len(edits),
        "manual_edits_applied": applied,
        "matched": matched,
        "matched_uncertain": uncertain,
        "unknown": unknown,
        "assigned": assigned,
        "assigned_rate": assigned / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
    }

    summary_json = report_root / "manual_identification_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = report_root / "manual_identification_report.md"
    lines = [
        "# Ручная проверка результатов идентификации",
        "",
        "## Назначение",
        "",
        "Модуль позволяет экспертно проверить спорные назначения SKU без изменения исходного файла результатов. Все решения сохраняются отдельно и применяются в corrected-версии таблицы.",
        "",
        "## Сводка",
        "",
        f"- Объектов всего: {summary['total_objects']}",
        f"- Ручных правок в журнале: {summary['manual_edits_count']}",
        f"- Применено правок: {summary['manual_edits_applied']}",
        f"- matched: {summary['matched']}",
        f"- matched_uncertain: {summary['matched_uncertain']}",
        f"- unknown: {summary['unknown']}",
        f"- assigned_rate: {summary['assigned_rate']:.4f}",
        "",
        "## Важное ограничение",
        "",
        "Ручная проверка повышает экспертную согласованность демонстрационного контура, но не превращает assigned_rate в top-1 accuracy без эталонной SKU-разметки всех объектов.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "corrected_results_csv": output_path,
        "summary_json": summary_json,
        "report_md": report_md,
    }
