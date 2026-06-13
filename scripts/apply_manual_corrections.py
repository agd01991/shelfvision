from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


STATUS_COLUMNS = ["sku_status", "status", "assignment_status"]
CORRECTION_COLUMNS = [
    "created_at",
    "image_name",
    "object_id",
    "old_status",
    "old_sku_id",
    "old_sku_name",
    "new_sku_id",
    "correction_type",
    "comment",
    "sku_confidence",
    "distinct_margin",
    "second_distinct_sku",
    "crop_path",
]


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Файл пустой: {path}")
    return df


def _normalize_object_id(value: Any) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return str(value).strip()


def _correction_key(image_name: Any, object_id: Any) -> tuple[str, str]:
    return str(image_name).strip(), _normalize_object_id(object_id)


def _latest_corrections(corrections: pd.DataFrame) -> Dict[tuple[str, str], pd.Series]:
    result: Dict[tuple[str, str], pd.Series] = {}
    if "created_at" in corrections.columns:
        corrections = corrections.sort_values("created_at")
    for _, row in corrections.iterrows():
        key = _correction_key(row.get("image_name", ""), row.get("object_id", ""))
        result[key] = row
    return result


def _apply_one(row: pd.Series, status_col: str, correction: pd.Series) -> pd.Series:
    action = str(correction.get("correction_type", "")).strip()
    new_sku_id = str(correction.get("new_sku_id", "")).strip()

    if action == "confirm_match":
        row[status_col] = "matched"
        if new_sku_id:
            row["sku_id"] = new_sku_id
        if "safe_sku_id" in row.index:
            row["safe_sku_id"] = row.get("sku_id", "")
        if "safe_sku_name" in row.index and "sku_name" in row.index:
            row["safe_sku_name"] = row.get("sku_name", "")
    elif action == "change_sku":
        row[status_col] = "matched"
        row["sku_id"] = new_sku_id
        if "sku_name" in row.index:
            row["sku_name"] = new_sku_id
        if "safe_sku_id" in row.index:
            row["safe_sku_id"] = new_sku_id
        if "safe_sku_name" in row.index:
            row["safe_sku_name"] = new_sku_id
    elif action == "mark_unknown":
        row[status_col] = "unknown"
        if "sku_id" in row.index:
            row["sku_id"] = ""
        if "sku_name" in row.index:
            row["sku_name"] = "unknown"
        if "safe_sku_id" in row.index:
            row["safe_sku_id"] = ""
        if "safe_sku_name" in row.index:
            row["safe_sku_name"] = ""
    elif action == "needs_review":
        row[status_col] = "matched_uncertain"

    row["manual_correction_type"] = action
    row["manual_comment"] = str(correction.get("comment", ""))
    row["manual_corrected_at"] = str(correction.get("created_at", ""))
    return row


def _summary(df: pd.DataFrame, status_col: str) -> Dict[str, Any]:
    statuses = df[status_col].astype(str)
    matched = int(statuses.eq("matched").sum())
    uncertain = int(statuses.eq("matched_uncertain").sum())
    unknown = int(statuses.eq("unknown").sum())
    total = int(len(df))
    assigned = matched + uncertain
    return {
        "total_objects": total,
        "matched": matched,
        "matched_uncertain": uncertain,
        "unknown": unknown,
        "assigned": assigned,
        "matched_rate": matched / total if total else 0.0,
        "matched_uncertain_rate": uncertain / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "assigned_rate": assigned / total if total else 0.0,
    }


def apply_manual_corrections(
    results_csv: str | Path,
    corrections_csv: str | Path,
    out_csv: str | Path,
    report_dir: str | Path | None = None,
) -> Dict[str, Path]:
    results_csv = Path(results_csv)
    corrections_csv = Path(corrections_csv)
    out_csv = Path(out_csv)
    report_dir = Path(report_dir) if report_dir else out_csv.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results = _read_csv(results_csv)
    corrections = _read_csv(corrections_csv)
    status_col = _status_column(results)
    if status_col is None:
        raise ValueError(f"В таблице результатов нет колонки статуса. Ожидается одна из: {STATUS_COLUMNS}")

    before = _summary(results, status_col)
    latest = _latest_corrections(corrections)
    corrected = results.copy()
    applied = 0
    missed = 0

    for index, row in corrected.iterrows():
        key = _correction_key(row.get("image_name", ""), row.get("object_id", ""))
        correction = latest.get(key)
        if correction is None:
            continue
        corrected.loc[index] = _apply_one(row.copy(), status_col, correction)
        applied += 1

    correction_keys = set(latest.keys())
    result_keys = {_correction_key(row.get("image_name", ""), row.get("object_id", "")) for _, row in results.iterrows()}
    missed = len(correction_keys - result_keys)

    corrected.to_csv(out_csv, index=False)
    after = _summary(corrected, status_col)

    summary = {
        "results_csv": str(results_csv),
        "corrections_csv": str(corrections_csv),
        "corrected_csv": str(out_csv),
        "manual_rows": int(len(corrections)),
        "manual_unique_objects": int(len(latest)),
        "applied": int(applied),
        "not_found_in_results": int(missed),
        "before": before,
        "after": after,
    }

    summary_json = report_dir / "manual_corrections_summary.json"
    report_md = report_dir / "manual_corrections_report.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Применение ручных решений ShelfVision",
        "",
        f"Файл исходных результатов: `{results_csv}`",
        f"Файл ручных решений: `{corrections_csv}`",
        f"Исправленная таблица: `{out_csv}`",
        "",
        "## Сводка применения",
        "",
        f"- строк ручных решений: {len(corrections)}",
        f"- уникальных объектов с ручным решением: {len(latest)}",
        f"- применено к результатам: {applied}",
        f"- не найдено в результатах: {missed}",
        "",
        "## До применения",
        "",
        f"- уверенно идентифицировано: {before['matched']}",
        f"- требует проверки: {before['matched_uncertain']}",
        f"- не определено: {before['unknown']}",
        "",
        "## После применения",
        "",
        f"- уверенно идентифицировано: {after['matched']}",
        f"- требует проверки: {after['matched_uncertain']}",
        f"- не определено: {after['unknown']}",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "corrected_results_csv": out_csv,
        "manual_corrections_summary_json": summary_json,
        "manual_corrections_report_md": report_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Применение ручных решений ShelfVision к таблице SKU-сопоставления")
    parser.add_argument("--results-csv", required=True, help="Путь к исходному identification_results.csv")
    parser.add_argument("--corrections-csv", required=True, help="Путь к manual_corrections.csv")
    parser.add_argument("--out-csv", required=True, help="Путь для identification_results_corrected.csv")
    parser.add_argument("--report-dir", default=None, help="Папка для отчёта применения ручных решений")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = apply_manual_corrections(
        results_csv=args.results_csv,
        corrections_csv=args.corrections_csv,
        out_csv=args.out_csv,
        report_dir=args.report_dir,
    )
    print("=== ShelfVision: ручные решения применены ===")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
