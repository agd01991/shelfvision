from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ALLOWED_STATUSES = {"matched", "matched_uncertain", "unknown"}
STATUS_COLUMNS = ["status", "sku_status", "assignment_status"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ValidationCheck:
    name: str
    status: str
    message: str
    path: str = ""


@dataclass
class ValidationSummary:
    status: str
    run_dir: str
    checks_total: int
    ok_count: int
    warning_count: int
    error_count: int
    crops_rows: int
    identification_rows: int
    matched: int
    matched_uncertain: int
    unknown: int


def _first_existing(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _find_visualized_dir(root: Path) -> Optional[Path]:
    candidates = [
        "04_identification/visualized",
        "04_identification/visualized_selected",
        "06_manual_gallery/manual_identification/visualized",
        "03_identification/visualized",
        "visualized",
    ]
    for rel in candidates:
        path = root / rel
        if path.exists() and path.is_dir():
            return path
    for path in root.rglob("visualized*"):
        if path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.rglob("*")):
            return path
    return None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _check_file(name: str, path: Optional[Path], required: bool = True) -> ValidationCheck:
    if path is None:
        return ValidationCheck(
            name=name,
            status="error" if required else "warning",
            message="не найден" if required else "не найден, но допускается",
        )
    if path.is_file():
        return ValidationCheck(name=name, status="ok", message="найден", path=str(path))
    return ValidationCheck(name=name, status="error", message="путь найден, но это не файл", path=str(path))


def _check_dir(name: str, path: Optional[Path], required: bool = True) -> ValidationCheck:
    if path is None:
        return ValidationCheck(
            name=name,
            status="error" if required else "warning",
            message="не найдена" if required else "не найдена, но допускается",
        )
    if path.is_dir():
        return ValidationCheck(name=name, status="ok", message="найдена", path=str(path))
    return ValidationCheck(name=name, status="error", message="путь найден, но это не папка", path=str(path))


def _check_not_empty(name: str, df: pd.DataFrame, path: Optional[Path]) -> ValidationCheck:
    if path is None:
        return ValidationCheck(name=name, status="error", message="файл не найден")
    if df.empty:
        return ValidationCheck(name=name, status="error", message="таблица пустая или не прочитана", path=str(path))
    return ValidationCheck(name=name, status="ok", message=f"строк: {len(df)}", path=str(path))


def _check_statuses(df: pd.DataFrame, path: Optional[Path]) -> ValidationCheck:
    if path is None or df.empty:
        return ValidationCheck(name="Статусы идентификации", status="error", message="нет данных для проверки")
    col = _status_column(df)
    if col is None:
        return ValidationCheck(
            name="Статусы идентификации",
            status="error",
            message=f"нет колонки статуса: ожидается одна из {STATUS_COLUMNS}",
            path=str(path),
        )
    statuses = set(df[col].dropna().astype(str).unique())
    unexpected = statuses - ALLOWED_STATUSES
    if unexpected:
        return ValidationCheck(
            name="Статусы идентификации",
            status="error",
            message=f"найдены недопустимые статусы: {sorted(unexpected)}",
            path=str(path),
        )
    return ValidationCheck(
        name="Статусы идентификации",
        status="ok",
        message=f"статусы корректны в колонке {col}: {sorted(statuses)}",
        path=str(path),
    )


def _check_crop_paths(df: pd.DataFrame, run_dir: Path, manifest_path: Optional[Path]) -> ValidationCheck:
    if manifest_path is None or df.empty:
        return ValidationCheck(name="Пути к вырезанным фрагментам", status="warning", message="нет данных для проверки")
    path_cols = [col for col in ["crop_path", "crop_file", "path"] if col in df.columns]
    if not path_cols:
        return ValidationCheck(name="Пути к вырезанным фрагментам", status="warning", message="колонка с путём к фрагменту не найдена", path=str(manifest_path))
    col = path_cols[0]
    values = [str(v) for v in df[col].dropna().tolist()[:200]]
    missing = []
    for value in values:
        path = Path(value)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(run_dir / path)
            candidates.append(manifest_path.parent / path)
        if not any(candidate.exists() for candidate in candidates):
            missing.append(value)
    if missing:
        return ValidationCheck(
            name="Пути к вырезанным фрагментам",
            status="warning",
            message=f"не найдены первые пути: {missing[:5]}",
            path=str(manifest_path),
        )
    return ValidationCheck(name="Пути к вырезанным фрагментам", status="ok", message=f"проверено путей: {len(values)}", path=str(manifest_path))


def validate_run(run_dir: Path) -> tuple[ValidationSummary, List[ValidationCheck], Dict[str, str]]:
    run_dir = run_dir.resolve()
    artifacts: Dict[str, Optional[Path]] = {
        "predictions_json": _first_existing(run_dir, ["03_query_inference/predictions.json", "01_inference/predictions.json", "predictions.json", "prediction.json"]),
        "summary_csv": _first_existing(run_dir, ["03_query_inference/summary.csv", "01_inference/summary.csv", "summary.csv"]),
        "crops_manifest_csv": _first_existing(run_dir, ["04_identification/crops_manifest.csv", "02_demo_gallery/crops_manifest.csv", "03_query_crops/crops_manifest.csv", "crops_manifest.csv"]),
        "gallery_csv": _first_existing(run_dir, ["02_demo_gallery/sku_gallery_final/gallery.csv", "02_demo_gallery/gallery.csv", "gallery.csv"]),
        "identification_results_csv": _first_existing(run_dir, ["04_identification/identification_results.csv", "03_identification/identification_results.csv", "06_manual_gallery/manual_identification/identification_results.csv", "identification_results.csv"]),
        "matched_uncertain_csv": _first_existing(run_dir, ["04_identification/matched_uncertain_candidates.csv", "06_manual_gallery/manual_identification/matched_uncertain_candidates.csv", "matched_uncertain_candidates.csv"]),
        "visualized_dir": _find_visualized_dir(run_dir),
    }

    checks: List[ValidationCheck] = [
        _check_file("predictions.json", artifacts["predictions_json"]),
        _check_file("summary.csv", artifacts["summary_csv"]),
        _check_file("crops_manifest.csv", artifacts["crops_manifest_csv"]),
        _check_file("gallery.csv", artifacts["gallery_csv"]),
        _check_file("identification_results.csv", artifacts["identification_results_csv"]),
        _check_file("matched_uncertain_candidates.csv", artifacts["matched_uncertain_csv"], required=False),
        _check_dir("visualized/", artifacts["visualized_dir"]),
    ]

    predictions_payload = _read_json(artifacts["predictions_json"])
    if predictions_payload is not None:
        checks.append(ValidationCheck("Содержимое predictions.json", "ok", "JSON читается", str(artifacts["predictions_json"])))
    else:
        checks.append(ValidationCheck("Содержимое predictions.json", "error", "JSON не читается или файл отсутствует", str(artifacts["predictions_json"] or "")))

    crops_df = _read_csv(artifacts["crops_manifest_csv"])
    ident_df = _read_csv(artifacts["identification_results_csv"])
    checks.append(_check_not_empty("Строки crops_manifest.csv", crops_df, artifacts["crops_manifest_csv"]))
    checks.append(_check_not_empty("Строки identification_results.csv", ident_df, artifacts["identification_results_csv"]))
    checks.append(_check_statuses(ident_df, artifacts["identification_results_csv"]))
    checks.append(_check_crop_paths(crops_df, run_dir, artifacts["crops_manifest_csv"]))

    status_col = _status_column(ident_df)
    if status_col:
        statuses = ident_df[status_col].astype(str)
        matched = int(statuses.eq("matched").sum())
        matched_uncertain = int(statuses.eq("matched_uncertain").sum())
        unknown = int(statuses.eq("unknown").sum())
    else:
        matched = matched_uncertain = unknown = 0

    error_count = sum(1 for item in checks if item.status == "error")
    warning_count = sum(1 for item in checks if item.status == "warning")
    ok_count = sum(1 for item in checks if item.status == "ok")
    status = "ERROR" if error_count else "WARNING" if warning_count else "OK"

    summary = ValidationSummary(
        status=status,
        run_dir=str(run_dir),
        checks_total=len(checks),
        ok_count=ok_count,
        warning_count=warning_count,
        error_count=error_count,
        crops_rows=len(crops_df),
        identification_rows=len(ident_df),
        matched=matched,
        matched_uncertain=matched_uncertain,
        unknown=unknown,
    )
    artifact_strings = {key: str(value) for key, value in artifacts.items() if value is not None}
    return summary, checks, artifact_strings


def write_reports(run_dir: Path, summary: ValidationSummary, checks: List[ValidationCheck], artifacts: Dict[str, str]) -> Dict[str, Path]:
    report_md = run_dir / "validation_report.md"
    summary_json = run_dir / "validation_summary.json"

    lines = [
        "# Проверка результата ShelfVision",
        "",
        f"Статус проверки: **{summary.status}**",
        "",
        "## Проверенные файлы",
        "",
    ]
    for name in ["predictions_json", "summary_csv", "crops_manifest_csv", "gallery_csv", "identification_results_csv", "matched_uncertain_csv", "visualized_dir"]:
        lines.append(f"- `{name}`: `{artifacts.get(name, 'не найден')}`")
    lines.extend(
        [
            "",
            "## Сводка",
            "",
            f"- строк в crops_manifest.csv: {summary.crops_rows}",
            f"- строк в identification_results.csv: {summary.identification_rows}",
            f"- matched: {summary.matched}",
            f"- matched_uncertain: {summary.matched_uncertain}",
            f"- unknown: {summary.unknown}",
            "",
            "## Замечания и проверки",
            "",
            "| Проверка | Статус | Сообщение | Путь |",
            "|---|---|---|---|",
        ]
    )
    for item in checks:
        lines.append(f"| {item.name} | {item.status} | {item.message} | `{item.path}` |")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    payload = {"summary": asdict(summary), "checks": [asdict(item) for item in checks], "artifacts": artifacts}
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"validation_report_md": report_md, "validation_summary_json": summary_json}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка выходной папки результата ShelfVision")
    parser.add_argument("--run-dir", required=True, help="Папка результата полного запуска ShelfVision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Папка результата не найдена: {run_dir}")
    summary, checks, artifacts = validate_run(run_dir)
    outputs = write_reports(run_dir, summary, checks, artifacts)
    print(f"Статус проверки: {summary.status}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if summary.status == "ERROR":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
