from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

PROJECT_REQUIRED_FILES = [
    "README.md",
    "config/vkr_final.yaml",
    "docs/DEFENSE_FAQ.md",
    "docs/SIMILAR_PROJECTS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/DEMO_SCRIPT_5_MIN.md",
    "data/README.md",
    "scripts/final_demo_app.py",
    "scripts/final_demo_history_app.py",
    "scripts/action_history.py",
    "scripts/identification_review_panel.py",
    "src/identification/manual_identification_editor.py",
    "src/identification/selected_sku_exporter.py",
    "src/reporting/defense_export.py",
]

EXPERIMENT_REQUIRED_FILES = [
    "04_identification/identification_results.csv",
]

EXPERIMENT_RECOMMENDED_FILES = [
    "00_manifest/all_images.csv",
    "00_manifest/gallery_images.csv",
    "00_manifest/query_images.csv",
    "00_manifest/split_params.json",
    "00_manifest/run_environment.json",
    "01_gallery_inference/predictions.json",
    "02_demo_gallery/demo_sku_gallery_summary.json",
    "03_query_inference/predictions.json",
    "04_identification/crops_manifest.csv",
    "05_reports/full_experiment_summary.md",
    "05_reports/full_experiment_summary.json",
    "05_reports/threshold_analysis.csv",
    "06_manual_identification/manual_identification_edits.csv",
    "06_manual_identification/identification_results_corrected.csv",
    "history/events.csv",
    "selected_sku_demo/selected_sku_report.md",
    "export/demo_artifacts.zip",
]

IMPORT_CHECKS = [
    "pandas",
    "yaml",
    "streamlit",
    "action_history",
    "src.identification.manual_identification_editor",
    "src.identification.selected_sku_exporter",
    "src.reporting.defense_export",
]

OPTIONAL_IMPORT_CHECKS = ["cv2"]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class SmokeReport:
    python: str
    platform: str
    project_root: str
    experiment_dir: str
    checks: List[CheckResult]

    @property
    def errors_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "error")

    @property
    def warnings_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warning")

    @property
    def status(self) -> str:
        return "error" if self.errors_count else "warning" if self.warnings_count else "ok"


def _check_file(path: Path, name: str, missing_status: str = "error") -> CheckResult:
    if path.exists():
        return CheckResult(name=name, status="ok", detail=str(path))
    return CheckResult(name=name, status=missing_status, detail=f"not found: {path}")


def _check_import(module_name: str, missing_status: str = "error") -> CheckResult:
    try:
        importlib.import_module(module_name)
        return CheckResult(name=f"import {module_name}", status="ok")
    except Exception as exc:
        return CheckResult(
            name=f"import {module_name}",
            status=missing_status,
            detail=str(exc),
        )


def build_report(experiment_dir: Path | None = None) -> SmokeReport:
    checks: List[CheckResult] = []

    for rel in PROJECT_REQUIRED_FILES:
        checks.append(_check_file(ROOT / rel, f"project file {rel}", missing_status="error"))

    for module in IMPORT_CHECKS:
        checks.append(_check_import(module, missing_status="error"))
    for module in OPTIONAL_IMPORT_CHECKS:
        checks.append(_check_import(module, missing_status="warning"))

    exp_text = ""
    if experiment_dir is not None:
        exp = Path(experiment_dir)
        exp_text = str(exp)
        if not exp.exists():
            checks.append(CheckResult("experiment directory", "warning", f"not found: {exp}"))
        else:
            checks.append(CheckResult("experiment directory", "ok", str(exp)))
            for rel in EXPERIMENT_REQUIRED_FILES:
                checks.append(
                    _check_file(
                        exp / rel,
                        f"experiment required {rel}",
                        missing_status="error",
                    )
                )
            for rel in EXPERIMENT_RECOMMENDED_FILES:
                checks.append(
                    _check_file(
                        exp / rel,
                        f"experiment recommended {rel}",
                        missing_status="warning",
                    )
                )

    return SmokeReport(
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        project_root=str(ROOT),
        experiment_dir=exp_text,
        checks=checks,
    )


def write_report(report: SmokeReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "demo_smoke_report.json"
    md_path = out_dir / "demo_smoke_report.md"

    payload = asdict(report)
    payload["status"] = report.status
    payload["errors_count"] = report.errors_count
    payload["warnings_count"] = report.warnings_count
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Smoke-проверка демонстрационного интерфейса",
        "",
        f"- Статус: **{report.status}**",
        f"- Ошибок: {report.errors_count}",
        f"- Предупреждений: {report.warnings_count}",
        f"- Python: `{report.python}`",
        f"- Platform: `{report.platform}`",
        f"- Project root: `{report.project_root}`",
        f"- Experiment dir: `{report.experiment_dir or 'не указан'}`",
        "",
        "## Проверки",
        "",
        "| Статус | Проверка | Детали |",
        "|---|---|---|",
    ]
    for check in report.checks:
        icon = "✅" if check.status == "ok" else "⚠️" if check.status == "warning" else "❌"
        detail = check.detail.replace("|", "\\|") if check.detail else ""
        lines.append(f"| {icon} {check.status} | {check.name} | {detail} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "md": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-проверка демонстрационного интерфейса")
    parser.add_argument(
        "--experiment-dir",
        default="",
        help="Папка эксперимента, если нужно проверить его артефакты",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Куда сохранить отчет. По умолчанию: experiment/export или results/demo_smoke",
    )
    parser.add_argument("--strict", action="store_true", help="Вернуть код 1 при ошибках")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir.strip() else None
    report = build_report(experiment_dir=experiment_dir)

    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    elif experiment_dir is not None:
        out_dir = experiment_dir / "export"
    else:
        out_dir = ROOT / "results" / "demo_smoke"
    outputs = write_report(report, out_dir)

    print(f"Demo smoke status: {report.status}")
    print(f"Errors: {report.errors_count}, warnings: {report.warnings_count}")
    for name, path in outputs.items():
        print(f"{name}: {path}")

    if args.strict and report.errors_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
