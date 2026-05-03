from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


CLI_SCRIPTS = [
    "run_inference.py",
    "run_evaluation.py",
    "run_recommendation.py",
    "run_compare.py",
    "run_density.py",
    "run_mini_report.py",
    "run_full_pipeline.py",
]


MODULES = [
    "src.inference.prediction",
    "src.inference.yolo_inference",
    "src.inference.rtdetr_inference",
    "src.inference.faster_rcnn_inference",
    "src.inference.ensemble_wbf",
    "src.visualization.draw_boxes",
    "src.evaluation.metrics",
    "src.evaluation.error_visualization",
    "src.evaluation.recommend_model",
    "src.evaluation.compare_models",
    "src.analytics.density",
    "src.reporting.mini_report",
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def run_help_check(script_name: str, timeout: int = 30) -> CheckResult:
    script_path = ROOT / script_name
    if not script_path.exists():
        return CheckResult(script_name, False, "file not found")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as exc:
        return CheckResult(script_name, False, str(exc))

    if result.returncode != 0:
        return CheckResult(script_name, False, result.stdout[-1000:])
    return CheckResult(script_name, True, "--help ok")


def run_import_check(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(module_name, False, str(exc))
    return CheckResult(module_name, True, "import ok")


def print_results(title: str, results: List[CheckResult]) -> None:
    print(f"\n=== {title} ===")
    for item in results:
        status = "OK" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision smoke checks for CLI scripts and modules")
    parser.add_argument("--skip-help", action="store_true", help="Не проверять запуск CLI-скриптов с --help")
    parser.add_argument("--skip-imports", action="store_true", help="Не проверять импорт модулей")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout для одного --help запуска")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: List[CheckResult] = []

    if not args.skip_imports:
        import_results = [run_import_check(module) for module in MODULES]
        print_results("Import checks", import_results)
        all_results.extend(import_results)

    if not args.skip_help:
        help_results = [run_help_check(script, timeout=args.timeout) for script in CLI_SCRIPTS]
        print_results("CLI --help checks", help_results)
        all_results.extend(help_results)

    failed = [item for item in all_results if not item.ok]
    print("\n=== Summary ===")
    print(f"Total checks: {len(all_results)}")
    print(f"Passed:       {len(all_results) - len(failed)}")
    print(f"Failed:       {len(failed)}")

    if failed:
        print("\nFailed checks:")
        for item in failed:
            print(f"- {item.name}: {item.details}")
        raise SystemExit(1)

    print("\nSmoke checks passed.")


if __name__ == "__main__":
    main()
