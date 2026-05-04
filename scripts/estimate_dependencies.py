from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from check_dependencies import RequirementStatus, check_one_requirement, read_requirement_lines


# Вес не задаёт фиксированное время всей установки. Он используется только для
# автоматической оценки по фактически недостающим пакетам.
PACKAGE_COST_SECONDS = {
    "torch": 420,
    "torchvision": 240,
    "ultralytics": 240,
    "opencv-python": 180,
    "scipy": 180,
    "streamlit": 160,
    "pandas": 120,
    "matplotlib": 120,
    "plotly": 90,
    "pyarrow": 90,
    "numba": 90,
    "nvidia": 180,
    "cuda": 180,
    "triton": 180,
}


def _normalized_name(status: RequirementStatus) -> str:
    return status.name.lower().replace("_", "-")


def _cost_for(status: RequirementStatus) -> int:
    name = _normalized_name(status)
    for marker, seconds in PACKAGE_COST_SECONDS.items():
        if name == marker or name.startswith(marker + "-") or marker in name:
            return seconds
    return 35


def estimate_dependency_seconds(requirements: str | Path, assume_empty: bool = False) -> Dict[str, object]:
    raw_requirements = read_requirement_lines(requirements)

    if assume_empty:
        statuses = [
            RequirementStatus(raw=raw, name=raw.split("==", 1)[0].split(">=", 1)[0].strip(), installed=False, reason="assume empty venv")
            for raw in raw_requirements
        ]
    else:
        statuses = [check_one_requirement(raw) for raw in raw_requirements]

    need_install = [item for item in statuses if not item.installed or item.version_ok is False]
    heavy = [item.raw for item in need_install if _cost_for(item) >= 120]
    estimated = 30 + sum(_cost_for(item) for item in need_install)

    # Верхняя граница защищает UI от слишком пессимистичных ETA при медленной сети.
    estimated = max(30, min(7200, estimated))

    return {
        "estimated_seconds": estimated,
        "requirements_total": len(statuses),
        "need_install_count": len(need_install),
        "already_ok_count": len(statuses) - len(need_install),
        "heavy_packages": heavy,
        "need_install": [item.raw for item in need_install],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate pip install time from current environment and requirements.txt")
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--assume-empty", action="store_true", help="Estimate as if no packages from requirements are installed")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate_dependency_seconds(args.requirements, assume_empty=args.assume_empty)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    print("=== dependency install estimate ===")
    print(f"Requirements total: {result['requirements_total']}")
    print(f"Already OK:          {result['already_ok_count']}")
    print(f"Need install:        {result['need_install_count']}")
    print(f"Estimated seconds:   {result['estimated_seconds']}")
    if result["heavy_packages"]:
        print("Heavy packages:")
        for item in result["heavy_packages"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
