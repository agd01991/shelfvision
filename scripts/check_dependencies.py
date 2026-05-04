from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


try:
    from packaging.requirements import Requirement
except Exception:  # packaging может отсутствовать в частично установленной среде
    try:
        from pip._vendor.packaging.requirements import Requirement  # type: ignore
    except Exception:
        Requirement = None  # type: ignore


@dataclass
class RequirementStatus:
    raw: str
    name: str
    installed: bool
    installed_version: str = ""
    version_ok: Optional[bool] = None
    reason: str = ""


def read_requirement_lines(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"requirements.txt не найден: {path}")

    lines: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement")):
            # В проекте сейчас нет вложенных requirements, но строку явно показываем.
            lines.append(line)
            continue
        if line.startswith(("-f ", "--find-links", "--extra-index-url", "--index-url")):
            continue
        lines.append(line)
    return lines


def fallback_name(raw: str) -> str:
    separators = ["==", ">=", "<=", "~=", "!=", ">", "<", "["]
    name = raw
    for sep in separators:
        if sep in name:
            name = name.split(sep, 1)[0]
    return name.strip().replace("_", "-")


def check_one_requirement(raw: str) -> RequirementStatus:
    if raw.startswith(("-r ", "--requirement")):
        return RequirementStatus(raw=raw, name=raw, installed=True, version_ok=None, reason="nested requirement is not expanded by this checker")

    req = None
    if Requirement is not None:
        try:
            req = Requirement(raw)
            name = req.name
        except Exception:
            name = fallback_name(raw)
    else:
        name = fallback_name(raw)

    try:
        installed_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return RequirementStatus(raw=raw, name=name, installed=False, reason="not installed")

    version_ok: Optional[bool] = None
    reason = "installed"
    if req is not None and str(req.specifier):
        try:
            version_ok = bool(req.specifier.contains(installed_version, prereleases=True))
            reason = "version ok" if version_ok else f"version mismatch: required {req.specifier}, installed {installed_version}"
        except Exception as exc:
            reason = f"installed, version check skipped: {exc}"

    return RequirementStatus(
        raw=raw,
        name=name,
        installed=True,
        installed_version=installed_version,
        version_ok=version_ok,
        reason=reason,
    )


def run_pip_check() -> int:
    print("\n=== pip check ===")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.strip() or "pip check не вернул текстового вывода")
    return int(result.returncode)


def print_statuses(statuses: List[RequirementStatus]) -> None:
    print("\n=== requirements status ===")
    for item in statuses:
        if not item.installed:
            print(f"[MISSING] {item.raw} -> {item.name}: {item.reason}")
        elif item.version_ok is False:
            print(f"[VERSION] {item.raw} -> {item.name} {item.installed_version}: {item.reason}")
        else:
            suffix = f" {item.installed_version}" if item.installed_version else ""
            print(f"[OK] {item.raw} -> {item.name}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check installed packages against requirements.txt")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when missing/version mismatched packages are found")
    parser.add_argument("--skip-pip-check", action="store_true", help="Do not run python -m pip check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== ShelfVision dependency check ===")
    print(f"Python:       {sys.executable}")
    print(f"Requirements: {Path(args.requirements).resolve()}")

    raw_requirements = read_requirement_lines(args.requirements)
    statuses = [check_one_requirement(raw) for raw in raw_requirements]
    print_statuses(statuses)

    missing = [item for item in statuses if not item.installed]
    mismatched = [item for item in statuses if item.version_ok is False]

    pip_check_code = 0 if args.skip_pip_check else run_pip_check()

    print("\n=== summary ===")
    print(f"Requirements total: {len(statuses)}")
    print(f"Installed:          {sum(1 for item in statuses if item.installed)}")
    print(f"Missing:            {len(missing)}")
    print(f"Version mismatch:   {len(mismatched)}")
    print(f"pip check code:     {pip_check_code}")

    if missing:
        print("\nMissing packages:")
        for item in missing:
            print(f"- {item.raw}")

    if mismatched:
        print("\nVersion mismatches:")
        for item in mismatched:
            print(f"- {item.raw}: installed {item.installed_version}")

    if args.strict and (missing or mismatched or pip_check_code != 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
