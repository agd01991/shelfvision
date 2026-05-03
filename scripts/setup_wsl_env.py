from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str]) -> None:
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WSL virtual environment and install ShelfVision dependencies")
    parser.add_argument("--venv-dir", default=".venv_wsl", help="Linux venv path inside repository")
    parser.add_argument("--requirements", default="requirements.txt", help="requirements.txt path")
    parser.add_argument("--upgrade-pip", action="store_true", default=True, help="Upgrade pip before install")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = args.venv_dir.replace("\\", "/")
    requirements = args.requirements.replace("\\", "/")

    script = f"""
set -e
cd "$(wslpath '{ROOT}')"
echo '[1/4] Current WSL directory:'
pwd
if ! command -v python3 >/dev/null 2>&1; then
  echo 'python3 is not installed in WSL. Install it first: sudo apt update && sudo apt install -y python3 python3-venv python3-pip'
  exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo 'python3-venv is not available. Run: sudo apt update && sudo apt install -y python3-venv python3-pip'
  exit 1
fi
echo '[2/4] Creating WSL virtual environment: {venv_dir}'
python3 -m venv '{venv_dir}'
echo '[3/4] Upgrading pip'
'{venv_dir}/bin/python' -m pip install --upgrade pip
echo '[4/4] Installing requirements: {requirements}'
'{venv_dir}/bin/python' -m pip install -r '{requirements}'
echo 'WSL environment is ready: {venv_dir}'
""".strip()

    run(["wsl", "bash", "-lc", script])


if __name__ == "__main__":
    main()
