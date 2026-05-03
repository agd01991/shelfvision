from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def quote_bash(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_wsl_python(venv_dir: str, script: str, script_args: List[str]) -> int:
    venv_dir = venv_dir.replace("\\", "/")
    script = script.replace("\\", "/")
    args = " ".join(quote_bash(arg.replace("\\", "/")) for arg in script_args)
    command = f"cd \"$(wslpath '{ROOT}')\" && {quote_bash(f'{venv_dir}/bin/python')} {quote_bash(script)} {args}"
    result = subprocess.run(["wsl", "bash", "-lc", command], cwd=str(ROOT), text=True)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShelfVision script through WSL virtual environment")
    parser.add_argument("--venv-dir", default=".venv_wsl", help="WSL venv directory inside repository")
    parser.add_argument("script", help="Python script to run, e.g. run_inference.py")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the target script")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_wsl_python(args.venv_dir, args.script, args.script_args))


if __name__ == "__main__":
    main()
