from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

from path_utils import to_wsl_arg


ROOT = Path(__file__).resolve().parents[1]


def quote_bash(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def normalize_wsl_arg(arg: str) -> str:
    if not isinstance(arg, str):
        return str(arg)
    return to_wsl_arg(arg)


def normalize_wsl_venv_dir(venv_dir: str) -> str:
    return to_wsl_arg(venv_dir)


def run_wsl_python(venv_dir: str, python_args: List[str]) -> int:
    venv_dir = normalize_wsl_venv_dir(venv_dir)
    normalized_args = [normalize_wsl_arg(arg) for arg in python_args]
    args = " ".join(quote_bash(arg) for arg in normalized_args)
    command = f"cd \"$(wslpath '{ROOT}')\" && {quote_bash(f'{venv_dir}/bin/python')} {args}"
    print("WSL command:", command, flush=True)
    result = subprocess.run(["wsl", "bash", "-lc", command], cwd=str(ROOT), text=True)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ShelfVision Python command through WSL virtual environment",
        usage="python scripts/wsl_runtime.py [--venv-dir .venv_wsl] <python args...>",
    )
    parser.add_argument("--venv-dir", default=".venv_wsl", help="WSL venv directory inside repository")
    known, python_args = parser.parse_known_args()
    if not python_args:
        parser.error("pass Python arguments, for example: run_inference.py --help or -m streamlit run scripts/video_app.py")
    known.python_args = python_args
    return known


def main() -> None:
    args = parse_args()
    raise SystemExit(run_wsl_python(args.venv_dir, args.python_args))


if __name__ == "__main__":
    main()
