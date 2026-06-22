from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/([A-Za-z])/(.*)$")


def _canonical(value: str | Path | None) -> str:
    raw = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    wsl = WSL_MOUNT_RE.match(raw)
    if wsl:
        raw = f"{wsl.group(1).lower()}:/{wsl.group(2)}"
    win = WINDOWS_DRIVE_RE.match(raw)
    if win:
        raw = f"{win.group(1).lower()}:/{win.group(2)}"
    return raw.rstrip("/").lower()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _manifest_roots(manifest_csv: Path) -> tuple[list[str], str]:
    try:
        df = pd.read_csv(manifest_csv).fillna("")
    except Exception:
        return [], ""
    if "image_path" not in df.columns:
        return [], ""

    paths = [_canonical(value) for value in df["image_path"].astype(str).tolist() if str(value).strip()]
    if not paths:
        return [], ""
    try:
        common = os.path.commonpath(paths).replace("\\", "/")
    except ValueError:
        common = ""
    return paths, common


def build_report(experiment_dir: Path, config_path: Path) -> dict[str, Any]:
    config = _read_yaml(config_path)
    full = config.get("full_photo_identification", {})
    config_images_dir = _canonical(full.get("images_dir"))

    manifest_csv = experiment_dir / "00_manifest" / "all_images.csv"
    environment_json = experiment_dir / "00_manifest" / "run_environment.json"
    split_json = experiment_dir / "00_manifest" / "split_params.json"

    manifest_paths, manifest_common_root = _manifest_roots(manifest_csv)
    environment = _read_json(environment_json)
    run_args = environment.get("args", {}) if isinstance(environment.get("args"), dict) else {}
    run_images_dir = _canonical(run_args.get("images_dir"))

    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    if manifest_csv.exists() and manifest_paths:
        add("manifest", "ok", f"images={len(manifest_paths)}, common_root={manifest_common_root}")
    else:
        add("manifest", "error", f"Не найден или пуст: {manifest_csv}")

    if environment_json.exists():
        add("run_environment", "ok", str(environment_json))
    else:
        add("run_environment", "warning", f"Не найден: {environment_json}")

    if split_json.exists():
        add("split_params", "ok", str(split_json))
    else:
        add("split_params", "warning", f"Не найден: {split_json}")

    if config_images_dir:
        add("config_images_dir", "ok", config_images_dir)
    else:
        add("config_images_dir", "warning", "images_dir не задан в конфигурации")

    if run_images_dir:
        add("run_images_dir", "ok", run_images_dir)
    else:
        add("run_images_dir", "warning", "images_dir не найден в run_environment.json")

    if config_images_dir and run_images_dir:
        status = "ok" if config_images_dir == run_images_dir else "error"
        add(
            "config_vs_run",
            status,
            f"config={config_images_dir}; run={run_images_dir}",
        )

    if manifest_common_root and run_images_dir:
        inside = manifest_common_root.startswith(run_images_dir) or run_images_dir.startswith(manifest_common_root)
        add(
            "manifest_vs_run",
            "ok" if inside else "error",
            f"manifest_root={manifest_common_root}; run={run_images_dir}",
        )

    errors = sum(1 for item in checks if item["status"] == "error")
    warnings = sum(1 for item in checks if item["status"] == "warning")
    status = "error" if errors else "warning" if warnings else "ok"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "config_images_dir": config_images_dir,
        "run_images_dir": run_images_dir,
        "manifest_common_root": manifest_common_root,
        "manifest_images_count": len(manifest_paths),
        "checks": checks,
        "note": "Название датасета определяется по фактическим путям и метаданным запуска, а не по имени проекта или README.",
    }


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_source_check.json"
    md_path = out_dir / "data_source_check.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Проверка источника данных эксперимента",
        "",
        f"- Статус: **{report['status']}**",
        f"- Ошибок: {report['errors']}",
        f"- Предупреждений: {report['warnings']}",
        f"- Изображений в manifest: {report['manifest_images_count']}",
        f"- Путь из конфигурации: `{report['config_images_dir'] or 'не задан'}`",
        f"- Путь из run_environment: `{report['run_images_dir'] or 'не найден'}`",
        f"- Общий корень manifest: `{report['manifest_common_root'] or 'не определён'}`",
        "",
        "## Проверки",
        "",
        "| Статус | Проверка | Детали |",
        "|---|---|---|",
    ]
    icons = {"ok": "✅", "warning": "⚠️", "error": "❌"}
    for item in report["checks"]:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {icons[item['status']]} {item['status']} | {item['name']} | {detail} |")
    lines.extend(["", report["note"]])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверить согласованность источника данных финального эксперимента")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--config", default="config/vkr_final.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir / "export"
    report = build_report(experiment_dir, config_path)
    outputs = write_report(report, out_dir)
    print(f"Data source check: {report['status']}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
