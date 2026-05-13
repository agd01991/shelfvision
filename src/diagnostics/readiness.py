from __future__ import annotations

import csv
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WEIGHT_EXTS = {".pt", ".pth", ".onnx"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ReadinessCheck:
    name: str
    status: str
    message: str
    path: str = ""
    suggestion: str = ""


@dataclass
class ReadinessSummary:
    status: str
    checks_total: int
    ok_count: int
    warning_count: int
    error_count: int
    out_dir: str


def _load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Некорректный YAML config: {path}")
    return data


def _windows_to_wsl_path(value: str) -> Optional[Path]:
    normalized = value.replace("\\", "/")
    match = WINDOWS_DRIVE_RE.match(normalized)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = match.group(2)
    return Path(f"/mnt/{drive}/{rest}")


def _candidate_paths(value: str | Path) -> List[Path]:
    text = str(value).strip()
    if not text:
        return []
    candidates = [Path(text)]
    wsl_candidate = _windows_to_wsl_path(text)
    if wsl_candidate is not None:
        candidates.append(wsl_candidate)
    return candidates


def _existing_path(value: str | Path) -> Optional[Path]:
    for candidate in _candidate_paths(value):
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def _path_text(value: Any) -> str:
    return str(value or "").strip()


def _check_required_file(name: str, path_value: str, allowed_exts: Optional[set[str]] = None) -> ReadinessCheck:
    if not path_value:
        return ReadinessCheck(name=name, status="error", message="Путь не указан", suggestion="Заполнить путь в настройках")
    existing = _existing_path(path_value)
    if existing is None:
        suggestion = "Проверить путь. Для WSL Windows-путь D:/... должен быть доступен как /mnt/d/..."
        return ReadinessCheck(name=name, status="error", message="Файл не найден", path=path_value, suggestion=suggestion)
    if not existing.is_file():
        return ReadinessCheck(name=name, status="error", message="Путь существует, но это не файл", path=str(existing))
    if allowed_exts and existing.suffix.lower() not in allowed_exts:
        return ReadinessCheck(name=name, status="warning", message=f"Неожиданное расширение: {existing.suffix}", path=str(existing))
    return ReadinessCheck(name=name, status="ok", message="Файл найден", path=str(existing))


def _check_optional_file(name: str, path_value: str, allowed_exts: Optional[set[str]] = None) -> ReadinessCheck:
    if not path_value:
        return ReadinessCheck(name=name, status="warning", message="Путь не указан, проверка пропущена")
    return _check_required_file(name, path_value, allowed_exts=allowed_exts)


def _check_output_dir(name: str, path_value: str) -> ReadinessCheck:
    if not path_value:
        return ReadinessCheck(name=name, status="error", message="Путь не указан", suggestion="Заполнить папку вывода")
    candidates = _candidate_paths(path_value)
    for candidate in candidates:
        try:
            target = candidate if candidate.suffix == "" else candidate.parent
            if target.exists():
                if target.is_dir() and os.access(str(target), os.W_OK):
                    return ReadinessCheck(name=name, status="ok", message="Папка вывода доступна для записи", path=str(target))
                return ReadinessCheck(name=name, status="error", message="Путь найден, но запись недоступна", path=str(target))
            parent = target.parent
            if parent.exists() and parent.is_dir() and os.access(str(parent), os.W_OK):
                return ReadinessCheck(name=name, status="ok", message="Папку вывода можно создать", path=str(target))
        except Exception:
            continue
    return ReadinessCheck(name=name, status="error", message="Папка вывода недоступна", path=path_value, suggestion="Проверить диск и права на запись")


def _check_gallery_csv(path_value: str) -> ReadinessCheck:
    base = _check_optional_file("SKU gallery.csv", path_value, allowed_exts={".csv"})
    if base.status != "ok":
        if base.status == "warning":
            base.suggestion = "Можно создать gallery.csv кнопкой «Проверить SKU-галерею и создать gallery.csv»"
        return base
    try:
        import pandas as pd

        df = pd.read_csv(base.path)
        required = {"sku_id", "sku_name", "image_path"}
        missing = required - set(df.columns)
        if missing:
            return ReadinessCheck(
                name="SKU gallery.csv",
                status="error",
                message=f"В CSV нет колонок: {sorted(missing)}",
                path=base.path,
                suggestion="Пересоздать gallery.csv через мастер SKU-галереи",
            )
        if len(df) == 0:
            return ReadinessCheck(name="SKU gallery.csv", status="error", message="CSV пустой", path=base.path)
        sku_count = df["sku_id"].nunique()
        return ReadinessCheck(name="SKU gallery.csv", status="ok", message=f"CSV найден: строк {len(df)}, SKU {sku_count}", path=base.path)
    except Exception as exc:
        return ReadinessCheck(name="SKU gallery.csv", status="error", message=f"Не удалось прочитать CSV: {exc}", path=base.path)


def _check_sku_gallery_dir(path_value: str, min_images_per_sku: int) -> ReadinessCheck:
    if not path_value:
        return ReadinessCheck(name="SKU gallery dir", status="error", message="Путь не указан")
    existing = _existing_path(path_value)
    if existing is None:
        return ReadinessCheck(name="SKU gallery dir", status="error", message="Папка не найдена", path=path_value, suggestion="Создать структуру sku_gallery/<sku_id>/*.jpg")
    if not existing.is_dir():
        return ReadinessCheck(name="SKU gallery dir", status="error", message="Путь существует, но это не папка", path=str(existing))
    sku_dirs = [p for p in existing.iterdir() if p.is_dir()]
    if not sku_dirs:
        return ReadinessCheck(name="SKU gallery dir", status="error", message="SKU-папки не найдены", path=str(existing), suggestion="Создать структуру sku_gallery/<sku_id>/*.jpg")
    weak = 0
    total_images = 0
    for sku_dir in sku_dirs:
        images = [p for p in sku_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        total_images += len(images)
        if len(images) < min_images_per_sku:
            weak += 1
    if total_images == 0:
        return ReadinessCheck(name="SKU gallery dir", status="error", message="Эталонные изображения не найдены", path=str(existing))
    if weak:
        return ReadinessCheck(name="SKU gallery dir", status="warning", message=f"SKU: {len(sku_dirs)}, изображений: {total_images}, слабых SKU: {weak}", path=str(existing), suggestion="Добавить эталоны или снизить min_images_per_sku")
    return ReadinessCheck(name="SKU gallery dir", status="ok", message=f"SKU: {len(sku_dirs)}, изображений: {total_images}", path=str(existing))


def _check_wsl_venv(config: Dict[str, Any]) -> ReadinessCheck:
    setup = config.get("setup", {})
    venv_dir = _path_text(setup.get("venv_dir_wsl", ".venv_wsl"))
    candidate = Path(venv_dir) / "bin" / "python"
    if candidate.exists():
        return ReadinessCheck(name="WSL .venv_wsl", status="ok", message="WSL python найден", path=str(candidate))
    return ReadinessCheck(name="WSL .venv_wsl", status="warning", message="WSL python не найден из текущего процесса", path=str(candidate), suggestion="Проверить через Control Panel кнопку проверки WSL-зависимостей")


def _check_windows_path_in_wsl(name: str, path_value: str, use_wsl_runtime: bool) -> Optional[ReadinessCheck]:
    if not use_wsl_runtime or not path_value:
        return None
    if WINDOWS_DRIVE_RE.match(path_value.replace("\\", "/")):
        wsl = _windows_to_wsl_path(path_value)
        if wsl is not None and not wsl.exists():
            return ReadinessCheck(
                name=f"WSL path compatibility: {name}",
                status="warning",
                message="Путь выглядит как Windows-путь. Внутри WSL он может быть недоступен без /mnt/d/...",
                path=path_value,
                suggestion=f"Проверить доступность: {wsl}",
            )
    return None


def build_readiness_report(config_path: str | Path, out_dir: str | Path) -> Dict[str, Path]:
    config = _load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = config.get("runtime", {})
    weights = config.get("weights", {})
    video = config.get("video", {})
    sku_gallery = config.get("sku_gallery", {})
    identification = config.get("identification", {})
    use_wsl_runtime = bool(runtime.get("use_wsl_runtime", True))

    video_model = _path_text(video.get("model", "yolo_seg"))
    video_weight = _path_text(weights.get("yolo_seg" if video_model == "yolo_seg" else "yolo", weights.get("yolo", "")))
    min_images_per_sku = int(sku_gallery.get("min_images_per_sku", 3) or 3)

    checks: List[ReadinessCheck] = [
        _check_required_file("Video input", _path_text(video.get("input_path", "")), allowed_exts=VIDEO_EXTS),
        _check_required_file(f"Video weights ({video_model})", video_weight, allowed_exts=WEIGHT_EXTS),
        _check_output_dir("Video output dir", _path_text(video.get("output_dir", "results/video/yolo"))),
        _check_sku_gallery_dir(_path_text(sku_gallery.get("gallery_dir", identification.get("gallery_dir", ""))), min_images_per_sku=min_images_per_sku),
        _check_gallery_csv(_path_text(sku_gallery.get("output_csv", identification.get("gallery_csv", "")))),
        _check_output_dir("SKU gallery report dir", _path_text(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery"))),
        _check_output_dir("Identification output dir", _path_text(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification"))),
        _check_wsl_venv(config) if use_wsl_runtime else ReadinessCheck(name="Runtime", status="ok", message="Запуск задач настроен без WSL"),
    ]

    for name, value in [
        ("video", _path_text(video.get("input_path", ""))),
        ("weights", video_weight),
        ("gallery_dir", _path_text(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "")))),
        ("gallery_csv", _path_text(sku_gallery.get("output_csv", identification.get("gallery_csv", "")))),
        ("video_out", _path_text(video.get("output_dir", ""))),
        ("identification_out", _path_text(identification.get("out_dir", ""))),
    ]:
        extra = _check_windows_path_in_wsl(name, value, use_wsl_runtime=use_wsl_runtime)
        if extra is not None:
            checks.append(extra)

    ok_count = sum(1 for item in checks if item.status == "ok")
    warning_count = sum(1 for item in checks if item.status == "warning")
    error_count = sum(1 for item in checks if item.status == "error")
    status = "ready" if error_count == 0 else "not_ready"
    if status == "ready" and warning_count:
        status = "ready_with_warnings"

    summary = ReadinessSummary(status=status, checks_total=len(checks), ok_count=ok_count, warning_count=warning_count, error_count=error_count, out_dir=str(out_dir))

    report_json = out_dir / "readiness_report.json"
    checks_csv = out_dir / "readiness_checks.csv"
    report_md = out_dir / "readiness_report.md"

    payload = {"summary": asdict(summary), "checks": [asdict(item) for item in checks]}
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with checks_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "message", "path", "suggestion"])
        writer.writeheader()
        writer.writerows([asdict(item) for item in checks])

    lines = [
        "# ShelfVision: диагностика готовности видео-идентификации",
        "",
        f"- Статус: **{summary.status}**",
        f"- OK: {summary.ok_count}",
        f"- Warning: {summary.warning_count}",
        f"- Error: {summary.error_count}",
        f"- ОС/среда: {platform.platform()}",
        "",
        "## Проверки",
        "",
        "| check | status | message | path | suggestion |",
        "|---|---|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item.name} | {item.status} | {item.message} | `{item.path}` | {item.suggestion} |")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {"report_json": report_json, "checks_csv": checks_csv, "report_md": report_md}
