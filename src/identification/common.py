from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("predictions JSON должен быть объектом или списком объектов")


def resolve_image_path(image_path: str, images_dir: str | Path | None = None) -> Path:
    raw = Path(str(image_path))
    if raw.exists():
        return raw
    if raw.is_absolute():
        return raw
    if images_dir:
        candidate = Path(images_dir) / raw.name
        if candidate.exists():
            return candidate
        candidate = Path(images_dir) / raw
        if candidate.exists():
            return candidate
    return raw


def safe_stem(path: str | Path) -> str:
    return Path(str(path)).stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
