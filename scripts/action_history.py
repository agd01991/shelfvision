from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

EVENT_COLUMNS = ["event_id", "created_at", "event_type", "title", "details", "snapshot_path"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def history_dir(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / "history"


def events_csv(experiment_dir: str | Path) -> Path:
    return history_dir(experiment_dir) / "events.csv"


def checkpoints_dir(experiment_dir: str | Path) -> Path:
    return history_dir(experiment_dir) / "checkpoints"


def _safe_name(value: str) -> str:
    allowed = []
    for ch in value.strip().replace(" ", "_"):
        allowed.append(ch if ch.isalnum() or ch in "._-" else "_")
    name = "".join(allowed).strip("_")
    return name or "checkpoint"


def append_event(
    experiment_dir: str | Path,
    event_type: str,
    title: str,
    details: str = "",
    snapshot_path: str | Path | None = None,
) -> Path:
    path = events_csv(experiment_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "event_id": uuid.uuid4().hex[:12],
                "created_at": _now(),
                "event_type": event_type,
                "title": title,
                "details": details,
                "snapshot_path": str(snapshot_path or ""),
            }
        )
    return path


def create_checkpoint(
    experiment_dir: str | Path,
    title: str,
    config: Dict[str, Any] | None = None,
    note: str = "",
    extra: Dict[str, Any] | None = None,
) -> Path:
    exp = Path(experiment_dir)
    out_dir = checkpoints_dir(exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{_safe_name(title)}.json"
    path = out_dir / filename
    payload = {
        "created_at": _now(),
        "title": title,
        "note": note,
        "experiment_dir": str(exp),
        "config": config or {},
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(exp, "checkpoint", title, note, path)
    return path


def list_checkpoints(experiment_dir: str | Path) -> List[Path]:
    root = checkpoints_dir(experiment_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), reverse=True)


def read_checkpoint(path: str | Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_events(experiment_dir: str | Path) -> List[Dict[str, str]]:
    path = events_csv(experiment_dir)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []
