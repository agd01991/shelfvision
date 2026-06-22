from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

import action_history


def _signature(path: Path) -> tuple[bool, int, int]:
    try:
        stat = path.stat()
        return True, int(stat.st_size), int(stat.st_mtime_ns)
    except OSError:
        return False, 0, 0


def _last_row_details(path: Path) -> str:
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return str(path)
    if df.empty:
        return str(path)

    row = df.iloc[-1].to_dict()
    keys = [
        "image_name",
        "object_id",
        "edit_type",
        "old_sku_id",
        "new_sku_id",
        "new_status",
        "target_sku_id",
        "comment",
    ]
    values = [
        f"{key}={row.get(key)}"
        for key in keys
        if key in row and str(row.get(key)).strip()
    ]
    return "; ".join(values) or str(path)


def render_review_with_history(
    experiment_dir: str | Path,
    config: Dict[str, Any],
    render_review: Callable[[Dict[str, Any]], None],
) -> None:
    """Отрисовать ручную проверку и записать изменившиеся журналы в общую историю."""

    exp = Path(experiment_dir)
    tracked = {
        "edits": exp / "06_manual_identification" / "manual_identification_edits.csv",
        "references": exp
        / "06_manual_identification"
        / "manual_reference_suggestions.csv",
        "corrected": exp
        / "06_manual_identification"
        / "identification_results_corrected.csv",
    }
    before = {name: _signature(path) for name, path in tracked.items()}

    render_review(config)

    after = {name: _signature(path) for name, path in tracked.items()}
    if before["edits"] != after["edits"] and tracked["edits"].exists():
        action_history.append_event(
            exp,
            "manual_identification_edit",
            "Добавлена ручная правка идентификации",
            _last_row_details(tracked["edits"]),
            tracked["edits"],
        )
    if (
        before["references"] != after["references"]
        and tracked["references"].exists()
    ):
        action_history.append_event(
            exp,
            "reference_suggestion",
            "Добавлен предложенный эталон",
            _last_row_details(tracked["references"]),
            tracked["references"],
        )
    if before["corrected"] != after["corrected"] and tracked["corrected"].exists():
        action_history.append_event(
            exp,
            "manual_edits_apply",
            "Применён журнал ручных правок",
            str(tracked["corrected"]),
            tracked["corrected"],
        )
