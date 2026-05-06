from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .matcher import IdentificationResult


@dataclass
class TrackSkuSummary:
    track_id: int
    frames_count: int
    objects_count: int
    stable_sku_id: Optional[str]
    stable_sku_name: str
    stable_confidence: float
    stable_status: str
    matched_votes: int
    unknown_votes: int


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_key(result: IdentificationResult) -> tuple[str, int]:
    return (Path(result.image_path).name, int(result.object_id))


def _prediction_key(prediction: Dict[str, Any]) -> str:
    return Path(str(prediction.get("image_path", ""))).name


def build_detection_track_index(video_predictions_json: str | Path) -> Dict[tuple[str, int], Optional[int]]:
    predictions = _read_json(video_predictions_json)
    index: Dict[tuple[str, int], Optional[int]] = {}
    for prediction in predictions:
        image_name = _prediction_key(prediction)
        boxes = prediction.get("boxes", []) or []
        track_ids = prediction.get("track_ids", []) or []
        for object_index, _ in enumerate(boxes, start=1):
            track_id = track_ids[object_index - 1] if object_index - 1 < len(track_ids) else None
            index[(image_name, object_index)] = int(track_id) if track_id is not None else None
    return index


def _choose_stable_sku(track_results: List[IdentificationResult]) -> TrackSkuSummary:
    track_id = int(getattr(track_results[0], "track_id", 0) or 0)
    frames = {Path(item.image_path).name for item in track_results}
    matched = [item for item in track_results if item.sku_status == "matched" and item.sku_id]
    unknown_votes = len(track_results) - len(matched)

    if not matched:
        return TrackSkuSummary(
            track_id=track_id,
            frames_count=len(frames),
            objects_count=len(track_results),
            stable_sku_id=None,
            stable_sku_name="unknown",
            stable_confidence=0.0,
            stable_status="unknown",
            matched_votes=0,
            unknown_votes=unknown_votes,
        )

    grouped: Dict[str, List[IdentificationResult]] = {}
    for item in matched:
        grouped.setdefault(str(item.sku_id), []).append(item)

    def score_group(items: List[IdentificationResult]) -> tuple[int, float]:
        return (len(items), sum(item.sku_confidence for item in items) / len(items))

    best_sku_id, best_items = max(grouped.items(), key=lambda pair: score_group(pair[1]))
    avg_confidence = sum(item.sku_confidence for item in best_items) / len(best_items)
    return TrackSkuSummary(
        track_id=track_id,
        frames_count=len(frames),
        objects_count=len(track_results),
        stable_sku_id=best_sku_id,
        stable_sku_name=best_items[0].sku_name,
        stable_confidence=avg_confidence,
        stable_status="matched",
        matched_votes=len(best_items),
        unknown_votes=unknown_votes,
    )


def stabilize_results_by_tracks(
    results: List[IdentificationResult],
    video_predictions_json: str | Path,
) -> tuple[List[IdentificationResult], List[TrackSkuSummary]]:
    track_index = build_detection_track_index(video_predictions_json)

    by_track: Dict[int, List[IdentificationResult]] = {}
    for result in results:
        track_id = track_index.get(_result_key(result))
        setattr(result, "track_id", track_id)
        if track_id is not None:
            by_track.setdefault(track_id, []).append(result)

    summaries = [_choose_stable_sku(items) for _, items in sorted(by_track.items()) if items]
    summary_by_track = {item.track_id: item for item in summaries}

    for result in results:
        track_id = getattr(result, "track_id", None)
        summary = summary_by_track.get(track_id)
        if summary is None:
            continue
        result.sku_id = summary.stable_sku_id
        result.sku_name = summary.stable_sku_name
        result.sku_confidence = summary.stable_confidence
        result.sku_status = summary.stable_status
        setattr(result, "track_stabilized", True)
        setattr(result, "track_frames_count", summary.frames_count)
        setattr(result, "track_matched_votes", summary.matched_votes)

    return results, summaries


def save_track_summaries(summaries: List[TrackSkuSummary], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "track_sku_summary.json"
    output_path.write_text(json.dumps([asdict(item) for item in summaries], ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
