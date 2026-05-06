from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2

from .matcher import IdentificationResult


BOX_COLOR_MATCHED = (0, 180, 0)
BOX_COLOR_UNKNOWN = (0, 0, 220)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)


@dataclass
class IdentifiedVideoSummary:
    input_video: str
    output_video: str
    frames_used: int
    fps: float
    width: int
    height: int
    matched_objects: int
    unknown_objects: int
    tracked_objects: int


def _draw_label(image, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, h + baseline + 6)
    cv2.rectangle(image, (x, y - h - baseline - 6), (x + w + 8, y + 3), TEXT_BG_COLOR, -1)
    cv2.putText(image, text, (x + 4, y - baseline - 2), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_identification_results(path: str | Path) -> List[IdentificationResult]:
    raw = _read_json(path)
    results: List[IdentificationResult] = []
    for item in raw:
        results.append(
            IdentificationResult(
                image_path=str(item.get("image_path", "")),
                image_name=str(item.get("image_name", "")),
                object_id=int(item.get("object_id", 0)),
                crop_path=str(item.get("crop_path", "")),
                x1=float(item.get("x1", 0.0)),
                y1=float(item.get("y1", 0.0)),
                x2=float(item.get("x2", 0.0)),
                y2=float(item.get("y2", 0.0)),
                source_type=str(item.get("source_type", "bbox")),
                detection_score=float(item.get("detection_score", 0.0)),
                label=str(item.get("label", "product")),
                class_id=int(item.get("class_id", 0)),
                sku_id=item.get("sku_id"),
                sku_name=str(item.get("sku_name", "unknown")),
                sku_confidence=float(item.get("sku_confidence", 0.0)),
                sku_status=str(item.get("sku_status", "unknown")),
                top_k=[],
                track_id=item.get("track_id"),
                track_stabilized=bool(item.get("track_stabilized", False)),
                track_frames_count=int(item.get("track_frames_count", 0) or 0),
                track_matched_votes=int(item.get("track_matched_votes", 0) or 0),
                track_unknown_votes=int(item.get("track_unknown_votes", 0) or 0),
            )
        )
    return results


def _group_results_by_image(results: Iterable[IdentificationResult]) -> Dict[str, List[IdentificationResult]]:
    grouped: Dict[str, List[IdentificationResult]] = {}
    for item in results:
        grouped.setdefault(str(Path(item.image_path)), []).append(item)
    return grouped


def _frame_sort_key(prediction: Dict[str, Any]) -> tuple[int, int, str]:
    metadata = prediction.get("metadata", {}) or {}
    source_frame_id = int(metadata.get("source_frame_id", 0))
    processed_frame_id = int(metadata.get("processed_frame_id", 0))
    return source_frame_id, processed_frame_id, str(prediction.get("image_path", ""))


def _output_fps(video_summary: Dict[str, Any], input_video: str | Path) -> float:
    frame_skip = max(1, int(video_summary.get("frame_skip", 1) or 1))
    capture = cv2.VideoCapture(str(input_video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()
    return max(1.0, fps / frame_skip)


def _draw_identification(image, results: List[IdentificationResult]) -> tuple[int, int, int]:
    matched = 0
    unknown = 0
    tracked = 0
    for item in results:
        color = BOX_COLOR_MATCHED if item.sku_status == "matched" else BOX_COLOR_UNKNOWN
        if item.sku_status == "matched":
            matched += 1
        else:
            unknown += 1
        if item.track_id is not None:
            tracked += 1
        x1, y1, x2, y2 = [int(round(v)) for v in (item.x1, item.y1, item.x2, item.y2)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        track_prefix = f"T{item.track_id} " if item.track_id is not None else ""
        stable_mark = "*" if item.track_stabilized else ""
        label = (
            f"{track_prefix}{item.sku_name}{stable_mark} {item.sku_confidence:.2f}"
            if item.sku_status == "matched"
            else f"{track_prefix}unknown {item.sku_confidence:.2f}"
        )
        _draw_label(image, label, x1, y1)
    return matched, unknown, tracked


def render_identified_video(
    video_predictions_json: str | Path,
    identification_results_json: str | Path,
    video_summary_json: str | Path,
    out_dir: str | Path,
    output_name: str = "identified_output_video.mp4",
    codec: str = "mp4v",
) -> Dict[str, Path]:
    """Builds an MP4 where video detections are labeled with SKU identification results."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_predictions = _read_json(video_predictions_json)
    video_summary = _read_json(video_summary_json)
    identification_results = _load_identification_results(identification_results_json)
    grouped = _group_results_by_image(identification_results)

    frames = sorted(video_predictions, key=_frame_sort_key)
    if not frames:
        raise ValueError("video_predictions.json не содержит кадров")

    first_frame_path = Path(frames[0].get("image_path", ""))
    first_image = cv2.imread(str(first_frame_path))
    if first_image is None:
        raise FileNotFoundError(f"Не удалось открыть первый кадр: {first_frame_path}")

    height, width = first_image.shape[:2]
    input_video = video_summary.get("input_video", "")
    fps = _output_fps(video_summary, input_video) if input_video else 25.0
    output_video_path = out_dir / output_name
    writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))

    matched_total = 0
    unknown_total = 0
    tracked_total = 0
    frames_used = 0

    for frame_prediction in frames:
        frame_path = Path(str(frame_prediction.get("image_path", "")))
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))

        frame_results = grouped.get(str(frame_path), [])
        matched, unknown, tracked = _draw_identification(image, frame_results)
        matched_total += matched
        unknown_total += unknown
        tracked_total += tracked
        writer.write(image)
        frames_used += 1

    writer.release()

    summary = IdentifiedVideoSummary(
        input_video=str(input_video),
        output_video=str(output_video_path),
        frames_used=frames_used,
        fps=fps,
        width=width,
        height=height,
        matched_objects=matched_total,
        unknown_objects=unknown_total,
        tracked_objects=tracked_total,
    )
    summary_path = out_dir / "identified_video_summary.json"
    summary_path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"identified_video": output_video_path, "identified_video_summary": summary_path}
