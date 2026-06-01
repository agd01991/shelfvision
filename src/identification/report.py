from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import load_predictions, save_json
from .matcher import IdentificationResult, result_to_dict


def _result_key(result: IdentificationResult) -> tuple[str, int]:
    return (Path(result.image_path).name, result.object_id)


def _get_indexed(values: List[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default


def build_identified_predictions(
    predictions_json: str | Path,
    results: List[IdentificationResult],
) -> List[Dict[str, Any]]:
    predictions = load_predictions(predictions_json)
    result_by_key = {_result_key(item): item for item in results}

    identified: List[Dict[str, Any]] = []
    for prediction in predictions:
        image_name = Path(str(prediction.get("image_path", ""))).name
        boxes = prediction.get("boxes", []) or []
        scores = prediction.get("scores", []) or []
        labels = prediction.get("labels", []) or []
        class_ids = prediction.get("class_ids", []) or []
        masks = prediction.get("masks", []) or []
        track_ids = prediction.get("track_ids", []) or []

        detections: List[Dict[str, Any]] = []
        for idx, box in enumerate(boxes, start=1):
            array_index = idx - 1
            matched = result_by_key.get((image_name, idx))
            detection = {
                "object_id": idx,
                "box": box,
                "score": _get_indexed(scores, array_index, 0.0),
                "label": _get_indexed(labels, array_index, "product"),
                "class_id": _get_indexed(class_ids, array_index, 0),
                "mask": _get_indexed(masks, array_index, None),
                "track_id": _get_indexed(track_ids, array_index, None),
            }
            if matched:
                detection.update(
                    {
                        "crop_path": matched.crop_path,
                        "sku_id": matched.sku_id,
                        "sku_name": matched.sku_name,
                        "sku_confidence": matched.sku_confidence,
                        "sku_status": matched.sku_status,
                        "safe_sku_id": matched.safe_sku_id,
                        "safe_sku_name": matched.safe_sku_name,
                        "best_distinct_sku": matched.best_distinct_sku,
                        "best_distinct_score": matched.best_distinct_score,
                        "second_distinct_sku": matched.second_distinct_sku,
                        "second_distinct_score": matched.second_distinct_score,
                        "distinct_margin": matched.distinct_margin,
                        "sku_top_k": [candidate.__dict__ for candidate in matched.top_k],
                        "track_id": matched.track_id,
                        "track_stabilized": matched.track_stabilized,
                        "track_frames_count": matched.track_frames_count,
                        "track_matched_votes": matched.track_matched_votes,
                        "track_unknown_votes": matched.track_unknown_votes,
                    }
                )
            detections.append(detection)

        enriched = dict(prediction)
        enriched["detections"] = detections
        enriched["identified_objects_count"] = sum(1 for item in detections if item.get("sku_status") == "matched")
        enriched["matched_uncertain_objects_count"] = sum(1 for item in detections if item.get("sku_status") == "matched_uncertain")
        enriched["unknown_objects_count"] = sum(1 for item in detections if item.get("sku_status") == "unknown")
        enriched["tracked_objects_count"] = sum(1 for item in detections if item.get("track_id") is not None)
        identified.append(enriched)
    return identified


def save_identification_outputs(
    predictions_json: str | Path,
    results: List[IdentificationResult],
    metrics: Dict[str, Any],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    save_json([result_to_dict(item) for item in results], out_dir / "identification_results.json")
    save_json(metrics, out_dir / "identification_metrics.json")
    save_json(build_identified_predictions(predictions_json, results), out_dir / "identified_predictions.json")
    save_identification_report(results=results, metrics=metrics, out_dir=out_dir)


def save_identification_report(
    results: List[IdentificationResult],
    metrics: Dict[str, Any],
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    stabilized_count = sum(1 for item in results if item.track_stabilized)
    tracks_count = len({item.track_id for item in results if item.track_id is not None})
    lines = [
        "# ShelfVision: отчёт по SKU-идентификации",
        "",
        "## Сводка",
        "",
        f"- Всего объектов: {metrics.get('total_objects', 0)}",
        f"- Сопоставлено с SKU уверенно: {metrics.get('matched', 0)}",
        f"- Matched uncertain: {metrics.get('matched_uncertain', 0)}",
        f"- Unknown: {metrics.get('unknown', 0)}",
        f"- Доля matched: {metrics.get('matched_rate', 0):.4f}",
        f"- Доля matched uncertain: {metrics.get('matched_uncertain_rate', 0):.4f}",
        f"- Доля unknown: {metrics.get('unknown_rate', 0):.4f}",
        f"- Средняя similarity: {metrics.get('avg_similarity', 0):.4f}",
        f"- Средний distinct margin: {metrics.get('mean_distinct_margin', 0):.4f}",
        f"- Треков в видео: {tracks_count}",
        f"- Объектов со стабилизированным SKU по треку: {stabilized_count}",
    ]
    if "top1_accuracy" in metrics:
        lines.extend(
            [
                f"- Top-1 accuracy: {metrics.get('top1_accuracy', 0):.4f}",
                f"- Top-k accuracy: {metrics.get('topk_accuracy', 0):.4f}",
                f"- False match rate: {metrics.get('false_match_rate', 0):.4f}",
            ]
        )

    lines.extend(["", "## Первые результаты", ""])
    lines.append("| image | object | track | status | sku | safe sku | confidence | margin | crop |")
    lines.append("|---|---:|---:|---|---|---|---:|---:|---|")
    for item in results[:30]:
        track = item.track_id if item.track_id is not None else ""
        margin = item.distinct_margin if item.distinct_margin is not None else 0.0
        safe_sku = item.safe_sku_name or ""
        lines.append(
            f"| {item.image_name} | {item.object_id} | {track} | {item.sku_status} | {item.sku_name} | "
            f"{safe_sku} | {item.sku_confidence:.4f} | {margin:.4f} | `{item.crop_path}` |"
        )

    report_path = out_dir / "identification_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
