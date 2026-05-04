from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import load_predictions, save_json
from .matcher import IdentificationResult, result_to_dict


def _result_key(result: IdentificationResult) -> tuple[str, int]:
    return (Path(result.image_path).name, result.object_id)


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
        detections: List[Dict[str, Any]] = []
        for idx, box in enumerate(boxes, start=1):
            matched = result_by_key.get((image_name, idx))
            detection = {
                "object_id": idx,
                "box": box,
                "score": (prediction.get("scores", []) or [0.0] * len(boxes))[idx - 1],
                "label": (prediction.get("labels", []) or ["product"] * len(boxes))[idx - 1],
                "class_id": (prediction.get("class_ids", []) or [0] * len(boxes))[idx - 1],
                "mask": (prediction.get("masks", []) or [None] * len(boxes))[idx - 1],
            }
            if matched:
                detection.update(
                    {
                        "crop_path": matched.crop_path,
                        "sku_id": matched.sku_id,
                        "sku_name": matched.sku_name,
                        "sku_confidence": matched.sku_confidence,
                        "sku_status": matched.sku_status,
                        "sku_top_k": [candidate.__dict__ for candidate in matched.top_k],
                    }
                )
            detections.append(detection)

        enriched = dict(prediction)
        enriched["detections"] = detections
        enriched["identified_objects_count"] = sum(1 for item in detections if item.get("sku_status") == "matched")
        enriched["unknown_objects_count"] = sum(1 for item in detections if item.get("sku_status") == "unknown")
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
    lines = [
        "# ShelfVision: отчёт по SKU-идентификации",
        "",
        "## Сводка",
        "",
        f"- Всего объектов: {metrics.get('total_objects', 0)}",
        f"- Сопоставлено с SKU: {metrics.get('matched', 0)}",
        f"- Unknown: {metrics.get('unknown', 0)}",
        f"- Доля matched: {metrics.get('matched_rate', 0):.4f}",
        f"- Доля unknown: {metrics.get('unknown_rate', 0):.4f}",
        f"- Средняя similarity: {metrics.get('avg_similarity', 0):.4f}",
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
    lines.append("| image | object | status | sku | confidence | crop |")
    lines.append("|---|---:|---|---|---:|---|")
    for item in results[:30]:
        lines.append(
            f"| {item.image_name} | {item.object_id} | {item.sku_status} | {item.sku_name} | "
            f"{item.sku_confidence:.4f} | `{item.crop_path}` |"
        )

    report_path = out_dir / "identification_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
