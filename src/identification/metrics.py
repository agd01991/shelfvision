from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .matcher import IdentificationResult


def summarize_identification(results: List[IdentificationResult]) -> Dict[str, float | int]:
    total = len(results)
    matched = sum(1 for item in results if item.sku_status == "matched")
    matched_uncertain = sum(1 for item in results if item.sku_status == "matched_uncertain")
    unknown = sum(1 for item in results if item.sku_status == "unknown")
    assigned = matched + matched_uncertain
    avg_similarity = sum(item.sku_confidence for item in results) / total if total else 0.0
    margins = [float(item.distinct_margin) for item in results if item.distinct_margin is not None]
    mean_distinct_margin = sum(margins) / len(margins) if margins else 0.0

    return {
        "total_objects": total,
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "assigned": assigned,
        "matched_rate": matched / total if total else 0.0,
        "matched_uncertain_rate": matched_uncertain / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "assigned_rate": assigned / total if total else 0.0,
        "avg_similarity": avg_similarity,
        "mean_distinct_margin": mean_distinct_margin,
    }


def evaluate_with_ground_truth(
    results: List[IdentificationResult],
    gt_csv: str | Path | None = None,
) -> Dict[str, float | int]:
    summary = summarize_identification(results)
    if not gt_csv:
        return summary

    gt = pd.read_csv(gt_csv)
    required = {"image_name", "object_id", "true_sku_id"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"В ground truth CSV отсутствуют колонки: {sorted(missing)}")

    gt_map = {
        (str(row["image_name"]), int(row["object_id"])): str(row["true_sku_id"])
        for _, row in gt.iterrows()
    }

    evaluated = 0
    top1_correct = 0
    topk_correct = 0
    false_match = 0
    uncertain_correct = 0
    uncertain_total = 0

    for item in results:
        true_sku = gt_map.get((item.image_name, item.object_id))
        if true_sku is None:
            continue
        evaluated += 1
        if item.sku_id == true_sku:
            top1_correct += 1
        if any(candidate.sku_id == true_sku for candidate in item.top_k):
            topk_correct += 1
        if item.sku_status == "matched" and item.sku_id != true_sku:
            false_match += 1
        if item.sku_status == "matched_uncertain":
            uncertain_total += 1
            if item.sku_id == true_sku:
                uncertain_correct += 1

    summary.update(
        {
            "evaluated_objects": evaluated,
            "top1_accuracy": top1_correct / evaluated if evaluated else 0.0,
            "topk_accuracy": topk_correct / evaluated if evaluated else 0.0,
            "false_match_rate": false_match / evaluated if evaluated else 0.0,
            "uncertain_total": uncertain_total,
            "uncertain_correct": uncertain_correct,
            "uncertain_accuracy": uncertain_correct / uncertain_total if uncertain_total else 0.0,
        }
    )
    return summary


def save_identification_metrics(
    metrics: Dict[str, float | int],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_dir / "identification_metrics.csv", index=False)
