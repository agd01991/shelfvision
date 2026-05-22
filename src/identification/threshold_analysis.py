from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .matcher import IdentificationResult


DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def build_threshold_analysis(
    results: List[IdentificationResult],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    total = len(results)
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        matched_items = [item for item in results if item.sku_confidence >= threshold]
        unknown_items = [item for item in results if item.sku_confidence < threshold]
        matched = len(matched_items)
        unknown = len(unknown_items)
        rows.append(
            {
                "threshold": threshold,
                "total_objects": total,
                "matched": matched,
                "unknown": unknown,
                "matched_rate": matched / total if total else 0.0,
                "unknown_rate": unknown / total if total else 0.0,
                "avg_similarity_all": sum(item.sku_confidence for item in results) / total if total else 0.0,
                "avg_similarity_matched": sum(item.sku_confidence for item in matched_items) / matched if matched else 0.0,
                "min_similarity_matched": min((item.sku_confidence for item in matched_items), default=0.0),
                "max_similarity_unknown": max((item.sku_confidence for item in unknown_items), default=0.0),
            }
        )
    return pd.DataFrame(rows)


def save_threshold_analysis(
    results: List[IdentificationResult],
    out_dir: str | Path,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_threshold_analysis(results, thresholds=thresholds)
    csv_path = out_dir / "threshold_analysis.csv"
    md_path = out_dir / "threshold_analysis.md"
    df.to_csv(csv_path, index=False)

    lines = [
        "# Анализ влияния порога similarity на идентификацию",
        "",
        "Порог `threshold` определяет, при каком значении похожести найденный объект считается сопоставленным с SKU-галереей.",
        "",
        "Чем ниже порог, тем больше объектов получают статус `matched`, но выше риск менее надёжных совпадений. Чем выше порог, тем больше объектов получают статус `unknown`, но совпадения становятся строже.",
        "",
        df.to_markdown(index=False),
        "",
        "## Формулировка для ВКР",
        "",
        "Для оценки устойчивости модуля идентификации проведён анализ влияния порога similarity. Результаты показывают, как изменение порога сопоставления влияет на долю объектов, которым был присвоен идентификатор SKU, и на долю объектов, отнесённых к unknown.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"threshold_analysis_csv": csv_path, "threshold_analysis_md": md_path}
