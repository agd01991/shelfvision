from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .matcher import IdentificationResult, results_to_dataframe


def build_sku_assignment_summary(results: List[IdentificationResult]) -> pd.DataFrame:
    df = results_to_dataframe(results)

    if df.empty or "sku_id" not in df.columns:
        return pd.DataFrame()

    df["sku_id_group"] = df["sku_id"].fillna("NA").astype(str)
    df["sku_confidence"] = pd.to_numeric(df.get("sku_confidence"), errors="coerce")
    df["distinct_margin"] = pd.to_numeric(df.get("distinct_margin"), errors="coerce")

    grouped = (
        df.groupby("sku_id_group")
        .agg(
            assigned_objects=("sku_id_group", "size"),
            mean_confidence=("sku_confidence", "mean"),
            median_confidence=("sku_confidence", "median"),
            min_confidence=("sku_confidence", "min"),
            max_confidence=("sku_confidence", "max"),
            mean_margin=("distinct_margin", "mean"),
            median_margin=("distinct_margin", "median"),
            matched_count=("sku_status", lambda s: int((s == "matched").sum())),
            uncertain_count=("sku_status", lambda s: int((s == "matched_uncertain").sum())),
            unknown_count=("sku_status", lambda s: int((s == "unknown").sum())),
        )
        .reset_index()
        .rename(columns={"sku_id_group": "assigned_sku_id"})
    )

    grouped["uncertain_rate"] = grouped["uncertain_count"] / grouped["assigned_objects"].clip(lower=1)

    return grouped.sort_values(
        ["assigned_objects", "uncertain_rate"],
        ascending=[False, False],
    )


def save_assignment_audit_outputs(
    results: List[IdentificationResult],
    out_dir: str | Path,
    threshold: float,
    ambiguity_margin: float,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)

    audit_csv = out_dir / "query_assignment_audit.csv"
    df.to_csv(audit_csv, index=False)

    uncertain_df = df[df["sku_status"].astype(str).eq("matched_uncertain")].copy()
    uncertain_csv = out_dir / "matched_uncertain_candidates.csv"
    uncertain_df.to_csv(uncertain_csv, index=False)

    sku_summary = build_sku_assignment_summary(results)
    sku_summary_csv = out_dir / "query_assignment_sku_summary.csv"
    sku_summary.to_csv(sku_summary_csv, index=False)

    total = len(results)
    matched = sum(1 for item in results if item.sku_status == "matched")
    matched_uncertain = sum(1 for item in results if item.sku_status == "matched_uncertain")
    unknown = sum(1 for item in results if item.sku_status == "unknown")

    margins = [
        float(item.distinct_margin)
        for item in results
        if item.distinct_margin is not None
    ]
    mean_distinct_margin = sum(margins) / len(margins) if margins else 0.0

    summary = {
        "threshold": float(threshold),
        "ambiguity_margin": float(ambiguity_margin),
        "objects": total,
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "matched_rate": matched / total if total else 0.0,
        "matched_uncertain_rate": matched_uncertain / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "assigned_rate": (matched + matched_uncertain) / total if total else 0.0,
        "mean_sku_confidence": sum(item.sku_confidence for item in results) / total if total else 0.0,
        "mean_distinct_margin": mean_distinct_margin,
    }

    summary_json = out_dir / "assignment_uncertainty_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    suspicious = sku_summary[
        (sku_summary["assigned_objects"] >= 50)
        & (
            (sku_summary["uncertain_rate"] >= 0.30)
            | (sku_summary["mean_confidence"] < 0.70)
        )
    ].copy() if not sku_summary.empty else pd.DataFrame()
    suspicious_csv = out_dir / "suspicious_absorber_sku.csv"
    suspicious.to_csv(suspicious_csv, index=False)

    report_md = out_dir / "assignment_uncertainty_report.md"
    lines = [
        "# ShelfVision: assignment uncertainty audit",
        "",
        "## Parameters",
        "",
        f"- threshold: `{threshold}`",
        f"- ambiguity_margin: `{ambiguity_margin}`",
        "",
        "## Summary",
        "",
        f"- objects: `{summary['objects']}`",
        f"- matched: `{summary['matched']}`",
        f"- matched_uncertain: `{summary['matched_uncertain']}`",
        f"- unknown: `{summary['unknown']}`",
        f"- matched_rate: `{summary['matched_rate']:.4f}`",
        f"- matched_uncertain_rate: `{summary['matched_uncertain_rate']:.4f}`",
        f"- unknown_rate: `{summary['unknown_rate']:.4f}`",
        f"- assigned_rate: `{summary['assigned_rate']:.4f}`",
        f"- mean_sku_confidence: `{summary['mean_sku_confidence']:.4f}`",
        f"- mean_distinct_margin: `{summary['mean_distinct_margin']:.4f}`",
        "",
        "## Files",
        "",
        f"- query assignment audit: `{audit_csv}`",
        f"- SKU summary: `{sku_summary_csv}`",
        f"- matched uncertain candidates: `{uncertain_csv}`",
        f"- suspicious absorber SKU: `{suspicious_csv}`",
        f"- summary JSON: `{summary_json}`",
        "",
        "## Interpretation",
        "",
        "`matched` означает уверенное назначение SKU. "
        "`matched_uncertain` означает, что top-1 и top-2 разные SKU слишком близки по similarity, "
        "поэтому для безопасного результата `safe_sku_id` не заполняется. "
        "`unknown` означает, что score ниже threshold или совпадение не найдено.",
        "",
        "## Формулировка для ВКР",
        "",
        "Для снижения риска ошибочного присвоения разных товаров одному SKU в модуль идентификации добавлен анализ неоднозначности. "
        "После расчёта top-k кандидатов определяется лучший и второй лучший различные SKU, затем вычисляется margin между ними. "
        "Если margin ниже заданного порога, объект получает статус `matched_uncertain`, что позволяет сохранить диагностическую информацию, "
        "но не использовать такое назначение как надёжное.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "query_assignment_audit_csv": audit_csv,
        "matched_uncertain_candidates_csv": uncertain_csv,
        "query_assignment_sku_summary_csv": sku_summary_csv,
        "suspicious_absorber_sku_csv": suspicious_csv,
        "assignment_uncertainty_summary_json": summary_json,
        "assignment_uncertainty_report_md": report_md,
    }
