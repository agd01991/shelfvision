from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.identification.assignment_audit import save_assignment_audit_outputs
from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.threshold_analysis import save_threshold_analysis
from src.identification.visualization import visualize_identification_results
from src.reporting.segmentation_identification_report import generate_segmentation_identification_report


@dataclass
class ExistingIdentificationSummary:
    out_dir: str
    query_predictions_json: str
    identification_dir: str
    gallery_dir: str
    gallery_csv: str
    threshold: float
    thresholds: str
    top_k: int
    enable_uncertain_status: bool
    ambiguity_margin: float
    visualized_dir: str
    cache_dir: str
    total_objects: int
    matched: int
    matched_uncertain: int
    unknown: int
    assigned: int
    matched_rate: float
    matched_uncertain_rate: float
    unknown_rate: float
    assigned_rate: float
    avg_similarity: float
    mean_distinct_margin: float
    elapsed_seconds: float
    threshold_analysis_csv: str
    threshold_analysis_md: str
    threshold_analysis_plot_png: str
    assignment_audit_csv: str
    matched_uncertain_candidates_csv: str
    query_assignment_sku_summary_csv: str
    suspicious_absorber_sku_csv: str
    assignment_uncertainty_summary_json: str
    assignment_uncertainty_report_md: str
    segmentation_identification_summary_json: str
    segmentation_identification_report_md: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun ShelfVision photo identification from existing query predictions and SKU gallery")
    parser.add_argument("--out-dir", required=True, help="Full experiment output directory")
    parser.add_argument("--query-predictions-json", default=None, help="Existing query predictions.json. Defaults to <out-dir>/03_query_inference/predictions.json")
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument("--gallery-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--enable-uncertain-status",
        action="store_true",
        help="Enable matched_uncertain when top-1/top-2 distinct SKU margin is below --ambiguity-margin.",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.03,
        help="If best distinct SKU score minus second distinct SKU score is below this margin, mark as matched_uncertain.",
    )
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--gt-csv", default=None)
    parser.add_argument("--visualize-limit", type=int, default=100)
    parser.add_argument("--bbox-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cache-dir", default=None, help="Feature cache directory. Defaults to <out-dir>/04_identification/feature_cache")
    return parser.parse_args()


def _parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    return values or [0.65]


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _save_summary(
    args: argparse.Namespace,
    query_predictions_json: Path,
    identification_dir: Path,
    reports_dir: Path,
    cache_dir: Path,
    metrics: Dict[str, object],
    threshold_outputs: Dict[str, Path],
    assignment_outputs: Dict[str, Path],
    segmentation_identification_outputs: Dict[str, Path],
    elapsed_seconds: float,
) -> Dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = ExistingIdentificationSummary(
        out_dir=str(args.out_dir),
        query_predictions_json=str(query_predictions_json),
        identification_dir=str(identification_dir),
        gallery_dir=str(args.gallery_dir),
        gallery_csv=str(args.gallery_csv),
        threshold=float(args.threshold),
        thresholds=str(args.thresholds),
        top_k=int(args.top_k),
        enable_uncertain_status=bool(args.enable_uncertain_status),
        ambiguity_margin=float(args.ambiguity_margin),
        visualized_dir=str(identification_dir / "visualized"),
        cache_dir=str(cache_dir),
        total_objects=int(metrics.get("total_objects", 0) or 0),
        matched=int(metrics.get("matched", 0) or 0),
        matched_uncertain=int(metrics.get("matched_uncertain", 0) or 0),
        unknown=int(metrics.get("unknown", 0) or 0),
        assigned=int(metrics.get("assigned", 0) or 0),
        matched_rate=float(metrics.get("matched_rate", 0.0) or 0.0),
        matched_uncertain_rate=float(metrics.get("matched_uncertain_rate", 0.0) or 0.0),
        unknown_rate=float(metrics.get("unknown_rate", 0.0) or 0.0),
        assigned_rate=float(metrics.get("assigned_rate", 0.0) or 0.0),
        avg_similarity=float(metrics.get("avg_similarity", 0.0) or 0.0),
        mean_distinct_margin=float(metrics.get("mean_distinct_margin", 0.0) or 0.0),
        elapsed_seconds=elapsed_seconds,
        threshold_analysis_csv=str(threshold_outputs.get("threshold_analysis_csv", "")),
        threshold_analysis_md=str(threshold_outputs.get("threshold_analysis_md", "")),
        threshold_analysis_plot_png=str(threshold_outputs.get("threshold_analysis_plot_png", "")),
        assignment_audit_csv=str(assignment_outputs.get("query_assignment_audit_csv", "")),
        matched_uncertain_candidates_csv=str(assignment_outputs.get("matched_uncertain_candidates_csv", "")),
        query_assignment_sku_summary_csv=str(assignment_outputs.get("query_assignment_sku_summary_csv", "")),
        suspicious_absorber_sku_csv=str(assignment_outputs.get("suspicious_absorber_sku_csv", "")),
        assignment_uncertainty_summary_json=str(assignment_outputs.get("assignment_uncertainty_summary_json", "")),
        assignment_uncertainty_report_md=str(assignment_outputs.get("assignment_uncertainty_report_md", "")),
        segmentation_identification_summary_json=str(segmentation_identification_outputs.get("segmentation_identification_summary_json", "")),
        segmentation_identification_report_md=str(segmentation_identification_outputs.get("segmentation_identification_report_md", "")),
    )
    summary_json = reports_dir / "existing_identification_summary.json"
    summary_md = reports_dir / "existing_identification_summary.md"
    raw = asdict(summary)
    summary_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Быстрый пересчёт идентификации по существующим predictions",
        "",
        "## Сводка",
        "",
        f"- Query objects: {summary.total_objects}",
        f"- Matched: {summary.matched}",
        f"- Matched uncertain: {summary.matched_uncertain}",
        f"- Unknown: {summary.unknown}",
        f"- Assigned total: {summary.assigned}",
        f"- Matched rate: {summary.matched_rate:.4f}",
        f"- Matched uncertain rate: {summary.matched_uncertain_rate:.4f}",
        f"- Unknown rate: {summary.unknown_rate:.4f}",
        f"- Assigned rate: {summary.assigned_rate:.4f}",
        f"- Avg similarity: {summary.avg_similarity:.4f}",
        f"- Mean distinct margin: {summary.mean_distinct_margin:.4f}",
        f"- Threshold: {summary.threshold:.2f}",
        f"- Ambiguity margin: {summary.ambiguity_margin:.4f}",
        f"- Enable uncertain status: {summary.enable_uncertain_status}",
        f"- Время пересчёта: {_format_eta(summary.elapsed_seconds)}",
        "",
        "## Файлы",
        "",
        f"- Query predictions: `{summary.query_predictions_json}`",
        f"- Identification dir: `{summary.identification_dir}`",
        f"- Feature cache: `{summary.cache_dir}`",
        f"- Threshold CSV: `{summary.threshold_analysis_csv}`",
        f"- Threshold plot: `{summary.threshold_analysis_plot_png}`",
        f"- Assignment audit CSV: `{summary.assignment_audit_csv}`",
        f"- Matched uncertain candidates: `{summary.matched_uncertain_candidates_csv}`",
        f"- SKU assignment summary: `{summary.query_assignment_sku_summary_csv}`",
        f"- Suspicious absorber SKU: `{summary.suspicious_absorber_sku_csv}`",
        f"- Assignment uncertainty report: `{summary.assignment_uncertainty_report_md}`",
        f"- Segmentation + identification report: `{summary.segmentation_identification_report_md}`",
        "",
        "## Примечание для ВКР",
        "",
        "Этот режим не выполняет повторный inference модели детекции. Он переиспользует уже рассчитанный `query predictions.json`, текущую demo SKU-галерею и feature cache. Если включён `matched_uncertain`, система дополнительно проверяет margin между лучшим и вторым различным SKU и помечает неоднозначные совпадения как диагностические, не считая их надёжным safe SKU.",
        "",
        "Дополнительно формируется отчёт по связке сегментации/локализации и идентификации, который явно показывает соответствие практической реализации теме ВКР.",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {"existing_identification_summary_json": summary_json, "existing_identification_summary_md": summary_md}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    query_predictions_json = Path(args.query_predictions_json) if args.query_predictions_json else out_dir / "03_query_inference" / "predictions.json"
    identification_dir = out_dir / "04_identification"
    reports_dir = out_dir / "05_reports"
    cache_dir = Path(args.cache_dir) if args.cache_dir else identification_dir / "feature_cache"

    if not query_predictions_json.exists():
        raise FileNotFoundError(f"Query predictions not found: {query_predictions_json}")
    if not Path(args.gallery_csv).exists():
        raise FileNotFoundError(f"Gallery CSV not found: {args.gallery_csv}")

    print("=== ShelfVision existing photo identification rerun ===", flush=True)
    print(f"Query predictions: {query_predictions_json}", flush=True)
    print(f"Gallery CSV: {args.gallery_csv}", flush=True)
    print(f"Feature cache: {cache_dir}", flush=True)
    print(f"Uncertain status: {bool(args.enable_uncertain_status)} margin={float(args.ambiguity_margin):.4f}", flush=True)

    results = run_sku_matching(
        predictions_json=query_predictions_json,
        images_dir=None,
        out_dir=identification_dir,
        gallery_csv=args.gallery_csv,
        gallery_dir=args.gallery_dir,
        use_masks=not args.bbox_only,
        threshold=args.threshold,
        top_k=args.top_k,
        padding_ratio=args.padding,
        progress_every=max(1, args.progress_every),
        cache_dir=cache_dir,
        enable_uncertain_status=bool(args.enable_uncertain_status),
        ambiguity_margin=float(args.ambiguity_margin),
    )
    metrics = evaluate_with_ground_truth(results, gt_csv=args.gt_csv)
    save_identification_metrics(metrics, out_dir=identification_dir)
    save_identification_outputs(
        predictions_json=query_predictions_json,
        results=results,
        metrics=metrics,
        out_dir=identification_dir,
    )
    visualize_identification_results(
        results=results,
        images_dir=None,
        out_dir=identification_dir,
        limit=max(0, args.visualize_limit),
    )
    threshold_outputs = save_threshold_analysis(results, out_dir=reports_dir, thresholds=_parse_thresholds(args.thresholds))
    assignment_outputs = save_assignment_audit_outputs(
        results=results,
        out_dir=identification_dir,
        threshold=float(args.threshold),
        ambiguity_margin=float(args.ambiguity_margin),
    )
    segmentation_identification_outputs = generate_segmentation_identification_report(
        out_dir=out_dir,
        query_predictions_json=query_predictions_json,
        identification_dir=identification_dir,
        reports_dir=reports_dir,
    )
    summary_outputs = _save_summary(
        args=args,
        query_predictions_json=query_predictions_json,
        identification_dir=identification_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        metrics=metrics,
        threshold_outputs=threshold_outputs,
        assignment_outputs=assignment_outputs,
        segmentation_identification_outputs=segmentation_identification_outputs,
        elapsed_seconds=time.perf_counter() - started,
    )

    print("=== Done ===", flush=True)
    for name, path in {**threshold_outputs, **assignment_outputs, **segmentation_identification_outputs, **summary_outputs}.items():
        print(f"Report {name}: {path}", flush=True)
    print(f"Objects: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Matched: {metrics.get('matched', 0)}", flush=True)
    print(f"Matched uncertain: {metrics.get('matched_uncertain', 0)}", flush=True)
    print(f"Unknown: {metrics.get('unknown', 0)}", flush=True)
    print(f"Segmentation-identification report: {segmentation_identification_outputs.get('segmentation_identification_report_md')}", flush=True)


if __name__ == "__main__":
    main()
