from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.threshold_analysis import save_threshold_analysis
from src.identification.visualization import visualize_identification_results


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
    visualized_dir: str
    cache_dir: str
    total_objects: int
    matched: int
    unknown: int
    matched_rate: float
    unknown_rate: float
    avg_similarity: float
    elapsed_seconds: float
    threshold_analysis_csv: str
    threshold_analysis_md: str
    threshold_analysis_plot_png: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun ShelfVision photo identification from existing query predictions and SKU gallery")
    parser.add_argument("--out-dir", required=True, help="Full experiment output directory")
    parser.add_argument("--query-predictions-json", default=None, help="Existing query predictions.json. Defaults to <out-dir>/03_query_inference/predictions.json")
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument("--gallery-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
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
        visualized_dir=str(identification_dir / "visualized"),
        cache_dir=str(cache_dir),
        total_objects=int(metrics.get("total_objects", 0) or 0),
        matched=int(metrics.get("matched", 0) or 0),
        unknown=int(metrics.get("unknown", 0) or 0),
        matched_rate=float(metrics.get("matched_rate", 0.0) or 0.0),
        unknown_rate=float(metrics.get("unknown_rate", 0.0) or 0.0),
        avg_similarity=float(metrics.get("avg_similarity", 0.0) or 0.0),
        elapsed_seconds=elapsed_seconds,
        threshold_analysis_csv=str(threshold_outputs.get("threshold_analysis_csv", "")),
        threshold_analysis_md=str(threshold_outputs.get("threshold_analysis_md", "")),
        threshold_analysis_plot_png=str(threshold_outputs.get("threshold_analysis_plot_png", "")),
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
        f"- Unknown: {summary.unknown}",
        f"- Matched rate: {summary.matched_rate:.4f}",
        f"- Unknown rate: {summary.unknown_rate:.4f}",
        f"- Avg similarity: {summary.avg_similarity:.4f}",
        f"- Threshold: {summary.threshold:.2f}",
        f"- Время пересчёта: {_format_eta(summary.elapsed_seconds)}",
        "",
        "## Файлы",
        "",
        f"- Query predictions: `{summary.query_predictions_json}`",
        f"- Identification dir: `{summary.identification_dir}`",
        f"- Feature cache: `{summary.cache_dir}`",
        f"- Threshold CSV: `{summary.threshold_analysis_csv}`",
        f"- Threshold plot: `{summary.threshold_analysis_plot_png}`",
        "",
        "## Примечание для ВКР",
        "",
        "Этот режим не выполняет повторный inference модели детекции. Он переиспользует уже рассчитанный `query predictions.json`, текущую demo SKU-галерею и feature cache, поэтому подходит для быстрого подбора порога идентификации и повторного формирования отчётов.",
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
    summary_outputs = _save_summary(
        args=args,
        query_predictions_json=query_predictions_json,
        identification_dir=identification_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        metrics=metrics,
        threshold_outputs=threshold_outputs,
        elapsed_seconds=time.perf_counter() - started,
    )

    print("=== Done ===", flush=True)
    for name, path in {**threshold_outputs, **summary_outputs}.items():
        print(f"Report {name}: {path}", flush=True)
    print(f"Objects: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Matched: {metrics.get('matched', 0)}", flush=True)
    print(f"Unknown: {metrics.get('unknown', 0)}", flush=True)


if __name__ == "__main__":
    main()
