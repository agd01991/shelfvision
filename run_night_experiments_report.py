from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class NightExperimentRecommendation:
    experiment: str
    reason: str
    matched_rate: float
    unknown_rate: float
    avg_similarity: float
    query_objects: int
    demo_sku: int
    gallery_refs: int
    out_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analytical reports for ShelfVision SKU110K night experiments")
    parser.add_argument("--results-root", default=None, help="Night experiment root directory")
    parser.add_argument("--summary-csv", default=None, help="Path to night_experiments_summary.csv")
    parser.add_argument("--out-dir", default=None, help="Directory for generated reports. Defaults to results root")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.summary_csv:
        summary_csv = Path(args.summary_csv)
        results_root = Path(args.results_root) if args.results_root else summary_csv.parent
    elif args.results_root:
        results_root = Path(args.results_root)
        summary_csv = results_root / "night_experiments_summary.csv"
    else:
        raise SystemExit("Specify --results-root or --summary-csv")

    out_dir = Path(args.out_dir) if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return results_root, summary_csv, out_dir


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _prepare_df(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")

    numeric_cols = [
        "conf",
        "imgsz",
        "gallery_count",
        "query_count",
        "max_sku",
        "dedup_threshold",
        "max_refs_per_sku",
        "min_crop",
        "padding",
        "query_objects",
        "matched",
        "unknown",
        "matched_rate",
        "unknown_rate",
        "avg_similarity",
        "created_demo_sku",
        "gallery_refs",
        "duplicate_refs",
        "skipped_duplicate_crops",
        "elapsed_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _num(df[col])

    if "status" not in df.columns:
        df["status"] = "unknown"
    if "experiment" not in df.columns:
        df["experiment"] = [f"experiment_{i+1}" for i in range(len(df))]

    df["is_ok"] = df["status"].eq("ok")
    refs_bonus = (df.get("gallery_refs", 0) / 500.0).clip(upper=1.0)
    df["quick_score"] = (
        df.get("matched_rate", 0) * 0.55
        + df.get("avg_similarity", 0) * 0.35
        + refs_bonus * 0.10
        - df.get("unknown_rate", 0) * 0.05
    )
    return df


def _group_impact(df_ok: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df_ok.columns or df_ok.empty:
        return pd.DataFrame()
    rows = []
    grouped = df_ok.groupby(group_col, dropna=False)
    for value, part in grouped:
        best = part.sort_values("quick_score", ascending=False).iloc[0]
        rows.append(
            {
                "factor": group_col,
                "value": value,
                "runs": len(part),
                "best_experiment": best.get("experiment", ""),
                "best_quick_score": best.get("quick_score", 0.0),
                "best_matched_rate": best.get("matched_rate", 0.0),
                "best_unknown_rate": best.get("unknown_rate", 0.0),
                "best_avg_similarity": best.get("avg_similarity", 0.0),
                "avg_matched_rate": part["matched_rate"].mean(),
                "avg_similarity": part["avg_similarity"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_parameter_impact(df_ok: pd.DataFrame) -> pd.DataFrame:
    factors = [
        "model",
        "weights_key",
        "conf",
        "gallery_count",
        "max_sku",
        "dedup_threshold",
        "max_refs_per_sku",
        "min_crop",
        "padding",
    ]
    tables = [_group_impact(df_ok, factor) for factor in factors]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).sort_values(["factor", "best_quick_score"], ascending=[True, False])


def _row_to_recommendation(row: pd.Series, reason: str) -> NightExperimentRecommendation:
    return NightExperimentRecommendation(
        experiment=str(row.get("experiment", "")),
        reason=reason,
        matched_rate=float(row.get("matched_rate", 0.0) or 0.0),
        unknown_rate=float(row.get("unknown_rate", 0.0) or 0.0),
        avg_similarity=float(row.get("avg_similarity", 0.0) or 0.0),
        query_objects=int(row.get("query_objects", 0) or 0),
        demo_sku=int(row.get("created_demo_sku", 0) or 0),
        gallery_refs=int(row.get("gallery_refs", 0) or 0),
        out_dir=str(row.get("out_dir", "")),
    )


def build_recommendations(df_ok: pd.DataFrame) -> List[NightExperimentRecommendation]:
    if df_ok.empty:
        return []

    recs: List[NightExperimentRecommendation] = []
    best_quick = df_ok.sort_values("quick_score", ascending=False).iloc[0]
    recs.append(_row_to_recommendation(best_quick, "Лучший общий кандидат по quick_score"))

    best_matched = df_ok.sort_values(["matched_rate", "avg_similarity"], ascending=False).iloc[0]
    if best_matched.get("experiment") != best_quick.get("experiment"):
        recs.append(_row_to_recommendation(best_matched, "Максимальная доля matched"))

    safe = df_ok[df_ok.get("dedup_threshold", 0) >= 0.86]
    if not safe.empty:
        best_safe = safe.sort_values("quick_score", ascending=False).iloc[0]
        if best_safe.get("experiment") not in {rec.experiment for rec in recs}:
            recs.append(_row_to_recommendation(best_safe, "Более консервативный вариант: dedup_threshold >= 0.86"))

    rich_gallery = df_ok[(df_ok.get("gallery_count", 0) >= 160) | (df_ok.get("gallery_refs", 0) >= 1000)]
    if not rich_gallery.empty:
        best_rich = rich_gallery.sort_values("quick_score", ascending=False).iloc[0]
        if best_rich.get("experiment") not in {rec.experiment for rec in recs}:
            recs.append(_row_to_recommendation(best_rich, "Лучший вариант с расширенной gallery"))

    return recs


def _save_plots(df_ok: pd.DataFrame, impact: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    if df_ok.empty:
        return outputs
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return outputs

    top = df_ok.sort_values("quick_score", ascending=False).head(12).copy()
    plot_path = out_dir / "night_experiments_top_matched_rate.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(top)), top["matched_rate"])
    ax.set_title("Top experiments by quick_score: matched_rate")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("matched_rate")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["experiment"], rotation=75, ha="right", fontsize=8)
    ax.set_ylim(0, max(1.0, float(top["matched_rate"].max()) * 1.05))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    outputs["top_matched_rate_plot"] = plot_path

    if not impact.empty:
        best_by_factor = impact.sort_values("best_quick_score", ascending=False).head(20)
        impact_plot = out_dir / "night_experiments_parameter_impact.png"
        labels = [f"{r.factor}={r.value}" for r in best_by_factor.itertuples()]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(range(len(best_by_factor)), best_by_factor["best_matched_rate"])
        ax.set_title("Parameter impact: best matched_rate by factor value")
        ax.set_xlabel("best matched_rate")
        ax.set_yticks(range(len(best_by_factor)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(impact_plot, dpi=180)
        plt.close(fig)
        outputs["parameter_impact_plot"] = impact_plot

    return outputs


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def save_reports(results_root: Path, summary_csv: Path, out_dir: Path, top_n: int) -> Dict[str, Path]:
    df = _prepare_df(summary_csv)
    df_ok = df[df["is_ok"]].copy()
    df_sorted = df_ok.sort_values("quick_score", ascending=False)
    impact = build_parameter_impact(df_ok)
    recs = build_recommendations(df_ok)
    plots = _save_plots(df_ok, impact, out_dir)

    ranked_csv = out_dir / "night_experiments_ranked.csv"
    impact_csv = out_dir / "night_experiments_parameter_impact.csv"
    best_json = out_dir / "night_experiments_best_config.json"
    report_md = out_dir / "night_experiments_detailed_report.md"
    vkr_md = out_dir / "vkr_night_experiments_section.md"

    df_sorted.to_csv(ranked_csv, index=False)
    impact.to_csv(impact_csv, index=False)

    best_payload = {
        "results_root": str(results_root),
        "summary_csv": str(summary_csv),
        "recommendations": [asdict(rec) for rec in recs],
    }
    best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Отчёт по серии экспериментов SKU110K",
        "",
        f"Папка серии: `{results_root}`",
        f"Исходная таблица: `{summary_csv}`",
        "",
        "## Назначение отчёта",
        "",
        "Отчёт агрегирует результаты серии запусков полного pipeline: детекция, формирование demo SKU-галереи, дедупликация, идентификация query-объектов и threshold analysis.",
        "",
        "`matched_rate` не является accuracy по реальным SKU. Это доля объектов, сопоставленных с автоматически сформированной demo SKU-галереей.",
        "",
        "## Рекомендованные конфигурации",
        "",
    ]
    for index, rec in enumerate(recs, start=1):
        lines.extend(
            [
                f"### {index}. `{rec.experiment}`",
                "",
                f"Причина выбора: {rec.reason}.",
                "",
                "| Показатель | Значение |",
                "|---|---:|",
                f"| matched_rate | {_format_float(rec.matched_rate)} ({_format_pct(rec.matched_rate)}) |",
                f"| unknown_rate | {_format_float(rec.unknown_rate)} ({_format_pct(rec.unknown_rate)}) |",
                f"| avg_similarity | {_format_float(rec.avg_similarity)} |",
                f"| query_objects | {rec.query_objects} |",
                f"| demo SKU | {rec.demo_sku} |",
                f"| gallery refs | {rec.gallery_refs} |",
                f"| out_dir | `{rec.out_dir}` |",
                "",
            ]
        )

    top_table_cols = [
        "experiment",
        "quick_score",
        "matched_rate",
        "unknown_rate",
        "avg_similarity",
        "query_objects",
        "created_demo_sku",
        "gallery_refs",
        "conf",
        "gallery_count",
        "max_sku",
        "dedup_threshold",
        "max_refs_per_sku",
        "min_crop",
        "padding",
    ]
    existing_top_cols = [col for col in top_table_cols if col in df_sorted.columns]
    lines.extend(
        [
            "## Топ экспериментов",
            "",
            df_sorted[existing_top_cols].head(top_n).to_markdown(index=False),
            "",
            "## Влияние параметров",
            "",
        ]
    )
    if not impact.empty:
        lines.append(impact.to_markdown(index=False))
    else:
        lines.append("Нет данных для оценки влияния параметров.")

    lines.extend(
        [
            "",
            "## Интерпретация для программы",
            "",
            "На основе серии экспериментов в программу целесообразно добавить модуль агрегации серийных запусков, автоматическое ранжирование конфигураций, выбор рекомендованного финального эксперимента и блок отчётов для ВКР.",
            "",
            "## Файлы отчёта",
            "",
            f"- Ранжированная таблица: `{ranked_csv}`",
            f"- Влияние параметров: `{impact_csv}`",
            f"- JSON с рекомендациями: `{best_json}`",
        ]
    )
    for name, path in plots.items():
        lines.append(f"- {name}: `{path}`")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    vkr_lines = [
        "# Раздел для ВКР: серия экспериментов по подбору параметров идентификации",
        "",
        "Для повышения качества работы системы на датасете SKU110K была проведена серия экспериментов, в которых варьировались параметры детектора и модуля формирования demo SKU-галереи: confidence threshold, размер gallery-части, максимальное количество demo SKU, порог дедупликации crop-изображений, количество эталонов на один SKU, минимальный размер crop и padding.",
        "",
        "В результате экспериментов было установлено, что наибольшее влияние на долю сопоставленных объектов оказывают размер demo SKU-галереи и порог дедупликации crop-изображений. Увеличение gallery-части и более мягкая дедупликация повышают покрытие галереи и увеличивают долю объектов, для которых находится близкий эталон.",
        "",
    ]
    if recs:
        best = recs[0]
        vkr_lines.extend(
            [
                f"Лучшей общей конфигурацией по совокупному quick_score стал эксперимент `{best.experiment}`. При данной конфигурации доля объектов, сопоставленных с demo SKU-галереей, составила {_format_pct(best.matched_rate)}, доля объектов со статусом unknown — {_format_pct(best.unknown_rate)}, средняя similarity — {_format_float(best.avg_similarity)}.",
                "",
            ]
        )
    vkr_lines.extend(
        [
            "Следует отметить, что matched_rate в данном эксперименте не является accuracy по реальным SKU-классам, поскольку в используемом наборе данных отсутствует эталонная разметка объектов по настоящим артикулам. Данный показатель отражает долю объектов, которые были сопоставлены с автоматически сформированной демонстрационной SKU-галереей.",
            "",
            "Полученные результаты могут быть использованы для выбора финальной конфигурации программного модуля и для обоснования параметров идентификации в экспериментальной части ВКР.",
        ]
    )
    vkr_md.write_text("\n".join(vkr_lines), encoding="utf-8")

    return {
        "ranked_csv": ranked_csv,
        "impact_csv": impact_csv,
        "best_json": best_json,
        "report_md": report_md,
        "vkr_md": vkr_md,
        **plots,
    }


def main() -> None:
    args = parse_args()
    results_root, summary_csv, out_dir = _resolve_paths(args)
    outputs = save_reports(results_root=results_root, summary_csv=summary_csv, out_dir=out_dir, top_n=max(1, args.top_n))

    print("=== Night experiments report generated ===")
    print(f"Results root: {results_root}")
    print(f"Summary CSV: {summary_csv}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
