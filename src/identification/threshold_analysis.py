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


def _save_threshold_plot(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Saves a VKR-ready plot for threshold analysis.

    The project uses Streamlit/CLI scripts, so plotting must work headlessly.
    If matplotlib is unavailable, the table files are still produced.
    """

    if df.empty:
        return {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    png_path = out_dir / "threshold_analysis_plot.png"
    svg_path = out_dir / "threshold_analysis_plot.svg"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["threshold"], df["matched_rate"], marker="o", label="matched_rate")
    ax.plot(df["threshold"], df["unknown_rate"], marker="o", label="unknown_rate")
    ax.set_title("Влияние порога similarity на идентификацию")
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Доля объектов")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"threshold_analysis_plot_png": png_path, "threshold_analysis_plot_svg": svg_path}


def _recommended_threshold_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "Недостаточно данных для выбора порога."

    # Prefer the lowest threshold that does not match absolutely everything,
    # because it demonstrates rejection of questionable objects while keeping
    # most useful matches. If there is no such threshold, fall back to 0.65 or
    # the nearest available value.
    candidates = df[(df["matched_rate"] >= 0.90) & (df["unknown"] > 0)]
    if candidates.empty:
        nearest = df.iloc[(df["threshold"] - 0.65).abs().argsort()[:1]]
        row = nearest.iloc[0]
    else:
        row = candidates.sort_values(["unknown", "threshold"]).iloc[0]

    return (
        "Рекомендуемый рабочий порог по текущей таблице: "
        f"`{row['threshold']:.2f}`. При нём matched = {int(row['matched'])} "
        f"из {int(row['total_objects'])}, unknown = {int(row['unknown'])}, "
        f"matched_rate = {row['matched_rate']:.4f}."
    )


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
    plot_outputs = _save_threshold_plot(df, out_dir)

    lines = [
        "# Анализ влияния порога similarity на идентификацию",
        "",
        "Порог `threshold` определяет, при каком значении похожести найденный объект считается сопоставленным с SKU-галереей.",
        "",
        "Чем ниже порог, тем больше объектов получают статус `matched`, но выше риск менее надёжных совпадений. Чем выше порог, тем больше объектов получают статус `unknown`, но совпадения становятся строже.",
        "",
        df.to_markdown(index=False),
        "",
        "## График",
        "",
    ]
    if "threshold_analysis_plot_png" in plot_outputs:
        lines.extend([
            "![График влияния threshold на matched_rate и unknown_rate](threshold_analysis_plot.png)",
            "",
        ])
    else:
        lines.extend([
            "График не был сформирован. Проверь, установлен ли `matplotlib`.",
            "",
        ])

    lines.extend(
        [
            "## Рекомендация по порогу",
            "",
            _recommended_threshold_text(df),
            "",
            "## Формулировка для ВКР",
            "",
            "Для оценки устойчивости модуля идентификации проведён анализ влияния порога similarity. Результаты показывают, как изменение порога сопоставления влияет на долю объектов, которым был присвоен идентификатор SKU, и на долю объектов, отнесённых к unknown.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"threshold_analysis_csv": csv_path, "threshold_analysis_md": md_path, **plot_outputs}
