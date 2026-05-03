from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .recommend_model import RecommendationWeights, load_metrics_tables, recommend_best_model


METRIC_COLUMNS = ["AP50-95", "AP50", "precision", "recall", "f1"]


@dataclass
class ComparisonResult:
    best_model: str
    best_score: float
    ranking: List[Dict[str, Any]]
    highlights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _best_by_metric(df: pd.DataFrame, metric: str) -> Optional[Dict[str, Any]]:
    if metric not in df.columns or df.empty:
        return None
    ordered = df.sort_values(metric, ascending=False)
    row = ordered.iloc[0].to_dict()
    return {"model_name": row.get("model_name", "-"), "metric": metric, "value": _safe_float(row.get(metric))}


def build_highlights(ranking: pd.DataFrame) -> List[str]:
    highlights: List[str] = []
    for metric in METRIC_COLUMNS:
        best = _best_by_metric(ranking, metric)
        if best:
            highlights.append(
                f"Лучшее значение {metric}: {best['model_name']} ({best['value']:.4f})."
            )

    if "time" in ranking.columns and (ranking["time"] > 0).any():
        fastest = ranking[ranking["time"] > 0].sort_values("time", ascending=True).iloc[0]
        highlights.append(
            f"Самая быстрая модель: {fastest['model_name']} ({float(fastest['time']):.4f})."
        )

    if not highlights:
        highlights.append("Недостаточно данных для автоматического выделения сильных сторон моделей.")
    return highlights


def compare_models(
    metrics_paths: Sequence[str | Path],
    labels: Optional[Sequence[str]] = None,
    weights: RecommendationWeights | None = None,
) -> tuple[ComparisonResult, pd.DataFrame]:
    """Собирает метрики нескольких моделей в единый рейтинг."""

    weights = weights or RecommendationWeights()
    metrics_df = load_metrics_tables(metrics_paths, labels=labels)
    recommendation, ranking = recommend_best_model(metrics_df, weights=weights)
    highlights = build_highlights(ranking)

    comparison = ComparisonResult(
        best_model=recommendation.model_name,
        best_score=recommendation.score,
        ranking=ranking.to_dict(orient="records"),
        highlights=highlights,
    )
    return comparison, ranking


def _plot_bar(ranking: pd.DataFrame, metric: str, out_path: Path) -> None:
    if metric not in ranking.columns or ranking.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(ranking["model_name"], ranking[metric])
    plt.title(f"Сравнение моделей по {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_grouped_quality(ranking: pd.DataFrame, out_path: Path) -> None:
    available = [metric for metric in METRIC_COLUMNS if metric in ranking.columns]
    if not available or ranking.empty:
        return

    x = list(range(len(ranking)))
    width = 0.8 / len(available)

    plt.figure(figsize=(11, 6))
    for idx, metric in enumerate(available):
        positions = [item - 0.4 + width / 2 + idx * width for item in x]
        plt.bar(positions, ranking[metric], width=width, label=metric)

    plt.xticks(x, ranking["model_name"], rotation=20, ha="right")
    plt.ylabel("Значение метрики")
    plt.title("Сравнение качества моделей")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_quality_vs_speed(ranking: pd.DataFrame, out_path: Path) -> None:
    if ranking.empty or "time" not in ranking.columns or "AP50-95" not in ranking.columns:
        return
    df = ranking[ranking["time"] > 0].copy()
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(df["time"], df["AP50-95"])
    for _, row in df.iterrows():
        plt.annotate(str(row["model_name"]), (row["time"], row["AP50-95"]))
    plt.xlabel("Время обработки")
    plt.ylabel("AP50-95")
    plt.title("Баланс качества и скорости")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_comparison_report(
    comparison: ComparisonResult,
    ranking: pd.DataFrame,
    out_dir: str | Path,
    weights: RecommendationWeights | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    weights = weights or RecommendationWeights()

    json_path = out_dir / "model_comparison.json"
    csv_path = out_dir / "model_comparison.csv"
    md_path = out_dir / "model_comparison.md"

    json_path.write_text(
        json.dumps(
            {
                "comparison": comparison.to_dict(),
                "weights": asdict(weights),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ranking.to_csv(csv_path, index=False)

    for metric in METRIC_COLUMNS + ["recommendation_score"]:
        if metric in ranking.columns:
            _plot_bar(ranking, metric, plots_dir / f"{metric.replace('-', '_')}_bar.png")
    _plot_grouped_quality(ranking, plots_dir / "quality_grouped.png")
    _plot_quality_vs_speed(ranking, plots_dir / "quality_vs_speed.png")

    lines = [
        "# Сравнение моделей ShelfVision",
        "",
        f"**Рекомендуемый pipeline:** {comparison.best_model}",
        f"**Интегральный score:** {comparison.best_score:.4f}",
        "",
        "## Ключевые выводы",
        "",
    ]
    lines.extend([f"- {item}" for item in comparison.highlights])
    lines.extend(
        [
            "",
            "## Рейтинг моделей",
            "",
            ranking.to_markdown(index=False),
            "",
            "## Использованные веса рекомендации",
            "",
            pd.DataFrame([asdict(weights)]).to_markdown(index=False),
            "",
            "## Графики",
            "",
            "Графики сохранены в папке `plots/`.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path, "plots_dir": plots_dir}
