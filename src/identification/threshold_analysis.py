from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .matcher import IdentificationResult


DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

THRESHOLD_ANALYSIS_COLUMNS_RU = {
    "threshold": "Порог идентификации",
    "total_objects": "Всего объектов",
    "matched": "Уверенные сопоставления",
    "matched_uncertain": "Спорные сопоставления",
    "unknown": "Неопределенные объекты",
    "assigned": "Объекты с назначенным кандидатом",
    "matched_rate": "Доля уверенных сопоставлений",
    "matched_uncertain_rate": "Доля спорных сопоставлений",
    "unknown_rate": "Доля неопределенных объектов",
    "assigned_rate": "Доля объектов с назначенным кандидатом",
    "avg_similarity_all": "Среднее сходство по всем объектам",
    "avg_similarity_matched": "Среднее сходство уверенных сопоставлений",
    "mean_margin_matched": "Средний margin уверенных сопоставлений",
}


def build_threshold_analysis(
    results: List[IdentificationResult],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    ambiguity_margin: float = 0.03,
    enable_uncertain_status: bool = True,
) -> pd.DataFrame:
    total = len(results)
    rows = []

    for threshold in thresholds:
        threshold = float(threshold)

        matched_items = []
        uncertain_items = []
        unknown_items = []

        for item in results:
            score = float(item.sku_confidence or 0.0)
            margin = item.distinct_margin

            if score < threshold:
                unknown_items.append(item)
            elif (
                enable_uncertain_status
                and margin is not None
                and float(margin) < ambiguity_margin
            ):
                uncertain_items.append(item)
            else:
                matched_items.append(item)

        matched = len(matched_items)
        matched_uncertain = len(uncertain_items)
        unknown = len(unknown_items)
        assigned = matched + matched_uncertain

        rows.append(
            {
                "threshold": threshold,
                "total_objects": total,
                "matched": matched,
                "matched_uncertain": matched_uncertain,
                "unknown": unknown,
                "assigned": assigned,
                "matched_rate": matched / total if total else 0.0,
                "matched_uncertain_rate": matched_uncertain / total if total else 0.0,
                "unknown_rate": unknown / total if total else 0.0,
                "assigned_rate": assigned / total if total else 0.0,
                "avg_similarity_all": sum(item.sku_confidence for item in results)
                / total
                if total
                else 0.0,
                "avg_similarity_matched": sum(
                    item.sku_confidence for item in matched_items
                )
                / matched
                if matched
                else 0.0,
                "mean_margin_matched": (
                    sum(
                        float(item.distinct_margin)
                        for item in matched_items
                        if item.distinct_margin is not None
                    )
                    / max(
                        1,
                        sum(
                            1
                            for item in matched_items
                            if item.distinct_margin is not None
                        ),
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def _display_threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.rename(columns=THRESHOLD_ANALYSIS_COLUMNS_RU)


def _save_threshold_plot(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Сохраняет график анализа порогов для отчёта ВКР.

    Проект запускается как из Streamlit, так и из CLI, поэтому график должен
    строиться без открытого окна. Если matplotlib недоступен, табличные отчёты
    всё равно будут сохранены.
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
    ax.plot(
        df["threshold"],
        df["matched_rate"],
        marker="o",
        label="доля уверенных совпадений",
    )
    ax.plot(
        df["threshold"],
        df["unknown_rate"],
        marker="o",
        label="доля неопределённых объектов",
    )
    ax.set_title("Влияние порога сходства на идентификацию")
    ax.set_xlabel("Порог визуального сходства")
    ax.set_ylabel("Доля объектов")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {
        "threshold_analysis_plot_png": png_path,
        "threshold_analysis_plot_svg": svg_path,
    }


def _recommended_threshold_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "Недостаточно данных для выбора порога."

    # Предпочитаем минимальный порог, который не сопоставляет абсолютно всё:
    # так в отчёте видно, что сомнительные объекты отбрасываются, но при этом
    # сохраняется большая часть полезных совпадений. Если такого порога нет,
    # используем 0.65 или ближайшее доступное значение.
    candidates = df[(df["matched_rate"] >= 0.90) & (df["unknown"] > 0)]
    if candidates.empty:
        nearest = df.iloc[(df["threshold"] - 0.65).abs().argsort()[:1]]
        row = nearest.iloc[0]
    else:
        row = candidates.sort_values(["unknown", "threshold"]).iloc[0]

    return (
        "Рекомендуемый рабочий порог по текущей таблице: "
        f"`{row['threshold']:.2f}`. При нём уверенно сопоставлено {int(row['matched'])} "
        f"из {int(row['total_objects'])} объектов, неопределённых объектов = {int(row['unknown'])}, "
        f"доля уверенных совпадений = {row['matched_rate']:.4f}."
    )


def save_threshold_analysis(
    results: List[IdentificationResult],
    out_dir: str | Path,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    ambiguity_margin: float = 0.03,
    enable_uncertain_status: bool = True,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_threshold_analysis(results, thresholds=thresholds)
    csv_path = out_dir / "threshold_analysis.csv"
    md_path = out_dir / "threshold_analysis.md"
    df.to_csv(csv_path, index=False)
    plot_outputs = _save_threshold_plot(df, out_dir)
    display_df = _display_threshold_analysis(df)

    lines = [
        "# Анализ влияния порога визуального сходства на идентификацию",
        "",
        "Порог идентификации определяет, при каком значении визуального сходства найденный объект считается сопоставленным с SKU-галереей.",
        "",
        "Чем ниже порог, тем больше объектов получают статус уверенного совпадения, но выше риск менее надёжных назначений. Чем выше порог, тем больше объектов остаются неопределёнными, но совпадения становятся строже.",
        "",
        display_df.to_markdown(index=False),
        "",
        "## График",
        "",
    ]
    if "threshold_analysis_plot_png" in plot_outputs:
        lines.extend(
            [
                "![График влияния порога на долю уверенных совпадений и неопределённых объектов](threshold_analysis_plot.png)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "График не был сформирован. Проверь, установлен ли `matplotlib`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Рекомендация по порогу",
            "",
            _recommended_threshold_text(df),
            "",
            "## Формулировка для ВКР",
            "",
            "Для оценки устойчивости модуля идентификации проведён анализ влияния порога визуального сходства. Результаты показывают, как изменение порога сопоставления влияет на долю объектов, которым был присвоен идентификатор SKU, и на долю объектов, оставшихся неопределёнными.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "threshold_analysis_csv": csv_path,
        "threshold_analysis_md": md_path,
        **plot_outputs,
    }
