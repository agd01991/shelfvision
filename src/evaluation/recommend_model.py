from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


@dataclass
class RecommendationWeights:
    """Веса критериев для выбора лучшего pipeline.

    Качество важнее скорости, потому что в ВКР основной акцент сделан на
    экспериментальном сравнении точности обнаружения и сегментации товаров.
    """

    ap50_95: float = 0.40
    ap50: float = 0.20
    recall: float = 0.15
    precision: float = 0.15
    f1: float = 0.05
    speed: float = 0.05


@dataclass
class ModelRecommendation:
    model_name: str
    score: float
    reason: str
    row: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


COLUMN_ALIASES = {
    "model": ["model", "model_name", "system", "name", "pipeline", "Модель"],
    "ap50_95": ["AP50-95", "AP50_95", "mAP50-95", "mAP50_95", "mAP5095", "mAP5095_box", "AP"],
    "ap50": ["AP50", "mAP50", "mAP50_box"],
    "precision": ["precision", "Precision", "P", "P_box"],
    "recall": ["recall", "Recall", "R", "R_box"],
    "f1": ["f1", "F1", "F1-score", "f1_score"],
    "time": ["inference_time", "ms_per_image", "time_ms", "seconds_per_image", "Время"],
}


def _find_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalize_metric(value: float) -> float:
    """Приводит метрику к диапазону 0..1.

    Некоторые внешние отчёты могут хранить AP в процентах 0..100, а наши новые
    метрики — в долях 0..1. Здесь это выравнивается.
    """

    if value > 1.0:
        return value / 100.0
    return max(0.0, min(1.0, value))


def _speed_score(value: float, min_time: float, max_time: float) -> float:
    """Чем меньше время, тем выше score."""

    if value <= 0:
        return 0.0
    if max_time <= min_time:
        return 1.0
    return 1.0 - ((value - min_time) / (max_time - min_time))


def _load_metrics_file(path: str | Path, fallback_model_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл метрик не найден: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "summary" in data:
            rows = [data["summary"]]
        elif isinstance(data, dict):
            rows = [data]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Неподдерживаемый JSON формат: {path}")
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    model_col = _find_column(df, "model")
    if model_col is None:
        df["model"] = fallback_model_name or path.parent.name or path.stem
    return df


def load_metrics_tables(paths: Sequence[str | Path], labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for idx, path in enumerate(paths):
        label = labels[idx] if labels and idx < len(labels) else None
        frames.append(_load_metrics_file(path, fallback_model_name=label))
    if not frames:
        raise ValueError("Передайте хотя бы один файл метрик")
    return pd.concat(frames, ignore_index=True)


def prepare_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    model_col = _find_column(df, "model")
    ap_col = _find_column(df, "ap50_95")
    ap50_col = _find_column(df, "ap50")
    precision_col = _find_column(df, "precision")
    recall_col = _find_column(df, "recall")
    f1_col = _find_column(df, "f1")
    time_col = _find_column(df, "time")

    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        model_name = str(row.get(model_col, f"model_{idx + 1}")) if model_col else f"model_{idx + 1}"
        rows.append(
            {
                "model_name": model_name,
                "AP50-95": _normalize_metric(_as_float(row.get(ap_col))) if ap_col else 0.0,
                "AP50": _normalize_metric(_as_float(row.get(ap50_col))) if ap50_col else 0.0,
                "precision": _normalize_metric(_as_float(row.get(precision_col))) if precision_col else 0.0,
                "recall": _normalize_metric(_as_float(row.get(recall_col))) if recall_col else 0.0,
                "f1": _normalize_metric(_as_float(row.get(f1_col))) if f1_col else 0.0,
                "time": _as_float(row.get(time_col), default=0.0) if time_col else 0.0,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    positive_times = table.loc[table["time"] > 0, "time"]
    if positive_times.empty:
        table["speed_score"] = 0.0
    else:
        min_time = float(positive_times.min())
        max_time = float(positive_times.max())
        table["speed_score"] = table["time"].apply(lambda value: _speed_score(float(value), min_time, max_time))
    return table


def recommend_best_model(
    metrics_df: pd.DataFrame,
    weights: RecommendationWeights | None = None,
) -> tuple[ModelRecommendation, pd.DataFrame]:
    weights = weights or RecommendationWeights()
    table = prepare_recommendation_table(metrics_df)
    if table.empty:
        raise ValueError("Таблица метрик пустая")

    table["recommendation_score"] = (
        table["AP50-95"] * weights.ap50_95
        + table["AP50"] * weights.ap50
        + table["recall"] * weights.recall
        + table["precision"] * weights.precision
        + table["f1"] * weights.f1
        + table["speed_score"] * weights.speed
    )

    table = table.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    reason = build_recommendation_reason(best, weights)
    return ModelRecommendation(
        model_name=str(best["model_name"]),
        score=float(best["recommendation_score"]),
        reason=reason,
        row=best,
    ), table


def build_recommendation_reason(row: Dict[str, Any], weights: RecommendationWeights) -> str:
    parts = [
        f"лучший интегральный score={row['recommendation_score']:.4f}",
        f"AP50-95={row['AP50-95']:.4f}",
        f"AP50={row['AP50']:.4f}",
        f"recall={row['recall']:.4f}",
        f"precision={row['precision']:.4f}",
    ]
    if row.get("time", 0.0):
        parts.append(f"time={row['time']:.4f}")
    return "Рекомендуется как основной pipeline: " + "; ".join(parts) + "."


def save_recommendation_report(
    recommendation: ModelRecommendation,
    ranking: pd.DataFrame,
    out_dir: str | Path,
    weights: RecommendationWeights | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or RecommendationWeights()

    json_path = out_dir / "recommendation.json"
    csv_path = out_dir / "recommendation_ranking.csv"
    md_path = out_dir / "recommendation.md"

    payload = {
        "best_model": recommendation.to_dict(),
        "weights": asdict(weights),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ranking.to_csv(csv_path, index=False)

    md = [
        "# Рекомендация лучшего pipeline",
        "",
        f"**Рекомендуемая модель:** {recommendation.model_name}",
        f"**Интегральный score:** {recommendation.score:.4f}",
        "",
        recommendation.reason,
        "",
        "## Рейтинг моделей",
        "",
        ranking.to_markdown(index=False),
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}
