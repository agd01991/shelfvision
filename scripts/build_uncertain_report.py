from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


STATUS_COLUMNS = ["status", "sku_status", "assignment_status"]
MARGIN_COLUMNS = ["margin", "distinct_margin"]
DEFAULT_COLUMNS = [
    "image_name",
    "object_id",
    "sku_status",
    "sku_id",
    "sku_name",
    "best_distinct_sku",
    "second_distinct_sku",
    "sku_confidence",
    "second_distinct_score",
    "distinct_margin",
    "crop_path",
]


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _margin_column(df: pd.DataFrame) -> Optional[str]:
    for col in MARGIN_COLUMNS:
        if col in df.columns:
            return col
    return None


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл результатов не найден: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Файл результатов пустой: {path}")
    if _status_column(df) is None:
        raise ValueError(f"В файле результатов нет колонки статуса. Ожидается одна из: {STATUS_COLUMNS}")
    return df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _prepare_uncertain(df: pd.DataFrame) -> pd.DataFrame:
    status_col = _status_column(df)
    if status_col is None:
        return pd.DataFrame()
    uncertain = df[df[status_col].astype(str).eq("matched_uncertain")].copy()
    if uncertain.empty:
        return uncertain

    margin_col = _margin_column(uncertain)
    if margin_col:
        uncertain["report_margin"] = _num(uncertain[margin_col])
        uncertain = uncertain.sort_values("report_margin", ascending=True)
    elif {"sku_confidence", "second_distinct_score"}.issubset(uncertain.columns):
        uncertain["report_margin"] = _num(uncertain["sku_confidence"]) - _num(uncertain["second_distinct_score"])
        uncertain = uncertain.sort_values("report_margin", ascending=True)
    elif {"best_similarity", "second_similarity"}.issubset(uncertain.columns):
        uncertain["report_margin"] = _num(uncertain["best_similarity"]) - _num(uncertain["second_similarity"])
        uncertain = uncertain.sort_values("report_margin", ascending=True)
    return uncertain


def _existing_columns(df: pd.DataFrame) -> List[str]:
    columns = [col for col in DEFAULT_COLUMNS if col in df.columns]
    if "report_margin" in df.columns and "report_margin" not in columns:
        columns.append("report_margin")
    for col in df.columns:
        if col not in columns and col.startswith("top"):
            columns.append(col)
    return columns or list(df.columns)


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "0.0000"


def build_uncertain_report(results_csv: str | Path, out_dir: str | Path, top_n: int = 100) -> Dict[str, Path]:
    results_csv = Path(results_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_results(results_csv)
    uncertain = _prepare_uncertain(df)
    top = uncertain.head(max(1, top_n)).copy() if not uncertain.empty else uncertain

    output_csv = out_dir / "matched_uncertain_top.csv"
    report_md = out_dir / "matched_uncertain_report.md"
    columns = _existing_columns(top) if not top.empty else DEFAULT_COLUMNS
    if not top.empty:
        top[columns].to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    margin_values = _num(uncertain["report_margin"]) if "report_margin" in uncertain.columns and not uncertain.empty else pd.Series(dtype=float)
    mean_margin = float(margin_values.mean()) if not margin_values.empty else 0.0
    min_margin = float(margin_values.min()) if not margin_values.empty else 0.0

    lines = [
        "# Спорные случаи SKU-сопоставления",
        "",
        f"Файл результатов: `{results_csv}`",
        "",
        "## Сводка",
        "",
        f"- Всего объектов в результатах: {len(df)}",
        f"- Всего спорных случаев: {len(uncertain)}",
        f"- Средний margin: {_format_float(mean_margin)}",
        f"- Минимальный margin: {_format_float(min_margin)}",
        f"- Таблица топ-случаев: `{output_csv}`",
        "",
        "## Интерпретация",
        "",
        "Статус `matched_uncertain` означает, что лучший и следующий SKU-кандидат имеют близкие оценки сходства. Такие объекты не считаются полностью уверенными и должны попадать в ручную проверку.",
        "",
    ]
    if not top.empty:
        preview_cols = [col for col in columns if col in top.columns]
        lines.extend(["## Топ спорных случаев", "", top[preview_cols].to_markdown(index=False), ""])
    else:
        lines.extend(["## Топ спорных случаев", "", "Спорные случаи не найдены.", ""])

    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {"matched_uncertain_top_csv": output_csv, "matched_uncertain_report_md": report_md}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Формирование отчёта по спорным случаям SKU-сопоставления")
    parser.add_argument("--results-csv", required=True, help="Путь к identification_results.csv")
    parser.add_argument("--out-dir", required=True, help="Папка для отчёта по спорным случаям")
    parser.add_argument("--top-n", type=int, default=100, help="Сколько спорных случаев сохранить в таблице")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_uncertain_report(args.results_csv, args.out_dir, top_n=args.top_n)
    print("=== ShelfVision: отчёт по спорным SKU-сопоставлениям сформирован ===")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
