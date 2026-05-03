from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.recommend_model import (
    RecommendationWeights,
    load_metrics_tables,
    recommend_best_model,
    save_recommendation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision model recommendation runner")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Один или несколько CSV/JSON файлов с метриками моделей",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Опциональные названия моделей, если в файлах нет model/model_name",
    )
    parser.add_argument("--out-dir", default="results/recommendation", help="Папка для отчёта рекомендации")
    parser.add_argument("--w-ap50-95", type=float, default=0.40, help="Вес AP50-95")
    parser.add_argument("--w-ap50", type=float, default=0.20, help="Вес AP50")
    parser.add_argument("--w-recall", type=float, default=0.15, help="Вес Recall")
    parser.add_argument("--w-precision", type=float, default=0.15, help="Вес Precision")
    parser.add_argument("--w-f1", type=float, default=0.05, help="Вес F1")
    parser.add_argument("--w-speed", type=float, default=0.05, help="Вес скорости")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = RecommendationWeights(
        ap50_95=args.w_ap50_95,
        ap50=args.w_ap50,
        recall=args.w_recall,
        precision=args.w_precision,
        f1=args.w_f1,
        speed=args.w_speed,
    )

    metrics_df = load_metrics_tables(args.metrics, labels=args.labels)
    recommendation, ranking = recommend_best_model(metrics_df, weights=weights)
    paths = save_recommendation_report(recommendation, ranking, args.out_dir, weights=weights)

    print("=== ShelfVision pipeline recommendation ===")
    print(f"Best model: {recommendation.model_name}")
    print(f"Score:      {recommendation.score:.4f}")
    print(recommendation.reason)
    print("Saved:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
