from __future__ import annotations

import argparse

from src.evaluation.compare_models import compare_models, save_comparison_report
from src.evaluation.recommend_model import RecommendationWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision multi-model comparison report")
    parser.add_argument("--metrics", nargs="+", required=True, help="CSV/JSON файлы метрик моделей")
    parser.add_argument("--labels", nargs="*", help="Названия моделей, если в файлах нет model/model_name")
    parser.add_argument("--out-dir", default="results/model_comparison", help="Папка для отчёта сравнения")
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

    comparison, ranking = compare_models(args.metrics, labels=args.labels, weights=weights)
    paths = save_comparison_report(comparison, ranking, args.out_dir, weights=weights)

    print("=== ShelfVision model comparison ===")
    print(f"Best pipeline: {comparison.best_model}")
    print(f"Score:         {comparison.best_score:.4f}")
    print("Highlights:")
    for item in comparison.highlights:
        print(f"- {item}")
    print("Saved:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
