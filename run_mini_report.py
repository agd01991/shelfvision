from __future__ import annotations

import argparse

from src.reporting.mini_report import build_mini_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision presentation mini report builder")
    parser.add_argument("--out-dir", default="results/mini_report", help="Папка для мини-отчёта")
    parser.add_argument("--title", default="ShelfVision: итоговый мини-отчёт", help="Заголовок отчёта")
    parser.add_argument("--comparison-json", help="results/model_comparison/model_comparison.json")
    parser.add_argument("--comparison-csv", help="results/model_comparison/model_comparison.csv")
    parser.add_argument("--recommendation-json", help="results/recommendation/recommendation.json")
    parser.add_argument("--density-json", help="results/density/.../density_report.json")
    parser.add_argument("--density-csv", help="results/density/.../density_summary.csv")
    parser.add_argument("--images-dir", help="Папка с изображениями для демонстрации")
    parser.add_argument("--image-limit", type=int, default=8, help="Сколько изображений добавить в отчёт")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_mini_report(
        out_dir=args.out_dir,
        title=args.title,
        comparison_json=args.comparison_json,
        comparison_csv=args.comparison_csv,
        recommendation_json=args.recommendation_json,
        density_json=args.density_json,
        density_csv=args.density_csv,
        images_dir=args.images_dir,
        image_limit=args.image_limit,
    )

    print("=== ShelfVision mini report ===")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
