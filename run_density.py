from __future__ import annotations

import argparse

from src.analytics.density import analyze_density_file


OUTPUT_LABELS_RU = {
    "zones_csv": "CSV плотности по зонам",
    "summary_csv": "CSV-сводка плотности",
    "report_json": "JSON-отчёт плотности",
}


def _label_output(name: str) -> str:
    return OUTPUT_LABELS_RU.get(str(name), str(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ плотности товаров на стеллаже в ShelfVision")
    parser.add_argument("--predictions", required=True, help="prediction.json или predictions.json из run_inference.py")
    parser.add_argument("--out-dir", default="results/density", help="Папка для отчёта плотности")
    parser.add_argument("--rows", type=int, default=3, help="Количество зон по вертикали")
    parser.add_argument("--cols", type=int, default=3, help="Количество зон по горизонтали")
    parser.add_argument("--no-visualize", action="store_true", help="Не сохранять изображения с сеткой плотности")
    parser.add_argument("--limit", type=int, default=0, help="Сколько изображений обработать, 0 — все")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = analyze_density_file(
        predictions_json=args.predictions,
        out_dir=args.out_dir,
        rows=args.rows,
        cols=args.cols,
        visualize=not args.no_visualize,
        limit=args.limit,
    )

    print("=== ShelfVision: анализ плотности товаров ===")
    for name, path in paths.items():
        print(f"- {_label_output(name)}: {path}")


if __name__ == "__main__":
    main()
