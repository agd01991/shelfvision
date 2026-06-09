from __future__ import annotations

import argparse

from src.diagnostics.readiness import build_readiness_report


OUTPUT_LABELS_RU = {
    "report_json": "JSON-отчёт диагностики",
    "checks_csv": "CSV-таблица проверок",
    "report_md": "Markdown-отчёт диагностики",
}


def _label_output(name: str) -> str:
    return OUTPUT_LABELS_RU.get(str(name), str(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Диагностика готовности видео-идентификации ShelfVision")
    parser.add_argument("--config", default="config/shelfvision.yaml", help="Путь к config/shelfvision.yaml")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/readiness", help="Папка для отчёта диагностики")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_readiness_report(config_path=args.config, out_dir=args.out_dir)
    print("=== ShelfVision: диагностика готовности ===", flush=True)
    for name, path in outputs.items():
        print(f"- {_label_output(name)}: {path}", flush=True)


if __name__ == "__main__":
    main()
