from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from run_metadata import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Быстрый демонстрационный запуск ShelfVision для защиты")
    parser.add_argument("--images-dir", default="data/raw/sku110k_small/images", help="Папка изображений полочных сцен")
    parser.add_argument("--weights", required=True, help="Путь к весам модели")
    parser.add_argument("--model", choices=["yolo", "yolo_seg", "rtdetr", "frcnn"], default="yolo", help="Модель для применения")
    parser.add_argument("--limit", type=int, default=10, help="Сколько изображений обработать")
    parser.add_argument("--out-dir", default="results/demo_defense", help="Папка демонстрационного результата")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения")
    parser.add_argument("--threshold", type=float, default=0.65, help="Порог SKU-сопоставления")
    parser.add_argument("--top-k", type=int, default=5, help="Количество ближайших SKU-кандидатов")
    parser.add_argument("--gallery-count", type=int, default=5, help="Сколько изображений отдать под демонстрационную SKU-галерею")
    parser.add_argument("--max-sku", type=int, default=30, help="Максимум демонстрационных SKU")
    parser.add_argument("--visualize-limit", type=int, default=20, help="Сколько итоговых визуализаций сохранить")
    parser.add_argument("--device", default="", help="Устройство запуска: 0, cpu, cuda:0")
    parser.add_argument("--no-validate", action="store_true", help="Не запускать проверку результата после демо")
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    gallery_dir = out_dir / "02_demo_gallery" / "sku_gallery_final"
    gallery_csv = gallery_dir / "gallery.csv"

    pipeline_cmd: List[str] = [
        sys.executable,
        "run_full_photo_identification_pipeline.py",
        "--model",
        args.model,
        "--weights",
        args.weights,
        "--images-dir",
        args.images_dir,
        "--out-dir",
        str(out_dir),
        "--gallery-dir",
        str(gallery_dir),
        "--gallery-csv",
        str(gallery_csv),
        "--limit",
        str(max(1, args.limit)),
        "--gallery-count",
        str(max(1, min(args.gallery_count, args.limit))),
        "--query-count",
        "0",
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--max-sku",
        str(args.max_sku),
        "--min-score",
        "0.25",
        "--min-width",
        "10",
        "--min-height",
        "10",
        "--padding",
        "0.05",
        "--prefix",
        "demo_sku_",
        "--dedup-threshold",
        "0.82",
        "--max-refs-per-sku",
        "5",
        "--gallery-build-mode",
        "greedy",
        "--threshold",
        str(args.threshold),
        "--thresholds",
        "0.55,0.60,0.65,0.70,0.75,0.80",
        "--top-k",
        str(args.top_k),
        "--enable-uncertain-status",
        "--ambiguity-margin",
        "0.03",
        "--visualize-limit",
        str(args.visualize_limit),
        "--progress-every",
        "10",
        "--shuffle",
        "--seed",
        "42",
        "--resume",
        "--skip-existing",
        "--no-visualize-inference",
    ]
    if args.device:
        pipeline_cmd.extend(["--device", args.device])

    print("=== ShelfVision: быстрый демонстрационный запуск ===", flush=True)
    _run(pipeline_cmd)

    metadata_outputs = write_run_metadata(
        out_dir,
        params={
            "model": args.model,
            "weights": args.weights,
            "images_dir": args.images_dir,
            "out_dir": str(out_dir),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "limit": args.limit,
            "gallery_count": args.gallery_count,
            "max_sku": args.max_sku,
        },
    )

    if not args.no_validate:
        _run([sys.executable, "scripts/validate_run_outputs.py", "--run-dir", str(out_dir)])

    demo_report = out_dir / "demo_report.md"
    manifest = out_dir / "run_manifest.json"
    demo_report.write_text(
        "\n".join(
            [
                "# Демонстрационный запуск ShelfVision",
                "",
                f"Папка результата: `{out_dir}`",
                f"Модель: `{args.model}`",
                f"Веса: `{args.weights}`",
                f"Папка изображений: `{args.images_dir}`",
                f"Лимит изображений: {args.limit}",
                f"Паспорт запуска: `{manifest}`",
                "",
                "## Основные файлы",
                "",
                "- `run_manifest.json` - паспорт запуска",
                "- `validation_report.md` - проверка выходных файлов",
                "- `04_identification/identification_results.csv` - результаты SKU-сопоставления",
                "- `04_identification/visualized/` - итоговые визуализации с найденными и идентифицированными товарами",
            ]
        ),
        encoding="utf-8",
    )

    print("=== ShelfVision: демонстрационный запуск завершён ===", flush=True)
    print(f"Папка результата: {out_dir}", flush=True)
    print(f"Демо-отчёт: {demo_report}", flush=True)
    for name, path in metadata_outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
