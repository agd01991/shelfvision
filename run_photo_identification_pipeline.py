from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from run_inference import main as inference_main
from src.identification.demo_gallery_builder import build_demo_sku_gallery_from_predictions
from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.visualization import visualize_identification_results
from src.reporting.experiment_summary import save_photo_identification_experiment_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пайплайн ShelfVision для фото-идентификации")
    parser.add_argument("--model", choices=["yolo", "yolo_seg", "rtdetr", "frcnn"], default="yolo", help="Модель детекции/сегментации")
    parser.add_argument("--weights", required=True, help="Путь к весам модели")
    parser.add_argument("--image", default=None, help="Путь к одному изображению")
    parser.add_argument("--images-dir", default=None, help="Папка изображений")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/photo_identification", help="Папка результатов пайплайна")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence детектора")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--bbox-only", action="store_true", help="Использовать bbox-crop даже при наличии масок")

    parser.add_argument("--gallery-dir", default="D:/1Diplom/sku_gallery", help="Папка demo SKU-галереи")
    parser.add_argument("--gallery-csv", default="D:/1Diplom/sku_gallery/gallery.csv", help="Путь к gallery.csv demo SKU-галереи")
    parser.add_argument("--max-sku", type=int, default=30, help="Максимальное число demo SKU")
    parser.add_argument("--min-score", type=float, default=0.35, help="Минимальная confidence детекции для эталона demo SKU")
    parser.add_argument("--min-width", type=int, default=20, help="Минимальная ширина эталонного crop")
    parser.add_argument("--min-height", type=int, default=20, help="Минимальная высота эталонного crop")
    parser.add_argument("--padding", type=float, default=0.05, help="Отступ вокруг crop")
    parser.add_argument("--prefix", default="sku_demo_", help="Префикс demo SKU")
    parser.add_argument("--keep-old-demo", action="store_true", help="Не удалять предыдущие папки demo SKU")

    parser.add_argument("--threshold", type=float, default=0.65, help="Порог идентификации SKU")
    parser.add_argument("--top-k", type=int, default=3, help="Количество ближайших SKU-кандидатов")
    parser.add_argument("--gt-csv", default=None, help="Необязательный GT CSV для метрик идентификации")
    parser.add_argument("--visualize-limit", type=int, default=50, help="Лимит визуализаций")
    return parser.parse_args()


def _run_inference(args: argparse.Namespace, inference_dir: Path) -> Path:
    import sys

    cli_args: List[str] = [
        "run_inference.py",
        "--model",
        args.model,
        "--weights",
        args.weights,
        "--out-dir",
        str(inference_dir),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
    ]
    if args.image:
        cli_args.extend(["--image", args.image])
    if args.images_dir:
        cli_args.extend(["--images-dir", args.images_dir])
    if args.device:
        cli_args.extend(["--device", args.device])
    if args.bbox_only:
        cli_args.append("--no-masks")

    old_argv = sys.argv
    try:
        sys.argv = cli_args
        inference_main()
    finally:
        sys.argv = old_argv

    prediction_file = inference_dir / ("prediction.json" if args.image else "predictions.json")
    if not prediction_file.exists():
        raise FileNotFoundError(f"Инференс завершился, но файл предсказаний не найден: {prediction_file}")
    return prediction_file


def _summary_params(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "weights": args.weights,
        "image": args.image or "",
        "images_dir": args.images_dir or "",
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device or "",
        "bbox_only": args.bbox_only,
        "gallery_dir": args.gallery_dir,
        "gallery_csv": args.gallery_csv,
        "max_sku": args.max_sku,
        "min_score": args.min_score,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "padding": args.padding,
        "prefix": args.prefix,
        "keep_old_demo": args.keep_old_demo,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "gt_csv": args.gt_csv or "",
        "visualize_limit": args.visualize_limit,
    }


def main() -> None:
    args = parse_args()
    if not args.image and not args.images_dir:
        raise SystemExit("Укажи --image или --images-dir")

    out_dir = Path(args.out_dir)
    inference_dir = out_dir / "01_inference"
    demo_gallery_out_dir = out_dir / "02_demo_gallery"
    identification_dir = out_dir / "03_identification"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ShelfVision: пайплайн фото-идентификации ===", flush=True)
    print("Шаг 1/3: инференс", flush=True)
    predictions_json = _run_inference(args, inference_dir=inference_dir)
    images_dir_for_matching = args.images_dir if args.images_dir else None

    print("Шаг 2/3: сборка demo SKU-галереи", flush=True)
    demo_outputs = build_demo_sku_gallery_from_predictions(
        predictions_json=predictions_json,
        images_dir=images_dir_for_matching,
        gallery_dir=args.gallery_dir,
        gallery_csv=args.gallery_csv,
        out_dir=demo_gallery_out_dir,
        max_sku=max(1, args.max_sku),
        min_score=args.min_score,
        min_width=max(1, args.min_width),
        min_height=max(1, args.min_height),
        use_masks=not args.bbox_only,
        padding_ratio=args.padding,
        prefix=args.prefix,
        clear_old_demo=not args.keep_old_demo,
    )

    print("Шаг 3/3: идентификация объектов на фото", flush=True)
    results = run_sku_matching(
        predictions_json=predictions_json,
        images_dir=images_dir_for_matching,
        out_dir=identification_dir,
        gallery_csv=args.gallery_csv,
        gallery_dir=args.gallery_dir,
        use_masks=not args.bbox_only,
        threshold=args.threshold,
        top_k=args.top_k,
        padding_ratio=args.padding,
    )
    metrics = evaluate_with_ground_truth(results, gt_csv=args.gt_csv)
    save_identification_metrics(metrics, out_dir=identification_dir)
    save_identification_outputs(
        predictions_json=predictions_json,
        results=results,
        metrics=metrics,
        out_dir=identification_dir,
    )
    visualize_identification_results(
        results=results,
        images_dir=images_dir_for_matching,
        out_dir=identification_dir,
        limit=max(0, args.visualize_limit),
    )
    summary_outputs = save_photo_identification_experiment_summary(
        pipeline_out_dir=out_dir,
        gallery_dir=args.gallery_dir,
        gallery_csv=args.gallery_csv,
        params=_summary_params(args),
    )

    print("=== Готово ===", flush=True)
    print(f"Папка результата пайплайна: {out_dir}", flush=True)
    print(f"Файл предсказаний: {predictions_json}", flush=True)
    print(f"Demo-галерея: {args.gallery_dir}", flush=True)
    print(f"CSV-файл галереи: {args.gallery_csv}", flush=True)
    for name, path in demo_outputs.items():
        print(f"Demo-артефакт {name}: {path}", flush=True)
    for name, path in summary_outputs.items():
        print(f"Сводка {name}: {path}", flush=True)
    print(f"Результаты идентификации: {identification_dir}", flush=True)
    print(f"Объектов: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Уверенных совпадений: {metrics.get('matched', 0)}", flush=True)
    print(f"Неоднозначных совпадений: {metrics.get('matched_uncertain', 0)}", flush=True)
    print(f"Неопределённых объектов: {metrics.get('unknown', 0)}", flush=True)


if __name__ == "__main__":
    main()
