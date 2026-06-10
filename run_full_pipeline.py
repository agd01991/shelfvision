from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


MODEL_LABELS = {
    "yolo": "YOLO",
    "yolo_seg": "YOLO-Seg",
    "rtdetr": "RT-DETR",
    "frcnn": "Faster-R-CNN",
    "wbf": "WBF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Полный пайплайн ShelfVision")

    parser.add_argument("--images-dir", required=True, help="Папка изображений для пакетного инференса")
    parser.add_argument("--out-dir", default="results/full_pipeline", help="Корневая папка для всех результатов")

    parser.add_argument("--yolo-weights", help="Путь к весам YOLO")
    parser.add_argument("--yolo-seg-weights", help="Путь к весам YOLO-Seg")
    parser.add_argument("--rtdetr-weights", help="Путь к весам RT-DETR")
    parser.add_argument("--frcnn-weights", help="Путь к весам Faster R-CNN")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "rtdetr", "wbf"],
        choices=["yolo", "yolo_seg", "rtdetr", "frcnn", "wbf"],
        help="Какие модели запускать",
    )

    parser.add_argument("--gt-coco", help="COCO JSON с эталонной bbox/segmentation-разметкой")
    parser.add_argument("--gt-yolo-labels", help="Папка YOLO labels для bbox-оценки")
    parser.add_argument("--gt-yolo-images", help="Папка изображений для YOLO labels. Если не указана, используется --images-dir")

    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU-порог для bbox/mask оценки")

    parser.add_argument("--wbf-iou", type=float, default=0.55, help="IoU-порог для WBF")
    parser.add_argument("--wbf-skip", type=float, default=0.001, help="Порог пропуска для WBF")
    parser.add_argument("--yolo-weight", type=float, default=1.0, help="Вес YOLO в WBF")
    parser.add_argument("--rtdetr-weight", type=float, default=1.0, help="Вес RT-DETR в WBF")

    parser.add_argument(
        "--density-model",
        default="yolo",
        choices=["yolo", "yolo_seg", "rtdetr", "frcnn", "wbf"],
        help="По какой модели считать плотность",
    )
    parser.add_argument("--density-rows", type=int, default=3, help="Количество зон по вертикали")
    parser.add_argument("--density-cols", type=int, default=3, help="Количество зон по горизонтали")
    parser.add_argument("--density-limit", type=int, default=20, help="Сколько изображений визуализировать для плотности")

    parser.add_argument("--run-identification", action="store_true", help="Запустить SKU-идентификацию после инференса")
    parser.add_argument(
        "--identification-model",
        default="yolo_seg",
        choices=["yolo", "yolo_seg", "rtdetr", "frcnn", "wbf"],
        help="По предсказаниям какой модели запускать идентификацию",
    )
    parser.add_argument("--sku-gallery-csv", default=None, help="CSV SKU-галереи: sku_id, sku_name, category, image_path")
    parser.add_argument("--sku-gallery-dir", default=None, help="Папка SKU-галереи вида <sku_id>/*.jpg")
    parser.add_argument("--sku-gt-csv", default=None, help="Необязательный GT CSV: image_name, object_id, true_sku_id")
    parser.add_argument("--sku-threshold", type=float, default=0.65, help="Порог визуального сходства для matched/unknown")
    parser.add_argument("--sku-top-k", type=int, default=3, help="Сколько кандидатов SKU сохранять")
    parser.add_argument("--sku-padding", type=float, default=0.05, help="Отступ вокруг bbox при извлечении crop")
    parser.add_argument("--sku-use-masks", action="store_true", help="Использовать mask-crop, если маски есть")

    parser.add_argument("--skip-evaluation", action="store_true", help="Пропустить bbox-оценку качества")
    parser.add_argument("--skip-segmentation-evaluation", action="store_true", help="Пропустить mask-оценку YOLO-Seg")
    parser.add_argument("--skip-density", action="store_true", help="Пропустить анализ плотности")
    parser.add_argument("--skip-mini-report", action="store_true", help="Пропустить мини-отчёт")
    parser.add_argument("--visualize-errors", action="store_true", help="Сохранять визуализации TP/FP/FN при bbox-оценке")
    parser.add_argument("--error-limit", type=int, default=20, help="Сколько error-визуализаций сохранять")
    return parser.parse_args()


def run(cmd: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def weight_args_for_model(args: argparse.Namespace, model: str) -> List[str]:
    if model == "yolo":
        require(bool(args.yolo_weights), "Для модели yolo укажите --yolo-weights")
        return ["--weights", args.yolo_weights]
    if model == "yolo_seg":
        require(bool(args.yolo_seg_weights), "Для модели yolo_seg укажите --yolo-seg-weights")
        return ["--weights", args.yolo_seg_weights]
    if model == "rtdetr":
        require(bool(args.rtdetr_weights), "Для модели rtdetr укажите --rtdetr-weights")
        return ["--weights", args.rtdetr_weights]
    if model == "frcnn":
        require(bool(args.frcnn_weights), "Для модели frcnn укажите --frcnn-weights")
        return ["--weights", args.frcnn_weights]
    if model == "wbf":
        require(bool(args.yolo_weights), "Для модели wbf укажите --yolo-weights")
        require(bool(args.rtdetr_weights), "Для модели wbf укажите --rtdetr-weights")
        return [
            "--yolo-weights",
            args.yolo_weights,
            "--rtdetr-weights",
            args.rtdetr_weights,
            "--wbf-iou",
            str(args.wbf_iou),
            "--wbf-skip",
            str(args.wbf_skip),
            "--yolo-weight",
            str(args.yolo_weight),
            "--rtdetr-weight",
            str(args.rtdetr_weight),
        ]
    raise ValueError(model)


def evaluation_gt_args(args: argparse.Namespace) -> List[str]:
    if args.gt_coco:
        return ["--gt-coco", args.gt_coco]
    if args.gt_yolo_labels:
        return [
            "--gt-yolo-labels",
            args.gt_yolo_labels,
            "--images-dir",
            args.gt_yolo_images or args.images_dir,
        ]
    raise SystemExit("Для оценки укажите --gt-coco или --gt-yolo-labels")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    inference_root = out_dir / "inference"
    evaluation_root = out_dir / "evaluation"
    segmentation_evaluation_root = out_dir / "segmentation_evaluation"
    identification_dir = out_dir / "identification"
    comparison_dir = out_dir / "model_comparison"
    recommendation_dir = out_dir / "recommendation"
    density_dir = out_dir / "density"
    mini_report_dir = out_dir / "mini_report"

    print("=== ShelfVision: полный пайплайн ===")
    print(f"Модели: {', '.join(args.models)}")
    print(f"Изображения: {args.images_dir}")
    print(f"Папка результатов: {out_dir}")

    # 1. Инференс
    for model in args.models:
        model_out = inference_root / model
        cmd = [
            sys.executable,
            "run_inference.py",
            "--model",
            model,
            *weight_args_for_model(args, model),
            "--images-dir",
            args.images_dir,
            "--out-dir",
            str(model_out),
            "--conf",
            str(args.conf),
            "--imgsz",
            str(args.imgsz),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        run(cmd, cwd=root)

    # 2. BBox-оценка
    metrics_files: List[Path] = []
    labels: List[str] = []
    if not args.skip_evaluation:
        gt_args = evaluation_gt_args(args)
        for model in args.models:
            predictions = inference_root / model / "predictions.json"
            eval_out = evaluation_root / model
            cmd = [
                sys.executable,
                "run_evaluation.py",
                "--predictions",
                str(predictions),
                *gt_args,
                "--out-dir",
                str(eval_out),
                "--iou",
                str(args.iou),
            ]
            if args.visualize_errors:
                cmd.extend(["--visualize-errors", "--limit", str(args.error_limit)])
            run(cmd, cwd=root)
            metrics_files.append(eval_out / "metrics_summary.csv")
            labels.append(MODEL_LABELS[model])

    # 2b. Mask-оценка YOLO-Seg
    if not args.skip_segmentation_evaluation and "yolo_seg" in args.models:
        require(bool(args.gt_coco), "Для mask-оценки YOLO-Seg нужен --gt-coco с segmentation-разметкой")
        run(
            [
                sys.executable,
                "run_segmentation_evaluation.py",
                "--predictions",
                str(inference_root / "yolo_seg" / "predictions.json"),
                "--gt-coco",
                args.gt_coco,
                "--out-dir",
                str(segmentation_evaluation_root / "yolo_seg"),
                "--iou",
                str(args.iou),
            ],
            cwd=root,
        )

    # 2c. SKU-идентификация
    if args.run_identification:
        require(args.identification_model in args.models, "Модель для идентификации должна быть указана в --models")
        require(bool(args.sku_gallery_csv or args.sku_gallery_dir), "Для идентификации укажите --sku-gallery-csv или --sku-gallery-dir")
        cmd = [
            sys.executable,
            "run_identification.py",
            "--predictions",
            str(inference_root / args.identification_model / "predictions.json"),
            "--images-dir",
            args.images_dir,
            "--out-dir",
            str(identification_dir / args.identification_model),
            "--threshold",
            str(args.sku_threshold),
            "--top-k",
            str(args.sku_top_k),
            "--padding",
            str(args.sku_padding),
        ]
        if args.sku_gallery_csv:
            cmd.extend(["--gallery-csv", args.sku_gallery_csv])
        if args.sku_gallery_dir:
            cmd.extend(["--gallery-dir", args.sku_gallery_dir])
        if args.sku_gt_csv:
            cmd.extend(["--gt-csv", args.sku_gt_csv])
        if args.sku_use_masks:
            cmd.append("--use-masks")
        run(cmd, cwd=root)

    # 3. Рекомендация и сравнение моделей
    if metrics_files:
        run(
            [
                sys.executable,
                "run_recommendation.py",
                "--metrics",
                *[str(path) for path in metrics_files],
                "--labels",
                *labels,
                "--out-dir",
                str(recommendation_dir),
            ],
            cwd=root,
        )
        run(
            [
                sys.executable,
                "run_compare.py",
                "--metrics",
                *[str(path) for path in metrics_files],
                "--labels",
                *labels,
                "--out-dir",
                str(comparison_dir),
            ],
            cwd=root,
        )

    # 4. Анализ плотности
    density_report = None
    density_summary = None
    density_images_dir = None
    if not args.skip_density and args.density_model in args.models:
        density_predictions = inference_root / args.density_model / "predictions.json"
        run(
            [
                sys.executable,
                "run_density.py",
                "--predictions",
                str(density_predictions),
                "--out-dir",
                str(density_dir / args.density_model),
                "--rows",
                str(args.density_rows),
                "--cols",
                str(args.density_cols),
                "--limit",
                str(args.density_limit),
            ],
            cwd=root,
        )
        density_report = density_dir / args.density_model / "density_report.json"
        density_summary = density_dir / args.density_model / "density_summary.csv"
        density_images_dir = density_dir / args.density_model / "visualized"

    # 5. Мини-отчёт
    if not args.skip_mini_report:
        cmd = [
            sys.executable,
            "run_mini_report.py",
            "--out-dir",
            str(mini_report_dir),
            "--title",
            "ShelfVision: итоговый отчёт полного пайплайна",
        ]
        if metrics_files:
            cmd.extend(
                [
                    "--comparison-json",
                    str(comparison_dir / "model_comparison.json"),
                    "--comparison-csv",
                    str(comparison_dir / "model_comparison.csv"),
                    "--recommendation-json",
                    str(recommendation_dir / "recommendation.json"),
                ]
            )
        if density_report and density_summary:
            cmd.extend(["--density-json", str(density_report), "--density-csv", str(density_summary)])
        if density_images_dir:
            cmd.extend(["--images-dir", str(density_images_dir)])
        run(cmd, cwd=root)

    print("\n=== Готово ===")
    print(f"Все результаты сохранены в: {out_dir}")
    if args.run_identification:
        print(f"Идентификация: {identification_dir / args.identification_model}")
    if not args.skip_mini_report:
        print(f"Мини-отчёт: {mini_report_dir / 'mini_report.html'}")


if __name__ == "__main__":
    main()
