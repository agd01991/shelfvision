from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


MODEL_LABELS = {
    "yolo": "YOLO",
    "rtdetr": "RT-DETR",
    "frcnn": "Faster-R-CNN",
    "wbf": "WBF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision full pipeline runner")

    parser.add_argument("--images-dir", required=True, help="Папка изображений для пакетного инференса")
    parser.add_argument("--out-dir", default="results/full_pipeline", help="Корневая папка для всех результатов")

    parser.add_argument("--yolo-weights", help="Путь к весам YOLO")
    parser.add_argument("--rtdetr-weights", help="Путь к весам RT-DETR")
    parser.add_argument("--frcnn-weights", help="Путь к весам Faster R-CNN")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "rtdetr", "wbf"],
        choices=["yolo", "rtdetr", "frcnn", "wbf"],
        help="Какие модели запускать",
    )

    parser.add_argument("--gt-coco", help="COCO JSON с эталонной разметкой")
    parser.add_argument("--gt-yolo-labels", help="Папка YOLO labels для оценки")
    parser.add_argument("--gt-yolo-images", help="Папка изображений для YOLO labels. Если не указана, используется --images-dir")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold для оценки")

    parser.add_argument("--wbf-iou", type=float, default=0.55, help="IoU threshold для WBF")
    parser.add_argument("--wbf-skip", type=float, default=0.001, help="Skip score threshold для WBF")
    parser.add_argument("--yolo-weight", type=float, default=1.0, help="Вес YOLO в WBF")
    parser.add_argument("--rtdetr-weight", type=float, default=1.0, help="Вес RT-DETR в WBF")

    parser.add_argument("--density-model", default="yolo", choices=["yolo", "rtdetr", "frcnn", "wbf"], help="По какой модели считать плотность")
    parser.add_argument("--density-rows", type=int, default=3, help="Количество зон по вертикали")
    parser.add_argument("--density-cols", type=int, default=3, help="Количество зон по горизонтали")
    parser.add_argument("--density-limit", type=int, default=20, help="Сколько изображений визуализировать для плотности")

    parser.add_argument("--skip-evaluation", action="store_true", help="Пропустить оценку качества")
    parser.add_argument("--skip-density", action="store_true", help="Пропустить анализ плотности")
    parser.add_argument("--skip-mini-report", action="store_true", help="Пропустить мини-отчёт")
    parser.add_argument("--visualize-errors", action="store_true", help="Сохранять визуализации TP/FP/FN при оценке")
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
    comparison_dir = out_dir / "model_comparison"
    recommendation_dir = out_dir / "recommendation"
    density_dir = out_dir / "density"
    mini_report_dir = out_dir / "mini_report"

    print("=== ShelfVision full pipeline ===")
    print(f"Models: {', '.join(args.models)}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {out_dir}")

    # 1. Inference
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

    # 2. Evaluation
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

    # 3. Recommendation and comparison
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

    # 4. Density
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

    # 5. Mini report
    if not args.skip_mini_report:
        cmd = [
            sys.executable,
            "run_mini_report.py",
            "--out-dir",
            str(mini_report_dir),
            "--title",
            "ShelfVision: итоговый отчёт полного pipeline",
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

    print("\n=== DONE ===")
    print(f"All results saved to: {out_dir}")
    if not args.skip_mini_report:
        print(f"Mini report: {mini_report_dir / 'mini_report.html'}")


if __name__ == "__main__":
    main()
