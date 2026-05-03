from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, List

from src.inference.ensemble_wbf import predict_wbf_folder, predict_wbf_image
from src.inference.faster_rcnn_inference import predict_faster_rcnn_folder, predict_faster_rcnn_image
from src.inference.prediction import ImagePrediction, save_prediction_json, save_predictions_json
from src.inference.rtdetr_inference import predict_rtdetr_folder, predict_rtdetr_image
from src.inference.yolo_inference import predict_yolo_folder, predict_yolo_image, prediction_summary
from src.visualization.draw_boxes import draw_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision inference runner")
    parser.add_argument("--model", choices=["yolo", "rtdetr", "frcnn", "wbf"], default="yolo", help="Модель для запуска")
    parser.add_argument("--weights", help="Путь к весам одной модели, например models/yolo/best.pt")
    parser.add_argument("--yolo-weights", help="Путь к весам YOLO для WBF")
    parser.add_argument("--rtdetr-weights", help="Путь к весам RT-DETR для WBF")
    parser.add_argument("--image", help="Путь к одному изображению")
    parser.add_argument("--images-dir", help="Путь к папке с изображениями для пакетной обработки")
    parser.add_argument("--out-dir", default="results/inference", help="Папка для результатов")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0 и т.д.")
    parser.add_argument("--no-masks", action="store_true", help="Не отрисовывать маски")
    parser.add_argument("--wbf-iou", type=float, default=0.55, help="IoU threshold для WBF")
    parser.add_argument("--wbf-skip", type=float, default=0.001, help="Минимальный score для WBF")
    parser.add_argument("--yolo-weight", type=float, default=1.0, help="Вес YOLO в WBF")
    parser.add_argument("--rtdetr-weight", type=float, default=1.0, help="Вес RT-DETR в WBF")
    return parser.parse_args()


def save_summary_csv(predictions: List[ImagePrediction], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [prediction_summary(item) for item in predictions]
    fieldnames = [
        "image_path",
        "model_name",
        "objects_count",
        "average_confidence",
        "min_confidence",
        "max_confidence",
        "inference_time",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def get_predictors(model: str) -> tuple[
    Callable[..., ImagePrediction],
    Callable[..., List[ImagePrediction]],
    str,
]:
    if model == "yolo":
        return predict_yolo_image, predict_yolo_folder, "YOLO"
    if model == "rtdetr":
        return predict_rtdetr_image, predict_rtdetr_folder, "RT-DETR-L"
    if model == "frcnn":
        return predict_faster_rcnn_image, predict_faster_rcnn_folder, "Faster R-CNN"
    raise ValueError(f"Неизвестная модель: {model}")


def run_wbf(args: argparse.Namespace, out_dir: Path, visualized_dir: Path) -> None:
    if not args.yolo_weights or not args.rtdetr_weights:
        raise SystemExit("Для --model wbf укажите --yolo-weights и --rtdetr-weights")

    if args.image:
        prediction = predict_wbf_image(
            yolo_model_path=args.yolo_weights,
            rtdetr_model_path=args.rtdetr_weights,
            image_path=args.image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            iou_thr=args.wbf_iou,
            skip_box_thr=args.wbf_skip,
            yolo_weight=args.yolo_weight,
            rtdetr_weight=args.rtdetr_weight,
        )
        save_prediction_json(prediction, out_dir / "prediction.json")
        draw_prediction(
            prediction,
            output_path=visualized_dir / Path(prediction.image_path).name,
            show_masks=not args.no_masks,
        )
        save_summary_csv([prediction], out_dir / "summary.csv")
        print(f"Done: model={prediction.model_name}, objects={prediction.objects_count}, avg_conf={prediction.average_confidence:.3f}")
        print(f"Results saved to: {out_dir}")
        return

    if not args.images_dir:
        raise SystemExit("Для WBF укажите --image или --images-dir")

    predictions = predict_wbf_folder(
        yolo_model_path=args.yolo_weights,
        rtdetr_model_path=args.rtdetr_weights,
        images_dir=args.images_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        iou_thr=args.wbf_iou,
        skip_box_thr=args.wbf_skip,
        yolo_weight=args.yolo_weight,
        rtdetr_weight=args.rtdetr_weight,
    )
    save_predictions_json(predictions, out_dir / "predictions.json")
    save_summary_csv(predictions, out_dir / "summary.csv")

    for prediction in predictions:
        rel_name = Path(prediction.image_path).name
        draw_prediction(
            prediction,
            output_path=visualized_dir / rel_name,
            show_masks=not args.no_masks,
        )

    print(f"Done: model=WBF(YOLO + RT-DETR), images={len(predictions)}")
    print(f"Results saved to: {out_dir}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    visualized_dir = out_dir / "visualized"

    if not args.image and not args.images_dir:
        raise SystemExit("Укажите --image или --images-dir")

    if args.model == "wbf":
        run_wbf(args, out_dir, visualized_dir)
        return

    if not args.weights:
        raise SystemExit("Для выбранной модели укажите --weights")

    predict_image, predict_folder, model_name = get_predictors(args.model)

    if args.image:
        prediction = predict_image(
            model_path=args.weights,
            image_path=args.image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            model_name=model_name,
        )
        save_prediction_json(prediction, out_dir / "prediction.json")
        draw_prediction(
            prediction,
            output_path=visualized_dir / Path(prediction.image_path).name,
            show_masks=not args.no_masks,
        )
        save_summary_csv([prediction], out_dir / "summary.csv")
        print(f"Done: model={model_name}, objects={prediction.objects_count}, avg_conf={prediction.average_confidence:.3f}")
        print(f"Results saved to: {out_dir}")
        return

    predictions = predict_folder(
        model_path=args.weights,
        images_dir=args.images_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        model_name=model_name,
    )
    save_predictions_json(predictions, out_dir / "predictions.json")
    save_summary_csv(predictions, out_dir / "summary.csv")

    for prediction in predictions:
        rel_name = Path(prediction.image_path).name
        draw_prediction(
            prediction,
            output_path=visualized_dir / rel_name,
            show_masks=not args.no_masks,
        )

    print(f"Done: model={model_name}, images={len(predictions)}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
