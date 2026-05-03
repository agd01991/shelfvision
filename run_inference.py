from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from src.inference.prediction import ImagePrediction, save_prediction_json, save_predictions_json
from src.inference.yolo_inference import predict_yolo_folder, predict_yolo_image, prediction_summary
from src.visualization.draw_boxes import draw_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision inference runner")
    parser.add_argument("--model", choices=["yolo"], default="yolo", help="Модель для запуска")
    parser.add_argument("--weights", required=True, help="Путь к весам модели, например models/yolo/best.pt")
    parser.add_argument("--image", help="Путь к одному изображению")
    parser.add_argument("--images-dir", help="Путь к папке с изображениями для пакетной обработки")
    parser.add_argument("--out-dir", default="results/inference", help="Папка для результатов")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0 и т.д.")
    parser.add_argument("--no-masks", action="store_true", help="Не отрисовывать маски")
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    visualized_dir = out_dir / "visualized"

    if not args.image and not args.images_dir:
        raise SystemExit("Укажите --image или --images-dir")

    if args.model != "yolo":
        raise SystemExit("На первом шаге подключён только YOLO. Остальные модели добавляются следующими этапами.")

    if args.image:
        prediction = predict_yolo_image(
            model_path=args.weights,
            image_path=args.image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            model_name="YOLO",
        )
        save_prediction_json(prediction, out_dir / "prediction.json")
        draw_prediction(
            prediction,
            output_path=visualized_dir / Path(prediction.image_path).name,
            show_masks=not args.no_masks,
        )
        save_summary_csv([prediction], out_dir / "summary.csv")
        print(f"Done: objects={prediction.objects_count}, avg_conf={prediction.average_confidence:.3f}")
        print(f"Results saved to: {out_dir}")
        return

    predictions = predict_yolo_folder(
        model_path=args.weights,
        images_dir=args.images_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        model_name="YOLO",
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

    print(f"Done: images={len(predictions)}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
