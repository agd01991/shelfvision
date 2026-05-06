from __future__ import annotations

import argparse

from src.inference.video_inference import process_yolo_video_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision video inference runner")
    parser.add_argument("--model", choices=["yolo"], default="yolo", help="Модель для видеоинференса")
    parser.add_argument("--weights", required=True, help="Путь к весам YOLO/YOLO-Seg")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument("--out-dir", default="results/video/yolo", help="Папка для результатов")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для модели")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--frame-skip", type=int, default=1, help="Обрабатывать каждый N-й кадр")
    parser.add_argument("--max-frames", type=int, default=0, help="Максимум обработанных кадров, 0 — всё видео")
    parser.add_argument("--no-save-video", action="store_true", help="Не сохранять итоговое видео")
    parser.add_argument("--sample-frames", type=int, default=8, help="Сколько первых кадров сохранить как примеры")
    parser.add_argument("--no-masks", action="store_true", help="Не отрисовывать masks")
    parser.add_argument("--codec", default="mp4v", help="Кодек для сохранения видео")
    parser.add_argument(
        "--save-frames-for-identification",
        action="store_true",
        help="Сохранять все обработанные кадры как изображения и делать video_predictions.json совместимым с run_identification.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "yolo":
        raise SystemExit("На первом этапе видеорежима поддерживается только YOLO.")

    outputs = process_yolo_video_file(
        model_path=args.weights,
        input_video=args.video,
        out_dir=args.out_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        frame_skip=max(1, args.frame_skip),
        max_frames=max(0, args.max_frames),
        save_video=not args.no_save_video,
        save_sample_frames=max(0, args.sample_frames),
        show_masks=not args.no_masks,
        codec=args.codec,
        model_name="YOLO Video",
        save_frames_for_identification=args.save_frames_for_identification,
    )

    print("=== ShelfVision video inference ===")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
