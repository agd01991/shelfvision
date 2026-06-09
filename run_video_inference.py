from __future__ import annotations

import argparse
import time
from typing import Any, Dict

from src.inference.video_inference import process_yolo_video_file


OUTPUT_LABELS_RU = {
    "output_video": "размеченное видео",
    "summary_json": "JSON-сводка",
    "frame_stats_csv": "CSV-статистика по кадрам",
    "sample_frames_dir": "папка кадров-примеров",
    "frames_for_identification_dir": "папка кадров для идентификации",
    "predictions_json": "JSON-предсказания",
    "video_predictions_json": "JSON-предсказания видео",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск видеоинференса ShelfVision")
    parser.add_argument("--model", choices=["yolo", "yolo_seg"], default="yolo", help="Модель для видео")
    parser.add_argument("--weights", required=True, help="Путь к весам YOLO или YOLO-Seg")
    parser.add_argument("--video", required=True, help="Входной видеофайл")
    parser.add_argument("--out-dir", default="results/video/yolo", help="Папка результатов")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения")
    parser.add_argument("--device", default=None, help="Устройство: 0, cpu, cuda:0")
    parser.add_argument("--frame-skip", type=int, default=1, help="Обрабатывать каждый N-й кадр")
    parser.add_argument("--max-frames", type=int, default=0, help="Максимум обработанных кадров, 0 означает всё видео")
    parser.add_argument("--no-save-video", action="store_true", help="Не сохранять размеченное видео")
    parser.add_argument("--sample-frames", type=int, default=8, help="Сколько кадров-примеров сохранить")
    parser.add_argument("--no-masks", action="store_true", help="Не отрисовывать маски")
    parser.add_argument("--codec", default="mp4v", help="Видеокодек")
    parser.add_argument("--save-frames-for-identification", action="store_true", help="Сохранить обработанные кадры и совместимый video_predictions.json")
    parser.add_argument("--no-tracking", action="store_true", help="Отключить простой IoU tracking")
    parser.add_argument("--tracking-iou", type=float, default=0.30, help="IoU-порог для tracking")
    parser.add_argument("--tracking-max-missing", type=int, default=5, help="Максимум пропущенных обработанных кадров для одного трека")
    parser.add_argument("--progress-every", type=int, default=10, help="Печатать прогресс каждые N обработанных кадров")
    return parser.parse_args()


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _build_progress_callback(progress_every: int, frame_skip: int, max_frames: int):
    progress_every = max(1, int(progress_every))
    frame_skip = max(1, int(frame_skip))
    last_print_at = 0.0

    def callback(update: Dict[str, Any]) -> None:
        nonlocal last_print_at
        processed = int(update.get("processed_frames", 0) or 0)
        source_frames = int(update.get("source_frames", 0) or 0)
        source_frame_id = int(update.get("source_frame_id", 0) or 0)
        avg_fps = float(update.get("avg_processing_fps", 0.0) or 0.0)
        objects_count = int(update.get("objects_count", 0) or 0)
        elapsed = float(update.get("elapsed_seconds", 0.0) or 0.0)

        expected = 0
        if source_frames > 0:
            expected = (source_frames + frame_skip - 1) // frame_skip
        if max_frames > 0:
            expected = min(expected, max_frames) if expected else max_frames

        now = time.perf_counter()
        should_print = processed == 1 or processed % progress_every == 0 or now - last_print_at >= 5.0
        if not should_print:
            return
        last_print_at = now

        if expected > 0 and avg_fps > 0:
            remaining_frames = max(0, expected - processed)
            eta = _format_eta(remaining_frames / avg_fps)
            progress = min(100.0, processed / expected * 100.0)
            progress_text = f"{progress:.1f}%"
        else:
            eta = "неизвестно"
            progress_text = "неизвестно"

        print(
            "VIDEO_PROGRESS "
            f"processed={processed} "
            f"expected={expected or 'unknown'} "
            f"source_frame={source_frame_id}/{source_frames or 'unknown'} "
            f"progress={progress_text} "
            f"objects={objects_count} "
            f"avg_fps={avg_fps:.2f} "
            f"elapsed={_format_eta(elapsed)} "
            f"eta={eta}",
            flush=True,
        )

    return callback


def _label_output(name: str) -> str:
    return OUTPUT_LABELS_RU.get(str(name), str(name))


def main() -> None:
    args = parse_args()
    print("=== ShelfVision: видеоинференс запущен ===", flush=True)
    print(f"Модель: {args.model}", flush=True)
    print(f"Видео: {args.video}", flush=True)
    print(f"Веса: {args.weights}", flush=True)
    print(f"Папка результатов: {args.out_dir}", flush=True)
    print(f"frame_skip: {max(1, args.frame_skip)}, max_frames: {max(0, args.max_frames)}", flush=True)
    print(
        f"tracking: {not args.no_tracking}, tracking_iou: {args.tracking_iou}, "
        f"max_missing: {max(0, args.tracking_max_missing)}",
        flush=True,
    )

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
        model_name="YOLO-Seg Video" if args.model == "yolo_seg" else "YOLO Video",
        save_frames_for_identification=args.save_frames_for_identification,
        tracking_enabled=not args.no_tracking,
        tracking_iou=args.tracking_iou,
        tracking_max_missing=max(0, args.tracking_max_missing),
        progress_callback=_build_progress_callback(args.progress_every, args.frame_skip, args.max_frames),
    )

    print("=== ShelfVision: видеоинференс завершён ===", flush=True)
    for name, path in outputs.items():
        print(f"- {_label_output(name)}: {path}", flush=True)


if __name__ == "__main__":
    main()
