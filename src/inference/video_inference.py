from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .prediction import DetectionPrediction, ImagePrediction, save_predictions_json
from .tracking import SimpleIoUTracker


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_BOX_COLOR = (0, 200, 0)
DEFAULT_MASK_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
VideoProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class FrameStats:
    frame_id: int
    source_frame_id: int
    timestamp_sec: float
    objects_count: int
    average_confidence: float
    min_confidence: float
    max_confidence: float
    inference_time: float
    fps: float


@dataclass
class VideoSummary:
    input_video: str
    output_video: Optional[str]
    model_name: str
    processed_frames: int
    source_frames: int
    frame_skip: int
    average_objects_per_frame: float
    average_confidence: float
    average_inference_time: float
    average_fps: float
    total_processing_time: float
    predictions_json: Optional[str] = None
    frames_for_identification_dir: Optional[str] = None
    tracking_enabled: bool = True
    tracking_iou: float = 0.3
    tracking_max_missing: int = 5


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _extract_masks(result: Any, boxes_count: int) -> List[Optional[List[List[float]]]]:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return [None for _ in range(boxes_count)]

    polygons: List[Optional[List[List[float]]]] = []
    for polygon in masks.xy:
        try:
            polygons.append([[float(x), float(y)] for x, y in polygon.tolist()])
        except Exception:
            polygons.append(None)

    if len(polygons) < boxes_count:
        polygons.extend([None] * (boxes_count - len(polygons)))
    return polygons[:boxes_count]


def _result_to_prediction(
    result: Any,
    image_path: str,
    model_name: str,
    inference_time: float,
    frame_width: int,
    frame_height: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImagePrediction:
    names = getattr(result, "names", {})
    detections: List[DetectionPrediction] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy)
        classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else [0] * len(xyxy)
        masks = _extract_masks(result, boxes_count=len(xyxy))

        for idx, box in enumerate(xyxy):
            class_id = int(classes[idx])
            detections.append(
                DetectionPrediction(
                    box=[float(v) for v in box.tolist()],
                    score=float(scores[idx]),
                    label=_class_name(names, class_id),
                    class_id=class_id,
                    mask=masks[idx] if idx < len(masks) else None,
                )
            )

    return ImagePrediction(
        image_path=image_path,
        model_name=model_name,
        detections=detections,
        inference_time=inference_time,
        image_width=frame_width,
        image_height=frame_height,
        metadata=metadata or {},
    )


def _draw_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, h + baseline + 4)
    cv2.rectangle(image, (x, y - h - baseline - 4), (x + w + 6, y + 2), TEXT_BG_COLOR, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 1), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_mask(image: np.ndarray, polygon: List[List[float]], alpha: float = 0.35) -> None:
    if not polygon or len(polygon) < 3:
        return
    points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], DEFAULT_MASK_COLOR)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.polylines(image, [points], isClosed=True, color=DEFAULT_MASK_COLOR, thickness=2)


def draw_video_prediction(
    frame: np.ndarray,
    prediction: ImagePrediction,
    show_masks: bool = True,
    show_footer: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    image = frame.copy()
    for detection in prediction.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.box]
        if show_masks and detection.mask:
            _draw_mask(image, detection.mask)
        cv2.rectangle(image, (x1, y1), (x2, y2), DEFAULT_BOX_COLOR, 2)
        track_text = f" id={detection.track_id}" if show_track_id and detection.track_id is not None else ""
        _draw_label(image, f"{detection.label}{track_text} {detection.score:.2f}", x1, y1)

    if show_footer:
        footer = (
            f"{prediction.model_name}: objects={prediction.objects_count}, "
            f"avg_conf={prediction.average_confidence:.3f}, time={prediction.inference_time:.3f}s"
        )
        _draw_label(image, footer, 8, image.shape[0] - 10)
    return image


def frame_stats_from_prediction(
    prediction: ImagePrediction,
    frame_id: int,
    source_frame_id: int,
    timestamp_sec: float,
) -> FrameStats:
    scores = [item.score for item in prediction.detections]
    inference_time = prediction.inference_time
    return FrameStats(
        frame_id=frame_id,
        source_frame_id=source_frame_id,
        timestamp_sec=timestamp_sec,
        objects_count=prediction.objects_count,
        average_confidence=prediction.average_confidence,
        min_confidence=min(scores) if scores else 0.0,
        max_confidence=max(scores) if scores else 0.0,
        inference_time=inference_time,
        fps=(1.0 / inference_time) if inference_time > 0 else 0.0,
    )


def save_frame_stats_csv(stats: List[FrameStats], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()) if stats else list(FrameStats.__annotations__.keys()))
        writer.writeheader()
        for item in stats:
            writer.writerow(asdict(item))
    return output_path


def build_video_summary(
    input_video: str | Path,
    output_video: Optional[str | Path],
    model_name: str,
    stats: List[FrameStats],
    source_frames: int,
    frame_skip: int,
    total_processing_time: float,
    predictions_json: Optional[str | Path] = None,
    frames_for_identification_dir: Optional[str | Path] = None,
    tracking_enabled: bool = True,
    tracking_iou: float = 0.3,
    tracking_max_missing: int = 5,
) -> VideoSummary:
    processed_frames = len(stats)
    return VideoSummary(
        input_video=str(input_video),
        output_video=str(output_video) if output_video else None,
        model_name=model_name,
        processed_frames=processed_frames,
        source_frames=source_frames,
        frame_skip=frame_skip,
        average_objects_per_frame=(sum(item.objects_count for item in stats) / processed_frames) if processed_frames else 0.0,
        average_confidence=(sum(item.average_confidence for item in stats) / processed_frames) if processed_frames else 0.0,
        average_inference_time=(sum(item.inference_time for item in stats) / processed_frames) if processed_frames else 0.0,
        average_fps=(sum(item.fps for item in stats) / processed_frames) if processed_frames else 0.0,
        total_processing_time=total_processing_time,
        predictions_json=str(predictions_json) if predictions_json else None,
        frames_for_identification_dir=str(frames_for_identification_dir) if frames_for_identification_dir else None,
        tracking_enabled=tracking_enabled,
        tracking_iou=tracking_iou,
        tracking_max_missing=tracking_max_missing,
    )


def save_video_summary_json(summary: VideoSummary, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _emit_video_progress(
    progress_callback: VideoProgressCallback | None,
    source_frame_id: int,
    source_frames: int,
    processed_frame_id: int,
    objects_count: int,
    inference_time: float,
    started_at: float,
) -> None:
    if progress_callback is None:
        return
    elapsed = time.perf_counter() - started_at
    progress_callback(
        {
            "source_frame_id": source_frame_id,
            "source_frames": source_frames,
            "processed_frames": processed_frame_id,
            "objects_count": objects_count,
            "last_inference_time": inference_time,
            "elapsed_seconds": elapsed,
            "avg_processing_fps": processed_frame_id / elapsed if elapsed > 0 else 0.0,
        }
    )


def process_yolo_video_file(
    model_path: str | Path,
    input_video: str | Path,
    out_dir: str | Path = "results/video/yolo",
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    frame_skip: int = 1,
    max_frames: int = 0,
    save_video: bool = True,
    save_sample_frames: int = 8,
    show_masks: bool = True,
    codec: str = "mp4v",
    model_name: str = "YOLO Video",
    save_frames_for_identification: bool = False,
    progress_callback: VideoProgressCallback | None = None,
    tracking_enabled: bool = True,
    tracking_iou: float = 0.3,
    tracking_max_missing: int = 5,
) -> Dict[str, Path]:
    """Processes a video file with YOLO/YOLO-Seg and saves video analytics."""

    from ultralytics import YOLO

    model_path = Path(model_path)
    input_video = Path(input_video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Не найдены веса модели: {model_path}")
    if not input_video.exists():
        raise FileNotFoundError(f"Не найден видеофайл: {input_video}")
    if input_video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Неподдерживаемое расширение видео: {input_video.suffix}")

    model = YOLO(str(model_path))
    tracker = SimpleIoUTracker(iou_threshold=tracking_iou, max_missing=tracking_max_missing) if tracking_enabled else None
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = out_dir / "output_video.mp4" if save_video else None
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_video_path), fourcc, source_fps / max(1, frame_skip), (source_width, source_height))

    stats: List[FrameStats] = []
    predictions: List[ImagePrediction] = []
    sample_dir = out_dir / "sample_frames"
    if save_sample_frames > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    identification_frames_dir = out_dir / "frames_for_identification" if save_frames_for_identification else None
    if identification_frames_dir is not None:
        identification_frames_dir.mkdir(parents=True, exist_ok=True)

    processed_frame_id = 0
    source_frame_id = 0
    start_total = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if source_frame_id % max(1, frame_skip) != 0:
            source_frame_id += 1
            continue

        if max_frames and processed_frame_id >= max_frames:
            break

        timestamp_sec = source_frame_id / source_fps if source_fps else 0.0

        frame_image_path: str
        if identification_frames_dir is not None:
            frame_path = identification_frames_dir / f"frame_{processed_frame_id:06d}_src_{source_frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_image_path = str(frame_path)
        else:
            frame_image_path = f"{input_video}#frame_{source_frame_id}"

        start_infer = time.perf_counter()
        results = model.predict(source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        inference_time = time.perf_counter() - start_infer

        prediction = _result_to_prediction(
            results[0],
            image_path=frame_image_path,
            model_name=model_name,
            inference_time=inference_time,
            frame_width=source_width,
            frame_height=source_height,
            metadata={
                "input_video": str(input_video),
                "source_frame_id": source_frame_id,
                "processed_frame_id": processed_frame_id,
                "timestamp_sec": timestamp_sec,
                "conf": conf,
                "imgsz": imgsz,
                "weights": str(model_path),
                "frame_skip": frame_skip,
                "tracking_enabled": tracking_enabled,
                "tracking_iou": tracking_iou,
                "tracking_max_missing": tracking_max_missing,
            },
        )
        if tracker is not None:
            prediction.detections = tracker.update(prediction.detections)
        predictions.append(prediction)

        drawn = draw_video_prediction(frame, prediction, show_masks=show_masks, show_track_id=tracking_enabled)
        if writer is not None:
            writer.write(drawn)

        if save_sample_frames > 0 and processed_frame_id < save_sample_frames:
            cv2.imwrite(str(sample_dir / f"frame_{processed_frame_id:06d}.jpg"), drawn)

        stats.append(frame_stats_from_prediction(prediction, processed_frame_id, source_frame_id, timestamp_sec))
        processed_frame_id += 1
        _emit_video_progress(progress_callback, source_frame_id, source_frames, processed_frame_id, prediction.objects_count, inference_time, start_total)
        source_frame_id += 1

    total_processing_time = time.perf_counter() - start_total
    capture.release()
    if writer is not None:
        writer.release()

    stats_csv = save_frame_stats_csv(stats, out_dir / "frame_stats.csv")
    predictions_json = save_predictions_json(predictions, out_dir / "video_predictions.json")
    summary = build_video_summary(
        input_video=input_video,
        output_video=output_video_path,
        model_name=model_name,
        stats=stats,
        source_frames=source_frames,
        frame_skip=frame_skip,
        total_processing_time=total_processing_time,
        predictions_json=predictions_json,
        frames_for_identification_dir=identification_frames_dir,
        tracking_enabled=tracking_enabled,
        tracking_iou=tracking_iou,
        tracking_max_missing=tracking_max_missing,
    )
    summary_json = save_video_summary_json(summary, out_dir / "video_summary.json")

    outputs: Dict[str, Path] = {"frame_stats_csv": stats_csv, "summary_json": summary_json, "predictions_json": predictions_json}
    if output_video_path is not None:
        outputs["output_video"] = output_video_path
    if save_sample_frames > 0:
        outputs["sample_frames_dir"] = sample_dir
    if identification_frames_dir is not None:
        outputs["frames_for_identification_dir"] = identification_frames_dir
    return outputs
