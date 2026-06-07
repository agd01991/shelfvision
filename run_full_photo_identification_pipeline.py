from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from run_inference import get_predictors, save_summary_csv
from src.identification.assignment_audit import save_assignment_audit_outputs
from src.identification.clustered_gallery_builder import build_clustered_demo_sku_gallery_from_predictions
from src.identification.demo_gallery_builder import build_demo_sku_gallery_from_predictions
from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.threshold_analysis import save_threshold_analysis
from src.identification.visualization import visualize_identification_results
from src.identification.vkr_report import generate_vkr_experiment_report
from src.inference.prediction import ImagePrediction, save_predictions_json
from src.reporting.segmentation_identification_report import generate_segmentation_identification_report
from src.visualization.draw_boxes import draw_prediction


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PARAM_LABELS_RU = {
    "model": "Модель",
    "weights": "Файл весов модели",
    "limit": "Общий лимит изображений",
    "gallery_count": "Изображений для галереи",
    "query_count": "Изображений для query",
    "shuffle": "Случайное перемешивание",
    "seed": "Начальное число генератора",
    "conf": "Порог confidence детектора",
    "imgsz": "Размер изображения для инференса",
    "threshold": "Порог идентификации SKU",
    "enable_uncertain_status": "Проверка неоднозначных совпадений",
    "ambiguity_margin": "Минимальный отрыв между двумя лучшими SKU",
    "thresholds": "Пороги для анализа качества",
    "top_k": "Количество ближайших кандидатов",
    "max_sku": "Максимум demo SKU",
    "min_score": "Минимальная confidence для crop галереи",
    "min_width": "Минимальная ширина crop",
    "min_height": "Минимальная высота crop",
    "padding": "Отступ вокруг crop",
    "deduplicate_gallery": "Объединять похожие эталоны галереи",
    "dedup_threshold": "Порог объединения эталонов",
    "max_refs_per_sku": "Максимум эталонов на SKU",
    "gallery_build_mode": "Режим построения галереи",
    "cluster_merge_threshold": "Порог объединения кластеров",
    "cluster_strong_merge_threshold": "Порог сильного совпадения кластеров",
    "cluster_min_similarity": "Минимальное сходство внутри кластера",
    "cluster_pair_report_threshold": "Порог пары для отчёта",
    "cluster_max_candidates": "Максимум предварительных кандидатов",
    "resume": "Продолжать по частичным результатам",
    "skip_existing": "Переиспользовать существующие predictions.json",
}


@dataclass
class ImageManifestRow:
    split: str
    index: int
    image_path: str
    image_name: str


@dataclass
class FullExperimentSummary:
    out_dir: str
    images_dir: str
    gallery_images_count: int
    query_images_count: int
    created_demo_sku_count: int
    extracted_gallery_crops_count: int
    query_objects_count: int
    matched: int
    matched_uncertain: int
    unknown: int
    assigned: int
    matched_rate: float
    matched_uncertain_rate: float
    unknown_rate: float
    assigned_rate: float
    avg_similarity: float
    mean_distinct_margin: float
    elapsed_seconds: float
    gallery_predictions_json: str
    query_predictions_json: str
    gallery_csv: str
    identification_results_csv: str
    visualized_dir: str
    threshold_analysis_csv: str
    assignment_uncertainty_report_md: str
    segmentation_identification_report_md: str
    params: Dict[str, str | int | float | bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Полный пайплайн ShelfVision для фото-идентификации с разделением на gallery/query")
    parser.add_argument("--model", choices=["yolo", "yolo_seg", "rtdetr", "frcnn"], default="yolo", help="Модель инференса")
    parser.add_argument("--weights", required=True, help="Файл весов модели")
    parser.add_argument("--images-dir", default=None, help="Общая папка изображений. Используется, если не заданы отдельные gallery/query папки")
    parser.add_argument("--gallery-images-dir", default=None, help="Отдельная папка изображений для формирования галереи")
    parser.add_argument("--query-images-dir", default=None, help="Отдельная папка query-изображений")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/full_photo_identification", help="Папка результатов полного эксперимента")
    parser.add_argument("--gallery-dir", default="D:/1Diplom/sku_gallery_full", help="Папка создаваемой SKU-галереи")
    parser.add_argument("--gallery-csv", default="D:/1Diplom/sku_gallery_full/gallery.csv", help="CSV-файл создаваемой SKU-галереи")

    parser.add_argument("--limit", type=int, default=0, help="Общий лимит изображений из --images-dir перед разделением. 0 означает все")
    parser.add_argument("--gallery-count", type=int, default=50, help="Сколько изображений из --images-dir отправить в gallery split")
    parser.add_argument("--query-count", type=int, default=0, help="Сколько изображений отправить в query split. 0 означает все оставшиеся")
    parser.add_argument("--gallery-limit", type=int, default=0, help="Лимит для явной gallery-папки. 0 означает все")
    parser.add_argument("--query-limit", type=int, default=0, help="Лимит для явной query-папки. 0 означает все")
    parser.add_argument("--shuffle", action="store_true", help="Перемешать изображения перед разделением на gallery/query")
    parser.add_argument("--seed", type=int, default=42, help="Начальное число для воспроизводимого перемешивания")

    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence для детектора")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для инференса")
    parser.add_argument("--device", default=None, help="Устройство инференса, например 0 или cpu")
    parser.add_argument("--bbox-only", action="store_true", help="Использовать только ограничивающие прямоугольники без масок")

    parser.add_argument("--max-sku", type=int, default=100, help="Максимальное число demo SKU")
    parser.add_argument("--min-score", type=float, default=0.35, help="Минимальная confidence для crop-объекта галереи")
    parser.add_argument("--min-width", type=int, default=20, help="Минимальная ширина crop-объекта")
    parser.add_argument("--min-height", type=int, default=20, help="Минимальная высота crop-объекта")
    parser.add_argument("--padding", type=float, default=0.05, help="Отступ вокруг crop-объекта")
    parser.add_argument("--prefix", default="sku_demo_", help="Префикс для создаваемых demo SKU")
    parser.add_argument("--keep-old-demo", action="store_true", help="Не удалять старые demo SKU перед сборкой новой галереи")
    parser.add_argument("--no-deduplicate-gallery", action="store_true", help="Отключить объединение визуально похожих crop-объектов в один SKU")
    parser.add_argument("--dedup-threshold", type=float, default=0.86, help="Порог сходства для жадного объединения crop-объектов в один demo SKU")
    parser.add_argument("--max-refs-per-sku", type=int, default=3, help="Максимум эталонных изображений на demo SKU после объединения")

    parser.add_argument("--gallery-build-mode", choices=["greedy", "cluster"], default="greedy", help="Режим построения demo SKU-галереи")
    parser.add_argument("--cluster-merge-threshold", type=float, default=0.82, help="Кластерная галерея: порог объединения по центроидам/парам")
    parser.add_argument("--cluster-strong-merge-threshold", type=float, default=0.88, help="Кластерная галерея: порог сильного совпадения по одной паре")
    parser.add_argument("--cluster-min-similarity", type=float, default=0.72, help="Кластерная галерея: минимальное сходство внутри объединённого кластера")
    parser.add_argument("--cluster-pair-report-threshold", type=float, default=0.75, help="Кластерная галерея: сохранять пары SKU выше этого сходства")
    parser.add_argument("--cluster-max-candidates", type=int, default=0, help="Кластерная галерея: максимум предварительных кандидатов. 0 означает автоматически")

    parser.add_argument("--threshold", type=float, default=0.65, help="Порог уверенного совпадения SKU")
    parser.add_argument(
        "--enable-uncertain-status",
        action="store_true",
        help="Включить статус matched_uncertain, если отрыв между двумя лучшими разными SKU ниже --ambiguity-margin.",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.03,
        help="Минимальный отрыв между лучшим и вторым лучшим разными SKU для уверенного назначения.",
    )
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90", help="Список порогов для анализа качества")
    parser.add_argument("--top-k", type=int, default=3, help="Количество ближайших SKU-кандидатов")
    parser.add_argument("--gt-csv", default=None, help="CSV с эталонными SKU, если он есть")
    parser.add_argument("--visualize-limit", type=int, default=100, help="Сколько визуализаций идентификации сохранить")

    parser.add_argument("--resume", action="store_true", help="Переиспользовать predictions_partial.jsonl при наличии")
    parser.add_argument("--skip-existing", action="store_true", help="Переиспользовать итоговый predictions.json, если он соответствует текущему split")
    parser.add_argument("--progress-every", type=int, default=10, help="Печатать прогресс каждые N изображений/объектов")
    parser.add_argument("--no-visualize-inference", action="store_true", help="Не сохранять визуализации инференса для gallery/query")
    return parser.parse_args()


def _parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    return values or [0.65]


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _list_images(images_dir: str | Path, limit: int = 0) -> List[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Папка изображений не найдена: {root}")
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if limit and limit > 0:
        files = files[:limit]
    return files


def _maybe_shuffle(images: List[Path], args: argparse.Namespace) -> List[Path]:
    if not bool(getattr(args, "shuffle", False)):
        return images
    shuffled = list(images)
    random.Random(int(getattr(args, "seed", 42))).shuffle(shuffled)
    return shuffled


def _split_images(args: argparse.Namespace) -> tuple[List[Path], List[Path]]:
    if args.gallery_images_dir or args.query_images_dir:
        if not args.gallery_images_dir or not args.query_images_dir:
            raise SystemExit("Укажи одновременно --gallery-images-dir и --query-images-dir либо используй только --images-dir")
        gallery = _maybe_shuffle(_list_images(args.gallery_images_dir, args.gallery_limit), args)
        query = _maybe_shuffle(_list_images(args.query_images_dir, args.query_limit), args)
        return gallery, query

    if not args.images_dir:
        raise SystemExit("Укажи --images-dir или одновременно --gallery-images-dir и --query-images-dir")
    all_images = _maybe_shuffle(_list_images(args.images_dir, args.limit), args)
    if not all_images:
        raise SystemExit("Изображения не найдены")
    gallery_count = max(1, min(args.gallery_count, len(all_images)))
    gallery = all_images[:gallery_count]
    query = all_images[gallery_count:]
    if args.query_count and args.query_count > 0:
        query = query[: args.query_count]
    if not query:
        raise SystemExit("Query split пустой. Увеличь --limit или уменьши --gallery-count")
    return gallery, query


def _save_manifest(out_dir: Path, gallery: Sequence[Path], query: Sequence[Path]) -> Dict[str, Path]:
    manifest_dir = out_dir / "00_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: List[ImageManifestRow] = []
    for idx, path in enumerate(gallery, start=1):
        rows.append(ImageManifestRow("gallery", idx, str(path), path.name))
    for idx, path in enumerate(query, start=1):
        rows.append(ImageManifestRow("query", idx, str(path), path.name))

    all_csv = manifest_dir / "all_images.csv"
    gallery_csv = manifest_dir / "gallery_images.csv"
    query_csv = manifest_dir / "query_images.csv"
    pd.DataFrame([asdict(row) for row in rows]).to_csv(all_csv, index=False)
    pd.DataFrame([asdict(row) for row in rows if row.split == "gallery"]).to_csv(gallery_csv, index=False)
    pd.DataFrame([asdict(row) for row in rows if row.split == "query"]).to_csv(query_csv, index=False)
    return {"all_images": all_csv, "gallery_images": gallery_csv, "query_images": query_csv}


def _prediction_records(raw: object) -> List[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("predictions", "items", "images"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _prediction_image_paths(predictions_json: Path) -> List[str]:
    try:
        raw = json.loads(predictions_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    paths: List[str] = []
    for item in _prediction_records(raw):
        value = item.get("image_path") or item.get("path") or item.get("file") or item.get("filename")
        if value:
            paths.append(str(value))
    return paths


def _can_reuse_predictions(predictions_json: Path, images: Sequence[Path]) -> bool:
    expected = [str(path) for path in images]
    existing = _prediction_image_paths(predictions_json)
    if len(existing) != len(expected):
        print(
            "PHOTO_PROGRESS "
            f"reuse_mismatch={predictions_json} reason=count "
            f"existing={len(existing)} expected={len(expected)}",
            flush=True,
        )
        return False
    if set(existing) != set(expected):
        print(
            "PHOTO_PROGRESS "
            f"reuse_mismatch={predictions_json} reason=image_set_changed",
            flush=True,
        )
        return False
    return True


def _load_partial(partial_jsonl: Path) -> Dict[str, ImagePrediction]:
    loaded: Dict[str, ImagePrediction] = {}
    if not partial_jsonl.exists():
        return loaded
    from src.inference.prediction import DetectionPrediction

    with partial_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            detections = []
            boxes = raw.get("boxes", []) or []
            scores = raw.get("scores", []) or []
            labels = raw.get("labels", []) or []
            class_ids = raw.get("class_ids", []) or []
            masks = raw.get("masks", []) or []
            track_ids = raw.get("track_ids", []) or []
            for i, box in enumerate(boxes):
                detections.append(
                    DetectionPrediction(
                        box=box,
                        score=float(scores[i] if i < len(scores) else 0.0),
                        label=str(labels[i] if i < len(labels) else "product"),
                        class_id=int(class_ids[i] if i < len(class_ids) else 0),
                        mask=masks[i] if i < len(masks) else None,
                        track_id=track_ids[i] if i < len(track_ids) else None,
                    )
                )
            pred = ImagePrediction(
                image_path=str(raw.get("image_path", "")),
                model_name=str(raw.get("model_name", "")),
                detections=detections,
                inference_time=float(raw.get("inference_time", 0.0) or 0.0),
                image_width=raw.get("image_width"),
                image_height=raw.get("image_height"),
                metadata=raw.get("metadata", {}) or {},
            )
            if pred.image_path:
                loaded[pred.image_path] = pred
    return loaded


def _save_progress(progress_json: Path, payload: Dict[str, object]) -> None:
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_split_inference(split_name: str, images: Sequence[Path], args: argparse.Namespace, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_json = out_dir / "predictions.json"
    partial_jsonl = out_dir / "predictions_partial.jsonl"
    progress_json = out_dir / "progress.json"
    visualized_dir = out_dir / "visualized"

    if args.skip_existing and predictions_json.exists() and _can_reuse_predictions(predictions_json, images):
        print(f"PHOTO_PROGRESS split={split_name} reused_existing={predictions_json}", flush=True)
        return predictions_json

    predict_image, _, model_name = get_predictors(args.model)
    expected_paths = {str(path) for path in images}
    predictions_by_path = _load_partial(partial_jsonl) if args.resume else {}
    predictions_by_path = {path: pred for path, pred in predictions_by_path.items() if path in expected_paths}
    processed_before = len(predictions_by_path)
    total = len(images)
    start_time = time.perf_counter()

    print(f"PHOTO_PROGRESS split={split_name} total={total} resume_loaded={processed_before}", flush=True)
    with partial_jsonl.open("a", encoding="utf-8") as partial:
        for image_path in images:
            key = str(image_path)
            if key in predictions_by_path:
                continue
            pred = predict_image(
                model_path=args.weights,
                image_path=str(image_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                model_name=model_name,
            )
            predictions_by_path[key] = pred
            partial.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            partial.flush()

            if not args.no_visualize_inference:
                draw_prediction(pred, output_path=visualized_dir / image_path.name, show_masks=not args.bbox_only)

            done = len(predictions_by_path)
            elapsed = time.perf_counter() - start_time
            rate = max(1e-9, (done - processed_before) / elapsed) if done > processed_before else 0.0
            remaining = max(0, total - done)
            eta = remaining / rate if rate > 0 else 0.0
            if done == 1 or done % max(1, args.progress_every) == 0 or done == total:
                objects = sum(item.objects_count for item in predictions_by_path.values())
                print(
                    "PHOTO_PROGRESS "
                    f"split={split_name} processed={done}/{total} "
                    f"objects={objects} elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}",
                    flush=True,
                )
            _save_progress(
                progress_json,
                {
                    "split": split_name,
                    "processed": done,
                    "total": total,
                    "objects": sum(item.objects_count for item in predictions_by_path.values()),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
            )

    ordered = [predictions_by_path[str(path)] for path in images if str(path) in predictions_by_path]
    save_predictions_json(ordered, predictions_json)
    save_summary_csv(ordered, out_dir / "summary.csv")
    return predictions_json


def _read_demo_summary(demo_dir: Path) -> Dict[str, object]:
    path = demo_dir / "demo_sku_gallery_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_demo_gallery(args: argparse.Namespace, gallery_predictions_json: Path, demo_dir: Path) -> Dict[str, Path]:
    if args.gallery_build_mode == "cluster":
        print("GALLERY_BUILD mode=cluster provisional_candidates=true", flush=True)
        return build_clustered_demo_sku_gallery_from_predictions(
            predictions_json=gallery_predictions_json,
            images_dir=None,
            gallery_dir=args.gallery_dir,
            gallery_csv=args.gallery_csv,
            out_dir=demo_dir,
            max_sku=max(1, args.max_sku),
            min_score=args.min_score,
            min_width=max(1, args.min_width),
            min_height=max(1, args.min_height),
            use_masks=not args.bbox_only,
            padding_ratio=args.padding,
            prefix=args.prefix,
            clear_old_demo=not args.keep_old_demo,
            merge_threshold=args.cluster_merge_threshold,
            strong_merge_threshold=args.cluster_strong_merge_threshold,
            min_cluster_similarity=args.cluster_min_similarity,
            pair_report_threshold=args.cluster_pair_report_threshold,
            max_candidates=args.cluster_max_candidates,
            max_refs_per_sku=args.max_refs_per_sku,
        )

    print("GALLERY_BUILD mode=greedy", flush=True)
    return build_demo_sku_gallery_from_predictions(
        predictions_json=gallery_predictions_json,
        images_dir=None,
        gallery_dir=args.gallery_dir,
        gallery_csv=args.gallery_csv,
        out_dir=demo_dir,
        max_sku=max(1, args.max_sku),
        min_score=args.min_score,
        min_width=max(1, args.min_width),
        min_height=max(1, args.min_height),
        use_masks=not args.bbox_only,
        padding_ratio=args.padding,
        prefix=args.prefix,
        clear_old_demo=not args.keep_old_demo,
        deduplicate=not args.no_deduplicate_gallery,
        dedup_threshold=args.dedup_threshold,
        max_refs_per_sku=args.max_refs_per_sku,
    )


def _save_full_summary(
    args: argparse.Namespace,
    out_dir: Path,
    gallery_images: Sequence[Path],
    query_images: Sequence[Path],
    gallery_predictions_json: Path,
    query_predictions_json: Path,
    demo_dir: Path,
    identification_dir: Path,
    metrics: Dict[str, object],
    elapsed_seconds: float,
    threshold_outputs: Dict[str, Path],
    assignment_outputs: Dict[str, Path],
    segmentation_identification_outputs: Dict[str, Path],
) -> Dict[str, Path]:
    reports_dir = out_dir / "05_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    demo_summary = _read_demo_summary(demo_dir)
    summary = FullExperimentSummary(
        out_dir=str(out_dir),
        images_dir=str(args.images_dir or ""),
        gallery_images_count=len(gallery_images),
        query_images_count=len(query_images),
        created_demo_sku_count=int(demo_summary.get("created_sku_count", 0) or 0),
        extracted_gallery_crops_count=int(demo_summary.get("extracted_crops_count", 0) or 0),
        query_objects_count=int(metrics.get("total_objects", 0) or 0),
        matched=int(metrics.get("matched", 0) or 0),
        matched_uncertain=int(metrics.get("matched_uncertain", 0) or 0),
        unknown=int(metrics.get("unknown", 0) or 0),
        assigned=int(metrics.get("assigned", 0) or 0),
        matched_rate=float(metrics.get("matched_rate", 0.0) or 0.0),
        matched_uncertain_rate=float(metrics.get("matched_uncertain_rate", 0.0) or 0.0),
        unknown_rate=float(metrics.get("unknown_rate", 0.0) or 0.0),
        assigned_rate=float(metrics.get("assigned_rate", 0.0) or 0.0),
        avg_similarity=float(metrics.get("avg_similarity", 0.0) or 0.0),
        mean_distinct_margin=float(metrics.get("mean_distinct_margin", 0.0) or 0.0),
        elapsed_seconds=elapsed_seconds,
        gallery_predictions_json=str(gallery_predictions_json),
        query_predictions_json=str(query_predictions_json),
        gallery_csv=str(args.gallery_csv),
        identification_results_csv=str(identification_dir / "identification_results.csv"),
        visualized_dir=str(identification_dir / "visualized"),
        threshold_analysis_csv=str(threshold_outputs.get("threshold_analysis_csv", "")),
        assignment_uncertainty_report_md=str(assignment_outputs.get("assignment_uncertainty_report_md", "")),
        segmentation_identification_report_md=str(segmentation_identification_outputs.get("segmentation_identification_report_md", "")),
        params={
            "model": args.model,
            "weights": args.weights,
            "limit": args.limit,
            "gallery_count": args.gallery_count,
            "query_count": args.query_count,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "threshold": args.threshold,
            "enable_uncertain_status": args.enable_uncertain_status,
            "ambiguity_margin": args.ambiguity_margin,
            "thresholds": args.thresholds,
            "top_k": args.top_k,
            "max_sku": args.max_sku,
            "min_score": args.min_score,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "padding": args.padding,
            "deduplicate_gallery": not args.no_deduplicate_gallery,
            "dedup_threshold": args.dedup_threshold,
            "max_refs_per_sku": args.max_refs_per_sku,
            "gallery_build_mode": args.gallery_build_mode,
            "cluster_merge_threshold": args.cluster_merge_threshold,
            "cluster_strong_merge_threshold": args.cluster_strong_merge_threshold,
            "cluster_min_similarity": args.cluster_min_similarity,
            "cluster_pair_report_threshold": args.cluster_pair_report_threshold,
            "cluster_max_candidates": args.cluster_max_candidates,
            "resume": args.resume,
            "skip_existing": args.skip_existing,
        },
    )
    raw = asdict(summary)
    json_path = reports_dir / "full_experiment_summary.json"
    csv_path = reports_dir / "full_experiment_summary.csv"
    md_path = reports_dir / "full_experiment_summary.md"
    json_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in raw.items() if k != "params"}]).to_csv(csv_path, index=False)

    lines = [
        "# ShelfVision: полный эксперимент фото-идентификации",
        "",
        "## Сводка",
        "",
        f"- Изображений для формирования галереи: {summary.gallery_images_count}",
        f"- Query-изображений: {summary.query_images_count}",
        f"- Создано demo SKU: {summary.created_demo_sku_count}",
        f"- Crop-объектов извлечено для галереи: {summary.extracted_gallery_crops_count}",
        f"- Query-объектов найдено: {summary.query_objects_count}",
        f"- Уверенные совпадения: {summary.matched}",
        f"- Неоднозначные совпадения: {summary.matched_uncertain}",
        f"- Неопределённые объекты: {summary.unknown}",
        f"- Всего назначений SKU: {summary.assigned}",
        f"- Доля уверенных совпадений: {summary.matched_rate:.4f}",
        f"- Доля неоднозначных совпадений: {summary.matched_uncertain_rate:.4f}",
        f"- Доля неопределённых объектов: {summary.unknown_rate:.4f}",
        f"- Доля всех назначений SKU: {summary.assigned_rate:.4f}",
        f"- Среднее визуальное сходство: {summary.avg_similarity:.4f}",
        f"- Средний отрыв между двумя лучшими SKU: {summary.mean_distinct_margin:.4f}",
        f"- Общее время: {_format_eta(summary.elapsed_seconds)}",
        "",
        "## Основные файлы",
        "",
        f"- Предсказания gallery: `{summary.gallery_predictions_json}`",
        f"- Предсказания query: `{summary.query_predictions_json}`",
        f"- CSV-файл SKU-галереи: `{summary.gallery_csv}`",
        f"- CSV-файл идентификации: `{summary.identification_results_csv}`",
        f"- Анализ порогов: `{summary.threshold_analysis_csv}`",
        f"- Отчёт аудита неоднозначности: `{summary.assignment_uncertainty_report_md}`",
        f"- Отчёт по связке сегментации и идентификации: `{summary.segmentation_identification_report_md}`",
        f"- Визуализации: `{summary.visualized_dir}`",
        "",
        "## Формулировка для ВКР",
        "",
        "Для полноценной проверки модуля идентификации исходный набор изображений разделяется на две части: gallery и query. По gallery-части автоматически формируется демонстрационная SKU-галерея, а по query-части выполняется независимое сопоставление найденных объектов с данной галереей. Дополнительно формируется отчёт по связке сегментации/локализации и идентификации, который показывает соответствие реализации теме ВКР.",
        "",
        "## Параметры запуска",
        "",
    ]
    for key, value in summary.params.items():
        label = PARAM_LABELS_RU.get(key, key)
        lines.append(f"- {label}: `{value}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary_json": json_path, "summary_csv": csv_path, "summary_md": md_path}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    gallery_inference_dir = out_dir / "01_gallery_inference"
    demo_dir = out_dir / "02_demo_gallery"
    query_inference_dir = out_dir / "03_query_inference"
    identification_dir = out_dir / "04_identification"
    reports_dir = out_dir / "05_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ShelfVision: полный пайплайн фото-идентификации ===", flush=True)
    gallery_images, query_images = _split_images(args)
    manifests = _save_manifest(out_dir, gallery_images, query_images)
    print(f"Манифест сохранён: {manifests['all_images']}", flush=True)
    print(f"Изображений для gallery: {len(gallery_images)} | изображений для query: {len(query_images)}", flush=True)
    if args.shuffle:
        print(f"Перемешивание включено: seed={args.seed}", flush=True)

    print("Шаг 1/5: инференс gallery-части", flush=True)
    gallery_predictions_json = _run_split_inference("gallery", gallery_images, args, gallery_inference_dir)

    print("Шаг 2/5: сборка demo SKU-галереи по gallery-части", flush=True)
    _build_demo_gallery(args=args, gallery_predictions_json=gallery_predictions_json, demo_dir=demo_dir)

    print("Шаг 3/5: инференс query-части", flush=True)
    query_predictions_json = _run_split_inference("query", query_images, args, query_inference_dir)

    print("Шаг 4/5: идентификация query-объектов", flush=True)
    results = run_sku_matching(
        predictions_json=query_predictions_json,
        images_dir=None,
        out_dir=identification_dir,
        gallery_csv=args.gallery_csv,
        gallery_dir=args.gallery_dir,
        use_masks=not args.bbox_only,
        threshold=args.threshold,
        top_k=args.top_k,
        padding_ratio=args.padding,
        progress_every=args.progress_every,
        enable_uncertain_status=bool(args.enable_uncertain_status),
        ambiguity_margin=float(args.ambiguity_margin),
    )
    metrics = evaluate_with_ground_truth(results, gt_csv=args.gt_csv)
    save_identification_metrics(metrics, out_dir=identification_dir)
    save_identification_outputs(
        predictions_json=query_predictions_json,
        results=results,
        metrics=metrics,
        out_dir=identification_dir,
    )
    visualize_identification_results(
        results=results,
        images_dir=None,
        out_dir=identification_dir,
        limit=max(0, args.visualize_limit),
    )
    threshold_outputs = save_threshold_analysis(results, out_dir=reports_dir, thresholds=_parse_thresholds(args.thresholds))
    assignment_outputs = save_assignment_audit_outputs(
        results=results,
        out_dir=identification_dir,
        threshold=float(args.threshold),
        ambiguity_margin=float(args.ambiguity_margin),
    )
    segmentation_identification_outputs = generate_segmentation_identification_report(
        out_dir=out_dir,
        query_predictions_json=query_predictions_json,
        identification_dir=identification_dir,
        reports_dir=reports_dir,
    )

    print("Шаг 5/5: сохранение итогового отчёта полного эксперимента", flush=True)
    summary_outputs = _save_full_summary(
        args=args,
        out_dir=out_dir,
        gallery_images=gallery_images,
        query_images=query_images,
        gallery_predictions_json=gallery_predictions_json,
        query_predictions_json=query_predictions_json,
        demo_dir=demo_dir,
        identification_dir=identification_dir,
        metrics=metrics,
        elapsed_seconds=time.perf_counter() - started,
        threshold_outputs=threshold_outputs,
        assignment_outputs=assignment_outputs,
        segmentation_identification_outputs=segmentation_identification_outputs,
    )
    vkr_outputs = generate_vkr_experiment_report(out_dir)

    print("=== Готово ===", flush=True)
    print(f"Папка результата: {out_dir}", flush=True)
    print(f"CSV-файл SKU-галереи: {args.gallery_csv}", flush=True)
    print(f"Результаты идентификации: {identification_dir}", flush=True)
    for name, path in {**threshold_outputs, **assignment_outputs, **segmentation_identification_outputs, **summary_outputs, **vkr_outputs}.items():
        print(f"Отчёт {name}: {path}", flush=True)
    print(f"Объектов: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Уверенных совпадений: {metrics.get('matched', 0)}", flush=True)
    print(f"Неоднозначных совпадений: {metrics.get('matched_uncertain', 0)}", flush=True)
    print(f"Неопределённых объектов: {metrics.get('unknown', 0)}", flush=True)
    print(f"Отчёт по связке сегментации и идентификации: {segmentation_identification_outputs.get('segmentation_identification_report_md')}", flush=True)


if __name__ == "__main__":
    main()
