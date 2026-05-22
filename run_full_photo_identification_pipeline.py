from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from run_inference import get_predictors, save_summary_csv
from src.identification.demo_gallery_builder import build_demo_sku_gallery_from_predictions
from src.identification.matcher import run_sku_matching
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.threshold_analysis import save_threshold_analysis
from src.identification.visualization import visualize_identification_results
from src.inference.prediction import ImagePrediction, save_predictions_json
from src.visualization.draw_boxes import draw_prediction


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    unknown: int
    matched_rate: float
    unknown_rate: float
    avg_similarity: float
    elapsed_seconds: float
    gallery_predictions_json: str
    query_predictions_json: str
    gallery_csv: str
    identification_results_csv: str
    visualized_dir: str
    threshold_analysis_csv: str
    params: Dict[str, str | int | float | bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision full photo identification pipeline with gallery/query split")
    parser.add_argument("--model", choices=["yolo", "yolo_seg", "rtdetr", "frcnn"], default="yolo")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images-dir", default=None, help="Full image directory. Used when gallery/query dirs are not specified")
    parser.add_argument("--gallery-images-dir", default=None, help="Optional explicit gallery source image directory")
    parser.add_argument("--query-images-dir", default=None, help="Optional explicit query image directory")
    parser.add_argument("--out-dir", default="D:/1Diplom/shelfvision_results/full_photo_identification")
    parser.add_argument("--gallery-dir", default="D:/1Diplom/sku_gallery_full")
    parser.add_argument("--gallery-csv", default="D:/1Diplom/sku_gallery_full/gallery.csv")

    parser.add_argument("--limit", type=int, default=0, help="Limit total images from --images-dir before split. 0 means all")
    parser.add_argument("--gallery-count", type=int, default=50, help="How many images from --images-dir go to gallery split")
    parser.add_argument("--query-count", type=int, default=0, help="How many images go to query split. 0 means all remaining")
    parser.add_argument("--gallery-limit", type=int, default=0, help="Limit explicit gallery-images-dir. 0 means all")
    parser.add_argument("--query-limit", type=int, default=0, help="Limit explicit query-images-dir. 0 means all")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bbox-only", action="store_true")

    parser.add_argument("--max-sku", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-width", type=int, default=20)
    parser.add_argument("--min-height", type=int, default=20)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--prefix", default="sku_demo_")
    parser.add_argument("--keep-old-demo", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gt-csv", default=None)
    parser.add_argument("--visualize-limit", type=int, default=100)

    parser.add_argument("--resume", action="store_true", help="Reuse partial predictions JSONL if present")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse final predictions.json for a split if present")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-visualize-inference", action="store_true", help="Do not save detection visualizations for gallery/query inference")
    return parser.parse_args()


def _parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
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
        raise FileNotFoundError(f"Images directory not found: {root}")
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if limit and limit > 0:
        files = files[:limit]
    return files


def _split_images(args: argparse.Namespace) -> tuple[List[Path], List[Path]]:
    if args.gallery_images_dir or args.query_images_dir:
        if not args.gallery_images_dir or not args.query_images_dir:
            raise SystemExit("Specify both --gallery-images-dir and --query-images-dir, or only --images-dir")
        gallery = _list_images(args.gallery_images_dir, args.gallery_limit)
        query = _list_images(args.query_images_dir, args.query_limit)
        return gallery, query

    if not args.images_dir:
        raise SystemExit("Specify --images-dir or both --gallery-images-dir and --query-images-dir")
    all_images = _list_images(args.images_dir, args.limit)
    if not all_images:
        raise SystemExit("No images found")
    gallery_count = max(1, min(args.gallery_count, len(all_images)))
    gallery = all_images[:gallery_count]
    query = all_images[gallery_count:]
    if args.query_count and args.query_count > 0:
        query = query[: args.query_count]
    if not query:
        raise SystemExit("Query split is empty. Increase --limit or reduce --gallery-count")
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
                from src.inference.prediction import DetectionPrediction

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

    if args.skip_existing and predictions_json.exists():
        print(f"PHOTO_PROGRESS split={split_name} reused_existing={predictions_json}", flush=True)
        return predictions_json

    predict_image, _, model_name = get_predictors(args.model)
    predictions_by_path = _load_partial(partial_jsonl) if args.resume else {}
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
            _save_progress(progress_json, {"split": split_name, "processed": done, "total": total})

    ordered = [predictions_by_path[str(path)] for path in images if str(path) in predictions_by_path]
    save_predictions_json(ordered, predictions_json)
    save_summary_csv(ordered, out_dir / "summary.csv")
    return predictions_json


def _read_demo_summary(demo_dir: Path) -> Dict[str, object]:
    path = demo_dir / "demo_sku_gallery_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        unknown=int(metrics.get("unknown", 0) or 0),
        matched_rate=float(metrics.get("matched_rate", 0.0) or 0.0),
        unknown_rate=float(metrics.get("unknown_rate", 0.0) or 0.0),
        avg_similarity=float(metrics.get("avg_similarity", 0.0) or 0.0),
        elapsed_seconds=elapsed_seconds,
        gallery_predictions_json=str(gallery_predictions_json),
        query_predictions_json=str(query_predictions_json),
        gallery_csv=str(args.gallery_csv),
        identification_results_csv=str(identification_dir / "identification_results.csv"),
        visualized_dir=str(identification_dir / "visualized"),
        threshold_analysis_csv=str(threshold_outputs.get("threshold_analysis_csv", "")),
        params={
            "model": args.model,
            "weights": args.weights,
            "limit": args.limit,
            "gallery_count": args.gallery_count,
            "query_count": args.query_count,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "threshold": args.threshold,
            "thresholds": args.thresholds,
            "top_k": args.top_k,
            "max_sku": args.max_sku,
            "min_score": args.min_score,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "padding": args.padding,
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
        f"- Изображений для формирования gallery: {summary.gallery_images_count}",
        f"- Query-изображений: {summary.query_images_count}",
        f"- Создано demo SKU: {summary.created_demo_sku_count}",
        f"- Crop-ов извлечено для gallery: {summary.extracted_gallery_crops_count}",
        f"- Query-объектов найдено: {summary.query_objects_count}",
        f"- Matched: {summary.matched}",
        f"- Unknown: {summary.unknown}",
        f"- Matched rate: {summary.matched_rate:.4f}",
        f"- Unknown rate: {summary.unknown_rate:.4f}",
        f"- Avg similarity: {summary.avg_similarity:.4f}",
        f"- Общее время: {_format_eta(summary.elapsed_seconds)}",
        "",
        "## Основные файлы",
        "",
        f"- Gallery predictions: `{summary.gallery_predictions_json}`",
        f"- Query predictions: `{summary.query_predictions_json}`",
        f"- Gallery CSV: `{summary.gallery_csv}`",
        f"- Identification CSV: `{summary.identification_results_csv}`",
        f"- Threshold analysis: `{summary.threshold_analysis_csv}`",
        f"- Visualized: `{summary.visualized_dir}`",
        "",
        "## Формулировка для ВКР",
        "",
        "Для полноценной проверки модуля идентификации исходный набор изображений разделяется на две части: gallery и query. По gallery-части автоматически формируется демонстрационная SKU-галерея, а по query-части выполняется независимое сопоставление найденных объектов с данной галереей. Это позволяет показать работу контура идентификации без ручной подготовки эталонной базы для датасета, в котором отсутствует исходная SKU-разметка.",
        "",
        "## Параметры запуска",
        "",
    ]
    for key, value in summary.params.items():
        lines.append(f"- {key}: `{value}`")
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

    print("=== ShelfVision full photo identification pipeline ===", flush=True)
    gallery_images, query_images = _split_images(args)
    manifests = _save_manifest(out_dir, gallery_images, query_images)
    print(f"Manifest saved: {manifests['all_images']}", flush=True)
    print(f"Gallery images: {len(gallery_images)} | Query images: {len(query_images)}", flush=True)

    print("Step 1/5: gallery inference", flush=True)
    gallery_predictions_json = _run_split_inference("gallery", gallery_images, args, gallery_inference_dir)

    print("Step 2/5: build demo SKU gallery from gallery split", flush=True)
    build_demo_sku_gallery_from_predictions(
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
    )

    print("Step 3/5: query inference", flush=True)
    query_predictions_json = _run_split_inference("query", query_images, args, query_inference_dir)

    print("Step 4/5: identify query objects", flush=True)
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

    print("Step 5/5: save full experiment report", flush=True)
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
    )

    print("=== Done ===", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Gallery CSV: {args.gallery_csv}", flush=True)
    print(f"Identification results: {identification_dir}", flush=True)
    for name, path in {**threshold_outputs, **summary_outputs}.items():
        print(f"Report {name}: {path}", flush=True)
    print(f"Objects: {metrics.get('total_objects', 0)}", flush=True)
    print(f"Matched: {metrics.get('matched', 0)}", flush=True)
    print(f"Unknown: {metrics.get('unknown', 0)}", flush=True)


if __name__ == "__main__":
    main()
