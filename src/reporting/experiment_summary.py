from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PARAM_LABELS_RU = {
    "model": "Модель",
    "weights": "Файл весов модели",
    "conf": "Порог confidence детектора",
    "imgsz": "Размер изображения для инференса",
    "threshold": "Порог идентификации SKU",
    "top_k": "Количество ближайших кандидатов",
    "padding": "Отступ вокруг crop",
    "dedup_threshold": "Порог объединения эталонов",
    "max_refs_per_sku": "Максимум эталонов на SKU",
    "max_sku": "Максимум demo SKU",
    "bbox_only": "Только ограничивающие прямоугольники",
    "use_masks": "Использовать маски",
}


def _read_json(path: str | Path) -> Dict[str, Any] | List[Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _count_visualized(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*_identified.jpg") if item.is_file())


def build_photo_identification_experiment_summary(
    pipeline_out_dir: str | Path,
    gallery_dir: str | Path,
    gallery_csv: str | Path,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pipeline_out_dir = Path(pipeline_out_dir)
    gallery_dir = Path(gallery_dir)
    gallery_csv = Path(gallery_csv)
    params = params or {}

    inference_dir = pipeline_out_dir / "01_inference"
    demo_dir = pipeline_out_dir / "02_demo_gallery"
    identification_dir = pipeline_out_dir / "03_identification"
    metrics = _read_json(identification_dir / "identification_metrics.json")
    demo_summary = _read_json(demo_dir / "demo_sku_gallery_summary.json")
    predictions = _read_json(inference_dir / "predictions.json")
    if not predictions:
        predictions = _read_json(inference_dir / "prediction.json")

    images_count = len(predictions) if isinstance(predictions, list) else 1 if predictions else 0
    total_objects = _safe_int(metrics.get("total_objects", 0) if isinstance(metrics, dict) else 0)
    matched = _safe_int(metrics.get("matched", 0) if isinstance(metrics, dict) else 0)
    matched_uncertain = _safe_int(metrics.get("matched_uncertain", 0) if isinstance(metrics, dict) else 0)
    unknown = _safe_int(metrics.get("unknown", 0) if isinstance(metrics, dict) else 0)
    assigned = _safe_int(metrics.get("assigned", matched + matched_uncertain) if isinstance(metrics, dict) else 0)
    matched_rate = _safe_float(metrics.get("matched_rate", 0.0) if isinstance(metrics, dict) else 0.0)
    matched_uncertain_rate = _safe_float(metrics.get("matched_uncertain_rate", 0.0) if isinstance(metrics, dict) else 0.0)
    unknown_rate = _safe_float(metrics.get("unknown_rate", 0.0) if isinstance(metrics, dict) else 0.0)
    assigned_rate = _safe_float(metrics.get("assigned_rate", 0.0) if isinstance(metrics, dict) else 0.0)
    avg_similarity = _safe_float(metrics.get("avg_similarity", 0.0) if isinstance(metrics, dict) else 0.0)
    mean_distinct_margin = _safe_float(metrics.get("mean_distinct_margin", 0.0) if isinstance(metrics, dict) else 0.0)

    created_sku_count = _safe_int(demo_summary.get("created_sku_count", 0) if isinstance(demo_summary, dict) else 0)
    extracted_crops_count = _safe_int(demo_summary.get("extracted_crops_count", 0) if isinstance(demo_summary, dict) else 0)
    visualized_count = _count_visualized(identification_dir / "visualized")

    summary = {
        "pipeline_out_dir": str(pipeline_out_dir),
        "inference_dir": str(inference_dir),
        "demo_gallery_report_dir": str(demo_dir),
        "identification_dir": str(identification_dir),
        "gallery_dir": str(gallery_dir),
        "gallery_csv": str(gallery_csv),
        "images_count": images_count,
        "total_objects": total_objects,
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "assigned": assigned,
        "matched_rate": matched_rate,
        "matched_uncertain_rate": matched_uncertain_rate,
        "unknown_rate": unknown_rate,
        "assigned_rate": assigned_rate,
        "avg_similarity": avg_similarity,
        "mean_distinct_margin": mean_distinct_margin,
        "created_demo_sku_count": created_sku_count,
        "extracted_crops_count": extracted_crops_count,
        "visualized_images_count": visualized_count,
        "predictions_json": str(inference_dir / "predictions.json"),
        "identification_results_csv": str(identification_dir / "identification_results.csv"),
        "identification_results_json": str(identification_dir / "identification_results.json"),
        "identified_predictions_json": str(identification_dir / "identified_predictions.json"),
        "identification_report_md": str(identification_dir / "identification_report.md"),
        "demo_gallery_report_md": str(demo_dir / "demo_sku_gallery_report.md"),
        "visualized_dir": str(identification_dir / "visualized"),
        "crops_dir": str(identification_dir / "crops"),
        "params": params,
    }
    return summary


def save_photo_identification_experiment_summary(
    pipeline_out_dir: str | Path,
    gallery_dir: str | Path,
    gallery_csv: str | Path,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    pipeline_out_dir = Path(pipeline_out_dir)
    summary = build_photo_identification_experiment_summary(
        pipeline_out_dir=pipeline_out_dir,
        gallery_dir=gallery_dir,
        gallery_csv=gallery_csv,
        params=params,
    )

    json_path = pipeline_out_dir / "experiment_summary.json"
    csv_path = pipeline_out_dir / "experiment_summary.csv"
    md_path = pipeline_out_dir / "experiment_summary.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in summary.items() if k != "params"}]).to_csv(csv_path, index=False)

    lines = [
        "# ShelfVision: итоговый отчёт эксперимента фото-идентификации",
        "",
        "## Сводка для ВКР",
        "",
        f"- Обработано изображений: {summary['images_count']}",
        f"- Найдено объектов: {summary['total_objects']}",
        f"- Уверенные совпадения: {summary['matched']}",
        f"- Неоднозначные совпадения: {summary['matched_uncertain']}",
        f"- Неопределённые объекты: {summary['unknown']}",
        f"- Всего назначений SKU: {summary['assigned']}",
        f"- Доля уверенных совпадений: {summary['matched_rate']:.4f}",
        f"- Доля неоднозначных совпадений: {summary['matched_uncertain_rate']:.4f}",
        f"- Доля неопределённых объектов: {summary['unknown_rate']:.4f}",
        f"- Доля всех назначений SKU: {summary['assigned_rate']:.4f}",
        f"- Среднее визуальное сходство: {summary['avg_similarity']:.4f}",
        f"- Средний отрыв между двумя лучшими SKU: {summary['mean_distinct_margin']:.4f}",
        f"- Создано demo SKU: {summary['created_demo_sku_count']}",
        f"- Извлечено crop-объектов: {summary['extracted_crops_count']}",
        f"- Визуализировано изображений: {summary['visualized_images_count']}",
        "",
        "## Основные файлы",
        "",
        f"- predictions.json: `{summary['predictions_json']}`",
        f"- gallery.csv: `{summary['gallery_csv']}`",
        f"- identification_results.csv: `{summary['identification_results_csv']}`",
        f"- identified_predictions.json: `{summary['identified_predictions_json']}`",
        f"- visualized: `{summary['visualized_dir']}`",
        f"- crops: `{summary['crops_dir']}`",
        "",
        "## Формулировка для ВКР",
        "",
        "Поскольку используемый датасет не содержит полноценной SKU-разметки, для проверки модуля идентификации автоматически формируется демонстрационная SKU-галерея на основе crop-изображений найденных объектов. Каждый выбранный crop рассматривается как условная эталонная товарная позиция, после чего найденные объекты сопоставляются с данной галереей и получают статус уверенного совпадения, неоднозначного совпадения или неопределённого объекта.",
    ]

    if summary.get("params"):
        lines.extend(["", "## Параметры запуска", ""])
        for key, value in summary["params"].items():
            label = PARAM_LABELS_RU.get(key, key)
            lines.append(f"- {label}: `{value}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"experiment_summary_json": json_path, "experiment_summary_csv": csv_path, "experiment_summary_md": md_path}
