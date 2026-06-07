from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


MODE_LABELS_RU = {
    "mask_segmentation": "сегментационный режим с масками",
    "bbox_fallback": "резервный режим по ограничивающим прямоугольникам",
    "not_found": "данные не найдены",
}

SOURCE_TYPE_LABELS_RU = {
    "mask": "crop по маске",
    "bbox": "crop по ограничивающему прямоугольнику",
    "unknown": "не указан",
}


def _mode_label(mode: str | None) -> str:
    value = str(mode or "")
    return MODE_LABELS_RU.get(value, value or "не указан")


def _source_type_label(source_type: str | None) -> str:
    value = str(source_type or "unknown")
    return SOURCE_TYPE_LABELS_RU.get(value, value)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _as_records(raw: Any) -> List[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ["predictions", "items", "images", "data"]:
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _mask_present(mask: Any) -> bool:
    if mask is None:
        return False
    if isinstance(mask, str):
        return bool(mask.strip())
    if isinstance(mask, dict):
        return bool(mask)
    if isinstance(mask, (list, tuple)):
        return len(mask) > 0
    return True


def _iter_detection_rows(predictions_json: Path) -> Iterable[dict]:
    raw = _read_json(predictions_json)
    records = _as_records(raw)

    for record in records:
        image_path = str(record.get("image_path") or record.get("path") or "")
        image_name = str(record.get("image_name") or Path(image_path).name)
        model_name = str(record.get("model_name") or record.get("model") or "")

        detections = record.get("detections")
        if isinstance(detections, list):
            for idx, det in enumerate(detections, start=1):
                if not isinstance(det, dict):
                    continue
                mask = det.get("mask")
                score = det.get("score", det.get("confidence", 0.0))
                yield {
                    "image_path": image_path,
                    "image_name": image_name,
                    "model_name": model_name,
                    "object_id": int(det.get("object_id", idx) or idx),
                    "score": float(score or 0.0),
                    "has_mask": _mask_present(mask),
                    "source_format": "detections",
                }
            continue

        boxes = record.get("boxes") or []
        scores = record.get("scores") or []
        masks = record.get("masks") or []
        if isinstance(boxes, list):
            for idx, _box in enumerate(boxes, start=1):
                score = scores[idx - 1] if idx - 1 < len(scores) else 0.0
                mask = masks[idx - 1] if idx - 1 < len(masks) else None
                yield {
                    "image_path": image_path,
                    "image_name": image_name,
                    "model_name": model_name,
                    "object_id": idx,
                    "score": float(score or 0.0),
                    "has_mask": _mask_present(mask),
                    "source_format": "flat",
                }


def _summarize_predictions(predictions_json: Path) -> Dict[str, Any]:
    rows = list(_iter_detection_rows(predictions_json))
    if not rows:
        return {
            "predictions_json": str(predictions_json),
            "images_count": 0,
            "segmented_or_detected_objects_count": 0,
            "objects_with_masks_count": 0,
            "objects_without_masks_count": 0,
            "mask_rate": 0.0,
            "average_detection_confidence": 0.0,
            "model_names": [],
            "mode": "not_found",
        }

    df = pd.DataFrame(rows)
    objects_count = len(df)
    mask_count = int(df["has_mask"].sum())
    no_mask_count = int(objects_count - mask_count)
    mode = "mask_segmentation" if mask_count > 0 else "bbox_fallback"

    return {
        "predictions_json": str(predictions_json),
        "images_count": int(df["image_name"].nunique()),
        "segmented_or_detected_objects_count": int(objects_count),
        "objects_with_masks_count": mask_count,
        "objects_without_masks_count": no_mask_count,
        "mask_rate": mask_count / max(1, objects_count),
        "average_detection_confidence": float(df["score"].mean()),
        "model_names": sorted([str(x) for x in df["model_name"].dropna().unique().tolist() if str(x)]),
        "mode": mode,
    }


def _summarize_identification(identification_csv: Path) -> Dict[str, Any]:
    df = _read_csv(identification_csv)
    if df.empty:
        return {
            "identification_csv": str(identification_csv),
            "objects_count": 0,
            "matched": 0,
            "matched_uncertain": 0,
            "unknown": 0,
            "matched_rate": 0.0,
            "matched_uncertain_rate": 0.0,
            "unknown_rate": 0.0,
            "safe_sku_count": 0,
            "safe_sku_rate": 0.0,
            "mean_sku_confidence": 0.0,
            "mean_distinct_margin": 0.0,
            "crop_source_type_counts": {},
        }

    total = len(df)
    status = df["sku_status"].fillna("unknown").astype(str) if "sku_status" in df.columns else pd.Series(["unknown"] * total)
    matched = int((status == "matched").sum())
    matched_uncertain = int((status == "matched_uncertain").sum())
    unknown = int((status == "unknown").sum())

    safe_sku_count = 0
    if "safe_sku_id" in df.columns:
        safe_sku = df["safe_sku_id"].fillna("").astype(str).str.strip()
        safe_sku_count = int((safe_sku != "").sum())

    mean_conf = 0.0
    if "sku_confidence" in df.columns:
        mean_conf = float(pd.to_numeric(df["sku_confidence"], errors="coerce").mean())

    mean_margin = 0.0
    if "distinct_margin" in df.columns:
        mean_margin = float(pd.to_numeric(df["distinct_margin"], errors="coerce").mean())

    source_counts: Dict[str, int] = {}
    if "source_type" in df.columns:
        source_counts = {
            str(key): int(value)
            for key, value in df["source_type"].fillna("unknown").astype(str).value_counts().to_dict().items()
        }

    return {
        "identification_csv": str(identification_csv),
        "objects_count": int(total),
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "matched_rate": matched / max(1, total),
        "matched_uncertain_rate": matched_uncertain / max(1, total),
        "unknown_rate": unknown / max(1, total),
        "safe_sku_count": safe_sku_count,
        "safe_sku_rate": safe_sku_count / max(1, total),
        "mean_sku_confidence": mean_conf,
        "mean_distinct_margin": mean_margin,
        "crop_source_type_counts": source_counts,
    }


def _find_first_existing(root: Path, names: List[str]) -> Path | None:
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _summarize_segmentation_metrics(out_dir: Path) -> Dict[str, Any]:
    path = _find_first_existing(
        out_dir,
        [
            "segmentation_metrics_summary.csv",
            "mask_ap_by_threshold.csv",
            "segmentation_metrics.json",
        ],
    )
    if path is None:
        return {"available": False, "path": "", "metrics": {}}

    if path.suffix.lower() == ".json":
        raw = _read_json(path)
        return {"available": raw is not None, "path": str(path), "metrics": raw if isinstance(raw, dict) else {}}

    df = _read_csv(path)
    if df.empty:
        return {"available": False, "path": str(path), "metrics": {}}

    first = df.iloc[0].to_dict()
    metrics = {}
    for key, value in first.items():
        try:
            metrics[str(key)] = float(value)
        except Exception:
            metrics[str(key)] = str(value)
    return {"available": True, "path": str(path), "metrics": metrics}


def _write_markdown_report(
    path: Path,
    segmentation: Dict[str, Any],
    identification: Dict[str, Any],
    mask_metrics: Dict[str, Any],
) -> None:
    mode = str(segmentation.get("mode", "unknown"))
    if mode == "mask_segmentation":
        mode_text = (
            "В предсказаниях обнаружены маски объектов. Это соответствует сегментационному режиму: "
            "товарные области выделяются не только рамками, но и контурами или масками."
        )
    elif mode == "bbox_fallback":
        mode_text = (
            "В предсказаниях маски не обнаружены. Используется резервный режим по ограничивающим прямоугольникам: "
            "товарные области выделяются прямоугольными рамками. Такой режим сохраняет работоспособность идентификации, "
            "но хуже соответствует строгой постановке сегментации."
        )
    else:
        mode_text = "Файл предсказаний не найден или не содержит объектов."

    lines = [
        "# ShelfVision: отчёт по связке сегментации и идентификации продукции",
        "",
        "## Назначение отчёта",
        "",
        "Отчёт показывает единый путь обработки изображения стеллажа: сначала система выделяет товарные области, затем извлекает объекты и выполняет идентификацию по SKU-галерее.",
        "",
        "Такой отчёт нужен для явного соответствия теме ВКР: `Система для семантической сегментации и идентификации продукции на стеллажах розничных магазинов`.",
        "",
        "## 1. Сегментация / выделение товарных объектов",
        "",
        f"- Файл предсказаний: `{segmentation.get('predictions_json', '')}`",
        f"- Режим: `{_mode_label(mode)}`",
        f"- Изображений: `{segmentation.get('images_count', 0)}`",
        f"- Выделено товарных объектов: `{segmentation.get('segmented_or_detected_objects_count', 0)}`",
        f"- Объектов с масками: `{segmentation.get('objects_with_masks_count', 0)}`",
        f"- Объектов без масок: `{segmentation.get('objects_without_masks_count', 0)}`",
        f"- Доля объектов с масками: `{float(segmentation.get('mask_rate', 0.0)):.4f}`",
        f"- Средняя confidence локализации: `{float(segmentation.get('average_detection_confidence', 0.0)):.4f}`",
        f"- Модели: `{', '.join(segmentation.get('model_names', []) or [])}`",
        "",
        mode_text,
        "",
        "## 2. Качество масок",
        "",
    ]

    if mask_metrics.get("available"):
        lines.extend([f"- Файл метрик масок: `{mask_metrics.get('path', '')}`", "", "| Метрика | Значение |", "|---|---:|"])
        for key, value in (mask_metrics.get("metrics") or {}).items():
            lines.append(f"| {key} | `{value}` |")
    else:
        lines.append(
            "Отдельные метрики масок в текущей папке результата не найдены. "
            "Для строгой оценки сегментации нужен запуск на наборе с COCO-разметкой сегментации."
        )

    lines.extend(
        [
            "",
            "## 3. Идентификация выделенных объектов",
            "",
            f"- Файл идентификации: `{identification.get('identification_csv', '')}`",
            f"- Всего объектов в идентификации: `{identification.get('objects_count', 0)}`",
            f"- Уверенные совпадения: `{identification.get('matched', 0)}`",
            f"- Неоднозначные совпадения: `{identification.get('matched_uncertain', 0)}`",
            f"- Неопределённые объекты: `{identification.get('unknown', 0)}`",
            f"- Доля уверенных совпадений: `{float(identification.get('matched_rate', 0.0)):.4f}`",
            f"- Доля неоднозначных совпадений: `{float(identification.get('matched_uncertain_rate', 0.0)):.4f}`",
            f"- Доля неопределённых объектов: `{float(identification.get('unknown_rate', 0.0)):.4f}`",
            f"- Безопасных назначений SKU: `{identification.get('safe_sku_count', 0)}`",
            f"- Доля безопасных назначений SKU: `{float(identification.get('safe_sku_rate', 0.0)):.4f}`",
            f"- Среднее визуальное сходство: `{float(identification.get('mean_sku_confidence', 0.0)):.4f}`",
            f"- Средний отрыв между двумя лучшими SKU: `{float(identification.get('mean_distinct_margin', 0.0)):.4f}`",
            "",
            "### Источник crop-объектов",
            "",
            "| Источник crop-объекта | Количество |",
            "|---|---:|",
        ]
    )

    source_counts = identification.get("crop_source_type_counts") or {}
    if source_counts:
        for key, value in source_counts.items():
            lines.append(f"| {_source_type_label(str(key))} | {value} |")
    else:
        lines.append("| нет данных | 0 |")

    lines.extend(
        [
            "",
            "## 4. Интерпретация для ВКР",
            "",
            "Разработанная система выполняет два связанных этапа. На первом этапе выделяются товарные области на изображениях стеллажей. Если используется YOLO-Seg, объект представлен маской; если маски отсутствуют, применяется резервный режим по ограничивающим прямоугольникам. На втором этапе каждый выделенный объект преобразуется в crop и сопоставляется с SKU-галереей.",
            "",
            "Для повышения надёжности идентификации используется статус `matched_uncertain`. Он означает, что лучший и второй лучший различные SKU имеют слишком близкие значения визуального сходства. В таких случаях система сохраняет диагностическую информацию, но не считает назначение безопасным SKU.",
            "",
            "В результате отчёт подтверждает соответствие программной реализации теме ВКР: сегментация или выделение продукции используется как входной этап для последующей идентификации продукции.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_segmentation_identification_report(
    out_dir: str | Path,
    query_predictions_json: str | Path | None = None,
    identification_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    query_predictions_json = Path(query_predictions_json) if query_predictions_json is not None else out_dir / "03_query_inference" / "predictions.json"
    identification_dir = Path(identification_dir) if identification_dir is not None else out_dir / "04_identification"
    reports_dir = Path(reports_dir) if reports_dir is not None else out_dir / "05_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    identification_csv = identification_dir / "identification_results.csv"
    segmentation_summary = _summarize_predictions(query_predictions_json)
    identification_summary = _summarize_identification(identification_csv)
    mask_metrics_summary = _summarize_segmentation_metrics(out_dir)

    summary = {
        "out_dir": str(out_dir),
        "segmentation_or_detection": segmentation_summary,
        "identification": identification_summary,
        "mask_metrics": mask_metrics_summary,
    }

    json_path = reports_dir / "segmentation_identification_summary.json"
    md_path = reports_dir / "segmentation_identification_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, segmentation_summary, identification_summary, mask_metrics_summary)

    return {
        "segmentation_identification_summary_json": json_path,
        "segmentation_identification_report_md": md_path,
    }
