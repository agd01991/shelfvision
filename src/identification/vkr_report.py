from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


THRESHOLD_COLUMNS_RU = {
    "threshold": "Порог идентификации",
    "total_objects": "Всего объектов",
    "matched": "Уверенные совпадения",
    "unknown": "Неопределённые объекты",
    "matched_rate": "Доля уверенных совпадений",
    "unknown_rate": "Доля неопределённых объектов",
    "avg_similarity_all": "Среднее сходство по всем объектам",
    "avg_similarity_matched": "Среднее сходство уверенных совпадений",
    "min_similarity_matched": "Минимальное сходство среди уверенных совпадений",
    "max_similarity_unknown": "Максимальное сходство среди неопределённых объектов",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_threshold_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _threshold_table_for_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.rename(columns=THRESHOLD_COLUMNS_RU)


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def _fmt_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def _best_threshold_line(threshold_df: pd.DataFrame) -> str:
    if threshold_df.empty:
        return "Таблица анализа порога визуального сходства не была сформирована."

    candidates = threshold_df[(threshold_df["matched_rate"] >= 0.90) & (threshold_df["unknown"] > 0)]
    if candidates.empty:
        row = threshold_df.iloc[(threshold_df["threshold"] - 0.65).abs().argsort()[:1]].iloc[0]
    else:
        row = candidates.sort_values(["unknown", "threshold"]).iloc[0]

    return (
        f"По результатам анализа в качестве рабочего может использоваться порог визуального сходства {float(row['threshold']):.2f}: "
        f"при данном значении доля уверенных совпадений составляет {_fmt_percent(row['matched_rate'])}, "
        f"а доля неопределённых объектов — {_fmt_percent(row['unknown_rate'])}."
    )


def _bool_text(value: Any) -> str:
    return "да" if bool(value) else "нет"


def generate_vkr_experiment_report(out_dir: str | Path) -> Dict[str, Path]:
    """Сформировать готовый раздел ВКР по экспериментальной проверке."""

    out_dir = Path(out_dir)
    reports_dir = out_dir / "05_reports"
    demo_dir = out_dir / "02_demo_gallery"
    identification_dir = out_dir / "04_identification"
    reports_dir.mkdir(parents=True, exist_ok=True)

    full_summary = _read_json(reports_dir / "full_experiment_summary.json")
    existing_summary = _read_json(reports_dir / "existing_identification_summary.json")
    summary = full_summary or existing_summary
    params = summary.get("params", {}) if isinstance(summary.get("params"), dict) else {}
    demo_summary = _read_json(demo_dir / "demo_sku_gallery_summary.json")
    segmentation_summary = _read_json(reports_dir / "segmentation_identification_summary.json")
    threshold_df = _read_threshold_table(reports_dir / "threshold_analysis.csv")

    path_md = reports_dir / "vkr_experiment_section.md"
    path_txt = reports_dir / "vkr_experiment_section.txt"

    gallery_images = int(summary.get("gallery_images_count", 0) or 0)
    query_images = int(summary.get("query_images_count", 0) or 0)
    query_objects = int(summary.get("query_objects_count", summary.get("total_objects", 0)) or 0)
    matched = int(summary.get("matched", 0) or 0)
    matched_uncertain = int(summary.get("matched_uncertain", 0) or 0)
    unknown = int(summary.get("unknown", 0) or 0)
    assigned = int(summary.get("assigned", matched + matched_uncertain) or 0)
    created_sku = int(summary.get("created_demo_sku_count", demo_summary.get("created_sku_count", 0)) or 0)
    extracted_crops = int(summary.get("extracted_gallery_crops_count", demo_summary.get("extracted_crops_count", 0)) or 0)
    gallery_refs = int(demo_summary.get("gallery_refs_count", 0) or 0)
    duplicate_refs = int(demo_summary.get("duplicate_refs_count", 0) or 0)
    deduplicate = demo_summary.get("deduplicate", None)
    threshold = params.get("threshold", summary.get("threshold", 0.65))
    ambiguity_margin = params.get("ambiguity_margin", summary.get("ambiguity_margin", 0.03))
    enable_uncertain_status = params.get("enable_uncertain_status", summary.get("enable_uncertain_status", False))
    shuffle = params.get("shuffle", False)
    seed = params.get("seed", 42)

    segmentation_block = segmentation_summary.get("segmentation_or_detection", {}) if isinstance(segmentation_summary, dict) else {}
    segmented_objects = int(segmentation_block.get("segmented_or_detected_objects_count", 0) or 0)
    objects_with_masks = int(segmentation_block.get("objects_with_masks_count", 0) or 0)
    mask_rate = float(segmentation_block.get("mask_rate", 0.0) or 0.0)
    segmentation_mode = str(segmentation_block.get("mode", ""))

    lines = [
        "# Раздел для ВКР: экспериментальная проверка сегментации и идентификации продукции",
        "",
        "## Цель эксперимента",
        "",
        "Целью экспериментальной проверки являлась оценка работоспособности разработанного программного комплекса ShelfVision, выполняющего выделение товарных областей на изображениях стеллажей и последующую идентификацию выделенных объектов по SKU-галерее. В рамках эксперимента проверялся полный контур обработки: инференс модели, извлечение crop-объектов, формирование демонстрационной SKU-галереи, сопоставление query-объектов с эталонами и анализ неоднозначных назначений.",
        "",
        "## Методика проведения эксперимента",
        "",
    ]

    if shuffle:
        lines.append(f"Для эксперимента была сформирована воспроизводимая случайная выборка изображений при фиксированном seed = {seed}.")
    else:
        lines.append("Для эксперимента была использована выборка изображений из исходного набора данных без дополнительного перемешивания.")

    lines.extend(
        [
            f"Исходная выборка была разделена на две части: {gallery_images} изображений использовались для автоматического формирования демонстрационной SKU-галереи, а {query_images} изображений применялись как query-часть для независимой идентификации.",
            "",
            "Демонстрационная SKU-галерея формировалась автоматически на основе crop-изображений объектов, найденных моделью в gallery-части. Для уменьшения количества дубликатов в галерее применялось объединение визуально похожих crop-изображений в один условный demo SKU.",
            "",
            "## Результаты выделения товарных объектов",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Режим выделения объектов | {segmentation_mode or 'не указан'} |",
            f"| Выделено товарных объектов в query-предсказаниях | {segmented_objects} |",
            f"| Объектов с масками | {objects_with_masks} |",
            f"| Доля объектов с масками | {_fmt_float(mask_rate)} ({_fmt_percent(mask_rate)}) |",
            "",
            "## Результаты формирования demo SKU-галереи",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Извлечено crop-изображений из gallery-части | {extracted_crops} |",
            f"| Создано уникальных demo SKU | {created_sku} |",
            f"| Эталонных изображений в галерее | {gallery_refs} |",
            f"| Повторных эталонов, добавленных к существующим SKU | {duplicate_refs} |",
            f"| Дедупликация галереи | {_bool_text(deduplicate)} |",
            "",
            "## Результаты идентификации query-объектов",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Query-изображений | {query_images} |",
            f"| Найдено товарных объектов | {query_objects} |",
            f"| Уверенные совпадения | {matched} |",
            f"| Неоднозначные совпадения | {matched_uncertain} |",
            f"| Неопределённые объекты | {unknown} |",
            f"| Всего назначений SKU | {assigned} |",
            f"| Доля уверенных совпадений | {_fmt_float(summary.get('matched_rate', 0.0))} ({_fmt_percent(summary.get('matched_rate', 0.0))}) |",
            f"| Доля неоднозначных совпадений | {_fmt_float(summary.get('matched_uncertain_rate', 0.0))} ({_fmt_percent(summary.get('matched_uncertain_rate', 0.0))}) |",
            f"| Доля неопределённых объектов | {_fmt_float(summary.get('unknown_rate', 0.0))} ({_fmt_percent(summary.get('unknown_rate', 0.0))}) |",
            f"| Среднее визуальное сходство | {_fmt_float(summary.get('avg_similarity', 0.0))} |",
            f"| Средний отрыв между двумя лучшими SKU | {_fmt_float(summary.get('mean_distinct_margin', 0.0))} |",
            f"| Рабочий порог идентификации | {_fmt_float(threshold, 2)} |",
            f"| Минимальный отрыв для уверенного назначения | {_fmt_float(ambiguity_margin, 3)} |",
            f"| Анализ неоднозначных совпадений | {_bool_text(enable_uncertain_status)} |",
            "",
            "## Анализ порога визуального сходства",
            "",
        ]
    )

    if not threshold_df.empty:
        lines.append(_threshold_table_for_report(threshold_df).to_markdown(index=False))
        lines.append("")
    lines.append(_best_threshold_line(threshold_df))
    lines.extend(
        [
            "",
            "График влияния порога визуального сходства на долю уверенных и неопределённых объектов сохраняется в файле `threshold_analysis_plot.png` и может быть использован в презентации и тексте ВКР.",
            "",
            "## Интерпретация результатов",
            "",
            "Полученные результаты подтверждают работоспособность разработанного программного комплекса. Система выделяет товарные области на изображениях, формирует демонстрационную SKU-галерею и сопоставляет найденные query-объекты с эталонными изображениями. Наличие неопределённых и неоднозначных объектов показывает, что модуль не только присваивает ближайший SKU, но и отделяет недостаточно уверенные совпадения от надёжных назначений.",
            "",
            "## Ограничения эксперимента",
            "",
            "В используемом датасете отсутствует полноценная разметка объектов по реальным SKU-классам. Поэтому показатель доли уверенных совпадений не следует интерпретировать как accuracy классификации по настоящим артикулам. В данной работе он отражает долю объектов, которые были сопоставлены с автоматически сформированной демонстрационной SKU-галереей. Для оценки точности по реальным SKU требуется отдельная эталонная галерея и разметка query-объектов по истинным артикулам.",
            "",
            "## Файлы, которые можно использовать в ВКР",
            "",
            f"- Сводка эксперимента: `{reports_dir / 'full_experiment_summary.md'}`",
            f"- Отчёт по связке сегментации и идентификации: `{reports_dir / 'segmentation_identification_report.md'}`",
            f"- Таблица анализа порогов: `{reports_dir / 'threshold_analysis.csv'}`",
            f"- График анализа порогов: `{reports_dir / 'threshold_analysis_plot.png'}`",
            f"- Отчёт demo-галереи: `{demo_dir / 'demo_sku_gallery_report.md'}`",
            f"- Отчёт аудита неоднозначности: `{identification_dir / 'assignment_uncertainty_report.md'}`",
            f"- Таблица результатов идентификации: `{identification_dir / 'identification_results.csv'}`",
            f"- Примеры визуализаций: `{identification_dir / 'visualized'}`",
        ]
    )

    text = "\n".join(lines)
    path_md.write_text(text, encoding="utf-8")
    path_txt.write_text(text, encoding="utf-8")
    return {"vkr_experiment_section_md": path_md, "vkr_experiment_section_txt": path_txt}
