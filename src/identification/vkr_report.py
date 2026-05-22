from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


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
        return "Таблица анализа порога similarity не была сформирована."

    candidates = threshold_df[(threshold_df["matched_rate"] >= 0.90) & (threshold_df["unknown"] > 0)]
    if candidates.empty:
        row = threshold_df.iloc[(threshold_df["threshold"] - 0.65).abs().argsort()[:1]].iloc[0]
    else:
        row = candidates.sort_values(["unknown", "threshold"]).iloc[0]

    return (
        f"По результатам анализа в качестве рабочего может использоваться порог similarity {float(row['threshold']):.2f}: "
        f"при данном значении доля сопоставленных объектов составляет {_fmt_percent(row['matched_rate'])}, "
        f"а доля объектов со статусом unknown — {_fmt_percent(row['unknown_rate'])}."
    )


def generate_vkr_experiment_report(out_dir: str | Path) -> Dict[str, Path]:
    """Generate a ready-to-insert VKR experiment section from pipeline artifacts."""

    out_dir = Path(out_dir)
    reports_dir = out_dir / "05_reports"
    demo_dir = out_dir / "02_demo_gallery"
    identification_dir = out_dir / "04_identification"
    reports_dir.mkdir(parents=True, exist_ok=True)

    full_summary = _read_json(reports_dir / "full_experiment_summary.json")
    existing_summary = _read_json(reports_dir / "existing_identification_summary.json")
    summary = full_summary or existing_summary
    demo_summary = _read_json(demo_dir / "demo_sku_gallery_summary.json")
    threshold_df = _read_threshold_table(reports_dir / "threshold_analysis.csv")

    path_md = reports_dir / "vkr_experiment_section.md"
    path_txt = reports_dir / "vkr_experiment_section.txt"

    gallery_images = int(summary.get("gallery_images_count", 0) or 0)
    query_images = int(summary.get("query_images_count", 0) or 0)
    query_objects = int(summary.get("query_objects_count", summary.get("total_objects", 0)) or 0)
    matched = int(summary.get("matched", 0) or 0)
    unknown = int(summary.get("unknown", 0) or 0)
    created_sku = int(summary.get("created_demo_sku_count", demo_summary.get("created_sku_count", 0)) or 0)
    extracted_crops = int(summary.get("extracted_gallery_crops_count", demo_summary.get("extracted_crops_count", 0)) or 0)
    gallery_refs = int(demo_summary.get("gallery_refs_count", 0) or 0)
    duplicate_refs = int(demo_summary.get("duplicate_refs_count", 0) or 0)
    deduplicate = demo_summary.get("deduplicate", None)
    threshold = summary.get("params", {}).get("threshold", summary.get("threshold", 0.65)) if isinstance(summary.get("params"), dict) else summary.get("threshold", 0.65)
    shuffle = summary.get("params", {}).get("shuffle", False) if isinstance(summary.get("params"), dict) else False
    seed = summary.get("params", {}).get("seed", 42) if isinstance(summary.get("params"), dict) else 42

    lines = [
        "# Раздел для ВКР: экспериментальная проверка модуля идентификации",
        "",
        "## Цель эксперимента",
        "",
        "Целью экспериментальной проверки являлась оценка работоспособности разработанного программного модуля идентификации товарных позиций на изображениях полочного пространства. В рамках эксперимента проверялся полный контур обработки: обнаружение товарных объектов, формирование демонстрационной SKU-галереи и сопоставление найденных query-объектов с эталонными изображениями галереи.",
        "",
        "## Методика проведения эксперимента",
        "",
    ]

    if shuffle:
        lines.append(f"Для эксперимента была сформирована воспроизводимая случайная выборка изображений при фиксированном seed = {seed}.")
    else:
        lines.append("Для эксперимента была использована ограниченная выборка изображений из исходного набора данных.")

    lines.extend(
        [
            f"Исходная выборка была разделена на две части: {gallery_images} изображений использовались для автоматического формирования демонстрационной SKU-галереи, а {query_images} изображений применялись как query-часть для независимой идентификации.",
            "",
            "Демонстрационная SKU-галерея формировалась автоматически на основе crop-изображений объектов, найденных детектором в gallery-части. Для уменьшения количества дубликатов в галерее применялось объединение визуально похожих crop-изображений в один условный demo SKU.",
            "",
            "## Результаты формирования demo SKU-галереи",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Извлечено crop-изображений из gallery-части | {extracted_crops} |",
            f"| Создано уникальных demo SKU | {created_sku} |",
            f"| Эталонных изображений в галерее | {gallery_refs} |",
            f"| Повторных refs, добавленных к существующим SKU | {duplicate_refs} |",
            f"| Дедупликация gallery | {deduplicate} |",
            "",
            "## Результаты идентификации query-объектов",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Query-изображений | {query_images} |",
            f"| Найдено товарных объектов | {query_objects} |",
            f"| Сопоставлено с demo SKU | {matched} |",
            f"| Получили статус unknown | {unknown} |",
            f"| Доля сопоставленных объектов | {_fmt_float(summary.get('matched_rate', 0.0))} ({_fmt_percent(summary.get('matched_rate', 0.0))}) |",
            f"| Доля unknown | {_fmt_float(summary.get('unknown_rate', 0.0))} ({_fmt_percent(summary.get('unknown_rate', 0.0))}) |",
            f"| Средняя similarity | {_fmt_float(summary.get('avg_similarity', 0.0))} |",
            f"| Рабочий threshold | {_fmt_float(threshold, 2)} |",
            "",
            "## Анализ порога similarity",
            "",
        ]
    )

    if not threshold_df.empty:
        lines.append(threshold_df.to_markdown(index=False))
        lines.append("")
    lines.append(_best_threshold_line(threshold_df))
    lines.extend(
        [
            "",
            "График влияния порога similarity на долю matched/unknown объектов сохраняется в файле `threshold_analysis_plot.png` и может быть использован в презентации и тексте ВКР.",
            "",
            "## Интерпретация результатов",
            "",
            "Полученные результаты подтверждают работоспособность разработанного модуля идентификации. Система выполняет обнаружение товарных объектов на изображениях, формирует демонстрационную SKU-галерею и сопоставляет найденные query-объекты с эталонными изображениями. Наличие объектов со статусом unknown показывает, что модуль не только присваивает ближайший SKU, но и способен отклонять недостаточно уверенные совпадения при заданном пороге similarity.",
            "",
            "## Ограничения эксперимента",
            "",
            "В используемом датасете отсутствует полноценная разметка объектов по реальным SKU-классам. Поэтому показатель доли сопоставленных объектов не следует интерпретировать как accuracy классификации по настоящим артикулам. В данной работе он отражает долю объектов, которые были сопоставлены с автоматически сформированной демонстрационной SKU-галереей. Для оценки точности по реальным SKU требуется отдельная эталонная галерея и разметка query-объектов по истинным артикулам.",
            "",
            "## Файлы, которые можно использовать в ВКР",
            "",
            f"- Сводка эксперимента: `{reports_dir / 'full_experiment_summary.md'}`",
            f"- Таблица threshold analysis: `{reports_dir / 'threshold_analysis.csv'}`",
            f"- График threshold analysis: `{reports_dir / 'threshold_analysis_plot.png'}`",
            f"- Отчёт demo gallery: `{demo_dir / 'demo_sku_gallery_report.md'}`",
            f"- Таблица результатов идентификации: `{identification_dir / 'identification_results.csv'}`",
            f"- Примеры визуализаций: `{identification_dir / 'visualized'}`",
        ]
    )

    text = "\n".join(lines)
    path_md.write_text(text, encoding="utf-8")
    path_txt.write_text(text, encoding="utf-8")
    return {"vkr_experiment_section_md": path_md, "vkr_experiment_section_txt": path_txt}
