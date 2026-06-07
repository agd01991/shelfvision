from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from control_panel_wsl import python_command
from panel_progress import CommandStep, run_steps_with_progress
from path_utils import to_current_os_path
from src.identification.manual_gallery_editor import (
    ManualGalleryEdit,
    append_manual_edit,
    build_manual_gallery_from_edits,
    infer_gallery_dir_from_experiment,
    list_sku_refs,
    read_manual_edits,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_path(raw: str | Path | None) -> Path:
    return to_current_os_path(raw)


def _read_csv(path: Path) -> pd.DataFrame:
    path = _safe_path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    path = _safe_path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 30_000) -> str:
    path = _safe_path(path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...текст сокращён для отображения..."
    return text


def _write_text_preview(path: Path) -> None:
    text = _read_text(path)
    if text:
        st.markdown(text)
    else:
        st.info(f"Файл не найден: `{_safe_path(path)}`")


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _select_experiment_from_summary(results_root: Path, summary_csv: Path) -> Path | None:
    df = _read_csv(summary_csv)
    if df.empty or not {"experiment", "out_dir"}.issubset(df.columns):
        return None
    cluster_df = df[df.get("gallery_build_mode", "").astype(str).eq("cluster")].copy() if "gallery_build_mode" in df.columns else df
    if cluster_df.empty:
        cluster_df = df
    options = [str(value) for value in cluster_df["experiment"].tolist()]
    selected = st.selectbox("Эксперимент", options, key="manual_editor_experiment")
    row = cluster_df[cluster_df["experiment"].astype(str).eq(selected)].iloc[0]
    return _safe_path(str(row.get("out_dir") or results_root / selected))


def _render_refs_grid(refs: List[Path], key_prefix: str, selectable: bool = True) -> List[str]:
    selected: List[str] = []
    if not refs:
        st.info("В выбранном SKU нет эталонных изображений.")
        return selected

    cols = st.columns(4)
    for index, ref in enumerate(refs):
        with cols[index % 4]:
            if ref.exists():
                st.image(str(ref), caption=ref.name, use_container_width=True)
            else:
                st.caption(ref.name)
            if selectable:
                checked = st.checkbox("выбрать", key=f"{key_prefix}_{ref.name}_{index}")
                if checked:
                    selected.append(ref.name)
    return selected


def _render_current_edits(edits_csv: Path) -> None:
    edits = read_manual_edits(_safe_path(edits_csv))
    if not edits:
        st.info(f"Ручных операций пока нет: `{_safe_path(edits_csv)}`")
        return
    df = pd.DataFrame([edit.__dict__ for edit in edits])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_merge_tab(source_gallery_dir: Path, edits_csv: Path) -> None:
    sku_refs = list_sku_refs(_safe_path(source_gallery_dir))
    sku_ids = sorted(sku_refs.keys())
    if len(sku_ids) < 2:
        st.info("Для объединения нужно минимум два SKU.")
        return

    c1, c2 = st.columns(2)
    with c1:
        source_sku = st.selectbox("SKU-источник: откуда забрать эталоны", sku_ids, key="manual_merge_source")
    with c2:
        target_options = [sku for sku in sku_ids if sku != source_sku]
        target_sku = st.selectbox("SKU-приёмник: куда объединить", target_options, key="manual_merge_target")

    st.caption("SKU-источник будет объединён в SKU-приёмник при сборке ручной галереи. Исходная галерея не изменяется.")
    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.markdown(f"#### SKU-источник `{source_sku}`")
        _render_refs_grid(sku_refs.get(source_sku, []), key_prefix="merge_source_preview", selectable=False)
    with preview_cols[1]:
        st.markdown(f"#### SKU-приёмник `{target_sku}`")
        _render_refs_grid(sku_refs.get(target_sku, []), key_prefix="merge_target_preview", selectable=False)

    comment = st.text_input("Комментарий к объединению", value="", placeholder="например: один и тот же товар", key="manual_merge_comment")
    if st.button("Добавить операцию объединения", use_container_width=True, key="manual_add_merge"):
        append_manual_edit(
            _safe_path(edits_csv),
            ManualGalleryEdit(
                operation="merge",
                source_sku_id=source_sku,
                target_sku_id=target_sku,
                comment=comment,
            ),
        )
        st.success(f"Операция объединения добавлена: {source_sku} -> {target_sku}")


def _render_split_tab(source_gallery_dir: Path, edits_csv: Path) -> None:
    sku_refs = list_sku_refs(_safe_path(source_gallery_dir))
    sku_ids = sorted(sku_refs.keys())
    if not sku_ids:
        st.info("SKU не найдены.")
        return

    source_sku = st.selectbox("SKU для разделения", sku_ids, key="manual_split_source")
    refs = sku_refs.get(source_sku, [])
    st.caption("Выбранные эталоны будут вынесены в новый SKU при сборке ручной галереи. Исходная галерея не изменяется.")
    selected_refs = _render_refs_grid(refs, key_prefix="manual_split_refs", selectable=True)

    c1, c2 = st.columns(2)
    with c1:
        new_sku_id = st.text_input("Новый SKU ID, необязательно", value="", placeholder="sku_demo_manual_001", key="manual_split_new_sku")
    with c2:
        comment = st.text_input("Комментарий к разделению", value="", placeholder="например: другой товар внутри кластера", key="manual_split_comment")

    if st.button("Добавить операцию разделения", use_container_width=True, key="manual_add_split"):
        if not selected_refs:
            st.warning("Выбери хотя бы один эталон для разделения.")
            return
        append_manual_edit(
            _safe_path(edits_csv),
            ManualGalleryEdit(
                operation="split",
                source_sku_id=source_sku,
                new_sku_id=new_sku_id.strip(),
                ref_files=";".join(selected_refs),
                comment=comment,
            ),
        )
        st.success(f"Операция разделения добавлена: {source_sku}, эталоны: {', '.join(selected_refs)}")


def _manual_paths(experiment_dir: Path) -> tuple[Path, Path, Path, Path]:
    experiment_dir = _safe_path(experiment_dir)
    manual_root = experiment_dir / "06_manual_gallery"
    edits_csv = manual_root / "manual_cluster_edits.csv"
    manual_gallery_dir = manual_root / "sku_gallery_manual"
    manual_gallery_csv = manual_gallery_dir / "gallery.csv"
    return manual_root, edits_csv, manual_gallery_dir, manual_gallery_csv


def _build_rerun_args(experiment_dir: Path, manual_gallery_dir: Path, manual_gallery_csv: Path) -> List[str]:
    experiment_dir = _safe_path(experiment_dir)
    query_predictions_json = experiment_dir / "03_query_inference" / "predictions.json"
    manual_identification_dir = experiment_dir / "06_manual_gallery" / "manual_identification"
    return [
        "--out-dir",
        str(manual_identification_dir),
        "--query-predictions-json",
        str(query_predictions_json),
        "--gallery-dir",
        str(manual_gallery_dir),
        "--gallery-csv",
        str(manual_gallery_csv),
        "--threshold",
        "0.65",
        "--thresholds",
        "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        "--top-k",
        "3",
        "--visualize-limit",
        "60",
        "--progress-every",
        "25",
    ]


def _extract_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    return {
        "objects": _to_int(raw.get("query_objects_count", raw.get("total_objects", 0))),
        "matched": _to_int(raw.get("matched", 0)),
        "unknown": _to_int(raw.get("unknown", 0)),
        "matched_rate": _to_float(raw.get("matched_rate", 0.0)),
        "unknown_rate": _to_float(raw.get("unknown_rate", 0.0)),
        "avg_similarity": _to_float(raw.get("avg_similarity", 0.0)),
    }


def _find_original_summary(experiment_dir: Path) -> tuple[Path | None, Dict[str, Any]]:
    experiment_dir = _safe_path(experiment_dir)
    candidates = [
        experiment_dir / "05_reports" / "full_experiment_summary.json",
        experiment_dir / "05_reports" / "existing_identification_summary.json",
        experiment_dir / "04_identification" / "identification_metrics.json",
    ]
    for path in candidates:
        raw = _read_json(path)
        metrics = _extract_metrics(raw)
        if metrics:
            return path, metrics
    return None, {}


def _find_manual_summary(manual_root: Path) -> tuple[Path | None, Dict[str, Any]]:
    manual_root = _safe_path(manual_root)
    candidates = [
        manual_root / "manual_identification" / "05_reports" / "existing_identification_summary.json",
        manual_root / "manual_identification" / "04_identification" / "identification_metrics.json",
    ]
    for path in candidates:
        raw = _read_json(path)
        metrics = _extract_metrics(raw)
        if metrics:
            return path, metrics
    return None, {}


def _comparison_rows(original: Dict[str, Any], manual: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    labels = {
        "objects": "Объекты query",
        "matched": "Уверенные совпадения",
        "unknown": "Неопределённые объекты",
        "matched_rate": "Доля уверенных совпадений",
        "unknown_rate": "Доля неопределённых объектов",
        "avg_similarity": "Среднее визуальное сходство",
    }
    for key, label in labels.items():
        before = original.get(key, 0)
        after = manual.get(key, 0)
        rows.append(
            {
                "метрика": label,
                "исходная галерея": before,
                "ручная галерея": after,
                "изменение": after - before,
            }
        )
    return pd.DataFrame(rows)


def _write_before_after_report(experiment_dir: Path, manual_root: Path, original: Dict[str, Any], manual: Dict[str, Any]) -> Path:
    manual_root = _safe_path(manual_root)
    report_path = manual_root / "manual_vs_original_report.md"
    rows = _comparison_rows(original, manual)

    delta_matched_rate = manual.get("matched_rate", 0.0) - original.get("matched_rate", 0.0)
    delta_unknown_rate = manual.get("unknown_rate", 0.0) - original.get("unknown_rate", 0.0)
    delta_similarity = manual.get("avg_similarity", 0.0) - original.get("avg_similarity", 0.0)

    verdict = "ручная галерея улучшила долю уверенных совпадений" if delta_matched_rate > 0 else "ручная галерея не улучшила долю уверенных совпадений"
    lines = [
        "# ShelfVision: сравнение исходной и ручной SKU-галереи",
        "",
        f"- Папка эксперимента: `{_safe_path(experiment_dir)}`",
        f"- Папка ручной галереи: `{manual_root}`",
        "",
        "## Итог",
        "",
        f"- Вывод: **{verdict}**",
        f"- Изменение доли уверенных совпадений: `{delta_matched_rate:.4f}`",
        f"- Изменение доли неопределённых объектов: `{delta_unknown_rate:.4f}`",
        f"- Изменение среднего визуального сходства: `{delta_similarity:.4f}`",
        "",
        "## Метрики",
        "",
        "| Метрика | Исходная галерея | Ручная галерея | Изменение |",
        "|---|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        metric = row["метрика"]
        original_value = row["исходная галерея"]
        manual_value = row["ручная галерея"]
        delta = row["изменение"]
        if "доля" in str(metric).lower() or "сходство" in str(metric).lower():
            lines.append(f"| {metric} | {original_value:.4f} | {manual_value:.4f} | {delta:.4f} |")
        else:
            lines.append(f"| {metric} | {int(original_value)} | {int(manual_value)} | {int(delta)} |")
    lines.extend(
        [
            "",
            "## Формулировка для ВКР",
            "",
            "После автоматического формирования SKU-галереи была выполнена ручная экспертная корректировка кластеров. "
            "Для проверки влияния корректировки идентификация была пересчитана на той же query-части, но с использованием ручной галереи. "
            "Такой подход позволяет оценивать эффект операций объединения и разделения без повторного запуска детектора.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _render_before_after_tab(experiment_dir: Path) -> None:
    manual_root, _, _, _ = _manual_paths(experiment_dir)
    original_path, original_metrics = _find_original_summary(experiment_dir)
    manual_path, manual_metrics = _find_manual_summary(manual_root)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Исходная галерея")
        if original_metrics:
            st.metric("Доля уверенных совпадений", f"{original_metrics['matched_rate']:.4f}")
            st.caption(f"Источник: `{original_path}`")
            st.write(f"Доля неопределённых объектов: `{original_metrics['unknown_rate']:.4f}`")
            st.write(f"Среднее визуальное сходство: `{original_metrics['avg_similarity']:.4f}`")
            st.write(f"Уверенные совпадения: `{original_metrics['matched']}`")
            st.write(f"Неопределённые объекты: `{original_metrics['unknown']}`")
        else:
            st.info("Сводка исходной галереи не найдена.")
    with c2:
        st.markdown("#### Ручная галерея")
        if manual_metrics:
            st.metric("Доля уверенных совпадений", f"{manual_metrics['matched_rate']:.4f}")
            st.caption(f"Источник: `{manual_path}`")
            st.write(f"Доля неопределённых объектов: `{manual_metrics['unknown_rate']:.4f}`")
            st.write(f"Среднее визуальное сходство: `{manual_metrics['avg_similarity']:.4f}`")
            st.write(f"Уверенные совпадения: `{manual_metrics['matched']}`")
            st.write(f"Неопределённые объекты: `{manual_metrics['unknown']}`")
        else:
            st.info("Сводка ручной галереи пока не найдена. Сначала нажми `Пересчитать идентификацию с ручной галереей`.")

    if not original_metrics or not manual_metrics:
        return

    comparison = _comparison_rows(original_metrics, manual_metrics)
    st.markdown("#### Таблица до/после")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    delta = manual_metrics["matched_rate"] - original_metrics["matched_rate"]
    if delta > 0:
        st.success(f"Ручная галерея улучшила долю уверенных совпадений на `{delta:.4f}`.")
    elif delta < 0:
        st.warning(f"Ручная галерея ухудшила долю уверенных совпадений на `{abs(delta):.4f}`. Проверь операции объединения и разделения.")
    else:
        st.info("Доля уверенных совпадений не изменилась.")

    if st.button("Сформировать markdown-отчёт до/после", use_container_width=True, key="manual_write_before_after_report"):
        report_path = _write_before_after_report(experiment_dir, manual_root, original_metrics, manual_metrics)
        st.success(f"Отчёт сохранён: `{report_path}`")

    report_path = manual_root / "manual_vs_original_report.md"
    if report_path.exists():
        with st.expander("Отчёт сравнения исходной и ручной галереи", expanded=True):
            _write_text_preview(report_path)


def _render_apply_tab(config: Dict[str, Any], experiment_dir: Path, source_gallery_dir: Path, edits_csv: Path) -> None:
    manual_root, _, manual_gallery_dir, manual_gallery_csv = _manual_paths(experiment_dir)
    st.markdown("#### Сборка ручной галереи")
    st.write(f"Папка ручной галереи: `{manual_root}`")
    st.write(f"Изображения ручной галереи: `{manual_gallery_dir}`")
    st.write(f"CSV-файл ручной галереи: `{manual_gallery_csv}`")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Применить ручные правки и собрать ручную галерею", use_container_width=True, key="manual_apply_edits"):
            try:
                outputs = build_manual_gallery_from_edits(
                    source_gallery_dir=_safe_path(source_gallery_dir),
                    output_gallery_dir=manual_gallery_dir,
                    output_gallery_csv=manual_gallery_csv,
                    edits_csv=_safe_path(edits_csv),
                    out_dir=manual_root,
                )
                st.success("Ручная галерея собрана.")
                for name, path in outputs.items():
                    st.caption(f"{name}: `{_safe_path(path)}`")
            except Exception as exc:
                st.error(f"Ошибка сборки ручной галереи: {exc}")
    with c2:
        if st.button("Пересчитать идентификацию с ручной галереей", use_container_width=True, key="manual_rerun_identification"):
            if not manual_gallery_csv.exists():
                st.warning("Сначала собери ручную галерею.")
            else:
                cmd = python_command(config, "run_existing_photo_identification.py", _build_rerun_args(experiment_dir, manual_gallery_dir, manual_gallery_csv))
                run_steps_with_progress(
                    [
                        CommandStep(
                            title="Пересчёт идентификации с ручной галереей",
                            cmd=cmd,
                            cwd=ROOT,
                            description="Переиспользуется существующий query predictions.json, но галерея берётся из ручной версии.",
                            estimated_seconds=None,
                        )
                    ],
                    title="Пересчёт идентификации с ручной галереей",
                    success_message="Идентификация с ручной галереей пересчитана.",
                    failure_message="Ошибка пересчёта идентификации с ручной галереей",
                )

    report_path = manual_root / "manual_gallery_report.md"
    if report_path.exists():
        with st.expander("Отчёт по ручной галерее", expanded=True):
            _write_text_preview(report_path)
    summary_json = manual_root / "manual_gallery_summary.json"
    if summary_json.exists():
        with st.expander("Сводка ручной галереи", expanded=False):
            st.json(_read_json(summary_json))

    manual_items = manual_root / "manual_gallery_items.csv"
    if manual_items.exists():
        with st.expander("Элементы ручной галереи", expanded=False):
            st.dataframe(_read_csv(manual_items), use_container_width=True, hide_index=True)

    manual_identification = manual_root / "manual_identification"
    identification_csv = manual_identification / "04_identification" / "identification_results.csv"
    if identification_csv.exists():
        with st.expander("Результаты идентификации с ручной галереей", expanded=False):
            st.dataframe(_read_csv(identification_csv).head(500), use_container_width=True, hide_index=True)


def page_manual_cluster_editor(config: Dict[str, Any]) -> None:
    st.subheader("Ручное объединение и разделение SKU-кластеров")
    st.caption("Редактор создаёт отдельную ручную галерею и не изменяет исходные результаты эксперимента.")

    night = config.setdefault("night_experiments", {})
    default_results = str(night.get("out_dir") or night.get("results_root") or "D:/1Diplom/shelfvision_results/cluster_compare_sku110k_2026-05-29_23-35-48")
    results_root = _safe_path(st.text_input("Папка серии экспериментов", value=default_results, key="manual_editor_results_root"))
    summary_csv = _safe_path(st.text_input("Сводная таблица серии экспериментов, необязательно", value=str(night.get("summary_csv") or results_root / "night_experiments_summary.csv"), key="manual_editor_summary_csv"))

    experiment_dir = _select_experiment_from_summary(results_root, summary_csv)
    if experiment_dir is None:
        raw_experiment = st.text_input("Папка конкретного эксперимента", value="", key="manual_editor_experiment_dir")
        if not raw_experiment.strip():
            st.info("Укажи сводную таблицу серии экспериментов или папку конкретного эксперимента.")
            return
        experiment_dir = _safe_path(raw_experiment)

    st.caption(f"Папка эксперимента: `{experiment_dir}`")
    inferred_gallery = infer_gallery_dir_from_experiment(experiment_dir)
    gallery_default = str(inferred_gallery or experiment_dir / "02_demo_gallery")
    source_gallery_dir = _safe_path(st.text_input("Исходная SKU-галерея", value=gallery_default, key="manual_editor_source_gallery"))
    if not source_gallery_dir.exists():
        st.warning(f"Исходная галерея не найдена: `{source_gallery_dir}`")
        return

    manual_root, edits_csv, manual_gallery_dir, manual_gallery_csv = _manual_paths(experiment_dir)
    st.caption(f"Файл ручных операций: `{edits_csv}`")

    sku_refs = list_sku_refs(source_gallery_dir)
    c1, c2, c3 = st.columns(3)
    c1.metric("SKU в исходной галерее", len(sku_refs))
    c2.metric("Эталонов в исходной галерее", sum(len(refs) for refs in sku_refs.values()))
    c3.metric("Ручных операций", len(read_manual_edits(edits_csv)))

    tabs = st.tabs(["Объединение", "Разделение", "Текущие операции", "Применить и пересчитать", "Сравнение до/после"])
    with tabs[0]:
        _render_merge_tab(source_gallery_dir, edits_csv)
    with tabs[1]:
        _render_split_tab(source_gallery_dir, edits_csv)
    with tabs[2]:
        _render_current_edits(edits_csv)
    with tabs[3]:
        _render_apply_tab(config, experiment_dir, source_gallery_dir, edits_csv)
    with tabs[4]:
        _render_before_after_tab(experiment_dir)
