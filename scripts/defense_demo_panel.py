from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from path_utils import to_current_os_path
from src.identification.selected_sku_exporter import export_selected_sku_demo
from src.reporting.defense_export import build_defense_export_zip

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_EXTS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _read_json(path: Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 12000) -> str:
    path = _p(path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return text[:max_chars] + ("\n\n..." if len(text) > max_chars else "")


def _read_csv(path: Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _default_experiment_dir(config: Dict[str, Any]) -> Path:
    full = config.get("full_photo_identification", {})
    return _p(full.get("out_dir") or "D:/1Diplom/shelfvision_results/full_photo_identification")


def _preview_images(root: Path, limit: int = 8) -> List[Path]:
    root = _p(root)
    if not root.exists():
        return []
    search_roots = [root]
    if root.is_dir():
        search_roots = [
            root / "04_identification" / "visualized",
            root / "03_query_inference" / "visualized",
            root / "01_gallery_inference" / "visualized",
            root / "visualized",
            root,
        ]
    images: List[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
                if len(images) >= limit:
                    return images
    return images


def _render_file_links(paths: Dict[str, Path]) -> None:
    for label, path in paths.items():
        path = _p(path)
        if path.exists():
            st.write(f"✅ {label}: `{_rel(path)}`")
        else:
            st.write(f"⚠️ {label}: `{_rel(path)}` — не найден")


def _load_final_profile() -> Dict[str, Any]:
    path = ROOT / "config" / "vkr_final.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _status_summary(df: pd.DataFrame) -> Dict[str, float | int]:
    if df.empty or "sku_status" not in df.columns:
        return {
            "total": 0,
            "matched": 0,
            "matched_uncertain": 0,
            "unknown": 0,
            "assigned": 0,
            "assigned_rate": 0.0,
            "unknown_rate": 0.0,
            "manual_edits": 0,
        }
    total = len(df)
    matched = int((df["sku_status"].astype(str) == "matched").sum())
    uncertain = int((df["sku_status"].astype(str) == "matched_uncertain").sum())
    unknown = int((df["sku_status"].astype(str) == "unknown").sum())
    assigned = matched + uncertain
    manual_edits = int((df.get("manual_edit_applied", pd.Series([False] * total)).astype(str).str.lower().isin({"true", "1", "yes"})).sum())
    return {
        "total": total,
        "matched": matched,
        "matched_uncertain": uncertain,
        "unknown": unknown,
        "assigned": assigned,
        "assigned_rate": assigned / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "manual_edits": manual_edits,
    }


def _render_final_profile(config: Dict[str, Any]) -> None:
    profile = _load_final_profile()
    full = profile.get("full_photo_identification", {}) if profile else config.get("full_photo_identification", {})
    runtime = profile.get("runtime", {}) if profile else config.get("runtime", {})
    feature = profile.get("feature_extractor", {}) if profile else {}

    st.markdown("#### Итоговый профиль ВКР")
    cols = st.columns(4)
    cols[0].metric("gallery", full.get("gallery_count", 160))
    cols[1].metric("query", full.get("query_count", 140))
    cols[2].metric("max SKU", full.get("max_sku", 200))
    cols[3].metric("top-k", full.get("top_k", 5))

    st.write(
        {
            "model": full.get("model", "yolo"),
            "conf": runtime.get("conf", 0.25),
            "imgsz": runtime.get("imgsz", 640),
            "threshold_tau": full.get("threshold", 0.65),
            "ambiguity_delta": full.get("ambiguity_margin", 0.03),
            "dedup_threshold": full.get("dedup_threshold", 0.82),
            "max_refs_per_sku": full.get("max_refs_per_sku", 15),
            "feature_extractor": feature or "HSV + ORB + cosine",
        }
    )


def _render_summary_metrics(experiment_dir: Path) -> None:
    summary = _read_json(experiment_dir / "05_reports" / "full_experiment_summary.json")
    demo_summary = _read_json(experiment_dir / "02_demo_gallery" / "demo_sku_gallery_summary.json")
    results = _read_csv(experiment_dir / "04_identification" / "identification_results.csv")

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("gallery изображений", _safe_int(summary.get("gallery_images_count")))
        c2.metric("query изображений", _safe_int(summary.get("query_images_count")))
        c3.metric("demo SKU", _safe_int(summary.get("created_demo_sku_count")))
        c4.metric("query объектов", _safe_int(summary.get("query_objects_count")))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("matched", _safe_int(summary.get("matched")))
        c6.metric("matched_uncertain", _safe_int(summary.get("matched_uncertain")))
        c7.metric("unknown", _safe_int(summary.get("unknown")))
        c8.metric("assigned_rate", f"{_safe_float(summary.get('assigned_rate')):.4f}")
    elif not results.empty:
        stats = _status_summary(results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("объектов", stats["total"])
        c2.metric("matched", stats["matched"])
        c3.metric("matched_uncertain", stats["matched_uncertain"])
        c4.metric("unknown", stats["unknown"])
    else:
        st.info("Итоговая сводка пока не найдена. Запусти полный контур или укажи другую папку эксперимента.")

    if demo_summary:
        st.caption(
            "Галерея: "
            f"извлечено фрагментов={demo_summary.get('extracted_crops_count', '')}, "
            f"отобрано={demo_summary.get('selected_crops_count', '')}, "
            f"эталонов={demo_summary.get('gallery_refs_count', '')}, "
            f"dedup={demo_summary.get('dedup_threshold', '')}."
        )


def _render_results_table(experiment_dir: Path) -> None:
    results_csv = experiment_dir / "04_identification" / "identification_results.csv"
    corrected_csv = experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"
    selected_csv = corrected_csv if corrected_csv.exists() else results_csv
    df = _read_csv(selected_csv)
    if df.empty:
        st.info("Таблица идентификации пока не найдена.")
        return

    st.caption(f"Показана таблица: `{_rel(selected_csv)}`")
    statuses = sorted([str(x) for x in df.get("sku_status", pd.Series(dtype=str)).dropna().unique().tolist()])
    selected_statuses = st.multiselect("Статусы", statuses, default=statuses, key="defense_status_filter")
    filtered = df[df["sku_status"].astype(str).isin(selected_statuses)] if selected_statuses and "sku_status" in df.columns else df

    show_cols = [
        col for col in [
            "image_name",
            "object_id",
            "crop_path",
            "sku_id",
            "sku_status",
            "sku_confidence",
            "distinct_margin",
            "top_k",
            "manual_edit_applied",
            "manual_edit_type",
        ] if col in filtered.columns
    ]
    st.dataframe(filtered[show_cols].head(300), use_container_width=True, hide_index=True)


def _render_crop_gallery(experiment_dir: Path) -> None:
    manifest_csv = experiment_dir / "04_identification" / "crops_manifest.csv"
    df = _read_csv(manifest_csv)
    if df.empty:
        st.info("`crops_manifest.csv` пока не найден.")
        return
    st.caption(f"Фрагментов в манифесте: {len(df)}")
    sample = df.head(24)
    cols = st.columns(4)
    for idx, row in sample.iterrows():
        crop_path = _p(str(row.get("crop_path", "")))
        with cols[idx % 4]:
            if crop_path.exists():
                st.image(str(crop_path), caption=f"obj {row.get('object_id', '')} / {row.get('source_type', '')}", use_container_width=True)
            else:
                st.caption(str(crop_path))


def _render_before_after(experiment_dir: Path) -> None:
    raw_csv = experiment_dir / "04_identification" / "identification_results.csv"
    corrected_csv = experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"
    edits_csv = experiment_dir / "06_manual_identification" / "manual_identification_edits.csv"
    raw = _read_csv(raw_csv)
    corrected = _read_csv(corrected_csv)
    edits = _read_csv(edits_csv)

    if raw.empty:
        st.info("Исходный `identification_results.csv` пока не найден.")
        return

    raw_stats = _status_summary(raw)
    corrected_stats = _status_summary(corrected) if not corrected.empty else raw_stats

    st.markdown("#### Сравнение исходного и corrected-результата")
    table = pd.DataFrame(
        [
            {"Показатель": "Всего объектов", "До": raw_stats["total"], "После": corrected_stats["total"]},
            {"Показатель": "matched", "До": raw_stats["matched"], "После": corrected_stats["matched"]},
            {"Показатель": "matched_uncertain", "До": raw_stats["matched_uncertain"], "После": corrected_stats["matched_uncertain"]},
            {"Показатель": "unknown", "До": raw_stats["unknown"], "После": corrected_stats["unknown"]},
            {"Показатель": "assigned_rate", "До": f"{raw_stats['assigned_rate']:.4f}", "После": f"{corrected_stats['assigned_rate']:.4f}"},
            {"Показатель": "ручных правок", "До": 0, "После": len(edits)},
            {"Показатель": "применено правок", "До": 0, "После": corrected_stats.get("manual_edits", 0)},
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    if corrected.empty:
        st.info("Corrected CSV пока не сформирован. Применить журнал правок можно в разделе `Запуск задач → Ручная проверка идентификации`.")
    if not edits.empty:
        with st.expander("Последние ручные правки", expanded=False):
            st.dataframe(edits.tail(100), use_container_width=True, hide_index=True)

    st.warning("Ручная проверка повышает экспертную согласованность демонстрационного контура, но не превращает assigned_rate в top-1 accuracy без эталонной SKU-разметки.")


def _render_selected_sku_export(experiment_dir: Path) -> None:
    corrected_csv = experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"
    raw_csv = experiment_dir / "04_identification" / "identification_results.csv"
    results_csv = corrected_csv if corrected_csv.exists() else raw_csv
    df = _read_csv(results_csv)
    if df.empty or "sku_id" not in df.columns:
        st.info("Таблица идентификации пока не найдена или не содержит sku_id.")
        return

    sku_counts = (
        df[df["sku_id"].astype(str).str.len() > 0]
        .groupby("sku_id")
        .size()
        .sort_values(ascending=False)
    )
    options = sku_counts.index.astype(str).tolist()
    default = options[: min(5, len(options))]
    selected = st.multiselect("SKU для демонстрации", options, default=default, key="selected_sku_demo_multiselect")
    include_unknown = st.checkbox("Включать unknown, где выбранный SKU встречается в top-k", value=False)
    max_rows = st.number_input("Максимум query-фрагментов на SKU", min_value=1, max_value=500, value=40)

    if selected:
        preview = df[df["sku_id"].astype(str).isin(selected)].copy()
        st.caption(f"Найдено строк по выбранным SKU: {len(preview)}")
        show_cols = [col for col in ["image_name", "object_id", "sku_id", "sku_status", "sku_confidence", "distinct_margin", "crop_path", "top_k"] if col in preview.columns]
        st.dataframe(preview[show_cols].head(200), use_container_width=True, hide_index=True)

    if st.button("Собрать демонстрационный набор по выбранным SKU", use_container_width=True):
        outputs = export_selected_sku_demo(
            experiment_dir=experiment_dir,
            selected_skus=selected,
            output_dir=experiment_dir / "selected_sku_demo",
            results_csv=results_csv,
            max_rows_per_sku=int(max_rows),
            include_unknown_similar=include_unknown,
        )
        st.success("Демонстрационный набор выбранных SKU собран.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{_rel(_p(path))}`")


def _render_export(experiment_dir: Path) -> None:
    output_zip = _p(
        st.text_input(
            "Путь к ZIP-архиву",
            value=str(experiment_dir / "defense_export" / "vkr_defense_artifacts.zip"),
            key="defense_export_zip_path",
        )
    )
    include_visuals = st.checkbox("Включить ограниченное число визуализаций", value=True)
    visual_limit = st.number_input("Лимит визуализаций на папку", min_value=0, max_value=300, value=30)

    if st.button("Собрать ZIP-архив материалов защиты", use_container_width=True):
        outputs = build_defense_export_zip(
            experiment_dir=experiment_dir,
            output_zip=output_zip,
            include_visualizations=include_visuals,
            visualized_limit_per_dir=int(visual_limit),
        )
        st.success("ZIP-архив материалов защиты сформирован.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{_rel(_p(path))}`")

    report = _read_text(output_zip.parent / "defense_export_report.md")
    if report:
        st.markdown(report)


def _render_faq() -> None:
    faq_path = ROOT / "docs" / "DEFENSE_FAQ.md"
    text = _read_text(faq_path)
    if text:
        st.markdown(text)
    else:
        st.info("Файл FAQ пока не найден.")


def _render_demo_script() -> None:
    script_path = ROOT / "docs" / "DEMO_SCRIPT_5_MIN.md"
    text = _read_text(script_path, max_chars=40000)
    if text:
        st.markdown(text)
    else:
        st.info("Файл сценария демонстрации пока не найден.")


def _render_readiness_checklist(experiment_dir: Path) -> None:
    st.markdown("#### Быстрый чек-лист перед показом")
    checks = {
        "есть итоговая сводка": experiment_dir / "05_reports" / "full_experiment_summary.md",
        "есть результаты идентификации": experiment_dir / "04_identification" / "identification_results.csv",
        "есть crop-манифест": experiment_dir / "04_identification" / "crops_manifest.csv",
        "есть corrected-результат или журнал правок": experiment_dir / "06_manual_identification" / "manual_identification_edits.csv",
        "есть FAQ защиты": ROOT / "docs" / "DEFENSE_FAQ.md",
        "есть сценарий на 5 минут": ROOT / "docs" / "DEMO_SCRIPT_5_MIN.md",
    }
    for title, path in checks.items():
        icon = "✅" if _p(path).exists() else "⚠️"
        st.write(f"{icon} {title}: `{_rel(_p(path))}`")


def page_defense_demo(config: Dict[str, Any]) -> None:
    st.header("Демо защиты: полный визуальный контур")
    st.caption(
        "Единый экран для показа сценария ВКР: данные → локализация → фрагменты → SKU-галерея → "
        "сопоставление → ручная проверка → отчетность. Название проекта в интерфейсе не используется как бренд."
    )

    experiment_dir = _p(
        st.text_input(
            "Папка итогового эксперимента",
            value=str(_default_experiment_dir(config)),
            key="defense_experiment_dir",
        )
    )

    tabs = st.tabs([
        "0. Сценарий защиты",
        "1. Обзор",
        "2. Данные и профиль",
        "3. Фрагменты",
        "4. Идентификация",
        "5. До/после",
        "6. Выбор SKU",
        "7. Отчеты",
        "8. Экспорт",
        "9. FAQ",
    ])

    with tabs[0]:
        _render_readiness_checklist(experiment_dir)
        st.divider()
        _render_demo_script()

    with tabs[1]:
        _render_summary_metrics(experiment_dir)
        st.markdown("#### Готовность артефактов")
        _render_file_links(
            {
                "manifest gallery/query": experiment_dir / "00_manifest" / "all_images.csv",
                "gallery predictions": experiment_dir / "01_gallery_inference" / "predictions.json",
                "demo gallery summary": experiment_dir / "02_demo_gallery" / "demo_sku_gallery_summary.json",
                "query predictions": experiment_dir / "03_query_inference" / "predictions.json",
                "identification results": experiment_dir / "04_identification" / "identification_results.csv",
                "full summary": experiment_dir / "05_reports" / "full_experiment_summary.md",
                "manual corrected results": experiment_dir / "06_manual_identification" / "identification_results_corrected.csv",
                "selected SKU demo": experiment_dir / "selected_sku_demo" / "selected_sku_report.md",
                "defense export ZIP": experiment_dir / "defense_export" / "vkr_defense_artifacts.zip",
            }
        )
        images = _preview_images(experiment_dir)
        if images:
            st.markdown("#### Превью визуализаций")
            cols = st.columns(min(4, len(images)))
            for idx, image in enumerate(images):
                with cols[idx % len(cols)]:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tabs[2]:
        _render_final_profile(config)
        st.markdown("#### Манифест изображений")
        for label, path in {
            "all_images.csv": experiment_dir / "00_manifest" / "all_images.csv",
            "gallery_images.csv": experiment_dir / "00_manifest" / "gallery_images.csv",
            "query_images.csv": experiment_dir / "00_manifest" / "query_images.csv",
        }.items():
            df = _read_csv(path)
            if df.empty:
                st.info(f"{label} не найден: `{_rel(path)}`")
            else:
                with st.expander(f"{label}: {len(df)} строк", expanded=False):
                    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

    with tabs[3]:
        _render_crop_gallery(experiment_dir)

    with tabs[4]:
        _render_results_table(experiment_dir)
        st.info("Для ручного изменения конкретного назначения открой раздел `Ручная проверка идентификации` в меню `Запуск задач`.")

    with tabs[5]:
        _render_before_after(experiment_dir)

    with tabs[6]:
        _render_selected_sku_export(experiment_dir)

    with tabs[7]:
        reports = {
            "full_experiment_summary.md": experiment_dir / "05_reports" / "full_experiment_summary.md",
            "threshold_analysis.csv": experiment_dir / "05_reports" / "threshold_analysis.csv",
            "assignment_uncertainty_report.md": experiment_dir / "04_identification" / "assignment_uncertainty_report.md",
            "manual_identification_report.md": experiment_dir / "06_manual_identification" / "manual_identification_report.md",
            "manual_gallery_report.md": experiment_dir / "06_manual_gallery" / "manual_gallery_report.md",
            "selected_sku_report.md": experiment_dir / "selected_sku_demo" / "selected_sku_report.md",
        }
        for label, path in reports.items():
            path = _p(path)
            if not path.exists():
                st.info(f"{label} не найден: `{_rel(path)}`")
                continue
            with st.expander(label, expanded=label == "full_experiment_summary.md"):
                if path.suffix.lower() == ".md":
                    st.markdown(_read_text(path))
                elif path.suffix.lower() == ".csv":
                    st.dataframe(_read_csv(path).head(300), use_container_width=True, hide_index=True)
                else:
                    st.code(_read_text(path), language="text")

    with tabs[8]:
        _render_export(experiment_dir)

    with tabs[9]:
        _render_faq()
