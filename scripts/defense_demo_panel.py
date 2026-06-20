from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from path_utils import to_current_os_path

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


def _status_badge(label: str, ok: bool, path: Path | None = None) -> None:
    icon = "✅" if ok else "⚠️"
    text = f"{icon} **{label}**"
    if path is not None:
        text += f"  
`{_rel(path)}`"
    st.markdown(text)


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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("объектов", len(results))
        c2.metric("matched", int((results.get("sku_status", "") == "matched").sum()))
        c3.metric("matched_uncertain", int((results.get("sku_status", "") == "matched_uncertain").sum()))
        c4.metric("unknown", int((results.get("sku_status", "") == "unknown").sum()))
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


def _render_faq() -> None:
    faq_path = ROOT / "docs" / "DEFENSE_FAQ.md"
    text = _read_text(faq_path)
    if text:
        st.markdown(text)
    else:
        st.info("Файл FAQ пока не найден.")


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
        "1. Обзор",
        "2. Данные и профиль",
        "3. Фрагменты",
        "4. Идентификация",
        "5. Отчеты",
        "6. FAQ",
    ])

    with tabs[0]:
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
            }
        )
        images = _preview_images(experiment_dir)
        if images:
            st.markdown("#### Превью визуализаций")
            cols = st.columns(min(4, len(images)))
            for idx, image in enumerate(images):
                with cols[idx % len(cols)]:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tabs[1]:
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

    with tabs[2]:
        _render_crop_gallery(experiment_dir)

    with tabs[3]:
        _render_results_table(experiment_dir)
        st.info("Для ручного изменения конкретного назначения открой раздел `Ручная проверка идентификации` в меню `Запуск задач`.")

    with tabs[4]:
        reports = {
            "full_experiment_summary.md": experiment_dir / "05_reports" / "full_experiment_summary.md",
            "threshold_analysis.csv": experiment_dir / "05_reports" / "threshold_analysis.csv",
            "assignment_uncertainty_report.md": experiment_dir / "04_identification" / "assignment_uncertainty_report.md",
            "manual_identification_report.md": experiment_dir / "06_manual_identification" / "manual_identification_report.md",
            "manual_gallery_report.md": experiment_dir / "06_manual_gallery" / "manual_gallery_report.md",
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

    with tabs[5]:
        _render_faq()
