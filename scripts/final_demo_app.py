from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from identification_review_panel import page_identification_review
from path_utils import to_current_os_path
from src.identification.selected_sku_exporter import export_selected_sku_demo
from src.reporting.defense_export import build_defense_export_zip

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIG_PATH = ROOT / "config" / "shelfvision.yaml"
FINAL_PROFILE_PATH = ROOT / "config" / "vkr_final.yaml"


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_json(path: Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _default_experiment_dir(config: Dict[str, Any]) -> Path:
    full = config.get("full_photo_identification", {})
    final_profile = _read_yaml(FINAL_PROFILE_PATH).get("full_photo_identification", {})
    return _p(full.get("out_dir") or final_profile.get("out_dir") or "D:/1Diplom/shelfvision_results/full_photo_identification")


def _status_summary(df: pd.DataFrame) -> Dict[str, float | int]:
    if df.empty or "sku_status" not in df.columns:
        return {"total": 0, "matched": 0, "matched_uncertain": 0, "unknown": 0, "assigned_rate": 0.0, "manual_edits": 0}
    total = len(df)
    matched = int((df["sku_status"].astype(str) == "matched").sum())
    uncertain = int((df["sku_status"].astype(str) == "matched_uncertain").sum())
    unknown = int((df["sku_status"].astype(str) == "unknown").sum())
    manual = int((df.get("manual_edit_applied", pd.Series([False] * total)).astype(str).str.lower().isin({"true", "1", "yes"})).sum())
    return {
        "total": total,
        "matched": matched,
        "matched_uncertain": uncertain,
        "unknown": unknown,
        "assigned_rate": (matched + uncertain) / total if total else 0.0,
        "manual_edits": manual,
    }


def _preview_images(experiment_dir: Path, limit: int = 8) -> List[Path]:
    search_roots = [
        experiment_dir / "04_identification" / "visualized",
        experiment_dir / "03_query_inference" / "visualized",
        experiment_dir / "01_gallery_inference" / "visualized",
        experiment_dir / "visualized",
    ]
    images: List[Path] = []
    for root in search_roots:
        root = _p(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
                if len(images) >= limit:
                    return images
    return images


def _file_status(label: str, path: Path) -> None:
    path = _p(path)
    if path.exists():
        st.write(f"✅ {label}: `{_rel(path)}`")
    else:
        st.write(f"⚠️ {label}: `{_rel(path)}` — не найден")


def _render_quick_scenario() -> None:
    st.markdown("#### Как пройти демо")
    st.markdown(
        """
1. Открой вкладку **Обзор** и проверь, что основные файлы найдены.
2. Открой **Параметры** и проверь текущий профиль запуска.
3. Открой **Фрагменты** и посмотри примеры вырезанных товаров.
4. Открой **Идентификация** и проверь статусы `matched`, `matched_uncertain`, `unknown`.
5. Открой **Ручная проверка**, выбери спорный объект и при необходимости измени назначение.
6. Открой **До/после**, чтобы увидеть эффект ручной проверки.
7. Открой **Выбор SKU**, чтобы собрать компактный набор по выбранным товарам.
8. Открой **Экспорт** и собери ZIP-архив с результатами.
"""
    )


def _render_overview(experiment_dir: Path) -> None:
    summary = _read_json(experiment_dir / "05_reports" / "full_experiment_summary.json")
    results = _read_csv(experiment_dir / "04_identification" / "identification_results.csv")
    stats = _status_summary(results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", _safe_int(summary.get("query_objects_count")) or stats["total"])
    c2.metric("matched", _safe_int(summary.get("matched")) or stats["matched"])
    c3.metric("matched_uncertain", _safe_int(summary.get("matched_uncertain")) or stats["matched_uncertain"])
    c4.metric("unknown", _safe_int(summary.get("unknown")) or stats["unknown"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("gallery изображений", _safe_int(summary.get("gallery_images_count")))
    c6.metric("query изображений", _safe_int(summary.get("query_images_count")))
    c7.metric("demo SKU", _safe_int(summary.get("created_demo_sku_count")))
    c8.metric("assigned_rate", f"{_safe_float(summary.get('assigned_rate')) or stats['assigned_rate']:.4f}")

    st.markdown("#### Основные файлы")
    files = {
        "manifest": experiment_dir / "00_manifest" / "all_images.csv",
        "gallery predictions": experiment_dir / "01_gallery_inference" / "predictions.json",
        "gallery summary": experiment_dir / "02_demo_gallery" / "demo_sku_gallery_summary.json",
        "query predictions": experiment_dir / "03_query_inference" / "predictions.json",
        "identification results": experiment_dir / "04_identification" / "identification_results.csv",
        "corrected results": experiment_dir / "06_manual_identification" / "identification_results_corrected.csv",
        "export ZIP": experiment_dir / "defense_export" / "vkr_defense_artifacts.zip",
    }
    for label, path in files.items():
        _file_status(label, path)

    images = _preview_images(experiment_dir)
    if images:
        st.markdown("#### Превью")
        cols = st.columns(min(4, len(images)))
        for idx, image_path in enumerate(images):
            with cols[idx % len(cols)]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _render_profile(config: Dict[str, Any]) -> None:
    profile = _read_yaml(FINAL_PROFILE_PATH)
    full = profile.get("full_photo_identification", {}) or config.get("full_photo_identification", {})
    runtime = profile.get("runtime", {}) or config.get("runtime", {})
    feature = profile.get("feature_extractor", {})

    st.markdown("#### Параметры запуска")
    cols = st.columns(4)
    cols[0].metric("gallery", full.get("gallery_count", 160))
    cols[1].metric("query", full.get("query_count", 140))
    cols[2].metric("max SKU", full.get("max_sku", 200))
    cols[3].metric("top-k", full.get("top_k", 5))

    st.json(
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


def _render_crops(experiment_dir: Path) -> None:
    manifest = _read_csv(experiment_dir / "04_identification" / "crops_manifest.csv")
    if manifest.empty:
        st.info("Манифест вырезанных фрагментов пока не найден.")
        return
    st.caption(f"Фрагментов в манифесте: {len(manifest)}")
    cols = st.columns(4)
    for idx, row in manifest.head(24).iterrows():
        crop_path = _p(str(row.get("crop_path", "")))
        with cols[idx % 4]:
            if crop_path.exists():
                st.image(str(crop_path), caption=f"obj {row.get('object_id', '')}", use_container_width=True)
            else:
                st.caption(str(crop_path))


def _render_identification_table(experiment_dir: Path) -> None:
    corrected = experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"
    raw = experiment_dir / "04_identification" / "identification_results.csv"
    source = corrected if _p(corrected).exists() else raw
    df = _read_csv(source)
    if df.empty:
        st.info("Таблица идентификации пока не найдена.")
        return
    st.caption(f"Источник: `{_rel(_p(source))}`")
    statuses = sorted([str(x) for x in df.get("sku_status", pd.Series(dtype=str)).dropna().unique().tolist()])
    selected = st.multiselect("Статусы", statuses, default=statuses)
    if selected:
        df = df[df["sku_status"].astype(str).isin(selected)]
    cols = [col for col in ["image_name", "object_id", "sku_id", "sku_status", "sku_confidence", "distinct_margin", "top_k", "crop_path"] if col in df.columns]
    st.dataframe(df[cols].head(300), use_container_width=True, hide_index=True)


def _render_before_after(experiment_dir: Path) -> None:
    raw = _read_csv(experiment_dir / "04_identification" / "identification_results.csv")
    corrected = _read_csv(experiment_dir / "06_manual_identification" / "identification_results_corrected.csv")
    edits = _read_csv(experiment_dir / "06_manual_identification" / "manual_identification_edits.csv")
    if raw.empty:
        st.info("Исходная таблица идентификации пока не найдена.")
        return
    raw_stats = _status_summary(raw)
    corrected_stats = _status_summary(corrected) if not corrected.empty else raw_stats
    table = pd.DataFrame(
        [
            {"Показатель": "Всего объектов", "До": raw_stats["total"], "После": corrected_stats["total"]},
            {"Показатель": "matched", "До": raw_stats["matched"], "После": corrected_stats["matched"]},
            {"Показатель": "matched_uncertain", "До": raw_stats["matched_uncertain"], "После": corrected_stats["matched_uncertain"]},
            {"Показатель": "unknown", "До": raw_stats["unknown"], "После": corrected_stats["unknown"]},
            {"Показатель": "assigned_rate", "До": f"{raw_stats['assigned_rate']:.4f}", "После": f"{corrected_stats['assigned_rate']:.4f}"},
            {"Показатель": "ручных правок", "До": 0, "После": len(edits)},
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("assigned_rate — доля объектов с назначенным кандидатом. Для оценки точности нужна эталонная SKU-разметка.")


def _render_selected_sku(experiment_dir: Path) -> None:
    corrected = experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"
    raw = experiment_dir / "04_identification" / "identification_results.csv"
    source = corrected if _p(corrected).exists() else raw
    df = _read_csv(source)
    if df.empty or "sku_id" not in df.columns:
        st.info("Таблица идентификации пока не найдена или не содержит sku_id.")
        return
    sku_counts = df[df["sku_id"].astype(str).str.len() > 0].groupby("sku_id").size().sort_values(ascending=False)
    options = sku_counts.index.astype(str).tolist()
    selected = st.multiselect("SKU для показа", options, default=options[: min(5, len(options))])
    include_unknown = st.checkbox("Включать unknown, где выбранный SKU встречается в top-k", value=False)
    max_rows = st.number_input("Максимум query-фрагментов на SKU", min_value=1, max_value=500, value=40)
    if selected:
        preview = df[df["sku_id"].astype(str).isin(selected)].copy()
        view_cols = [col for col in ["image_name", "object_id", "sku_id", "sku_status", "sku_confidence", "distinct_margin", "crop_path", "top_k"] if col in preview.columns]
        st.dataframe(preview[view_cols].head(200), use_container_width=True, hide_index=True)
    if st.button("Собрать набор по выбранным SKU", use_container_width=True):
        outputs = export_selected_sku_demo(
            experiment_dir=experiment_dir,
            selected_skus=selected,
            output_dir=experiment_dir / "selected_sku_demo",
            results_csv=source,
            max_rows_per_sku=int(max_rows),
            include_unknown_similar=include_unknown,
        )
        st.success("Набор выбранных SKU собран.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{_rel(_p(path))}`")


def _render_export(experiment_dir: Path) -> None:
    output_zip = _p(st.text_input("Путь к ZIP-архиву", value=str(experiment_dir / "defense_export" / "vkr_defense_artifacts.zip")))
    include_visuals = st.checkbox("Включить ограниченное число визуализаций", value=True)
    visual_limit = st.number_input("Лимит визуализаций на папку", min_value=0, max_value=300, value=30)
    if st.button("Собрать ZIP-архив материалов", use_container_width=True):
        outputs = build_defense_export_zip(
            experiment_dir=experiment_dir,
            output_zip=output_zip,
            include_visualizations=include_visuals,
            visualized_limit_per_dir=int(visual_limit),
        )
        st.success("ZIP-архив материалов сформирован.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{_rel(_p(path))}`")


def _render_faq() -> None:
    st.markdown(
        """
### Что делает интерфейс?
Он показывает полный визуальный контур: загрузка/выбор изображений, локализация товаров, вырезанные фрагменты, SKU-галерея, сопоставление, ручная проверка и экспорт результатов.

### Что означают matched, matched_uncertain и unknown?
- `matched`: кандидат прошёл порог сходства и имеет достаточный отрыв от альтернатив.
- `matched_uncertain`: кандидат прошёл порог, но близок к другому SKU.
- `unknown`: кандидат не прошёл порог или отсутствует.

### Где хранятся векторы?
В локальном файловом кэше: `.npy` для вектора и `.json` для метаданных.

### Какие признаки используются?
HSV-гистограмма + ORB-признаки, затем нормализация и косинусное сходство.

### Почему assigned_rate не является accuracy?
Это доля объектов с назначенным кандидатом. Для строгой accuracy нужна эталонная SKU-разметка каждого объекта.
"""
    )


def main() -> None:
    st.set_page_config(page_title="Демо анализа полочных сцен", page_icon="🧰", layout="wide")
    config = _read_yaml(CONFIG_PATH)
    experiment_dir = _p(
        st.sidebar.text_input("Папка итогового эксперимента", value=str(_default_experiment_dir(config)))
    )

    st.title("🧰 Демо анализа полочных сцен")
    st.caption("Просмотр результатов, ручная проверка идентификации и экспорт материалов.")

    tabs = st.tabs([
        "Сценарий",
        "Обзор",
        "Параметры",
        "Фрагменты",
        "Идентификация",
        "Ручная проверка",
        "До/после",
        "Выбор SKU",
        "Экспорт",
        "FAQ",
    ])

    with tabs[0]:
        _render_quick_scenario()
    with tabs[1]:
        _render_overview(experiment_dir)
    with tabs[2]:
        _render_profile(config)
    with tabs[3]:
        _render_crops(experiment_dir)
    with tabs[4]:
        _render_identification_table(experiment_dir)
    with tabs[5]:
        review_config = dict(config)
        review_config.setdefault("full_photo_identification", {})["out_dir"] = str(experiment_dir)
        page_identification_review(review_config)
    with tabs[6]:
        _render_before_after(experiment_dir)
    with tabs[7]:
        _render_selected_sku(experiment_dir)
    with tabs[8]:
        _render_export(experiment_dir)
    with tabs[9]:
        _render_faq()


if __name__ == "__main__":
    main()
