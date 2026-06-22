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
FINAL_PROFILE_PATH = ROOT / "config" / "vkr_final.yaml"
CONFIG_PATH = FINAL_PROFILE_PATH

STATUS_LABELS = {
    "matched": "Уверенное совпадение (matched)",
    "matched_uncertain": "Требует проверки (matched_uncertain)",
    "unknown": "Не определено (unknown)",
}


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


@st.cache_data(show_spinner=False)
def _load_yaml_cached(path_text: str, mtime_ns: int) -> Dict[str, Any]:
    del mtime_ns
    try:
        return yaml.safe_load(Path(path_text).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_json_cached(path_text: str, mtime_ns: int) -> Dict[str, Any]:
    del mtime_ns
    try:
        return json.loads(Path(path_text).read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_csv_cached(path_text: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    try:
        return pd.read_csv(path_text).fillna("")
    except Exception:
        return pd.DataFrame()


def _mtime(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


def _read_yaml(path: Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    return _load_yaml_cached(str(path), _mtime(path))


def _read_json(path: Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    return _load_json_cached(str(path), _mtime(path))


def _read_csv(path: Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    return _load_csv_cached(str(path), _mtime(path)).copy()


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
    path = _p(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _default_experiment_dir(config: Dict[str, Any]) -> Path:
    full = config.get("full_photo_identification", {})
    return _p(
        full.get("out_dir")
        or "D:/1Diplom/shelfvision_results/full_photo_identification_vkr_final"
    )


def experiment_paths(experiment_dir: str | Path) -> Dict[str, Path]:
    exp = _p(experiment_dir)
    return {
        "manifest": exp / "00_manifest" / "all_images.csv",
        "environment": exp / "00_manifest" / "run_environment.json",
        "split": exp / "00_manifest" / "split_params.json",
        "gallery_predictions": exp / "01_gallery_inference" / "predictions.json",
        "gallery_summary": exp / "02_demo_gallery" / "demo_sku_gallery_summary.json",
        "query_predictions": exp / "03_query_inference" / "predictions.json",
        "crops": exp / "04_identification" / "crops_manifest.csv",
        "results": exp / "04_identification" / "identification_results.csv",
        "corrected": exp
        / "06_manual_identification"
        / "identification_results_corrected.csv",
        "edits": exp
        / "06_manual_identification"
        / "manual_identification_edits.csv",
        "summary": exp / "05_reports" / "full_experiment_summary.json",
        "thresholds": exp / "05_reports" / "threshold_analysis.csv",
        "source_check": exp / "export" / "data_source_check.json",
        "export": exp / "export" / "demo_artifacts.zip",
    }


def result_source(experiment_dir: str | Path) -> Path:
    paths = experiment_paths(experiment_dir)
    return paths["corrected"] if _p(paths["corrected"]).exists() else paths["results"]


def _status_summary(df: pd.DataFrame) -> Dict[str, float | int]:
    if df.empty or "sku_status" not in df.columns:
        return {
            "total": 0,
            "matched": 0,
            "matched_uncertain": 0,
            "unknown": 0,
            "assigned_rate": 0.0,
            "manual_edits": 0,
        }
    total = len(df)
    statuses = df["sku_status"].astype(str)
    matched = int((statuses == "matched").sum())
    uncertain = int((statuses == "matched_uncertain").sum())
    unknown = int((statuses == "unknown").sum())
    manual = int(
        (
            df.get("manual_edit_applied", pd.Series([False] * total))
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        ).sum()
    )
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


def _file_status(label: str, path: Path) -> bool:
    path = _p(path)
    if path.exists():
        st.write(f"✅ {label}: `{_rel(path)}`")
        return True
    st.write(f"⚠️ {label}: `{_rel(path)}` — не найден")
    return False


def _download_file(path: Path, label: str, mime: str, key: str) -> None:
    path = _p(path)
    if not path.exists() or not path.is_file():
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def _render_quick_scenario() -> None:
    st.markdown("#### Как пройти демо")
    st.info(
        "Интерфейс читает результаты уже выполненного эксперимента. "
        "Обучение и полный инференс запускаются отдельными скриптами."
    )
    st.markdown(
        """
1. Откройте **Обзор** и убедитесь, что основные файлы найдены.
2. Откройте **Параметры** и проверьте профиль запуска и источник данных.
3. Откройте **Фрагменты** и оцените качество выделения товаров.
4. Откройте **Идентификация** и отфильтруйте уверенные, спорные и неопределённые результаты.
5. В **Ручной проверке** откройте спорный объект и при необходимости измените назначение.
6. В **Истории** сохраните контрольную точку или просмотрите выполненные действия.
7. В **До/после** сравните исходный и скорректированный результат.
8. В **Выборе SKU** соберите компактный набор по выбранным товарам.
9. В **Экспорте** сформируйте ZIP-архив результатов.
"""
    )


def _render_overview(experiment_dir: Path) -> None:
    paths = experiment_paths(experiment_dir)
    summary = _read_json(paths["summary"])
    source = result_source(experiment_dir)
    results = _read_csv(source)
    stats = _status_summary(results)

    if results.empty:
        st.warning(
            "Результаты идентификации не найдены. Проверьте путь к папке "
            "эксперимента или сначала выполните полный конвейер."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", stats["total"] or _safe_int(summary.get("query_objects_count")))
    c2.metric("Уверенные", stats["matched"])
    c3.metric("Требуют проверки", stats["matched_uncertain"])
    c4.metric("Не определены", stats["unknown"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Изображений галереи", _safe_int(summary.get("gallery_images_count")))
    c6.metric("Проверяемых изображений", _safe_int(summary.get("query_images_count")))
    c7.metric("Демонстрационных SKU", _safe_int(summary.get("created_demo_sku_count")))
    c8.metric("Доля с кандидатом", f"{stats['assigned_rate']:.4f}")
    st.caption(
        "Доля с кандидатом показывает распределение статусов и не является "
        "точностью SKU-распознавания без эталонной SKU-разметки."
    )

    source_check = _read_json(paths["source_check"])
    if source_check:
        status = str(source_check.get("status", "warning"))
        message = (
            f"Проверка источника данных: {status}; "
            f"manifest={source_check.get('manifest_images_count', 0)} изображений."
        )
        if status == "ok":
            st.success(message)
        elif status == "error":
            st.error(message)
        else:
            st.warning(message)
    else:
        st.info(
            "Источник данных ещё не подтверждён отдельным отчётом. "
            "Запустите scripts/verify_experiment_source.py."
        )

    with st.expander("Основные файлы", expanded=False):
        labels = {
            "manifest": "Манифест изображений",
            "environment": "Среда запуска",
            "split": "Параметры разбиения",
            "gallery_predictions": "Предсказания галерейной части",
            "gallery_summary": "Сводка галереи",
            "query_predictions": "Предсказания проверяемой части",
            "results": "Исходные результаты идентификации",
            "corrected": "Скорректированные результаты",
            "thresholds": "Анализ порогов",
            "export": "ZIP-архив",
        }
        for name, label in labels.items():
            _file_status(label, paths[name])

    images = _preview_images(experiment_dir)
    if images:
        st.markdown("#### Примеры визуализаций")
        cols = st.columns(min(4, len(images)))
        for idx, image_path in enumerate(images):
            with cols[idx % len(cols)]:
                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True,
                )


def _render_profile(config: Dict[str, Any]) -> None:
    profile = _read_yaml(FINAL_PROFILE_PATH) or config
    full = profile.get("full_photo_identification", {})
    runtime = profile.get("runtime", {})
    feature = profile.get("feature_extractor", {})

    st.markdown("#### Параметры запуска")
    cols = st.columns(4)
    cols[0].metric("Изображений галереи", full.get("gallery_count", 160))
    cols[1].metric("Проверяемых изображений", full.get("query_count", 140))
    cols[2].metric("Максимум SKU", full.get("max_sku", 200))
    cols[3].metric("Кандидатов top-k", full.get("top_k", 5))

    st.markdown("#### Источник и выходные данные")
    st.code(
        "\n".join(
            [
                f"images_dir: {full.get('images_dir', '')}",
                f"out_dir: {full.get('out_dir', '')}",
                f"gallery_dir: {full.get('gallery_dir', '')}",
            ]
        ),
        language="text",
    )
    st.caption(
        "Фактический источник необходимо сверять с all_images.csv и "
        "run_environment.json конкретного запуска."
    )

    st.markdown("#### Модель и сопоставление")
    st.json(
        {
            "model": full.get("model", "yolo"),
            "conf": runtime.get("conf", 0.25),
            "imgsz": runtime.get("imgsz", 640),
            "threshold_tau": full.get("threshold", 0.65),
            "ambiguity_delta": full.get("ambiguity_margin", 0.03),
            "dedup_threshold": full.get("dedup_threshold", 0.82),
            "max_refs_per_sku": full.get("max_refs_per_sku", 15),
            "feature_extractor": feature,
        }
    )


def _render_crops(experiment_dir: Path) -> None:
    manifest = _read_csv(experiment_paths(experiment_dir)["crops"])
    if manifest.empty:
        st.info(
            "Манифест фрагментов не найден. После полного запуска должен "
            "появиться файл 04_identification/crops_manifest.csv."
        )
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        available = sorted(
            str(value)
            for value in manifest.get("source_type", pd.Series(dtype=str)).unique()
            if str(value)
        )
        selected_types = st.multiselect(
            "Источник фрагмента",
            available,
            default=available,
        )
    with c2:
        show_limit = st.slider(
            "Количество карточек",
            min_value=4,
            max_value=min(100, max(4, len(manifest))),
            value=min(24, max(4, len(manifest))),
            step=4,
        )

    filtered = manifest
    if selected_types and "source_type" in filtered.columns:
        filtered = filtered[
            filtered["source_type"].astype(str).isin(selected_types)
        ]

    st.caption(
        f"Фрагментов всего: {len(manifest)}; показано: {min(show_limit, len(filtered))}."
    )
    cols = st.columns(4)
    for card_index, (_, row) in enumerate(filtered.head(show_limit).iterrows()):
        crop_path = _p(str(row.get("crop_path", "")))
        with cols[card_index % 4]:
            if crop_path.exists():
                st.image(
                    str(crop_path),
                    caption=(
                        f"obj {row.get('object_id', '')} · "
                        f"{row.get('source_type', '')} · "
                        f"score={_safe_float(row.get('score')):.3f}"
                    ),
                    use_container_width=True,
                )
            else:
                st.warning(f"Фрагмент не найден: {crop_path}")


def _render_identification_table(experiment_dir: Path) -> None:
    source = result_source(experiment_dir)
    df = _read_csv(source)
    if df.empty:
        st.info("Таблица идентификации не найдена.")
        return

    st.caption(f"Источник: `{_rel(_p(source))}`")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        statuses = sorted(
            str(value)
            for value in df.get("sku_status", pd.Series(dtype=str)).unique()
            if str(value)
        )
        selected = st.multiselect(
            "Статусы",
            statuses,
            default=statuses,
            format_func=lambda value: STATUS_LABELS.get(value, value),
        )
    with f2:
        search = st.text_input("Поиск по изображению или SKU", value="")
    with f3:
        row_limit = st.number_input(
            "Максимум строк",
            min_value=50,
            max_value=5000,
            value=300,
            step=50,
        )

    filtered = df
    if selected:
        filtered = filtered[filtered["sku_status"].astype(str).isin(selected)]
    if search.strip():
        query = search.strip()
        mask = pd.Series(False, index=filtered.index)
        for column in ["image_name", "sku_id", "sku_name"]:
            if column in filtered.columns:
                mask = mask | filtered[column].astype(str).str.contains(
                    query,
                    case=False,
                    na=False,
                )
        filtered = filtered[mask]

    st.caption(f"Найдено строк: {len(filtered)}")
    columns = [
        column
        for column in [
            "image_name",
            "object_id",
            "sku_id",
            "sku_status",
            "sku_confidence",
            "distinct_margin",
            "top_k",
            "crop_path",
        ]
        if column in filtered.columns
    ]
    st.dataframe(
        filtered[columns].head(int(row_limit)),
        use_container_width=True,
        hide_index=True,
    )
    _download_file(
        _p(source),
        "Скачать текущую таблицу CSV",
        "text/csv",
        "download_identification_csv",
    )


def _render_before_after(experiment_dir: Path) -> None:
    paths = experiment_paths(experiment_dir)
    raw = _read_csv(paths["results"])
    corrected = _read_csv(paths["corrected"])
    edits = _read_csv(paths["edits"])
    if raw.empty:
        st.info("Исходная таблица идентификации не найдена.")
        return

    raw_stats = _status_summary(raw)
    corrected_stats = _status_summary(corrected) if not corrected.empty else raw_stats
    table = pd.DataFrame(
        [
            {
                "Показатель": "Всего объектов",
                "До": raw_stats["total"],
                "После": corrected_stats["total"],
            },
            {
                "Показатель": "Уверенные (matched)",
                "До": raw_stats["matched"],
                "После": corrected_stats["matched"],
            },
            {
                "Показатель": "Требуют проверки (matched_uncertain)",
                "До": raw_stats["matched_uncertain"],
                "После": corrected_stats["matched_uncertain"],
            },
            {
                "Показатель": "Не определены (unknown)",
                "До": raw_stats["unknown"],
                "После": corrected_stats["unknown"],
            },
            {
                "Показатель": "Доля с кандидатом",
                "До": f"{raw_stats['assigned_rate']:.4f}",
                "После": f"{corrected_stats['assigned_rate']:.4f}",
            },
            {
                "Показатель": "Ручных правок",
                "До": 0,
                "После": len(edits),
            },
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Ручная проверка изменяет экспертно скорректированную версию результатов, "
        "но не превращает долю назначений в accuracy без эталонной разметки."
    )


def _render_selected_sku(experiment_dir: Path) -> None:
    source = result_source(experiment_dir)
    df = _read_csv(source)
    if df.empty or "sku_id" not in df.columns:
        st.info("Таблица идентификации не найдена или не содержит sku_id.")
        return

    sku_counts = (
        df[df["sku_id"].astype(str).str.len() > 0]
        .groupby("sku_id")
        .size()
        .sort_values(ascending=False)
    )
    options = sku_counts.index.astype(str).tolist()
    selected = st.multiselect(
        "SKU для набора",
        options,
        default=options[: min(5, len(options))],
    )
    include_unknown = st.checkbox(
        "Включать unknown, где выбранный SKU встречается в top-k",
        value=False,
    )
    max_rows = st.number_input(
        "Максимум проверяемых фрагментов на SKU",
        min_value=1,
        max_value=500,
        value=40,
    )

    if selected:
        preview = df[df["sku_id"].astype(str).isin(selected)].copy()
        view_columns = [
            column
            for column in [
                "image_name",
                "object_id",
                "sku_id",
                "sku_status",
                "sku_confidence",
                "distinct_margin",
                "crop_path",
                "top_k",
            ]
            if column in preview.columns
        ]
        st.dataframe(
            preview[view_columns].head(200),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Собрать набор по выбранным SKU",
        use_container_width=True,
        disabled=not selected,
    ):
        with st.spinner("Копирование эталонов и проверяемых фрагментов..."):
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
    output_zip = _p(
        st.text_input(
            "Путь к ZIP-архиву",
            value=str(experiment_dir / "export" / "demo_artifacts.zip"),
        )
    )
    include_visuals = st.checkbox(
        "Включить ограниченное число визуализаций",
        value=True,
    )
    visual_limit = st.number_input(
        "Лимит визуализаций на папку",
        min_value=0,
        max_value=300,
        value=30,
    )
    if st.button("Собрать ZIP-архив", use_container_width=True):
        with st.spinner("Формирование архива..."):
            outputs = build_defense_export_zip(
                experiment_dir=experiment_dir,
                output_zip=output_zip,
                include_visualizations=include_visuals,
                visualized_limit_per_dir=int(visual_limit),
            )
        st.success("ZIP-архив сформирован.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{_rel(_p(path))}`")

    _download_file(
        output_zip,
        "Скачать ZIP-архив",
        "application/zip",
        "download_demo_zip",
    )


def _render_faq() -> None:
    st.markdown(
        """
### Что делает этот интерфейс?
Он читает результаты уже выполненного полного эксперимента и позволяет просматривать фрагменты и назначения SKU, выполнять ручную проверку, сравнивать результат до/после и экспортировать материалы. Обучение модели и полный инференс запускаются отдельными скриптами.

### Что означают статусы?
- `matched` — кандидат прошёл порог сходства и имеет достаточный отрыв от альтернатив.
- `matched_uncertain` — кандидат прошёл порог, но близок к другому SKU и требует проверки.
- `unknown` — кандидат не прошёл порог или отсутствует.

### Где хранятся векторы?
В локальном файловом кэше: `.npy` содержит вектор, `.json` — метаданные исходного файла и параметров экстрактора.

### Какие признаки используются?
HSV-гистограмма 16×16×8 объединяется с усреднённым ORB-дескриптором длины 64. Итоговый 2112-мерный вектор нормализуется, сходство рассчитывается косинусной мерой.

### Что означает контрольная точка истории?
Она сохраняет конфигурацию и ссылку на каталог результатов. Это не полный транзакционный откат файлов.

### Почему доля с кандидатом не является accuracy?
Для строгой accuracy необходима эталонная SKU-разметка каждого проверяемого объекта.
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="Демо анализа полочных сцен",
        page_icon="🧰",
        layout="wide",
    )
    config = _read_yaml(CONFIG_PATH)
    experiment_dir = _p(
        st.sidebar.text_input(
            "Папка с результатами",
            value=str(_default_experiment_dir(config)),
            help="Укажите каталог полного эксперимента с папками 00_manifest–05_reports.",
        )
    )

    st.title("🧰 Демо анализа полочных сцен")
    st.caption(
        "Просмотр результатов, ручная проверка идентификации, история действий "
        "и экспорт материалов."
    )

    tabs = st.tabs(
        [
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
        ]
    )

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
        review_config.setdefault("full_photo_identification", {})[
            "out_dir"
        ] = str(experiment_dir)
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
