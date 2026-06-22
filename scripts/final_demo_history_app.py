from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

import action_history
import final_demo_app as core
from demo_sku_correction_panel import page_demo_sku_correction
from history_review_bridge import render_review_with_history
from identification_review_panel import page_identification_review
from src.identification.selected_sku_exporter import export_selected_sku_demo
from src.reporting.defense_export import build_defense_export_zip
from user_photos_panel import page_user_photos
from verify_experiment_source import (
    build_report as build_source_report,
    write_report as write_source_report,
)


def _config_snapshot(config: Dict[str, Any], exp: Path) -> Dict[str, Any]:
    full = config.get("full_photo_identification", {})
    paths = core.experiment_paths(exp) if hasattr(core, "experiment_paths") else {}
    return {
        "experiment_dir": str(exp),
        "full_photo_identification": full,
        "files": {name: str(path) for name, path in paths.items()},
    }


def _sidebar(config: Dict[str, Any]) -> Path:
    st.sidebar.header("Настройки")
    default_exp = st.session_state.get("demo_experiment_dir") or str(
        core._default_experiment_dir(config)
    )
    exp = core._p(
        st.sidebar.text_input(
            "Папка с результатами",
            value=default_exp,
            key="history_exp_dir",
            help="Каталог полного эксперимента с папками 00_manifest–05_reports.",
        )
    )
    st.session_state["demo_experiment_dir"] = str(exp)

    st.sidebar.caption(
        "Папка с результатами полного запуска: предсказания, фрагменты, идентификация и отчёты."
    )
    st.sidebar.divider()
    if st.sidebar.button("Создать контрольную точку", use_container_width=True):
        checkpoint = action_history.create_checkpoint(
            exp,
            title="Быстрая контрольная точка",
            config=_config_snapshot(config, exp),
            note="Создано из боковой панели",
        )
        st.sidebar.success(f"Сохранено: {checkpoint.name}")
    return exp


def _render_source_check(exp: Path) -> None:
    st.markdown("#### Проверка источника данных")
    report_path = exp / "export" / "data_source_check.json"

    if st.button(
        "Сверить конфигурацию, manifest и среду запуска",
        use_container_width=True,
        disabled=not exp.exists(),
    ):
        with st.spinner("Проверка фактических путей изображений..."):
            report = build_source_report(
                experiment_dir=exp,
                config_path=core.FINAL_PROFILE_PATH,
            )
            outputs = write_source_report(report, exp / "export")
            action_history.append_event(
                exp,
                "data_source_check",
                "Проверен источник данных",
                (
                    f"status={report.get('status')}; "
                    f"config={report.get('config_images_dir')}; "
                    f"run={report.get('run_images_dir')}; "
                    f"manifest={report.get('manifest_common_root')}"
                ),
                outputs["json"],
            )

        if report.get("status") == "ok":
            st.success("Конфигурация, manifest и среда запуска согласованы.")
        elif report.get("status") == "error":
            st.error("Обнаружено расхождение источника данных. Откройте отчёт ниже.")
        else:
            st.warning("Проверка завершена с предупреждениями.")

    existing = core._read_json(report_path)
    if existing:
        status = str(existing.get("status", "warning"))
        summary = (
            f"Статус: {status}; изображений: "
            f"{existing.get('manifest_images_count', 0)}; "
            f"manifest: {existing.get('manifest_common_root', '')}"
        )
        if status == "ok":
            st.success(summary)
        elif status == "error":
            st.error(summary)
        else:
            st.warning(summary)
        with st.expander("Детали проверки", expanded=False):
            st.json(existing)
    elif not exp.exists():
        st.info("Сначала укажите существующую папку результатов.")
    else:
        st.caption("Отчёт ещё не сформирован.")


def _render_start(exp: Path) -> None:
    st.subheader("Старт")
    st.info(
        "Интерфейс помогает пройти полный контур: обзор результатов, фрагменты, "
        "идентификация, пользовательские фото, ручная проверка, коррекция SKU-галереи, "
        "сравнение до/после и экспорт."
    )
    paths = core.experiment_paths(exp) if hasattr(core, "experiment_paths") else {}
    if paths:
        cols = st.columns(4)
        checks = [
            ("Файлы", paths.get("manifest")),
            ("Фрагменты", paths.get("crops")),
            ("Идентификация", paths.get("results")),
            ("Правки", paths.get("edits")),
        ]
        for col, (label, path) in zip(cols, checks):
            with col:
                ok = path is not None and core._p(path).exists()
                st.metric(label, "готово" if ok else "нет")

    _render_source_check(exp)

    st.markdown(
        """
#### Рекомендуемый порядок
1. **Обзор** — убедиться, что файлы найдены.
2. **Параметры** — проверить профиль и фактический источник данных.
3. **Фрагменты** — проверить качество вырезанных товаров.
4. **Идентификация** — отфильтровать статусы и найти спорные случаи.
5. **Свои фото** — запустить детекцию и идентификацию на пользовательской папке без разметки.
6. **Ручная проверка** — подтвердить или изменить назначение SKU.
7. **Коррекция SKU** — улучшить автоматически сформированную галерею.
8. **История** — сохранить контрольную точку после важных действий.
9. **До/после** — сравнить исходный и скорректированный результат.
10. **Экспорт** — собрать ZIP-архив с материалами.
"""
    )


def _render_selected_sku_with_history(exp: Path) -> None:
    source = core.result_source(exp)
    df = core._read_csv(source)
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
    selected = st.multiselect(
        "SKU для показа",
        options,
        default=options[: min(5, len(options))],
    )
    include_unknown = st.checkbox(
        "Включать unknown, где выбранный SKU встречается в top-k",
        value=False,
    )
    max_rows = st.number_input(
        "Максимум query-фрагментов на SKU",
        min_value=1,
        max_value=500,
        value=40,
    )

    if selected:
        preview = df[df["sku_id"].astype(str).isin(selected)].copy()
        view_cols = [
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
            preview[view_cols].head(200),
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
                experiment_dir=exp,
                selected_skus=selected,
                output_dir=exp / "selected_sku_demo",
                results_csv=source,
                max_rows_per_sku=int(max_rows),
                include_unknown_similar=include_unknown,
            )
        action_history.append_event(
            exp,
            "selected_sku_export",
            "Собран набор SKU",
            ", ".join(selected),
            outputs.get("summary_json"),
        )
        st.success("Набор выбранных SKU собран.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{core._rel(core._p(path))}`")


def _render_export_with_history(exp: Path) -> None:
    output = core._p(
        st.text_input(
            "Путь к ZIP-архиву",
            value=str(exp / "export" / "demo_artifacts.zip"),
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
    if st.button("Собрать ZIP-архив материалов", use_container_width=True):
        with st.spinner("Формирование архива..."):
            outputs = build_defense_export_zip(
                experiment_dir=exp,
                output_zip=output,
                include_visualizations=include_visuals,
                visualized_limit_per_dir=int(visual_limit),
            )
        action_history.append_event(
            exp,
            "export",
            "Собран ZIP-архив",
            str(output),
            outputs.get("zip"),
        )
        st.success("ZIP-архив материалов сформирован.")
        for name, path in outputs.items():
            st.write(f"- {name}: `{core._rel(core._p(path))}`")

    core._download_file(
        output,
        "Скачать ZIP-архив",
        "application/zip",
        "history_download_demo_zip",
    )


def _render_history(exp: Path, config: Dict[str, Any]) -> None:
    st.subheader("История")
    st.caption("Сохраняйте контрольные точки и просматривайте журнал действий.")
    st.info(
        "Контрольная точка хранит конфигурацию и путь к каталогу результатов. "
        "Она не копирует и не откатывает все файлы автоматически."
    )

    title = st.text_input("Название контрольной точки", value="Проверенный этап")
    note = st.text_area("Комментарий", value="", height=90)
    if st.button(
        "Сохранить контрольную точку",
        type="primary",
        use_container_width=True,
    ):
        checkpoint = action_history.create_checkpoint(
            exp,
            title=title,
            note=note,
            config=_config_snapshot(config, exp),
            extra={"active_page": "История"},
        )
        st.success(f"Контрольная точка сохранена: {checkpoint.name}")

    checkpoints = action_history.list_checkpoints(exp)
    if checkpoints:
        selected = st.selectbox(
            "Сохранённые контрольные точки",
            checkpoints,
            format_func=lambda path: path.name,
        )
        data = action_history.read_checkpoint(selected)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write(
                {
                    "created_at": data.get("created_at"),
                    "title": data.get("title"),
                    "note": data.get("note"),
                }
            )
        with c2:
            if st.button("Открыть этот этап", use_container_width=True):
                target = data.get("experiment_dir") or str(exp)
                st.session_state["demo_experiment_dir"] = target
                st.success(f"Открыт каталог: {target}")
                st.rerun()
        with st.expander("Содержимое контрольной точки", expanded=False):
            st.json(data)
    else:
        st.info("Контрольных точек пока нет.")

    events = action_history.read_events(exp)
    if events:
        st.markdown("#### События")
        st.dataframe(
            pd.DataFrame(events).tail(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Событий пока нет.")


def _render_review(exp: Path, config: Dict[str, Any]) -> None:
    review_config = dict(config)
    review_config.setdefault("full_photo_identification", {})["out_dir"] = str(exp)
    render_review_with_history(exp, review_config, page_identification_review)


def main() -> None:
    st.set_page_config(
        page_title="Демо анализа полочных сцен",
        page_icon="🧰",
        layout="wide",
    )
    config = core._read_yaml(core.CONFIG_PATH)
    exp = _sidebar(config)

    st.title("🧰 Демо анализа полочных сцен")
    st.caption(
        "Просмотр результатов, запуск на пользовательских фото, ручная проверка, "
        "коррекция SKU-галереи, история действий и экспорт материалов."
    )

    tabs = st.tabs(
        [
            "Старт",
            "Обзор",
            "Параметры",
            "Фрагменты",
            "Идентификация",
            "Свои фото",
            "Ручная проверка",
            "Коррекция SKU",
            "История",
            "До/после",
            "Выбор SKU",
            "Экспорт",
            "FAQ",
        ]
    )
    with tabs[0]:
        _render_start(exp)
    with tabs[1]:
        core._render_overview(exp)
    with tabs[2]:
        core._render_profile(config)
    with tabs[3]:
        core._render_crops(exp)
    with tabs[4]:
        core._render_identification_table(exp)
    with tabs[5]:
        page_user_photos(config, exp)
    with tabs[6]:
        _render_review(exp, config)
    with tabs[7]:
        page_demo_sku_correction(config, exp)
    with tabs[8]:
        _render_history(exp, config)
    with tabs[9]:
        core._render_before_after(exp)
    with tabs[10]:
        _render_selected_sku_with_history(exp)
    with tabs[11]:
        _render_export_with_history(exp)
    with tabs[12]:
        core._render_faq()


if __name__ == "__main__":
    main()
