from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from control_panel import save_config
from control_panel_wsl import python_command
from panel_progress import CommandStep, run_steps_with_progress
from path_utils import to_current_os_path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PATH_COLUMNS = {"out_dir", "log_file", "weights", "primary_ref"}


def _safe_path(raw: str | Path | None) -> Path:
    return to_current_os_path(raw)


def _path_for_display(path: str | Path | None) -> str:
    return str(to_current_os_path(path))


def _normalize_path_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    for col in PATH_COLUMNS.intersection(view.columns):
        view[col] = view[col].astype(str).map(_path_for_display)
    return view


def _read_json(path: Path) -> Dict[str, Any]:
    path = _safe_path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 40_000) -> str:
    path = _safe_path(path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...текст сокращён для отображения в панели..."
    return text


def _read_csv(path: Path) -> pd.DataFrame:
    path = _safe_path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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


def _format_rate(value: Any) -> str:
    return f"{_to_float(value):.4f}"


def _format_percent(value: Any) -> str:
    return f"{_to_float(value) * 100:.2f}%"


def _render_csv_table(path: Path, title: str, max_rows: int = 500) -> None:
    path = _safe_path(path)
    if not path.exists():
        st.info(f"Файл не найден: `{path}`")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Не удалось прочитать `{path}`: {exc}")
        return

    st.markdown(f"#### {title}")
    st.dataframe(_normalize_path_columns_for_display(df.head(max_rows)), use_container_width=True, hide_index=True)
    st.caption(f"Файл: `{path}`")


def _existing_images(path: Path, limit: int = 24) -> List[Path]:
    path = _safe_path(path)
    if not path.exists():
        return []
    return [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS][:limit]


def _build_report_args(config: Dict[str, Any]) -> List[str]:
    night = config.setdefault("night_experiments", {})
    args: List[str] = []
    results_root = str(night.get("results_root", "")).strip()
    summary_csv = str(night.get("summary_csv", "")).strip()
    out_dir = str(night.get("out_dir", "")).strip()
    top_n = int(night.get("top_n", 16) or 16)

    if results_root:
        args.extend(["--results-root", results_root])
    if summary_csv:
        args.extend(["--summary-csv", summary_csv])
    if out_dir:
        args.extend(["--out-dir", out_dir])
    args.extend(["--top-n", str(top_n)])
    return args


def _guess_summary_csv(results_root: Path) -> Path:
    return _safe_path(results_root) / "night_experiments_summary.csv"


def _render_settings(config: Dict[str, Any]) -> None:
    night = config.setdefault("night_experiments", {})
    with st.expander("Настройки отчётов серии экспериментов", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            night["results_root"] = st.text_input(
                "Папка серии экспериментов",
                value=str(night.get("results_root", "D:/1Diplom/shelfvision_results/night_sku110k_v2_2026-05-28_00-16-10")),
                key="night_results_root",
                help="Можно вводить Windows-путь D:/... или WSL-путь /mnt/d/... Панель сама адаптирует путь для чтения и запуска WSL-команд.",
            )
            night["summary_csv"] = st.text_input(
                "night_experiments_summary.csv, опционально",
                value=str(night.get("summary_csv", "")),
                key="night_summary_csv",
                help="Если пусто, будет использован <results_root>/night_experiments_summary.csv",
            )
        with c2:
            night["out_dir"] = st.text_input(
                "Папка для сгенерированных отчётов, опционально",
                value=str(night.get("out_dir", "")),
                key="night_report_out_dir",
                help="Если пусто, отчёты сохраняются в results_root",
            )
            night["top_n"] = st.number_input(
                "Сколько лучших экспериментов показывать",
                min_value=1,
                max_value=100,
                value=int(night.get("top_n", 16) or 16),
                key="night_top_n",
            )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Сохранить настройки отчётов", use_container_width=True, key="save_night_reports_settings"):
                save_config(config)
                st.success("Настройки отчётов серии экспериментов сохранены.")
        with b2:
            if st.button("Сгенерировать/обновить отчёты", use_container_width=True, key="generate_night_reports"):
                save_config(config)
                cmd = python_command(config, "run_night_experiments_report.py", _build_report_args(config))
                run_steps_with_progress(
                    [
                        CommandStep(
                            title="Генерация отчётов серии SKU110K",
                            cmd=cmd,
                            cwd=ROOT,
                            description="Агрегируются night_experiments_summary.csv, ranked-таблица, влияние параметров, графики и раздел для ВКР.",
                            estimated_seconds=None,
                        )
                    ],
                    title="Отчёты серии экспериментов",
                    success_message="Отчёты серии экспериментов обновлены. Ниже можно посмотреть таблицы, графики и текст для ВКР.",
                    failure_message="Ошибка генерации отчётов серии экспериментов",
                )


def _render_best_config(best_json: Path) -> None:
    best_json = _safe_path(best_json)
    payload = _read_json(best_json)
    recommendations = payload.get("recommendations", []) if payload else []
    if not recommendations:
        st.info(f"Файл рекомендаций пока не найден или пустой: `{best_json}`")
        return

    st.markdown("#### Рекомендованные конфигурации")
    cols = st.columns(min(3, len(recommendations)))
    for index, rec in enumerate(recommendations[:3]):
        with cols[index % len(cols)]:
            st.metric("matched_rate", f"{float(rec.get('matched_rate', 0.0) or 0.0):.4f}")
            st.caption(f"**{rec.get('experiment', '')}**")
            st.caption(str(rec.get("reason", "")))
            st.write(f"unknown_rate: `{float(rec.get('unknown_rate', 0.0) or 0.0):.4f}`")
            st.write(f"avg_similarity: `{float(rec.get('avg_similarity', 0.0) or 0.0):.4f}`")
            st.write(f"gallery_refs: `{rec.get('gallery_refs', '')}`")
            if rec.get("out_dir"):
                st.caption(f"out_dir: `{_path_for_display(rec.get('out_dir'))}`")


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    sortable = df.copy()
    for col in ["matched_rate", "avg_similarity", "gallery_refs"]:
        if col in sortable.columns:
            sortable[col] = pd.to_numeric(sortable[col], errors="coerce").fillna(0.0)
    return sortable.sort_values(["matched_rate", "avg_similarity", "gallery_refs"], ascending=False).iloc[0]


def _render_result_card(title: str, row: pd.Series | None) -> None:
    st.markdown(f"#### {title}")
    if row is None:
        st.info("Нет данных.")
        return
    st.metric("matched_rate", _format_rate(row.get("matched_rate", 0.0)), help=_format_percent(row.get("matched_rate", 0.0)))
    st.caption(f"**{row.get('experiment', '')}**")
    st.write(f"unknown_rate: `{_format_rate(row.get('unknown_rate', 0.0))}`")
    st.write(f"avg_similarity: `{_format_rate(row.get('avg_similarity', 0.0))}`")
    st.write(f"gallery_refs: `{_to_int(row.get('gallery_refs', 0))}`")
    st.write(f"demo_sku: `{_to_int(row.get('created_demo_sku', 0))}`")


def _render_cluster_contact_sheets(experiment_dir: Path, key_prefix: str) -> None:
    experiment_dir = _safe_path(experiment_dir)
    sheets_dir = experiment_dir / "02_demo_gallery" / "cluster_contact_sheets"
    if not sheets_dir.exists():
        st.info(f"Contact sheets не найдены: `{sheets_dir}`")
        return

    limit = st.slider("Сколько contact sheets показать", 1, 30, 10, key=f"{key_prefix}_contact_sheets_limit")
    images = _existing_images(sheets_dir, limit=limit)
    if not images:
        st.info("В contact_sheets нет изображений.")
        return

    cols = st.columns(2)
    for index, image in enumerate(images):
        with cols[index % 2]:
            st.image(str(image), caption=image.name, use_container_width=True)


def _render_cluster_tables(experiment_dir: Path) -> None:
    experiment_dir = _safe_path(experiment_dir)
    demo_dir = experiment_dir / "02_demo_gallery"
    _render_csv_table(demo_dir / "sku_cluster_summary.csv", "Cluster summary", max_rows=300)
    _render_csv_table(demo_dir / "sku_similarity_pairs.csv", "Similarity pairs", max_rows=300)
    _render_csv_table(demo_dir / "sku_merge_decisions.csv", "Merge decisions", max_rows=300)
    _render_csv_table(demo_dir / "sku_clusters.csv", "Final clusters", max_rows=300)


def _render_greedy_vs_cluster(summary_csv: Path, results_root: Path) -> None:
    summary_csv = _safe_path(summary_csv)
    results_root = _safe_path(results_root)
    st.markdown("### Greedy vs Clustered gallery")
    st.caption("Сравнение старого жадного объединения crop-ов и нового режима provisional SKU + clustering.")

    df = _read_csv(summary_csv)
    if df.empty:
        st.warning(f"Summary CSV пустой или не читается: `{summary_csv}`")
        return

    if "gallery_build_mode" not in df.columns:
        st.info("В summary нет колонки `gallery_build_mode`. Это обычная ночная серия без сравнения greedy/cluster.")
        return

    ok_df = df[df.get("status", "").astype(str).str.startswith("ok")].copy()
    if ok_df.empty:
        st.warning("Нет успешных запусков для сравнения.")
        _render_csv_table(summary_csv, "Исходная summary-таблица", max_rows=100)
        return

    for col in ["matched_rate", "unknown_rate", "avg_similarity", "gallery_refs", "created_demo_sku"]:
        if col in ok_df.columns:
            ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce").fillna(0.0)

    greedy_df = ok_df[ok_df["gallery_build_mode"].astype(str).eq("greedy")]
    cluster_df = ok_df[ok_df["gallery_build_mode"].astype(str).eq("cluster")]
    best_overall = _best_row(ok_df)
    best_greedy = _best_row(greedy_df)
    best_cluster = _best_row(cluster_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_result_card("Лучший общий", best_overall)
    with c2:
        _render_result_card("Лучший greedy", best_greedy)
    with c3:
        _render_result_card("Лучший cluster", best_cluster)

    if best_greedy is not None and best_cluster is not None:
        delta_matched = _to_float(best_greedy.get("matched_rate")) - _to_float(best_cluster.get("matched_rate"))
        delta_similarity = _to_float(best_greedy.get("avg_similarity")) - _to_float(best_cluster.get("avg_similarity"))
        delta_refs = _to_int(best_greedy.get("gallery_refs")) - _to_int(best_cluster.get("gallery_refs"))
        if delta_matched >= 0:
            st.success(
                "Текущий вывод: основной финальный режим лучше оставить `greedy`. "
                f"Он выше cluster по matched_rate на `{delta_matched:.4f}`, "
                f"по avg_similarity на `{delta_similarity:.4f}` и даёт на `{delta_refs}` больше gallery refs."
            )
        else:
            st.success(
                "Текущий вывод: cluster-режим обогнал greedy по matched_rate. "
                "Перед фиксацией как основного режима нужно визуально проверить contact sheets."
            )

    st.info(
        "Рекомендация для программы: использовать `greedy` как основной режим идентификации, "
        "а `cluster` оставить как диагностический режим для анализа похожих SKU, contact sheets и отчётов по merge decisions."
    )

    comparison_md = results_root / "cluster_comparison_summary.md"
    comparison_text = _read_text(comparison_md)
    with st.expander("Cluster comparison summary.md", expanded=True):
        if comparison_text:
            st.markdown(comparison_text)
        else:
            st.info(f"Файл не найден: `{comparison_md}`")

    display_cols = [
        "experiment",
        "gallery_build_mode",
        "matched_rate",
        "unknown_rate",
        "avg_similarity",
        "created_demo_sku",
        "gallery_refs",
        "duplicate_refs",
        "cluster_merge_threshold",
        "cluster_strong_merge_threshold",
        "cluster_min_similarity",
        "max_refs_per_sku",
        "out_dir",
    ]
    existing_cols = [col for col in display_cols if col in ok_df.columns]
    st.markdown("#### Таблица сравнения")
    st.dataframe(
        _normalize_path_columns_for_display(ok_df.sort_values(["matched_rate", "avg_similarity"], ascending=False)[existing_cols]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Проверка cluster contact sheets")
    if cluster_df.empty:
        st.info("Cluster-запусков нет.")
        return

    cluster_options = [str(value) for value in cluster_df.sort_values("matched_rate", ascending=False)["experiment"].tolist()]
    selected = st.selectbox("Cluster-эксперимент", cluster_options, key="cluster_compare_selected_experiment")
    selected_row = cluster_df[cluster_df["experiment"].astype(str).eq(selected)].iloc[0]
    experiment_dir = _safe_path(str(selected_row.get("out_dir") or results_root / selected))
    st.caption(f"Папка эксперимента: `{experiment_dir}`")

    sheet_tab, table_tab, report_tab = st.tabs(["Contact sheets", "Cluster tables", "Cluster report"])
    with sheet_tab:
        _render_cluster_contact_sheets(experiment_dir, key_prefix="cluster_compare")
    with table_tab:
        _render_cluster_tables(experiment_dir)
    with report_tab:
        report_path = experiment_dir / "02_demo_gallery" / "sku_merge_report.md"
        text = _read_text(report_path)
        if text:
            st.markdown(text)
        else:
            st.info(f"Файл не найден: `{report_path}`")


def _render_plots(results_root: Path) -> None:
    results_root = _safe_path(results_root)
    plot_paths = [
        results_root / "night_experiments_top_matched_rate.png",
        results_root / "night_experiments_parameter_impact.png",
    ]
    existing = [path for path in plot_paths if path.exists()]
    if not existing:
        st.info("Графики серии экспериментов пока не найдены. Нажми «Сгенерировать/обновить отчёты».")
        return

    for path in existing:
        st.image(str(path), caption=path.name, use_container_width=True)


def _render_report_texts(results_root: Path) -> None:
    results_root = _safe_path(results_root)
    reports = [
        ("Greedy vs Cluster summary", results_root / "cluster_comparison_summary.md"),
        ("Подробный отчёт", results_root / "night_experiments_detailed_report.md"),
        ("Раздел для ВКР", results_root / "vkr_night_experiments_section.md"),
        ("Краткий summary ночного запуска", results_root / "night_experiments_summary.md"),
    ]
    for title, path in reports:
        with st.expander(title, expanded=title in {"Раздел для ВКР", "Greedy vs Cluster summary"}):
            text = _read_text(path)
            if text:
                st.markdown(text)
            else:
                st.info(f"Файл не найден: `{path}`")


def _render_visualized_for_best(best_json: Path, summary_csv: Path | None = None) -> None:
    best_json = _safe_path(best_json)
    payload = _read_json(best_json)
    recommendations = payload.get("recommendations", []) if payload else []

    options: List[str] = []
    out_dirs: Dict[str, Path] = {}
    if recommendations:
        options = [str(rec.get("experiment", "")) for rec in recommendations if rec.get("experiment")]
        out_dirs = {str(rec.get("experiment", "")): _safe_path(str(rec.get("out_dir", ""))) for rec in recommendations}
    elif summary_csv is not None and _safe_path(summary_csv).exists():
        df = _read_csv(summary_csv)
        if not df.empty and {"experiment", "out_dir"}.issubset(df.columns):
            options = [str(value) for value in df["experiment"].head(20).tolist()]
            out_dirs = {str(row["experiment"]): _safe_path(str(row["out_dir"])) for _, row in df.head(20).iterrows()}

    if not options:
        st.info("Нет рекомендаций или summary для предпросмотра visualized.")
        return

    selected = st.selectbox("Эксперимент для предпросмотра", options, key="night_visualized_experiment")
    visualized_dir = out_dirs[selected] / "04_identification" / "visualized"
    limit = st.slider("Сколько картинок показать", 1, 24, 8, key="night_visualized_limit")
    images = _existing_images(visualized_dir, limit)
    if not images:
        st.info(f"Папка visualized не найдена или была очищена: `{visualized_dir}`")
        return

    cols = st.columns(2)
    for index, image in enumerate(images):
        with cols[index % 2]:
            st.image(str(image), caption=image.name, use_container_width=True)


def page_night_experiments_reports(config: Dict[str, Any]) -> None:
    st.subheader("Отчёты по серии экспериментов SKU110K")
    st.caption("Здесь можно посмотреть результаты ночных запусков, greedy vs cluster, ранжирование конфигураций, влияние параметров и готовый текст для ВКР.")
    _render_settings(config)

    night = config.setdefault("night_experiments", {})
    results_root = _safe_path(str(night.get("out_dir") or night.get("results_root") or ""))
    if not str(results_root):
        st.info("Укажи папку серии экспериментов.")
        return

    summary_csv = _safe_path(str(night.get("summary_csv") or _guess_summary_csv(results_root)))
    ranked_csv = results_root / "night_experiments_ranked.csv"
    impact_csv = results_root / "night_experiments_parameter_impact.csv"
    best_json = results_root / "night_experiments_best_config.json"

    st.caption(f"Папка отчётов: `{results_root}`")
    if not summary_csv.exists():
        st.warning(f"Исходная таблица серии не найдена: `{summary_csv}`")
        return

    tabs = st.tabs(["Greedy vs Cluster", "Рекомендации", "Ranked", "Влияние параметров", "Графики", "Текст отчётов", "Visualized"])
    with tabs[0]:
        _render_greedy_vs_cluster(summary_csv, results_root)
    with tabs[1]:
        _render_best_config(best_json)
        _render_csv_table(summary_csv, "Исходная summary-таблица", max_rows=100)
    with tabs[2]:
        _render_csv_table(ranked_csv, "Ранжированная таблица экспериментов", max_rows=100)
    with tabs[3]:
        _render_csv_table(impact_csv, "Влияние параметров", max_rows=200)
    with tabs[4]:
        _render_plots(results_root)
    with tabs[5]:
        _render_report_texts(results_root)
    with tabs[6]:
        _render_visualized_for_best(best_json, summary_csv=summary_csv)
