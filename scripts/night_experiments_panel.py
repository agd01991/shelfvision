from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from control_panel import save_config
from control_panel_wsl import python_command, use_wsl_runtime
from panel_progress import CommandStep, run_steps_with_progress


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_path(raw: str | Path) -> Path:
    return Path(str(raw).strip().strip('"').strip("'"))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 40_000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...текст сокращён для отображения в панели..."
    return text


def _render_csv_table(path: Path, title: str, max_rows: int = 500) -> None:
    if not path.exists():
        st.info(f"Файл не найден: `{path}`")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Не удалось прочитать `{path}`: {exc}")
        return

    st.markdown(f"#### {title}")
    st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)
    st.caption(f"Файл: `{path}`")


def _existing_images(path: Path, limit: int = 24) -> List[Path]:
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
    return results_root / "night_experiments_summary.csv"


def _render_settings(config: Dict[str, Any]) -> None:
    night = config.setdefault("night_experiments", {})
    with st.expander("Настройки отчётов серии экспериментов", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            night["results_root"] = st.text_input(
                "Папка серии экспериментов",
                value=str(night.get("results_root", "/mnt/d/1Diplom/shelfvision_results/night_sku110k_v2_2026-05-28_00-16-10")),
                key="night_results_root",
                help="Папка, где лежит night_experiments_summary.csv и подпапки экспериментов 01/02/...",
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


def _render_plots(results_root: Path) -> None:
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
    reports = [
        ("Подробный отчёт", results_root / "night_experiments_detailed_report.md"),
        ("Раздел для ВКР", results_root / "vkr_night_experiments_section.md"),
        ("Краткий summary ночного запуска", results_root / "night_experiments_summary.md"),
    ]
    for title, path in reports:
        with st.expander(title, expanded=title == "Раздел для ВКР"):
            text = _read_text(path)
            if text:
                st.markdown(text)
            else:
                st.info(f"Файл не найден: `{path}`")


def _render_visualized_for_best(best_json: Path) -> None:
    payload = _read_json(best_json)
    recommendations = payload.get("recommendations", []) if payload else []
    if not recommendations:
        st.info("Нет рекомендаций для предпросмотра visualized.")
        return

    options = [str(rec.get("experiment", "")) for rec in recommendations if rec.get("experiment")]
    out_dirs = {str(rec.get("experiment", "")): Path(str(rec.get("out_dir", ""))) for rec in recommendations}
    if not options:
        st.info("В recommendations нет out_dir.")
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
    st.caption("Здесь можно посмотреть результаты ночных запусков, ранжирование конфигураций, влияние параметров и готовый текст для ВКР.")
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

    tabs = st.tabs(["Рекомендации", "Ranked", "Влияние параметров", "Графики", "Текст отчётов", "Visualized"])
    with tabs[0]:
        _render_best_config(best_json)
        _render_csv_table(summary_csv, "Исходная summary-таблица", max_rows=100)
    with tabs[1]:
        _render_csv_table(ranked_csv, "Ранжированная таблица экспериментов", max_rows=100)
    with tabs[2]:
        _render_csv_table(impact_csv, "Влияние параметров", max_rows=200)
    with tabs[3]:
        _render_plots(results_root)
    with tabs[4]:
        _render_report_texts(results_root)
    with tabs[5]:
        _render_visualized_for_best(best_json)
