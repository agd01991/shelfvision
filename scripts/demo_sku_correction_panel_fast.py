from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from demo_sku_correction_panel import (
    _manual_paths,
    _p,
    _render_apply,
    _render_before_after,
    _render_focus_sku,
    _render_journal,
    _render_pair_comparison,
    _render_split,
    _source_gallery_dir,
)
from src.identification.manual_gallery_editor import list_sku_refs, read_manual_edits


@st.cache_data(show_spinner=False)
def _cached_sku_refs(gallery_dir_text: str, marker: int) -> Dict[str, List[str]]:
    del marker
    refs_by_sku = list_sku_refs(_p(gallery_dir_text))
    return {sku: [str(path) for path in refs] for sku, refs in refs_by_sku.items()}


def _gallery_marker(gallery_dir: Path) -> int:
    gallery_dir = _p(gallery_dir)
    candidates = [
        gallery_dir / "gallery.csv",
        gallery_dir.parent / "demo_sku_gallery_summary.json",
        gallery_dir.parent / "manual_gallery_summary.json",
        gallery_dir,
    ]
    values: List[int] = []
    for path in candidates:
        try:
            values.append(int(path.stat().st_mtime_ns))
        except OSError:
            values.append(0)
    return max(values) if values else 0


def _gallery_stats_fast(gallery_dir: Path) -> tuple[Dict[str, List[Path]], int, int]:
    raw = _cached_sku_refs(str(_p(gallery_dir)), _gallery_marker(gallery_dir))
    refs_by_sku = {sku: [_p(path) for path in refs] for sku, refs in raw.items()}
    refs_count = sum(len(refs) for refs in refs_by_sku.values())
    return refs_by_sku, len(refs_by_sku), refs_count


def _active_display_gallery(base_gallery_dir: Path, manual_gallery_dir: Path, manual_gallery_csv: Path) -> Path:
    """Gallery used only for visual editing screens.

    If the user has already built `06_manual_gallery/sku_gallery_manual`, the
    correction screen should show that current manual state, not the original
    automatically generated gallery. Building still uses the original source in
    `_render_apply`, so the full edit log remains reproducible from the base
    experiment.
    """

    manual_gallery_dir = _p(manual_gallery_dir)
    manual_gallery_csv = _p(manual_gallery_csv)
    if manual_gallery_dir.exists() and manual_gallery_csv.exists():
        return manual_gallery_dir
    return _p(base_gallery_dir)


def page_demo_sku_correction(config: Dict[str, Any], experiment_dir: str | Path) -> None:
    exp = _p(experiment_dir)
    st.subheader("Коррекция SKU-галереи")
    st.caption(
        "Визуальное объединение похожих SKU и разделение смешанных SKU-кластеров. "
        "Для ускорения отрисовывается только выбранный раздел, а не все вкладки сразу."
    )

    if not exp.exists():
        st.info("Сначала укажите существующую папку результатов в боковой панели.")
        return

    base_gallery_dir = _source_gallery_dir(exp, config)
    if not base_gallery_dir.exists():
        st.warning(f"Исходная SKU-галерея не найдена: `{base_gallery_dir}`")
        return

    _, edits_csv, manual_gallery_dir, manual_gallery_csv, _ = _manual_paths(exp)
    display_gallery_dir = _active_display_gallery(base_gallery_dir, manual_gallery_dir, manual_gallery_csv)
    refs_by_sku, sku_count, refs_count = _gallery_stats_fast(display_gallery_dir)
    edits = read_manual_edits(edits_csv)
    is_manual_view = display_gallery_dir == _p(manual_gallery_dir)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKU в галерее", sku_count)
    c2.metric("Эталонов", refs_count)
    c3.metric("Ручных операций", len(edits))
    c4.metric("Ручная галерея", "активна" if is_manual_view else "нет")

    st.info(
        "Используйте этот экран для демонстрации экспертной корректировки: "
        "объедините два SKU, если это один товар, или вынесите ошибочные эталоны в новый SKU."
    )
    st.caption(f"Исходная галерея для воспроизводимой сборки: `{base_gallery_dir}`")
    if is_manual_view:
        st.success(f"Отображается текущая ручная галерея: `{display_gallery_dir}`")
    else:
        st.caption(f"Отображается исходная галерея: `{display_gallery_dir}`")
    st.caption(f"Журнал операций: `{edits_csv}`")

    st.session_state.setdefault("demo_focus_candidate_limit", 24)
    sections = ["Улучшить SKU", "Похожие SKU", "Разделить SKU", "Журнал", "Применить", "До/после"]
    section = st.radio(
        "Раздел коррекции SKU",
        sections,
        horizontal=True,
        label_visibility="collapsed",
        key="sku_correction_fast_section",
    )

    if section == "Улучшить SKU":
        _render_focus_sku(exp, edits_csv, refs_by_sku)
    elif section == "Похожие SKU":
        _render_pair_comparison(exp, edits_csv, refs_by_sku)
    elif section == "Разделить SKU":
        _render_split(exp, edits_csv, refs_by_sku)
    elif section == "Журнал":
        _render_journal(exp, edits_csv)
    elif section == "Применить":
        _render_apply(exp, config, base_gallery_dir, edits_csv)
    else:
        _render_before_after(exp)
