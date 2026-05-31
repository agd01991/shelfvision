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
    infer_gallery_dir_from_experiment,
)
from ui_settings import is_advanced


ROOT = Path(__file__).resolve().parents[1]


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
        return text[:max_chars] + "\n\n...текст сокращён..."
    return text


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

    if "gallery_build_mode" in df.columns:
        cluster_df = df[df["gallery_build_mode"].astype(str).eq("cluster")].copy()
        if not cluster_df.empty:
            df = cluster_df

    options = [str(value) for value in df["experiment"].tolist()]
    selected = st.selectbox("Эксперимент", options, key="purity_experiment")

    row = df[df["experiment"].astype(str).eq(selected)].iloc[0]
    return _safe_path(str(row.get("out_dir") or results_root / selected))


def _manual_edits_csv(experiment_dir: Path) -> Path:
    return _safe_path(experiment_dir) / "06_manual_gallery" / "manual_cluster_edits.csv"


def _purity_out_dir(experiment_dir: Path) -> Path:
    return _safe_path(experiment_dir) / "07_sku_purity_audit"


def _build_purity_args(
    gallery_dir: Path,
    out_dir: Path,
    own_centroid_threshold: float,
    own_mean_threshold: float,
    other_margin: float,
    min_other_similarity: float,
    max_refs_per_sku: int,
) -> List[str]:
    return [
        "--gallery-dir",
        str(_safe_path(gallery_dir)),
        "--out-dir",
        str(_safe_path(out_dir)),
        "--own-centroid-threshold",
        str(own_centroid_threshold),
        "--own-mean-threshold",
        str(own_mean_threshold),
        "--other-margin",
        str(other_margin),
        "--min-other-similarity",
        str(min_other_similarity),
        "--max-refs-per-sku",
        str(max_refs_per_sku),
    ]


def _render_ref_cards(candidates: pd.DataFrame, max_items: int = 40) -> List[str]:
    selected_refs: List[str] = []

    if candidates.empty:
        st.info("Для выбранного SKU нет ref-кандидатов.")
        return selected_refs

    rows = list(candidates.head(max_items).iterrows())
    cols = st.columns(4)

    for index, (_, row) in enumerate(rows):
        ref_path = _safe_path(str(row.get("ref_path", "")))
        ref_file = str(row.get("ref_file", ""))

        with cols[index % 4]:
            if ref_path.exists():
                st.image(str(ref_path), caption=ref_file, use_container_width=True)
            else:
                st.caption(ref_file)

            st.caption(
                f"decision: `{row.get('decision', '')}`\n\n"
                f"own: `{_to_float(row.get('own_centroid_similarity')):.3f}` | "
                f"other: `{_to_float(row.get('nearest_other_similarity')):.3f}`"
            )

            if st.checkbox("Вынести в новый SKU", key=f"purity_select_{ref_file}_{index}"):
                selected_refs.append(ref_file)

    return selected_refs


def _render_purity_results(experiment_dir: Path, gallery_dir: Path) -> None:
    out_dir = _purity_out_dir(experiment_dir)

    summary_json = out_dir / "sku_purity_audit_summary.json"
    mixed_csv = out_dir / "mixed_sku_candidates.csv"
    outliers_csv = out_dir / "ref_outlier_candidates.csv"
    ref_purity_csv = out_dir / "sku_ref_purity.csv"
    report_md = out_dir / "sku_purity_audit_report.md"

    summary = _read_json(summary_json)
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKU проверено", _to_int(summary.get("usable_sku_count")))
        c2.metric("Refs проверено", _to_int(summary.get("checked_refs_count")))
        c3.metric("Mixed SKU", _to_int(summary.get("mixed_sku_count")))
        c4.metric("Wrong/outliers", _to_int(summary.get("likely_wrong_sku_count")) + _to_int(summary.get("possible_outliers_count")))

    if report_md.exists():
        with st.expander("Отчёт SKU purity audit", expanded=False):
            st.markdown(_read_text(report_md))

    mixed_df = _read_csv(mixed_csv)
    outliers_df = _read_csv(outliers_csv)

    if mixed_df.empty and outliers_df.empty:
        st.info("Кандидаты на split пока не найдены. Запусти audit или ослабь пороги.")
        full_df = _read_csv(ref_purity_csv)
        if not full_df.empty:
            with st.expander("Полная таблица sku_ref_purity.csv", expanded=False):
                st.dataframe(full_df.head(500), use_container_width=True, hide_index=True)
        return

    if not mixed_df.empty:
        st.markdown("#### Mixed SKU candidates")
        st.dataframe(mixed_df.head(300), use_container_width=True, hide_index=True)

    if not outliers_df.empty:
        st.markdown("#### Ref outlier candidates")
        st.dataframe(outliers_df.head(500), use_container_width=True, hide_index=True)

    sku_options = sorted(outliers_df["sku_id"].astype(str).unique().tolist())
    if not sku_options:
        return

    selected_sku = st.selectbox("SKU для split-проверки", sku_options, key="purity_selected_sku")
    sku_candidates = outliers_df[outliers_df["sku_id"].astype(str).eq(selected_sku)].copy()

    st.markdown(f"#### Проверка refs внутри `{selected_sku}`")
    selected_refs = _render_ref_cards(sku_candidates)

    suggested = ""
    if not sku_candidates.empty:
        suggested = str(sku_candidates.iloc[0].get("suggested_new_sku_id", ""))

    c1, c2 = st.columns(2)
    with c1:
        new_sku_id = st.text_input(
            "Новый SKU ID",
            value=suggested,
            key="purity_new_sku_id",
            placeholder="sku_demo_manual_001",
        )
    with c2:
        comment = st.text_input(
            "Комментарий",
            value="split from SKU purity audit",
            key="purity_split_comment",
        )

    edits_csv = _manual_edits_csv(experiment_dir)
    if st.button("Добавить split-операцию в manual editor", use_container_width=True, key="purity_add_split"):
        if not selected_refs:
            st.warning("Выбери хотя бы один ref для split.")
            return

        append_manual_edit(
            _safe_path(edits_csv),
            ManualGalleryEdit(
                operation="split",
                source_sku_id=selected_sku,
                new_sku_id=new_sku_id.strip(),
                ref_files=";".join(selected_refs),
                comment=comment,
            ),
        )

        st.success(f"Split-операция добавлена в `{edits_csv}`")


def page_sku_purity_audit(config: Dict[str, Any]) -> None:
    st.subheader("Проверка смешанных SKU")
    st.caption(
        "Этот модуль ищет случаи, когда разные товары попали в один `sku_id`, "
        "и помогает добавить split-операции в manual editor."
    )

    night = config.setdefault("night_experiments", {})
    default_root = str(
        night.get("out_dir")
        or night.get("results_root")
        or "D:/1Diplom/shelfvision_results/cluster_compare_sku110k_2026-05-29_23-35-48"
    )

    results_root = _safe_path(
        st.text_input(
            "Папка серии/экспериментов",
            value=default_root,
            key="purity_results_root",
        )
    )

    summary_csv = _safe_path(
        st.text_input(
            "Summary CSV",
            value=str(night.get("summary_csv") or results_root / "night_experiments_summary.csv"),
            key="purity_summary_csv",
        )
    )

    experiment_dir = _select_experiment_from_summary(results_root, summary_csv)
    if experiment_dir is None:
        raw_experiment = st.text_input(
            "Папка конкретного эксперимента",
            value="",
            key="purity_experiment_dir",
        )
        if not raw_experiment.strip():
            st.info("Укажи summary CSV или папку конкретного эксперимента.")
            return
        experiment_dir = _safe_path(raw_experiment)

    st.caption(f"Папка эксперимента: `{experiment_dir}`")

    inferred_gallery = infer_gallery_dir_from_experiment(experiment_dir)
    gallery_dir = _safe_path(
        st.text_input(
            "Source SKU gallery",
            value=str(inferred_gallery or experiment_dir / "02_demo_gallery"),
            key="purity_gallery_dir",
        )
    )

    if not gallery_dir.exists():
        st.warning(f"Source gallery не найдена: `{gallery_dir}`")
        return

    out_dir = _purity_out_dir(experiment_dir)
    st.caption(f"Purity audit out: `{out_dir}`")
    st.caption(f"Manual edits CSV: `{_manual_edits_csv(experiment_dir)}`")

    advanced = is_advanced(config, page_key="actions")

    if advanced:
        with st.expander("Расширенные параметры purity audit", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                own_centroid_threshold = st.slider("Own centroid threshold", 0.30, 0.99, 0.65, 0.01)
                own_mean_threshold = st.slider("Own mean threshold", 0.30, 0.99, 0.60, 0.01)
            with c2:
                other_margin = st.slider("Other margin", 0.00, 0.50, 0.08, 0.01)
                min_other_similarity = st.slider("Min other similarity", 0.30, 0.99, 0.68, 0.01)
            with c3:
                max_refs_per_sku = st.number_input("Max refs per SKU", 1, 500, 50)
    else:
        own_centroid_threshold = 0.65
        own_mean_threshold = 0.60
        other_margin = 0.08
        min_other_similarity = 0.68
        max_refs_per_sku = 50

        st.info(
            "Используются рекомендованные пороги purity audit. "
            "Для изменения порогов включи режим «Для уверенных пользователей»."
        )

    if st.button("Запустить SKU purity audit", use_container_width=True, key="run_sku_purity_audit"):
        cmd = python_command(
            config,
            "run_sku_purity_audit.py",
            _build_purity_args(
                gallery_dir=gallery_dir,
                out_dir=out_dir,
                own_centroid_threshold=float(own_centroid_threshold),
                own_mean_threshold=float(own_mean_threshold),
                other_margin=float(other_margin),
                min_other_similarity=float(min_other_similarity),
                max_refs_per_sku=int(max_refs_per_sku),
            ),
        )

        run_steps_with_progress(
            [
                CommandStep(
                    title="SKU purity audit",
                    cmd=cmd,
                    cwd=ROOT,
                    description="Проверяется чистота SKU-папок и ищутся refs, которые могут относиться к другому товару.",
                    estimated_seconds=None,
                )
            ],
            title="SKU purity audit",
            success_message="SKU purity audit завершён.",
            failure_message="Ошибка SKU purity audit",
        )

    _render_purity_results(experiment_dir, gallery_dir)
