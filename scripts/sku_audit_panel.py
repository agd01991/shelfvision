from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from control_panel_wsl import python_command
from panel_progress import CommandStep, run_steps_with_progress
from path_utils import to_current_os_path
from src.identification.manual_gallery_editor import ManualGalleryEdit, append_manual_edit, infer_gallery_dir_from_experiment

ROOT = Path(__file__).resolve().parents[1]


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _csv(path: Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _select_experiment(results_root: Path, summary_csv: Path) -> Path | None:
    df = _csv(summary_csv)
    if df.empty or not {"experiment", "out_dir"}.issubset(df.columns):
        return None
    options = [str(x) for x in df["experiment"].tolist()]
    selected = st.selectbox("Experiment", options, key="sku_audit_experiment")
    row = df[df["experiment"].astype(str).eq(selected)].iloc[0]
    return _p(str(row.get("out_dir") or results_root / selected))


def _audit_args(gallery_dir: Path, out_dir: Path, pair_threshold: float, candidate_threshold: float, top_n: int, sheet_limit: int, max_refs: int) -> List[str]:
    return [
        "--gallery-dir", str(_p(gallery_dir)),
        "--out-dir", str(_p(out_dir)),
        "--pair-report-threshold", str(pair_threshold),
        "--candidate-threshold", str(candidate_threshold),
        "--top-n", str(top_n),
        "--contact-sheet-limit", str(sheet_limit),
        "--max-refs-per-sku", str(max_refs),
    ]


def page_sku_audit(config: Dict[str, Any]) -> None:
    st.subheader("SKU-to-SKU similarity audit")
    st.caption("Find visually similar SKU folders and add merge operations to the manual editor.")

    night = config.setdefault("night_experiments", {})
    default_root = str(night.get("out_dir") or night.get("results_root") or "D:/1Diplom/shelfvision_results/cluster_compare_sku110k_2026-05-29_23-35-48")
    results_root = _p(st.text_input("Experiment series root", value=default_root, key="sku_audit_results_root"))
    summary_csv = _p(st.text_input("Summary CSV", value=str(night.get("summary_csv") or results_root / "night_experiments_summary.csv"), key="sku_audit_summary_csv"))

    experiment_dir = _select_experiment(results_root, summary_csv)
    if experiment_dir is None:
        raw_exp = st.text_input("Experiment dir", value="", key="sku_audit_experiment_dir")
        if not raw_exp.strip():
            st.info("Provide summary CSV or experiment dir.")
            return
        experiment_dir = _p(raw_exp)

    st.caption(f"Experiment dir: `{experiment_dir}`")
    inferred_gallery = infer_gallery_dir_from_experiment(experiment_dir)
    gallery_dir = _p(st.text_input("Source SKU gallery", value=str(inferred_gallery or experiment_dir / "02_demo_gallery"), key="sku_audit_gallery_dir"))
    if not gallery_dir.exists():
        st.warning(f"Gallery not found: `{gallery_dir}`")
        return

    audit_dir = experiment_dir / "07_sku_audit"
    manual_edits_csv = experiment_dir / "06_manual_gallery" / "manual_cluster_edits.csv"
    st.caption(f"Audit dir: `{audit_dir}`")
    st.caption(f"Manual edits CSV: `{manual_edits_csv}`")

    c1, c2, c3 = st.columns(3)
    with c1:
        pair_threshold = st.slider("Pair report threshold", 0.50, 0.99, 0.75, 0.01, key="sku_audit_pair_threshold")
        candidate_threshold = st.slider("Candidate threshold", 0.50, 0.99, 0.82, 0.01, key="sku_audit_candidate_threshold")
    with c2:
        top_n = st.number_input("Top-N pairs", 1, 5000, 200, key="sku_audit_top_n")
        sheet_limit = st.number_input("Contact sheet limit", 0, 500, 80, key="sku_audit_sheet_limit")
    with c3:
        max_refs = st.number_input("Max refs per SKU", 1, 100, 10, key="sku_audit_max_refs")

    if st.button("Run SKU similarity audit", use_container_width=True, key="sku_audit_run"):
        cmd = python_command(
            config,
            "run_sku_similarity_audit.py",
            _audit_args(gallery_dir, audit_dir, float(pair_threshold), float(candidate_threshold), int(top_n), int(sheet_limit), int(max_refs)),
        )
        run_steps_with_progress(
            [CommandStep(title="SKU similarity audit", cmd=cmd, cwd=ROOT, description="Computing similarities between SKU folders.", estimated_seconds=None)],
            title="SKU similarity audit",
            success_message="Audit finished.",
            failure_message="Audit failed.",
        )

    candidates = _csv(audit_dir / "merge_candidates.csv")
    pairs = _csv(audit_dir / "sku_to_sku_similarity.csv")
    if candidates.empty:
        st.info("No merge candidates yet. Run audit first or lower thresholds.")
        if not pairs.empty:
            st.dataframe(pairs.head(300), use_container_width=True, hide_index=True)
        return

    st.markdown("#### Merge candidates")
    st.dataframe(candidates.head(300), use_container_width=True, hide_index=True)
    options = [
        f"{idx}: {row['sku_a']} / {row['sku_b']} | centroid={float(row.get('centroid_similarity', 0) or 0):.3f}"
        for idx, row in candidates.head(300).iterrows()
    ]
    selected = st.selectbox("Candidate", options, key="sku_audit_candidate")
    idx = int(str(selected).split(":", 1)[0])
    row = candidates.loc[idx]
    sku_a = str(row["sku_a"])
    sku_b = str(row["sku_b"])
    target = st.radio("Target SKU", [sku_a, sku_b], horizontal=True, key="sku_audit_target")
    source = sku_b if target == sku_a else sku_a
    comment = st.text_input("Comment", value=f"audit candidate: {sku_a} / {sku_b}", key="sku_audit_comment")
    if st.button("Add merge operation to manual editor", use_container_width=True, key="sku_audit_add_merge"):
        append_manual_edit(_p(manual_edits_csv), ManualGalleryEdit(operation="merge", source_sku_id=source, target_sku_id=target, comment=comment))
        st.success(f"Merge added: {source} -> {target}")
