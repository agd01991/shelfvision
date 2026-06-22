from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from path_utils import to_current_os_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _is_non_empty_file(path: str | Path | None) -> bool:
    if path is None:
        return False
    path = _p(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def manual_gallery_experiment_dir(experiment_dir: str | Path) -> Path:
    return _p(experiment_dir) / "06_manual_gallery" / "manual_identification"


def active_experiment_dir(experiment_dir: str | Path) -> Path:
    manual_exp = manual_gallery_experiment_dir(experiment_dir)
    if _is_non_empty_file(manual_exp / "04_identification" / "identification_results.csv"):
        return manual_exp
    return _p(experiment_dir)


def active_result_source(experiment_dir: str | Path) -> Path:
    exp = _p(experiment_dir)
    manual_exp = manual_gallery_experiment_dir(exp)
    candidates = [
        manual_exp / "06_manual_identification" / "identification_results_corrected.csv",
        manual_exp / "04_identification" / "identification_results.csv",
        exp / "06_manual_identification" / "identification_results_corrected.csv",
        exp / "04_identification" / "identification_results.csv",
    ]
    for path in candidates:
        if _is_non_empty_file(path):
            return _p(path)
    return exp / "04_identification" / "identification_results.csv"


def active_crops_source(experiment_dir: str | Path) -> Path:
    exp = _p(experiment_dir)
    manual_exp = manual_gallery_experiment_dir(exp)
    manual_crops = manual_exp / "04_identification" / "crops_manifest.csv"
    if _is_non_empty_file(manual_crops):
        return manual_crops
    return exp / "04_identification" / "crops_manifest.csv"


def active_summary_source(experiment_dir: str | Path) -> Path:
    exp = _p(experiment_dir)
    manual_summary = manual_gallery_experiment_dir(exp) / "05_reports" / "existing_identification_summary.json"
    if _is_non_empty_file(manual_summary):
        return manual_summary
    return exp / "05_reports" / "full_experiment_summary.json"


def active_gallery_csv(experiment_dir: str | Path) -> Path | None:
    summary = active_summary_source(experiment_dir)
    if not summary.exists():
        return None
    try:
        import json

        raw = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = raw.get("gallery_csv") or raw.get("output_gallery_csv")
    if not value:
        return None
    path = _p(str(value))
    return path if path.exists() else None


def active_gallery_dir(experiment_dir: str | Path) -> Path | None:
    gallery_csv = active_gallery_csv(experiment_dir)
    if gallery_csv is not None:
        return gallery_csv.parent
    summary = active_summary_source(experiment_dir)
    if not summary.exists():
        return None
    try:
        import json

        raw = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = raw.get("gallery_dir") or raw.get("output_gallery_dir")
    if not value:
        return None
    path = _p(str(value))
    return path if path.exists() else None


def _status_summary(core: Any, df: pd.DataFrame) -> Dict[str, float | int]:
    if hasattr(core, "_status_summary"):
        return core._status_summary(df)
    if df.empty or "sku_status" not in df.columns:
        return {"total": 0, "matched": 0, "matched_uncertain": 0, "unknown": 0, "assigned_rate": 0.0, "manual_edits": 0}
    statuses = df["sku_status"].astype(str)
    total = len(df)
    matched = int((statuses == "matched").sum())
    uncertain = int((statuses == "matched_uncertain").sum())
    unknown = int((statuses == "unknown").sum())
    return {"total": total, "matched": matched, "matched_uncertain": uncertain, "unknown": unknown, "assigned_rate": (matched + uncertain) / total if total else 0.0, "manual_edits": 0}


def _read_csv(core: Any, path: Path) -> pd.DataFrame:
    if hasattr(core, "_read_csv"):
        return core._read_csv(path)
    if not _is_non_empty_file(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _read_json(core: Any, path: Path) -> Dict[str, Any]:
    if hasattr(core, "_read_json"):
        return core._read_json(path)
    if not _is_non_empty_file(path):
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_int(core: Any, value: Any) -> int:
    if hasattr(core, "_safe_int"):
        return core._safe_int(value)
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _render_active_source_notice(source: Path) -> None:
    source = _p(source)
    if "06_manual_gallery" in str(source).replace("\\", "/"):
        st.success(f"Активный результат: идентификация после пересчёта с ручной SKU-галереей: `{source}`")
    elif "06_manual_identification" in str(source).replace("\\", "/"):
        st.success(f"Активный результат: таблица после ручной проверки объектов: `{source}`")
    else:
        st.info(f"Активный результат: исходная таблица идентификации: `{source}`")


def patch_final_demo_app(core: Any) -> None:
    if getattr(core, "_manual_gallery_active_patch_applied", False):
        return

    original_experiment_paths = core.experiment_paths
    original_result_source = core.result_source

    def patched_experiment_paths(experiment_dir: str | Path) -> Dict[str, Path]:
        paths = dict(original_experiment_paths(experiment_dir))
        exp = _p(experiment_dir)
        manual_exp = manual_gallery_experiment_dir(exp)
        paths.update(
            {
                "manual_gallery_experiment": manual_exp,
                "manual_gallery_results": manual_exp / "04_identification" / "identification_results.csv",
                "manual_gallery_crops": manual_exp / "04_identification" / "crops_manifest.csv",
                "manual_gallery_corrected": manual_exp / "06_manual_identification" / "identification_results_corrected.csv",
                "manual_gallery_edits": manual_exp / "06_manual_identification" / "manual_identification_edits.csv",
                "manual_gallery_summary": manual_exp / "05_reports" / "existing_identification_summary.json",
                "active_results": active_result_source(exp),
                "active_crops": active_crops_source(exp),
                "active_summary": active_summary_source(exp),
            }
        )
        return paths

    def patched_result_source(experiment_dir: str | Path) -> Path:
        return active_result_source(experiment_dir)

    def patched_preview_images(experiment_dir: Path, limit: int = 8) -> List[Path]:
        exp = _p(experiment_dir)
        manual_exp = manual_gallery_experiment_dir(exp)
        search_roots = [
            manual_exp / "04_identification" / "visualized",
            exp / "04_identification" / "visualized",
            exp / "03_query_inference" / "visualized",
            exp / "01_gallery_inference" / "visualized",
            exp / "visualized",
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

    def patched_render_crops(experiment_dir: Path) -> None:
        crops_path = active_crops_source(experiment_dir)
        manifest = _read_csv(core, crops_path)
        if manifest.empty:
            st.info("Манифест фрагментов не найден. После полного запуска должен появиться файл 04_identification/crops_manifest.csv.")
            return
        _render_active_source_notice(active_result_source(experiment_dir))
        st.caption(f"Манифест фрагментов: `{crops_path}`")

        c1, c2 = st.columns([1, 1])
        with c1:
            available = sorted(str(value) for value in manifest.get("source_type", pd.Series(dtype=str)).unique() if str(value))
            selected_types = st.multiselect("Источник фрагмента", available, default=available)
        with c2:
            show_limit = st.slider("Количество карточек", min_value=4, max_value=min(100, max(4, len(manifest))), value=min(24, max(4, len(manifest))), step=4)

        filtered = manifest
        if selected_types and "source_type" in filtered.columns:
            filtered = filtered[filtered["source_type"].astype(str).isin(selected_types)]
        st.caption(f"Фрагментов всего: {len(manifest)}; показано: {min(show_limit, len(filtered))}.")
        cols = st.columns(4)
        for card_index, (_, row) in enumerate(filtered.head(show_limit).iterrows()):
            crop_path = _p(str(row.get("crop_path", "")))
            with cols[card_index % 4]:
                if crop_path.exists():
                    score = row.get("score", row.get("detection_score", 0))
                    try:
                        score_value = float(score or 0)
                    except Exception:
                        score_value = 0.0
                    st.image(str(crop_path), caption=f"obj {row.get('object_id', '')} · {row.get('source_type', '')} · score={score_value:.3f}", use_container_width=True)
                else:
                    st.warning(f"Фрагмент не найден: {crop_path}")

    def patched_render_before_after(experiment_dir: Path) -> None:
        paths = patched_experiment_paths(experiment_dir)
        raw = _read_csv(core, paths["results"])
        active_path = active_result_source(experiment_dir)
        active = _read_csv(core, active_path)
        object_edits = _read_csv(core, paths.get("edits", Path("")))
        manual_gallery_edits = _read_csv(core, paths.get("manual_gallery_edits", Path("")))
        cluster_edits = _read_csv(core, _p(experiment_dir) / "06_manual_gallery" / "manual_cluster_edits.csv")

        if raw.empty:
            st.info("Исходная таблица идентификации не найдена.")
            return
        if active.empty:
            active = raw

        raw_stats = _status_summary(core, raw)
        active_stats = _status_summary(core, active)
        st.caption(f"До: `{paths['results']}`")
        st.caption(f"После: `{active_path}`")
        table = pd.DataFrame(
            [
                {"Показатель": "Всего объектов", "До": raw_stats["total"], "После": active_stats["total"]},
                {"Показатель": "Уверенные (matched)", "До": raw_stats["matched"], "После": active_stats["matched"]},
                {"Показатель": "Требуют проверки (matched_uncertain)", "До": raw_stats["matched_uncertain"], "После": active_stats["matched_uncertain"]},
                {"Показатель": "Не определены (unknown)", "До": raw_stats["unknown"], "После": active_stats["unknown"]},
                {"Показатель": "Доля с кандидатом", "До": f"{raw_stats['assigned_rate']:.4f}", "После": f"{active_stats['assigned_rate']:.4f}"},
                {"Показатель": "Правок отдельных объектов", "До": 0, "После": len(object_edits) + len(manual_gallery_edits)},
                {"Показатель": "Операций с SKU-галереей", "До": 0, "После": len(cluster_edits)},
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("После пересчёта ручной SKU-галереи вкладки используют активную таблицу из 06_manual_gallery/manual_identification. Доля назначений не является accuracy без эталонной SKU-разметки.")

    core.experiment_paths = patched_experiment_paths
    core.result_source = patched_result_source
    core._preview_images = patched_preview_images
    core._render_crops = patched_render_crops
    core._render_before_after = patched_render_before_after
    core._manual_gallery_active_patch_applied = True
    core._original_result_source = original_result_source
