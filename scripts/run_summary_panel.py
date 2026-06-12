from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _first_existing(root: Path, candidates: list[str]) -> Optional[Path]:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_visualized(root: Path, limit: int = 12) -> list[Path]:
    candidates = [
        root / "04_identification" / "visualized",
        root / "04_identification" / "visualized_selected",
        root / "06_manual_gallery" / "manual_identification" / "visualized",
        root / "03_identification" / "visualized",
        root / "visualized",
    ]
    images: list[Path] = []
    for directory in candidates:
        if directory.exists():
            images.extend(path for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    return images[:limit]


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _metric_from_manifest(manifest: Dict[str, Any], key: str, fallback: Any = 0) -> Any:
    return manifest.get(key, fallback)


def collect_summary(run_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    summary_csv = _first_existing(run_dir, ["03_query_inference/summary.csv", "01_inference/summary.csv", "summary.csv"])
    crops_csv = _first_existing(run_dir, ["03_query_crops/crops_manifest.csv", "02_demo_gallery/crops_manifest.csv", "crops_manifest.csv"])
    gallery_csv = _first_existing(run_dir, ["02_demo_gallery/sku_gallery_final/gallery.csv", "02_demo_gallery/gallery.csv", "gallery.csv"])
    results_csv = _first_existing(run_dir, ["04_identification/identification_results.csv", "03_identification/identification_results.csv", "identification_results.csv"])
    validation_json = _read_json(run_dir / "validation_summary.json")

    summary_df = _read_csv(summary_csv)
    crops_df = _read_csv(crops_csv)
    gallery_df = _read_csv(gallery_csv)
    results_df = _read_csv(results_csv)

    statuses = results_df["status"].astype(str) if "status" in results_df.columns else pd.Series(dtype=str)
    matched = int(statuses.eq("matched").sum()) if not statuses.empty else int(_metric_from_manifest(manifest, "matched", 0))
    matched_uncertain = int(statuses.eq("matched_uncertain").sum()) if not statuses.empty else int(_metric_from_manifest(manifest, "matched_uncertain", 0))
    unknown = int(statuses.eq("unknown").sum()) if not statuses.empty else int(_metric_from_manifest(manifest, "unknown", 0))
    total_objects = len(results_df) if not results_df.empty else int(_metric_from_manifest(manifest, "detected_objects", 0))

    if "objects_count" in summary_df.columns:
        detected_objects = int(pd.to_numeric(summary_df["objects_count"], errors="coerce").fillna(0).sum())
    else:
        detected_objects = total_objects

    avg_similarity = _metric_from_manifest(manifest, "avg_similarity", 0.0)
    for col in ["best_similarity", "similarity", "score"]:
        if col in results_df.columns:
            avg_similarity = float(pd.to_numeric(results_df[col], errors="coerce").fillna(0).mean())
            break

    avg_margin = _metric_from_manifest(manifest, "avg_margin", 0.0)
    if "margin" in results_df.columns:
        avg_margin = float(pd.to_numeric(results_df["margin"], errors="coerce").fillna(0).mean())

    assigned = matched + matched_uncertain
    assigned_share = assigned / total_objects if total_objects else 0.0

    return {
        "manifest": manifest,
        "validation": validation_json,
        "summary_csv": summary_csv,
        "crops_csv": crops_csv,
        "gallery_csv": gallery_csv,
        "results_csv": results_csv,
        "summary_df": summary_df,
        "crops_df": crops_df,
        "gallery_df": gallery_df,
        "results_df": results_df,
        "processed_images": len(summary_df) if not summary_df.empty else int(_metric_from_manifest(manifest, "processed_images", 0)),
        "detected_objects": detected_objects,
        "crops_count": len(crops_df) if not crops_df.empty else int(_metric_from_manifest(manifest, "crops_count", 0)),
        "sku_count": int(gallery_df["sku_id"].nunique()) if "sku_id" in gallery_df.columns else int(_metric_from_manifest(manifest, "gallery_items", 0)),
        "gallery_refs": len(gallery_df) if not gallery_df.empty else int(_metric_from_manifest(manifest, "gallery_refs", 0)),
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "assigned_share": assigned_share,
        "avg_similarity": _num(avg_similarity),
        "avg_margin": _num(avg_margin),
        "visualized": _find_visualized(run_dir),
    }


def main() -> None:
    st.set_page_config(page_title="ShelfVision: итоговый отчёт запуска", page_icon="📊", layout="wide")
    st.title("ShelfVision: итоговый отчёт запуска")
    st.caption("Экран для демонстрации результата полного запуска перед защитой ВКР.")

    run_dir_raw = st.text_input("Папка результата", value="results/demo_defense")
    run_dir = Path(run_dir_raw).expanduser()
    if not run_dir.exists():
        st.warning(f"Папка не найдена: {run_dir}")
        return

    data = collect_summary(run_dir)
    validation_summary = data["validation"].get("summary", {}) if isinstance(data["validation"], dict) else {}
    validation_status = validation_summary.get("status", "не проверено")

    st.subheader("Паспорт и проверка")
    c1, c2, c3 = st.columns(3)
    c1.metric("Статус проверки", validation_status)
    c2.metric("Обработано изображений", data["processed_images"])
    c3.metric("Найдено объектов", data["detected_objects"])

    st.subheader("Итоговые показатели")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Вырезанных фрагментов", data["crops_count"])
    c2.metric("SKU в галерее", data["sku_count"])
    c3.metric("Эталонов в галерее", data["gallery_refs"])
    c4.metric("Доля с кандидатом", f"{data['assigned_share'] * 100:.2f}%")

    c5, c6, c7, c8, c9 = st.columns(5)
    c5.metric("matched", data["matched"])
    c6.metric("matched_uncertain", data["matched_uncertain"])
    c7.metric("unknown", data["unknown"])
    c8.metric("Средняя оценка сходства", f"{data['avg_similarity']:.4f}")
    c9.metric("Средний margin", f"{data['avg_margin']:.4f}")

    st.subheader("Ключевые файлы")
    for label, path in [
        ("summary.csv", data["summary_csv"]),
        ("crops_manifest.csv", data["crops_csv"]),
        ("gallery.csv", data["gallery_csv"]),
        ("identification_results.csv", data["results_csv"]),
        ("run_manifest.json", run_dir / "run_manifest.json"),
        ("validation_report.md", run_dir / "validation_report.md"),
    ]:
        st.write(f"- **{label}:** `{path or 'не найден'}`")

    if data["visualized"]:
        st.subheader("Итоговые визуализации")
        cols = st.columns(3)
        for index, image_path in enumerate(data["visualized"]):
            with cols[index % 3]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

    with st.expander("Таблица результатов SKU-сопоставления", expanded=False):
        if data["results_df"].empty:
            st.info("Таблица identification_results.csv не найдена или пуста.")
        else:
            st.dataframe(data["results_df"], use_container_width=True, height=420)


if __name__ == "__main__":
    main()
