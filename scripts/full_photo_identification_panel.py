from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from control_panel import save_config
from control_panel_wsl import python_command, use_wsl_runtime
from panel_progress import CommandStep, run_steps_with_progress
from ui_settings import is_advanced


ROOT = Path(__file__).resolve().parents[1]
PHOTO_MODEL_LABELS = {
    "yolo": "YOLO",
    "yolo_seg": "YOLO-Seg",
    "rtdetr": "RT-DETR-L",
    "frcnn": "Faster R-CNN",
}
GALLERY_BUILD_MODE_LABELS = {
    "greedy": "Greedy deduplication",
    "cluster": "Clustered provisional SKU",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

FULL_PHOTO_PRESETS = {
    "fast": {
        "label": "Быстрый",
        "description": "Минимальный прогон для проверки работоспособности.",
        "limit": 60,
        "gallery_count": 30,
        "query_count": 30,
        "max_sku": 50,
        "threshold": 0.65,
        "top_k": 3,
        "visualize_limit": 20,
        "progress_every": 50,
        "gallery_build_mode": "greedy",
        "dedup_threshold": 0.86,
        "max_refs_per_sku": 3,
        "padding": 0.05,
        "min_score": 0.35,
        "min_width": 20,
        "min_height": 20,
        "enable_uncertain_status": True,
        "ambiguity_margin": 0.03,
    },
    "balanced": {
        "label": "Сбалансированный",
        "description": "Рекомендуемый режим для основной проверки качества.",
        "limit": 0,
        "gallery_count": 120,
        "query_count": 140,
        "max_sku": 150,
        "threshold": 0.65,
        "top_k": 5,
        "visualize_limit": 100,
        "progress_every": 100,
        "gallery_build_mode": "greedy",
        "dedup_threshold": 0.82,
        "max_refs_per_sku": 10,
        "padding": 0.05,
        "min_score": 0.35,
        "min_width": 20,
        "min_height": 20,
        "enable_uncertain_status": True,
        "ambiguity_margin": 0.03,
    },
    "quality": {
        "label": "Качественный",
        "description": "Более полный режим: больше gallery/query и больше refs на SKU.",
        "limit": 0,
        "gallery_count": 160,
        "query_count": 140,
        "max_sku": 200,
        "threshold": 0.65,
        "top_k": 5,
        "visualize_limit": 150,
        "progress_every": 100,
        "gallery_build_mode": "greedy",
        "dedup_threshold": 0.82,
        "max_refs_per_sku": 15,
        "padding": 0.05,
        "min_score": 0.35,
        "min_width": 20,
        "min_height": 20,
        "enable_uncertain_status": True,
        "ambiguity_margin": 0.03,
    },
}


def _apply_full_photo_preset(full: Dict[str, Any], preset_key: str) -> None:
    preset = FULL_PHOTO_PRESETS.get(preset_key, FULL_PHOTO_PRESETS["balanced"])
    for key, value in preset.items():
        if key in {"label", "description"}:
            continue
        full[key] = value


def _photo_weight_path(config: Dict[str, Any], model: str) -> str:
    weights = config.get("weights", {})
    if model == "yolo_seg":
        return str(weights.get("yolo_seg", weights.get("yolo", "")))
    return str(weights.get(model, weights.get("yolo", "")))


def _shuffle_implied_by_paths(full: Dict[str, Any]) -> bool:
    paths = [
        str(full.get("out_dir", "")),
        str(full.get("gallery_dir", "")),
        str(full.get("gallery_csv", "")),
    ]
    return any("shuffle" in path.lower() or "seed" in path.lower() for path in paths)


def _safe_path(raw: str | Path) -> Path:
    return Path(str(raw).strip().strip('"'))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 20_000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...текст сокращён для отображения в панели..."
    return text


def _existing_images(path: Path, limit: int) -> List[Path]:
    if not path.exists():
        return []
    images = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return images[: max(0, limit)]


def _build_full_photo_args(config: Dict[str, Any]) -> List[str]:
    runtime = config.setdefault("runtime", {})
    full = config.setdefault("full_photo_identification", {})
    model = str(full.get("model", "yolo"))
    gallery_build_mode = str(full.get("gallery_build_mode", "greedy"))

    args: List[str] = [
        "--model", model,
        "--weights", _photo_weight_path(config, model),
        "--out-dir", str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")),
        "--gallery-dir", str(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")),
        "--gallery-csv", str(full.get("gallery_csv", "D:/1Diplom/sku_gallery_full/gallery.csv")),
        "--limit", str(full.get("limit", 100)),
        "--gallery-count", str(full.get("gallery_count", 30)),
        "--query-count", str(full.get("query_count", 70)),
        "--gallery-limit", str(full.get("gallery_limit", 0)),
        "--query-limit", str(full.get("query_limit", 0)),
        "--conf", str(runtime.get("conf", 0.25)),
        "--imgsz", str(runtime.get("imgsz", 640)),
        "--max-sku", str(full.get("max_sku", 50)),
        "--min-score", str(full.get("min_score", 0.35)),
        "--min-width", str(full.get("min_width", 20)),
        "--min-height", str(full.get("min_height", 20)),
        "--padding", str(full.get("padding", 0.05)),
        "--prefix", str(full.get("prefix", "sku_demo_")),
        "--dedup-threshold", str(full.get("dedup_threshold", 0.86)),
        "--max-refs-per-sku", str(full.get("max_refs_per_sku", 3)),
        "--gallery-build-mode", gallery_build_mode,
        "--cluster-merge-threshold", str(full.get("cluster_merge_threshold", 0.82)),
        "--cluster-strong-merge-threshold", str(full.get("cluster_strong_merge_threshold", 0.88)),
        "--cluster-min-similarity", str(full.get("cluster_min_similarity", 0.72)),
        "--cluster-pair-report-threshold", str(full.get("cluster_pair_report_threshold", 0.75)),
        "--cluster-max-candidates", str(full.get("cluster_max_candidates", 0)),
        "--threshold", str(full.get("threshold", 0.65)),
        "--thresholds", str(full.get("thresholds", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")),
        "--top-k", str(full.get("top_k", 3)),
        "--visualize-limit", str(full.get("visualize_limit", 100)),
        "--progress-every", str(full.get("progress_every", 10)),
    ]

    gallery_images_dir = str(full.get("gallery_images_dir", "")).strip()
    query_images_dir = str(full.get("query_images_dir", "")).strip()
    if gallery_images_dir or query_images_dir:
        args.extend(["--gallery-images-dir", gallery_images_dir, "--query-images-dir", query_images_dir])
    else:
        args.extend(["--images-dir", str(full.get("images_dir", "D:/1Diplom/data/raw/d2s_full/images"))])

    device = str(runtime.get("device", "")).strip()
    if device:
        args.extend(["--device", device])
    gt_csv = str(full.get("gt_csv", "")).strip()
    if gt_csv:
        args.extend(["--gt-csv", gt_csv])

    shuffle_enabled = bool(full.get("shuffle", False)) or _shuffle_implied_by_paths(full)
    if shuffle_enabled:
        args.append("--shuffle")
        args.extend(["--seed", str(full.get("seed", 42))])

    if gallery_build_mode == "greedy" and not bool(full.get("deduplicate_gallery", True)):
        args.append("--no-deduplicate-gallery")
    if bool(full.get("bbox_only", False)):
        args.append("--bbox-only")
    if bool(full.get("keep_old_demo", False)):
        args.append("--keep-old-demo")
    if bool(full.get("resume", True)):
        args.append("--resume")
    if bool(full.get("skip_existing", True)):
        args.append("--skip-existing")
    if bool(full.get("no_visualize_inference", False)):
        args.append("--no-visualize-inference")
    return args


def _build_existing_photo_args(config: Dict[str, Any]) -> List[str]:
    full = config.setdefault("full_photo_identification", {})
    args: List[str] = [
        "--out-dir", str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")),
        "--gallery-dir", str(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")),
        "--gallery-csv", str(full.get("gallery_csv", "D:/1Diplom/sku_gallery_full/gallery.csv")),
        "--threshold", str(full.get("threshold", 0.65)),
        "--ambiguity-margin", str(full.get("ambiguity_margin", 0.03)),
        "--thresholds", str(full.get("thresholds", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")),
        "--top-k", str(full.get("top_k", 3)),
        "--padding", str(full.get("padding", 0.05)),
        "--visualize-limit", str(full.get("visualize_limit", 100)),
        "--progress-every", str(full.get("progress_every", 10)),
    ]
    query_predictions_json = str(full.get("query_predictions_json", "")).strip()
    if query_predictions_json:
        args.extend(["--query-predictions-json", query_predictions_json])
    gt_csv = str(full.get("gt_csv", "")).strip()
    if gt_csv:
        args.extend(["--gt-csv", gt_csv])
    cache_dir = str(full.get("feature_cache_dir", "")).strip()
    if cache_dir:
        args.extend(["--cache-dir", cache_dir])
    if bool(full.get("enable_uncertain_status", True)):
        args.append("--enable-uncertain-status")
    if bool(full.get("bbox_only", False)):
        args.append("--bbox-only")
    return args


def _render_full_photo_settings(config: Dict[str, Any]) -> None:
    full = config.setdefault("full_photo_identification", {})
    advanced = is_advanced(config, page_key="actions")

    st.markdown("#### Настройки идентификации")

    if not advanced:
        st.caption(
            "В рекомендованном режиме показаны только основные поля. "
            "Пороги, min crop, cache, cluster-настройки и технические флаги скрыты."
        )

        preset_options = list(FULL_PHOTO_PRESETS.keys())
        current_preset = str(full.get("recommended_preset", "balanced"))
        if current_preset not in preset_options:
            current_preset = "balanced"

        selected_preset = st.selectbox(
            "Профиль качества",
            preset_options,
            index=preset_options.index(current_preset),
            format_func=lambda key: FULL_PHOTO_PRESETS[key]["label"],
            key="full_photo_recommended_preset",
        )
        full["recommended_preset"] = selected_preset
        _apply_full_photo_preset(full, selected_preset)

        st.info(FULL_PHOTO_PRESETS[selected_preset]["description"])

        c1, c2 = st.columns(2)
        with c1:
            model_options = list(PHOTO_MODEL_LABELS.keys())
            current_model = str(full.get("model", "yolo"))
            full["model"] = st.selectbox(
                "Модель",
                model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                format_func=lambda x: PHOTO_MODEL_LABELS[x],
                key="full_photo_model_simple",
            )
            full["images_dir"] = st.text_input(
                "Папка изображений",
                value=str(full.get("images_dir", "D:/1Diplom/data/raw/d2s_full/images")),
                key="full_photo_images_dir_simple",
            )
            full["out_dir"] = st.text_input(
                "Папка результатов",
                value=str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")),
                key="full_photo_out_dir_simple",
            )
        with c2:
            full["gallery_dir"] = st.text_input(
                "Папка SKU-галереи",
                value=str(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")),
                key="full_photo_gallery_dir_simple",
            )
            full["gallery_csv"] = st.text_input(
                "gallery.csv",
                value=str(full.get("gallery_csv", "D:/1Diplom/sku_gallery_full/gallery.csv")),
                key="full_photo_gallery_csv_simple",
            )
            full["query_predictions_json"] = st.text_input(
                "Query predictions.json, если уже есть",
                value=str(full.get("query_predictions_json", "")),
                key="full_photo_query_predictions_json_simple",
                help="Если пусто, будет использован <out_dir>/03_query_inference/predictions.json",
            )

        with st.expander("Какие параметры применит выбранный профиль", expanded=False):
            preset = FULL_PHOTO_PRESETS[selected_preset]
            st.write(
                {
                    "limit": preset["limit"],
                    "gallery_count": preset["gallery_count"],
                    "query_count": preset["query_count"],
                    "max_sku": preset["max_sku"],
                    "threshold": preset["threshold"],
                    "top_k": preset["top_k"],
                    "dedup_threshold": preset["dedup_threshold"],
                    "max_refs_per_sku": preset["max_refs_per_sku"],
                    "enable_uncertain_status": preset["enable_uncertain_status"],
                    "ambiguity_margin": preset["ambiguity_margin"],
                }
            )

        if st.button("Сохранить рекомендованные настройки", use_container_width=True, key="save_full_photo_settings_simple"):
            save_config(config)
            st.success("Рекомендованные настройки сохранены.")
        return

    with st.expander("Настройки полного эксперимента", expanded=True):
        st.warning(
            "Расширенный режим: доступны технические параметры. "
            "Их изменение влияет на качество, время работы и размер результатов."
        )
        c1, c2 = st.columns(2)
        with c1:
            full["images_dir"] = st.text_input("Полная папка изображений", value=str(full.get("images_dir", "D:/1Diplom/data/raw/d2s_full/images")), key="full_photo_images_dir")
            full["gallery_images_dir"] = st.text_input("Отдельная gallery-папка, опционально", value=str(full.get("gallery_images_dir", "")), key="full_photo_gallery_images_dir")
            full["query_images_dir"] = st.text_input("Отдельная query-папка, опционально", value=str(full.get("query_images_dir", "")), key="full_photo_query_images_dir")
            full["out_dir"] = st.text_input("Папка результатов полного эксперимента", value=str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")), key="full_photo_out_dir")
            full["gallery_dir"] = st.text_input("Папка full SKU-галереи", value=str(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")), key="full_photo_gallery_dir")
            full["gallery_csv"] = st.text_input("Full gallery.csv", value=str(full.get("gallery_csv", "D:/1Diplom/sku_gallery_full/gallery.csv")), key="full_photo_gallery_csv")
            model_options = list(PHOTO_MODEL_LABELS.keys())
            current_model = str(full.get("model", "yolo"))
            full["model"] = st.selectbox("Модель полного эксперимента", model_options, index=model_options.index(current_model) if current_model in model_options else 0, format_func=lambda x: PHOTO_MODEL_LABELS[x], key="full_photo_model")
        with c2:
            full["limit"] = st.number_input("Общий limit, 0 — все", 0, 1_000_000, int(full.get("limit", 100)), key="full_photo_limit")
            full["gallery_count"] = st.number_input("Gallery images", 1, 1_000_000, int(full.get("gallery_count", 30)), key="full_photo_gallery_count")
            full["query_count"] = st.number_input("Query images, 0 — все оставшиеся", 0, 1_000_000, int(full.get("query_count", 70)), key="full_photo_query_count")
            full["max_sku"] = st.number_input("Максимум demo SKU", 1, 10_000, int(full.get("max_sku", 50)), key="full_photo_max_sku")
            full["threshold"] = st.slider("Порог SKU matching", 0.0, 1.0, float(full.get("threshold", 0.65)), 0.01, key="full_photo_threshold")
            full["enable_uncertain_status"] = st.checkbox("Включить статус matched_uncertain", value=bool(full.get("enable_uncertain_status", True)), key="full_photo_enable_uncertain_status")
            full["ambiguity_margin"] = st.slider("Ambiguity margin", 0.0, 0.20, float(full.get("ambiguity_margin", 0.03)), 0.005, key="full_photo_ambiguity_margin")
            full["thresholds"] = st.text_input("Пороги для threshold analysis", value=str(full.get("thresholds", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")), key="full_photo_thresholds")
            full["top_k"] = st.number_input("Top-k", 1, 50, int(full.get("top_k", 3)), key="full_photo_top_k")
            full["visualize_limit"] = st.number_input("Лимит визуализаций", 0, 10_000, int(full.get("visualize_limit", 100)), key="full_photo_visualize_limit")
            full["progress_every"] = st.number_input("Прогресс каждые N фото", 1, 10_000, int(full.get("progress_every", 10)), key="full_photo_progress_every")
            full["resume"] = st.checkbox("Resume по predictions_partial.jsonl", value=bool(full.get("resume", True)), key="full_photo_resume")
            full["skip_existing"] = st.checkbox("Skip existing predictions.json", value=bool(full.get("skip_existing", True)), key="full_photo_skip_existing")

        c3, c4 = st.columns(2)
        with c3:
            full["min_score"] = st.slider("Min score для gallery crop", 0.0, 1.0, float(full.get("min_score", 0.35)), 0.01, key="full_photo_min_score")
            full["min_width"] = st.number_input("Min crop width", 1, 5000, int(full.get("min_width", 20)), key="full_photo_min_width")
            full["min_height"] = st.number_input("Min crop height", 1, 5000, int(full.get("min_height", 20)), key="full_photo_min_height")
            implied_shuffle = _shuffle_implied_by_paths(full)
            full["shuffle"] = st.checkbox("Случайная воспроизводимая выборка", value=bool(full.get("shuffle", False)) or implied_shuffle, key="full_photo_shuffle")
            full["seed"] = st.number_input("Seed для случайной выборки", 0, 1_000_000, int(full.get("seed", 42)), key="full_photo_seed")
            if implied_shuffle and not bool(full.get("shuffle", False)):
                st.warning("В названии папок есть `shuffle/seed`, поэтому команда будет запущена с `--shuffle --seed`, даже если настройка не была сохранена ранее.")
        with c4:
            mode_options = list(GALLERY_BUILD_MODE_LABELS.keys())
            current_mode = str(full.get("gallery_build_mode", "greedy"))
            full["gallery_build_mode"] = st.selectbox("Режим формирования demo SKU-галереи", mode_options, index=mode_options.index(current_mode) if current_mode in mode_options else 0, format_func=lambda x: GALLERY_BUILD_MODE_LABELS[x], key="full_photo_gallery_build_mode")
            full["deduplicate_gallery"] = st.checkbox("Объединять похожие crop-ы в один demo SKU", value=bool(full.get("deduplicate_gallery", True)), key="full_photo_deduplicate_gallery")
            full["dedup_threshold"] = st.slider("Порог объединения crop-ов в gallery", 0.0, 1.0, float(full.get("dedup_threshold", 0.86)), 0.01, key="full_photo_dedup_threshold")
            full["max_refs_per_sku"] = st.number_input("Максимум эталонов на один demo SKU", 1, 50, int(full.get("max_refs_per_sku", 3)), key="full_photo_max_refs_per_sku")
            full["padding"] = st.slider("Crop padding", 0.0, 0.5, float(full.get("padding", 0.05)), 0.01, key="full_photo_padding")
            full["prefix"] = st.text_input("SKU prefix", value=str(full.get("prefix", "sku_demo_")), key="full_photo_prefix")
            full["keep_old_demo"] = st.checkbox("Не удалять старые sku_demo_*", value=bool(full.get("keep_old_demo", False)), key="full_photo_keep_old")
            full["bbox_only"] = st.checkbox("Не использовать masks", value=bool(full.get("bbox_only", False)), key="full_photo_bbox_only")
            full["no_visualize_inference"] = st.checkbox("Не сохранять визуализации инференса", value=bool(full.get("no_visualize_inference", False)), key="full_photo_no_visualize_inference")

        with st.expander("Clustered gallery: параметры объединения provisional SKU", expanded=False):
            st.caption("Эти настройки используются только при режиме `Clustered provisional SKU`.")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                full["cluster_merge_threshold"] = st.slider("Merge threshold", 0.0, 1.0, float(full.get("cluster_merge_threshold", 0.82)), 0.01, key="full_photo_cluster_merge_threshold")
                full["cluster_strong_merge_threshold"] = st.slider("Strong merge threshold", 0.0, 1.0, float(full.get("cluster_strong_merge_threshold", 0.88)), 0.01, key="full_photo_cluster_strong_merge_threshold")
            with cc2:
                full["cluster_min_similarity"] = st.slider("Min cluster similarity", 0.0, 1.0, float(full.get("cluster_min_similarity", 0.72)), 0.01, key="full_photo_cluster_min_similarity")
                full["cluster_pair_report_threshold"] = st.slider("Pair report threshold", 0.0, 1.0, float(full.get("cluster_pair_report_threshold", 0.75)), 0.01, key="full_photo_cluster_pair_report_threshold")
            with cc3:
                full["cluster_max_candidates"] = st.number_input("Max provisional candidates, 0 — auto", 0, 100_000, int(full.get("cluster_max_candidates", 0)), key="full_photo_cluster_max_candidates")
                st.info("Для SKU110K удобно начать с auto или 1500–2500. Чем больше кандидатов, тем тяжелее pairwise similarity.")

        with st.expander("Дополнительно для быстрого пересчёта идентификации", expanded=False):
            full["query_predictions_json"] = st.text_input("Query predictions.json, опционально", value=str(full.get("query_predictions_json", "")), key="full_photo_query_predictions_json", help="Если пусто, будет использован <out_dir>/03_query_inference/predictions.json")
            full["feature_cache_dir"] = st.text_input("Feature cache dir, опционально", value=str(full.get("feature_cache_dir", "")), key="full_photo_feature_cache_dir", help="Если пусто, будет использован <out_dir>/04_identification/feature_cache")

        if st.button("Сохранить настройки полного эксперимента", use_container_width=True, key="save_full_photo_settings"):
            save_config(config)
            st.success("Настройки полного эксперимента сохранены.")


def _render_summary_cards(summary: Dict[str, Any]) -> None:
    if not summary:
        st.warning("Сводка full_experiment_summary.json пока не найдена. Сначала запусти полный эксперимент.")
        return

    st.markdown("#### Сводка эксперимента")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gallery images", int(summary.get("gallery_images_count", 0) or 0))
    c2.metric("Query images", int(summary.get("query_images_count", 0) or 0))
    c3.metric("Query objects", int(summary.get("query_objects_count", 0) or 0))
    c4.metric("Demo SKU", int(summary.get("created_demo_sku_count", 0) or 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Matched", int(summary.get("matched", 0) or 0))
    c6.metric("Unknown", int(summary.get("unknown", 0) or 0))
    c7.metric("Matched rate", f"{float(summary.get('matched_rate', 0.0) or 0.0):.4f}")
    c8.metric("Avg similarity", f"{float(summary.get('avg_similarity', 0.0) or 0.0):.4f}")

    st.caption("Matched rate — это доля сопоставленных объектов с demo SKU-галереей, а не accuracy по реальным SKU-классам.")


def _render_csv_table(path: Path, title: str, max_rows: int = 300) -> None:
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


def _render_visualized_examples(visualized_dir: Path) -> None:
    st.markdown("#### Примеры визуализации идентификации")
    if not visualized_dir.exists():
        st.info(f"Папка visualized пока не найдена: `{visualized_dir}`")
        return

    limit = st.slider("Сколько примеров показать", 1, 24, 8, key="full_photo_visualized_preview_limit")
    images = _existing_images(visualized_dir, limit)
    if not images:
        st.info("В папке visualized нет изображений для предпросмотра.")
        return

    cols = st.columns(2)
    for index, image_path in enumerate(images):
        with cols[index % 2]:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _render_contact_sheets(contact_sheets_dir: Path) -> None:
    st.markdown("#### Contact sheets финальных SKU")
    if not contact_sheets_dir.exists():
        st.info(f"Папка contact sheets не найдена: `{contact_sheets_dir}`")
        return
    limit = st.slider("Сколько SKU contact sheets показать", 1, 24, 8, key="full_photo_contact_sheet_limit")
    images = _existing_images(contact_sheets_dir, limit)
    if not images:
        st.info("Contact sheets пока не созданы.")
        return
    cols = st.columns(2)
    for index, image_path in enumerate(images):
        with cols[index % 2]:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _render_full_photo_results(config: Dict[str, Any]) -> None:
    full = config.setdefault("full_photo_identification", {})
    out_dir = _safe_path(str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")))
    advanced = is_advanced(config, page_key="actions")

    with st.expander("Результаты последнего полного эксперимента", expanded=True):
        st.caption(f"Папка результатов: `{out_dir}`")
        reports_dir = out_dir / "05_reports"
        demo_dir = out_dir / "02_demo_gallery"
        identification_dir = out_dir / "04_identification"
        visualized_dir = identification_dir / "visualized"

        summary_json = reports_dir / "full_experiment_summary.json"
        summary_md = reports_dir / "full_experiment_summary.md"
        threshold_csv = reports_dir / "threshold_analysis.csv"
        threshold_md = reports_dir / "threshold_analysis.md"
        threshold_plot_png = reports_dir / "threshold_analysis_plot.png"
        demo_report_md = demo_dir / "demo_sku_gallery_report.md"
        demo_items_csv = demo_dir / "demo_sku_gallery_items.csv"
        identification_csv = identification_dir / "identification_results.csv"
        existing_summary_md = reports_dir / "existing_identification_summary.md"
        assignment_report_md = identification_dir / "assignment_uncertainty_report.md"
        assignment_summary_csv = identification_dir / "query_assignment_sku_summary.csv"
        uncertain_csv = identification_dir / "matched_uncertain_candidates.csv"
        suspicious_csv = identification_dir / "suspicious_absorber_sku.csv"
        provisional_csv = demo_dir / "provisional_sku_items.csv"
        similarity_pairs_csv = demo_dir / "sku_similarity_pairs.csv"
        merge_decisions_csv = demo_dir / "sku_merge_decisions.csv"
        clusters_csv = demo_dir / "sku_clusters.csv"
        cluster_summary_csv = demo_dir / "sku_cluster_summary.csv"
        cluster_report_md = demo_dir / "sku_merge_report.md"
        contact_sheets_dir = demo_dir / "cluster_contact_sheets"

        summary = _read_json(summary_json)
        _render_summary_cards(summary)

        if advanced:
            tabs = st.tabs(["Threshold", "Demo gallery", "Cluster merge", "Identification", "Uncertain audit", "Visualized", "Markdown reports"])
        else:
            tabs = st.tabs(["Identification", "Uncertain audit", "Reports"])

        if advanced:
            with tabs[0]:
                if threshold_plot_png.exists():
                    st.image(str(threshold_plot_png), caption="Threshold analysis", use_container_width=True)
                else:
                    st.info(f"График threshold analysis пока не найден: `{threshold_plot_png}`")
                _render_csv_table(threshold_csv, "Таблица threshold analysis")

            with tabs[1]:
                demo_report = _read_text(demo_report_md)
                if demo_report:
                    st.markdown(demo_report)
                else:
                    st.info(f"Отчёт demo gallery пока не найден: `{demo_report_md}`")
                _render_csv_table(demo_items_csv, "Demo SKU refs", max_rows=500)

            with tabs[2]:
                st.caption("Эта вкладка заполняется при режиме `Clustered provisional SKU`.")
                cluster_report = _read_text(cluster_report_md)
                if cluster_report:
                    st.markdown(cluster_report)
                else:
                    st.info(f"Cluster report пока не найден: `{cluster_report_md}`")
                _render_csv_table(provisional_csv, "Provisional SKU items", max_rows=500)
                _render_csv_table(cluster_summary_csv, "Cluster summary", max_rows=500)
                _render_csv_table(similarity_pairs_csv, "Similarity pairs", max_rows=500)
                _render_csv_table(merge_decisions_csv, "Merge decisions", max_rows=500)
                _render_csv_table(clusters_csv, "Final clusters", max_rows=500)
                _render_contact_sheets(contact_sheets_dir)

            identification_tab = tabs[3]
            uncertain_tab = tabs[4]
            visualized_tab = tabs[5]
            reports_tab = tabs[6]
        else:
            identification_tab = tabs[0]
            uncertain_tab = tabs[1]
            reports_tab = tabs[2]
            visualized_tab = None

        with identification_tab:
            _render_csv_table(identification_csv, "Результаты идентификации", max_rows=500)

        with uncertain_tab:
            report = _read_text(assignment_report_md)
            if report:
                st.markdown(report)
            else:
                st.info(f"Assignment uncertainty report пока не найден: `{assignment_report_md}`")
            _render_csv_table(assignment_summary_csv, "SKU assignment summary", max_rows=300)
            _render_csv_table(suspicious_csv, "Suspicious absorber SKU", max_rows=300)
            if advanced:
                _render_csv_table(uncertain_csv, "Matched uncertain candidates", max_rows=500)

        if visualized_tab is not None:
            with visualized_tab:
                _render_visualized_examples(visualized_dir)

        with reports_tab:
            report_items = [
                ("Full experiment summary", summary_md),
                ("Existing identification rerun", existing_summary_md),
                ("Assignment uncertainty", assignment_report_md),
            ]
            if advanced:
                report_items.extend([
                    ("Threshold analysis", threshold_md),
                    ("Demo SKU gallery report", demo_report_md),
                    ("Cluster merge report", cluster_report_md),
                ])
            for title, path in report_items:
                with st.expander(title, expanded=False):
                    text = _read_text(path)
                    if text:
                        st.markdown(text)
                    else:
                        st.info(f"Файл не найден: `{path}`")


def page_full_photo_identification(config: Dict[str, Any]) -> None:
    st.subheader("Полная фото-идентификация gallery/query")
    st.caption("Полноценный сценарий для ВКР: часть изображений формирует demo SKU-галерею, а другая часть используется как query для независимой идентификации.")
    _render_full_photo_settings(config)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Запустить полный pipeline", use_container_width=True, key="run_full_photo_identification"):
            save_config(config)
            cmd = python_command(config, "run_full_photo_identification_pipeline.py", _build_full_photo_args(config))
            run_steps_with_progress(
                [CommandStep(title="Полная фото-идентификация gallery/query", cmd=cmd, cwd=ROOT, description="Идёт split на gallery/query, инференс, сборка demo SKU-галереи и идентификация query-объектов. В логе должны появляться строки PHOTO_PROGRESS и PROGRESS_JSON.", estimated_seconds=None)],
                title="Полная фото-идентификация",
                success_message="Полный эксперимент завершён. Ниже можно посмотреть summary, threshold-график, таблицы и визуализации.",
                failure_message="Ошибка полного эксперимента фото-идентификации",
            )
        if st.button("Быстро пересчитать только идентификацию", use_container_width=True, key="rerun_existing_photo_identification"):
            save_config(config)
            cmd = python_command(config, "run_existing_photo_identification.py", _build_existing_photo_args(config))
            run_steps_with_progress(
                [CommandStep(title="Быстрый пересчёт идентификации", cmd=cmd, cwd=ROOT, description="Переиспользуются существующие query predictions.json, gallery.csv и feature cache. YOLO-инференс заново не запускается.", estimated_seconds=None)],
                title="Быстрый пересчёт идентификации",
                success_message="Идентификация, threshold analysis и assignment uncertainty audit пересчитаны.",
                failure_message="Ошибка быстрого пересчёта идентификации",
            )
    with c2:
        full = config.setdefault("full_photo_identification", {})
        out_dir = str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification"))
        st.info(f"Результаты будут сохранены в: `{out_dir}`")
        st.caption(f"Режим запуска: {'WSL .venv_wsl' if use_wsl_runtime(config) else 'Windows/local .venv'}")
        if is_advanced(config, page_key="actions"):
            st.caption("Для изменения `threshold`, `thresholds`, `top_k` и визуализаций можно использовать быстрый пересчёт. Для изменения `dedup_threshold`, `max_refs_per_sku`, `max_sku` или режима gallery build нужен полный pipeline.")
        else:
            st.caption("В рекомендованном режиме используется профиль качества и безопасная проверка matched_uncertain.")

    _render_full_photo_results(config)
