from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from control_panel import save_config
from control_panel_wsl import python_command, use_wsl_runtime
from panel_progress import CommandStep, run_steps_with_progress


ROOT = Path(__file__).resolve().parents[1]
PHOTO_MODEL_LABELS = {
    "yolo": "YOLO",
    "yolo_seg": "YOLO-Seg",
    "rtdetr": "RT-DETR-L",
    "frcnn": "Faster R-CNN",
}


def _photo_weight_path(config: Dict[str, Any], model: str) -> str:
    weights = config.get("weights", {})
    if model == "yolo_seg":
        return str(weights.get("yolo_seg", weights.get("yolo", "")))
    return str(weights.get(model, weights.get("yolo", "")))


def _build_full_photo_args(config: Dict[str, Any]) -> List[str]:
    runtime = config.setdefault("runtime", {})
    full = config.setdefault("full_photo_identification", {})
    model = str(full.get("model", "yolo"))

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
    if bool(full.get("shuffle", False)):
        args.append("--shuffle")
        args.extend(["--seed", str(full.get("seed", 42))])
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


def _render_full_photo_settings(config: Dict[str, Any]) -> None:
    full = config.setdefault("full_photo_identification", {})
    with st.expander("Настройки полного эксперимента", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            full["images_dir"] = st.text_input(
                "Полная папка изображений",
                value=str(full.get("images_dir", "D:/1Diplom/data/raw/d2s_full/images")),
                key="full_photo_images_dir",
            )
            full["gallery_images_dir"] = st.text_input(
                "Отдельная gallery-папка, опционально",
                value=str(full.get("gallery_images_dir", "")),
                key="full_photo_gallery_images_dir",
            )
            full["query_images_dir"] = st.text_input(
                "Отдельная query-папка, опционально",
                value=str(full.get("query_images_dir", "")),
                key="full_photo_query_images_dir",
            )
            full["out_dir"] = st.text_input(
                "Папка результатов полного эксперимента",
                value=str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")),
                key="full_photo_out_dir",
            )
            full["gallery_dir"] = st.text_input(
                "Папка full SKU-галереи",
                value=str(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")),
                key="full_photo_gallery_dir",
            )
            full["gallery_csv"] = st.text_input(
                "Full gallery.csv",
                value=str(full.get("gallery_csv", "D:/1Diplom/sku_gallery_full/gallery.csv")),
                key="full_photo_gallery_csv",
            )
            model_options = list(PHOTO_MODEL_LABELS.keys())
            current_model = str(full.get("model", "yolo"))
            full["model"] = st.selectbox(
                "Модель полного эксперимента",
                model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                format_func=lambda x: PHOTO_MODEL_LABELS[x],
                key="full_photo_model",
            )
        with c2:
            full["limit"] = st.number_input("Общий limit, 0 — все", 0, 1_000_000, int(full.get("limit", 100)), key="full_photo_limit")
            full["gallery_count"] = st.number_input("Gallery images", 1, 1_000_000, int(full.get("gallery_count", 30)), key="full_photo_gallery_count")
            full["query_count"] = st.number_input("Query images, 0 — все оставшиеся", 0, 1_000_000, int(full.get("query_count", 70)), key="full_photo_query_count")
            full["max_sku"] = st.number_input("Максимум demo SKU", 1, 10_000, int(full.get("max_sku", 50)), key="full_photo_max_sku")
            full["threshold"] = st.slider("Порог SKU matching", 0.0, 1.0, float(full.get("threshold", 0.65)), 0.01, key="full_photo_threshold")
            full["thresholds"] = st.text_input(
                "Пороги для threshold analysis",
                value=str(full.get("thresholds", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")),
                key="full_photo_thresholds",
                help="Через запятую. Эти значения попадут в 05_reports/threshold_analysis.csv и .md",
            )
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
            full["shuffle"] = st.checkbox(
                "Случайная воспроизводимая выборка",
                value=bool(full.get("shuffle", False)),
                key="full_photo_shuffle",
                help="Перемешивает список изображений перед split gallery/query. Для ВКР лучше включать и фиксировать seed.",
            )
            full["seed"] = st.number_input("Seed для случайной выборки", 0, 1_000_000, int(full.get("seed", 42)), key="full_photo_seed")
        with c4:
            full["padding"] = st.slider("Crop padding", 0.0, 0.5, float(full.get("padding", 0.05)), 0.01, key="full_photo_padding")
            full["prefix"] = st.text_input("SKU prefix", value=str(full.get("prefix", "sku_demo_")), key="full_photo_prefix")
            full["keep_old_demo"] = st.checkbox("Не удалять старые sku_demo_*", value=bool(full.get("keep_old_demo", False)), key="full_photo_keep_old")
            full["bbox_only"] = st.checkbox("Не использовать masks", value=bool(full.get("bbox_only", False)), key="full_photo_bbox_only")
            full["no_visualize_inference"] = st.checkbox("Не сохранять визуализации инференса", value=bool(full.get("no_visualize_inference", False)), key="full_photo_no_visualize_inference")

        if st.button("Сохранить настройки полного эксперимента", use_container_width=True, key="save_full_photo_settings"):
            save_config(config)
            st.success("Настройки полного эксперимента сохранены.")


def page_full_photo_identification(config: Dict[str, Any]) -> None:
    st.subheader("Полная фото-идентификация gallery/query")
    st.caption(
        "Полноценный сценарий для ВКР: часть изображений формирует demo SKU-галерею, "
        "а другая часть используется как query для независимой идентификации."
    )
    _render_full_photo_settings(config)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Тест full pipeline на текущем limit", use_container_width=True, key="run_full_photo_identification"):
            save_config(config)
            cmd = python_command(config, "run_full_photo_identification_pipeline.py", _build_full_photo_args(config))
            run_steps_with_progress(
                [
                    CommandStep(
                        title="Полная фото-идентификация gallery/query",
                        cmd=cmd,
                        cwd=ROOT,
                        description="Идёт split на gallery/query, инференс, сборка demo SKU-галереи и идентификация query-объектов. В логе должны появляться строки PHOTO_PROGRESS.",
                        estimated_seconds=None,
                    )
                ],
                title="Полная фото-идентификация",
                success_message="Полный эксперимент завершён. Проверь 00_manifest, 04_identification и 05_reports/full_experiment_summary.md.",
                failure_message="Ошибка полного эксперимента фото-идентификации",
            )
    with c2:
        full = config.setdefault("full_photo_identification", {})
        out_dir = str(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification"))
        st.info(f"Результаты будут сохранены в: `{out_dir}`")
        st.caption(f"Режим запуска: {'WSL .venv_wsl' if use_wsl_runtime(config) else 'Windows/local .venv'}")
