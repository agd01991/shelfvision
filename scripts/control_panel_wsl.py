from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from control_panel import (
    MODEL_LABELS,
    build_weight_args,
    ensure_config_exists,
    load_config,
    page_downloads,
    page_results,
    resolve_path,
    run_command,
    save_config,
    venv_python,
)
from panel_progress import CommandStep, run_steps_with_progress
from setup_pages import page_setup
from ui_settings import is_advanced, render_settings_mode_switch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "config/shelfvision.yaml"
PHOTO_MODEL_LABELS = {
    "yolo": "YOLO",
    "yolo_seg": "YOLO-Seg",
    "rtdetr": "RT-DETR-L",
    "frcnn": "Faster R-CNN",
}
STREAMLIT_PORTS = {
    "scripts/interface_app.py": 8502,
    "scripts/inference_app.py": 8503,
    "scripts/video_app.py": 8504,
}


def use_wsl_runtime(config: Dict[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("use_wsl_runtime", True))


def wsl_venv_dir(config: Dict[str, Any]) -> str:
    return str(config.get("setup", {}).get("venv_dir_wsl", ".venv_wsl"))


def python_command(config: Dict[str, Any], script: str, args: List[str]) -> List[str]:
    if use_wsl_runtime(config):
        return [sys.executable, "scripts/wsl_runtime.py", "--venv-dir", wsl_venv_dir(config), script, *args]
    return [str(venv_python(config)), script, *args]


def streamlit_port(app_script: str) -> int:
    return int(STREAMLIT_PORTS.get(app_script.replace("\\", "/"), 0) or 0)


def streamlit_url(app_script: str) -> str:
    port = streamlit_port(app_script)
    return f"http://localhost:{port}" if port else "http://localhost:8501"


def streamlit_command(config: Dict[str, Any], app_script: str) -> List[str]:
    extra_args: List[str] = ["streamlit", "run", app_script]
    port = streamlit_port(app_script)
    if port:
        extra_args.extend(["--server.port", str(port), "--server.headless", "true"])
    if use_wsl_runtime(config):
        return python_command(config, "-m", extra_args)
    return [str(venv_python(config)), "-m", *extra_args]


def render_command_result(result) -> None:
    if result.returncode == 0:
        st.success(f"Команда выполнена успешно: returncode={result.returncode}")
    else:
        st.error(f"Команда завершилась с ошибкой: returncode={result.returncode}")
    st.code(result.stdout or "", language="text")


def _launch_background(cmd: List[str]) -> None:
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _video_weight_path(config: Dict[str, Any], video_model: str) -> str:
    weights = config.get("weights", {})
    if video_model == "yolo_seg":
        return str(weights.get("yolo_seg", weights.get("yolo", "")))
    return str(weights.get("yolo", ""))


def _photo_weight_path(config: Dict[str, Any], model: str) -> str:
    weights = config.get("weights", {})
    if model == "yolo_seg":
        return str(weights.get("yolo_seg", weights.get("yolo", "")))
    return str(weights.get(model, weights.get("yolo", "")))


def _run_live_command(title: str, cmd: List[str], description: str, success: str, failure: str) -> None:
    run_steps_with_progress(
        [CommandStep(title=title, cmd=cmd, cwd=ROOT, description=description, estimated_seconds=None)],
        title=title,
        success_message=success,
        failure_message=failure,
    )


def _video_expected_seconds(config: Dict[str, Any]) -> int | None:
    video = config.get("video", {})
    max_frames = int(video.get("max_frames", 0) or 0)
    if max_frames > 0:
        return max(30, max_frames // 2)
    return None


def page_config_wsl(config: Dict[str, Any]) -> Dict[str, Any]:
    st.header("3. Настройки проекта")
    render_settings_mode_switch(config, page_key="settings")
    advanced = is_advanced(config, page_key="settings")

    with st.form("config_form_wsl"):
        runtime = config.setdefault("runtime", {})
        paths = config.setdefault("paths", {})
        weights = config.setdefault("weights", {})
        setup = config.setdefault("setup", {})
        wbf = config.setdefault("wbf", {})
        density = config.setdefault("density", {})
        video = config.setdefault("video", {})
        readiness = config.setdefault("readiness", {})
        sku_gallery = config.setdefault("sku_gallery", {})
        demo_sku_gallery = config.setdefault("demo_sku_gallery", {})
        identification = config.setdefault("identification", {})
        presentation = config.setdefault("presentation_assets", {})

        st.subheader("Режим запуска")
        runtime["use_wsl_runtime"] = st.checkbox("Запускать задачи через WSL .venv_wsl", value=bool(runtime.get("use_wsl_runtime", True)))
        if not advanced:
            st.caption("Технические пути виртуальных окружений, requirements.txt, WBF, tracking и служебные лимиты скрыты. Включи расширенный режим, если нужно менять их вручную.")

        st.subheader("Пути")
        paths["image"] = st.text_input("Одно изображение", value=str(paths.get("image", "")))
        paths["images_dir"] = st.text_input("Папка изображений", value=str(paths.get("images_dir", "")))
        paths["out_dir"] = st.text_input("Папка результатов", value=str(paths.get("out_dir", "results/control_panel")))
        if advanced:
            paths["gt_coco"] = st.text_input("Файл COCO annotations.json", value=str(paths.get("gt_coco", "")))
            paths["gt_yolo_labels"] = st.text_input("Папка YOLO labels", value=str(paths.get("gt_yolo_labels", "")))

        st.subheader("Веса моделей")
        weights["yolo"] = st.text_input("Веса YOLO", value=str(weights.get("yolo", "")))
        weights["yolo_seg"] = st.text_input("Веса YOLO-Seg", value=str(weights.get("yolo_seg", "")))
        weights["rtdetr"] = st.text_input("Веса RT-DETR", value=str(weights.get("rtdetr", "")))
        if advanced:
            weights["frcnn"] = st.text_input("Веса Faster R-CNN", value=str(weights.get("frcnn", "")))

        if advanced:
            st.subheader("Настройка окружения")
            setup["venv_dir"] = st.text_input("Windows/local venv для панели", value=str(setup.get("venv_dir", ".venv")))
            setup["venv_dir_wsl"] = st.text_input("WSL venv для запуска задач", value=str(setup.get("venv_dir_wsl", ".venv_wsl")))
            setup["requirements"] = st.text_input("Файл requirements.txt", value=str(setup.get("requirements", "requirements.txt")))

        st.subheader("Параметры инференса")
        runtime["conf"] = st.slider("Порог confidence", 0.01, 0.95, float(runtime.get("conf", 0.25)), 0.01)
        imgsz_options = [416, 512, 640, 768, 1024]
        imgsz_value = int(runtime.get("imgsz", 640))
        runtime["imgsz"] = st.selectbox("Размер изображения", imgsz_options, index=imgsz_options.index(imgsz_value) if imgsz_value in imgsz_options else 2)
        if advanced:
            runtime["device"] = st.text_input("Устройство запуска", value=str(runtime.get("device", "0")))
            runtime["models"] = st.multiselect(
                "Модели для полного пайплайна",
                options=list(MODEL_LABELS.keys()),
                default=[m for m in runtime.get("models", ["yolo", "rtdetr", "wbf"]) if m in MODEL_LABELS],
                format_func=lambda x: MODEL_LABELS[x],
            )

            st.subheader("Объединение предсказаний WBF")
            wbf["iou"] = st.slider("IoU-порог WBF", 0.1, 0.9, float(wbf.get("iou", 0.55)), 0.01)
            wbf["skip"] = st.slider("Порог пропуска WBF", 0.0, 0.5, float(wbf.get("skip", 0.001)), 0.001)
            wbf["yolo_weight"] = st.number_input("Вес YOLO", 0.1, 5.0, float(wbf.get("yolo_weight", 1.0)), 0.1)
            wbf["rtdetr_weight"] = st.number_input("Вес RT-DETR", 0.1, 5.0, float(wbf.get("rtdetr_weight", 1.0)), 0.1)

            st.subheader("Плотность")
            density["model"] = st.selectbox(
                "Модель для анализа плотности",
                list(MODEL_LABELS.keys()),
                index=list(MODEL_LABELS.keys()).index(density.get("model", "yolo")) if density.get("model", "yolo") in MODEL_LABELS else 0,
                format_func=lambda x: MODEL_LABELS[x],
            )
            density["rows"] = st.number_input("Строк сетки", 1, 10, int(density.get("rows", 3)))
            density["cols"] = st.number_input("Столбцов сетки", 1, 10, int(density.get("cols", 3)))
            density["limit"] = st.number_input("Лимит визуализаций", 0, 1000, int(density.get("limit", 20)))

            st.subheader("Диагностика готовности")
            readiness["out_dir"] = st.text_input("Папка отчётов диагностики", value=str(readiness.get("out_dir", "D:/1Diplom/shelfvision_results/readiness")))

        st.subheader("Видео")
        video["model"] = st.selectbox("Модель для видео", ["yolo", "yolo_seg"], index=0 if video.get("model", "yolo_seg") == "yolo" else 1, format_func=lambda x: "YOLO-Seg" if x == "yolo_seg" else "YOLO")
        video["input_path"] = st.text_input("Видеофайл", value=str(video.get("input_path", "data/video/test.mp4")))
        video["output_dir"] = st.text_input("Папка результатов видео", value=str(video.get("output_dir", "results/video/yolo")))
        video["frame_skip"] = st.number_input("Обрабатывать каждый N-й кадр", 1, 120, int(video.get("frame_skip", 3)))
        video["max_frames"] = st.number_input("Максимум кадров, 0 — всё видео", 0, 100000, int(video.get("max_frames", 0)))
        video["save_video"] = st.checkbox("Сохранять размеченное видео", value=bool(video.get("save_video", True)))
        video["sample_frames"] = st.number_input("Кадры-примеры", 0, 100, int(video.get("sample_frames", 8)))
        video["show_masks"] = st.checkbox("Показывать маски на видео", value=bool(video.get("show_masks", True)))
        if advanced:
            video["save_frames_for_identification"] = st.checkbox("Сохранять кадры для идентификации", value=bool(video.get("save_frames_for_identification", True)))
            video["tracking_enabled"] = st.checkbox("Включить IoU tracking для видео", value=bool(video.get("tracking_enabled", True)))
            video["tracking_iou"] = st.slider("IoU-порог tracking", 0.05, 0.95, float(video.get("tracking_iou", 0.30)), 0.01)
            video["tracking_max_missing"] = st.number_input("Максимум пропущенных кадров tracking", 0, 100, int(video.get("tracking_max_missing", 5)))
            video["progress_every"] = st.number_input("Печатать прогресс каждые N кадров", 1, 1000, int(video.get("progress_every", 10)))
            video["codec"] = st.text_input("Кодек", value=str(video.get("codec", "mp4v")))

        st.subheader("SKU-галерея")
        sku_gallery["gallery_dir"] = st.text_input("Папка SKU-галереи", value=str(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "D:/1Diplom/sku_gallery"))))
        sku_gallery["output_csv"] = st.text_input("Куда сохранить gallery.csv", value=str(sku_gallery.get("output_csv", identification.get("gallery_csv", "D:/1Diplom/sku_gallery/gallery.csv"))))
        sku_gallery["out_dir"] = st.text_input("Папка отчётов SKU-галереи", value=str(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery")))
        if advanced:
            sku_gallery["min_images_per_sku"] = st.number_input("Минимум эталонов на SKU", 1, 100, int(sku_gallery.get("min_images_per_sku", 3)))

        st.subheader("Demo SKU-галерея")
        demo_sku_gallery["images_dir"] = st.text_input("Папка изображений для фото-идентификации", value=str(demo_sku_gallery.get("images_dir", paths.get("images_dir", "D:/1Diplom/data/raw/d2s_full/images"))))
        demo_sku_gallery["out_dir"] = st.text_input("Папка результатов фото-идентификации", value=str(demo_sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/photo_identification")))
        demo_sku_gallery["gallery_dir"] = st.text_input("Папка demo SKU-галереи", value=str(demo_sku_gallery.get("gallery_dir", sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery"))))
        demo_sku_gallery["gallery_csv"] = st.text_input("CSV-файл demo SKU-галереи", value=str(demo_sku_gallery.get("gallery_csv", sku_gallery.get("output_csv", "D:/1Diplom/sku_gallery/gallery.csv"))))
        demo_sku_gallery["model"] = st.selectbox("Модель для фото-идентификации", list(PHOTO_MODEL_LABELS.keys()), index=list(PHOTO_MODEL_LABELS.keys()).index(demo_sku_gallery.get("model", "yolo")) if demo_sku_gallery.get("model", "yolo") in PHOTO_MODEL_LABELS else 0, format_func=lambda x: PHOTO_MODEL_LABELS[x])
        demo_sku_gallery["max_sku"] = st.number_input("Максимум demo SKU", 1, 500, int(demo_sku_gallery.get("max_sku", 30)))
        if advanced:
            demo_sku_gallery["min_score"] = st.slider("Минимальная confidence для эталона", 0.0, 1.0, float(demo_sku_gallery.get("min_score", 0.35)), 0.01)
            demo_sku_gallery["min_width"] = st.number_input("Минимальная ширина crop", 1, 1000, int(demo_sku_gallery.get("min_width", 20)))
            demo_sku_gallery["min_height"] = st.number_input("Минимальная высота crop", 1, 1000, int(demo_sku_gallery.get("min_height", 20)))
            demo_sku_gallery["padding"] = st.slider("Отступ вокруг crop", 0.0, 0.5, float(demo_sku_gallery.get("padding", 0.05)), 0.01)
            demo_sku_gallery["threshold"] = st.slider("Порог идентификации SKU", 0.0, 1.0, float(demo_sku_gallery.get("threshold", identification.get("threshold", 0.65))), 0.01)
            demo_sku_gallery["top_k"] = st.number_input("Количество ближайших кандидатов", 1, 20, int(demo_sku_gallery.get("top_k", identification.get("top_k", 3))))
            demo_sku_gallery["visualize_limit"] = st.number_input("Лимит визуализаций фото", 0, 1000, int(demo_sku_gallery.get("visualize_limit", 50)))
            demo_sku_gallery["use_masks"] = st.checkbox("Использовать маски для demo crop", value=bool(demo_sku_gallery.get("use_masks", True)))
            demo_sku_gallery["prefix"] = st.text_input("Префикс demo SKU", value=str(demo_sku_gallery.get("prefix", "sku_demo_")))
            demo_sku_gallery["keep_old_demo"] = st.checkbox("Не удалять старые sku_demo_*", value=bool(demo_sku_gallery.get("keep_old_demo", False)))

        st.subheader("Идентификация SKU")
        identification["predictions"] = st.text_input("Файл predictions.json", value=str(identification.get("predictions", "results/inference/yolo_seg_batch/predictions.json")))
        identification["images_dir"] = st.text_input("Папка изображений для predictions", value=str(identification.get("images_dir", "data/yolo_cache/d2s_small_seg/images/test")))
        identification["out_dir"] = st.text_input("Папка результатов идентификации", value=str(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")))
        identification["gallery_csv"] = st.text_input("CSV-файл SKU-галереи", value=str(identification.get("gallery_csv", sku_gallery.get("output_csv", "D:/1Diplom/sku_gallery/gallery.csv"))))
        identification["gallery_dir"] = st.text_input("Папка SKU-галереи", value=str(identification.get("gallery_dir", sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery"))))
        identification["threshold"] = st.slider("Порог визуального сходства SKU", 0.0, 1.0, float(identification.get("threshold", 0.65)), 0.01)
        identification["top_k"] = st.number_input("Количество ближайших SKU-кандидатов", 1, 20, int(identification.get("top_k", 3)))
        if advanced:
            identification["gt_csv"] = st.text_input("GT CSV, необязательно", value=str(identification.get("gt_csv", "")))
            identification["padding"] = st.slider("Отступ вокруг bbox", 0.0, 0.5, float(identification.get("padding", 0.05)), 0.01)
            identification["use_masks"] = st.checkbox("Использовать маски для crop", value=bool(identification.get("use_masks", True)))
            identification["no_visualize"] = st.checkbox("Не сохранять визуализации", value=bool(identification.get("no_visualize", False)))
            identification["visualize_limit"] = st.number_input("Лимит визуализаций", 0, 1000, int(identification.get("visualize_limit", 30)))
            identification["stabilize_tracks"] = st.checkbox("Стабилизировать SKU по track_id", value=bool(identification.get("stabilize_tracks", True)))
            identification["render_identified_video"] = st.checkbox("Собрать видео с подписями SKU", value=bool(identification.get("render_identified_video", True)))
            identification["video_summary"] = st.text_input("video_summary.json, необязательно", value=str(identification.get("video_summary", "")))
            identification["identified_video_codec"] = st.text_input("Кодек identified video", value=str(identification.get("identified_video_codec", "mp4v")))

        st.subheader("Материалы презентации")
        presentation["out_dir"] = st.text_input("Папка материалов презентации", value=str(presentation.get("out_dir", "D:/1Diplom/presentation_assets")))
        if advanced:
            presentation["results_root"] = st.text_input("Корень результатов", value=str(presentation.get("results_root", "D:/1Diplom/shelfvision_results")))
            presentation["video_dir"] = st.text_input("Папка видео результатов", value=str(presentation.get("video_dir", video.get("output_dir", "results/video/yolo"))))
            presentation["identification_dir"] = st.text_input("Папка результатов идентификации для слайдов", value=str(presentation.get("identification_dir", demo_sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/photo_identification") + "/03_identification")))
            presentation["video_frame"] = st.number_input("Кадр из видео для слайдов", 0, 100000, int(presentation.get("video_frame", 0)))

        if st.form_submit_button("Сохранить настройки"):
            identification["gallery_dir"] = str(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "")))
            identification["gallery_csv"] = str(sku_gallery.get("output_csv", identification.get("gallery_csv", "")))
            sku_gallery["gallery_dir"] = str(demo_sku_gallery.get("gallery_dir", sku_gallery.get("gallery_dir", "")))
            sku_gallery["output_csv"] = str(demo_sku_gallery.get("gallery_csv", sku_gallery.get("output_csv", "")))
            save_config(config)
            st.success("Настройки сохранены")
    return config

def _build_readiness_args(config: Dict[str, Any]) -> List[str]:
    readiness = config.setdefault("readiness", {})
    return ["--config", CONFIG_PATH, "--out-dir", str(readiness.get("out_dir", "D:/1Diplom/shelfvision_results/readiness"))]


def _build_gallery_args(config: Dict[str, Any]) -> List[str]:
    sku_gallery = config.setdefault("sku_gallery", {})
    return [
        "--gallery-dir", str(sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery")),
        "--output-csv", str(sku_gallery.get("output_csv", "D:/1Diplom/sku_gallery/gallery.csv")),
        "--out-dir", str(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery")),
        "--min-images-per-sku", str(sku_gallery.get("min_images_per_sku", 3)),
    ]


def _build_photo_identification_args(config: Dict[str, Any]) -> List[str]:
    runtime = config.setdefault("runtime", {})
    demo = config.setdefault("demo_sku_gallery", {})
    model = str(demo.get("model", "yolo"))
    args = [
        "--model", model,
        "--weights", _photo_weight_path(config, model),
        "--images-dir", str(demo.get("images_dir", config.get("paths", {}).get("images_dir", ""))),
        "--out-dir", str(demo.get("out_dir", "D:/1Diplom/shelfvision_results/photo_identification")),
        "--gallery-dir", str(demo.get("gallery_dir", "D:/1Diplom/sku_gallery")),
        "--gallery-csv", str(demo.get("gallery_csv", "D:/1Diplom/sku_gallery/gallery.csv")),
        "--max-sku", str(demo.get("max_sku", 30)),
        "--min-score", str(demo.get("min_score", 0.35)),
        "--min-width", str(demo.get("min_width", 20)),
        "--min-height", str(demo.get("min_height", 20)),
        "--conf", str(runtime.get("conf", 0.25)),
        "--imgsz", str(runtime.get("imgsz", 640)),
        "--threshold", str(demo.get("threshold", config.get("identification", {}).get("threshold", 0.65))),
        "--top-k", str(demo.get("top_k", config.get("identification", {}).get("top_k", 3))),
        "--visualize-limit", str(demo.get("visualize_limit", 50)),
        "--padding", str(demo.get("padding", 0.05)),
        "--prefix", str(demo.get("prefix", "sku_demo_")),
    ]
    device = str(runtime.get("device", "")).strip()
    if device:
        args.extend(["--device", device])
    if not bool(demo.get("use_masks", True)):
        args.append("--bbox-only")
    if bool(demo.get("keep_old_demo", False)):
        args.append("--keep-old-demo")
    gt_csv = str(demo.get("gt_csv", "")).strip()
    if gt_csv:
        args.extend(["--gt-csv", gt_csv])
    return args


def _build_video_args(config: Dict[str, Any], out_base: Path) -> List[str]:
    runtime = config["runtime"]
    video = config.get("video", {})
    video_model = str(video.get("model", "yolo_seg"))
    args = [
        "--model", video_model,
        "--weights", _video_weight_path(config, video_model),
        "--video", str(video.get("input_path", "data/video/test.mp4")),
        "--out-dir", str(video.get("output_dir", str(out_base / "video"))),
        "--conf", str(runtime.get("conf", 0.25)),
        "--imgsz", str(runtime.get("imgsz", 640)),
        "--frame-skip", str(video.get("frame_skip", 3)),
        "--max-frames", str(video.get("max_frames", 0)),
        "--sample-frames", str(video.get("sample_frames", 8)),
        "--codec", str(video.get("codec", "mp4v")),
        "--tracking-iou", str(video.get("tracking_iou", 0.30)),
        "--tracking-max-missing", str(video.get("tracking_max_missing", 5)),
        "--progress-every", str(video.get("progress_every", 10)),
    ]
    if bool(video.get("save_frames_for_identification", True)):
        args.append("--save-frames-for-identification")
    if not bool(video.get("save_video", True)):
        args.append("--no-save-video")
    if not bool(video.get("show_masks", True)):
        args.append("--no-masks")
    if not bool(video.get("tracking_enabled", True)):
        args.append("--no-tracking")
    device = str(runtime.get("device", "")).strip()
    if device:
        args.extend(["--device", device])
    return args


def _build_identification_args(config: Dict[str, Any]) -> List[str]:
    identification = config.setdefault("identification", {})
    args = [
        "--predictions", str(identification.get("predictions", "")),
        "--out-dir", str(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")),
        "--threshold", str(identification.get("threshold", 0.65)),
        "--top-k", str(identification.get("top_k", 3)),
        "--padding", str(identification.get("padding", 0.05)),
        "--visualize-limit", str(identification.get("visualize_limit", 30)),
    ]
    for key, cli_name in [("images_dir", "--images-dir"), ("gallery_csv", "--gallery-csv"), ("gallery_dir", "--gallery-dir"), ("gt_csv", "--gt-csv")]:
        value = str(identification.get(key, "")).strip()
        if value:
            args.extend([cli_name, value])
    if bool(identification.get("use_masks", True)):
        args.append("--use-masks")
    if bool(identification.get("no_visualize", False)):
        args.append("--no-visualize")
    if bool(identification.get("stabilize_tracks", True)):
        args.append("--stabilize-tracks")
    if bool(identification.get("render_identified_video", False)):
        args.append("--render-identified-video")
        video_summary = str(identification.get("video_summary", "")).strip()
        if video_summary:
            args.extend(["--video-summary", video_summary])
        args.extend(["--identified-video-codec", str(identification.get("identified_video_codec", "mp4v"))])
    return args


def _build_presentation_assets_args(config: Dict[str, Any]) -> List[str]:
    presentation = config.setdefault("presentation_assets", {})
    demo = config.setdefault("demo_sku_gallery", {})
    sku_gallery = config.setdefault("sku_gallery", {})
    return [
        "--project-root", ".",
        "--out-dir", str(presentation.get("out_dir", "D:/1Diplom/presentation_assets")),
        "--results-root", str(presentation.get("results_root", "D:/1Diplom/shelfvision_results")),
        "--video-dir", str(presentation.get("video_dir", config.get("video", {}).get("output_dir", "results/video/yolo"))),
        "--identification-dir", str(presentation.get("identification_dir", str(demo.get("out_dir", "D:/1Diplom/shelfvision_results/photo_identification")) + "/03_identification")),
        "--sku-gallery-dir", str(demo.get("gallery_dir", sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery"))),
        "--sku-gallery-report-dir", str(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery")),
        "--video-frame", str(presentation.get("video_frame", 0)),
    ]


def page_actions_wsl(config: Dict[str, Any]) -> None:
    st.header("4. Запуск задач кнопками")
    st.info(f"Текущий режим запуска задач: **{'WSL .venv_wsl' if use_wsl_runtime(config) else 'Windows/local .venv'}**")

    paths = config["paths"]
    runtime = config["runtime"]
    out_base = Path(paths.get("out_dir", "results/control_panel"))

    st.subheader("Интерфейсы")
    c1, c2, c3 = st.columns(3)
    with c1:
        app_script = "scripts/interface_app.py"
        if st.button("Открыть интерфейс экспериментов", use_container_width=True):
            _launch_background(streamlit_command(config, app_script))
            st.success("Интерфейс экспериментов запускается в отдельном процессе.")
            st.markdown(f"Открой: [{streamlit_url(app_script)}]({streamlit_url(app_script)})")
    with c2:
        app_script = "scripts/inference_app.py"
        if st.button("Открыть интерфейс инференса", use_container_width=True):
            _launch_background(streamlit_command(config, app_script))
            st.success("Интерфейс инференса запускается в отдельном процессе.")
            st.markdown(f"Открой: [{streamlit_url(app_script)}]({streamlit_url(app_script)})")
    with c3:
        app_script = "scripts/video_app.py"
        if st.button("Открыть видеоинтерфейс", use_container_width=True):
            _launch_background(streamlit_command(config, app_script))
            st.success("Видеоинтерфейс запускается в отдельном процессе.")
            st.markdown(f"Открой: [{streamlit_url(app_script)}]({streamlit_url(app_script)})")

    st.subheader("Диагностика")
    st.caption("Быстрая проверка перед тяжёлым запуском: видео, веса, SKU-галерея, gallery.csv, папки вывода и WSL-среда.")
    if st.button("Проверить готовность видео-идентификации", use_container_width=True):
        save_config(config)
        cmd = python_command(config, "run_readiness_check.py", _build_readiness_args(config))
        _run_live_command(
            title="Диагностика готовности",
            cmd=cmd,
            description="Проверяются пути, веса, видео, SKU-галерея, gallery.csv, папки вывода и совместимость с WSL.",
            success="Диагностика завершена. Проверь readiness_report.md/json и readiness_checks.csv.",
            failure="Ошибка диагностики готовности",
        )

    st.subheader("Идентификация по фото")
    st.caption("Полный рабочий сценарий для защиты: изображения → инференс → demo SKU-галерея → gallery.csv → идентификация → визуализации.")
    if st.button("Запустить полную идентификацию по фото", use_container_width=True):
        save_config(config)
        cmd = python_command(config, "run_photo_identification_pipeline.py", _build_photo_identification_args(config))
        _run_live_command(
            title="Идентификация по фото",
            cmd=cmd,
            description="Запускается полный цикл: инференс по папке изображений, автоматическая demo SKU-галерея, сопоставление с галереей и визуализация результатов.",
            success="Фото-идентификация завершена. Проверь 01_inference, 02_demo_gallery и 03_identification/visualized.",
            failure="Ошибка фото-идентификации",
        )

    st.subheader("Инференс одного изображения")
    selected_model = st.selectbox("Модель", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])
    if st.button("Запустить инференс через выбранный режим", use_container_width=True):
        out_dir = out_base / "inference" / selected_model
        args = ["--model", selected_model, *build_weight_args(config, selected_model), "--image", paths["image"], "--out-dir", str(out_dir), "--conf", str(runtime["conf"]), "--imgsz", str(runtime["imgsz"])]
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        render_command_result(run_command(python_command(config, "run_inference.py", args)))

    st.subheader("Видеоинференс")
    st.caption("Видео запускается с live-log: строки VIDEO_PROGRESS показывают кадры, FPS, объекты и ETA.")
    if st.button("Обработать видео через выбранный режим", use_container_width=True):
        cmd = python_command(config, "run_video_inference.py", _build_video_args(config, out_base))
        step = CommandStep(
            title="Видеоинференс YOLO/YOLO-Seg",
            cmd=cmd,
            cwd=ROOT,
            description="Идёт обработка видео. В логе ниже будут появляться строки VIDEO_PROGRESS с количеством кадров, FPS и примерным ETA.",
            estimated_seconds=_video_expected_seconds(config),
        )
        run_steps_with_progress(
            [step],
            title="Видеоинференс с live-progress",
            success_message="Видео обработано. Проверь output_video.mp4, video_predictions.json, video_summary.json и frames_for_identification.",
            failure_message="Ошибка обработки видео",
        )

    st.subheader("SKU-галерея")
    st.caption("Проверяет эталонную базу товаров, создаёт gallery.csv и отчёты качества галереи.")
    if st.button("Проверить SKU-галерею и создать gallery.csv", use_container_width=True):
        cmd = python_command(config, "run_gallery_manager.py", _build_gallery_args(config))
        _run_live_command(
            title="SKU-галерея",
            cmd=cmd,
            description="Сканируется sku_gallery/<sku_id>/*.jpg, проверяются битые изображения, создаётся gallery.csv и отчёт по галерее.",
            success="SKU-галерея проверена. Проверь gallery.csv, sku_gallery_report.md/json и csv-таблицы.",
            failure="Ошибка проверки SKU-галереи",
        )
        sku_gallery = config.setdefault("sku_gallery", {})
        identification = config.setdefault("identification", {})
        identification["gallery_dir"] = str(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "")))
        identification["gallery_csv"] = str(sku_gallery.get("output_csv", identification.get("gallery_csv", "")))
        save_config(config)

    st.subheader("Идентификация SKU")
    identification = config.setdefault("identification", {})
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Подставить результаты последнего видео", use_container_width=True):
            video_out = Path(config.get("video", {}).get("output_dir", "results/video/yolo"))
            identification["predictions"] = str(video_out / "video_predictions.json")
            identification["images_dir"] = str(video_out / "frames_for_identification")
            identification["video_summary"] = str(video_out / "video_summary.json")
            identification["render_identified_video"] = True
            identification["stabilize_tracks"] = True
            save_config(config)
            st.success("В настройки идентификации подставлены video_predictions.json, frames_for_identification и video_summary.json")
    with c2:
        if st.button("Запустить идентификацию SKU", use_container_width=True):
            cmd = python_command(config, "run_identification.py", _build_identification_args(config))
            _run_live_command(
                title="Идентификация SKU",
                cmd=cmd,
                description="Запускается извлечение crop-объектов, сопоставление с SKU-галереей, стабилизация по track_id и сборка identified video при включённой опции.",
                success="Идентификация завершена. Проверь identification_results.json/csv, track_sku_summary.json и identified_output_video.mp4.",
                failure="Ошибка идентификации SKU",
            )

    st.subheader("Материалы для презентации")
    if st.button("Собрать скрины и файлы по слайдам", use_container_width=True):
        save_config(config)
        cmd = python_command(config, "scripts/prepare_presentation_assets.py", _build_presentation_assets_args(config))
        _run_live_command(
            title="Материалы презентации",
            cmd=cmd,
            description="Файлы результатов раскладываются по папкам slide_01...slide_10 для вставки в Gamma/PowerPoint.",
            success="Материалы презентации подготовлены. Проверь папку presentation_assets.",
            failure="Ошибка подготовки материалов презентации",
        )

    st.subheader("Полный пайплайн")
    if st.button("Запустить полный пайплайн через выбранный режим", use_container_width=True):
        models = runtime.get("models", ["yolo", "rtdetr", "wbf"])
        args = ["--images-dir", paths["images_dir"], "--yolo-weights", config["weights"]["yolo"], "--rtdetr-weights", config["weights"]["rtdetr"], "--models", *models, "--out-dir", str(out_base / "full_pipeline"), "--conf", str(runtime["conf"]), "--imgsz", str(runtime["imgsz"]), "--density-model", config["density"].get("model", "yolo"), "--density-rows", str(config["density"].get("rows", 3)), "--density-cols", str(config["density"].get("cols", 3)), "--density-limit", str(config["density"].get("limit", 20))]
        if "frcnn" in models:
            args.extend(["--frcnn-weights", config["weights"]["frcnn"]])
        if paths.get("gt_coco") and resolve_path(paths["gt_coco"]).exists():
            args.extend(["--gt-coco", paths["gt_coco"]])
        else:
            args.extend(["--gt-yolo-labels", paths["gt_yolo_labels"]])
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        _run_live_command("Полный пайплайн", python_command(config, "run_full_pipeline.py", args), "Запускается полный пайплайн.", "Пайплайн завершён.", "Ошибка полного пайплайна")

    st.subheader("Служебные проверки")
    if st.button("Smoke-проверка через выбранный режим", use_container_width=True):
        _run_live_command("Smoke-проверка", python_command(config, "scripts/smoke_cli.py", []), "Проверяется базовая работоспособность CLI.", "Smoke-проверка завершена.", "Ошибка smoke-проверки")


def main() -> None:
    st.set_page_config(page_title="Панель управления ShelfVision", page_icon="🧰", layout="wide")
    ensure_config_exists()
    config = load_config()
    config.setdefault("runtime", {}).setdefault("use_wsl_runtime", True)

    st.title("🧰 Панель управления ShelfVision")
    st.caption("Панель первого запуска и управления. Рабочие задачи по умолчанию запускаются через WSL .venv_wsl.")

    page = st.sidebar.radio("Раздел", ["Первый запуск", "Скачивание файлов", "Настройки", "Запуск задач", "Результаты", "config YAML"])
    if page == "Первый запуск":
        page_setup(config)
    elif page == "Скачивание файлов":
        page_downloads(config)
    elif page == "Настройки":
        page_config_wsl(config)
    elif page == "Запуск задач":
        page_actions_wsl(config)
    elif page == "Результаты":
        page_results(config)
    elif page == "config YAML":
        st.header("config/shelfvision.yaml")
        st.code(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), language="yaml")
        if st.button("Сохранить текущий YAML"):
            save_config(config)
            st.success("Сохранено")


if __name__ == "__main__":
    main()
