from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "config/shelfvision.yaml"


def use_wsl_runtime(config: Dict[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("use_wsl_runtime", True))


def wsl_venv_dir(config: Dict[str, Any]) -> str:
    return str(config.get("setup", {}).get("venv_dir_wsl", ".venv_wsl"))


def python_command(config: Dict[str, Any], script: str, args: List[str]) -> List[str]:
    if use_wsl_runtime(config):
        return [sys.executable, "scripts/wsl_runtime.py", "--venv-dir", wsl_venv_dir(config), script, *args]
    return [str(venv_python(config)), script, *args]


def streamlit_command(config: Dict[str, Any], app_script: str) -> List[str]:
    if use_wsl_runtime(config):
        return python_command(config, "-m", ["streamlit", "run", app_script])
    return [str(venv_python(config)), "-m", "streamlit", "run", app_script]


def render_command_result(result) -> None:
    if result.returncode == 0:
        st.success(f"Команда выполнена успешно: returncode={result.returncode}")
    else:
        st.error(f"Команда завершилась с ошибкой: returncode={result.returncode}")
    st.code(result.stdout or "", language="text")


def _video_weight_path(config: Dict[str, Any], video_model: str) -> str:
    weights = config.get("weights", {})
    if video_model == "yolo_seg":
        return str(weights.get("yolo_seg", weights.get("yolo", "")))
    return str(weights.get("yolo", ""))


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
        identification = config.setdefault("identification", {})

        st.subheader("Режим запуска")
        runtime["use_wsl_runtime"] = st.checkbox("Запускать задачи через WSL .venv_wsl", value=bool(runtime.get("use_wsl_runtime", True)))

        st.subheader("Пути")
        paths["image"] = st.text_input("Одно изображение", value=str(paths.get("image", "")))
        paths["images_dir"] = st.text_input("Папка изображений", value=str(paths.get("images_dir", "")))
        paths["gt_coco"] = st.text_input("COCO annotations.json", value=str(paths.get("gt_coco", "")))
        paths["gt_yolo_labels"] = st.text_input("YOLO labels", value=str(paths.get("gt_yolo_labels", "")))
        paths["out_dir"] = st.text_input("Папка результатов", value=str(paths.get("out_dir", "results/control_panel")))

        st.subheader("Веса")
        weights["yolo"] = st.text_input("YOLO weights", value=str(weights.get("yolo", "")))
        weights["yolo_seg"] = st.text_input("YOLO-Seg weights", value=str(weights.get("yolo_seg", "")))
        weights["rtdetr"] = st.text_input("RT-DETR weights", value=str(weights.get("rtdetr", "")))
        weights["frcnn"] = st.text_input("Faster R-CNN weights", value=str(weights.get("frcnn", "")))

        st.subheader("Setup")
        setup["venv_dir"] = st.text_input("Windows/local venv для панели", value=str(setup.get("venv_dir", ".venv")))
        setup["venv_dir_wsl"] = st.text_input("WSL venv для запуска задач", value=str(setup.get("venv_dir_wsl", ".venv_wsl")))
        setup["requirements"] = st.text_input("requirements.txt", value=str(setup.get("requirements", "requirements.txt")))

        st.subheader("Runtime")
        runtime["conf"] = st.slider("Confidence", 0.01, 0.95, float(runtime.get("conf", 0.25)), 0.01)
        imgsz_options = [416, 512, 640, 768, 1024]
        imgsz_value = int(runtime.get("imgsz", 640))
        runtime["imgsz"] = st.selectbox("imgsz", imgsz_options, index=imgsz_options.index(imgsz_value) if imgsz_value in imgsz_options else 2)
        runtime["device"] = st.text_input("device", value=str(runtime.get("device", "0")))
        runtime["models"] = st.multiselect(
            "Модели для полного pipeline",
            options=list(MODEL_LABELS.keys()),
            default=[m for m in runtime.get("models", ["yolo", "rtdetr", "wbf"]) if m in MODEL_LABELS],
            format_func=lambda x: MODEL_LABELS[x],
        )

        st.subheader("WBF")
        wbf["iou"] = st.slider("WBF IoU", 0.1, 0.9, float(wbf.get("iou", 0.55)), 0.01)
        wbf["skip"] = st.slider("WBF skip", 0.0, 0.5, float(wbf.get("skip", 0.001)), 0.001)
        wbf["yolo_weight"] = st.number_input("YOLO weight", 0.1, 5.0, float(wbf.get("yolo_weight", 1.0)), 0.1)
        wbf["rtdetr_weight"] = st.number_input("RT-DETR weight", 0.1, 5.0, float(wbf.get("rtdetr_weight", 1.0)), 0.1)

        st.subheader("Плотность")
        density["model"] = st.selectbox(
            "Модель для density",
            list(MODEL_LABELS.keys()),
            index=list(MODEL_LABELS.keys()).index(density.get("model", "yolo")) if density.get("model", "yolo") in MODEL_LABELS else 0,
            format_func=lambda x: MODEL_LABELS[x],
        )
        density["rows"] = st.number_input("Rows", 1, 10, int(density.get("rows", 3)))
        density["cols"] = st.number_input("Cols", 1, 10, int(density.get("cols", 3)))
        density["limit"] = st.number_input("Visualize limit", 0, 1000, int(density.get("limit", 20)))

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
        video["show_masks"] = st.checkbox("Показывать masks на видео", value=bool(video.get("show_masks", True)))
        video["save_frames_for_identification"] = st.checkbox("Сохранять кадры для идентификации", value=bool(video.get("save_frames_for_identification", True)))
        video["tracking_enabled"] = st.checkbox("Включить IoU tracking для видео", value=bool(video.get("tracking_enabled", True)))
        video["tracking_iou"] = st.slider("Tracking IoU threshold", 0.05, 0.95, float(video.get("tracking_iou", 0.30)), 0.01)
        video["tracking_max_missing"] = st.number_input("Tracking max missing frames", 0, 100, int(video.get("tracking_max_missing", 5)))
        video["progress_every"] = st.number_input("Печатать прогресс каждые N кадров", 1, 1000, int(video.get("progress_every", 10)))
        video["codec"] = st.text_input("Кодек", value=str(video.get("codec", "mp4v")))

        st.subheader("SKU-галерея")
        sku_gallery["gallery_dir"] = st.text_input("Папка SKU-галереи", value=str(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "D:/1Diplom/sku_gallery"))))
        sku_gallery["output_csv"] = st.text_input("Куда сохранить gallery.csv", value=str(sku_gallery.get("output_csv", identification.get("gallery_csv", "D:/1Diplom/sku_gallery/gallery.csv"))))
        sku_gallery["out_dir"] = st.text_input("Папка отчётов SKU-галереи", value=str(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery")))
        sku_gallery["min_images_per_sku"] = st.number_input("Минимум эталонов на SKU", 1, 100, int(sku_gallery.get("min_images_per_sku", 3)))

        st.subheader("Идентификация SKU")
        identification["predictions"] = st.text_input("predictions.json", value=str(identification.get("predictions", "results/inference/yolo_seg_batch/predictions.json")))
        identification["images_dir"] = st.text_input("images_dir для predictions", value=str(identification.get("images_dir", "data/yolo_cache/d2s_small_seg/images/test")))
        identification["out_dir"] = st.text_input("Папка результатов идентификации", value=str(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")))
        identification["gallery_csv"] = st.text_input("SKU gallery.csv", value=str(identification.get("gallery_csv", sku_gallery.get("output_csv", "D:/1Diplom/sku_gallery/gallery.csv"))))
        identification["gallery_dir"] = st.text_input("SKU gallery dir", value=str(identification.get("gallery_dir", sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery"))))
        identification["gt_csv"] = st.text_input("GT CSV, опционально", value=str(identification.get("gt_csv", "")))
        identification["threshold"] = st.slider("SKU similarity threshold", 0.0, 1.0, float(identification.get("threshold", 0.65)), 0.01)
        identification["top_k"] = st.number_input("Top-k кандидатов SKU", 1, 20, int(identification.get("top_k", 3)))
        identification["padding"] = st.slider("Padding вокруг bbox", 0.0, 0.5, float(identification.get("padding", 0.05)), 0.01)
        identification["use_masks"] = st.checkbox("Использовать masks для crop", value=bool(identification.get("use_masks", True)))
        identification["no_visualize"] = st.checkbox("Не сохранять визуализации", value=bool(identification.get("no_visualize", False)))
        identification["visualize_limit"] = st.number_input("Лимит визуализаций", 0, 1000, int(identification.get("visualize_limit", 30)))
        identification["stabilize_tracks"] = st.checkbox("Стабилизировать SKU по track_id", value=bool(identification.get("stabilize_tracks", True)))
        identification["render_identified_video"] = st.checkbox("Собрать видео с подписями SKU", value=bool(identification.get("render_identified_video", True)))
        identification["video_summary"] = st.text_input("video_summary.json, опционально", value=str(identification.get("video_summary", "")))
        identification["identified_video_codec"] = st.text_input("Кодек identified video", value=str(identification.get("identified_video_codec", "mp4v")))

        if st.form_submit_button("Сохранить настройки"):
            identification["gallery_dir"] = str(sku_gallery.get("gallery_dir", identification.get("gallery_dir", "")))
            identification["gallery_csv"] = str(sku_gallery.get("output_csv", identification.get("gallery_csv", "")))
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


def _build_video_args(config: Dict[str, Any], out_base: Path) -> List[str]:
    runtime = config["runtime"]
    video = config.get("video", {})
    video_model = str(video.get("model", "yolo_seg"))
    args = [
        "--model", "yolo",
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


def page_actions_wsl(config: Dict[str, Any]) -> None:
    st.header("4. Запуск задач кнопками")
    st.info(f"Текущий режим запуска задач: **{'WSL .venv_wsl' if use_wsl_runtime(config) else 'Windows/local .venv'}**")

    paths = config["paths"]
    runtime = config["runtime"]
    out_base = Path(paths.get("out_dir", "results/control_panel"))

    st.subheader("Интерфейсы")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Открыть интерфейс экспериментов", use_container_width=True):
            render_command_result(run_command(streamlit_command(config, "scripts/interface_app.py")))
    with c2:
        if st.button("Открыть интерфейс инференса", use_container_width=True):
            render_command_result(run_command(streamlit_command(config, "scripts/inference_app.py")))
    with c3:
        if st.button("Открыть видеоинтерфейс", use_container_width=True):
            render_command_result(run_command(streamlit_command(config, "scripts/video_app.py")))

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

    st.subheader("Инференс одного изображения")
    selected_model = st.selectbox("Модель", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])
    if st.button("Запустить инференс через выбранный runtime", use_container_width=True):
        out_dir = out_base / "inference" / selected_model
        args = ["--model", selected_model, *build_weight_args(config, selected_model), "--image", paths["image"], "--out-dir", str(out_dir), "--conf", str(runtime["conf"]), "--imgsz", str(runtime["imgsz"])]
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        render_command_result(run_command(python_command(config, "run_inference.py", args)))

    st.subheader("Видеоинференс")
    st.caption("Видео запускается с live-log: строки VIDEO_PROGRESS показывают кадры, FPS, объекты и ETA.")
    if st.button("Обработать видео через выбранный runtime", use_container_width=True):
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
                description="Запускается crop extraction, сопоставление с SKU-галереей, стабилизация по track_id и сборка identified video при включённой опции.",
                success="Идентификация завершена. Проверь identification_results.json/csv, track_sku_summary.json и identified_output_video.mp4.",
                failure="Ошибка идентификации SKU",
            )

    st.subheader("Полный pipeline")
    if st.button("Запустить полный pipeline через выбранный runtime", use_container_width=True):
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
        _run_live_command("Полный pipeline", python_command(config, "run_full_pipeline.py", args), "Запускается полный pipeline.", "Pipeline завершён.", "Ошибка полного pipeline")

    st.subheader("Служебные проверки")
    if st.button("Smoke-проверка через выбранный runtime", use_container_width=True):
        _run_live_command("Smoke-проверка", python_command(config, "scripts/smoke_cli.py", []), "Проверяется базовая работоспособность CLI.", "Smoke-проверка завершена.", "Ошибка smoke-проверки")


def main() -> None:
    st.set_page_config(page_title="ShelfVision Control Panel", page_icon="🧰", layout="wide")
    ensure_config_exists()
    config = load_config()
    config.setdefault("runtime", {}).setdefault("use_wsl_runtime", True)

    st.title("🧰 ShelfVision Control Panel")
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
