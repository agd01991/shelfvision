from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from control_panel import (
    DEFAULT_CONFIG_PATH,
    MODEL_LABELS,
    build_weight_args,
    ensure_config_exists,
    load_config,
    page_downloads,
    page_results,
    rel_path,
    resolve_path,
    run_command,
    save_config,
    venv_python,
)
from setup_pages import page_setup


ROOT = Path(__file__).resolve().parents[1]


def use_wsl_runtime(config: Dict[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("use_wsl_runtime", True))


def wsl_venv_dir(config: Dict[str, Any]) -> str:
    return str(config.get("setup", {}).get("venv_dir_wsl", ".venv_wsl"))


def python_command(config: Dict[str, Any], script: str, args: List[str]) -> List[str]:
    """Build command for running project scripts.

    When runtime.use_wsl_runtime=true, the target script runs through WSL using
    .venv_wsl/bin/python. The Windows .venv is used only to launch Streamlit.
    """

    if use_wsl_runtime(config):
        return [
            sys.executable,
            "scripts/wsl_runtime.py",
            "--venv-dir",
            wsl_venv_dir(config),
            script,
            *args,
        ]
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


def _video_weight_options(config: Dict[str, Any]) -> Dict[str, str]:
    weights = config.get("weights", {})
    return {
        "yolo": str(weights.get("yolo", "")),
        "yolo_seg": str(weights.get("yolo_seg", weights.get("yolo", ""))),
    }


def _video_weight_path(config: Dict[str, Any], video_model: str) -> str:
    return _video_weight_options(config).get(video_model, config.get("weights", {}).get("yolo", ""))


def page_config_wsl(config: Dict[str, Any]) -> Dict[str, Any]:
    st.header("3. Настройки проекта")

    with st.form("config_form_wsl"):
        st.subheader("Режим запуска")
        runtime = config.setdefault("runtime", {})
        runtime["use_wsl_runtime"] = st.checkbox(
            "Запускать задачи через WSL .venv_wsl",
            value=bool(runtime.get("use_wsl_runtime", True)),
            help="Если включено, инференс, полный pipeline и smoke-проверка идут через WSL и .venv_wsl/bin/python.",
        )

        st.subheader("Пути")
        paths = config["paths"]
        paths["image"] = st.text_input("Одно изображение", value=str(paths.get("image", "")))
        paths["images_dir"] = st.text_input("Папка изображений", value=str(paths.get("images_dir", "")))
        paths["gt_coco"] = st.text_input("COCO annotations.json", value=str(paths.get("gt_coco", "")))
        paths["gt_yolo_labels"] = st.text_input("YOLO labels", value=str(paths.get("gt_yolo_labels", "")))
        paths["out_dir"] = st.text_input("Папка результатов", value=str(paths.get("out_dir", "")))

        st.subheader("Веса")
        weights = config["weights"]
        weights["yolo"] = st.text_input("YOLO weights", value=str(weights.get("yolo", "")))
        weights["yolo_seg"] = st.text_input("YOLO-Seg weights", value=str(weights.get("yolo_seg", "")))
        weights["rtdetr"] = st.text_input("RT-DETR weights", value=str(weights.get("rtdetr", "")))
        weights["frcnn"] = st.text_input("Faster R-CNN weights", value=str(weights.get("frcnn", "")))

        st.subheader("Setup")
        setup = config["setup"]
        setup["venv_dir"] = st.text_input("Windows/local venv для панели", value=str(setup.get("venv_dir", ".venv")))
        setup["venv_dir_wsl"] = st.text_input("WSL venv для запуска задач", value=str(setup.get("venv_dir_wsl", ".venv_wsl")))
        setup["requirements"] = st.text_input("requirements.txt", value=str(setup.get("requirements", "requirements.txt")))

        st.subheader("Runtime")
        runtime["conf"] = st.slider("Confidence", 0.01, 0.95, float(runtime.get("conf", 0.25)), 0.01)
        imgsz_options = [416, 512, 640, 768, 1024]
        imgsz_value = int(runtime.get("imgsz", 640))
        runtime["imgsz"] = st.selectbox(
            "imgsz",
            imgsz_options,
            index=imgsz_options.index(imgsz_value) if imgsz_value in imgsz_options else 2,
        )
        runtime["device"] = st.text_input("device", value=str(runtime.get("device", "0")))
        runtime["models"] = st.multiselect(
            "Модели для полного pipeline",
            options=list(MODEL_LABELS.keys()),
            default=[m for m in runtime.get("models", ["yolo", "rtdetr", "wbf"]) if m in MODEL_LABELS],
            format_func=lambda x: MODEL_LABELS[x],
        )

        st.subheader("WBF")
        wbf = config["wbf"]
        wbf["iou"] = st.slider("WBF IoU", 0.1, 0.9, float(wbf.get("iou", 0.55)), 0.01)
        wbf["skip"] = st.slider("WBF skip", 0.0, 0.5, float(wbf.get("skip", 0.001)), 0.001)
        wbf["yolo_weight"] = st.number_input("YOLO weight", 0.1, 5.0, float(wbf.get("yolo_weight", 1.0)), 0.1)
        wbf["rtdetr_weight"] = st.number_input("RT-DETR weight", 0.1, 5.0, float(wbf.get("rtdetr_weight", 1.0)), 0.1)

        st.subheader("Плотность")
        density = config["density"]
        density["model"] = st.selectbox(
            "Модель для density",
            list(MODEL_LABELS.keys()),
            index=list(MODEL_LABELS.keys()).index(density.get("model", "yolo")) if density.get("model", "yolo") in MODEL_LABELS else 0,
            format_func=lambda x: MODEL_LABELS[x],
        )
        density["rows"] = st.number_input("Rows", 1, 10, int(density.get("rows", 3)))
        density["cols"] = st.number_input("Cols", 1, 10, int(density.get("cols", 3)))
        density["limit"] = st.number_input("Visualize limit", 0, 1000, int(density.get("limit", 20)))

        st.subheader("Видео")
        video = config.setdefault("video", {})
        video["model"] = st.selectbox(
            "Модель для видео",
            options=["yolo", "yolo_seg"],
            index=0 if video.get("model", "yolo") == "yolo" else 1,
            format_func=lambda x: "YOLO-Seg" if x == "yolo_seg" else "YOLO",
        )
        video["input_path"] = st.text_input("Видеофайл", value=str(video.get("input_path", "data/video/test.mp4")))
        video["output_dir"] = st.text_input("Папка результатов видео", value=str(video.get("output_dir", "results/video/yolo")))
        video["frame_skip"] = st.number_input("Обрабатывать каждый N-й кадр", 1, 120, int(video.get("frame_skip", 3)))
        video["max_frames"] = st.number_input("Максимум кадров, 0 — всё видео", 0, 100000, int(video.get("max_frames", 0)))
        video["save_video"] = st.checkbox("Сохранять размеченное видео", value=bool(video.get("save_video", True)))
        video["sample_frames"] = st.number_input("Кадры-примеры", 0, 100, int(video.get("sample_frames", 8)))
        video["show_masks"] = st.checkbox("Показывать masks на видео", value=bool(video.get("show_masks", True)))
        video["save_frames_for_identification"] = st.checkbox(
            "Сохранять кадры для идентификации",
            value=bool(video.get("save_frames_for_identification", True)),
            help="Нужно для связки video_predictions.json → run_identification.py.",
        )
        video["codec"] = st.text_input("Кодек", value=str(video.get("codec", "mp4v")))

        st.subheader("Идентификация SKU")
        identification = config.setdefault("identification", {})
        identification["predictions"] = st.text_input("predictions.json", value=str(identification.get("predictions", "results/inference/yolo_seg_batch/predictions.json")))
        identification["images_dir"] = st.text_input("images_dir для predictions", value=str(identification.get("images_dir", "data/yolo_cache/d2s_small_seg/images/test")))
        identification["out_dir"] = st.text_input("Папка результатов идентификации", value=str(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")))
        identification["gallery_csv"] = st.text_input("SKU gallery.csv", value=str(identification.get("gallery_csv", "D:/1Diplom/sku_gallery/gallery.csv")))
        identification["gallery_dir"] = st.text_input("SKU gallery dir", value=str(identification.get("gallery_dir", "D:/1Diplom/sku_gallery")))
        identification["gt_csv"] = st.text_input("GT CSV, опционально", value=str(identification.get("gt_csv", "")))
        identification["threshold"] = st.slider("SKU similarity threshold", 0.0, 1.0, float(identification.get("threshold", 0.65)), 0.01)
        identification["top_k"] = st.number_input("Top-k кандидатов SKU", 1, 20, int(identification.get("top_k", 3)))
        identification["padding"] = st.slider("Padding вокруг bbox", 0.0, 0.5, float(identification.get("padding", 0.05)), 0.01)
        identification["use_masks"] = st.checkbox("Использовать masks для crop", value=bool(identification.get("use_masks", True)))
        identification["no_visualize"] = st.checkbox("Не сохранять визуализации", value=bool(identification.get("no_visualize", False)))
        identification["visualize_limit"] = st.number_input("Лимит визуализаций", 0, 1000, int(identification.get("visualize_limit", 30)))
        identification["render_identified_video"] = st.checkbox(
            "Собрать видео с подписями SKU",
            value=bool(identification.get("render_identified_video", True)),
            help="Работает для video_predictions.json, если рядом есть video_summary.json или путь указан ниже.",
        )
        identification["video_summary"] = st.text_input("video_summary.json, опционально", value=str(identification.get("video_summary", "")))
        identification["identified_video_codec"] = st.text_input("Кодек identified video", value=str(identification.get("identified_video_codec", "mp4v")))

        submitted = st.form_submit_button("Сохранить настройки")
        if submitted:
            save_config(config)
            st.success("Настройки сохранены")
    return config


def page_actions_wsl(config: Dict[str, Any]) -> None:
    st.header("4. Запуск задач кнопками")

    runtime_mode = "WSL .venv_wsl" if use_wsl_runtime(config) else "Windows/local .venv"
    st.info(f"Текущий режим запуска задач: **{runtime_mode}**")

    paths = config["paths"]
    runtime = config["runtime"]
    out_base = Path(paths.get("out_dir", "results/control_panel"))

    st.subheader("Интерфейсы")
    st.caption("Панель управления запускается из Windows .venv. Остальные рабочие задачи можно запускать через WSL.")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Открыть интерфейс экспериментов", use_container_width=True):
            result = run_command(streamlit_command(config, "scripts/interface_app.py"))
            render_command_result(result)
    with c2:
        if st.button("Открыть интерфейс инференса", use_container_width=True):
            result = run_command(streamlit_command(config, "scripts/inference_app.py"))
            render_command_result(result)
    with c3:
        if st.button("Открыть видеоинтерфейс", use_container_width=True):
            result = run_command(streamlit_command(config, "scripts/video_app.py"))
            render_command_result(result)

    st.subheader("Инференс одного изображения")
    selected_model = st.selectbox("Модель", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])
    if st.button("Запустить инференс через выбранный runtime", use_container_width=True):
        out_dir = out_base / "inference" / selected_model
        args = [
            "--model",
            selected_model,
            *build_weight_args(config, selected_model),
            "--image",
            paths["image"],
            "--out-dir",
            str(out_dir),
            "--conf",
            str(runtime["conf"]),
            "--imgsz",
            str(runtime["imgsz"]),
        ]
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        result = run_command(python_command(config, "run_inference.py", args))
        render_command_result(result)

    st.subheader("Видеоинференс")
    st.caption("При включённом сохранении кадров результат `video_predictions.json` можно сразу передать в идентификацию SKU.")
    if st.button("Обработать видео через выбранный runtime", use_container_width=True):
        video = config.get("video", {})
        video_model = str(video.get("model", "yolo"))
        args = [
            "--model",
            "yolo",
            "--weights",
            _video_weight_path(config, video_model),
            "--video",
            video.get("input_path", "data/video/test.mp4"),
            "--out-dir",
            video.get("output_dir", str(out_base / "video")),
            "--conf",
            str(runtime["conf"]),
            "--imgsz",
            str(runtime["imgsz"]),
            "--frame-skip",
            str(video.get("frame_skip", 3)),
            "--max-frames",
            str(video.get("max_frames", 0)),
            "--sample-frames",
            str(video.get("sample_frames", 8)),
            "--codec",
            str(video.get("codec", "mp4v")),
        ]
        if bool(video.get("save_frames_for_identification", True)):
            args.append("--save-frames-for-identification")
        if not bool(video.get("save_video", True)):
            args.append("--no-save-video")
        if not bool(video.get("show_masks", True)):
            args.append("--no-masks")
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        result = run_command(python_command(config, "run_video_inference.py", args))
        render_command_result(result)

    st.subheader("Идентификация SKU")
    st.caption("Можно запускать по обычному predictions.json или по video_predictions.json после обработки видео.")
    identification = config.setdefault("identification", {})
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Подставить результаты последнего видео", use_container_width=True):
            video_out = Path(config.get("video", {}).get("output_dir", "results/video/yolo"))
            identification["predictions"] = str(video_out / "video_predictions.json")
            identification["images_dir"] = str(video_out / "frames_for_identification")
            identification["video_summary"] = str(video_out / "video_summary.json")
            identification["render_identified_video"] = True
            save_config(config)
            st.success("В настройки идентификации подставлены video_predictions.json, frames_for_identification и video_summary.json")
    with c2:
        if st.button("Запустить идентификацию SKU", use_container_width=True):
            args = [
                "--predictions",
                str(identification.get("predictions", "")),
                "--out-dir",
                str(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")),
                "--threshold",
                str(identification.get("threshold", 0.65)),
                "--top-k",
                str(identification.get("top_k", 3)),
                "--padding",
                str(identification.get("padding", 0.05)),
                "--visualize-limit",
                str(identification.get("visualize_limit", 30)),
            ]
            images_dir = str(identification.get("images_dir", "")).strip()
            if images_dir:
                args.extend(["--images-dir", images_dir])
            gallery_csv = str(identification.get("gallery_csv", "")).strip()
            gallery_dir = str(identification.get("gallery_dir", "")).strip()
            if gallery_csv:
                args.extend(["--gallery-csv", gallery_csv])
            if gallery_dir:
                args.extend(["--gallery-dir", gallery_dir])
            gt_csv = str(identification.get("gt_csv", "")).strip()
            if gt_csv:
                args.extend(["--gt-csv", gt_csv])
            if bool(identification.get("use_masks", True)):
                args.append("--use-masks")
            if bool(identification.get("no_visualize", False)):
                args.append("--no-visualize")
            if bool(identification.get("render_identified_video", False)):
                args.append("--render-identified-video")
                video_summary = str(identification.get("video_summary", "")).strip()
                if video_summary:
                    args.extend(["--video-summary", video_summary])
                args.extend(["--identified-video-codec", str(identification.get("identified_video_codec", "mp4v"))])

            result = run_command(python_command(config, "run_identification.py", args))
            render_command_result(result)

    st.subheader("Полный pipeline")
    if st.button("Запустить полный pipeline через выбранный runtime", use_container_width=True):
        models = runtime.get("models", ["yolo", "rtdetr", "wbf"])
        args = [
            "--images-dir",
            paths["images_dir"],
            "--yolo-weights",
            config["weights"]["yolo"],
            "--rtdetr-weights",
            config["weights"]["rtdetr"],
            "--models",
            *models,
            "--out-dir",
            str(out_base / "full_pipeline"),
            "--conf",
            str(runtime["conf"]),
            "--imgsz",
            str(runtime["imgsz"]),
            "--density-model",
            config["density"].get("model", "yolo"),
            "--density-rows",
            str(config["density"].get("rows", 3)),
            "--density-cols",
            str(config["density"].get("cols", 3)),
            "--density-limit",
            str(config["density"].get("limit", 20)),
        ]
        if "frcnn" in models:
            args.extend(["--frcnn-weights", config["weights"]["frcnn"]])
        if paths.get("gt_coco") and resolve_path(paths["gt_coco"]).exists():
            args.extend(["--gt-coco", paths["gt_coco"]])
        else:
            args.extend(["--gt-yolo-labels", paths["gt_yolo_labels"]])
        device = str(runtime.get("device", "")).strip()
        if device:
            args.extend(["--device", device])
        result = run_command(python_command(config, "run_full_pipeline.py", args))
        render_command_result(result)

    st.subheader("Служебные проверки")
    if st.button("Smoke-проверка через выбранный runtime", use_container_width=True):
        result = run_command(python_command(config, "scripts/smoke_cli.py", []))
        render_command_result(result)


def main() -> None:
    st.set_page_config(page_title="ShelfVision Control Panel", page_icon="🧰", layout="wide")
    ensure_config_exists()
    config = load_config()
    config.setdefault("runtime", {}).setdefault("use_wsl_runtime", True)

    st.title("🧰 ShelfVision Control Panel")
    st.caption("Панель первого запуска и управления. Рабочие задачи по умолчанию запускаются через WSL .venv_wsl.")

    page = st.sidebar.radio(
        "Раздел",
        [
            "Первый запуск",
            "Скачивание файлов",
            "Настройки",
            "Запуск задач",
            "Результаты",
            "config YAML",
        ],
    )

    if page == "Первый запуск":
        page_setup(config)
    elif page == "Скачивание файлов":
        page_downloads(config)
    elif page == "Настройки":
        config = page_config_wsl(config)
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
