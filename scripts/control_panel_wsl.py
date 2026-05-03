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
    page_setup,
    rel_path,
    resolve_path,
    run_command,
    save_config,
    venv_python,
)


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
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Открыть интерфейс экспериментов", use_container_width=True):
            result = run_command(streamlit_command(config, "scripts/interface_app.py"))
            render_command_result(result)
    with c2:
        if st.button("Открыть интерфейс инференса", use_container_width=True):
            result = run_command(streamlit_command(config, "scripts/inference_app.py"))
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
