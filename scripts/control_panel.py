from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

from ui_settings import is_advanced, render_settings_mode_switch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "shelfvision.yaml"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "shelfvision.example.yaml"


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "images_dir": "data/test/images",
        "image": "data/test/image_001.jpg",
        "gt_coco": "data/test/annotations.json",
        "gt_yolo_labels": "data/test/labels",
        "out_dir": "results/control_panel",
    },
    "weights": {
        "yolo": "models/yolo/best.pt",
        "rtdetr": "models/rtdetr/best.pt",
        "frcnn": "models/faster_rcnn/model_final.pth",
    },
    "runtime": {
        "conf": 0.25,
        "imgsz": 640,
        "device": "0",
        "models": ["yolo", "rtdetr", "wbf"],
    },
    "wbf": {
        "iou": 0.55,
        "skip": 0.001,
        "yolo_weight": 1.0,
        "rtdetr_weight": 1.0,
    },
    "density": {
        "model": "yolo",
        "rows": 3,
        "cols": 3,
        "limit": 20,
    },
    "setup": {
        "venv_dir": ".venv",
        "venv_dir_wsl": ".venv_wsl",
        "requirements": "requirements.txt",
        "downloads": [
            {"name": "yolo_weights", "url": "", "output": "models/yolo/best.pt"},
            {"name": "rtdetr_weights", "url": "", "output": "models/rtdetr/best.pt"},
            {"name": "test_archive", "url": "", "output": "data/downloads/test_data.zip"},
        ],
    },
}


MODEL_LABELS = {
    "yolo": "YOLO",
    "rtdetr": "RT-DETR-L",
    "frcnn": "Faster R-CNN",
    "wbf": "WBF (YOLO + RT-DETR)",
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return deep_merge(DEFAULT_CONFIG, data)
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).strip().strip('"').strip("'"))
    if path.is_absolute():
        return path
    return ROOT / path


def rel_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def venv_python(config: Dict[str, Any]) -> Path:
    venv_dir = resolve_path(config["setup"].get("venv_dir", ".venv"))
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_command(cmd: List[str], cwd: Path = ROOT, timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def render_command_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        st.success(f"Команда выполнена успешно: returncode={result.returncode}")
    else:
        st.error(f"Команда завершилась с ошибкой: returncode={result.returncode}")
    st.code(result.stdout or "", language="text")


def ensure_config_exists() -> None:
    if DEFAULT_CONFIG_PATH.exists():
        return
    if EXAMPLE_CONFIG_PATH.exists():
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXAMPLE_CONFIG_PATH, DEFAULT_CONFIG_PATH)
    else:
        save_config(DEFAULT_CONFIG, DEFAULT_CONFIG_PATH)


def download_file(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        output.write_bytes(response.read())
    return output


def check_path(label: str, value: str, should_exist: bool = True) -> None:
    path = resolve_path(value)
    if should_exist and path.exists():
        st.success(f"{label}: найдено — {rel_path(path)}")
    elif should_exist:
        st.warning(f"{label}: не найдено — {rel_path(path)}")
    else:
        st.info(f"{label}: {rel_path(path)}")


def build_weight_args(config: Dict[str, Any], model: str) -> List[str]:
    weights = config["weights"]
    wbf = config["wbf"]
    if model == "yolo":
        return ["--weights", weights["yolo"]]
    if model == "rtdetr":
        return ["--weights", weights["rtdetr"]]
    if model == "frcnn":
        return ["--weights", weights["frcnn"]]
    if model == "wbf":
        return [
            "--yolo-weights",
            weights["yolo"],
            "--rtdetr-weights",
            weights["rtdetr"],
            "--wbf-iou",
            str(wbf["iou"]),
            "--wbf-skip",
            str(wbf["skip"]),
            "--yolo-weight",
            str(wbf["yolo_weight"]),
            "--rtdetr-weight",
            str(wbf["rtdetr_weight"]),
        ]
    raise ValueError(model)


def page_setup(config: Dict[str, Any]) -> None:
    st.header("1. Первый запуск и установка")
    st.caption("Здесь собраны действия, которые обычно приходится делать вручную после скачивания проекта.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Создать/обновить config/shelfvision.yaml", use_container_width=True):
            path = save_config(config)
            st.success(f"Конфигурация сохранена: {rel_path(path)}")

        if st.button("Создать виртуальную среду .venv (Windows/local)", use_container_width=True):
            venv_dir = resolve_path(config["setup"].get("venv_dir", ".venv"))
            result = run_command([sys.executable, "-m", "venv", str(venv_dir)])
            render_command_result(result)

        if st.button("Обновить pip в .venv (Windows/local)", use_container_width=True):
            py = venv_python(config)
            result = run_command([str(py), "-m", "pip", "install", "--upgrade", "pip"])
            render_command_result(result)

        if st.button("Установить зависимости в .venv (Windows/local)", use_container_width=True):
            py = venv_python(config)
            req = resolve_path(config["setup"].get("requirements", "requirements.txt"))
            result = run_command([str(py), "-m", "pip", "install", "-r", str(req)])
            render_command_result(result)

    with col2:
        st.subheader("WSL")
        st.warning(
            "Установка WSL может требовать прав администратора и перезагрузки. "
            "Для установки зависимостей через WSL должна быть установлена Linux-система и пакеты python3, python3-venv, python3-pip."
        )
        if st.button("Проверить WSL", use_container_width=True):
            result = run_command(["wsl", "--status"])
            render_command_result(result)

        if st.button("Запустить wsl --install", use_container_width=True):
            result = run_command(["wsl", "--install"])
            render_command_result(result)

        if st.button("Создать WSL venv и установить зависимости", use_container_width=True):
            result = run_command(
                [
                    sys.executable,
                    "scripts/setup_wsl_env.py",
                    "--venv-dir",
                    config["setup"].get("venv_dir_wsl", ".venv_wsl"),
                    "--requirements",
                    config["setup"].get("requirements", "requirements.txt"),
                ]
            )
            render_command_result(result)

        if st.button("Smoke-проверка CLI", use_container_width=True):
            py = venv_python(config)
            result = run_command([str(py), "scripts/smoke_cli.py"])
            render_command_result(result)

    st.subheader("Проверка путей")
    check_path("requirements.txt", config["setup"].get("requirements", "requirements.txt"))
    check_path("Python .venv", str(venv_python(config)))
    check_path("WSL venv", config["setup"].get("venv_dir_wsl", ".venv_wsl"))
    check_path("Веса YOLO", config["weights"].get("yolo", ""))
    check_path("Веса RT-DETR", config["weights"].get("rtdetr", ""))
    check_path("Веса Faster R-CNN", config["weights"].get("frcnn", ""))
    check_path("Папка изображений", config["paths"].get("images_dir", ""))


def page_downloads(config: Dict[str, Any]) -> None:
    st.header("2. Скачивание файлов")
    render_settings_mode_switch(config, page_key="downloads")
    advanced = is_advanced(config, page_key="downloads")
    st.caption("Сюда можно добавить ссылки на веса моделей, архивы датасета или другие файлы.")

    downloads = config["setup"].setdefault("downloads", [])

    if advanced:
        edited = st.data_editor(
            downloads,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Название"),
                "url": st.column_config.TextColumn("URL"),
                "output": st.column_config.TextColumn("Куда сохранить"),
            },
        )
        config["setup"]["downloads"] = edited

        if st.button("Сохранить список загрузок", use_container_width=True):
            save_config(config)
            st.success("Список загрузок сохранён в config/shelfvision.yaml")
    else:
        edited = [item for item in downloads if item.get("output") or item.get("url")]
        st.info("Показаны только уже заданные загрузки. В расширенном режиме можно добавлять, удалять и редактировать URL/output.")

    st.divider()
    for item in edited:
        name = item.get("name") or "файл"
        url = item.get("url") or ""
        output = item.get("output") or ""
        expanded = advanced
        with st.expander(f"{name}: {output}", expanded=expanded):
            if advanced:
                st.write(f"URL: `{url or 'не указан'}`")
            elif not url:
                st.caption("URL не задан. Включи расширенный режим, чтобы отредактировать список загрузок.")
            st.write(f"Куда сохранить: `{output or 'не указан'}`")
            if output:
                check_path("Файл", output, should_exist=True)
            if st.button(f"Скачать {name}", key=f"download_{name}_{output}"):
                if not url or not output:
                    st.error("Для скачивания нужно указать URL и путь сохранения.")
                else:
                    try:
                        saved = download_file(url, resolve_path(output))
                        st.success(f"Файл скачан: {rel_path(saved)}")
                    except Exception as exc:
                        st.error(f"Ошибка скачивания: {exc}")


def page_config(config: Dict[str, Any]) -> Dict[str, Any]:
    st.header("3. Настройки проекта")

    with st.form("config_form"):
        st.subheader("Пути")
        paths = config["paths"]
        paths["image"] = st.text_input("Одно изображение", value=str(paths.get("image", "")))
        paths["images_dir"] = st.text_input("Папка изображений", value=str(paths.get("images_dir", "")))
        paths["gt_coco"] = st.text_input("Файл COCO annotations.json", value=str(paths.get("gt_coco", "")))
        paths["gt_yolo_labels"] = st.text_input("Папка YOLO labels", value=str(paths.get("gt_yolo_labels", "")))
        paths["out_dir"] = st.text_input("Папка результатов", value=str(paths.get("out_dir", "")))

        st.subheader("Веса моделей")
        weights = config["weights"]
        weights["yolo"] = st.text_input("Веса YOLO", value=str(weights.get("yolo", "")))
        weights["rtdetr"] = st.text_input("Веса RT-DETR", value=str(weights.get("rtdetr", "")))
        weights["frcnn"] = st.text_input("Веса Faster R-CNN", value=str(weights.get("frcnn", "")))

        st.subheader("Настройка окружения")
        setup = config["setup"]
        setup["venv_dir"] = st.text_input("Windows/local venv", value=str(setup.get("venv_dir", ".venv")))
        setup["venv_dir_wsl"] = st.text_input("WSL venv", value=str(setup.get("venv_dir_wsl", ".venv_wsl")))
        setup["requirements"] = st.text_input("Файл requirements.txt", value=str(setup.get("requirements", "requirements.txt")))

        st.subheader("Параметры инференса")
        runtime = config["runtime"]
        runtime["conf"] = st.slider("Порог confidence", 0.01, 0.95, float(runtime.get("conf", 0.25)), 0.01)
        runtime["imgsz"] = st.selectbox("Размер изображения", [416, 512, 640, 768, 1024], index=[416, 512, 640, 768, 1024].index(int(runtime.get("imgsz", 640))) if int(runtime.get("imgsz", 640)) in [416, 512, 640, 768, 1024] else 2)
        runtime["device"] = st.text_input("Устройство запуска", value=str(runtime.get("device", "0")))
        runtime["models"] = st.multiselect(
            "Модели для полного пайплайна",
            options=list(MODEL_LABELS.keys()),
            default=[m for m in runtime.get("models", ["yolo", "rtdetr", "wbf"]) if m in MODEL_LABELS],
            format_func=lambda x: MODEL_LABELS[x],
        )

        st.subheader("Объединение предсказаний WBF")
        wbf = config["wbf"]
        wbf["iou"] = st.slider("IoU-порог WBF", 0.1, 0.9, float(wbf.get("iou", 0.55)), 0.01)
        wbf["skip"] = st.slider("Порог пропуска WBF", 0.0, 0.5, float(wbf.get("skip", 0.001)), 0.001)
        wbf["yolo_weight"] = st.number_input("Вес YOLO", 0.1, 5.0, float(wbf.get("yolo_weight", 1.0)), 0.1)
        wbf["rtdetr_weight"] = st.number_input("Вес RT-DETR", 0.1, 5.0, float(wbf.get("rtdetr_weight", 1.0)), 0.1)

        st.subheader("Плотность")
        density = config["density"]
        density["model"] = st.selectbox("Модель для анализа плотности", list(MODEL_LABELS.keys()), index=list(MODEL_LABELS.keys()).index(density.get("model", "yolo")) if density.get("model", "yolo") in MODEL_LABELS else 0, format_func=lambda x: MODEL_LABELS[x])
        density["rows"] = st.number_input("Строк сетки", 1, 10, int(density.get("rows", 3)))
        density["cols"] = st.number_input("Столбцов сетки", 1, 10, int(density.get("cols", 3)))
        density["limit"] = st.number_input("Лимит визуализаций", 0, 1000, int(density.get("limit", 20)))

        submitted = st.form_submit_button("Сохранить настройки")
        if submitted:
            save_config(config)
            st.success("Настройки сохранены")
    return config


def page_actions(config: Dict[str, Any]) -> None:
    st.header("4. Запуск задач кнопками")
    py = venv_python(config)
    paths = config["paths"]
    runtime = config["runtime"]
    out_base = Path(paths.get("out_dir", "results/control_panel"))

    st.subheader("Интерфейсы")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Открыть интерфейс экспериментов", use_container_width=True):
            result = run_command([str(py), "-m", "streamlit", "run", "scripts/interface_app.py"])
            render_command_result(result)
    with c2:
        if st.button("Открыть интерфейс инференса", use_container_width=True):
            result = run_command([str(py), "-m", "streamlit", "run", "scripts/inference_app.py"])
            render_command_result(result)

    st.subheader("Инференс одного изображения")
    selected_model = st.selectbox("Модель", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])
    if st.button("Запустить инференс", use_container_width=True):
        out_dir = out_base / "inference" / selected_model
        cmd = [
            str(py),
            "run_inference.py",
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
            cmd.extend(["--device", device])
        result = run_command(cmd)
        render_command_result(result)

    st.subheader("Полный пайплайн")
    if st.button("Запустить полный пайплайн", use_container_width=True):
        models = runtime.get("models", ["yolo", "rtdetr", "wbf"])
        cmd = [
            str(py),
            "run_full_pipeline.py",
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
            cmd.extend(["--frcnn-weights", config["weights"]["frcnn"]])
        if paths.get("gt_coco") and resolve_path(paths["gt_coco"]).exists():
            cmd.extend(["--gt-coco", paths["gt_coco"]])
        else:
            cmd.extend(["--gt-yolo-labels", paths["gt_yolo_labels"]])
        device = str(runtime.get("device", "")).strip()
        if device:
            cmd.extend(["--device", device])
        result = run_command(cmd)
        render_command_result(result)

    st.subheader("Служебные проверки")
    if st.button("Smoke-проверка", use_container_width=True):
        result = run_command([str(py), "scripts/smoke_cli.py"])
        render_command_result(result)


def page_results(config: Dict[str, Any]) -> None:
    st.header("5. Результаты")
    out_dir = resolve_path(config["paths"].get("out_dir", "results/control_panel"))
    st.write(f"Папка результатов: `{rel_path(out_dir)}`")

    if not out_dir.exists():
        st.info("Пока папка результатов не создана.")
        return

    files = sorted([p for p in out_dir.rglob("*") if p.is_file()])
    st.write(f"Файлов найдено: {len(files)}")

    for path in files[:200]:
        st.write(f"- `{rel_path(path)}`")

    report_candidates = list(out_dir.rglob("mini_report.html"))
    if report_candidates:
        st.subheader("Мини-отчёт")
        st.write(f"HTML: `{rel_path(report_candidates[0])}`")


def main() -> None:
    st.set_page_config(page_title="Панель управления ShelfVision", page_icon="🧰", layout="wide")
    ensure_config_exists()
    config = load_config()

    st.title("🧰 Панель управления ShelfVision")
    st.caption("Мастер первого запуска, настройки и кнопки для основных сценариев ВКР-программы.")

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
        config = page_config(config)
    elif page == "Запуск задач":
        page_actions(config)
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
