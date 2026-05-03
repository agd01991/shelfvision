from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from control_panel import ROOT, check_path, rel_path, resolve_path, save_config, venv_python
from panel_progress import CommandStep, run_steps_with_progress


def _windows_python_path(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _windows_venv_steps(config: Dict[str, Any], install_full_requirements: bool = False) -> List[CommandStep]:
    venv_dir = resolve_path(config["setup"].get("venv_dir", ".venv"))
    py = _windows_python_path(venv_dir)
    req = resolve_path(config["setup"].get("requirements", "requirements.txt"))

    steps = [
        CommandStep(
            title="Создание Windows .venv",
            cmd=[sys.executable, "-m", "venv", str(venv_dir)],
            cwd=ROOT,
            description="Создаётся локальная Windows-среда. Она нужна для запуска Streamlit-панели.",
        ),
        CommandStep(
            title="Обновление pip",
            cmd=[str(py), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=ROOT,
            description="Обновляется pip внутри Windows .venv.",
        ),
        CommandStep(
            title="Установка минимальных пакетов панели",
            cmd=[str(py), "-m", "pip", "install", "streamlit", "PyYAML", "pandas"],
            cwd=ROOT,
            description="Ставятся минимальные зависимости, чтобы открыть Control Panel.",
        ),
    ]

    if install_full_requirements:
        steps.append(
            CommandStep(
                title="Установка всех зависимостей requirements.txt в Windows .venv",
                cmd=[str(py), "-m", "pip", "install", "-r", str(req)],
                cwd=ROOT,
                description="Ставятся все зависимости проекта в Windows .venv. Для рабочей схемы через WSL это обычно не требуется.",
            )
        )
    return steps


def _wsl_setup_steps(config: Dict[str, Any]) -> List[CommandStep]:
    return [
        CommandStep(
            title="Создание WSL .venv_wsl и установка requirements.txt",
            cmd=[
                sys.executable,
                "scripts/setup_wsl_env.py",
                "--venv-dir",
                config["setup"].get("venv_dir_wsl", ".venv_wsl"),
                "--requirements",
                config["setup"].get("requirements", "requirements.txt"),
            ],
            cwd=ROOT,
            description="Команда создаёт Linux-среду внутри WSL и устанавливает зависимости проекта.",
        )
    ]


def _wsl_reset_steps(config: Dict[str, Any]) -> List[CommandStep]:
    venv_dir = config["setup"].get("venv_dir_wsl", ".venv_wsl")
    return [
        CommandStep(
            title="Удаление старой WSL .venv_wsl",
            cmd=["wsl", "bash", "-lc", f"cd \"$(wslpath '{ROOT}')\" && rm -rf '{venv_dir}'"],
            cwd=ROOT,
            description="Удаляется старая WSL-среда. Использовать только если зависимости сломались.",
        ),
        *_wsl_setup_steps(config),
    ]


def page_setup(config: Dict[str, Any]) -> None:
    st.header("1. Первый запуск и установка")
    st.caption("Здесь видно, какой процесс запущен, какой шаг выполняется, и есть живой лог команды.")

    st.info(
        "Обычный режим: Windows `.venv` нужна только для открытия панели, а рабочие зависимости ставятся в WSL `.venv_wsl`. "
        "Пересоздавать среды каждый раз не нужно."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Windows .venv для панели")

        if st.button("Создать/обновить config/shelfvision.yaml", use_container_width=True):
            path = save_config(config)
            st.success(f"Конфигурация сохранена: {rel_path(path)}")

        if st.button("Создать/починить Windows .venv для панели", use_container_width=True):
            ok = run_steps_with_progress(
                _windows_venv_steps(config, install_full_requirements=False),
                title="Создание Windows .venv",
                success_message="Windows .venv готова. Можно запускать Control Panel.",
                failure_message="Не удалось подготовить Windows .venv",
            )
            if ok:
                st.balloons()

        if st.button("Пересоздать Windows .venv с нуля", use_container_width=True):
            venv_dir = resolve_path(config["setup"].get("venv_dir", ".venv"))
            with st.spinner(f"Удаляется {rel_path(venv_dir)}..."):
                _reset_directory(venv_dir)
            ok = run_steps_with_progress(
                _windows_venv_steps(config, install_full_requirements=False),
                title="Полная пересборка Windows .venv",
                success_message="Windows .venv пересоздана.",
                failure_message="Не удалось пересоздать Windows .venv",
            )
            if ok:
                st.balloons()

        if st.button("Установить ВСЕ зависимости в Windows .venv", use_container_width=True):
            st.warning(
                "Обычно для проекта используется WSL .venv_wsl. Установка всех зависимостей в Windows .venv нужна только для отдельного Windows-запуска."
            )
            run_steps_with_progress(
                _windows_venv_steps(config, install_full_requirements=True)[1:],
                title="Установка зависимостей в Windows .venv",
                success_message="Зависимости установлены в Windows .venv.",
                failure_message="Ошибка установки зависимостей в Windows .venv",
            )

    with col2:
        st.subheader("WSL .venv_wsl для рабочих задач")
        st.warning(
            "Если WSL ещё не установлен, кнопка `wsl --install` может потребовать права администратора и перезагрузку."
        )

        if st.button("Проверить WSL", use_container_width=True):
            run_steps_with_progress(
                [CommandStep("Проверка WSL", ["wsl", "--status"], ROOT, "Проверяется доступность WSL.")],
                title="Проверка WSL",
                success_message="WSL доступен.",
                failure_message="WSL не отвечает или не установлен",
            )

        if st.button("Запустить wsl --install", use_container_width=True):
            run_steps_with_progress(
                [CommandStep("Установка WSL", ["wsl", "--install"], ROOT, "Запускается установка WSL.")],
                title="Установка WSL",
                success_message="Команда wsl --install выполнена. Может потребоваться перезагрузка.",
                failure_message="Ошибка выполнения wsl --install",
            )

        if st.button("Создать WSL venv и установить зависимости", use_container_width=True):
            ok = run_steps_with_progress(
                _wsl_setup_steps(config),
                title="Установка зависимостей через WSL",
                success_message="WSL .venv_wsl готова. Рабочие задачи можно запускать через WSL runtime.",
                failure_message="Ошибка установки зависимостей через WSL",
            )
            if ok:
                st.balloons()

        if st.button("Пересоздать WSL .venv_wsl с нуля", use_container_width=True):
            ok = run_steps_with_progress(
                _wsl_reset_steps(config),
                title="Полная пересборка WSL .venv_wsl",
                success_message="WSL .venv_wsl пересоздана и зависимости установлены.",
                failure_message="Ошибка пересборки WSL .venv_wsl",
            )
            if ok:
                st.balloons()

    st.subheader("Проверка путей")
    check_path("requirements.txt", config["setup"].get("requirements", "requirements.txt"))
    check_path("Python .venv", str(venv_python(config)))
    check_path("WSL venv", config["setup"].get("venv_dir_wsl", ".venv_wsl"))
    check_path("YOLO weights", config["weights"].get("yolo", ""))
    check_path("RT-DETR weights", config["weights"].get("rtdetr", ""))
    check_path("Faster R-CNN weights", config["weights"].get("frcnn", ""))
    check_path("Images dir", config["paths"].get("images_dir", ""))
