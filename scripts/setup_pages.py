from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from asset_discovery import AssetCandidate, discover_assets
from control_panel import ROOT, check_path, rel_path, resolve_path, save_config, venv_python
from panel_progress import CommandStep, run_long_task_with_progress, run_steps_with_progress


def _windows_python_path(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _wsl_bash_command(script: str) -> List[str]:
    return ["wsl", "bash", "-lc", script]


def _wsl_python_script(config: Dict[str, Any], script: str, extra_args: List[str] | None = None) -> str:
    venv_dir = str(config["setup"].get("venv_dir_wsl", ".venv_wsl")).replace("\\", "/")
    extra_args = extra_args or []
    safe_args = " ".join(f"'{arg}'" for arg in extra_args)
    return (
        f"cd \"$(wslpath '{ROOT}')\" && "
        f"if [ ! -x '{venv_dir}/bin/python' ]; then "
        f"echo 'WSL virtual environment not found: {venv_dir}/bin/python'; "
        f"echo 'Сначала создайте WSL .venv_wsl или установите зависимости.'; exit 2; fi && "
        f"'{venv_dir}/bin/python' '{script}' {safe_args}"
    )


def _wsl_pip_install_script(config: Dict[str, Any]) -> str:
    venv_dir = str(config["setup"].get("venv_dir_wsl", ".venv_wsl")).replace("\\", "/")
    requirements = str(config["setup"].get("requirements", "requirements.txt")).replace("\\", "/")
    return (
        f"cd \"$(wslpath '{ROOT}')\" && "
        f"if [ ! -x '{venv_dir}/bin/python' ]; then "
        f"echo 'WSL virtual environment not found: {venv_dir}/bin/python'; "
        f"echo 'Сначала нажмите Создать WSL venv и установить зависимости.'; exit 2; fi && "
        f"'{venv_dir}/bin/python' -m pip install -r '{requirements}'"
    )


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
            estimated_seconds=45,
        ),
        CommandStep(
            title="Обновление pip",
            cmd=[str(py), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=ROOT,
            description="Обновляется pip внутри Windows .venv.",
            estimated_seconds=45,
        ),
        CommandStep(
            title="Установка минимальных пакетов панели",
            cmd=[str(py), "-m", "pip", "install", "streamlit", "PyYAML", "pandas"],
            cwd=ROOT,
            description="Ставятся минимальные зависимости, чтобы открыть Control Panel.",
            estimated_seconds=180,
        ),
    ]

    if install_full_requirements:
        steps.append(
            CommandStep(
                title="Установка всех зависимостей requirements.txt в Windows .venv",
                cmd=[str(py), "-m", "pip", "install", "-r", str(req)],
                cwd=ROOT,
                description="Ставятся все зависимости проекта в Windows .venv. Для рабочей схемы через WSL это обычно не требуется.",
                estimated_seconds=1800,
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
            description=(
                "Команда создаёт Linux-среду внутри WSL и устанавливает зависимости проекта. "
                "На этапе torch/ultralytics/pip загрузка может выглядеть долгой, но таймер ниже показывает, что процесс живой."
            ),
            estimated_seconds=2400,
        )
    ]


def _wsl_check_dependency_steps(config: Dict[str, Any], strict: bool = False) -> List[CommandStep]:
    requirements = str(config["setup"].get("requirements", "requirements.txt")).replace("\\", "/")
    args = ["--requirements", requirements]
    if strict:
        args.append("--strict")
    return [
        CommandStep(
            title="Проверка зависимостей в WSL .venv_wsl",
            cmd=_wsl_bash_command(_wsl_python_script(config, "scripts/check_dependencies.py", args)),
            cwd=ROOT,
            description="Проверяется, какие пакеты из requirements.txt уже установлены в WSL .venv_wsl, без установки новых пакетов.",
            estimated_seconds=30,
        )
    ]


def _wsl_install_missing_steps(config: Dict[str, Any]) -> List[CommandStep]:
    return [
        CommandStep(
            title="Доустановка зависимостей в существующую WSL .venv_wsl",
            cmd=_wsl_bash_command(_wsl_pip_install_script(config)),
            cwd=ROOT,
            description=(
                "Запускается pip install -r requirements.txt в уже созданной WSL-среде. "
                "pip сам пропустит уже установленные подходящие пакеты и доустановит недостающие."
            ),
            estimated_seconds=1800,
        ),
        *_wsl_check_dependency_steps(config, strict=False),
    ]


def _wsl_reset_steps(config: Dict[str, Any]) -> List[CommandStep]:
    venv_dir = config["setup"].get("venv_dir_wsl", ".venv_wsl")
    return [
        CommandStep(
            title="Удаление старой WSL .venv_wsl",
            cmd=["wsl", "bash", "-lc", f"cd \"$(wslpath '{ROOT}')\" && rm -rf '{venv_dir}'"],
            cwd=ROOT,
            description="Удаляется старая WSL-среда. Использовать только если зависимости сломались.",
            estimated_seconds=60,
        ),
        *_wsl_setup_steps(config),
    ]


def _to_config_path(path: str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _candidate_options(candidates: List[AssetCandidate]) -> List[str]:
    return [candidate.path for candidate in candidates]


def _render_candidate_select(label: str, candidates: List[AssetCandidate], key: str) -> str | None:
    if not candidates:
        st.warning(f"{label}: кандидаты не найдены")
        return None
    options = _candidate_options(candidates)
    selected = st.selectbox(
        label,
        options=options,
        key=key,
        format_func=lambda value: f"{_to_config_path(value)}",
    )
    selected_candidate = next((item for item in candidates if item.path == selected), None)
    if selected_candidate:
        st.caption(f"score={selected_candidate.score}; {selected_candidate.reason}")
    return selected


def _default_search_roots() -> List[str]:
    roots = [
        ROOT,
        ROOT / "models",
        ROOT / "data",
        Path("D:/1Diplom"),
        Path("D:/1Diplom/models"),
        Path("D:/1Diplom/data"),
        Path.home() / "Downloads",
        Path.home() / "Documents",
    ]

    unique: List[str] = []
    seen = set()
    for root in roots:
        value = str(root)
        normalized = value.lower().replace("\\", "/")
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _estimate_asset_discovery_seconds(roots: List[Path]) -> int:
    estimate = 20
    for root in roots:
        root_text = str(root).replace("\\", "/").lower()
        if not root.exists():
            continue
        if root == ROOT or root_text.endswith("/models") or root_text.endswith("/data"):
            estimate += 20
        elif "1diplom" in root_text:
            estimate += 45
        elif "downloads" in root_text or "documents" in root_text:
            estimate += 90
        else:
            estimate += 60
    return max(30, min(600, estimate))


def _render_asset_discovery(config: Dict[str, Any]) -> None:
    st.subheader("Автопоиск файлов")
    st.caption(
        "Можно указать папки, где лежат веса, изображения и видео. Программа найдёт подходящие файлы и предложит записать пути в config."
    )

    default_roots = "\n".join(_default_search_roots())
    raw_roots = st.text_area(
        "Где искать",
        value=default_roots,
        height=170,
        help="Каждая папка с новой строки. Лучше не указывать весь диск C: или D:, чтобы поиск не был слишком долгим. Для твоего случая добавлен D:/1Diplom.",
    )
    limit = st.number_input("Сколько кандидатов показывать на каждый тип", 1, 30, 10)

    if st.button("Найти веса, изображения и видео", use_container_width=True):
        roots = [Path(line.strip()) for line in raw_roots.splitlines() if line.strip()]
        estimate = _estimate_asset_discovery_seconds(roots)
        st.session_state["asset_discovery_results"] = run_long_task_with_progress(
            func=lambda: discover_assets(roots, limit=int(limit)),
            title="Автопоиск файлов",
            description=(
                "Идёт поиск весов моделей, папок изображений и видеофайлов. "
                "Если выбраны большие папки вроде Documents или Downloads, поиск может занять несколько минут."
            ),
            estimated_seconds=estimate,
            progress_text="Поиск файлов",
        )
        st.success("Поиск завершён")

    results = st.session_state.get("asset_discovery_results")
    if not results:
        return

    st.divider()
    st.write("Выбери найденные пути и нажми **Применить выбранные пути в config**.")

    selected: Dict[str, str | None] = {}
    col1, col2 = st.columns(2)
    with col1:
        selected["yolo"] = _render_candidate_select("YOLO weights", results.get("yolo", []), "auto_yolo")
        selected["rtdetr"] = _render_candidate_select("RT-DETR weights", results.get("rtdetr", []), "auto_rtdetr")
        selected["frcnn"] = _render_candidate_select("Faster R-CNN weights", results.get("frcnn", []), "auto_frcnn")
    with col2:
        selected["images_dir"] = _render_candidate_select("Папка изображений", results.get("images_dir", []), "auto_images")
        selected["video"] = _render_candidate_select("Видео", results.get("video", []), "auto_video")

    if st.button("Применить выбранные пути в config", use_container_width=True):
        if selected.get("yolo"):
            config["weights"]["yolo"] = _to_config_path(selected["yolo"] or "")
        if selected.get("rtdetr"):
            config["weights"]["rtdetr"] = _to_config_path(selected["rtdetr"] or "")
        if selected.get("frcnn"):
            config["weights"]["frcnn"] = _to_config_path(selected["frcnn"] or "")
        if selected.get("images_dir"):
            config["paths"]["images_dir"] = _to_config_path(selected["images_dir"] or "")
        if selected.get("video"):
            config.setdefault("video", {})["input_path"] = _to_config_path(selected["video"] or "")
        save_config(config)
        st.success("Выбранные пути сохранены в config/shelfvision.yaml")


def page_setup(config: Dict[str, Any]) -> None:
    st.header("1. Первый запуск и установка")
    st.caption("Здесь видно, какой процесс запущен, какой шаг выполняется, таймер, примерное оставшееся время и живой лог команды.")

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
                [CommandStep("Проверка WSL", ["wsl", "--status"], ROOT, "Проверяется доступность WSL.", estimated_seconds=10)],
                title="Проверка WSL",
                success_message="WSL доступен.",
                failure_message="WSL не отвечает или не установлен",
            )

        if st.button("Проверить зависимости в WSL .venv_wsl", use_container_width=True):
            run_steps_with_progress(
                _wsl_check_dependency_steps(config, strict=False),
                title="Проверка зависимостей WSL .venv_wsl",
                success_message="Проверка зависимостей завершена. Посмотри строки MISSING/VERSION в логе ниже.",
                failure_message="Проверка зависимостей WSL завершилась с ошибкой",
            )

        if st.button("Доустановить недостающие зависимости в WSL .venv_wsl", use_container_width=True):
            run_steps_with_progress(
                _wsl_install_missing_steps(config),
                title="Доустановка зависимостей WSL .venv_wsl",
                success_message="Доустановка завершена, повторная проверка выполнена.",
                failure_message="Ошибка доустановки зависимостей WSL .venv_wsl",
            )

        if st.button("Запустить wsl --install", use_container_width=True):
            run_steps_with_progress(
                [CommandStep("Установка WSL", ["wsl", "--install"], ROOT, "Запускается установка WSL.", estimated_seconds=600)],
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

    st.divider()
    _render_asset_discovery(config)
