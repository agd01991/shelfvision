from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from path_utils import to_current_os_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_PORT = 8515
MAIN_APP = "scripts/control_panel_wsl_app.py"


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _launch_main_interface(port: int) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        MAIN_APP,
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(int(port)),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _main_url(port: int) -> str:
    return f"http://localhost:{int(port)}"


def page_main_interface_bridge(config: Dict[str, Any], experiment_dir: str | Path) -> None:
    st.subheader("Переход в основной интерфейс")
    st.caption(
        "Основной интерфейс предназначен для воспроизведения экспериментов, настройки путей, "
        "запуска полного контура, ручной проверки и просмотра всех результатов."
    )

    exp = _p(experiment_dir)
    st.info(
        "Демонстрационный интерфейс показывает подготовленный результат для защиты. "
        "Основной интерфейс используется как рабочая панель экспериментов: там можно менять конфигурацию, "
        "запускать полный pipeline, смотреть результаты, аудит SKU и ручную корректировку."
    )
    st.write(f"Текущий каталог результатов из демо: `{exp}`")

    port = st.number_input(
        "Порт основного интерфейса",
        min_value=8501,
        max_value=8999,
        value=int(st.session_state.get("main_interface_port", DEFAULT_MAIN_PORT)),
        step=1,
        key="main_interface_port_input",
    )
    st.session_state["main_interface_port"] = int(port)
    url = _main_url(int(port))

    running = _is_port_open(int(port))
    c1, c2 = st.columns([1, 1])
    with c1:
        if running:
            st.success(f"Основной интерфейс уже доступен: {url}")
        else:
            st.warning("Основной интерфейс на выбранном порту пока не запущен.")
    with c2:
        st.link_button("Открыть основной интерфейс", url, use_container_width=True, disabled=not running)

    if st.button("Запустить основной интерфейс", type="primary", use_container_width=True, disabled=running):
        process = _launch_main_interface(int(port))
        st.session_state["main_interface_pid"] = process.pid
        time.sleep(1.5)
        if _is_port_open(int(port)):
            st.success(f"Основной интерфейс запущен: {url}")
            st.link_button("Перейти в основной интерфейс", url, use_container_width=True)
        else:
            st.info(
                "Процесс запуска отправлен. Подождите несколько секунд и нажмите кнопку открытия ещё раз. "
                "Если ссылка не открылась, проверьте терминал запуска демо."
            )

    if "main_interface_pid" in st.session_state:
        st.caption(f"PID последнего запущенного процесса: {st.session_state['main_interface_pid']}")

    st.markdown("#### Что делать в основном интерфейсе")
    st.markdown(
        """
1. **Настройки** — проверить пути к данным, весам моделей, результирующим папкам и режим WSL.
2. **Запуск и проверка** — воспроизвести полный контур фото-идентификации gallery/query.
3. **Дополнительные инструменты проверки SKU** — запустить аудит похожих SKU, смешанных кластеров и ручную корректировку.
4. **Результаты** — открыть папки экспериментов, отчеты, CSV/JSON/MD и визуализации.
5. **config YAML** — быстро проверить или сохранить текущую конфигурацию.
"""
    )

    st.markdown("#### Команда ручного запуска")
    st.code(
        "PYTHONPATH=. .venv_wsl/bin/python -m streamlit run scripts/control_panel_wsl_app.py "
        f"--server.address 127.0.0.1 --server.port {int(port)} --browser.gatherUsageStats false",
        language="bash",
    )

    with st.expander("Связь с текущим демо", expanded=False):
        full = dict(config.get("full_photo_identification", {}) or {})
        st.write(
            {
                "demo_experiment_dir": str(exp),
                "configured_full_photo_out_dir": full.get("out_dir", ""),
                "configured_gallery_dir": full.get("gallery_dir", ""),
                "configured_gallery_csv": full.get("gallery_csv", ""),
            }
        )
        st.caption(
            "Если нужно, чтобы основной интерфейс работал именно с текущей папкой демо, "
            "укажите этот путь в разделе `Настройки` основного интерфейса в блоке full_photo_identification.out_dir."
        )
