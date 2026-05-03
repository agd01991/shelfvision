from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import streamlit as st


@dataclass
class CommandStep:
    title: str
    cmd: List[str]
    cwd: Path
    description: str = ""


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)


def run_command_with_live_log(
    step: CommandStep,
    log_placeholder,
    command_placeholder,
    max_log_lines: int = 250,
) -> int:
    command_placeholder.info(f"Выполняется: `{_format_cmd(step.cmd)}`")

    log_lines: List[str] = []
    start = time.perf_counter()

    try:
        process = subprocess.Popen(
            step.cmd,
            cwd=str(step.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:
        log_placeholder.code(f"Не удалось запустить команду:\n{exc}", language="text")
        return 1

    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.rstrip()
        if clean_line:
            log_lines.append(clean_line)
        visible = "\n".join(log_lines[-max_log_lines:])
        log_placeholder.code(visible or "Ожидание вывода команды...", language="text")

    return_code = process.wait()
    elapsed = time.perf_counter() - start

    if not log_lines:
        log_lines.append("Команда завершилась без текстового вывода.")
    log_lines.append(f"\nreturncode={return_code}, elapsed={elapsed:.1f}s")
    log_placeholder.code("\n".join(log_lines[-max_log_lines:]), language="text")
    return return_code


def run_steps_with_progress(
    steps: List[CommandStep],
    title: str = "Выполнение команды",
    success_message: str = "Готово",
    failure_message: str = "Команда завершилась с ошибкой",
) -> bool:
    if not steps:
        st.warning("Нет шагов для выполнения.")
        return False

    st.subheader(title)
    progress = st.progress(0, text="Подготовка...")
    status_placeholder = st.empty()
    command_placeholder = st.empty()
    log_placeholder = st.empty()

    total = len(steps)
    for idx, step in enumerate(steps, start=1):
        progress.progress((idx - 1) / total, text=f"[{idx}/{total}] {step.title}")
        if step.description:
            status_placeholder.info(step.description)
        else:
            status_placeholder.info(f"Выполняется шаг {idx} из {total}: {step.title}")

        return_code = run_command_with_live_log(step, log_placeholder, command_placeholder)
        if return_code != 0:
            progress.progress(idx / total, text=f"Ошибка на шаге: {step.title}")
            status_placeholder.error(f"{failure_message}: {step.title}, returncode={return_code}")
            return False

    progress.progress(1.0, text="Готово")
    command_placeholder.empty()
    status_placeholder.success(success_message)
    return True
