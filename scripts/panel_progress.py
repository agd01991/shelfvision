from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

import streamlit as st


T = TypeVar("T")


@dataclass
class CommandStep:
    title: str
    cmd: List[str]
    cwd: Path
    description: str = ""
    estimated_seconds: Optional[int] = None


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _eta_text(elapsed: float, estimated_seconds: Optional[int]) -> str:
    if not estimated_seconds or estimated_seconds <= 0:
        return "оставшееся время: оценивается по ходу выполнения"

    remaining = estimated_seconds - elapsed
    if remaining <= 0:
        overrun = elapsed - estimated_seconds
        return (
            "операция идёт дольше первичной оценки "
            f"на {_format_duration(overrun)}; точное оставшееся время определить нельзя"
        )

    return f"примерно осталось: {_format_duration(remaining)}"


def _progress_fraction(elapsed: float, estimated_seconds: Optional[int]) -> float:
    if not estimated_seconds or estimated_seconds <= 0:
        return min(0.95, 0.1 + (elapsed % 45) / 55)

    if elapsed <= estimated_seconds:
        return min(0.9, max(0.0, elapsed / estimated_seconds * 0.9))

    # После превышения оценки не показываем, будто задача почти завершена.
    # Держим живую анимацию в диапазоне 90-97%.
    return 0.9 + ((elapsed - estimated_seconds) % 20) / 20 * 0.07


def _reader_thread(stdout, output_queue: "queue.Queue[str]") -> None:
    try:
        for line in stdout:
            output_queue.put(line.rstrip())
    finally:
        output_queue.put("__STREAM_CLOSED__")


def _render_live_state(
    timer_placeholder,
    log_placeholder,
    log_lines: List[str],
    start: float,
    estimated_seconds: Optional[int],
    process_alive: bool,
    max_log_lines: int,
    hint: str,
) -> None:
    elapsed = time.perf_counter() - start
    eta = _eta_text(elapsed, estimated_seconds)
    timer_placeholder.info(f"⏱ Прошло: **{_format_duration(elapsed)}** · {eta}. {hint}")
    visible = "\n".join(log_lines[-max_log_lines:])
    if not visible and process_alive:
        visible = "Команда запущена. Ожидание первого вывода..."
    log_placeholder.code(visible or "Команда завершилась без текстового вывода.", language="text")


def run_long_task_with_progress(
    func: Callable[[], T],
    title: str,
    description: str = "",
    estimated_seconds: Optional[int] = None,
    progress_text: str = "Выполняется операция...",
    status_text: str = "Операция выполняется...",
    hint: str = "Если операция идёт по большим папкам или зависит от сети, это может занять несколько минут.",
) -> T:
    """Runs a long synchronous Python task in a worker thread and keeps Streamlit UI alive.

    Useful for tasks that do not stream stdout, for example local filesystem search.
    """

    st.subheader(title)
    if description:
        st.info(description)

    progress = st.progress(0.0, text="Подготовка...")
    timer_placeholder = st.empty()
    status_placeholder = st.empty()

    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def target() -> None:
        try:
            result_queue.put(("result", func()))
        except Exception as exc:  # noqa: BLE001 - показываем ошибку в интерфейсе
            result_queue.put(("error", exc))

    start = time.perf_counter()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()

    while worker.is_alive():
        elapsed = time.perf_counter() - start
        progress.progress(
            _progress_fraction(elapsed, estimated_seconds),
            text=f"{progress_text}: прошло {_format_duration(elapsed)}, {_eta_text(elapsed, estimated_seconds)}",
        )
        timer_placeholder.info(f"⏱ Прошло: **{_format_duration(elapsed)}** · {_eta_text(elapsed, estimated_seconds)}. {hint}")
        status_placeholder.info(status_text)
        time.sleep(1.0)

    elapsed = time.perf_counter() - start
    kind, payload = result_queue.get()
    progress.progress(1.0, text=f"Готово за {_format_duration(elapsed)}")
    timer_placeholder.success(f"⏱ Общее время выполнения: **{_format_duration(elapsed)}**")

    if kind == "error":
        status_placeholder.error(f"Операция завершилась с ошибкой: {payload}")
        raise payload  # type: ignore[misc]

    status_placeholder.success("Операция завершена")
    return payload  # type: ignore[return-value]


def run_command_with_live_log(
    step: CommandStep,
    log_placeholder,
    command_placeholder,
    timer_placeholder=None,
    step_progress_placeholder=None,
    max_log_lines: int = 250,
) -> int:
    command_placeholder.info(f"Выполняется: `{_format_cmd(step.cmd)}`")
    timer_placeholder = timer_placeholder or st.empty()
    step_progress_placeholder = step_progress_placeholder or st.empty()

    log_lines: List[str] = []
    start = time.perf_counter()
    output_queue: "queue.Queue[str]" = queue.Queue()

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
    thread = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    thread.start()

    stream_closed = False
    last_render = 0.0

    while process.poll() is None or not output_queue.empty() or not stream_closed:
        changed = False
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            if line == "__STREAM_CLOSED__":
                stream_closed = True
                continue
            if line:
                log_lines.append(line)
                changed = True

        now = time.perf_counter()
        elapsed = now - start
        if changed or now - last_render >= 1.0:
            last_render = now
            step_progress_placeholder.progress(
                _progress_fraction(elapsed, step.estimated_seconds),
                text=f"{step.title}: прошло {_format_duration(elapsed)}, {_eta_text(elapsed, step.estimated_seconds)}",
            )
            _render_live_state(
                timer_placeholder=timer_placeholder,
                log_placeholder=log_placeholder,
                log_lines=log_lines,
                start=start,
                estimated_seconds=step.estimated_seconds,
                process_alive=process.poll() is None,
                max_log_lines=max_log_lines,
                hint="Если лог не меняется несколько минут, это всё ещё может быть нормальной установкой тяжёлых пакетов.",
            )

        time.sleep(0.2)

    return_code = process.wait()
    elapsed = time.perf_counter() - start

    if not log_lines:
        log_lines.append("Команда завершилась без текстового вывода.")
    log_lines.append(f"\nreturncode={return_code}, elapsed={elapsed:.1f}s")
    step_progress_placeholder.progress(1.0, text=f"{step.title}: завершено за {_format_duration(elapsed)}")
    timer_placeholder.info(f"⏱ Итого: **{_format_duration(elapsed)}** · returncode={return_code}")
    log_placeholder.code("\n".join(log_lines[-max_log_lines:]), language="text")
    return return_code


def _total_estimated_seconds(steps: List[CommandStep]) -> Optional[int]:
    estimates = [step.estimated_seconds for step in steps if step.estimated_seconds and step.estimated_seconds > 0]
    if len(estimates) != len(steps):
        return None
    return int(sum(estimates))


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
    step_progress_placeholder = st.empty()
    timer_placeholder = st.empty()
    status_placeholder = st.empty()
    command_placeholder = st.empty()
    log_placeholder = st.empty()

    total = len(steps)
    total_estimate = _total_estimated_seconds(steps)
    all_start = time.perf_counter()

    for idx, step in enumerate(steps, start=1):
        elapsed_total = time.perf_counter() - all_start
        total_eta = _eta_text(elapsed_total, total_estimate) if total_estimate else "оставшееся время зависит от текущего шага"
        progress.progress(
            _progress_fraction(elapsed_total, total_estimate),
            text=f"[{idx}/{total}] {step.title} · прошло {_format_duration(elapsed_total)} · {total_eta}",
        )

        if step.description:
            status_placeholder.info(step.description)
        else:
            status_placeholder.info(f"Выполняется шаг {idx} из {total}: {step.title}")

        return_code = run_command_with_live_log(
            step,
            log_placeholder,
            command_placeholder,
            timer_placeholder=timer_placeholder,
            step_progress_placeholder=step_progress_placeholder,
        )
        if return_code != 0:
            progress.progress(idx / total, text=f"Ошибка на шаге: {step.title}")
            status_placeholder.error(f"{failure_message}: {step.title}, returncode={return_code}")
            return False

    elapsed_total = time.perf_counter() - all_start
    progress.progress(1.0, text=f"Готово за {_format_duration(elapsed_total)}")
    command_placeholder.empty()
    step_progress_placeholder.empty()
    timer_placeholder.success(f"⏱ Общее время выполнения: **{_format_duration(elapsed_total)}**")
    status_placeholder.success(success_message)
    return True
