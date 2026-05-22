from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

import streamlit as st


T = TypeVar("T")
ProgressEmit = Callable[[Dict[str, object]], None]


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


def _duration_to_seconds(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        return float(text)
    except ValueError:
        return None


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

    return 0.9 + ((elapsed - estimated_seconds) % 20) / 20 * 0.07


def _progress_details_text(update: Dict[str, object] | None, fallback: str) -> str:
    if not update:
        return fallback

    current_root = update.get("current_root") or "—"
    current_path = update.get("current_path") or "—"
    return (
        f"Текущая папка: {current_root}\n\n"
        f"Текущий путь: {current_path}\n\n"
        f"Просканировано папок: {update.get('dirs_scanned', 0)}\n"
        f"Просканировано файлов: {update.get('files_scanned', 0)}\n"
        f"Найдено весов: {update.get('weight_files', 0)}\n"
        f"Найдено изображений: {update.get('image_files', 0)}\n"
        f"Папок с изображениями: {update.get('image_dirs', 0)}\n"
        f"Найдено видео: {update.get('video_files', 0)}\n"
        f"Пропущено служебных папок: {update.get('skipped_dirs', 0)}"
    )


def _reader_thread(stdout, output_queue: "queue.Queue[str]") -> None:
    try:
        for line in stdout:
            output_queue.put(line.rstrip())
    finally:
        output_queue.put("__STREAM_CLOSED__")


def _parse_key_value_progress(payload: str) -> Dict[str, object]:
    update: Dict[str, object] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", payload):
        if "/" in value and key in {"processed", "downloaded"}:
            done_raw, total_raw = value.split("/", 1)
            try:
                update[key] = int(done_raw)
                update["total"] = int(total_raw)
            except ValueError:
                update[key] = value
            continue
        try:
            if "." in value:
                update[key] = float(value)
            else:
                update[key] = int(value)
        except ValueError:
            update[key] = value
    return update


def _parse_progress_line(line: str) -> Optional[Dict[str, object]]:
    """Extracts structured progress from command stdout.

    Supported formats:
    - PROGRESS_JSON {"stage":"query", "processed":10, "total":20, "eta_seconds":4}
    - PHOTO_PROGRESS split=query processed=10/20 objects=13 elapsed=00:07 eta=00:04
    - DOWNLOAD_PROGRESS downloaded=1048576/2097152 speed_bps=123456 eta=00:08
    """

    stripped = line.strip()
    if stripped.startswith("PROGRESS_JSON"):
        raw = stripped.removeprefix("PROGRESS_JSON").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            payload.setdefault("kind", "json")
            return payload
        return None

    for prefix in ("PHOTO_PROGRESS", "DOWNLOAD_PROGRESS", "SCAN_PROGRESS", "SETUP_PROGRESS"):
        if stripped.startswith(prefix):
            update = _parse_key_value_progress(stripped.removeprefix(prefix).strip())
            update["kind"] = prefix.lower()
            if prefix == "PHOTO_PROGRESS" and "split" in update:
                update["stage"] = str(update.get("split"))
            return update
    return None


def _progress_from_update(update: Dict[str, object] | None) -> Optional[float]:
    if not update:
        return None
    if "progress" in update:
        try:
            progress = float(update["progress"])
            return max(0.0, min(1.0, progress))
        except (TypeError, ValueError):
            return None

    processed = update.get("processed")
    if processed is None:
        processed = update.get("downloaded")
    total = update.get("total")
    if processed is None or total is None:
        if update.get("reused_existing"):
            return 1.0
        return None
    try:
        done = float(processed)
        total_value = float(total)
    except (TypeError, ValueError):
        return None
    if total_value <= 0:
        return None
    return max(0.0, min(1.0, done / total_value))


def _eta_from_update(update: Dict[str, object] | None, local_elapsed: float) -> str:
    if not update:
        return "оставшееся время: оценивается по ходу выполнения"

    for key in ("eta_seconds", "eta"):
        if key in update:
            seconds = _duration_to_seconds(update.get(key))
            if seconds is not None:
                return f"примерно осталось: {_format_duration(seconds)}"

    processed = update.get("processed") or update.get("downloaded")
    total = update.get("total")
    elapsed = _duration_to_seconds(update.get("elapsed_seconds"))
    if elapsed is None:
        elapsed = _duration_to_seconds(update.get("elapsed"))
    if elapsed is None:
        elapsed = local_elapsed

    try:
        done = float(processed) if processed is not None else 0.0
        total_value = float(total) if total is not None else 0.0
    except (TypeError, ValueError):
        return "оставшееся время: оценивается по ходу выполнения"

    if done <= 0 or total_value <= 0 or done >= total_value:
        return "оставшееся время: оценивается по ходу выполнения"
    speed = done / max(elapsed, 1e-9)
    eta = (total_value - done) / max(speed, 1e-9)
    return f"примерно осталось: {_format_duration(eta)}"


def _progress_status_text(step_title: str, update: Dict[str, object] | None, elapsed: float, estimated_seconds: Optional[int]) -> tuple[float, str]:
    progress = _progress_from_update(update)
    if progress is None:
        fraction = _progress_fraction(elapsed, estimated_seconds)
        return fraction, f"{step_title}: прошло {_format_duration(elapsed)}, {_eta_text(elapsed, estimated_seconds)}"

    stage = update.get("stage") or update.get("split") or update.get("kind") or "этап"
    processed = update.get("processed") or update.get("downloaded")
    total = update.get("total")
    objects = update.get("objects")
    parts = [f"{step_title}: {stage}"]
    if processed is not None and total is not None:
        parts.append(f"{processed}/{total}")
        parts.append(f"{progress * 100:.1f}%")
    if objects is not None:
        parts.append(f"объектов: {objects}")
    parts.append(_eta_from_update(update, elapsed))
    return progress, " · ".join(parts)


def _progress_hint_text(update: Dict[str, object] | None, elapsed: float, estimated_seconds: Optional[int]) -> str:
    if not update:
        return _eta_text(elapsed, estimated_seconds)
    progress = _progress_from_update(update)
    if progress is not None:
        processed = update.get("processed") or update.get("downloaded")
        total = update.get("total")
        stage = update.get("stage") or update.get("split") or "текущий этап"
        if processed is not None and total is not None:
            return f"{stage}: {processed}/{total} ({progress * 100:.1f}%), {_eta_from_update(update, elapsed)}"
    return _eta_from_update(update, elapsed)


def _render_live_state(
    timer_placeholder,
    log_placeholder,
    log_lines: List[str],
    start: float,
    estimated_seconds: Optional[int],
    process_alive: bool,
    max_log_lines: int,
    hint: str,
    progress_update: Dict[str, object] | None = None,
) -> None:
    elapsed = time.perf_counter() - start
    eta = _progress_hint_text(progress_update, elapsed, estimated_seconds)
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
    return run_long_task_with_callback(
        func=lambda emit: func(),
        title=title,
        description=description,
        estimated_seconds=estimated_seconds,
        progress_text=progress_text,
        status_text=status_text,
        hint=hint,
    )


def run_long_task_with_callback(
    func: Callable[[ProgressEmit], T],
    title: str,
    description: str = "",
    estimated_seconds: Optional[int] = None,
    progress_text: str = "Выполняется операция...",
    status_text: str = "Операция выполняется...",
    hint: str = "Если операция идёт по большим папкам или зависит от сети, это может занять несколько минут.",
) -> T:
    """Runs a long Python task in a worker thread and keeps Streamlit UI alive.

    The task receives an emit(update_dict) callback and can publish live counters.
    """

    st.subheader(title)
    if description:
        st.info(description)

    progress = st.progress(0.0, text="Подготовка...")
    timer_placeholder = st.empty()
    status_placeholder = st.empty()
    details_placeholder = st.empty()

    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
    update_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()

    def emit(update: Dict[str, object]) -> None:
        update_queue.put(update)

    def target() -> None:
        try:
            result_queue.put(("result", func(emit)))
        except Exception as exc:  # noqa: BLE001
            result_queue.put(("error", exc))

    start = time.perf_counter()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    latest_update: Dict[str, object] | None = None

    while worker.is_alive():
        while True:
            try:
                latest_update = update_queue.get_nowait()
            except queue.Empty:
                break

        elapsed = time.perf_counter() - start
        fraction, text = _progress_status_text(progress_text, latest_update, elapsed, estimated_seconds)
        progress.progress(fraction, text=text)
        timer_placeholder.info(f"⏱ Прошло: **{_format_duration(elapsed)}** · {_progress_hint_text(latest_update, elapsed, estimated_seconds)}. {hint}")
        status_placeholder.info(status_text)
        details_placeholder.code(_progress_details_text(latest_update, "Ожидание первых данных прогресса..."), language="text")
        time.sleep(1.0)

    while True:
        try:
            latest_update = update_queue.get_nowait()
        except queue.Empty:
            break

    elapsed = time.perf_counter() - start
    kind, payload = result_queue.get()
    progress.progress(1.0, text=f"Готово за {_format_duration(elapsed)}")
    timer_placeholder.success(f"⏱ Общее время выполнения: **{_format_duration(elapsed)}**")
    details_placeholder.code(_progress_details_text(latest_update, "Операция завершена."), language="text")

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
    latest_progress_update: Dict[str, object] | None = None

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
                progress_update = _parse_progress_line(line)
                if progress_update is not None:
                    latest_progress_update = progress_update
                changed = True

        now = time.perf_counter()
        elapsed = now - start
        if changed or now - last_render >= 1.0:
            last_render = now
            fraction, text = _progress_status_text(step.title, latest_progress_update, elapsed, step.estimated_seconds)
            step_progress_placeholder.progress(fraction, text=text)
            _render_live_state(
                timer_placeholder=timer_placeholder,
                log_placeholder=log_placeholder,
                log_lines=log_lines,
                start=start,
                estimated_seconds=step.estimated_seconds,
                process_alive=process.poll() is None,
                max_log_lines=max_log_lines,
                hint="Если лог не меняется несколько минут, это всё ещё может быть нормальной установкой тяжёлых пакетов.",
                progress_update=latest_progress_update,
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
