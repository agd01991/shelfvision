from __future__ import annotations

import os
import re
from pathlib import Path


WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def clean_path_value(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def windows_to_wsl_path(value: str | Path | None) -> str:
    """Convert Windows path D:/dir/file to /mnt/d/dir/file.

    Non-path values and already-WSL paths are returned mostly unchanged.
    This is suitable for command arguments that will be executed inside WSL.
    """

    raw = clean_path_value(value).replace("\\", "/")
    match = WINDOWS_DRIVE_RE.match(raw)
    if not match:
        return raw
    drive = match.group(1).lower()
    rest = match.group(2)
    return f"/mnt/{drive}/{rest}"


def wsl_to_windows_path(value: str | Path | None) -> str:
    """Convert WSL path /mnt/d/dir/file to D:/dir/file.

    Non-path values and already-Windows paths are returned mostly unchanged.
    This is suitable for local file reading when Streamlit runs on Windows.
    """

    raw = clean_path_value(value).replace("\\", "/")
    match = WSL_MOUNT_RE.match(raw)
    if not match:
        return raw
    drive = match.group(1).upper()
    rest = match.group(2)
    return f"{drive}:/{rest}"


def to_current_os_path(value: str | Path | None) -> Path:
    """Return a Path readable by the currently running Python process.

    Standard in config/UI can be either Windows-style (D:/...) or WSL-style
    (/mnt/d/...). This helper adapts paths for local existence checks, pandas,
    image preview and markdown rendering.
    """

    raw = clean_path_value(value)
    if os.name == "nt":
        return Path(wsl_to_windows_path(raw))
    return Path(windows_to_wsl_path(raw))


def to_display_path(value: str | Path | None) -> str:
    """Return a normalized path string for display in the current UI process."""

    return str(to_current_os_path(value))


def to_wsl_arg(value: str | Path | None) -> str:
    """Return a path string suitable for an argument passed into WSL."""

    return windows_to_wsl_path(value)
