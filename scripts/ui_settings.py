from __future__ import annotations

from typing import Any, Dict

import streamlit as st


RECOMMENDED_MODE = "recommended"
ADVANCED_MODE = "advanced"

MODE_LABELS = {
    RECOMMENDED_MODE: "Рекомендованные настройки",
    ADVANCED_MODE: "Для уверенных пользователей",
}

PAGE_LABELS = {
    "setup": "Первый запуск",
    "downloads": "Скачивание файлов",
    "settings": "Настройки",
    "actions": "Запуск задач",
    "results": "Результаты",
    "config_yaml": "config YAML",
}


def _ui_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.setdefault("ui", {})


def _normalize_mode(value: Any) -> str:
    mode = str(value or RECOMMENDED_MODE)
    return mode if mode in MODE_LABELS else RECOMMENDED_MODE


def get_settings_mode(config: Dict[str, Any], page_key: str | None = None) -> str:
    """Return settings mode for a page.

    The previous implementation stored a single global `ui.settings_mode`, so the
    radio button did not express page-specific behaviour.  New pages read from
    `ui.settings_mode_by_page[page_key]` and fall back to the legacy global value.
    """

    ui = _ui_config(config)
    legacy_mode = _normalize_mode(ui.get("settings_mode", RECOMMENDED_MODE))
    ui["settings_mode"] = legacy_mode

    if not page_key:
        return legacy_mode

    modes_by_page = ui.setdefault("settings_mode_by_page", {})
    page_mode = _normalize_mode(modes_by_page.get(page_key, legacy_mode))
    modes_by_page[page_key] = page_mode
    return page_mode


def is_advanced(config: Dict[str, Any], page_key: str | None = None) -> bool:
    return get_settings_mode(config, page_key=page_key) == ADVANCED_MODE


def render_settings_mode_switch(config: Dict[str, Any], page_key: str | None = None) -> str:
    """Render mode switch for one page and store the result in config."""

    ui = _ui_config(config)
    current = get_settings_mode(config, page_key=page_key)
    options = [RECOMMENDED_MODE, ADVANCED_MODE]
    page_label = PAGE_LABELS.get(str(page_key or ""), "текущей страницы")
    widget_key = f"settings_mode_{page_key or 'global'}"

    selected = st.radio(
        f"Режим настроек — {page_label}",
        options,
        index=options.index(current),
        format_func=lambda value: MODE_LABELS[value],
        horizontal=True,
        key=widget_key,
        help=(
            "Рекомендованные настройки показывают основные безопасные поля. "
            "Режим для уверенных пользователей открывает пороги, лимиты, "
            "технические параметры, служебные действия и детальные списки."
        ),
    )

    if page_key:
        ui.setdefault("settings_mode_by_page", {})[page_key] = selected
    else:
        ui["settings_mode"] = selected

    if selected == RECOMMENDED_MODE:
        st.info(
            "Включён режим рекомендованных настроек для этой страницы: "
            "показаны только основные поля и безопасные действия."
        )
    else:
        st.warning(
            "Включён расширенный режим для этой страницы: доступны пороги, "
            "лимиты, технические параметры и потенциально разрушительные действия."
        )

    return selected
