from __future__ import annotations

from typing import Any, Dict

import streamlit as st


RECOMMENDED_MODE = "recommended"
ADVANCED_MODE = "advanced"

MODE_LABELS = {
    RECOMMENDED_MODE: "Рекомендованные настройки",
    ADVANCED_MODE: "Для уверенных пользователей",
}


def get_settings_mode(config: Dict[str, Any]) -> str:
    ui = config.setdefault("ui", {})
    mode = str(ui.get("settings_mode", RECOMMENDED_MODE))
    if mode not in MODE_LABELS:
        mode = RECOMMENDED_MODE
    ui["settings_mode"] = mode
    return mode


def is_advanced(config: Dict[str, Any]) -> bool:
    return get_settings_mode(config) == ADVANCED_MODE


def render_settings_mode_switch(config: Dict[str, Any]) -> str:
    ui = config.setdefault("ui", {})
    current = get_settings_mode(config)
    options = [RECOMMENDED_MODE, ADVANCED_MODE]

    selected = st.radio(
        "Режим настроек",
        options,
        index=options.index(current),
        format_func=lambda value: MODE_LABELS[value],
        horizontal=True,
        key="global_settings_mode",
        help=(
            "Рекомендованные настройки показывают только основные поля и безопасные пресеты. "
            "Режим для уверенных пользователей открывает пороги, лимиты, технические параметры и пути."
        ),
    )

    ui["settings_mode"] = selected

    if selected == RECOMMENDED_MODE:
        st.info(
            "Включён режим рекомендованных настроек. "
            "Скрыты технические параметры, которые обычно не нужно менять."
        )
    else:
        st.warning(
            "Включён расширенный режим. "
            "Изменение порогов и лимитов может сильно повлиять на результат экспериментов."
        )

    return selected
