from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


# -----------------------------------------------------------------------------
# ShelfVision visual interface
# -----------------------------------------------------------------------------

APP_TITLE = "ShelfVision: интерфейс экспериментов"
CSV_FILES = {
    "DIR1_models": "reports/all_stats/DIR1_models.csv",
    "YOLO_11_ablations": "reports/all_stats/YOLO_11_ablations.csv",
    "DIR3_WBF": "reports/all_stats/DIR3_WBF.csv",
    "DIR5_robustness": "reports/all_stats/DIR5_robustness.csv",
    "D2S_YOLO_SEG_last": "reports/all_stats/D2S_YOLO_SEG_last.csv",
    "OVERALL_detection_min": "reports/all_stats/OVERALL_detection_min.csv",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def repo_root() -> Path:
    # scripts/interface_app.py -> repository root
    return Path(__file__).resolve().parents[1]


def existing_csvs(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for name, rel in CSV_FILES.items():
        p = root / rel
        if p.exists():
            result[name] = p
    return result


def load_table(root: Path, name: str) -> Optional[pd.DataFrame]:
    p = root / CSV_FILES[name]
    if not p.exists():
        return None
    try:
        return read_csv_cached(str(p))
    except Exception as exc:
        st.error(f"Не удалось прочитать {p}: {exc}")
        return None


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_float(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def show_dataframe(df: pd.DataFrame, height: int = 360) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def newest_files(root: Path, patterns: Iterable[str], limit: int = 20) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def render_sidebar(root: Path) -> None:
    st.sidebar.header("Навигация")
    st.sidebar.caption(f"Проект: `{root}`")
    st.sidebar.caption(f"Python: `{sys.executable}`")
    st.sidebar.caption(f"OS: `{platform.system()} {platform.release()}`")

    st.sidebar.divider()
    st.sidebar.subheader("Найденные таблицы")
    csvs = existing_csvs(root)
    if not csvs:
        st.sidebar.warning("CSV-таблицы отчётов не найдены")
    else:
        for name, path in csvs.items():
            st.sidebar.write(f"✅ {name}")
            st.sidebar.caption(relative(path, root))

    if px is None:
        st.sidebar.warning("Plotly не установлен: графики скрыты. Запусти start_control_panel.bat ещё раз или установи: pip install plotly")


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

def page_home(root: Path) -> None:
    st.header("Главная")
    st.write(
        "Этот интерфейс собирает ключевые результаты проекта ShelfVision: сравнение моделей, "
        "абляции YOLO, WBF-ансамбль, устойчивость к искажениям, YOLO-Seg и визуальные примеры."
    )

    if px is None:
        st.warning(
            "Plotly не установлен в текущем Python-окружении, поэтому интерактивные графики не отображаются. "
            "Если интерфейс открыт из основной панели, перезапусти `start_control_panel.bat`: он доустановит `plotly`. "
            "Или выполни вручную: `.venv\\Scripts\\python.exe -m pip install plotly`."
        )

    csvs = existing_csvs(root)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Таблиц отчётов", str(len(csvs)))
    with c2:
        images = newest_files(root, ["*.png", "*.jpg", "*.jpeg"], limit=9999)
        metric_card("Изображений/графиков", str(len(images)))
    with c3:
        models = newest_files(root, ["best.pt", "*.pth"], limit=9999)
        metric_card("Файлов весов", str(len(models)))

    st.subheader("Что показывать на защите")
    st.markdown(
        """
        1. **Детекция** — сравнение YOLO, RT-DETR, Faster R-CNN и WBF.  
        2. **YOLO-абляции** — выбор лучшей конфигурации модели.  
        3. **YOLO-Seg** — переход от рамок к маскам.  
        4. **Визуальные примеры** — итоговые изображения, графики, crop-результаты.  
        5. **АПК** — краткая схема аппаратно-программного комплекса.
        """
    )

    st.subheader("Быстрые ссылки на папки")
    for folder in ["reports", "reports/all_stats", "runs", "models", "results", "data"]:
        p = root / folder
        st.write(f"- `{folder}` — {'есть' if p.exists() else 'нет'}")
