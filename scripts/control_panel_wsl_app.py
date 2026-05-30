from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import yaml

from control_panel import ensure_config_exists, load_config, page_downloads, save_config
from control_panel_wsl import page_actions_wsl, page_config_wsl
from full_photo_identification_panel import page_full_photo_identification
from manual_cluster_editor_panel import page_manual_cluster_editor
from night_experiments_panel import page_night_experiments_reports
from setup_pages import page_setup


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_EXTS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
APP_PORTS = {
    "scripts/interface_app.py": 8502,
    "scripts/inference_app.py": 8503,
    "scripts/video_app.py": 8504,
}


def _as_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip().strip('"').strip("'")
    if not value:
        return None
    if value.startswith("/mnt/") and len(value) > 6 and value[6] == "/":
        drive = value[5].upper()
        return Path(f"{drive}:/{value[7:]}")
    return Path(value)


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _candidate_dirs(config: Dict[str, Any]) -> List[Tuple[str, Path | None, str]]:
    paths = config.get("paths", {})
    demo = config.get("demo_sku_gallery", {})
    full = config.get("full_photo_identification", {})
    night = config.get("night_experiments", {})
    identification = config.get("identification", {})
    video = config.get("video", {})
    readiness = config.get("readiness", {})
    sku_gallery = config.get("sku_gallery", {})
    presentation = config.get("presentation_assets", {})

    return [
        ("Основные результаты / старый pipeline", _as_path(paths.get("out_dir", "results/control_panel")), "paths.out_dir"),
        ("Фото-идентификация", _as_path(demo.get("out_dir", "D:/1Diplom/shelfvision_results/photo_identification")), "demo_sku_gallery.out_dir"),
        ("Полная фото-идентификация gallery/query", _as_path(full.get("out_dir", "D:/1Diplom/shelfvision_results/full_photo_identification")), "full_photo_identification.out_dir"),
        ("Серия экспериментов SKU110K", _as_path(night.get("out_dir") or night.get("results_root", "")), "night_experiments.results_root / night_experiments.out_dir"),
        ("Идентификация SKU", _as_path(identification.get("out_dir", "D:/1Diplom/shelfvision_results/identification")), "identification.out_dir"),
        ("Видео", _as_path(video.get("output_dir", "results/video/yolo")), "video.output_dir"),
        ("Диагностика готовности", _as_path(readiness.get("out_dir", "D:/1Diplom/shelfvision_results/readiness")), "readiness.out_dir"),
        ("Отчёты SKU-галереи", _as_path(sku_gallery.get("out_dir", "D:/1Diplom/shelfvision_results/sku_gallery")), "sku_gallery.out_dir"),
        ("Материалы презентации", _as_path(presentation.get("out_dir", "D:/1Diplom/presentation_assets")), "presentation_assets.out_dir"),
        ("SKU-галерея", _as_path(demo.get("gallery_dir") or sku_gallery.get("gallery_dir", "D:/1Diplom/sku_gallery")), "demo_sku_gallery.gallery_dir / sku_gallery.gallery_dir"),
        ("Full SKU-галерея", _as_path(full.get("gallery_dir", "D:/1Diplom/sku_gallery_full")), "full_photo_identification.gallery_dir"),
    ]


def _list_files(root: Path, limit: int = 300) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*") if p.is_file()])[:limit]


def _find_key_files(root: Path) -> List[Path]:
    names = [
        "manual_gallery_report.md",
        "manual_gallery_summary.json",
        "manual_cluster_edits.csv",
        "manual_cluster_edits_applied.csv",
        "night_experiments_detailed_report.md",
        "vkr_night_experiments_section.md",
        "night_experiments_ranked.csv",
        "night_experiments_parameter_impact.csv",
        "night_experiments_best_config.json",
        "night_experiments_summary.md",
        "night_experiments_summary.csv",
        "full_experiment_summary.md",
        "full_experiment_summary.csv",
        "full_experiment_summary.json",
        "experiment_summary.md",
        "experiment_summary.csv",
        "identification_results.csv",
        "identification_report.md",
        "identification_metrics.json",
        "identified_predictions.json",
        "predictions.json",
        "prediction.json",
        "summary.csv",
        "demo_sku_gallery_report.md",
        "demo_sku_gallery_summary.json",
        "gallery.csv",
        "sku_gallery_report.md",
        "readiness_report.md",
        "readiness_checks.csv",
        "video_summary.json",
    ]
    result: List[Path] = []
    if not root.exists():
        return result
    for name in names:
        if root.is_file() and root.name == name:
            result.append(root)
        elif root.is_dir():
            result.extend(sorted(root.rglob(name)))
    seen = set()
    unique: List[Path] = []
    for path in result:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _find_preview_images(root: Path, limit: int = 12) -> List[Path]:
    if not root.exists():
        return []
    search_roots = []
    if root.is_dir():
        for sub in [
            root / "04_identification" / "visualized",
            root / "06_manual_gallery" / "manual_identification" / "visualized",
            root / "03_identification" / "visualized",
            root / "visualized",
            root,
        ]:
            if sub.exists():
                search_roots.append(sub)
    elif root.suffix.lower() in IMAGE_EXTS:
        return [root]

    images: List[Path] = []
    for base in search_roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
                if len(images) >= limit:
                    return images
    return images


def _render_markdown_preview(path: Path, max_chars: int = 6000) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Не удалось прочитать файл: {exc}")
        return
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n..."
    st.markdown(text)


def _render_text_preview(path: Path, max_chars: int = 6000) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Не удалось прочитать файл: {exc}")
        return
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n..."
    st.code(text, language="text")


def _render_result_dir(title: str, root: Path | None, config_key: str) -> None:
    st.subheader(title)
    if root is None:
        st.info(f"Путь не задан: `{config_key}`")
        return

    st.write(f"Путь: `{_rel_or_abs(root)}`")
    if not root.exists():
        st.info("Папка пока не создана.")
        return

    files = _list_files(root)
    key_files = _find_key_files(root)
    images = _find_preview_images(root)

    c1, c2, c3 = st.columns(3)
    c1.metric("Файлов", len(files))
    c2.metric("Ключевых файлов", len(key_files))
    c3.metric("Превью-изображений", len(images))

    if images:
        st.write("Превью изображений:")
        cols = st.columns(min(4, len(images)))
        for idx, image_path in enumerate(images[:12]):
            with cols[idx % len(cols)]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

    if key_files:
        st.write("Ключевые файлы:")
        for path in key_files[:30]:
            with st.expander(_rel_or_abs(path), expanded=path.name in {"manual_gallery_report.md", "full_experiment_summary.md", "experiment_summary.md", "night_experiments_detailed_report.md", "vkr_night_experiments_section.md"}):
                if path.suffix.lower() == ".md":
                    _render_markdown_preview(path)
                elif path.suffix.lower() in TEXT_EXTS:
                    _render_text_preview(path)
                else:
                    st.write(f"`{_rel_or_abs(path)}`")

    with st.expander("Все файлы", expanded=False):
        for path in files[:300]:
            st.write(f"- `{_rel_or_abs(path)}`")
        if len(files) >= 300:
            st.caption("Показаны первые 300 файлов.")


def _streamlit_url(script_path: str) -> str:
    port = APP_PORTS.get(script_path.replace("\\", "/"), 8501)
    return f"http://localhost:{port}"


def _start_streamlit_app(script_path: str) -> str:
    port = APP_PORTS.get(script_path.replace("\\", "/"), 8501)
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            script_path,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"http://localhost:{port}"


def page_interface_shortcuts() -> None:
    st.header("Быстрый запуск интерфейсов")
    st.caption("`run_interface.bat` запускает старый экспериментальный интерфейс: `python -m streamlit run scripts/interface_app.py`.")

    app_script = "scripts/interface_app.py"
    if st.button(
        "Открыть интерфейс из run_interface.bat",
        use_container_width=True,
        key="shortcut_open_legacy_interface_app",
    ):
        url = _start_streamlit_app(app_script)
        st.success("Интерфейс `scripts/interface_app.py` запускается в отдельном процессе.")
        st.markdown(f"Открой: [{url}]({url})")

    st.caption(f"Адрес этого интерфейса: `{_streamlit_url(app_script)}`. Остальные интерфейсы ниже уже есть в основном блоке `Интерфейсы`.")


def page_actions_app(config: Dict[str, Any]) -> None:
    page_interface_shortcuts()
    st.divider()
    page_full_photo_identification(config)
    st.divider()
    page_night_experiments_reports(config)
    st.divider()
    page_manual_cluster_editor(config)
    st.divider()
    page_actions_wsl(config)


def page_results_wsl(config: Dict[str, Any]) -> None:
    st.header("5. Результаты")
    st.caption("Здесь показаны все основные папки результатов, а не только старый `results/control_panel`.")

    candidates = _candidate_dirs(config)
    existing = [(title, path, key) for title, path, key in candidates if path is not None and path.exists()]
    missing = [(title, path, key) for title, path, key in candidates if path is None or not path.exists()]

    st.success(f"Найдено рабочих папок: {len(existing)} из {len(candidates)}")

    st.info("Подробный просмотр серии экспериментов SKU110K и ручной редактор кластеров находятся в разделе `Запуск задач`.")

    selected_titles = st.multiselect(
        "Какие разделы показать",
        options=[title for title, _, _ in candidates],
        default=[title for title, _, _ in existing[:4]] or ["Полная фото-идентификация gallery/query", "Фото-идентификация"],
    )

    for title, path, key in candidates:
        if title in selected_titles:
            _render_result_dir(title, path, key)
            st.divider()

    if missing:
        with st.expander("Папки, которые пока не найдены", expanded=False):
            for title, path, key in missing:
                st.write(f"- **{title}** (`{key}`): `{_rel_or_abs(path) if path else 'не задано'}`")


def main() -> None:
    st.set_page_config(page_title="ShelfVision Control Panel", page_icon="🧰", layout="wide")
    ensure_config_exists()
    config = load_config()
    config.setdefault("runtime", {}).setdefault("use_wsl_runtime", True)

    st.title("🧰 ShelfVision Control Panel")
    st.caption("Панель первого запуска и управления. Рабочие задачи по умолчанию запускаются через WSL .venv_wsl.")

    page = st.sidebar.radio("Раздел", ["Первый запуск", "Скачивание файлов", "Настройки", "Запуск задач", "Результаты", "config YAML"])
    if page == "Первый запуск":
        page_setup(config)
    elif page == "Скачивание файлов":
        page_downloads(config)
    elif page == "Настройки":
        page_config_wsl(config)
    elif page == "Запуск задач":
        page_actions_app(config)
    elif page == "Результаты":
        page_results_wsl(config)
    elif page == "config YAML":
        st.header("config/shelfvision.yaml")
        st.code(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), language="yaml")
        if st.button("Сохранить текущий YAML"):
            save_config(config)
            st.success("Сохранено")


if __name__ == "__main__":
    main()
