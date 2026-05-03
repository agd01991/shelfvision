from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.video_inference import process_yolo_video_file


APP_TITLE = "ShelfVision: видеоинференс"
CONFIG_PATH = ROOT / "config" / "shelfvision.yaml"
UPLOAD_DIR = ROOT / "artifacts" / "video_uploads"
DEFAULT_OUT_DIR = ROOT / "results" / "video" / "streamlit"


DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {"yolo": "models/yolo/best.pt"},
    "runtime": {"conf": 0.25, "imgsz": 640, "device": "0"},
    "video": {
        "input_path": "data/video/test.mp4",
        "output_dir": "results/video/yolo",
        "frame_skip": 3,
        "max_frames": 0,
        "save_video": True,
        "sample_frames": 8,
        "show_masks": True,
        "codec": "mp4v",
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return deep_merge(DEFAULT_CONFIG, data)
    return DEFAULT_CONFIG.copy()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).strip().strip('"').strip("'"))
    if path.is_absolute():
        return path
    return ROOT / path


def save_uploaded_video(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    output_path = UPLOAD_DIR / f"uploaded_video{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def show_saved_outputs(outputs: Dict[str, Path]) -> None:
    st.subheader("Сохранённые результаты")
    for name, path in outputs.items():
        st.write(f"**{name}:** `{path}`")

    output_video = outputs.get("output_video")
    if output_video and output_video.exists():
        st.subheader("Размеченное видео")
        st.video(str(output_video))

    summary_json = outputs.get("summary_json")
    if summary_json and summary_json.exists():
        st.subheader("Сводка")
        st.json(summary_json.read_text(encoding="utf-8"))

    stats_csv = outputs.get("frame_stats_csv")
    if stats_csv and stats_csv.exists():
        st.subheader("Статистика по кадрам")
        df = pd.read_csv(stats_csv)
        st.dataframe(df, use_container_width=True, height=420)

        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Обработано кадров", len(df))
            with c2:
                st.metric("Среднее число объектов", f"{df['objects_count'].mean():.2f}")
            with c3:
                st.metric("Средний confidence", f"{df['average_confidence'].mean():.4f}")
            with c4:
                st.metric("Средний FPS", f"{df['fps'].mean():.2f}")

    sample_dir = outputs.get("sample_frames_dir")
    if sample_dir and sample_dir.exists():
        images = sorted(sample_dir.glob("*.jpg"))
        if images:
            st.subheader("Примеры обработанных кадров")
            cols = st.columns(2)
            for idx, image_path in enumerate(images):
                with cols[idx % 2]:
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
    config = load_config()
    video_config = config.get("video", {})
    runtime = config.get("runtime", {})
    weights = config.get("weights", {})

    st.title(APP_TITLE)
    st.caption("Обработка видеофайла: YOLO детектирует товары на кадрах, сохраняет размеченное видео и статистику.")

    with st.sidebar:
        st.header("Настройки")
        yolo_weights = resolve_path(st.text_input("YOLO weights", value=str(weights.get("yolo", "models/yolo/best.pt"))))
        conf = st.slider("Confidence", 0.01, 0.95, float(runtime.get("conf", 0.25)), 0.01)
        imgsz_options = [416, 512, 640, 768, 1024]
        imgsz_value = int(runtime.get("imgsz", 640))
        imgsz = st.selectbox("imgsz", imgsz_options, index=imgsz_options.index(imgsz_value) if imgsz_value in imgsz_options else 2)
        device = st.text_input("device", value=str(runtime.get("device", "0")))
        device = device.strip() or None

        st.divider()
        st.subheader("Видео")
        frame_skip = st.number_input("Обрабатывать каждый N-й кадр", min_value=1, max_value=120, value=int(video_config.get("frame_skip", 3)))
        max_frames = st.number_input("Максимум кадров, 0 — всё видео", min_value=0, max_value=100000, value=int(video_config.get("max_frames", 0)))
        save_video = st.checkbox("Сохранять размеченное видео", value=bool(video_config.get("save_video", True)))
        sample_frames = st.number_input("Сколько кадров-примеров сохранить", min_value=0, max_value=100, value=int(video_config.get("sample_frames", 8)))
        show_masks = st.checkbox("Показывать masks, если модель их вернула", value=bool(video_config.get("show_masks", True)))
        codec = st.text_input("Кодек", value=str(video_config.get("codec", "mp4v")))
        out_dir = resolve_path(st.text_input("Папка результатов", value=str(video_config.get("output_dir", DEFAULT_OUT_DIR))))

    source_mode = st.radio("Источник видео", ["Загрузить файл", "Указать путь"], horizontal=True)
    video_path: Optional[Path] = None

    if source_mode == "Загрузить файл":
        uploaded_video = st.file_uploader("Видео", type=["mp4", "avi", "mov", "mkv", "webm"])
        if uploaded_video is not None:
            video_path = save_uploaded_video(uploaded_video)
    else:
        raw_path = st.text_input("Путь к видео", value=str(video_config.get("input_path", "data/video/test.mp4")))
        if raw_path.strip():
            video_path = resolve_path(raw_path)

    if video_path is not None:
        st.subheader("Исходное видео")
        if video_path.exists():
            st.write(f"Файл: `{video_path}`")
            st.video(str(video_path))
        else:
            st.warning(f"Видео не найдено: {video_path}")

    run_button = st.button("Обработать видео", type="primary", disabled=video_path is None)

    if run_button and video_path is not None:
        if not yolo_weights.exists():
            st.error(f"Не найдены веса YOLO: {yolo_weights}")
            return
        if not video_path.exists():
            st.error(f"Видео не найдено: {video_path}")
            return

        with st.spinner("Идёт обработка видео. Это может занять время, особенно без GPU."):
            try:
                outputs = process_yolo_video_file(
                    model_path=yolo_weights,
                    input_video=video_path,
                    out_dir=out_dir,
                    conf=conf,
                    imgsz=int(imgsz),
                    device=device,
                    frame_skip=int(frame_skip),
                    max_frames=int(max_frames),
                    save_video=save_video,
                    save_sample_frames=int(sample_frames),
                    show_masks=show_masks,
                    codec=codec,
                    model_name="YOLO Video",
                )
            except Exception as exc:
                st.error(f"Ошибка обработки видео: {exc}")
                return

        st.success("Видео обработано")
        show_saved_outputs(outputs)

    st.divider()
    st.info(
        "Для демонстрации на защите лучше использовать короткий ролик или поставить frame_skip=3/5, "
        "чтобы обработка шла быстрее. Итоговое видео и CSV можно вставить в отчёт или презентацию."
    )


if __name__ == "__main__":
    main()
