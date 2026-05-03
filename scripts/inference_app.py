from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# scripts/inference_app.py -> repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.ensemble_wbf import predict_wbf_image
from src.inference.faster_rcnn_inference import predict_faster_rcnn_image
from src.inference.prediction import ImagePrediction, save_prediction_json
from src.inference.rtdetr_inference import predict_rtdetr_image
from src.inference.yolo_inference import predict_yolo_image, prediction_summary
from src.visualization.draw_boxes import draw_prediction


APP_TITLE = "ShelfVision: демонстрация инференса"
UPLOAD_DIR = ROOT / "artifacts" / "interface_uploads"
RESULTS_DIR = ROOT / "results" / "interface_inference"


MODEL_OPTIONS = {
    "YOLO": "yolo",
    "RT-DETR-L": "rtdetr",
    "Faster R-CNN": "frcnn",
    "WBF(YOLO + RT-DETR)": "wbf",
}


DEFAULT_WEIGHTS = {
    "yolo": "models/yolo/best.pt",
    "rtdetr": "models/rtdetr/best.pt",
    "frcnn": "models/faster_rcnn/model_final.pth",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(str(path).strip().strip('"').strip("'"))
    if path.is_absolute():
        return path
    return ROOT / path


def save_uploaded_image(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    output_path = UPLOAD_DIR / f"uploaded{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def detections_to_dataframe(prediction: ImagePrediction) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, detection in enumerate(prediction.detections, start=1):
        x1, y1, x2, y2 = detection.box
        rows.append(
            {
                "№": idx,
                "Класс": detection.label,
                "Confidence": round(detection.score, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "Площадь bbox": round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 1),
            }
        )
    return pd.DataFrame(rows)


def run_selected_model(
    model_key: str,
    image_path: Path,
    conf: float,
    imgsz: int,
    device: Optional[str],
    yolo_weights: Path,
    rtdetr_weights: Path,
    frcnn_weights: Path,
    wbf_iou: float,
    wbf_skip: float,
    yolo_weight: float,
    rtdetr_weight: float,
) -> ImagePrediction:
    if model_key == "yolo":
        return predict_yolo_image(
            model_path=yolo_weights,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            model_name="YOLO",
        )

    if model_key == "rtdetr":
        return predict_rtdetr_image(
            model_path=rtdetr_weights,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            model_name="RT-DETR-L",
        )

    if model_key == "frcnn":
        return predict_faster_rcnn_image(
            model_path=frcnn_weights,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            model_name="Faster R-CNN",
        )

    if model_key == "wbf":
        return predict_wbf_image(
            yolo_model_path=yolo_weights,
            rtdetr_model_path=rtdetr_weights,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            iou_thr=wbf_iou,
            skip_box_thr=wbf_skip,
            yolo_weight=yolo_weight,
            rtdetr_weight=rtdetr_weight,
            model_name="WBF(YOLO + RT-DETR)",
        )

    raise ValueError(f"Неизвестная модель: {model_key}")


def render_metrics(prediction: ImagePrediction) -> None:
    summary = prediction_summary(prediction)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Найдено объектов", summary["objects_count"])
    with c2:
        st.metric("Средний confidence", f"{summary['average_confidence']:.4f}")
    with c3:
        st.metric("Мин. confidence", f"{summary['min_confidence']:.4f}")
    with c4:
        st.metric("Время обработки", f"{summary['inference_time']:.3f} сек")


def save_outputs(prediction: ImagePrediction, show_masks: bool) -> tuple[Path, Path, Path]:
    run_dir = RESULTS_DIR / prediction.model_name.replace("/", "_").replace(" ", "_").replace("+", "plus")
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_prediction_json(prediction, run_dir / "prediction.json")
    image_path = draw_prediction(
        prediction,
        output_path=run_dir / "visualized.jpg",
        show_masks=show_masks,
    )
    table_path = run_dir / "detections.csv"
    detections_to_dataframe(prediction).to_csv(table_path, index=False)
    return json_path, run_dir / "visualized.jpg", table_path


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.title(APP_TITLE)
    st.caption("Интерактивный запуск моделей ShelfVision на одном изображении товарной полки.")

    with st.sidebar:
        st.header("Параметры")
        selected_model_title = st.selectbox("Модель", list(MODEL_OPTIONS.keys()))
        model_key = MODEL_OPTIONS[selected_model_title]

        conf = st.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01)
        imgsz = st.selectbox("imgsz", [416, 512, 640, 768, 1024], index=2)
        device = st.text_input("device", value="0", help="Например: 0, cpu, cuda:0")
        device = device.strip() or None
        show_masks = st.checkbox("Показывать masks, если модель их вернула", value=True)

        st.divider()
        st.subheader("Веса моделей")
        yolo_weights = resolve_path(st.text_input("YOLO weights", value=DEFAULT_WEIGHTS["yolo"]))
        rtdetr_weights = resolve_path(st.text_input("RT-DETR weights", value=DEFAULT_WEIGHTS["rtdetr"]))
        frcnn_weights = resolve_path(st.text_input("Faster R-CNN weights", value=DEFAULT_WEIGHTS["frcnn"]))

        st.divider()
        st.subheader("WBF")
        wbf_iou = st.slider("WBF IoU", 0.1, 0.9, 0.55, 0.01)
        wbf_skip = st.slider("WBF skip score", 0.0, 0.5, 0.001, 0.001)
        yolo_weight = st.number_input("YOLO weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        rtdetr_weight = st.number_input("RT-DETR weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    input_mode = st.radio("Источник изображения", ["Загрузить файл", "Указать путь"], horizontal=True)
    image_path: Optional[Path] = None

    if input_mode == "Загрузить файл":
        uploaded_file = st.file_uploader("Изображение полки", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded_file is not None:
            image_path = save_uploaded_image(uploaded_file)
    else:
        raw_path = st.text_input("Путь к изображению", value="data/test/image_001.jpg")
        if raw_path.strip():
            image_path = resolve_path(raw_path)

    if image_path is not None:
        st.subheader("Исходное изображение")
        if image_path.exists():
            st.image(str(image_path), caption=str(image_path), use_container_width=True)
        else:
            st.warning(f"Изображение не найдено: {image_path}")

    run_button = st.button("Обработать изображение", type="primary", disabled=image_path is None)

    if run_button and image_path is not None:
        missing = []
        if model_key in {"yolo", "wbf"} and not yolo_weights.exists():
            missing.append(f"YOLO: {yolo_weights}")
        if model_key in {"rtdetr", "wbf"} and not rtdetr_weights.exists():
            missing.append(f"RT-DETR: {rtdetr_weights}")
        if model_key == "frcnn" and not frcnn_weights.exists():
            missing.append(f"Faster R-CNN: {frcnn_weights}")

        if missing:
            st.error("Не найдены веса модели:\n" + "\n".join(missing))
            return

        with st.spinner(f"Запуск модели: {selected_model_title}..."):
            try:
                prediction = run_selected_model(
                    model_key=model_key,
                    image_path=image_path,
                    conf=conf,
                    imgsz=int(imgsz),
                    device=device,
                    yolo_weights=yolo_weights,
                    rtdetr_weights=rtdetr_weights,
                    frcnn_weights=frcnn_weights,
                    wbf_iou=wbf_iou,
                    wbf_skip=wbf_skip,
                    yolo_weight=yolo_weight,
                    rtdetr_weight=rtdetr_weight,
                )
                json_path, visualized_path, table_path = save_outputs(prediction, show_masks=show_masks)
            except Exception as exc:
                st.error(f"Ошибка инференса: {exc}")
                return

        st.success("Обработка завершена")
        render_metrics(prediction)

        st.subheader("Результат")
        st.image(str(visualized_path), caption=str(visualized_path), use_container_width=True)

        st.subheader("Таблица найденных объектов")
        detections_df = detections_to_dataframe(prediction)
        st.dataframe(detections_df, use_container_width=True, height=420)

        st.subheader("Сохранённые файлы")
        st.code(
            "\n".join(
                [
                    f"JSON: {json_path}",
                    f"CSV:  {table_path}",
                    f"IMG:  {visualized_path}",
                ]
            ),
            language="text",
        )

        with st.expander("JSON результата"):
            st.json(prediction.to_dict())

    st.divider()
    st.info(
        "Этот интерфейс предназначен для демонстрации практической части ВКР: "
        "можно загрузить изображение, выбрать модель, получить визуальный результат, "
        "таблицу найденных объектов и файлы для отчёта."
    )


if __name__ == "__main__":
    main()
