from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cv2
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def first_existing(paths: Iterable[str | Path | None]) -> Optional[Path]:
    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists():
            return path
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_exists(src: str | Path | None, dst_dir: Path, dst_name: str | None = None) -> Optional[Path]:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.exists() or not src_path.is_file():
        return None
    dst = dst_dir / (dst_name or src_path.name)
    shutil.copy2(src_path, dst)
    return dst


def find_first_image(root: str | Path | None) -> Optional[Path]:
    if not root:
        return None
    root_path = Path(str(root))
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTS:
        return root_path
    if not root_path.exists() or not root_path.is_dir():
        return None
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def find_images(root: str | Path | None, limit: int = 6) -> list[Path]:
    if not root:
        return []
    root_path = Path(str(root))
    if not root_path.exists():
        return []
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTS:
        return [root_path]
    result = []
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            result.append(path)
            if len(result) >= limit:
                break
    return result


def extract_video_frame(video_path: str | Path | None, out_path: Path, frame_index: int = 0) -> Optional[Path]:
    if not video_path:
        return None
    video = Path(str(video_path))
    if not video.exists() or video.suffix.lower() not in VIDEO_EXTS:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path


def csv_preview(csv_path: str | Path | None, out_path: Path, rows: int = 12) -> Optional[Path]:
    if not csv_path:
        return None
    path = Path(str(csv_path))
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    df = pd.read_csv(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(rows).to_csv(out_path, index=False)
    return out_path


def read_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(str(path))
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_note(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")


def prepare_assets(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    project_root = Path(args.project_root)
    results_root = Path(args.results_root)
    video_dir = Path(args.video_dir)
    identification_dir = Path(args.identification_dir)
    sku_gallery_dir = Path(args.sku_gallery_dir)
    sku_gallery_report_dir = Path(args.sku_gallery_report_dir)

    slide_notes: dict[str, list[str]] = {
        "01": [
            "Вставить на слайд 1: исходная полка + результат с bbox/масками.",
            "Если готового результата нет, вставь generated placeholder и потом замени вручную.",
        ],
        "02": ["Вставить на слайд 2: плотная товарная полка. Можно использовать исходное изображение датасета."],
        "03": ["Вставить на слайд 3: слева исходная полка, справа результат с подписями/рамками."],
        "04": ["Вставить на слайд 4: пример изображения из SKU-110K и скрин структуры sku_gallery."],
        "05": ["Вставить на слайд 5: пример bbox YOLO и масок YOLO-Seg."],
        "06": ["Вставить на слайд 6: скрин Control Panel и config/shelfvision.yaml."],
        "07": ["Вставить на слайд 7: crop найденного товара, эталон SKU и preview identification_results.csv."],
        "08": ["Вставить на слайд 8: кадр output_video.mp4, кадр identified_output_video.mp4 и лог VIDEO_PROGRESS."],
        "09": ["Вставить на слайд 9: таблицу метрик / model_comparison.csv / metrics_summary.csv и визуализацию bbox/mask."],
        "10": ["Вставить на слайд 10: финальный кадр identified_output_video.mp4 или результат идентификации с SKU-подписями."],
    }

    for slide_id, notes in slide_notes.items():
        slide_dir = ensure_dir(out_root / f"slide_{slide_id}")
        write_note(slide_dir / "README.md", f"Слайд {slide_id}", notes)

    first_dataset_image = first_existing([
        find_first_image(project_root / "data" / "test" / "images"),
        find_first_image(project_root / "data"),
        find_first_image(results_root),
    ])

    # Slides 1-4: source/dataset examples.
    for slide_id, name in [("01", "source_or_result.jpg"), ("02", "dense_shelf_example.jpg"), ("03", "source_left.jpg"), ("04", "dataset_example.jpg")]:
        copy_if_exists(first_dataset_image, out_root / f"slide_{slide_id}", name)

    # Results images from inference / segmentation.
    result_images = find_images(results_root, limit=10)
    if result_images:
        copy_if_exists(result_images[0], out_root / "slide_01", "result_bbox_or_mask.jpg")
        copy_if_exists(result_images[0], out_root / "slide_03", "result_right.jpg")
        copy_if_exists(result_images[0], out_root / "slide_05", "yolo_bbox_or_mask.jpg")
        copy_if_exists(result_images[0], out_root / "slide_09", "metric_visualization_or_prediction.jpg")
    if len(result_images) > 1:
        copy_if_exists(result_images[1], out_root / "slide_05", "yolo_seg_mask_example.jpg")

    # Slide 6: config copy.
    copy_if_exists(project_root / "config" / "shelfvision.yaml", out_root / "slide_06", "shelfvision.yaml")
    copy_if_exists(project_root / "config" / "shelfvision.example.yaml", out_root / "slide_06", "shelfvision.example.yaml")

    # Slide 7: identification artifacts.
    crops_dir = identification_dir / "crops"
    visualized_dir = identification_dir / "visualized"
    crop = find_first_image(crops_dir)
    visualized = find_first_image(visualized_dir)
    gallery_ref = find_first_image(sku_gallery_dir)
    copy_if_exists(crop, out_root / "slide_07", "crop_found_product.jpg")
    copy_if_exists(gallery_ref, out_root / "slide_07", "sku_gallery_reference.jpg")
    copy_if_exists(visualized, out_root / "slide_07", "identified_visualization.jpg")
    csv_preview(identification_dir / "identification_results.csv", out_root / "slide_07" / "identification_results_preview.csv")

    # Slide 8 and 10: video frames.
    extract_video_frame(video_dir / "output_video.mp4", out_root / "slide_08" / "output_video_frame.jpg", frame_index=args.video_frame)
    extract_video_frame(identification_dir / "identified_output_video.mp4", out_root / "slide_08" / "identified_output_video_frame.jpg", frame_index=args.video_frame)
    extract_video_frame(identification_dir / "identified_output_video.mp4", out_root / "slide_10" / "final_identified_video_frame.jpg", frame_index=args.video_frame)
    copy_if_exists(identification_dir / "track_sku_summary.json", out_root / "slide_08", "track_sku_summary.json")

    # Slide 9: metrics tables.
    metric_candidates = [
        results_root / "model_comparison.csv",
        results_root / "metrics_summary.csv",
        results_root / "full_pipeline" / "model_comparison.csv",
        results_root / "full_pipeline" / "metrics_summary.csv",
    ]
    for candidate in metric_candidates:
        if candidate.exists():
            copy_if_exists(candidate, out_root / "slide_09")
            csv_preview(candidate, out_root / "slide_09" / f"{candidate.stem}_preview.csv")
            break

    # Gallery reports for slide 4 or 7.
    copy_if_exists(sku_gallery_report_dir / "sku_gallery_report.md", out_root / "slide_04", "sku_gallery_report.md")
    copy_if_exists(sku_gallery_report_dir / "sku_gallery_sku_stats.csv", out_root / "slide_04", "sku_gallery_sku_stats.csv")

    # Summary index.
    index_lines = [
        "Папки подготовлены по слайдам.",
        "Открой каждую папку slide_XX и вставь файлы в соответствующий слайд Gamma/PowerPoint.",
        "Если какой-то файл не появился, значит соответствующий этап программы ещё не был запущен или путь указан неверно.",
        "",
        "Рекомендуемый порядок запуска перед подготовкой скринов:",
        "1. Проверить SKU-галерею и создать gallery.csv.",
        "2. Проверить готовность видео-идентификации.",
        "3. Обработать видео.",
        "4. Подставить результаты последнего видео.",
        "5. Запустить идентификацию SKU.",
    ]
    write_note(out_root / "README.md", "Presentation assets", index_lines)
    print(f"Готово: {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShelfVision presentation assets by slide")
    parser.add_argument("--project-root", default=".", help="Корень проекта ShelfVision")
    parser.add_argument("--out-dir", default="D:/1Diplom/presentation_assets", help="Куда сложить материалы по слайдам")
    parser.add_argument("--results-root", default="results", help="Папка с общими результатами")
    parser.add_argument("--video-dir", default="results/video/yolo", help="Папка видеоинференса")
    parser.add_argument("--identification-dir", default="D:/1Diplom/shelfvision_results/identification", help="Папка результатов идентификации")
    parser.add_argument("--sku-gallery-dir", default="D:/1Diplom/sku_gallery", help="Папка SKU-галереи")
    parser.add_argument("--sku-gallery-report-dir", default="D:/1Diplom/shelfvision_results/sku_gallery", help="Папка отчётов SKU-галереи")
    parser.add_argument("--video-frame", type=int, default=0, help="Номер кадра для извлечения из видео")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_assets(parse_args())
