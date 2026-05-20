from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

from .crop_extractor import CropRecord, extract_crops_from_predictions_file
from .gallery_manager import build_sku_gallery


@dataclass
class DemoGalleryItem:
    sku_id: str
    sku_name: str
    source_image: str
    source_object_id: int
    source_crop_path: str
    gallery_image_path: str
    score: float
    width: int
    height: int
    source_type: str


@dataclass
class DemoGallerySummary:
    predictions_json: str
    images_dir: str
    gallery_dir: str
    gallery_csv: str
    crops_dir: str
    requested_sku_count: int
    created_sku_count: int
    extracted_crops_count: int
    selected_crops_count: int
    min_score: float
    min_width: int
    min_height: int
    use_masks: bool
    status: str
    warning: str = ""


def _crop_size(path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return int(width), int(height)


def _select_demo_crops(
    crops: List[CropRecord],
    max_sku: int,
    min_score: float,
    min_width: int,
    min_height: int,
) -> List[tuple[CropRecord, int, int]]:
    candidates: List[tuple[CropRecord, int, int]] = []
    for crop in crops:
        width, height = _crop_size(crop.crop_path)
        if width < min_width or height < min_height:
            continue
        if crop.score < min_score:
            continue
        candidates.append((crop, width, height))

    # Сначала берём наиболее уверенные и крупные crop-ы: для demo-галереи они выглядят лучше.
    candidates.sort(key=lambda item: (item[0].score, item[1] * item[2]), reverse=True)
    return candidates[:max_sku]


def _clear_old_demo_skus(gallery_dir: Path, prefix: str) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for child in gallery_dir.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            shutil.rmtree(child)


def _copy_demo_items(
    selected: List[tuple[CropRecord, int, int]],
    gallery_dir: Path,
    prefix: str,
) -> List[DemoGalleryItem]:
    items: List[DemoGalleryItem] = []
    for index, (crop, width, height) in enumerate(selected, start=1):
        sku_id = f"{prefix}{index:03d}"
        sku_name = sku_id.replace("_", " ")
        sku_dir = gallery_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)
        dst = sku_dir / "ref_001.jpg"
        shutil.copy2(crop.crop_path, dst)
        items.append(
            DemoGalleryItem(
                sku_id=sku_id,
                sku_name=sku_name,
                source_image=crop.image_path,
                source_object_id=crop.object_id,
                source_crop_path=crop.crop_path,
                gallery_image_path=str(dst),
                score=crop.score,
                width=width,
                height=height,
                source_type=crop.source_type,
            )
        )
    return items


def build_demo_sku_gallery_from_predictions(
    predictions_json: str | Path,
    images_dir: str | Path | None,
    gallery_dir: str | Path,
    gallery_csv: str | Path,
    out_dir: str | Path,
    max_sku: int = 30,
    min_score: float = 0.35,
    min_width: int = 20,
    min_height: int = 20,
    use_masks: bool = True,
    padding_ratio: float = 0.05,
    prefix: str = "sku_demo_",
    clear_old_demo: bool = True,
) -> Dict[str, Path]:
    """Automatically creates a demo SKU gallery from detected object crops.

    This is intended for datasets such as SKU110K where bbox annotations exist,
    but true SKU-level gallery is not provided. Each selected crop becomes a
    conditional demo SKU class: sku_demo_001, sku_demo_002, ...
    """

    predictions_json = Path(predictions_json)
    gallery_dir = Path(gallery_dir)
    gallery_csv = Path(gallery_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_json.exists():
        raise FileNotFoundError(f"predictions_json не найден: {predictions_json}")

    crops_out_dir = out_dir / "demo_gallery_crops"
    crops = extract_crops_from_predictions_file(
        predictions_json=predictions_json,
        images_dir=images_dir,
        out_dir=crops_out_dir,
        use_masks=use_masks,
        padding_ratio=padding_ratio,
    )
    selected = _select_demo_crops(
        crops=crops,
        max_sku=max(1, max_sku),
        min_score=min_score,
        min_width=min_width,
        min_height=min_height,
    )

    if clear_old_demo:
        _clear_old_demo_skus(gallery_dir, prefix=prefix)
    else:
        gallery_dir.mkdir(parents=True, exist_ok=True)

    demo_items = _copy_demo_items(selected, gallery_dir=gallery_dir, prefix=prefix)
    items_json = out_dir / "demo_sku_gallery_items.json"
    items_csv = out_dir / "demo_sku_gallery_items.csv"
    items_json.write_text(json.dumps([asdict(item) for item in demo_items], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in demo_items]).to_csv(items_csv, index=False)

    warning = ""
    status = "ok"
    if not demo_items:
        status = "error"
        warning = "Не удалось выбрать ни одного crop для demo SKU-галереи. Проверь predictions_json, images_dir и фильтры."
    elif len(demo_items) < max_sku:
        status = "warning"
        warning = f"Создано меньше SKU, чем запрошено: {len(demo_items)} из {max_sku}."

    summary = DemoGallerySummary(
        predictions_json=str(predictions_json),
        images_dir=str(images_dir or ""),
        gallery_dir=str(gallery_dir),
        gallery_csv=str(gallery_csv),
        crops_dir=str(crops_out_dir / "crops"),
        requested_sku_count=max_sku,
        created_sku_count=len(demo_items),
        extracted_crops_count=len(crops),
        selected_crops_count=len(selected),
        min_score=min_score,
        min_width=min_width,
        min_height=min_height,
        use_masks=use_masks,
        status=status,
        warning=warning,
    )
    summary_json = out_dir / "demo_sku_gallery_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    gallery_outputs: Dict[str, Path] = {}
    if demo_items:
        # Для demo-галереи у каждого SKU один эталон, поэтому min_images_per_sku=1.
        gallery_outputs = build_sku_gallery(
            gallery_dir=gallery_dir,
            output_csv=gallery_csv,
            out_dir=out_dir / "gallery_check",
            min_images_per_sku=1,
        )

    report_md = out_dir / "demo_sku_gallery_report.md"
    lines = [
        "# ShelfVision: demo SKU-галерея",
        "",
        "## Назначение",
        "",
        "Эта галерея автоматически сформирована из crop-изображений найденных товаров.",
        "Она предназначена для демонстрации модуля идентификации на датасетах без настоящей SKU-разметки, например SKU110K.",
        "",
        "## Сводка",
        "",
        f"- Статус: {summary.status}",
        f"- Извлечено crop-ов: {summary.extracted_crops_count}",
        f"- Создано demo SKU: {summary.created_sku_count}",
        f"- Папка галереи: `{summary.gallery_dir}`",
        f"- gallery.csv: `{summary.gallery_csv}`",
        f"- Предупреждение: {summary.warning or 'нет'}",
        "",
        "## Первые demo SKU",
        "",
        "| sku_id | score | size | source | gallery image |",
        "|---|---:|---|---|---|",
    ]
    for item in demo_items[:30]:
        lines.append(
            f"| {item.sku_id} | {item.score:.4f} | {item.width}x{item.height} | "
            f"{Path(item.source_image).name}#{item.source_object_id} | `{item.gallery_image_path}` |"
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    outputs: Dict[str, Path] = {
        "summary_json": summary_json,
        "items_json": items_json,
        "items_csv": items_csv,
        "report_md": report_md,
    }
    outputs.update({f"gallery_{key}": value for key, value in gallery_outputs.items()})
    return outputs
