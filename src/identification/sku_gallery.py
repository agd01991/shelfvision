from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class SkuGalleryItem:
    sku_id: str
    sku_name: str
    category: str
    image_path: str


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sku_id", "sku_name", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В gallery.csv отсутствуют колонки: {sorted(missing)}")
    if "category" not in df.columns:
        df["category"] = ""
    return df


def load_gallery_csv(path: str | Path) -> List[SkuGalleryItem]:
    path = Path(path)
    df = _normalize_columns(pd.read_csv(path))
    items: List[SkuGalleryItem] = []
    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            candidate = path.parent / image_path
            if candidate.exists():
                image_path = candidate
        items.append(
            SkuGalleryItem(
                sku_id=str(row["sku_id"]),
                sku_name=str(row["sku_name"]),
                category=str(row.get("category", "")),
                image_path=str(image_path),
            )
        )
    return items


def scan_gallery_dir(gallery_dir: str | Path, output_csv: str | Path | None = None) -> List[SkuGalleryItem]:
    """Сканирует папку вида data/sku_gallery/<sku_id>/*.jpg.

    Если gallery.csv ещё нет, этот способ позволяет быстро собрать минимальную базу SKU:
    имя папки используется как sku_id и sku_name.
    """

    gallery_dir = Path(gallery_dir)
    items: List[SkuGalleryItem] = []
    for sku_dir in sorted(item for item in gallery_dir.iterdir() if item.is_dir()):
        sku_id = sku_dir.name
        sku_name = sku_id.replace("_", " ")
        for image_path in sorted(sku_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTS:
                items.append(
                    SkuGalleryItem(
                        sku_id=sku_id,
                        sku_name=sku_name,
                        category="",
                        image_path=str(image_path),
                    )
                )
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([item.__dict__ for item in items]).to_csv(output_csv, index=False)
    return items


def load_gallery(gallery_csv: str | Path | None = None, gallery_dir: str | Path | None = None) -> List[SkuGalleryItem]:
    if gallery_csv:
        return load_gallery_csv(gallery_csv)
    if gallery_dir:
        return scan_gallery_dir(gallery_dir)
    raise ValueError("Укажите --gallery-csv или --gallery-dir")
