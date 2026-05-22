from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .crop_extractor import CropRecord, extract_crops_from_predictions_file
from .feature_extractor import VisualFeatureExtractor, cosine_similarity
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
    ref_index: int = 1
    is_primary_ref: bool = True
    matched_existing_sku: bool = False
    dedup_similarity: float = 0.0


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
    gallery_refs_count: int
    duplicate_refs_count: int
    skipped_duplicate_crops_count: int
    min_score: float
    min_width: int
    min_height: int
    use_masks: bool
    deduplicate: bool
    dedup_threshold: float
    max_refs_per_sku: int
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
    if max_sku <= 0:
        return candidates
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
) -> tuple[List[DemoGalleryItem], int]:
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
    return items, 0


def _mean_feature(features: List[np.ndarray]) -> np.ndarray:
    if not features:
        return np.array([], dtype=np.float32)
    vector = np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _copy_demo_items_deduplicated(
    candidates: List[tuple[CropRecord, int, int]],
    gallery_dir: Path,
    prefix: str,
    max_sku: int,
    dedup_threshold: float,
    max_refs_per_sku: int,
) -> tuple[List[DemoGalleryItem], int]:
    """Creates demo SKU gallery with lightweight duplicate merging.

    If a new crop is visually similar to an already created SKU, it is copied as
    ref_002/ref_003/... inside the existing SKU folder instead of creating a new
    synthetic SKU id. This reduces cases where the same real product receives
    several demo identifiers such as SKU-006 and SKU-038.
    """

    extractor = VisualFeatureExtractor()
    items: List[DemoGalleryItem] = []
    sku_features: Dict[str, List[np.ndarray]] = {}
    sku_refs_count: Dict[str, int] = {}
    sku_primary_item: Dict[str, DemoGalleryItem] = {}
    skipped_duplicate_crops = 0

    for crop, width, height in candidates:
        try:
            feature = extractor.extract_from_path(crop.crop_path)
        except Exception:
            continue

        best_sku_id = ""
        best_similarity = 0.0
        for sku_id, features in sku_features.items():
            centroid = _mean_feature(features)
            if centroid.size == 0:
                continue
            similarity = cosine_similarity(feature, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_sku_id = sku_id

        if best_sku_id and best_similarity >= dedup_threshold:
            # Same visual product candidate. Add it as another reference, unless
            # the SKU already has enough reference examples.
            if sku_refs_count.get(best_sku_id, 0) >= max_refs_per_sku:
                skipped_duplicate_crops += 1
                continue
            sku_id = best_sku_id
            ref_index = sku_refs_count[sku_id] + 1
            sku_refs_count[sku_id] = ref_index
            sku_features[sku_id].append(feature)
            sku_name = sku_primary_item[sku_id].sku_name
            matched_existing_sku = True
            is_primary_ref = False
            dedup_similarity = best_similarity
        else:
            if len(sku_features) >= max_sku:
                continue
            sku_number = len(sku_features) + 1
            sku_id = f"{prefix}{sku_number:03d}"
            sku_name = sku_id.replace("_", " ")
            ref_index = 1
            sku_refs_count[sku_id] = 1
            sku_features[sku_id] = [feature]
            matched_existing_sku = False
            is_primary_ref = True
            dedup_similarity = 0.0

        sku_dir = gallery_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)
        dst = sku_dir / f"ref_{ref_index:03d}.jpg"
        shutil.copy2(crop.crop_path, dst)
        item = DemoGalleryItem(
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
            ref_index=ref_index,
            is_primary_ref=is_primary_ref,
            matched_existing_sku=matched_existing_sku,
            dedup_similarity=dedup_similarity,
        )
        if is_primary_ref:
            sku_primary_item[sku_id] = item
        items.append(item)

    return items, skipped_duplicate_crops


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
    deduplicate: bool = True,
    dedup_threshold: float = 0.86,
    max_refs_per_sku: int = 3,
) -> Dict[str, Path]:
    """Automatically creates a demo SKU gallery from detected object crops.

    This is intended for datasets such as SKU110K where bbox annotations exist,
    but true SKU-level gallery is not provided. In deduplication mode, visually
    similar crops are merged into one conditional demo SKU with multiple refs.
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
        max_sku=0 if deduplicate else max(1, max_sku),
        min_score=min_score,
        min_width=min_width,
        min_height=min_height,
    )

    if clear_old_demo:
        _clear_old_demo_skus(gallery_dir, prefix=prefix)
    else:
        gallery_dir.mkdir(parents=True, exist_ok=True)

    if deduplicate:
        demo_items, skipped_duplicate_crops = _copy_demo_items_deduplicated(
            selected,
            gallery_dir=gallery_dir,
            prefix=prefix,
            max_sku=max(1, max_sku),
            dedup_threshold=dedup_threshold,
            max_refs_per_sku=max(1, max_refs_per_sku),
        )
    else:
        demo_items, skipped_duplicate_crops = _copy_demo_items(selected, gallery_dir=gallery_dir, prefix=prefix)

    items_json = out_dir / "demo_sku_gallery_items.json"
    items_csv = out_dir / "demo_sku_gallery_items.csv"
    items_json.write_text(json.dumps([asdict(item) for item in demo_items], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in demo_items]).to_csv(items_csv, index=False)

    created_sku_count = len({item.sku_id for item in demo_items})
    duplicate_refs_count = len([item for item in demo_items if item.matched_existing_sku])

    warning = ""
    status = "ok"
    if not demo_items:
        status = "error"
        warning = "Не удалось выбрать ни одного crop для demo SKU-галереи. Проверь predictions_json, images_dir и фильтры."
    elif created_sku_count < max_sku:
        status = "warning"
        warning = f"Создано меньше уникальных SKU, чем запрошено: {created_sku_count} из {max_sku}."

    summary = DemoGallerySummary(
        predictions_json=str(predictions_json),
        images_dir=str(images_dir or ""),
        gallery_dir=str(gallery_dir),
        gallery_csv=str(gallery_csv),
        crops_dir=str(crops_out_dir / "crops"),
        requested_sku_count=max_sku,
        created_sku_count=created_sku_count,
        extracted_crops_count=len(crops),
        selected_crops_count=len(selected),
        gallery_refs_count=len(demo_items),
        duplicate_refs_count=duplicate_refs_count,
        skipped_duplicate_crops_count=skipped_duplicate_crops,
        min_score=min_score,
        min_width=min_width,
        min_height=min_height,
        use_masks=use_masks,
        deduplicate=deduplicate,
        dedup_threshold=dedup_threshold,
        max_refs_per_sku=max(1, max_refs_per_sku),
        status=status,
        warning=warning,
    )
    summary_json = out_dir / "demo_sku_gallery_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    gallery_outputs: Dict[str, Path] = {}
    if demo_items:
        # Demo SKU can have one or several reference images after deduplication.
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
        f"- Отобрано crop-кандидатов: {summary.selected_crops_count}",
        f"- Создано уникальных demo SKU: {summary.created_sku_count}",
        f"- Эталонных изображений в галерее: {summary.gallery_refs_count}",
        f"- Добавлено повторных refs к существующим SKU: {summary.duplicate_refs_count}",
        f"- Пропущено повторных crop-ов сверх max_refs_per_sku: {summary.skipped_duplicate_crops_count}",
        f"- Дедупликация: {summary.deduplicate}",
        f"- Порог дедупликации: {summary.dedup_threshold}",
        f"- Максимум refs на SKU: {summary.max_refs_per_sku}",
        f"- Папка галереи: `{summary.gallery_dir}`",
        f"- gallery.csv: `{summary.gallery_csv}`",
        f"- Предупреждение: {summary.warning or 'нет'}",
        "",
        "## Первые demo SKU refs",
        "",
        "| sku_id | ref | duplicate | dedup_similarity | score | size | source | gallery image |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]
    for item in demo_items[:50]:
        lines.append(
            f"| {item.sku_id} | {item.ref_index} | {item.matched_existing_sku} | {item.dedup_similarity:.4f} | "
            f"{item.score:.4f} | {item.width}x{item.height} | "
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
