from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .crop_extractor import CropRecord, extract_crops_from_predictions_file
from .feature_cache import VisualFeatureCache
from .feature_extractor import VisualFeatureExtractor, cosine_similarity
from .sku_gallery import SkuGalleryItem, load_gallery


@dataclass
class SkuCandidate:
    sku_id: str
    sku_name: str
    category: str
    score: float
    gallery_image_path: str


@dataclass
class IdentificationResult:
    image_path: str
    image_name: str
    object_id: int
    crop_path: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_type: str
    detection_score: float
    label: str
    class_id: int
    sku_id: Optional[str]
    sku_name: str
    sku_confidence: float
    sku_status: str
    top_k: List[SkuCandidate]
    track_id: Optional[int] = None
    track_stabilized: bool = False
    track_frames_count: int = 0
    track_matched_votes: int = 0
    track_unknown_votes: int = 0


def _progress(stage: str, processed: int, total: int, started: float, **extra: object) -> None:
    elapsed = time.perf_counter() - started
    speed = processed / max(elapsed, 1e-9) if processed else 0.0
    eta = (total - processed) / max(speed, 1e-9) if speed > 0 and total > processed else 0.0
    payload: Dict[str, object] = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
    }
    payload.update(extra)
    print(f"PROGRESS_JSON {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _build_gallery_features(
    items: List[SkuGalleryItem],
    cache: VisualFeatureCache,
    progress_every: int = 10,
) -> List[tuple[SkuGalleryItem, np.ndarray]]:
    features: List[tuple[SkuGalleryItem, np.ndarray]] = []
    total = len(items)
    started = time.perf_counter()
    _progress("gallery_features", 0, total, started)

    for index, item in enumerate(items, start=1):
        try:
            features.append((item, cache.get_or_extract(item.image_path)))
        except FileNotFoundError:
            pass

        if index == 1 or index % max(1, progress_every) == 0 or index == total:
            _progress("gallery_features", index, total, started, features=len(features))

    if not features:
        raise FileNotFoundError("Не найдено ни одного доступного изображения в SKU-галерее")
    return features


def _match_one_crop(
    crop: CropRecord,
    gallery_features: List[tuple[SkuGalleryItem, np.ndarray]],
    cache: VisualFeatureCache,
    threshold: float,
    top_k: int,
) -> IdentificationResult:
    crop_feature = cache.get_or_extract(crop.crop_path)
    candidates: List[SkuCandidate] = []
    for item, gallery_feature in gallery_features:
        score = cosine_similarity(crop_feature, gallery_feature)
        candidates.append(
            SkuCandidate(
                sku_id=item.sku_id,
                sku_name=item.sku_name,
                category=item.category,
                score=score,
                gallery_image_path=item.image_path,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0] if candidates else None
    status = "matched" if best and best.score >= threshold else "unknown"
    return IdentificationResult(
        image_path=crop.image_path,
        image_name=crop.image_name,
        object_id=crop.object_id,
        crop_path=crop.crop_path,
        x1=crop.x1,
        y1=crop.y1,
        x2=crop.x2,
        y2=crop.y2,
        source_type=crop.source_type,
        detection_score=crop.score,
        label=crop.label,
        class_id=crop.class_id,
        sku_id=best.sku_id if status == "matched" and best else None,
        sku_name=best.sku_name if status == "matched" and best else "unknown",
        sku_confidence=best.score if best else 0.0,
        sku_status=status,
        top_k=candidates[:top_k],
    )


def result_to_dict(result: IdentificationResult) -> Dict[str, Any]:
    data = asdict(result)
    data["top_k"] = [asdict(item) for item in result.top_k]
    return data


def results_to_dataframe(results: List[IdentificationResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in results:
        row = result_to_dict(item)
        row["top_k"] = " | ".join(f"{cand.sku_id}:{cand.score:.4f}" for cand in item.top_k)
        rows.append(row)
    return pd.DataFrame(rows)


def run_sku_matching(
    predictions_json: str | Path,
    images_dir: str | Path | None,
    out_dir: str | Path,
    gallery_csv: str | Path | None = None,
    gallery_dir: str | Path | None = None,
    use_masks: bool = True,
    threshold: float = 0.65,
    top_k: int = 3,
    padding_ratio: float = 0.05,
    progress_every: int = 10,
    cache_dir: str | Path | None = None,
) -> List[IdentificationResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_started = time.perf_counter()
    _progress("extract_query_crops", 0, 1, stage_started)
    crops = extract_crops_from_predictions_file(
        predictions_json=predictions_json,
        images_dir=images_dir,
        out_dir=out_dir,
        use_masks=use_masks,
        padding_ratio=padding_ratio,
    )
    _progress("extract_query_crops", 1, 1, stage_started, objects=len(crops))

    gallery_items = load_gallery(gallery_csv=gallery_csv, gallery_dir=gallery_dir)
    extractor = VisualFeatureExtractor()
    cache = VisualFeatureCache(cache_dir or (out_dir / "feature_cache"), extractor)
    gallery_features = _build_gallery_features(gallery_items, cache, progress_every=progress_every)

    results: List[IdentificationResult] = []
    total = len(crops)
    started = time.perf_counter()
    _progress("identify", 0, total, started, objects=0, cache_dir=str(cache.cache_dir))
    for index, crop in enumerate(crops, start=1):
        result = _match_one_crop(
            crop=crop,
            gallery_features=gallery_features,
            cache=cache,
            threshold=threshold,
            top_k=top_k,
        )
        results.append(result)
        if index == 1 or index % max(1, progress_every) == 0 or index == total:
            matched = sum(1 for item in results if item.sku_status == "matched")
            _progress(
                "identify",
                index,
                total,
                started,
                objects=index,
                matched=matched,
                unknown=index - matched,
                cache_dir=str(cache.cache_dir),
            )

    results_to_dataframe(results).to_csv(out_dir / "identification_results.csv", index=False)
    return results
