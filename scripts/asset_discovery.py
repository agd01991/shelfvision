from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WEIGHT_EXTS = {".pt", ".pth", ".onnx"}
MODEL_KINDS = ("yolo", "rtdetr", "frcnn")


@dataclass
class AssetCandidate:
    kind: str
    path: str
    score: int
    reason: str


def _safe_iter_files(root: Path, max_files: int = 30000) -> Iterable[Path]:
    """Legacy bounded file iterator kept for compatibility with helper functions."""

    count = 0
    try:
        iterator = root.rglob("*")
        for item in iterator:
            if count >= max_files:
                break
            try:
                if item.is_file():
                    count += 1
                    yield item
            except Exception:
                continue
    except Exception:
        return


def _safe_iter_dirs(root: Path, max_dirs: int = 10000) -> Iterable[Path]:
    """Legacy bounded dir iterator kept for compatibility with helper functions."""

    count = 0
    try:
        iterator = root.rglob("*")
        for item in iterator:
            if count >= max_dirs:
                break
            try:
                if item.is_dir():
                    count += 1
                    yield item
            except Exception:
                continue
    except Exception:
        return


def _safe_iter_tree(root: Path, max_items: int = 60000) -> Iterable[Path]:
    """Bounded recursive iterator for the optimized one-pass discovery.

    The old implementation walked every root several times: weights for YOLO,
    weights for RT-DETR, weights for Faster R-CNN, video files and image dirs.
    This iterator is used by discover_assets() to walk each selected root once.
    """

    count = 0
    try:
        for item in root.rglob("*"):
            if count >= max_items:
                break
            count += 1
            yield item
    except Exception:
        return


def _score_weight(path: Path, kind: str) -> AssetCandidate:
    lower = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    score = 0
    reasons: List[str] = []

    if path.suffix.lower() in WEIGHT_EXTS:
        score += 10
        reasons.append("расширение весов")
    if name in {"best.pt", "model_final.pth", "last.pt"}:
        score += 30
        reasons.append("типовое имя файла")
    if kind == "yolo" and "yolo" in lower:
        score += 35
        reasons.append("путь похож на YOLO")
    if kind == "rtdetr" and ("rtdetr" in lower or "rt-detr" in lower):
        score += 35
        reasons.append("путь похож на RT-DETR")
    if kind == "frcnn" and ("faster" in lower or "frcnn" in lower or "faster_rcnn" in lower):
        score += 35
        reasons.append("путь похож на Faster R-CNN")
    if "models" in lower:
        score += 10
        reasons.append("лежит в models")
    if "runs" in lower or "train" in lower:
        score += 5
        reasons.append("похоже на результат обучения")

    return AssetCandidate(kind=kind, path=str(path), score=score, reason=", ".join(reasons) or "кандидат")


def _score_video(path: Path) -> AssetCandidate:
    lower = str(path).replace("\\", "/").lower()
    score = 10
    reasons = ["видео"]
    if "video" in lower:
        score += 20
        reasons.append("путь похож на video")
    if "test" in lower or "demo" in lower:
        score += 10
        reasons.append("похоже на тестовый ролик")
    return AssetCandidate("video", str(path), score, ", ".join(reasons))


def _score_images_dir_from_count(path: Path, images_count: int) -> Optional[AssetCandidate]:
    if images_count <= 0:
        return None

    lower = str(path).replace("\\", "/").lower()
    score = min(images_count, 100)
    reasons = [f"изображений: {images_count}"]
    if "images" in lower:
        score += 30
        reasons.append("папка называется images")
    if "test" in lower or "val" in lower:
        score += 15
        reasons.append("похоже на test/val")
    if "data" in lower:
        score += 10
        reasons.append("лежит в data")
    return AssetCandidate(kind="images_dir", path=str(path), score=score, reason=", ".join(reasons))


def _score_images_dir(path: Path) -> Optional[AssetCandidate]:
    try:
        files = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS]
    except Exception:
        return None
    return _score_images_dir_from_count(path, len(files))


def _top_candidates(candidates: List[AssetCandidate], limit: int) -> List[AssetCandidate]:
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def find_weight_candidates(search_roots: List[Path], kind: str, limit: int = 10) -> List[AssetCandidate]:
    candidates: List[AssetCandidate] = []
    for root in search_roots:
        if not root.exists():
            continue
        for file_path in _safe_iter_files(root):
            if file_path.suffix.lower() in WEIGHT_EXTS:
                candidates.append(_score_weight(file_path, kind))
    return _top_candidates(candidates, limit)


def find_images_dir_candidates(search_roots: List[Path], limit: int = 10) -> List[AssetCandidate]:
    candidates: List[AssetCandidate] = []
    for root in search_roots:
        if not root.exists():
            continue
        direct = _score_images_dir(root)
        if direct:
            candidates.append(direct)
        for dir_path in _safe_iter_dirs(root):
            candidate = _score_images_dir(dir_path)
            if candidate:
                candidates.append(candidate)
    return _top_candidates(candidates, limit)


def find_video_candidates(search_roots: List[Path], limit: int = 10) -> List[AssetCandidate]:
    candidates: List[AssetCandidate] = []
    for root in search_roots:
        if not root.exists():
            continue
        for file_path in _safe_iter_files(root):
            if file_path.suffix.lower() in VIDEO_EXTS:
                candidates.append(_score_video(file_path))
    return _top_candidates(candidates, limit)


def discover_assets(search_roots: List[Path], limit: int = 10) -> Dict[str, List[AssetCandidate]]:
    """Find model weights, image folders and video files in one pass per root.

    This replaces the earlier multi-pass implementation that recursively scanned
    every root separately for YOLO, RT-DETR, Faster R-CNN, images and video.
    """

    weight_candidates: Dict[str, List[AssetCandidate]] = {kind: [] for kind in MODEL_KINDS}
    video_candidates: List[AssetCandidate] = []
    image_dir_counts: DefaultDict[Path, int] = defaultdict(int)

    for root in search_roots:
        if not root.exists():
            continue

        direct = _score_images_dir(root)
        if direct:
            image_dir_counts[root] += int(direct.reason.split("изображений: ", 1)[1].split(",", 1)[0])

        for item in _safe_iter_tree(root):
            try:
                if not item.is_file():
                    continue
            except Exception:
                continue

            suffix = item.suffix.lower()
            if suffix in WEIGHT_EXTS:
                for kind in MODEL_KINDS:
                    weight_candidates[kind].append(_score_weight(item, kind))
            elif suffix in VIDEO_EXTS:
                video_candidates.append(_score_video(item))
            elif suffix in IMAGE_EXTS:
                image_dir_counts[item.parent] += 1

    image_candidates = [
        candidate
        for candidate in (_score_images_dir_from_count(path, count) for path, count in image_dir_counts.items())
        if candidate is not None
    ]

    return {
        "yolo": _top_candidates(weight_candidates["yolo"], limit),
        "rtdetr": _top_candidates(weight_candidates["rtdetr"], limit),
        "frcnn": _top_candidates(weight_candidates["frcnn"], limit),
        "images_dir": _top_candidates(image_candidates, limit),
        "video": _top_candidates(video_candidates, limit),
    }
