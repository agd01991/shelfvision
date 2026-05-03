from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WEIGHT_EXTS = {".pt", ".pth", ".onnx"}


@dataclass
class AssetCandidate:
    kind: str
    path: str
    score: int
    reason: str


def _safe_iter_files(root: Path, max_files: int = 30000) -> Iterable[Path]:
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


def find_weight_candidates(search_roots: List[Path], kind: str, limit: int = 10) -> List[AssetCandidate]:
    candidates: List[AssetCandidate] = []
    for root in search_roots:
        if not root.exists():
            continue
        for file_path in _safe_iter_files(root):
            if file_path.suffix.lower() in WEIGHT_EXTS:
                candidates.append(_score_weight(file_path, kind))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def _score_images_dir(path: Path) -> Optional[AssetCandidate]:
    try:
        files = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS]
    except Exception:
        return None
    if not files:
        return None

    lower = str(path).replace("\\", "/").lower()
    score = min(len(files), 100)
    reasons = [f"изображений: {len(files)}"]
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
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def find_video_candidates(search_roots: List[Path], limit: int = 10) -> List[AssetCandidate]:
    candidates: List[AssetCandidate] = []
    for root in search_roots:
        if not root.exists():
            continue
        for file_path in _safe_iter_files(root):
            if file_path.suffix.lower() in VIDEO_EXTS:
                lower = str(file_path).replace("\\", "/").lower()
                score = 10
                reasons = ["видео"]
                if "video" in lower:
                    score += 20
                    reasons.append("путь похож на video")
                if "test" in lower or "demo" in lower:
                    score += 10
                    reasons.append("похоже на тестовый ролик")
                candidates.append(AssetCandidate("video", str(file_path), score, ", ".join(reasons)))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def discover_assets(search_roots: List[Path], limit: int = 10) -> Dict[str, List[AssetCandidate]]:
    return {
        "yolo": find_weight_candidates(search_roots, "yolo", limit=limit),
        "rtdetr": find_weight_candidates(search_roots, "rtdetr", limit=limit),
        "frcnn": find_weight_candidates(search_roots, "frcnn", limit=limit),
        "images_dir": find_images_dir_candidates(search_roots, limit=limit),
        "video": find_video_candidates(search_roots, limit=limit),
    }
