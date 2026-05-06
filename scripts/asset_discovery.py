from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WEIGHT_EXTS = {".pt", ".pth", ".onnx"}
MODEL_KINDS = ("yolo", "rtdetr", "frcnn")

# Это не лимит поиска, а список служебных папок, которые почти никогда не нужны
# для выбора весов/датасетов и сильно тормозят обход.
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".venv_wsl",
    "venv",
    "env",
    "node_modules",
    ".ipynb_checkpoints",
}


@dataclass
class AssetCandidate:
    kind: str
    path: str
    score: int
    reason: str


@dataclass
class DiscoveryRootStats:
    root: str
    elapsed_seconds: float
    dirs_scanned: int
    files_scanned: int
    weight_files: int
    image_files: int
    image_dirs: int
    video_files: int
    skipped_dirs: int
    status: str = "ok"


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


def _top_candidates(candidates: List[AssetCandidate], limit: int) -> List[AssetCandidate]:
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def _iter_tree_unlimited(root: Path, excluded_dir_names: set[str] | None = None) -> Iterable[tuple[Path, str]]:
    """Рекурсивный обход без искусственных лимитов по количеству файлов.

    Возвращает пары (path, kind), где kind — file, dir или skipped_dir.
    Используется os.scandir, чтобы служебные папки можно было отрезать до входа в них.
    """

    excluded_dir_names = excluded_dir_names or DEFAULT_EXCLUDED_DIRS
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    try:
                        path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            if path.name in excluded_dir_names:
                                yield path, "skipped_dir"
                                continue
                            stack.append(path)
                            yield path, "dir"
                        elif entry.is_file(follow_symlinks=False):
                            yield path, "file"
                    except Exception:
                        continue
        except Exception:
            continue


def analyze_and_discover_assets(search_roots: List[Path], limit: int = 10) -> Dict[str, object]:
    """Глубокий анализ и автопоиск за один полный проход.

    Эту функцию следует вызывать на первом этапе интерфейса. Она сразу возвращает
    и статистику обхода, и найденные кандидаты. Второй этап интерфейса может
    использовать готовые результаты без повторного сканирования диска.
    """

    started = time.perf_counter()
    weight_candidates: Dict[str, List[AssetCandidate]] = {kind: [] for kind in MODEL_KINDS}
    video_candidates: List[AssetCandidate] = []
    image_dir_counts: DefaultDict[Path, int] = defaultdict(int)
    root_stats: List[DiscoveryRootStats] = []

    for root in search_roots:
        root_started = time.perf_counter()
        if not root.exists():
            root_stats.append(
                DiscoveryRootStats(
                    root=str(root),
                    elapsed_seconds=0.0,
                    dirs_scanned=0,
                    files_scanned=0,
                    weight_files=0,
                    image_files=0,
                    image_dirs=0,
                    video_files=0,
                    skipped_dirs=0,
                    status="not_found",
                )
            )
            continue

        dirs_scanned = 0
        files_scanned = 0
        weight_files = 0
        image_files = 0
        video_files = 0
        skipped_dirs = 0
        root_image_dirs: set[Path] = set()

        for item, item_kind in _iter_tree_unlimited(root):
            if item_kind == "skipped_dir":
                skipped_dirs += 1
                continue
            if item_kind == "dir":
                dirs_scanned += 1
                continue
            if item_kind != "file":
                continue

            files_scanned += 1
            suffix = item.suffix.lower()
            if suffix in WEIGHT_EXTS:
                weight_files += 1
                for kind in MODEL_KINDS:
                    weight_candidates[kind].append(_score_weight(item, kind))
            elif suffix in VIDEO_EXTS:
                video_files += 1
                video_candidates.append(_score_video(item))
            elif suffix in IMAGE_EXTS:
                image_files += 1
                image_dir_counts[item.parent] += 1
                root_image_dirs.add(item.parent)

        root_stats.append(
            DiscoveryRootStats(
                root=str(root),
                elapsed_seconds=time.perf_counter() - root_started,
                dirs_scanned=dirs_scanned,
                files_scanned=files_scanned,
                weight_files=weight_files,
                image_files=image_files,
                image_dirs=len(root_image_dirs),
                video_files=video_files,
                skipped_dirs=skipped_dirs,
            )
        )

    image_candidates = [
        candidate
        for candidate in (_score_images_dir_from_count(path, count) for path, count in image_dir_counts.items())
        if candidate is not None
    ]

    results = {
        "yolo": _top_candidates(weight_candidates["yolo"], limit),
        "rtdetr": _top_candidates(weight_candidates["rtdetr"], limit),
        "frcnn": _top_candidates(weight_candidates["frcnn"], limit),
        "images_dir": _top_candidates(image_candidates, limit),
        "video": _top_candidates(video_candidates, limit),
    }

    return {
        "results": results,
        "stats": root_stats,
        "elapsed_seconds": time.perf_counter() - started,
        "excluded_dirs": sorted(DEFAULT_EXCLUDED_DIRS),
    }


def discover_assets(search_roots: List[Path], limit: int = 10) -> Dict[str, List[AssetCandidate]]:
    """Простой API для других скриптов: возвращает только кандидатов."""

    return analyze_and_discover_assets(search_roots, limit=limit)["results"]  # type: ignore[return-value]


def find_weight_candidates(search_roots: List[Path], kind: str, limit: int = 10) -> List[AssetCandidate]:
    return discover_assets(search_roots, limit=limit).get(kind, [])


def find_images_dir_candidates(search_roots: List[Path], limit: int = 10) -> List[AssetCandidate]:
    return discover_assets(search_roots, limit=limit).get("images_dir", [])


def find_video_candidates(search_roots: List[Path], limit: int = 10) -> List[AssetCandidate]:
    return discover_assets(search_roots, limit=limit).get("video", [])
