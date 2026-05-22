from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .feature_extractor import VisualFeatureExtractor


class VisualFeatureCache:
    """Disk cache for visual feature vectors.

    The cache is intentionally simple and transparent: one feature vector is
    saved as one .npy file, and metadata is saved next to it as .json. A cached
    vector is reused only when the source path, file size, modification time and
    extractor configuration are the same.
    """

    def __init__(self, cache_dir: str | Path, extractor: VisualFeatureExtractor) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        self.extractor_signature = {
            "image_size": extractor.image_size,
            "hist_bins": list(extractor.hist_bins),
            "orb_features": extractor.orb_features,
            "orb_vector_size": extractor.orb_vector_size,
        }

    def _file_meta(self, image_path: str | Path) -> Dict[str, object]:
        path = Path(image_path)
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "extractor": self.extractor_signature,
        }

    def _cache_stem(self, meta: Dict[str, object]) -> str:
        raw = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get_or_extract(self, image_path: str | Path) -> np.ndarray:
        meta = self._file_meta(image_path)
        stem = self._cache_stem(meta)
        feature_path = self.cache_dir / f"{stem}.npy"
        meta_path = self.cache_dir / f"{stem}.json"

        if feature_path.exists() and meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if cached_meta == meta:
                    return np.load(feature_path).astype(np.float32)
            except Exception:
                pass

        feature = self.extractor.extract_from_path(image_path).astype(np.float32)
        np.save(feature_path, feature)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return feature
