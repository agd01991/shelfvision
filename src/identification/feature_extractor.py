from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VisualFeatureExtractor:
    """Лёгкий baseline-экстрактор признаков для начальной SKU-идентификации.

    Он не требует torch/faiss и работает на текущих зависимостях проекта:
    - HSV color histogram описывает цветовую схему упаковки;
    - ORB Bag-of-Visual-Words-like vector грубо описывает локальные детали.

    Такой baseline нужен не как промышленное распознавание SKU, а как воспроизводимый
    первый контур: crop товара -> feature vector -> cosine similarity -> SKU/unknown.
    """

    def __init__(
        self,
        image_size: int = 224,
        hist_bins: tuple[int, int, int] = (16, 16, 8),
        orb_features: int = 256,
        orb_vector_size: int = 64,
    ) -> None:
        self.image_size = image_size
        self.hist_bins = hist_bins
        self.orb_features = orb_features
        self.orb_vector_size = orb_vector_size
        self._orb = cv2.ORB_create(nfeatures=orb_features)

    def read_image(self, path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
        return image

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def _color_hist(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.hist_bins),
            ranges=[0, 180, 0, 256, 0, 256],
        ).astype(np.float32).flatten()
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _orb_descriptor_vector(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.orb_vector_size, dtype=np.float32)

        descriptors = descriptors.astype(np.float32) / 255.0
        mean_desc = descriptors.mean(axis=0)
        if mean_desc.size >= self.orb_vector_size:
            vector = mean_desc[: self.orb_vector_size]
        else:
            vector = np.pad(mean_desc, (0, self.orb_vector_size - mean_desc.size))
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector.astype(np.float32)

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        prepared = self._prepare(image)
        vector = np.concatenate([self._color_hist(prepared), self._orb_descriptor_vector(prepared)]).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def extract_from_path(self, path: str | Path) -> np.ndarray:
        return self.extract_from_image(self.read_image(path))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
