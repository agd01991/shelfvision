"""Модуль начальной SKU-идентификации ShelfVision.

Контур идентификации строится после детекции/сегментации:
предсказания модели -> crop товара -> признаки crop -> matching с галереей SKU.
"""

from .crop_extractor import extract_crops_from_predictions_file
from .matcher import run_sku_matching

__all__ = ["extract_crops_from_predictions_file", "run_sku_matching"]
