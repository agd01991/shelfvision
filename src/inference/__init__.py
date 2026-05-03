"""Inference layer for ShelfVision.

This package contains adapters that convert outputs of different models
(YOLO, RT-DETR, Faster R-CNN, WBF) to one common prediction format.
"""

from .prediction import DetectionPrediction, ImagePrediction

__all__ = ["DetectionPrediction", "ImagePrediction"]
