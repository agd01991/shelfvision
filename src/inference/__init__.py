"""Inference layer for ShelfVision.

This package contains adapters that convert outputs of different models
(YOLO, RT-DETR, Faster R-CNN, WBF) to one common prediction format.
"""

from .prediction import DetectionPrediction, ImagePrediction
from .rtdetr_inference import predict_rtdetr_folder, predict_rtdetr_image
from .yolo_inference import predict_yolo_folder, predict_yolo_image

__all__ = [
    "DetectionPrediction",
    "ImagePrediction",
    "predict_yolo_image",
    "predict_yolo_folder",
    "predict_rtdetr_image",
    "predict_rtdetr_folder",
]
