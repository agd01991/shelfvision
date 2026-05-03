"""Inference layer for ShelfVision.

This package contains adapters that convert outputs of different models
(YOLO, RT-DETR, Faster R-CNN, WBF) to one common prediction format.
"""

from .ensemble_wbf import combine_predictions_wbf, predict_wbf_folder, predict_wbf_image
from .faster_rcnn_inference import predict_faster_rcnn_folder, predict_faster_rcnn_image
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
    "predict_faster_rcnn_image",
    "predict_faster_rcnn_folder",
    "combine_predictions_wbf",
    "predict_wbf_image",
    "predict_wbf_folder",
]
