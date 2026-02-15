# src/infer/engine.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal
import numpy as np
import cv2

from src.core.timer import timed
from src.infer.merge_tiles import merge_detections
from src.models.sku_id.crops import build_crops
from src.infer.report_builder import build_report

TaskMode = Literal["single", "tile"]


@dataclass
class DetObject:
    bbox: list[float]  # [x, y, w, h]
    score: float
    mask_rle: Optional[dict] = None
    sku_id: Optional[str] = None
    topk: Optional[list[dict]] = None


@dataclass
class InferenceResult:
    report: dict
    timings_ms: dict
    vis_path: Optional[str] = None
    report_path: Optional[str] = None


class InferenceEngine:
    def __init__(self, detector, sku_identifier, cfg: dict):
        self.detector = detector
        self.sku_identifier = sku_identifier
        self.cfg = cfg

    def run_image(self, img_path: str | Path, meta: dict) -> InferenceResult:
        timings = {}
        img_path = str(img_path)

        with timed("decode", timings):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mode: TaskMode = self.cfg.get("mode", "single")
        if mode == "tile":
            with timed("detect", timings):
                dets = self.detector.detect_tiled(img)
            with timed("merge_tiles", timings):
                dets = merge_detections(dets, self.cfg["merge"])
        else:
            with timed("detect", timings):
                dets = self.detector.detect(img)

        with timed("crops", timings):
            crops, crop_meta = build_crops(img, dets, self.cfg["crops"])

        with timed("sku_id", timings):
            topk = self.sku_identifier.predict_topk(
                crops, k=int(self.cfg["sku"]["topk"])
            )
            for obj, tk in zip(dets, topk):
                obj.topk = tk
                # выбранный sku_id и unknown-логика
                best = tk[0] if tk else None
                if best and best["score"] >= float(self.cfg["sku"]["unknown_thr"]):
                    obj.sku_id = best["sku_id"]
                else:
                    obj.sku_id = None  # unknown

        with timed("report", timings):
            report = build_report(meta, dets, crop_meta, self.cfg["report"])

        return InferenceResult(report=report, timings_ms=timings)
