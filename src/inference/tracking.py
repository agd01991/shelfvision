from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .prediction import DetectionPrediction


Box = Sequence[float]


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class TrackState:
    track_id: int
    box: List[float]
    class_id: int
    last_seen_frame: int
    missed: int = 0


class SimpleIoUTracker:
    """Small dependency-free IoU tracker for shelf video experiments.

    It is intentionally simple: each detection is assigned to the best unmatched
    previous track if IoU is high enough. This is enough for диплом experiments
    and does not require ByteTrack/DeepSORT dependencies.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 5, same_class_only: bool = True) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missing = int(max_missing)
        self.same_class_only = bool(same_class_only)
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.frame_index = 0

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()
        self.frame_index = 0

    def _new_track(self, detection: DetectionPrediction) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = TrackState(
            track_id=track_id,
            box=[float(v) for v in detection.box],
            class_id=int(detection.class_id),
            last_seen_frame=self.frame_index,
            missed=0,
        )
        return track_id

    def _best_track_for(self, detection: DetectionPrediction, used_tracks: set[int]) -> Tuple[Optional[int], float]:
        best_track_id: Optional[int] = None
        best_iou = 0.0
        for track_id, track in self.tracks.items():
            if track_id in used_tracks:
                continue
            if self.same_class_only and int(detection.class_id) != int(track.class_id):
                continue
            current_iou = box_iou(detection.box, track.box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_track_id = track_id
        if best_track_id is None or best_iou < self.iou_threshold:
            return None, best_iou
        return best_track_id, best_iou

    def update(self, detections: List[DetectionPrediction]) -> List[DetectionPrediction]:
        self.frame_index += 1
        used_tracks: set[int] = set()
        seen_tracks: set[int] = set()

        for detection in detections:
            track_id, _ = self._best_track_for(detection, used_tracks)
            if track_id is None:
                track_id = self._new_track(detection)
            else:
                track = self.tracks[track_id]
                track.box = [float(v) for v in detection.box]
                track.class_id = int(detection.class_id)
                track.last_seen_frame = self.frame_index
                track.missed = 0

            detection.track_id = track_id
            used_tracks.add(track_id)
            seen_tracks.add(track_id)

        for track_id, track in list(self.tracks.items()):
            if track_id not in seen_tracks:
                track.missed += 1
            if track.missed > self.max_missing:
                del self.tracks[track_id]

        return detections
