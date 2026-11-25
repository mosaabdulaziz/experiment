"""SORT baseline (single-stage Kalman + IoU Hungarian; DeepStream DUMMY proxy)."""

from __future__ import annotations

import numpy as np

from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.costs import iou_matrix

from .base import BaseTracker, Detection, TrackOutput
from .kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


class _SortTrack:
    def __init__(self, track_id: int, det: Detection, kf: KalmanFilterXYAH) -> None:
        self.id = track_id
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.hits = 1
        self.time_since_update = 0

    def predict(self, dt_steps: float = 1.0) -> np.ndarray:
        if self.mean[7] + self.mean[3] <= 0:
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov, dt_steps)
        self.time_since_update += 1
        return xyah_to_xyxy(self.mean[:4])

    def update(self, det: Detection) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.hits += 1
        self.time_since_update = 0

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


class SORTTracker(BaseTracker):
    """Single-stage Kalman + IoU Hungarian baseline."""

    name = "sort"

    def __init__(
        self,
        conf_thresh: float = 0.60,
        iou_thresh: float = 0.25,
        max_age: int = 30,
        min_hits: int = 2,
        check_class_match: bool = False,
        dt_aware: bool = False,
        ref_fps: float = 30.0,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.check_class_match = check_class_match
        self.dt_aware = dt_aware
        self.ref_fps = ref_fps
        self.kf = KalmanFilterXYAH()
        self.tracks: list[_SortTrack] = []
        self._next_id = 1
        self.frame_idx = 0

    def reset(self, fps: float = 30.0) -> None:
        if fps > 0:
            self.ref_fps = float(fps)
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0

    def update(
        self,
        detections: list[Detection],
        dt: float | None = None,
    ) -> list[TrackOutput]:
        self.frame_idx += 1
        dt_steps = (
            float(np.clip(dt * self.ref_fps, 1e-4, 30.0))
            if (self.dt_aware and dt is not None and dt > 0)
            else 1.0
        )
        high_dets = [d for d in detections if d.conf >= self.conf_thresh]

        for trk in self.tracks:
            trk.predict(dt_steps)

        if self.tracks and high_dets:
            t_boxes = np.stack([t.bbox for t in self.tracks], axis=0)
            d_boxes = np.stack([d.bbox for d in high_dets], axis=0)
            cost = 1.0 - iou_matrix(t_boxes, d_boxes)
            if self.check_class_match:
                tc = np.array([t.cls_id for t in self.tracks])[:, None]
                dc = np.array([d.cls_id for d in high_dets])[None, :]
                cost = np.where(tc != dc, np.inf, cost)
            matches, _, u_dets = linear_assignment(cost, 1.0 - self.iou_thresh)
            for ti, di in matches:
                self.tracks[ti].update(high_dets[di])
        else:
            u_dets = list(range(len(high_dets)))

        for di in u_dets:
            trk = _SortTrack(self._next_id, high_dets[di], self.kf)
            self._next_id += 1
            if self.frame_idx == 1:
                trk.hits = max(trk.hits, self.min_hits)
            self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [
            TrackOutput(t.id, t.bbox, t.conf, t.cls_id, False)
            for t in self.tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]
