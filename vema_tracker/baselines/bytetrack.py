"""Faithful ByteTrack baseline (8D Kalman + 2-stage high/low-conf IoU Hungarian)."""

from __future__ import annotations

import numpy as np

from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.costs import iou_matrix

from .base import BaseTracker, Detection, TrackOutput
from .kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


class _ByteTrackItem:
    def __init__(self, track_id: int, det: Detection, kf: KalmanFilterXYAH) -> None:
        self.id = track_id
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.is_activated = False
        self.state = "new"  # "tracked", "lost", "new"
        self.frame_lost = 0
        self.hits = 1

    def activate(self, frame_idx: int) -> None:
        self.state = "tracked"
        self.is_activated = frame_idx == 1
        self.frame_lost = 0

    def re_activate(self, det: Detection) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.state = "tracked"
        self.is_activated = True
        self.frame_lost = 0
        self.hits += 1

    def update(self, det: Detection) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.state = "tracked"
        self.is_activated = True
        self.frame_lost = 0
        self.hits += 1

    def predict(self, dt_steps: float = 1.0) -> None:
        if self.state != "tracked":
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov, dt_steps)

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


class ByteTrackBaseline(BaseTracker):
    """Two-stage Kalman + IoU tracker (Zhang et al., ECCV 2022)."""

    name = "bytetrack"

    def __init__(
        self,
        high_thresh: float = 0.60,
        low_thresh: float = 0.15,
        new_track_thresh: float = 0.60,
        match_thresh_high: float = 0.20,
        match_thresh_low: float = 0.50,
        match_thresh_unconfirmed: float = 0.30,
        track_buffer: int = 30,
        check_class_match: bool = False,
        dt_aware: bool = False,
        ref_fps: float = 30.0,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.match_thresh_unconfirmed = match_thresh_unconfirmed
        self.track_buffer = track_buffer
        self.check_class_match = check_class_match
        self.dt_aware = dt_aware
        self.ref_fps = ref_fps
        self.kf = KalmanFilterXYAH()
        self.tracked: list[_ByteTrackItem] = []
        self.lost: list[_ByteTrackItem] = []
        self._next_id = 1
        self.frame_idx = 0

    def reset(self, fps: float = 30.0) -> None:
        if fps > 0:
            self.ref_fps = float(fps)
        self.tracked = []
        self.lost = []
        self._next_id = 1
        self.frame_idx = 0

    def _iou_cost(self, tracks: list[_ByteTrackItem], dets: list[Detection]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float64)
        t_boxes = np.stack([t.bbox for t in tracks], axis=0)
        d_boxes = np.stack([d.bbox for d in dets], axis=0)
        cost = 1.0 - iou_matrix(t_boxes, d_boxes)
        if self.check_class_match:
            tc = np.array([t.cls_id for t in tracks])[:, None]
            dc = np.array([d.cls_id for d in dets])[None, :]
            cost = np.where(tc != dc, np.inf, cost)
        return cost

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

        high_dets = [d for d in detections if d.conf >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.conf < self.high_thresh]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        strack_pool = tracked + self.lost

        for t in strack_pool:
            t.predict(dt_steps)

        activated: list[_ByteTrackItem] = []
        refind: list[_ByteTrackItem] = []
        new_lost: list[_ByteTrackItem] = []

        # Stage 1: strack_pool <-> high_dets
        cost1 = self._iou_cost(strack_pool, high_dets)
        m1, u_trk1, u_det1 = linear_assignment(cost1, 1.0 - self.match_thresh_high)
        for ti, di in m1:
            trk = strack_pool[ti]
            det = high_dets[di]
            if trk.state == "tracked":
                trk.update(det)
                activated.append(trk)
            else:
                trk.re_activate(det)
                refind.append(trk)

        # Stage 2: remaining tracked (NOT lost) <-> low_dets
        rem_tracked = [strack_pool[ti] for ti in u_trk1 if strack_pool[ti].state == "tracked"]
        cost2 = self._iou_cost(rem_tracked, low_dets)
        m2, u_trk2, _ = linear_assignment(cost2, 1.0 - self.match_thresh_low)
        for ti, di in m2:
            trk = rem_tracked[ti]
            trk.update(low_dets[di])
            activated.append(trk)

        for ti in u_trk2:
            trk = rem_tracked[ti]
            trk.state = "lost"
            trk.frame_lost += 1
            new_lost.append(trk)

        # Keep still-unmatched previously lost tracks
        for ti in u_trk1:
            trk = strack_pool[ti]
            if trk.state == "lost":
                trk.frame_lost += 1

        # Stage 3: unconfirmed <-> remaining high_dets
        rem_high = [high_dets[di] for di in u_det1]
        cost3 = self._iou_cost(unconfirmed, rem_high)
        m3, _, u_det3 = linear_assignment(cost3, 1.0 - self.match_thresh_unconfirmed)
        for ti, di in m3:
            trk = unconfirmed[ti]
            trk.update(rem_high[di])
            activated.append(trk)

        # Spawn new tracks for remaining high_dets
        for di in u_det3:
            det = rem_high[di]
            if det.conf < self.new_track_thresh:
                continue
            trk = _ByteTrackItem(self._next_id, det, self.kf)
            self._next_id += 1
            trk.activate(self.frame_idx)
            activated.append(trk)

        # Prune lost tracks exceeding buffer
        surviving_lost = [
            t for t in (self.lost + new_lost)
            if t.state == "lost" and t.frame_lost <= self.track_buffer
        ]

        # Deduplicate tracked list
        seen_ids: set[int] = set()
        self.tracked = []
        for t in activated + refind:
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                self.tracked.append(t)
        self.lost = [t for t in surviving_lost if t.id not in seen_ids]

        return [
            TrackOutput(t.id, t.bbox, t.conf, t.cls_id, False)
            for t in self.tracked
            if t.is_activated
        ]
