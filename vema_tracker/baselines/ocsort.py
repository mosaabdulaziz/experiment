"""Faithful OC-SORT baseline (8D Kalman + OCM + OCR + ORU, Cao et al. CVPR 2023)."""

from __future__ import annotations

import numpy as np

from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.costs import iou_matrix

from .base import BaseTracker, Detection, TrackOutput
from .kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


def _k_previous_obs(observations: dict[int, np.ndarray], cur_age: int, k: int) -> np.ndarray | None:
    if not observations:
        return None
    for delta in range(k):
        dt = k - delta
        if (cur_age - dt) in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    c1 = np.array([0.5 * (bbox1[0] + bbox1[2]), 0.5 * (bbox1[1] + bbox1[3])], dtype=np.float64)
    c2 = np.array([0.5 * (bbox2[0] + bbox2[2]), 0.5 * (bbox2[1] + bbox2[3])], dtype=np.float64)
    diff = c2 - c1
    norm = float(np.linalg.norm(diff)) + 1e-6
    return diff / norm


class _OCSortTrack:
    def __init__(self, track_id: int, det: Detection, kf: KalmanFilterXYAH, delta_t: int = 3) -> None:
        self.id = track_id
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.delta_t = delta_t
        self.observations: dict[int, np.ndarray] = {0: det.bbox.copy()}
        self.last_observation: np.ndarray = det.bbox.copy()
        self.velocity: np.ndarray | None = None
        self.frozen = False
        self.history_frozen: tuple[np.ndarray, np.ndarray] | None = None

    def predict(self, dt_steps: float = 1.0) -> np.ndarray:
        if self.mean[7] + self.mean[3] <= 0:
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov, dt_steps)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return xyah_to_xyxy(self.mean[:4])

    def update(self, det: Detection | None) -> None:
        if det is None:
            if not self.frozen:
                self.frozen = True
                self.history_frozen = (self.mean.copy(), self.cov.copy())
            return

        bbox = det.bbox
        # ORU: Observation-Centric Re-Update across occluded gap
        if self.frozen and self.history_frozen is not None:
            gap = max(self.time_since_update, 1)
            prev_box = self.last_observation
            self.mean, self.cov = self.history_frozen
            for step in range(1, gap):
                t = step / float(gap)
                virt_box = prev_box + t * (bbox - prev_box)
                self.mean, self.cov = self.kf.predict(self.mean, self.cov, 1.0)
                self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(virt_box))
            self.mean, self.cov = self.kf.predict(self.mean, self.cov, 1.0)
            self.frozen = False
            self.history_frozen = None

        prev_obs = _k_previous_obs(self.observations, self.age, self.delta_t)
        if prev_obs is not None:
            self.velocity = _speed_direction(prev_obs, bbox)

        self.last_observation = bbox.copy()
        self.observations[self.age] = bbox.copy()
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(bbox))
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


class OCSORTBaseline(BaseTracker):
    """Observation-Centric SORT (Cao et al., CVPR 2023) with OCM, OCR, and ORU."""

    name = "ocsort"

    def __init__(
        self,
        det_thresh: float = 0.60,
        low_thresh: float = 0.15,
        iou_thresh: float = 0.25,
        low_iou_thresh: float = 0.30,
        max_age: int = 30,
        min_hits: int = 2,
        delta_t: int = 3,
        inertia: float = 0.20,
        use_byte: bool = True,
        check_class_match: bool = False,
        dt_aware: bool = False,
        ref_fps: float = 30.0,
    ) -> None:
        self.det_thresh = det_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.low_iou_thresh = low_iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.check_class_match = check_class_match
        self.dt_aware = dt_aware
        self.ref_fps = ref_fps
        self.kf = KalmanFilterXYAH()
        self.tracks: list[_OCSortTrack] = []
        self._next_id = 1
        self.frame_idx = 0

    def reset(self, fps: float = 30.0) -> None:
        if fps > 0:
            self.ref_fps = float(fps)
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0

    def _ocm_cost(self, tracks: list[_OCSortTrack], dets: list[Detection]) -> np.ndarray:
        t_boxes = np.stack([t.bbox for t in tracks], axis=0)
        d_boxes = np.stack([d.bbox for d in dets], axis=0)
        iou = iou_matrix(t_boxes, d_boxes)

        # Velocity direction consistency cost (OCM)
        angle_cost = np.zeros_like(iou)
        for ti, trk in enumerate(tracks):
            prev_obs = _k_previous_obs(trk.observations, trk.age, trk.delta_t)
            if trk.velocity is None or prev_obs is None:
                continue
            for di, det in enumerate(dets):
                cand_dir = _speed_direction(prev_obs, det.bbox)
                cos_val = float(np.clip(np.dot(trk.velocity, cand_dir), -1.0, 1.0))
                # Normalized angular difference in [0, 1]
                angle_cost[ti, di] = np.arccos(cos_val) / np.pi

        cost = (1.0 - iou) + self.inertia * angle_cost
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

        high_dets = [d for d in detections if d.conf >= self.det_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.conf < self.det_thresh]

        for trk in self.tracks:
            trk.predict(dt_steps)

        matched_trks: set[int] = set()
        u_trks = list(range(len(self.tracks)))
        u_high = list(range(len(high_dets)))

        # Stage 1: OCM (IoU + velocity angle) <-> high_dets
        if self.tracks and high_dets:
            cost1 = self._ocm_cost(self.tracks, high_dets)
            # Cost threshold slightly looser than 1 - iou_thresh to permit inertia term
            m1, u_trks, u_high = linear_assignment(cost1, (1.0 - self.iou_thresh) + self.inertia * 0.5)
            for ti, di in m1:
                self.tracks[ti].update(high_dets[di])
                matched_trks.add(ti)

        # Stage 2 (BYTE): remaining unmatched tracks <-> low_dets via IoU
        if self.use_byte and u_trks and low_dets:
            rem_trks = [self.tracks[ti] for ti in u_trks]
            t_boxes = np.stack([t.bbox for t in rem_trks], axis=0)
            d_boxes = np.stack([d.bbox for d in low_dets], axis=0)
            cost2 = 1.0 - iou_matrix(t_boxes, d_boxes)
            if self.check_class_match:
                tc = np.array([t.cls_id for t in rem_trks])[:, None]
                dc = np.array([d.cls_id for d in low_dets])[None, :]
                cost2 = np.where(tc != dc, np.inf, cost2)
            m2, u_local2, _ = linear_assignment(cost2, 1.0 - self.low_iou_thresh)
            for lti, di in m2:
                ti = u_trks[lti]
                self.tracks[ti].update(low_dets[di])
                matched_trks.add(ti)
            u_trks = [u_trks[lti] for lti in u_local2]

        # Stage 3 (OCR): Observation-Centric Recovery using last_observation <-> remaining high_dets
        if u_trks and u_high:
            rem_trks = [self.tracks[ti] for ti in u_trks]
            rem_dets = [high_dets[di] for di in u_high]
            t_obs = np.stack([t.last_observation for t in rem_trks], axis=0)
            d_boxes = np.stack([d.bbox for d in rem_dets], axis=0)
            cost3 = 1.0 - iou_matrix(t_obs, d_boxes)
            if self.check_class_match:
                tc = np.array([t.cls_id for t in rem_trks])[:, None]
                dc = np.array([d.cls_id for d in rem_dets])[None, :]
                cost3 = np.where(tc != dc, np.inf, cost3)
            m3, u_local3, u_det3 = linear_assignment(cost3, 1.0 - self.iou_thresh)
            for lti, ldi in m3:
                ti = u_trks[lti]
                di = u_high[ldi]
                self.tracks[ti].update(high_dets[di])
                matched_trks.add(ti)
            u_trks = [u_trks[lti] for lti in u_local3]
            u_high = [u_high[ldi] for ldi in u_det3]

        # Mark remaining unmatched tracks as missed (freeze Kalman for ORU)
        for ti in u_trks:
            self.tracks[ti].update(None)

        # Spawn new tracks for unmatched high_dets
        for di in u_high:
            trk = _OCSortTrack(self._next_id, high_dets[di], self.kf, delta_t=self.delta_t)
            self._next_id += 1
            if self.frame_idx == 1:
                trk.hits = max(trk.hits, self.min_hits)
            self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [
            TrackOutput(t.id, t.bbox, t.conf, t.cls_id, False)
            for t in self.tracks
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits)
        ]
