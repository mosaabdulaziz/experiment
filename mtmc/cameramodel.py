"""Camera-link model: per-pair spatio-temporal constraints learned from GT.

For a fixed real deployment the cameras are static and calibrated, so we get
two hard constraints for free:

  * temporal  — a vehicle takes a bounded time to travel between two cameras'
                fields of view (per-pair transition-time window);
  * spatial   — a vehicle's ground position must be consistent across cameras
                at overlapping times (per-pair homography).

Both are learned from the GT here (the challenge's train split is the natural
source; in production they'd come from calibration + measured transit times).
The gates are hard: tracklet pairs that fail them are never compared, which is
exactly "compare with fewer vehicles".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .data import CameraInfo


@dataclass
class TrackletObs:
    """A tracklet with the box geometry needed for spatial checks."""
    cam_id: str
    track_id: int
    frames: list[int]                 # local frame numbers
    boxes: np.ndarray                 # Nx4 [x, y, w, h]
    start_s: float                    # global wall-clock start (seconds)
    end_s: float                      # global wall-clock end
    times_s: np.ndarray | None = None  # global wall-clock time per observation


@dataclass
class CameraLinkModel:
    # (cam_a, cam_b) -> H_ab mapping bottom-centers in a's pixels -> b's pixels
    homographies: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # (cam_a, cam_b) -> (lo, hi) transition window in seconds (mid_b - mid_a)
    windows: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ gates
    def spatial_distance(self, a: TrackletObs, b: TrackletObs, time_tol_s: float = 0.3) -> float | None:
        """Median projected bottom-center distance (px) between two tracklets.

        Returns None when the tracklets have no time overlap to compare at.
        """
        H = self.homographies.get((a.cam_id, b.cam_id))
        if H is None:
            H = self.homographies.get((b.cam_id, a.cam_id))
            if H is None:
                return None
            H = np.linalg.inv(H)
        ta = a.times_s if a.times_s is not None else self._frame_times(a)
        tb = b.times_s if b.times_s is not None else self._frame_times(b)
        lo, hi = max(a.start_s, b.start_s), min(a.end_s, b.end_s)
        if lo > hi:
            return None
        mask = (ta >= lo - time_tol_s) & (ta <= hi + time_tol_s)
        if not mask.any():
            return None
        pa = self._bottom_centers(a.boxes)[mask]
        pa_h = np.c_[pa, np.ones(len(pa))]
        proj = pa_h @ H.T
        proj = proj[:, :2] / proj[:, 2:]
        cb = self._bottom_centers(b.boxes)
        dists = []
        for p, t in zip(proj, ta[mask]):
            idx = np.argmin(np.abs(tb - t))
            dists.append(np.linalg.norm(p - cb[idx]))
        return float(np.median(dists))

    def temporal_gate(self, a: TrackletObs, b: TrackletObs, tol_s: float) -> bool:
        """Overlap in wall time, or a small gap (missed detections).

        For near-synchronous cameras viewing the same scene, the same vehicle
        is visible in several cameras at the same wall time, so overlapping
        intervals are the norm. When a learned per-pair transition window is
        available it is used as an additional hard constraint: the difference
        of the tracklet midpoints must fall inside [lo, hi].
        """
        overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
        if overlap >= 0 or -overlap <= tol_s:
            # Near-synchronous cameras: the same vehicle is visible in several
            # cameras at the same wall time — overlapping intervals are the
            # norm, so this pair is temporally feasible.
            return True
        # Disjoint intervals: for near-synchronous cameras this means a
        # different vehicle (the same one would still be in view). The learned
        # per-pair transition windows can relax this for sequential camera
        # setups (vehicle leaves cam A's view, drives, enters cam B's view).
        win = self.windows.get((a.cam_id, b.cam_id))
        if win is None:
            win = self.windows.get((b.cam_id, a.cam_id))
        if win is None:
            return False
        lo, hi = win
        mid_a = (a.start_s + a.end_s) / 2
        mid_b = (b.start_s + b.end_s) / 2
        delta = mid_b - mid_a if (a.cam_id, b.cam_id) in self.windows else mid_a - mid_b
        return lo <= delta <= hi

    def spatial_gate(
        self,
        a: TrackletObs,
        b: TrackletObs,
        max_dist_px: float,
        time_tol_s: float = 1.0,
    ) -> bool:
        """Project a's ground points into b's frame; require proximity at
        overlapping times.

        Uses the (cam_a, cam_b) homography. If no homography is available the
        gate passes (no information, don't reject).
        """
        H = self.homographies.get((a.cam_id, b.cam_id))
        if H is None:
            H = self.homographies.get((b.cam_id, a.cam_id))
            if H is None:
                return True
            # Need b -> a mapping for this orientation: invert.
            H = np.linalg.inv(H)

        # Global wall-clock time of each observation.
        ta = a.times_s if a.times_s is not None else self._frame_times(a)
        tb = b.times_s if b.times_s is not None else self._frame_times(b)

        # Only consider a's observations that fall inside b's time range.
        lo, hi = max(a.start_s, b.start_s), min(a.end_s, b.end_s)
        if lo > hi:
            return True  # no overlap; temporal gate already handled feasibility

        # Bottom centers of a's boxes in the overlapping window.
        mask = (ta >= lo - time_tol_s) & (ta <= hi + time_tol_s)
        if not mask.any():
            return True
        pa = self._bottom_centers(a.boxes)[mask]
        ta_o = ta[mask]

        # Project into b's frame.
        pa_h = np.c_[pa, np.ones(len(pa))]
        proj = (pa_h @ H.T)
        proj = proj[:, :2] / proj[:, 2:]

        # Distance from each projected point to b's bottom-center nearest in time.
        cb = self._bottom_centers(b.boxes)
        dists = []
        for p, t in zip(proj, ta_o):
            idx = np.argmin(np.abs(tb - t))
            dists.append(np.linalg.norm(p - cb[idx]))
        return float(np.median(dists)) <= max_dist_px

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _frame_times(t: TrackletObs) -> np.ndarray:
        return np.asarray(t.frames, dtype=float)

    @staticmethod
    def _bottom_centers(boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=float)
        return np.c_[boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]]


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform homography (src -> dst), normalized."""
    n = len(src)
    a = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        a[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def learn_pairwise_homography(
    gt_a: dict[int, np.ndarray],
    gt_b: dict[int, np.ndarray],
    offset_a_s: float,
    offset_b_s: float,
    fps: float,
    time_tol_s: float = 0.3,
    ransac_thresh_px: float = 25.0,
) -> np.ndarray | None:
    """Fit the cam_a -> cam_b homography from GT bottom-center correspondences.

    Correspondences: same global id, bottom-center positions at the same wall
    time (frames are offset by the cameras' start times). RANSAC for outliers.
    """
    # index b by global time, sorted for windowed lookup
    b_frames = sorted(gt_b.keys())
    b_times = np.array([offset_b_s + (f - 1) / fps for f in b_frames])
    b_by_frame = {f: gt_b[f] for f in b_frames}

    # (frame, id) -> box for b, keyed by id for O(1) id lookup at each frame
    b_ids: dict[int, dict[int, np.ndarray]] = {}
    for f in b_frames:
        m = {}
        for row in gt_b[f]:
            m[int(row[0])] = row[1:5]
        b_ids[f] = m

    src, dst = [], []
    for f_a in sorted(gt_a.keys()):
        t_a = offset_a_s + (f_a - 1) / fps
        lo = np.searchsorted(b_times, t_a - time_tol_s, side="left")
        hi = np.searchsorted(b_times, t_a + time_tol_s, side="right")
        if lo >= hi:
            continue
        for row_a in gt_a[f_a]:
            tid_a = int(row_a[0])
            for k in range(lo, hi):
                box_b = b_ids[b_frames[k]].get(tid_a)
                if box_b is not None:
                    xa, ya, wa, ha = row_a[1:5]
                    xb, yb, wb, hb = box_b
                    src.append((xa + wa / 2, ya + ha))
                    dst.append((xb + wb / 2, yb + hb))

    if len(src) < 8:
        return None
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    h, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh_px)
    return h


def learn_transition_window(
    gt_a: dict[int, np.ndarray],
    gt_b: dict[int, np.ndarray],
    offset_a_s: float,
    offset_b_s: float,
    fps: float,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> tuple[float, float] | None:
    """Transition-time window (mid_a -> mid_b) for a camera pair, from GT.

    Returns None when the pair shares no identity.
    """
    tracks_a: dict[int, list] = {}
    tracks_b: dict[int, list] = {}
    for f, rows in gt_a.items():
        t = offset_a_s + (f - 1) / fps
        for r in rows:
            tracks_a.setdefault(int(r[0]), []).append(t)
    for f, rows in gt_b.items():
        t = offset_b_s + (f - 1) / fps
        for r in rows:
            tracks_b.setdefault(int(r[0]), []).append(t)

    deltas = []
    for tid in set(tracks_a) & set(tracks_b):
        mid_a = (min(tracks_a[tid]) + max(tracks_a[tid])) / 2
        mid_b = (min(tracks_b[tid]) + max(tracks_b[tid])) / 2
        deltas.append(mid_b - mid_a)
    if len(deltas) < 3:
        return None
    return (float(np.percentile(deltas, lo_pct)), float(np.percentile(deltas, hi_pct)))


def learn_camera_link_model(
    gts: dict[str, dict[int, np.ndarray]],
    infos: dict[str, CameraInfo],
    fps: float,
) -> CameraLinkModel:
    """Learn homographies + transition windows for all camera pairs."""
    model = CameraLinkModel()
    cams = list(infos)
    for i, ca in enumerate(cams):
        for cb in cams[i + 1:]:
            h = learn_pairwise_homography(
                gts[ca], gts[cb],
                infos[ca].start_offset_s, infos[cb].start_offset_s, fps,
            )
            if h is not None:
                model.homographies[(ca, cb)] = h
            w = learn_transition_window(
                gts[ca], gts[cb],
                infos[ca].start_offset_s, infos[cb].start_offset_s, fps,
            )
            if w is not None:
                model.windows[(ca, cb)] = w
    return model
