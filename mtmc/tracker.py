"""Single-camera tracking (SCT): turn per-frame detections into tracklets.

A lightweight greedy IoU tracker — no motion model needed since detections are
provided and vehicles (incl. parked ones) move slowly relative to the 10 fps
frame rate. Each tracklet is a list of (frame, x, y, w, h) observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PipelineConfig


@dataclass
class Tracklet:
    cam_id: str
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)  # [x, y, w, h]

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def last_box(self) -> np.ndarray:
        return self.boxes[-1]

    def add(self, frame: int, box: np.ndarray) -> None:
        self.frames.append(frame)
        self.boxes.append(box)

    def to_xyxy(self) -> np.ndarray:
        b = np.asarray(self.boxes, dtype=float)
        out = b.copy()
        out[:, 2] = b[:, 0] + b[:, 2]
        out[:, 3] = b[:, 1] + b[:, 3]
        return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (Nx4 xyxy) and b (Mx4 xyxy)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one assignment of rows->cols by descending IoU."""
    pairs = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    for score, i, j in sorted(flat, key=lambda t: -t[0]):
        if score < threshold or i in used_rows or j in used_cols:
            continue
        pairs.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return pairs


def track_detections(
    cam_id: str,
    detections: dict[int, np.ndarray],
    cfg: PipelineConfig,
    max_frame: int | None = None,
) -> list[Tracklet]:
    """Associate per-frame dets into tracklets for one camera.

    detections: {frame: Nx4 [x,y,w,h]}. Frames with no dets are skipped and
    count toward `max_unmatched_frames` for each active tracklet.
    """
    tracklets: list[Tracklet] = []
    next_id = 0
    active: dict[int, Tracklet] = {}  # track_id -> tracklet (in last-seen order)

    frames = sorted(detections.keys())
    if max_frame is None:
        max_frame = max(frames, default=0)

    # Walk every frame in the video so empty frames age the tracklets correctly.
    last_seen: dict[int, int] = {}
    for frame in range(1, max_frame + 1):
        dets = detections.get(frame)
        if dets is None or len(dets) == 0:
            # Age all active tracklets; drop expired ones.
            for tid in list(active):
                if frame - last_seen[tid] > cfg.max_unmatched_frames:
                    tracklets.append(active.pop(tid))
            continue

        cur_xyxy = np.c_[dets[:, 0], dets[:, 1], dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]]
        if active:
            active_ids = list(active)
            prev_xyxy = np.vstack([active[t].to_xyxy()[-1] for t in active_ids])
            iou = iou_matrix(cur_xyxy, prev_xyxy)
            matches = _greedy_match(iou, cfg.iou_threshold)
            matched_cur: set[int] = set()
            matched_prev: set[int] = set()
            for ci, pi in matches:
                tid = active_ids[pi]
                active[tid].add(frame, dets[ci])
                last_seen[tid] = frame
                matched_cur.add(ci)
                matched_prev.add(pi)
        else:
            matched_cur, matched_prev = set(), set()

        # Unmatched prev tracklets: age them.
        for pi, tid in enumerate(active_ids if active else []):
            if pi not in matched_prev:
                if frame - last_seen[tid] > cfg.max_unmatched_frames:
                    tracklets.append(active.pop(tid))

        # Unmatched new dets start fresh tracklets.
        for ci in range(len(dets)):
            if ci in matched_cur:
                continue
            t = Tracklet(cam_id=cam_id, track_id=next_id)
            next_id += 1
            t.add(frame, dets[ci])
            active[t.track_id] = t
            last_seen[t.track_id] = frame

    # Flush remaining active tracklets.
    tracklets.extend(active.values())

    # Filter: keep tracklets long enough to carry a stable Re-ID embedding.
    kept = []
    for t in tracklets:
        n = len(t.frames)
        if n >= cfg.min_tracklet_frames:
            duration = (t.end_frame - t.start_frame) / cfg.fps
            if duration >= cfg.min_tracklet_duration_s:
                kept.append(t)
    return kept
