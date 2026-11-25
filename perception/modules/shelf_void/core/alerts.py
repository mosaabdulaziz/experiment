"""Turns per-frame void regions into empty-shelf alert events.

Copied unchanged from the standalone shelf-monitoring project (~/IRIS/shelf-monitoring, commit 02538da,
shelf_void/alerts.py). In IRIS `ts` is the engine's monotonic `now` and no frame is passed, so RAISED carries no
snapshot (the module asks the engine for a clip instead).

Each empty place is tracked across frames (IoU matching). An alert is RAISED once
the place has been seen for `confirm_s` seconds, and CLEARED once it has been
missing for `clear_s` seconds (e.g. after restocking). A place that becomes empty
again after being cleared gets a new alert id.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .detector import Box, VoidRegion


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


@dataclass
class AlertConfig:
    camera_id: str = "cam-01"
    confirm_s: float = 2.0     # empty place must persist this long before alerting
    clear_s: float = 3.0       # must be gone this long before clearing
    match_iou: float = 0.3     # same place across frames
    snapshot_pad: int = 20     # px around the box in the snapshot crop


@dataclass
class EmptyAlertEvent:
    event: str                 # "RAISED" | "UPDATED" | "CLEARED"
    alert_id: str
    camera_id: str
    bbox: Box                  # x1, y1, x2, y2 in frame pixels
    score: float
    first_seen: float          # stream timestamp (s)
    timestamp: float
    snapshot: Optional[np.ndarray] = field(default=None, repr=False)  # BGR crop, on RAISED

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "snapshot"}
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class _Track:
    box: Box
    score: float
    first_seen: float
    last_seen: float
    alert_id: Optional[str] = None


class EmptyShelfAlerter:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.tracks: Dict[int, _Track] = {}
        self._ids = itertools.count()

    def update(self, regions: List[VoidRegion], ts: float,
               frame: Optional[np.ndarray] = None) -> List[EmptyAlertEvent]:
        events: List[EmptyAlertEvent] = []
        unmatched = list(regions)

        # match existing tracks to this frame's regions (greedy, best IoU first)
        for tid, t in sorted(self.tracks.items(), key=lambda kv: kv[1].first_seen):
            best = max(unmatched, key=lambda r: iou(t.box, r.box), default=None)
            if best is not None and iou(t.box, best.box) >= self.cfg.match_iou:
                t.box, t.score, t.last_seen = best.box, best.score, ts
                unmatched.remove(best)

        for r in unmatched:
            self.tracks[next(self._ids)] = _Track(r.box, r.score, ts, ts)

        for tid, t in list(self.tracks.items()):
            seen_now = t.last_seen == ts
            if t.alert_id is None and seen_now and ts - t.first_seen >= self.cfg.confirm_s:
                t.alert_id = uuid.uuid4().hex[:12]
                events.append(self._event("RAISED", t, ts, self._crop(frame, t.box)))
            elif t.alert_id is not None and not seen_now and ts - t.last_seen >= self.cfg.clear_s:
                events.append(self._event("CLEARED", t, ts))
                del self.tracks[tid]
            elif t.alert_id is None and not seen_now and ts - t.last_seen >= self.cfg.clear_s:
                del self.tracks[tid]  # never confirmed, drop silently
        return events

    def active(self) -> List[EmptyAlertEvent]:
        return [self._event("UPDATED", t, t.last_seen) for t in self.tracks.values() if t.alert_id]

    def _event(self, kind, t: _Track, ts, snap=None):
        return EmptyAlertEvent(kind, t.alert_id, self.cfg.camera_id, t.box, t.score,
                               t.first_seen, ts, snap)

    def _crop(self, frame, box):
        if frame is None:
            return None
        p = self.cfg.snapshot_pad
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        return frame[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)].copy()
