"""Adapter wrapping PolarEMATracker (VEMA v2) behind BaseTracker."""

from __future__ import annotations

import numpy as np

from vema_tracker.core.config import VEMAConfig
from vema_tracker.core.polar_config import VEMA2Config
from vema_tracker.core.polar_tracker import PolarEMATracker

from .base import BaseTracker, Detection, TrackOutput


class VEMA2BaselineAdapter(BaseTracker):
    """BaseTracker adapter around PolarEMATracker (VEMA v2)."""

    name = "vema2"

    def __init__(self, config: VEMAConfig | VEMA2Config | None = None) -> None:
        if isinstance(config, VEMA2Config):
            self.config = config
        elif isinstance(config, VEMAConfig):
            self.config = VEMA2Config.from_vema_config(config)
        else:
            self.config = VEMA2Config()
        self.tracker = PolarEMATracker(self.config)

    def reset(self, fps: float = 30.0) -> None:
        self.tracker.reset(fps=fps)

    def update(
        self,
        detections: list[Detection],
        dt: float | None = None,
    ) -> list[TrackOutput]:
        if detections:
            boxes = np.stack([d.bbox for d in detections], axis=0)
            scores = np.array([d.conf for d in detections], dtype=np.float64)
            clss = np.array([d.cls_id for d in detections], dtype=np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float64)
            scores = np.empty((0,), dtype=np.float64)
            clss = np.empty((0,), dtype=np.int64)

        raw_out = self.tracker.update(boxes, scores, clss, dt=dt)
        return [
            TrackOutput(track_id=tid, bbox=bbox, conf=conf, cls_id=cid, is_shadow=shadow)
            for tid, bbox, conf, cid, shadow in raw_out
        ]
