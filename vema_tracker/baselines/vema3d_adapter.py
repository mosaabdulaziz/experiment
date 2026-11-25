"""Adapter wrapping VEMA3DTracker behind BaseTracker."""

from __future__ import annotations

import numpy as np

from vema_tracker.core.config import VEMAConfig
from vema_tracker.core.vema3d import VEMA3DTracker

from .base import BaseTracker, Detection, TrackOutput


class VEMA3DBaselineAdapter(BaseTracker):
    """BaseTracker adapter around VEMA3DTracker."""

    name = "vema3d"

    def __init__(
        self,
        config: VEMAConfig | None = None,
        img_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        self.config = config or VEMAConfig()
        self.tracker = VEMA3DTracker(self.config, img_size=img_size)

    def reset(
        self,
        fps: float = 30.0,
        img_size: tuple[int, int] | None = None,
    ) -> None:
        self.tracker.reset(fps=fps, img_size=img_size)

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
