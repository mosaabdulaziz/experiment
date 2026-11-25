"""Stable empty-shelf (void) region detection on a video stream.

Copied from the standalone shelf-monitoring project (~/IRIS/shelf-monitoring, commit 02538da, shelf_void/detector.py).
IRIS change: the Ultralytics predict function is gone; in IRIS the boxes come from nvinfer (pgie_shelf) and the module
feeds them in with `update_boxes`. `update(frame)` with a `predict_fn` stays for the ported tests.

The raw YOLO model is low-confidence on angled shelf views, so single-frame
detections are noisy. We keep a decaying heat map of void boxes: real voids stay
in place frame after frame and build up heat, noise flickers and fades out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
RawBox = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf
PredictFn = Callable[[np.ndarray], Sequence[RawBox]]


@dataclass
class DetectorConfig:
    min_area: int = 400          # px^2, drops tiny boxes
    decay: float = 0.85          # heat memory; closer to 1 = steadier but slower
    show_thresh: float = 0.5     # heat needed for a region to count as stable
    close_kernel: Tuple[int, int] = (40, 9)  # joins void fragments along the shelf
    roi: Optional[List[Tuple[int, int]]] = None  # optional polygon; ignore voids outside


@dataclass
class VoidRegion:
    box: Box
    score: float   # mean heat inside the region (0..1), a stability score not a probability


class ShelfVoidDetector:
    def __init__(self, cfg: DetectorConfig, predict_fn: Optional[PredictFn] = None):
        self.cfg = cfg
        self.predict = predict_fn
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(cfg.close_kernel))
        self.heat: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None

    def reset(self):
        self.heat = None

    def _init(self, h: int, w: int):
        self.heat = np.zeros((h, w), np.float32)
        if self.cfg.roi:
            self.roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self.roi_mask, [np.array(self.cfg.roi, np.int32)], 1)

    def update(self, frame: np.ndarray) -> List[VoidRegion]:
        """Feed one frame through `predict_fn`; returns the stable void regions for this frame."""
        h, w = frame.shape[:2]
        return self.update_boxes(self.predict(frame), (w, h))

    def update_boxes(self, boxes: Sequence[RawBox], size: Tuple[int, int]) -> List[VoidRegion]:
        """Feed one frame's void boxes (x1, y1, x2, y2, conf) in a frame of `size` (width, height) pixels; returns the
        stable void regions for this frame."""
        w, h = int(size[0]), int(size[1])
        if self.heat is None or self.heat.shape != (h, w):
            self._init(h, w)

        mask = np.zeros((h, w), np.float32)
        for x1, y1, x2, y2, _conf in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if (x2 - x1) * (y2 - y1) >= self.cfg.min_area:
                mask[y1:y2, x1:x2] = 1
        if self.roi_mask is not None:
            mask *= self.roi_mask

        d = self.cfg.decay
        self.heat = d * self.heat + (1 - d) * mask
        stable = (self.heat > self.cfg.show_thresh).astype(np.uint8)
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, self.kernel)

        regions = []
        contours, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh >= self.cfg.min_area:
                score = float(self.heat[y:y + bh, x:x + bw].mean())
                regions.append(VoidRegion((x, y, x + bw, y + bh), round(score, 3)))
        return regions
