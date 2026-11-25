"""A person's segmentation mask -> a compact outline for iris:meta (chair engine, 2026-10-04).

DeepStream's instance mask (obj.mask_params) is a score grid laid over the object's box
(Task 1 confirmed the layout). The outline is the largest blob's contour, simplified to at most
max_points corners, mapped to the frame and normalised 0..1 like the zones -- about 0.5 KB per
person in the stream instead of a full mask."""
from __future__ import annotations

import cv2
import numpy as np


def outline(scores: np.ndarray, threshold: float, box, frame_wh, max_points: int = 40) -> list[float] | None:
    m = (np.asarray(scores) >= threshold).astype(np.uint8)
    if int(m.sum()) < 4:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_points:
        eps *= 1.5
        approx = cv2.approxPolyDP(c, eps, True)
    if len(approx) < 3:
        return None
    mh, mw = m.shape
    left, top, w, h = (float(v) for v in box)
    fw, fh = (float(v) for v in frame_wh)
    pts = approx.reshape(-1, 2).astype(np.float64)
    xs = (left + (pts[:, 0] + 0.5) * w / mw) / fw
    ys = (top + (pts[:, 1] + 0.5) * h / mh) / fh
    return [round(min(1.0, max(0.0, float(v))), 4) for xy in zip(xs, ys) for v in xy]
