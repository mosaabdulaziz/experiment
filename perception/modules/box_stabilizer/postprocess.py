"""Post-processing shared by the review annotator and the live pipeline.

The 1-class ``vehicle`` detector occasionally emits a tight box on the tractor
unit (cab) fully inside the box covering the whole vehicle (cab+cargo).  Plain
NMS does not remove it because the IoU of the inner box with its container is
only ``inner_area / outer_area`` (~0.4-0.6), well under the iou=0.45-0.7 NMS bar.
Containment suppression drops any box whose area is at least ``ratio`` inside
another box of the same frame, keeping the outer (whole-vehicle) box.

This is a live-pipeline copy of the top-level ``mtmc/detection_postprocess.py`` (offline tooling); the two are
kept in sync by hand.
"""
from __future__ import annotations

import numpy as np


def containment_suppression(xyxy: np.ndarray, ratio: float = 0.85,
                            conf: np.ndarray | None = None,
                            center: bool = True,
                            iou_min: float = 0.3) -> np.ndarray:
    """Return a boolean keep-mask over ``xyxy`` (Nx4 rows x1,y1,x2,y2).

    A box is dropped when it is nested inside a LARGER box of the same frame,
    where "nested" means either:
      * area-containment: >= ``ratio`` (0.85) of the box's area is covered by
        the larger box, or
      * center-residual: the box's CENTER lies inside the larger box AND their
        IoU >= ``iou_min`` (0.3).  This catches a cab/tractor box that covers a
        big chunk of the whole-vehicle box (containment 0.5-0.8) or pokes
        slightly outside it -- the case that pure area containment misses.
    Two boxes that are mutually nested (nearly identical) keep the higher
    confidence one, otherwise the outer/container box wins.  Separated vehicles
    (disjoint centers, low overlap) are untouched.

    Args:
        xyxy: (N, 4) float boxes in x1,y1,x2,y2 order.
        ratio: minimum area-containment fraction to suppress.
        conf: optional (N,) confidence values used to break near-identical ties.
        center: enable the center + IoU residual check (default True).
        iou_min: minimum IoU for the center-based check.

    Returns:
        Boolean array of length N, True = keep.
    """
    n = len(xyxy)
    if n < 2:
        return np.ones(n, dtype=bool)

    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = np.maximum(0.0, (x2 - x1) * (y2 - y1))
    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    np.fill_diagonal(inter, 0.0)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / np.where(union > 0, union, 1.0)
        frac = inter / areas[:, None]  # frac[i, j] = fraction of box i inside box j

    bigger = areas[None, :] > areas[:, None]  # bigger[a, b] = areas[b] > areas[a]

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dropped = False
        for j in range(n):
            if i == j:
                continue
            if not bigger[i, j]:  # only a LARGER box (j) may swallow box i
                continue
            contained = frac[i, j] >= ratio
            centered = center and (_in_box(x1[i], y1[i], x2[i], y2[i],
                                           x1[j], y1[j], x2[j], y2[j])
                                   and iou[i, j] >= iou_min)
            if not (contained or centered):
                continue
            near_dup = (iou[i, j] >= 0.7
                        or (frac[j, i] >= ratio
                            and _in_box(x1[j], y1[j], x2[j], y2[j],
                                        x1[i], y1[i], x2[i], y2[i])))
            if near_dup:
                # duplicate detections of the same object -> keep higher conf
                scores = conf if conf is not None else areas
                if scores[j] >= scores[i]:
                    keep[i] = False
                else:
                    keep[j] = False
            else:
                keep[i] = False
            dropped = True
            break
    return keep


def _in_box(x1: float, y1: float, x2: float, y2: float,
            bx1: float, by1: float, bx2: float, by2: float) -> bool:
    """True if the center of box (x1..y2) lies inside box (bx1..by2)."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return bx1 <= cx <= bx2 and by1 <= cy <= by2
