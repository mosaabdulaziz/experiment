"""A detected object's instance mask -> the few points the proximity rule needs, in mux pixels.

DeepStream attaches each object's mask as a score grid laid over its box (`perception.engine.nvds.InstanceMask`; the
YOLO26-seg parser makes every grid netW/4 x netH/4, whatever the box size). Only the largest blob counts, so a stray
fragment under a person does not move their foot, and the grid is first thinned to about one cell per mux pixel, so a
small, far person costs no more than its pixels.

Person: the foot (lowest mask point, x averaged over the bottom 5 % of the mask's height), the head (highest point,
likewise) and whether the mask reaches the frame's bottom edge. Vehicle: the lower outline (the lowest mask point of
each column), its ground reference point (median of the outline's lowest third) and the mask area. Objects without a
mask fall back to their box.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import cv2
import numpy as np

BAND = 0.05                 # foot / head band, as a share of the mask's height
OUTLINE_POINTS = 64         # at most this many points on a vehicle's lower outline
EDGE_PX = 2.0               # a mask this close to the frame's bottom edge is cut off by it
RIDER_OVERLAP = 0.8         # a person box this much inside a vehicle box may be on it
FLATBED_MARGIN = 0.25       # person heights the feet must be above the vehicle's outline to stand on it


@dataclass
class Blob:
    """The largest blob of an object's mask: a boolean grid over its box, each cell `cell_w` x `cell_h` mux px."""
    grid: np.ndarray
    left: float
    top: float
    cell_w: float
    cell_h: float

    def xy(self, rows, cols):
        """Mux coordinates of cell centres."""
        return (self.left + (np.asarray(cols, float) + 0.5) * self.cell_w,
                self.top + (np.asarray(rows, float) + 0.5) * self.cell_h)

    def contains(self, x: float, y: float) -> bool:
        r, c = int((y - self.top) // self.cell_h), int((x - self.left) // self.cell_w)
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and bool(self.grid[r, c])

    @property
    def area(self) -> float:
        return float(self.grid.sum()) * self.cell_w * self.cell_h

    @cached_property
    def filled(self) -> np.ndarray:
        """The grid with its holes filled: pixels another object took inside it (a seated driver's) count as its."""
        grid = self.grid.astype(np.uint8)
        contours, _h = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(grid)
        cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
        return out.astype(bool) | self.grid

    def contains_filled(self, x: float, y: float) -> bool:
        """`contains`, on the grid with its holes filled."""
        r, c = int((y - self.top) // self.cell_h), int((x - self.left) // self.cell_w)
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and bool(self.filled[r, c])


def outline_polygon(blob: Blob, epsilon_cells: float = 0.75) -> list:
    """The blob's outer contour, through its edge cells' centres and simplified to within `epsilon_cells`, as
    [[x, y], ...] in mux px: how the preview and the clips draw a mask. [] when it is too small to outline."""
    contours, _h = cv2.findContours(blob.grid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon_cells, True)[:, 0]
    if len(c) < 3:
        return []
    xs, ys = blob.xy(c[:, 1], c[:, 0])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)]


def blob_of(mask, box) -> Optional[Blob]:
    """The largest blob of `mask` (an InstanceMask, or None) laid over `box` (left, top, width, height in mux px), or
    None when there is no usable mask."""
    if mask is None or mask.width <= 0 or mask.height <= 0 or len(mask.values) != mask.width * mask.height:
        return None
    left, top, w, h = (float(v) for v in box)
    if w < 1 or h < 1:
        return None
    sy, sx = max(1, int(mask.height // h)), max(1, int(mask.width // w))
    scores = np.asarray(mask.values, np.float32).reshape(mask.height, mask.width)[::sy, ::sx]
    grid = (scores >= mask.threshold).astype(np.uint8)
    if int(grid.sum()) < 4:
        return None
    _n, labels, stats, _c = cv2.connectedComponentsWithStats(grid, connectivity=8)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return Blob(labels == keep, left, top, w * sx / mask.width, h * sy / mask.height)


@dataclass
class PersonShape:
    foot: tuple                 # (x, y) mux px: the lowest point of the mask as seen
    head: tuple                 # (x, y) the highest point
    touches_bottom: bool        # the mask reaches the frame's bottom edge: the feet are below it
    blob: Optional[Blob] = None


def _band_col(grid: np.ndarray, r0: int, r1: int) -> float:
    """Mean column of the mask cells in rows r0..r1 (inclusive)."""
    return float(np.nonzero(grid[r0:r1 + 1])[1].mean())


def person_shape(blob: Blob, frame_h: float) -> PersonShape:
    rows = np.flatnonzero(blob.grid.any(axis=1))
    r0, r1 = int(rows[0]), int(rows[-1])
    band = max(1, int(round(BAND * (r1 - r0 + 1))))
    fx, fy = blob.xy(r1, _band_col(blob.grid, r1 - band + 1, r1))
    hx, hy = blob.xy(r0, _band_col(blob.grid, r0, r0 + band - 1))
    return PersonShape((float(fx), float(fy)), (float(hx), float(hy)), fy + blob.cell_h / 2 >= frame_h - EDGE_PX, blob)


def person_shape_from_box(box, frame_h: float) -> PersonShape:
    left, top, w, h = (float(v) for v in box)
    return PersonShape((left + w / 2, top + h), (left + w / 2, top), top + h >= frame_h - EDGE_PX)


@dataclass
class VehicleShape:
    outline: np.ndarray         # (n, 2) the lowest mask point of each column, left to right, mux px
    ref: tuple                  # ground reference point: median of the outline's lowest third
    area: float                 # mux px^2
    blob: Optional[Blob] = None


def _ref(outline: np.ndarray) -> tuple:
    low = outline[outline[:, 1] >= np.percentile(outline[:, 1], 66)]
    return float(np.median(low[:, 0])), float(np.median(low[:, 1]))


def vehicle_shape(blob: Blob) -> VehicleShape:
    g = blob.grid
    cols = np.flatnonzero(g.any(axis=0))
    cols = cols[::max(1, -(-len(cols) // OUTLINE_POINTS))]
    lowest = g.shape[0] - 1 - np.argmax(g[::-1, cols], axis=0)
    xs, ys = blob.xy(lowest, cols)
    outline = np.stack([xs, ys], axis=1).astype(np.float64)
    return VehicleShape(outline, _ref(outline), blob.area, blob)


def vehicle_shape_from_box(box) -> VehicleShape:
    left, top, w, h = (float(v) for v in box)
    xs = np.linspace(left, left + w, 16)
    outline = np.stack([xs, np.full_like(xs, top + h)], axis=1)
    return VehicleShape(outline, (left + w / 2, top + h), w * h)


def gap_to_outline(foot: tuple, outline: np.ndarray) -> tuple:
    """(distance in mux px from the foot to the nearest outline point, that point)."""
    d = np.hypot(outline[:, 0] - foot[0], outline[:, 1] - foot[1])
    j = int(np.argmin(d))
    return float(d[j]), (float(outline[j, 0]), float(outline[j, 1]))


def mostly_inside(person_box, vehicle_box) -> bool:
    """At least RIDER_OVERLAP of the person's box lies inside the vehicle's box."""
    pl, pt, pw, ph = person_box
    vl, vt, vw, vh = vehicle_box
    ix = max(0.0, min(pl + pw, vl + vw) - max(pl, vl))
    iy = max(0.0, min(pt + ph, vt + vh) - max(pt, vt))
    return ix * iy >= RIDER_OVERLAP * pw * ph


def is_rider(person_box, visible_foot: tuple, vehicle_box, vehicle: VehicleShape, ph: float,
             feet_hidden: bool) -> bool:
    """Whether a person could be on the vehicle -- its driver, a rider, someone standing on its bed (RESEARCH 5.5), by
    geometry alone: their box lies mostly inside the vehicle's box, and either their visible foot is on the vehicle's
    mask (or the vehicle has no mask), or, with their feet in view, it stands above the vehicle's lower outline at that
    column (a flatbed), or, with their feet hidden, it lies in a hole of the vehicle's mask -- a seated driver's own
    pixels cut one where his lowest visible point is (C189, Ticket 3). A pedestrian just behind or in front of the
    vehicle can pass this too, so the rule decides with how the person moves against the vehicle (camera.py
    `_riding`)."""
    if not mostly_inside(person_box, vehicle_box):
        return False
    if vehicle.blob is None or vehicle.blob.contains(*visible_foot):
        return True
    if feet_hidden:
        return vehicle.blob.contains_filled(*visible_foot)
    j = int(np.argmin(np.abs(vehicle.outline[:, 0] - visible_foot[0])))
    return visible_foot[1] < vehicle.outline[j, 1] - FLATBED_MARGIN * ph
