"""Polygon rasters for the chair-first seat rule (project 3, 2026-10-04).

Zones and mask outlines are normalised polygons. Coverage is measured on a fixed GRID (the p1b
spike measured at 960x540: lower-mask overlaps are small, so a coarser grid loses them). A raster
covers only its polygon's bounding box; pixel (c, r) is inside when its centre (c + .5, r + .5)
is, by the even-odd rule, filled one scanline at a time (numpy only, no cv2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = (960, 540)


@dataclass(frozen=True)
class Raster:
    x0: int
    y0: int
    mask: np.ndarray          # bool, rows x cols, over the polygon's box

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def lower(self, frac: float) -> "Raster":
        """The bottom `frac` of the mask's own vertical extent (hips, not head)."""
        rows = np.flatnonzero(self.mask.any(axis=1))
        if frac >= 1.0 or not len(rows):
            return self
        top, bottom = int(rows[0]), int(rows[-1])
        keep = max(1, int(round(frac * (bottom - top + 1))))
        m = self.mask.copy()
        m[: bottom + 1 - keep] = False
        return Raster(self.x0, self.y0, m)

    def overlap(self, other: "Raster") -> int:
        h, w = self.mask.shape
        oh, ow = other.mask.shape
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x0 + w, other.x0 + ow), min(self.y0 + h, other.y0 + oh)
        if x1 <= x0 or y1 <= y0:
            return 0
        a = self.mask[y0 - self.y0:y1 - self.y0, x0 - self.x0:x1 - self.x0]
        b = other.mask[y0 - other.y0:y1 - other.y0, x0 - other.x0:x1 - other.x0]
        return int(np.count_nonzero(a & b))


def _points(points) -> np.ndarray | None:
    if points is None:
        return None
    p = np.asarray(points, np.float64)
    if p.ndim == 1:
        if len(p) % 2:
            return None
        p = p.reshape(-1, 2)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        return None
    return p


def raster(points, grid=GRID) -> Raster | None:
    p = _points(points)
    if p is None:
        return None
    gw, gh = grid
    p = p * (gw, gh)
    x0, x1 = max(0, int(np.floor(p[:, 0].min()))), min(gw, int(np.ceil(p[:, 0].max())) + 1)
    y0, y1 = max(0, int(np.floor(p[:, 1].min()))), min(gh, int(np.ceil(p[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    ys = (np.arange(y0, y1) + 0.5)[:, None]
    a, b = p, np.roll(p, -1, axis=0)
    ya, yb = a[:, 1][None, :], b[:, 1][None, :]
    crosses = (ya > ys) != (yb > ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0][None, :] + (ys - ya) * (b[:, 0] - a[:, 0])[None, :] / (yb - ya)
    xs = np.sort(np.where(crosses, xs, np.inf), axis=1)
    cols = (np.arange(x0, x1) + 0.5)[None, :]
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    for k in range(0, int(crosses.sum(axis=1).max(initial=0)) - 1, 2):
        mask |= (cols >= xs[:, k:k + 1]) & (cols < xs[:, k + 1:k + 2])
    return Raster(x0, y0, mask)
