"""Deterministic color palette for MTMC global IDs.

Each global_id maps to a unique, visually distinct color via golden-ratio
spacing in HSV hue space.  The mapping is stable: the same global_id always
produces the same color, regardless of how many IDs exist.

Pure Python — no external deps.
"""
from __future__ import annotations

import colorsys
from functools import lru_cache

# Number of distinct hues before cycling.  36 gives ~10° spacing —
# easily distinguishable by eye.
_N_HUES = 36

# Pre-computed palette: BGR tuples for OpenCV drawing.
# Hue is spaced by the golden ratio so相邻 colors are maximally distinct.
_PALETTE_BGR: list[tuple[int, int, int]] = []

for _i in range(_N_HUES):
    h = ((_i * 0.618033988749895) % 1.0)  # golden-ratio spacing
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    _PALETTE_BGR.append((int(b * 255), int(g * 255), int(r * 255)))


def gid_color(global_id: int) -> tuple[int, int, int]:
    """Return a BGR color tuple for the given global_id.

    Same global_id → same color, always.  Colors cycle after _N_HUES ids
    but adjacent ids are maximally distinct.
    """
    return _PALETTE_BGR[global_id % _N_HUES]


def gid_color_rgb(global_id: int) -> tuple[int, int, int]:
    """Return an RGB color tuple (for text/UI)."""
    b, g, r = _PALETTE_BGR[global_id % _N_HUES]
    return (r, g, b)
