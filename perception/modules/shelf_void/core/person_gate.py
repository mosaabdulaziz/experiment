# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/person_gate.py. Copied unchanged below this header: fix it there first, then copy it again.
"""Pause void detection in a region while a person is in or near it.

A person standing in front of a shelf hides products (looks like a gap) or hides gaps.
So a region is only "active" once no (padded) person box has touched it for `resume_s` seconds.
At 25 fps the original 10-frame rule is resume_s = 0.4.
"""
import cv2
import numpy as np

from .tracker import EPS


def pad_boxes(boxes, pad, w, h):
    """Grow each (x1, y1, x2, y2) box by `pad` of its size per side, clipped to the frame; ints."""
    out = []
    for x1, y1, x2, y2 in boxes:
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        out.append((max(int(x1 - pw), 0), max(int(y1 - ph), 0),
                    min(int(x2 + pw), w), min(int(y2 + ph), h)))
    return out


def region_masks(regions, w, h):
    """One uint8 mask (1 inside) per pixel polygon."""
    masks = []
    for reg in regions:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.asarray(reg, np.int32)], 1)
        masks.append(m)
    return masks


class PersonGate:
    """Tracks, per region, when a person last touched it."""

    def __init__(self, masks, resume_s):
        self.masks, self.resume_s = masks, resume_s
        self.last_touched = [None] * len(masks)  # None = never touched, so regions start active

    def touched(self, persons):
        """Which regions any (already padded) person box overlaps."""
        return [any(m[y1:y2, x1:x2].any() for x1, y1, x2, y2 in persons) for m in self.masks]

    def update(self, persons, now):
        """Record person boxes seen at `now`; return the list of active (person-free) flags."""
        for i, hit in enumerate(self.touched(persons)):
            if hit:
                self.last_touched[i] = now
        return self.active(now)

    def active(self, now):
        return [lt is None or now - lt >= self.resume_s - EPS for lt in self.last_touched]
