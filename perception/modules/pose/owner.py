"""Which person a keypoint set belongs to, and which crop the network saw.

Pure Python (no gi/pyds imports), so it stays unit-testable off-GPU.

The pose network runs on ONE tracked person's crop, so its keypoints are that crop's network-input
pixels. Putting them back in the frame needs two things:

- the CROP RECT: the rectangle the network actually saw. The keypoints are in network-input
  pixels; without the crop rect they cannot be placed at all. It is found by geometry: undo the
  letterbox for a candidate person's rect and count the keypoints that land inside the item's own
  box. The right rect puts the joints on the person, so most of them land inside the box the
  network reported for it; a wrong rect puts them elsewhere.
- the OWNER: the tracked person the item belongs to. The secondary GIE's parent pointer arrives
  unset on this pipeline (the same finding the face module works around), so the owner is decided
  by position: the person's box that contains the item's centre, the tightest such box.

Forked from perception/modules/face_detect/owner.py: a module never imports another module
(perception/modules/README.md), so the pattern is copied here rather than shared.
"""
from __future__ import annotations


def hits_inside(points, box) -> int:
    """How many of the points fall inside the box (left, top, width, height), bounds included."""
    left, top, width, height = box
    x1, y1, x2, y2 = left, top, left + width, top + height
    return sum(1 for x, y in points if x1 <= x <= x2 and y1 <= y <= y2)


def crop_for(item_box, persons, to_points, min_hits: int):
    """The person's rect whose letterbox inverse explains the item, and that person.

    `persons` -- the tracked people as (left, top, width, height, id).
    `to_points` -- (rect) -> the keypoints in mux pixels for that rect, or None.
    Returns (person, hits); (None, 0) when no rect explains the item."""
    best, best_hits = None, 0
    for person in persons:
        points = to_points(person[:4])
        if not points:
            continue
        hits = hits_inside(points, item_box)
        if hits > best_hits:
            best, best_hits = person, hits
    if best is None or best_hits < min_hits:
        return None, 0
    return best, best_hits


def owner_by_containment(item_box, persons):
    """The person whose box contains the item's centre, or None.

    The tightest such box wins: the more specific box around the point is the one the item is in."""
    cx = item_box[0] + item_box[2] / 2.0
    cy = item_box[1] + item_box[3] / 2.0
    inside = [p for p in persons
              if p[0] <= cx <= p[0] + p[2] and p[1] <= cy <= p[1] + p[3]]
    if not inside:
        return None
    return min(inside, key=lambda p: p[2] * p[3])
