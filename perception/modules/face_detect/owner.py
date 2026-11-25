"""Which person box a face belongs to when person boxes overlap.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU.

The face network runs on every person box (config_sgie_face.txt:
process-mode=2) and the face's parent pointer arrives unset on this pipeline,
so the probe decides by position. The old rule -- the smallest box containing
the face centre -- gave a barber's face to the seated customer she leaned
over (a seated box is the smaller one), and the barber's name then bounced
between the two tracks (barber clip, 2026-09-15: two thirds of all name moves
were one barber's). Now, among the boxes containing the face centre:

1. the track this face was on a moment ago -- faces move smoothly, so a face
   drifting into a second box stays with the body it came from; else
2. the box whose top is nearest the face top -- a head is at the top of its
   own body box -- never one the face sticks out above; else
3. the box with the highest top -- the face sticks out above every box
   (a clipped head), and the body it sits on is the tallest reaching it,
   not the smaller, seated box.

The same face found in two overlapping boxes arrives twice; it is kept once.

Which of these a product gets is its own setting (`face_owner.mode` in its
sidecar yaml), because the choice changes recognition for every camera of the
process: `tracked` is the rule above (barber, whose barbers lean over seated
customers), `legacy` is `assign_legacy` below -- what attendance's face-ID
thresholds were tuned against (2026-09-22).
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence

logger = logging.getLogger("iris.face_owner")

# (left, top, width, height, confidence, parent_id) -- the probe's face tuple
Face = tuple
# (left, top, width, height, object_id)
Person = tuple


class FaceOwnership:
    """One per pipeline; history is kept per source."""

    def __init__(self, memory_s: float = 2.0, follow_radius: float = 1.0,
                 stick_out: float = 0.25, same_face: float = 0.5,
                 fallback: str = "highest-top"):
        self.memory_s = memory_s            # a track's last face position is trusted this long...
        self.follow_radius = follow_radius  # ...for faces within this many face diagonals of it
        self.stick_out = stick_out          # a box top may sit this share of the face height below the face top
        self.same_face = same_face          # two detections overlapping this share of the smaller are one face
        if fallback not in ("highest-top", "smallest"):
            raise ValueError(f"face_owner fallback must be 'highest-top' or 'smallest', got {fallback!r}")
        self.fallback = fallback            # last resort when rules 1 and 2 decide nothing
        # How often rule 3 (the highest-top fallback, 2026-09-20) decides, and
        # how often it picks another box than the old smallest-box rule did --
        # it went live bundled with other changes and was never measured alone.
        # Logged and reset every report_s.
        self.report_s = 60.0
        self.stats = {"contested": 0, "fallback": 0, "fallback_changed": 0}
        self._window_start: Optional[float] = None
        # source -> track id -> (time, face centre x, y)
        self._last: dict[str, dict[int, tuple[float, float, float]]] = {}

    def assign(self, source_id: str, faces: Sequence[Face], persons: Sequence[Person],
               now: float) -> list[Face]:
        """The frame's faces, duplicates dropped, each with its owner's track
        id in the parent_id slot (the face's own parent when no box holds it)."""
        last = self._last.setdefault(source_id, {})
        for tid in [t for t, (ts, _x, _y) in last.items() if now - ts > self.memory_s]:
            del last[tid]
        out = [(*f[:5], self._owner(f, top, persons, last)) for f, top in _one_per_face(faces, self.same_face)]
        for f in out:
            if f[5] is not None:
                last[f[5]] = (now, f[0] + f[2] / 2.0, f[1] + f[3] / 2.0)
        self._report(now)
        return out

    def _report(self, now: float) -> None:
        if self._window_start is None:
            self._window_start = now
        elif now - self._window_start >= self.report_s:
            logger.info("face ownership, last %.0f s: %d faces in 2+ boxes, %d settled by the highest-top "
                        "fallback, %d of those differ from the old smallest-box rule",
                        now - self._window_start, self.stats["contested"], self.stats["fallback"],
                        self.stats["fallback_changed"])
            self.stats = dict.fromkeys(self.stats, 0)
            self._window_start = now

    def _owner(self, f: Face, face_top: float, persons: Sequence[Person],
               last: dict[int, tuple[float, float, float]]) -> Optional[int]:
        left, _top, w, h = f[:4]
        cx, cy = left + w / 2.0, f[1] + h / 2.0
        inside = [p for p in persons if p[0] <= cx <= p[0] + p[2] and p[1] <= cy <= p[1] + p[3]]
        if not inside:
            return f[5]
        if len(inside) == 1:
            return inside[0][4]
        self.stats["contested"] += 1
        reach = self.follow_radius * math.hypot(w, h)
        near = [(math.hypot(cx - last[p[4]][1], cy - last[p[4]][2]), p[4]) for p in inside if p[4] in last]
        near = [n for n in near if n[0] <= reach]
        if near:
            return min(near)[1]
        heads = [(abs(face_top - p[1]), p[4]) for p in inside if p[1] <= face_top + self.stick_out * h]
        if heads:
            return min(heads)[1]
        highest = min(inside, key=lambda p: p[1])[4]
        smallest = min(inside, key=lambda p: p[2] * p[3])[4]
        self.stats["fallback"] += 1
        if highest != smallest:
            self.stats["fallback_changed"] += 1
        return highest if self.fallback == "highest-top" else smallest


def _overlap(a: Face, b: Face) -> float:
    """Intersection over the smaller box's area (0 when disjoint)."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return ix * iy / smaller if smaller > 0 else 0.0


def _one_per_face(faces: Sequence[Face], same_face: float) -> list[tuple[Face, float]]:
    """Each face once: the most confident detection, with the highest top any
    detection of it reached -- a box that cut the face off at its own top edge
    reports a shorter face than the one the face really is. Input order."""
    kept: list[list] = []  # [index, face, top]
    for i, f in sorted(enumerate(faces), key=lambda e: -e[1][4]):
        for k in kept:
            if _overlap(f, k[1]) >= same_face:
                k[2] = min(k[2], f[1])
                break
        else:
            kept.append([i, f, f[1]])
    return [(f, top) for _i, f, top in sorted(kept, key=lambda k: k[0])]


def crop_box(face: Face, persons: Sequence[Person],
             to_points: Callable[[Person], Optional[list]], slack: float = 0.2) -> Optional[tuple]:
    """The person box the face network ran on: its landmarks come in network
    coords for that box, and only that box maps them back inside the face
    (to_points: person -> 5 [x, y] points in the face's coords, or None).
    The owner can be a different box -- a face found in the customer's box
    may be the barber's."""
    left, top, w, h = face[:4]
    x0, x1 = left - slack * w, left + w + slack * w
    y0, y1 = top - slack * h, top + h + slack * h
    best, best_hits = None, 3   # at least 4 of the 5 points inside
    for p in persons:
        pts = to_points(p)
        if not pts:
            continue
        hits = sum(1 for x, y in pts if x0 <= x <= x1 and y0 <= y <= y1)
        if hits > best_hits:
            best, best_hits = tuple(p[:4]), hits
    return best


def assign_legacy(faces: Sequence[Face], persons: Sequence[Person]) -> list[Face]:
    """The rule attendance ran before `tracked` existed, kept as a choice.

    Only a face the detector left unowned is matched, and it goes to the
    SMALLEST person box containing its centre. No continuity, no head-at-top,
    no duplicate removal -- a face the detector already attached to a person is
    never moved. Attendance's recognition thresholds (votes, lock, margin) were
    calibrated against exactly this behaviour, so switching a product to
    `tracked` re-opens that calibration.
    """
    out: list = []
    for f in faces:
        left, top, w, h, conf, pid = f[0], f[1], f[2], f[3], f[4], f[5]
        if pid is None and persons:
            cx, cy = left + w / 2.0, top + h / 2.0
            best, best_area = None, float("inf")
            for pl, pt, pw, ph, oid in persons:
                if pl <= cx <= pl + pw and pt <= cy <= pt + ph and pw * ph < best_area:
                    best, best_area = oid, pw * ph
            pid = best
        out.append((left, top, w, h, conf, pid))
    return out
