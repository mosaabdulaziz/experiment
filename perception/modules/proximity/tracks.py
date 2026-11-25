"""What the rule remembers about each track between frames: whether a vehicle is moving, whether a person moves with
it, and where a person's feet were last seen, to put back feet that are hidden now (RESEARCH 5.4 item 4, 5.5, 5.9)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

AREA_JUMP = 0.3        # a mask area change larger than this between two samples is a merge/split, not motion
BORDER_PX = 8.0        # a box edge this near the picture's border is cut off by it: it cannot show motion
MIN_SAMPLES = 6        # positions a window needs before it says anything
MIN_SPAN_S = 0.5       # ... spread over at least this long
LINK_S = 2.0           # a new vehicle id this soon after a lost one, where it was lost, continues its history
SHORT = 0.8            # visible height under this share of the standing height: feet hidden (or bent over)
RATIO_SIGHTINGS = 5    # a person's height ratio: the median of their last this many sightings with the feet in view
                       # (15 reached back ~2 s and put feet 0.03 ph high walking away; Ticket 4 measurement),
RATIO_KEEP_S = 30.0    # ... kept this long after the last one
MIN_VISIBLE = 0.5      # feet hidden, and under this share of a standing person in view: where they stand is a guess


def displacement(points, t: float, window_s: float) -> Optional[tuple]:
    """How far a track moved over the last `window_s` seconds, robustly: the median position of the window's last third
    minus that of its first third, so one frame's jump (a passer-by hiding part of a vehicle) moves neither. `points`
    are (t, x, y) in time order; None with fewer than MIN_SAMPLES of them or under MIN_SPAN_S."""
    win = [p for p in points if 0 <= t - p[0] <= window_s]
    if len(win) < MIN_SAMPLES or win[-1][0] - win[0][0] < min(MIN_SPAN_S, window_s):
        return None
    xy = np.asarray([(p[1], p[2]) for p in win])
    k = len(xy) // 3
    d = np.median(xy[-k:], axis=0) - np.median(xy[:k], axis=0)
    return float(d[0]), float(d[1])


def co_motion(person_points, vehicle_points, t: float, window_s: float) -> Optional[tuple]:
    """How a person moved against a vehicle over the last `window_s`: (their displacement / the vehicle's, the share of
    the vehicle's displacement they made along its direction). A driver or rider travels about as far as the vehicle
    (the direction can differ while it turns: the reference point and the cab move differently); someone standing still
    travels about nothing. Both are measured over the same span -- the window, or less when either track is younger
    (a new track id) -- or a young track would read as moving less. None when either has too little history to tell,
    or the vehicle did not move."""
    if not person_points or not vehicle_points:
        return None
    span = t - max(t - window_s, person_points[0][0], vehicle_points[0][0])
    if span <= 0:
        return None
    dv = displacement(vehicle_points, t, span)
    dp = displacement(person_points, t, span)
    if dv is None or dp is None:
        return None
    travel = math.hypot(*dv)
    if travel < 1e-6:
        return None
    return math.hypot(*dp) / travel, (dp[0] * dv[0] + dp[1] * dv[1]) / (travel * travel)


class Positions:
    """Recent (t, x, y) of each track, forgotten `keep_s` after it was last seen."""

    def __init__(self, keep_s: float = 10.0) -> None:
        self.keep_s = keep_s
        self._pos: dict = {}
        self._seen: dict = {}

    def add(self, tid, t: float, xy: tuple) -> None:
        pos = self._pos.setdefault(tid, deque())
        pos.append((t, float(xy[0]), float(xy[1])))
        self._seen[tid] = t
        while pos and t - pos[0][0] > self.keep_s:
            pos.popleft()

    def points(self, tid):
        return self._pos.get(tid, ())

    def forget(self, t: float) -> None:
        for tid in [k for k, seen in self._seen.items() if t - seen > self.keep_s]:
            self._pos.pop(tid, None)
            self._seen.pop(tid, None)


class VehicleMotion:
    """Whether a tracked vehicle is moving: the displacement of its ground reference point over the last `window_s`
    seconds is more than `motion_px`. A sample whose mask area jumped by more than AREA_JUMP against the one before (two
    vehicles merged, a split, a partial occlusion) adds no position. A new id that appears within `link_px` of a vehicle
    lost less than LINK_S ago is that vehicle re-identified: it takes over its history (ids still break, RESEARCH 4.4).
    A vehicle on the current frame (`present`) is never "lost", wherever it comes in the frame's object list.

    With `box_motion` > 0 its box must move too, by more than that share of `motion_px`: a parked vehicle whose mask
    merges with a neighbour for a few frames moves its reference point, not its box (C189, Ticket 3). The box moved by
    the second largest of its four edges' displacements -- a merge stretches one edge, a vehicle driving sideways moves
    two, one coming towards the camera more -- or, when the box touches the picture's border (a vehicle entering or
    leaving it: one edge moves), by the largest of its edges off the border. A vehicle with no box history yet is
    judged by its point alone.

    The state is steady (Ticket 5: a manoeuvring forklift switched 18 times in 20 s on warehouse camera C26). A moving
    vehicle reads parked only once its window has read parked for `parked_hold_s` in a row; a parked one reads moving
    at once. A vehicle whose window holds too little to tell -- back from a detector dropout, or re-identified -- keeps
    its last state, even if its point came back elsewhere: a parked truck re-detected partly hidden does that too (C189;
    reading it as driving made critical false alarms in Ticket 5's evaluation). The memory is per vehicle, shared by
    the zones that ask."""

    def __init__(self, keep_s: float = 10.0, frame_size: tuple = (0, 0), box_motion: float = 0.0,
                 parked_hold_s: float = 0.0) -> None:
        self.keep_s = keep_s
        self.frame_w, self.frame_h = frame_size
        self.box_motion = float(box_motion)
        self.parked_hold_s = float(parked_hold_s)
        self._pos: dict = {}        # vehicle id -> deque[(t, x, y)]
        self._box: dict = {}        # vehicle id -> deque[(t, left, top, right, bottom)]
        self._area: dict = {}       # vehicle id -> last mask area
        self._last: dict = {}       # vehicle id -> last reference point
        self._seen: dict = {}       # vehicle id -> last time observed
        self._verdict: dict = {}    # vehicle id -> its last state, "moving" or "stationary"
        self._still: dict = {}      # vehicle id -> when its window first read parked since it last read moving

    def observe(self, vid: int, t: float, ref: tuple, area: float, link_px: float = 0.0, present=(),
                box: Optional[tuple] = None) -> None:
        if vid not in self._seen and link_px > 0:
            self._adopt(vid, t, ref, link_px, present)
        prev = self._area.get(vid)
        self._area[vid], self._last[vid], self._seen[vid] = area, ref, t
        pos = self._pos.setdefault(vid, deque())
        if prev is None or abs(area - prev) <= AREA_JUMP * max(prev, 1e-6):
            pos.append((t, float(ref[0]), float(ref[1])))
        while pos and t - pos[0][0] > self.keep_s:
            pos.popleft()
        if box is not None:
            boxes = self._box.setdefault(vid, deque())
            x, y, w, h = (float(v) for v in box)
            boxes.append((t, x, y, x + w, y + h))
            while boxes and t - boxes[0][0] > self.keep_s:
                boxes.popleft()

    def _adopt(self, vid: int, t: float, ref: tuple, link_px: float, present) -> None:
        lost = [(math.hypot(ref[0] - r[0], ref[1] - r[1]), old) for old, r in self._last.items()
                if old not in present and 0 < t - self._seen[old] <= LINK_S]
        near = [c for c in lost if c[0] <= link_px]
        if not near:
            return
        _d, old = min(near)
        self._pos[vid], self._area[vid] = self._pos.pop(old), self._area.pop(old)
        for memory in (self._box, self._verdict, self._still):
            if old in memory:
                memory[vid] = memory.pop(old)
        self._last.pop(old)
        self._seen.pop(old)

    def state(self, vid: int, t: float, window_s: float, motion_px: float) -> Optional[str]:
        """"moving", "stationary", or None while the vehicle has not been seen long enough to tell."""
        verdict = self._window_state(vid, t, window_s, motion_px)
        last = self._verdict.get(vid)
        if verdict is None:
            if last is None:
                return None
            verdict = last
        if verdict == "stationary" and last == "moving" and self.parked_hold_s > 0:
            if t - self._still.setdefault(vid, t) < self.parked_hold_s:
                return "moving"
        self._still.pop(vid, None)
        self._verdict[vid] = verdict
        return verdict

    def _window_state(self, vid: int, t: float, window_s: float, motion_px: float) -> Optional[str]:
        """What the window alone says (no memory): "moving", "stationary", or None with too few positions."""
        d = displacement(self._pos.get(vid, ()), t, window_s)
        if d is None:
            return None
        if math.hypot(*d) <= motion_px:
            return "stationary"
        if self.box_motion > 0:
            moved = self._box_moved(vid, t, window_s)
            if moved is not None and moved <= self.box_motion * motion_px:
                return "stationary"
        return "moving"

    def _box_moved(self, vid: int, t: float, window_s: float) -> Optional[float]:
        boxes = self._box.get(vid)
        if not boxes:
            return None
        edges = [displacement([(b[0], b[1 + e], 0.0) for b in boxes], t, window_s) for e in range(4)]
        if edges[0] is None:
            return None
        moved = [abs(d[0]) for d in edges]
        left, top, right, bottom = boxes[-1][1:]
        cut = (left <= BORDER_PX, top <= BORDER_PX, right >= self.frame_w - BORDER_PX,
               bottom >= self.frame_h - BORDER_PX)
        if any(cut):
            free = [m for m, c in zip(moved, cut) if not c]
            return max(free) if free else None
        return sorted(moved)[-2]

    def points(self, vid: int):
        return self._pos.get(vid, ())

    def forget(self, t: float) -> None:
        for vid in [v for v, seen in self._seen.items() if t - seen > self.keep_s]:
            for memory in (self._pos, self._box, self._area, self._last, self._seen, self._verdict, self._still):
                memory.pop(vid, None)


@dataclass
class Foot:
    point: tuple        # (x, y) mux px: where the rule measures the gap and the zone from
    hidden: bool        # the feet are not in view (behind something, below the frame edge, or the person is bent)
    source: str         # "visible" | "track" (from the person's own height, or before a map their last foot moved
                        # with the head) | "head" (head row + scale map) | "unrecovered" (hidden, nothing to recover
                        # them from: the visible foot is used)
    trusted: bool = True  # False: feet hidden with under MIN_VISIBLE of the person in view (Ticket 4: a man behind a
                          # counter at D41, head and shoulders in view, got feet in a truck's path) -- not measured


class FootAnchors:
    """Where each person's feet were last seen, and their head then; and how tall they are against the scale map's
    standing height (their height ratio), from their recent sightings with the feet in view."""

    def __init__(self, max_age_s: float = 4.0, ratio_keep_s: float = RATIO_KEEP_S) -> None:
        self.max_age_s, self.ratio_keep_s = max_age_s, ratio_keep_s
        self._last: dict = {}       # person id -> (t, foot, head)
        self._ratios: dict = {}     # person id -> (t of the last, deque of visible height / the map's standing height)

    def seen(self, pid: int, t: float, foot: tuple, head: tuple, scale=None) -> None:
        self._last[pid] = (t, foot, head)
        if scale is not None and scale.ready:
            ratios = self._ratios.get(pid, (t, deque(maxlen=RATIO_SIGHTINGS)))[1]
            ratios.append((foot[1] - head[1]) / scale.standing_height_from_head(head[1]))
            self._ratios[pid] = (t, ratios)

    def place(self, pid: int, t: float, head: tuple, scale) -> Optional[tuple]:
        """The hidden foot from the person's own height: the head row plus their height ratio times the map's standing
        height at that row; the column, the last foot's offset from the head. (Moving the last foot with the head put
        it 0.12-0.19 person heights too low walking away from the camera: the head moves less than the feet. Ticket 4
        spike.) None without a ratio from the last ratio_keep_s, or without a map."""
        got, last = self._ratios.get(pid), self._last.get(pid)
        if got is None or last is None or not scale.ready or t - got[0] > self.ratio_keep_s:
            return None
        _t, foot, head0 = last
        ratio = float(np.median(got[1]))
        return (head[0] + foot[0] - head0[0], head[1] + ratio * scale.standing_height_from_head(head[1]))

    def recover(self, pid: int, t: float, head: tuple) -> Optional[tuple]:
        """The last visible foot moved by the head's displacement since (a walking person's head moves with their
        feet; ~0.05 ph accurate at 1-2 s, RESEARCH 4.7), or None when it is older than `max_age_s` or there is none.
        Used until the scale map is ready."""
        last = self._last.get(pid)
        if last is None or t - last[0] > self.max_age_s:
            return None
        _t, foot, head0 = last
        return (foot[0] + head[0] - head0[0], foot[1] + head[1] - head0[1])

    def forget(self, t: float) -> None:
        keep = max(self.max_age_s, self.ratio_keep_s)
        for pid in [p for p, a in self._last.items() if t - a[0] > keep]:
            del self._last[pid]
        for pid in [p for p, (seen, _r) in self._ratios.items() if t - seen > self.ratio_keep_s]:
            del self._ratios[pid]


def feet_hidden(shape, scale) -> bool:
    """The mask is cut off by the frame's bottom edge, or it is shorter than SHORT of the standing height its head row
    predicts (that needs a ready scale map)."""
    if shape.touches_bottom:
        return True
    if not scale.ready:
        return False
    return shape.foot[1] - shape.head[1] < SHORT * scale.standing_height_from_head(shape.head[1])


def resolve_foot(pid: int, t: float, shape, scale, anchors: FootAnchors) -> Foot:
    """Where the person stands. Visible feet are used as seen (and remembered, with the person's height ratio); hidden
    ones come from the person's own height (FootAnchors.place), else the last visible foot moved with the head (no
    map yet), else the head row and the scale map. With a map, hidden feet of someone showing under MIN_VISIBLE of a
    standing person are not trusted."""
    if not feet_hidden(shape, scale):
        anchors.seen(pid, t, shape.foot, shape.head, scale)
        return Foot(shape.foot, False, "visible")
    trusted = (not scale.ready or
               shape.foot[1] - shape.head[1] >= MIN_VISIBLE * scale.standing_height_from_head(shape.head[1]))
    point = anchors.place(pid, t, shape.head, scale) or anchors.recover(pid, t, shape.head)
    if point is not None:
        return Foot(point, True, "track", trusted)
    if scale.ready:
        return Foot((shape.head[0], shape.head[1] + scale.standing_height_from_head(shape.head[1])), True, "head",
                    trusted)
    return Foot(shape.foot, True, "unrecovered")
