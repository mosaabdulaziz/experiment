"""Same worker or not, from box geometry and time only (no appearance features): the optional re-ID grace, and the
track links the duplicate-alert hint (state.py) groups alerts by.

Track links (always on, TrackLinks): a track that appears, a new id or an id back after missing a frame, continues the
nearest LOST track (not in this frame, last seen at most `max_gap_s` 4 s ago, centre at most
`max_dist` 1.2 of its last box's heights away; one-to-one, closest first). No height test: a close-up or a partial box
changes the height a lot (test2.mp4: a worker re-found at 0.42 x their height). Linked tracks form a chain, one worker
as far as geometry can tell. On test2.mp4 (12 workers, 46 alerts) this links 11 re-finds of the same worker and 3 of
two different workers, so it only groups alerts (possible_duplicate_of); it never drops or merges one.

The test is ppe-compliance's track-stitching rule (`same_worker_distance` in its tracking.py), measured on the
evaluation clips: heights within a factor of `max_height_ratio`, centres at most max(min_dist, max_speed * gap)
person heights apart. One IRIS change: the lost person is represented by their TALLEST box of the last `recent_s`
seconds they were seen, not their last box, because a worker walking behind something often ends with a partial box
(test.mp4: the coverall worker's last box was 52 px, against 97 px when they reappeared).

Re-ID grace (off by default, `enabled`): when nvtracker gives a worker a new id, the new id takes over a recently lost
one (lost at most `max_gap_s` ago, one-to-one, closest pair first, never for boxes taller than `max_box_height` of the
frame, where close-up workers got joined wrongly in ppe-compliance, its issue #13) and smoothing continues on the old
(canonical) id: its window, open episode and cooldown. The canonical id is internal to the temporal filter; alerts and
the overlay always name the tracker's current id. Off by default because a wrong match is worse than a duplicate
alert: on test.mp4 it joined the coverall worker to a compliant worker walking next to them, so the alert named the
wrong person, and a wrong match can also swallow a real alert. With it off, the track links group repeats.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class ReidConfig:
    enabled: bool = False
    max_gap_s: float = 2.0               # how long a lost id can still be taken over
    max_speed: float = 0.7               # person heights per second (ppe-compliance stitching: 1.0; slower here)
    min_dist: float = 0.5                # person heights, allowed whatever the gap
    max_height_ratio: float = 1.5
    max_box_height: float = 0.5          # fraction of the frame height; taller boxes are never matched
    recent_s: float = 1.0                # a lost person is represented by their tallest box of this last stretch


@dataclass
class LinkConfig:
    max_gap_s: float = 4.0               # the lost track was last seen at most this long before
    max_dist: float = 1.2                # centre distance, in heights of the lost track's last box


def centre_distance(a, b) -> float:
    """Distance between the centres of two (x1, y1, x2, y2) boxes, in pixels."""
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return (dx * dx + dy * dy) ** 0.5


def same_worker_distance(old, old_t: float, new, new_t: float, *, max_gap_s: float, max_speed: float,
                         min_dist: float, max_height_ratio: float):
    """Centre distance in (old) person heights if `new` (seen at new_t) can be the worker seen as `old` until old_t,
    else None. Boxes are (x1, y1, x2, y2)."""
    gap = new_t - old_t
    if gap <= 0 or gap > max_gap_s:
        return None
    h_old, h_new = old[3] - old[1], new[3] - new[1]
    if h_old <= 0 or h_new <= 0 or max(h_old, h_new) / min(h_old, h_new) > max_height_ratio:
        return None
    dx = (old[0] + old[2] - new[0] - new[2]) / 2
    dy = (old[1] + old[3] - new[1] - new[3]) / 2
    dist = (dx * dx + dy * dy) ** 0.5 / h_old
    return dist if dist <= max(min_dist, max_speed * gap) else None


class TrackMemory:
    """Per key (a tracker or canonical id): when and where it was first seen, when it was last seen, and its boxes of
    the last `recent_s` seconds."""

    def __init__(self, recent_s: float) -> None:
        self.recent_s = recent_s
        self._m: dict = {}

    def see(self, key, t: float, box) -> None:
        e = self._m.get(key)
        if e is None:
            self._m[key] = {"first": (t, box), "last_t": t, "recent": deque([(t, box)])}
            return
        e["last_t"] = t
        recent = e["recent"]
        recent.append((t, box))
        while t - recent[0][0] > self.recent_s:
            recent.popleft()

    def __contains__(self, key) -> bool:
        return key in self._m

    def keys(self) -> list:
        return list(self._m)

    def first(self, key) -> tuple:
        return self._m[key]["first"]

    def last_t(self, key) -> float:
        return self._m[key]["last_t"]

    def lost_box(self, key) -> tuple:
        """(last time seen, tallest box of the last recent_s seconds)."""
        e = self._m[key]
        return e["last_t"], max(e["recent"], key=lambda tb: tb[1][3] - tb[1][1])[1]

    def forget(self, now: float, older_than: float, keep=()) -> None:
        for k in [k for k, e in self._m.items() if now - e["last_t"] > older_than and k not in keep]:
            del self._m[k]


class ReidGrace:
    def __init__(self, cfg: ReidConfig, forget_after_s: float) -> None:
        self.cfg = cfg
        self.forget_after_s = forget_after_s
        self._canonical: dict = {}       # tracker id -> canonical id (itself, or the lost id it took over)
        self._id_seen: dict = {}         # tracker id -> last time seen (to forget old mappings)
        self._lost = TrackMemory(cfg.recent_s)   # canonical id -> where it was, while it can still be taken over
        # (time, new tracker id, canonical id, distance in heights, (lost time, lost box), new box), for diagnostics
        self.matches: list = []

    def observe(self, now: float, persons: list, frame_height: float) -> dict:
        """persons: this frame's [(tracker id, (x1, y1, x2, y2))]. Returns {tracker id: canonical id} for them; with
        re-ID off every id is its own."""
        cfg = self.cfg
        if not cfg.enabled:
            return {tid: tid for tid, _ in persons}
        max_h = cfg.max_box_height * frame_height if frame_height else float("inf")
        new = [(tid, box) for tid, box in persons if tid not in self._canonical]
        present = {self._canonical[tid] for tid, _ in persons if tid in self._canonical}
        if new:
            pairs = []
            for cid in self._lost.keys():
                if cid in present:
                    continue
                t, old = self._lost.lost_box(cid)
                if old[3] - old[1] > max_h:
                    continue
                for tid, box in new:
                    if box[3] - box[1] > max_h:
                        continue
                    d = same_worker_distance(old, t, box, now, max_gap_s=cfg.max_gap_s, max_speed=cfg.max_speed,
                                             min_dist=cfg.min_dist, max_height_ratio=cfg.max_height_ratio)
                    if d is not None:
                        pairs.append((d, cid, tid, (t, old), box))
            taken_c, taken_t = set(), set()
            for d, cid, tid, lost, box in sorted(pairs, key=lambda p: (p[0], p[1], p[2])):
                if cid in taken_c or tid in taken_t:
                    continue
                taken_c.add(cid)
                taken_t.add(tid)
                self._canonical[tid] = cid
                self.matches.append((now, tid, cid, round(d, 3), lost, box))
        for tid, _ in new:
            self._canonical.setdefault(tid, tid)
        out = {}
        # A canonical id stands for one worker per frame. If an old id comes back (e.g. out of shadow frames) while a
        # new id has already taken it over, both are there: the old id keeps it, the newer one goes back to its own.
        for tid, box in sorted(persons, key=lambda p: self._canonical[p[0]] != p[0]):
            cid = self._canonical[tid]
            if cid in out.values():
                cid = self._canonical[tid] = tid
            out[tid] = cid
            self._id_seen[tid] = now
            self._lost.see(cid, now, box)
        # A lost canonical id can be taken over only within max_gap_s; the id mappings live as long as the
        # smoothing state does (forget_after_s), since a tracker id can come back unchanged after a while.
        self._lost.forget(now, cfg.max_gap_s, keep=set(out.values()))
        for tid in [t for t, seen in self._id_seen.items() if now - seen > self.forget_after_s]:
            del self._id_seen[tid]
            del self._canonical[tid]
        return out


class TrackLinks:
    """Which earlier track each track continues (module docstring, "Track links"). Fed every evaluated frame."""

    def __init__(self, cfg: LinkConfig, forget_after_s: float) -> None:
        self.cfg = cfg
        self.forget_after_s = forget_after_s
        self._last: dict = {}            # tracker id -> (time, box) of its last sighting
        self._chain: dict = {}           # tracker id -> chain of its current appearance
        self._link: dict = {}            # tracker id -> (earlier id, gap s, distance) its current appearance continues
        self._taken: set = set()         # lost ids another appearance has already continued
        self._prev: set = set()          # the ids of the previous evaluated frame
        self._chains = 0

    def observe(self, now: float, persons: list) -> None:
        """persons: this frame's [(tracker id, (x1, y1, x2, y2))]."""
        cfg = self.cfg
        ids = {tid for tid, _ in persons}
        appearing = [(tid, box) for tid, box in persons if tid not in self._prev]
        if appearing:
            # lost: not in this frame (an id switch often happens from one frame to the next), seen within max_gap_s;
            # an id that is back can also continue its own last sighting
            lost = [(tid, t, box) for tid, (t, box) in self._last.items()
                    if (tid not in ids or tid not in self._prev) and tid not in self._taken
                    and 0 < now - t <= cfg.max_gap_s]
            pairs = []
            for old, t, old_box in lost:
                h = old_box[3] - old_box[1]
                for tid, box in appearing:
                    if old in ids and old != tid:   # still here: only itself can continue it
                        continue
                    if h > 0 and (d := centre_distance(old_box, box) / h) <= cfg.max_dist:
                        pairs.append((d, now - t, old, tid))
            done_old, done_new = set(), set()
            for d, gap, old, tid in sorted(pairs, key=lambda p: (p[0], p[1], str(p[2]), str(p[3]))):
                if old in done_old or tid in done_new:
                    continue
                done_old.add(old)
                done_new.add(tid)
                self._taken.add(old)
                if old != tid:                   # an id back where it was lost keeps its chain and link
                    self._chain[tid], self._link[tid] = self._chain[old], (old, gap, d)
            for tid, _ in appearing:
                if tid not in done_new and tid not in self._chain:
                    self._chains += 1            # a new id near no lost track: a worker of its own
                    self._chain[tid], self._link[tid] = self._chains, None
                # an id back with no lost track near it keeps its chain: the tracker re-found it itself
        for tid, box in persons:
            self._last[tid] = (now, box)
            self._taken.discard(tid)
        self._prev = ids
        for tid in [t for t, (seen, _) in self._last.items() if now - seen > self.forget_after_s]:
            del self._last[tid], self._chain[tid], self._link[tid]
            self._taken.discard(tid)

    def chain(self, tid):
        return self._chain.get(tid)

    def link(self, tid) -> "tuple | None":
        """(earlier tracker id, gap s, distance in heights) the track's current appearance continues, or None."""
        return self._link.get(tid)
