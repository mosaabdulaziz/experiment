"""Per-camera body state: where each tracked body is (zones -> shop areas,
and the seat it sits in), whether it is still, whether it looks like staff
(vote over sampled crops), its mean body embedding, and door crossings.

Zone membership uses the box's lower torso (FrameGeometry.anchor): a chair
or sofa hides a seated person's feet. A seat is tested along the box's
centre line from its middle down to the anchor (FrameGeometry.seat_points).
The entrance line uses the feet."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from domain import Appearance, Frame
from zones import AREA_TYPES, CHAIR, ENTRANCE_LINE, SEAT, DoorLine, FrameGeometry, area_of, point_in_polygon


def unit(v) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def cosine(a, b) -> float:
    """Both unit length."""
    return sum(x * y for x, y in zip(a, b))


def merge(a, na: int, b, nb: int):
    """Weighted mean of two unit-mean embeddings, kept unit length."""
    if a is None:
        return b, nb
    if b is None:
        return a, na
    return unit(tuple(x * na + y * nb for x, y in zip(a, b))), na + nb


def fold(mean, n: int, v):
    return merge(mean, n, v, 1)


@dataclass(eq=False)
class Tracklet:
    camera: str
    track_id: int
    first_seen: float
    last_seen: float
    anchor: tuple[float, float]
    foot: tuple[float, float]
    box: tuple[float, float, float, float]
    areas: frozenset = frozenset()
    entered: dict = field(default_factory=dict)       # area -> when this track entered it
    seat: str | None = None                           # chair area whose seat holds the body
    history: deque = field(default_factory=lambda: deque(maxlen=256))   # (ts, box middle)
    height: float = 0.0                               # normalized box height
    staff_samples: deque = field(default_factory=lambda: deque(maxlen=15))
    reid: tuple | None = None
    reid_n: int = 0
    door_side: int = 0            # +1 inside, -1 outside, 0 not known yet
    appeared_at_door: bool = False
    person: str | None = None     # global person id (identity.py)
    retired: bool = False         # its person's visit closed: never joins a person again
    staff_outside: bool = False   # its staff vote was True outside every seat staff_outside_confirm_s
    staff_outside_since: float | None = None   # ...since when: start of the current run (a seat frame or non-staff vote restarts it)
    customer: bool = False        # latched: sat in a seat seat_dwell_s -- never staff after
    seat_since: float | None = None
    seat_last: float | None = None
    # 2026-09-27 (39-min testbed spec, section 4), set by the engine each frame:
    owned: frozenset = frozenset()          # its areas its camera owns, or that have no owner
    counted: frozenset | None = None        # its areas it is evidence for (identity_model); None = not set: areas
    reid_owned: tuple | None = None         # mean of the samples taken while owned was non-empty
    reid_owned_n: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera, self.track_id)

    @property
    def evidence(self) -> frozenset:
        """The areas this track is evidence for: counted, or areas before the engine set it."""
        return self.areas if self.counted is None else self.counted

    def still(self, window_s: float, radius: float, box_frac: float = 0.0) -> bool:
        """Seen for at least window_s, and its box middle never farther from
        where it is now than radius, or box_frac x the box's height when that
        is larger (a big box near the camera shakes more), during the last
        window_s."""
        if self.last_seen - self.first_seen < window_s or not self.history:
            return False
        cutoff = self.last_seen - window_s
        r = max(radius, box_frac * self.height)
        now = self.history[-1][1]
        return all(math.hypot(p[0] - now[0], p[1] - now[1]) <= r for t, p in self.history if t >= cutoff)

    def staff(self, min_samples: int, threshold: float, window: int = 15) -> bool | None:
        """None = not enough sampled crops to say (none at all is never enough);
        else the mean of the last `window` crops against threshold."""
        if not self.staff_samples or len(self.staff_samples) < min_samples:
            return None
        return self.staff_mean(window) >= threshold

    def staff_mean(self, window: int = 15) -> float:
        recent = list(self.staff_samples)[-window:]
        return sum(recent) / len(recent)


def _vote(t: Tracklet, policy: dict) -> bool | None:
    return t.staff(int(policy["staff_min_samples"]), policy["staff_threshold"], int(policy.get("staff_window", 15)))


def staff_vote(t: Tracklet, policy: dict) -> bool | None:
    """The staff vote at staff_threshold over the last staff_window crops, the
    same everywhere -- except that a track latched as a customer (see seat_ok)
    is never staff, unless staff_latched_release > 0 and that many last crops
    are all at or above seat_staff_threshold (2026-09-29: a tracker id swap
    handed the seated customer's track to the barber over him)."""
    if t.customer:
        k = int(policy.get("staff_latched_release", 0))
        recent = list(t.staff_samples)[-k:] if k > 0 else []
        return len(recent) == k > 0 and all(p >= policy["seat_staff_threshold"] for p in recent)
    return _vote(t, policy)


def seat_ok(t: Tracklet, policy: dict) -> bool:
    """May this body sit (and start a haircut) in a seat? The zero-shot staff
    score of caped, seated customers overlaps the barbers' (Task 16 tune), so
    in a seat only a mean at or above seat_staff_threshold rules a body out --
    unless its vote already said staff while it stood outside every seat."""
    if t.customer:
        return True
    if t.staff_outside:
        return False
    if _vote(t, policy) is not True:
        return True
    return t.staff_mean(int(policy.get("staff_window", 15))) < policy["seat_staff_threshold"]


@dataclass
class TrackEvents:
    new: list = field(default_factory=list)
    entered: list = field(default_factory=list)       # (tracklet, area)
    left: list = field(default_factory=list)
    crossed_in: list = field(default_factory=list)
    crossed_out: list = field(default_factory=list)
    ended: list = field(default_factory=list)         # tracks over after this camera's frame gap


class TrackletBook:
    def __init__(self, camera: str, zones: list[dict], geometry: FrameGeometry):
        self.camera, self.geometry = camera, geometry
        self.tracks: dict[int, Tracklet] = {}
        self.latest: float | None = None
        self.set_zones(zones)

    def set_zones(self, zones: list[dict]) -> None:
        self.areas = [(area_of(z), z["points"]) for z in zones if z["type"] in AREA_TYPES]
        chairs = {z["id"]: area_of(z) for z in zones if z["type"] == CHAIR}
        self.seats = [(chairs[z["seat_of"]], z["points"]) for z in zones
                      if z["type"] == SEAT and z["seat_of"] in chairs]
        self.doors = [DoorLine.from_zone(z) for z in zones if z["type"] == ENTRANCE_LINE]

    def observe(self, frame: Frame, policy: dict) -> TrackEvents:
        ev = TrackEvents()
        if self.latest is not None and frame.ts - self.latest > policy["track_gone_s"]:
            # 2026-09-26 (final B review C1a): this camera sent no frame for track_gone_s --
            # a stall or a sidecar restart, and the tracker's ids restart with the sidecar
            # (config_tracker_nvdcf_barber.yml useUniqueID 0): every track from before is
            # over, and an id seen again is someone new. The caller releases them. (Keyed on
            # the camera's frame gap, not each id's: a track back after ~2 s in a running
            # stream is the tracker's own reacquisition, which expire() judges per cycle.)
            ev.ended = list(self.tracks.values())
            self.tracks = {}
        self.latest = frame.ts if self.latest is None else max(self.latest, frame.ts)
        for obs in frame.tracks:
            anchor, foot = self.geometry.anchor(obs.box), self.geometry.foot(obs.box)
            t = self.tracks.get(obs.track_id)
            is_new = t is None
            if is_new:
                t = Tracklet(self.camera, obs.track_id, frame.ts, frame.ts, anchor, foot, obs.box)
                self.tracks[obs.track_id] = t
                ev.new.append(t)
            t.last_seen, t.anchor, t.foot, t.box = frame.ts, anchor, foot, obs.box
            t.history.append((frame.ts, self.geometry.middle(obs.box)))
            t.height = obs.box[3] / self.geometry.height
            areas = frozenset(a for a, pts in self.areas if point_in_polygon(anchor[0], anchor[1], pts))
            for a in sorted(areas - t.areas):
                t.entered[a] = frame.ts
                ev.entered.append((t, a))
            for a in sorted(t.areas - areas):
                t.entered.pop(a, None)
                ev.left.append((t, a))
            t.areas = areas
            seat_pts = self.geometry.seat_points(obs.box)
            t.seat = next((a for a, pts in self.seats if any(point_in_polygon(x, y, pts) for x, y in seat_pts)),
                          None)
            self._latch(t, frame.ts, policy)
            self._door(t, policy, ev, is_new)
        return ev

    def _latch(self, t: Tracklet, ts: float, policy: dict) -> None:
        """A body in a seat that may sit there (seat_ok) for seat_dwell_s --
        out of it no longer than seat_jitter_s -- is a customer for the rest of
        its track: standing up later doesn't turn it into staff. A body whose
        vote says staff outside every seat for staff_outside_confirm_s on end
        (a seat frame or a non-staff vote restarts the count) is ruled out
        of the seats for good (seat_ok)."""
        if t.customer:
            return
        if t.seat is None:
            if _vote(t, policy) is not True:
                t.staff_outside_since = None
                return
            if t.staff_outside_since is None:
                t.staff_outside_since = ts
            if ts - t.staff_outside_since >= policy["staff_outside_confirm_s"]:
                t.staff_outside = True
            return
        t.staff_outside_since = None
        if not seat_ok(t, policy):
            t.seat_since = None
            return
        if t.seat_since is None or ts - t.seat_last > policy["seat_jitter_s"]:
            t.seat_since = ts
        t.seat_last = ts
        if ts - t.seat_since >= policy["seat_dwell_s"]:
            t.customer = True

    def _door(self, t: Tracklet, policy: dict, ev: TrackEvents, is_new: bool) -> None:
        for door in self.doors:
            if not door.within_extent(t.foot):
                continue
            s = door.signed(t.foot)
            if is_new:
                t.appeared_at_door = 0.0 < s <= policy["door_band"]
            side = 1 if s > policy["line_band"] else -1 if s < -policy["line_band"] else 0
            if side == 0:
                continue
            if t.door_side == -side:
                (ev.crossed_in if side == 1 else ev.crossed_out).append(t)
            t.door_side = side

    def add_appearance(self, app: Appearance) -> Tracklet | None:
        t = self.tracks.get(app.track_id)
        if t is None:
            return None
        if app.staff_p is not None:
            t.staff_samples.append(app.staff_p)
        if app.reid is not None:
            t.reid, t.reid_n = fold(t.reid, t.reid_n, app.reid)
            if t.owned:
                t.reid_owned, t.reid_owned_n = fold(t.reid_owned, t.reid_owned_n, app.reid)
        return t

    def expire(self, gone_s: float) -> list[Tracklet]:
        """Tracks unseen for gone_s on THIS camera's clock: a camera whose
        frames stall ends nobody."""
        if self.latest is None:
            return []
        gone = [t for t in self.tracks.values() if self.latest - t.last_seen > gone_s]
        for t in gone:
            del self.tracks[t.track_id]
        return gone
