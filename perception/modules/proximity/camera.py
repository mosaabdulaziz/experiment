"""The proximity rule for one camera, frame by frame, with no engine in sight (RESEARCH 5.4-5.5, 5.9).

The module turns each analysed frame's objects into Persons and Vehicles; `ProximityCamera.step` decides which pairs
are close, which of them alert, what the overlay draws, and which pairs need a depth measurement. A pair is close when,
in a zone with proximity on: the person's foot (recovered when hidden) and the vehicle's reference point are inside;
the person is not riding the vehicle; the vehicle is moving (its reference point, and its box: VehicleMotion); and
the gap from the foot to the lowest vehicle_band of the vehicle's mask is within max_gap person heights. It alerts once close for persist_s and confirmed (by the image, the depth check, or
-- hidden feet with nothing to confirm them -- flagged "none"), when it opens a new episode or makes one more severe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from collections import deque

import numpy as np

from .episodes import Episode, EpisodeBook, PairLatch
from .rules import ZoneRule
from .scale import PersonScale, is_clear_standing
from .shapes import PersonShape, VehicleShape, gap_to_outline, is_rider, mostly_inside
from .tracks import Foot, FootAnchors, Positions, VehicleMotion, co_motion, resolve_foot

FORGET_PAIRS_S = 5.0
REWIND_S = 1.0          # time going back by more than this: a file looped or a source restarted its clock
DEPTH_WAIT_S = 0.5      # a pair never measured waits persist_s plus this from its first request, then goes without
ASKED_KEEP_S = 30.0     # first-request times are forgotten after this
RIDER_SHARE = 0.5       # on a vehicle by its geometry and moving along with it (this share of its travel): its rider
RIDER_KEEP = 0.25       # a rider keeps that while they travel at least this much of its distance (turning or not) ...
RIDER_HOLD_S = 5.0      # ... for this long after it was last confirmed
CAB_TRAVEL = 0.8        # a person beside a vehicle's box, feet hidden, travelling this much of its distance ...
CAB_SHARE = 0.6         # ... this much along its direction, is in its cab (the detector boxed only part of it)
SMOOTH_FRAME_S = 0.3    # a pair's history older than (its length) * this is stale: the medians start afresh
CAB_MARGIN = 0.5        # person heights: how close to the vehicle's box "beside" is
SAMPLE_EVERY_S = 1.0    # one scale-map sample per person per second: someone who lingers counts like a passer-by
NEW_TRACK_S = 2.0       # a person track this new, inside a moving vehicle (on it, or feet hidden), is its occupant
BOARDING_S = 1.5        # just got on a moving vehicle (by the geometry, box mostly inside), keeping up: its occupant
BOARDING_GAP_S = 0.5    # ... for this long; out of view longer than this, they get on afresh when back ...
BOARDING_DRIFT = 0.5    # ... while their head stays within this many person heights of where it got on


@dataclass
class Person:
    id: int
    label: str
    conf: float
    box: tuple                  # (left, top, width, height) mux px
    shape: PersonShape


@dataclass
class Vehicle:
    id: int
    label: str
    box: tuple
    shape: VehicleShape


@dataclass
class Line:
    """A person and a vehicle in a proximity zone, as the overlay draws them: the foot to the nearest point the gap is
    measured to (a moving vehicle: the lowest vehicle_band of its mask; a parked or not yet known one: its lower
    outline, as no gap of theirs is ever judged). Every such pair is measured (riders and drivers aside), close or not, for the preview's line
    range and the alert clips; only the close ones count for the rule."""
    foot: tuple
    outline_point: tuple
    gap: float                  # person heights
    latched: bool               # close for persist_s already
    person_id: int = 0
    person_box: tuple = ()      # (left, top, width, height) mux px
    vehicle_id: int = 0
    vehicle_box: tuple = ()
    vehicle_label: str = ""
    close: bool = True          # within the zone's max gap of a moving vehicle
    vehicle_state: str = "moving"   # "moving" | "parked" | "unknown" (not seen long enough to tell)
    foot_hidden: bool = False   # the foot is the estimated one: the person's feet are not in view
    seen_foot: tuple = ()       # the lowest point of the person's mask as seen (where the drawn leg starts)
    depth: Optional[tuple] = None   # in a depth-check zone, a close pair's (state, gap): "pending", "same", "apart"
                                    # (gap measured) or "timeout" (no answer in time; gap None)

    @property
    def style(self) -> str:
        """"latched", "close", or "far": beyond the max gap, or the vehicle is not moving."""
        return "latched" if self.latched else "close" if self.close else "far"


_LINE_RANK = {"far": 0, "close": 1, "latched": 2}
_STATE_RANK = {"parked": 0, "unknown": 1, "moving": 2}


def _strongest(lines: list) -> list:
    """One line per person and vehicle (zones may overlap): the most severe, then the shortest."""
    best: dict = {}
    for ln in lines:
        key = (ln.person_id, ln.vehicle_id)
        cur = best.get(key)
        if cur is None or (_LINE_RANK[ln.style], -ln.gap) > (_LINE_RANK[cur.style], -cur.gap):
            best[key] = ln
    return list(best.values())


@dataclass
class DepthRequest:
    """A pair whose depth check is due: what the depth worker measures it from (mux px)."""
    key: tuple                  # (zone id, person id, vehicle id)
    person_blob: object         # shapes.Blob or None
    person_box: tuple
    vehicle_blob: object
    vehicle_box: tuple


@dataclass
class Alert:
    zone: ZoneRule
    episode: Episode
    person: Person
    vehicle: Vehicle
    foot: Foot
    gap: float                  # person heights: the smallest while the pair was close
    held_s: float               # how long the pair had been close
    confirmation: str           # "image" | "depth" | "none" (hidden feet, nothing to confirm them)
    severity: str               # "critical" | "warning" | "info"
    escalated: bool = False     # the episode had alerted already, less severely
    depth_gap: Optional[float] = None   # the depth check's latest reading for the pair (a depth-check zone)


@dataclass
class Suppressed:
    """A pair close for persist_s that the depth check says is apart: no alert, one quiet record per episode of such
    pairs (Ticket 6), so a zone's depth cut-off can be checked against what it held back."""
    zone: ZoneRule
    episode: Episode
    person: Person
    vehicle: Vehicle
    foot: Foot
    gap: float                  # person heights: the smallest while the pair was close
    held_s: float
    severity: str               # what it would have alerted at, had the depth check put it at the same depth
    depth_gap: float            # the reading that held it back (over zone.depth_max_gap)


@dataclass
class Step:
    alerts: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    depth_requests: list = field(default_factory=list)
    parked: set = field(default_factory=set)        # vehicle ids found stationary inside a proximity zone
    states: dict = field(default_factory=dict)      # vehicle id -> "moving" | "parked" | "unknown", in a proximity zone
    closed: list = field(default_factory=list)      # episodes that ended
    suppressed: list = field(default_factory=list)  # quiet records: pairs the depth check held back (Suppressed)
    refit: bool = False                             # the scale map was refitted (worth saving)
    active: bool = False                            # a zone holds a vehicle that is moving or not yet known parked


def _beside(a: tuple, b: tuple, margin: float) -> bool:
    """Box `a` touches box `b` grown by `margin` on every side ((left, top, width, height))."""
    return (a[0] < b[0] + b[2] + margin and b[0] - margin < a[0] + a[2]
            and a[1] < b[1] + b[3] + margin and b[1] - margin < a[1] + a[3])


def severity(confirmation: str, gap: float, max_gap: float) -> str:
    """Critical under half the max gap, a warning otherwise, info when unconfirmed (RESEARCH 5.6, 5.9)."""
    if confirmation == "none":
        return "info"
    return "critical" if gap < max_gap / 2 else "warning"


class ProximityCamera:
    def __init__(self, frame_size: tuple, scale: PersonScale,
                 depth_refresh_s: float = 0.5, gap_smoothing_frames: int = 1, line_steady_frames: int = 1,
                 vehicle_band: float = 0.0, box_motion: float = 0.0, parked_hold_s: float = 0.0) -> None:
        self.frame_w, self.frame_h = frame_size
        self.scale = scale
        self.depth_refresh_s = depth_refresh_s
        self.gap_smoothing_frames = int(gap_smoothing_frames)
        self.line_steady_frames = int(line_steady_frames)
        self.vehicle_band = float(vehicle_band)
        self.box_motion = float(box_motion)
        self.parked_hold_s = float(parked_hold_s)
        self._last_t: Optional[float] = None
        self._rewind()

    def _rewind(self) -> None:
        """Fresh per-track memory (the scale map is the camera's, so it stays)."""
        self.motion = VehicleMotion(frame_size=(self.frame_w, self.frame_h), box_motion=self.box_motion,
                                    parked_hold_s=self.parked_hold_s)
        self.heads = Positions()
        self.anchors = FootAnchors()
        self.latch = PairLatch()
        self.episodes = EpisodeBook()
        self.quiet = EpisodeBook()      # episodes of pairs the depth check held back: one quiet record each
        self._asked: dict = {}      # pair key -> when its first depth measurement was asked for
        self._riders: dict = {}     # (person id, vehicle id) -> when the person was last confirmed on that vehicle
        self._aboard: dict = {}     # (person id, vehicle id) -> (since, last, head from the box centre then): on it
        self._sampled: dict = {}    # person id -> when they last taught the scale map
        self._born: dict = {}       # person id -> when the track was first seen
        self._gaps: dict = {}       # pair key -> its last measurements (t, gap, line end - foot) near a moving vehicle

    def step(self, t: float, persons: list, vehicles: list, rules: list,
             depth: Optional[Callable[[tuple], Optional[tuple]]] = None) -> Step:
        """One analysed frame at time `t` (s). `depth(key)` is the pair's latest depth measurement, (the time it was
        measured, the relative depth gap), or None; `depth=None` means this camera has no depth model."""
        lost: list = []
        if self._last_t is not None and t < self._last_t - REWIND_S:
            lost = self.episodes.open           # the clock went back: these episodes are over
            self._rewind()
            self._last_t = None
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)    # grace and holds count from frame to frame
        self._last_t = t
        out = Step()
        for p in persons:
            if (is_clear_standing(p.conf, p.box, self.frame_w, self.frame_h)
                    and t - self._sampled.get(p.id, t - SAMPLE_EVERY_S) >= SAMPLE_EVERY_S):
                self._sampled[p.id] = t
                out.refit = self.scale.add(p.shape.foot[1], p.box[3], p.id) or out.refit
            self.heads.add(p.id, t, p.shape.head)
            self._born.setdefault(p.id, t)
        feet = {p.id: resolve_foot(p.id, t, p.shape, self.scale, self.anchors) for p in persons}
        people_px = float(np.median([p.box[3] for p in persons])) if persons else None
        units = {v.id: self._unit(v, people_px) for v in vehicles}
        present = {v.id for v in vehicles}
        for v in vehicles:
            self.motion.observe(v.id, t, v.shape.ref, v.shape.area, link_px=units[v.id], present=present, box=v.box)
        for rule in rules:
            self._zone(t, dt, rule, persons, feet, vehicles, units, depth, out)
        out.lines = _strongest(out.lines)
        ended = self.episodes.expire(t)
        for e in ended:                         # a pair still close later starts its own span: a new episode is its own
            self.latch.drop(e.zone_id, e.person_ids, e.vehicle_ids)
        out.closed = lost + ended
        self.quiet.expire(t)
        self.latch.forget(t, FORGET_PAIRS_S)
        for key in [k for k, h in self._gaps.items() if t - h[-1][0] > FORGET_PAIRS_S]:
            del self._gaps[key]
        self.motion.forget(t)
        self.heads.forget(t)
        self.anchors.forget(t)
        self._asked = {k: first for k, first in self._asked.items() if t - first <= ASKED_KEEP_S}
        self._riders = {k: last for k, last in self._riders.items() if t - last <= RIDER_HOLD_S}
        self._aboard = {k: on for k, on in self._aboard.items() if t - on[1] <= BOARDING_GAP_S}
        self._sampled = {k: last for k, last in self._sampled.items() if t - last < SAMPLE_EVERY_S}
        self._born = {k: born for k, born in self._born.items() if self.heads.points(k)}
        return out

    def _ph(self, foot_y: float, person: Person) -> float:
        """A person height in mux px at this foot row: the scale map's, or the person's own box height until the map
        is ready."""
        return self.scale.height(foot_y) if self.scale.ready else max(1.0, float(person.box[3]))

    def _unit(self, v: Vehicle, people_px: Optional[float]) -> float:
        """A person height at the vehicle, in mux px: the scale map's at its reference row; until the map is ready the
        people in view (median box height), else the vehicle's own box height."""
        if self.scale.ready:
            return self.scale.height(v.shape.ref[1])
        return max(1.0, people_px if people_px else float(v.box[3]))

    def _inside(self, rule: ZoneRule, point: tuple) -> bool:
        """Zone membership of a point clamped into the frame: feet recovered below the bottom edge are still where the
        person is in the image."""
        x = min(max(point[0], 0.0), self.frame_w - 1.0)
        y = min(max(point[1], 0.0), self.frame_h - 1.0)
        return rule.contains(x, y)

    def _moving(self, t: float, rule: ZoneRule, vehicles: list, units: dict, out: Step) -> tuple:
        """(the zone's moving vehicles, its parked ones, those not seen long enough to tell); parked ones also go to
        `out.parked`, every one's state to `out.states` (a vehicle in several zones: the most active), and any vehicle
        not known to be parked makes the step `active`."""
        moving, parked, unknown = [], [], []
        for v in vehicles:
            if v.label not in rule.vehicle_classes or not rule.contains(*v.shape.ref):
                continue
            state = self.motion.state(v.id, t, rule.stationary_window_s, rule.stationary_motion * units[v.id])
            out.active = out.active or state != "stationary"
            if state == "moving":
                moving.append(v)
            elif state == "stationary":
                parked.append(v)
                out.parked.add(v.id)
            else:
                unknown.append(v)
            name = {"moving": "moving", "stationary": "parked"}.get(state, "unknown")
            if _STATE_RANK[name] >= _STATE_RANK.get(out.states.get(v.id), -1):
                out.states[v.id] = name
        return moving, parked, unknown

    def _riding(self, t: float, rule: ZoneRule, p: Person, foot: Foot, v: Vehicle, ph: float) -> bool:
        """The vehicle's driver or a rider. Status per person and vehicle, set only by positive evidence: on it by the
        geometry (RESEARCH 5.5) and moving along with it; on it by the geometry while it was parked (`_zone`); or feet
        hidden, right beside its box and travelling with it -- a cab the detector left out of the vehicle's box, or an
        open frame. It is kept while the person keeps up with the vehicle in any direction (a turning vehicle's
        reference point and cab move differently) for up to RIDER_HOLD_S since last confirmed, and lost when they stand
        still while it moves. A pedestrian walking toward or across a vehicle never gets it; nor does someone walking
        with it with their feet in view. With no history yet the geometry decides, for that frame only. Failing all
        that, a track younger than NEW_TRACK_S is too new for co-motion to tell (a driver coming into view, or first
        seen mid-turn): while its box is mostly inside the vehicle's and it is on it by the geometry or has its feet
        hidden, it is the vehicle's occupant -- for that frame, so a pedestrian is judged on their motion once the
        track is older. Likewise someone who has just got on, by the geometry with the box mostly inside, for under
        BOARDING_S, who has stayed put on it since (BOARDING_DRIFT) and keeps up with it (RIDER_KEEP, any direction):
        the co-motion window still holds what they did before -- the walk up to it as it set off, or a dropout that
        leaves their history shorter than the vehicle's. Someone standing still while it passes, or walking past
        behind it, never qualifies."""
        key = (p.id, v.id)
        motion = co_motion(self.heads.points(p.id), self.motion.points(v.id), t, rule.stationary_window_s)
        geometric = is_rider(p.box, p.shape.foot, v.box, v.shape, ph, foot.hidden)
        inside = mostly_inside(p.box, v.box)
        new = (t - self._born[p.id] < NEW_TRACK_S and (geometric or foot.hidden) and inside)
        boarding = self._boarding(t, key, bool(geometric) and inside, p, v, ph)
        if motion is None:
            return geometric or new or key in self._riders
        travel, share = motion
        if (geometric and share >= RIDER_SHARE) or (
                foot.hidden and travel >= CAB_TRAVEL and share >= CAB_SHARE and _beside(p.box, v.box, CAB_MARGIN * ph)):
            self._riders[key] = t
            return True
        if key in self._riders and travel >= RIDER_KEEP:
            return True
        self._riders.pop(key, None)
        return new or (boarding and travel >= RIDER_KEEP)

    def _boarding(self, t: float, key: tuple, on_it: bool, p: Person, v: Vehicle, ph: float) -> bool:
        """Whether the person has been on the vehicle (by the geometry, box mostly inside) for under BOARDING_S, their
        head within BOARDING_DRIFT person heights of where it was on its box when they got on; notes this frame. A pair
        not seen on it for over BOARDING_GAP_S is forgotten (`step`): back on after that, they got on afresh."""
        if not on_it:
            self._aboard.pop(key, None)
            return False
        at = (p.shape.head[0] - (v.box[0] + v.box[2] / 2), p.shape.head[1] - (v.box[1] + v.box[3] / 2))
        since, _last, start = self._aboard.get(key, (t, t, at))
        self._aboard[key] = (since, t, start)
        return t - since < BOARDING_S and math.hypot(at[0] - start[0], at[1] - start[1]) <= BOARDING_DRIFT * ph

    def _zone(self, t, dt, rule, persons, feet, vehicles, units, depth, out) -> None:
        moving, parked, unknown = self._moving(t, rule, vehicles, units, out)
        for p in persons:                      # someone sitting in a parked vehicle is its driver when it drives off
            for v in parked:
                if is_rider(p.box, p.shape.foot, v.box, v.shape, self._ph(feet[p.id].point[1], p), feet[p.id].hidden):
                    self._riders[(p.id, v.id)] = t
        grace = rule.grace_s + dt
        for p in persons:
            foot = feet[p.id]
            if not foot.trusted:                     # feet hidden, under half of them in view: where they stand is a guess
                continue
            if p.label not in rule.person_classes or not self._inside(rule, foot.point):
                continue
            ph = self._ph(foot.point[1], p)
            # vehicles that are not moving never count: measured for the overlay only (no rider history is kept here)
            for v, state in [(v, "parked") for v in parked] + [(v, "unknown") for v in unknown]:
                if (p.id, v.id) in self._riders or is_rider(p.box, p.shape.foot, v.box, v.shape, ph, foot.hidden):
                    continue
                dist, near = gap_to_outline(foot.point, v.shape.outline)
                out.lines.append(Line(foot.point, near, dist / ph, False, p.id, p.box, v.id, v.box, v.label,
                                      close=False, vehicle_state=state, foot_hidden=foot.hidden,
                                      seen_foot=p.shape.foot))
            for v in moving:
                if self._riding(t, rule, p, foot, v, ph):
                    continue
                key = (rule.zone_id, p.id, v.id)
                gap, near = self._measured(t, key, foot.point, v, ph)
                if gap > rule.max_gap:
                    out.lines.append(Line(foot.point, near, gap, False, p.id, p.box, v.id, v.box, v.label,
                                          close=False, foot_hidden=foot.hidden, seen_foot=p.shape.foot))
                    continue
                latched = self.latch.hot(key, t, gap, rule.persist_s, grace, where=(foot.point, v.shape.ref, ph))
                line = Line(foot.point, near, gap, latched is not None, p.id, p.box, v.id, v.box, v.label,
                            foot_hidden=foot.hidden, seen_foot=p.shape.foot)
                out.lines.append(line)
                confirmation = self._confirm(t, rule, key, p, foot, v, depth, out, line)
                if latched is not None and confirmation is None and line.depth is not None and line.depth[0] == "apart":
                    self._held_back(t, rule, key, p, foot, v, latched, ph, grace, line.depth[1], out)
                if latched is None or confirmation is None:
                    continue
                held, min_gap = latched
                level = severity(confirmation, min_gap, rule.max_gap)
                episode, what = self.episodes.add(rule.zone_id, t, t - held, p.id, v.id, foot.point, v.shape.ref,
                                                  min_gap, ph, grace + rule.cooldown_s, level,
                                                  min_gap_t=self.latch.min_time(key))
                if what != "joined":
                    out.alerts.append(Alert(rule, episode, p, v, foot, min_gap, held, confirmation, level,
                                            escalated=what == "escalated",
                                            depth_gap=line.depth[1] if line.depth is not None else None))

    def _held_back(self, t, rule, key, p, foot, v, latched, ph, grace, depth_gap, out) -> None:
        """A latched pair the depth check says is apart: a quiet record when it opens an episode of its own -- not when
        an episode that alerted covers it (its reading flipped near the cut-off: the incident is already there)."""
        if self.episodes.covering(rule.zone_id, t, foot.point, v.shape.ref, ph) is not None:
            return
        held, min_gap = latched
        level = severity("depth", min_gap, rule.max_gap)
        episode, what = self.quiet.add(rule.zone_id, t, t - held, p.id, v.id, foot.point, v.shape.ref, min_gap, ph,
                                       grace + rule.cooldown_s, level, min_gap_t=self.latch.min_time(key))
        if what == "opened":
            out.suppressed.append(Suppressed(rule, episode, p, v, foot, min_gap, held, level, depth_gap))

    def _measured(self, t: float, key, foot: tuple, v: Vehicle, ph: float) -> tuple:
        """The pair's gap (person heights) and where its drawn line ends on the vehicle.
        - gap: as measured, or with gap_smoothing_frames > 1 the median of the pair's last measurements (Ticket 3
          found no gain from it on held-out footage, so it is off by default);
        - line end: with line_steady_frames > 1, where the pair's recent ends lay around the foot, put back on the
          vehicle's outline: the line no longer jumps along the vehicle's edge from frame to frame (for the eye
          only: the rule never reads it)."""
        dist, near = gap_to_outline(foot, self._target(v))
        n = max(self.gap_smoothing_frames, self.line_steady_frames)
        if n <= 1:
            return dist / ph, near
        hist = self._gaps.get(key)
        if hist is None or t - hist[-1][0] > n * SMOOTH_FRAME_S:
            hist = self._gaps[key] = deque(maxlen=n)
        hist.append((t, dist / ph, near[0] - foot[0], near[1] - foot[1]))
        recent = list(hist)
        gap = (float(np.median([h[1] for h in recent[-self.gap_smoothing_frames:]]))
               if self.gap_smoothing_frames > 1 else dist / ph)
        if self.line_steady_frames > 1:
            ends = recent[-self.line_steady_frames:]
            target = (foot[0] + float(np.median([h[2] for h in ends])), foot[1] + float(np.median([h[3] for h in ends])))
            near = gap_to_outline(target, self._target(v))[1]
        return gap, near

    def _target(self, v: Vehicle) -> np.ndarray:
        """What a person's gap to a moving vehicle is measured to: its lower outline, or with vehicle_band > 0 every
        mask cell in the lowest vehicle_band of the mask's rows (wheels, side bumpers, forks near the ground, not the
        body above; Ticket 3: beat the lower outline and the whole mask on held-out footage). No mask: the outline."""
        blob = v.shape.blob
        if self.vehicle_band <= 0 or blob is None:
            return v.shape.outline
        rows, cols = np.nonzero(blob.grid)
        if not len(rows):
            return v.shape.outline
        keep = rows >= rows.max() - self.vehicle_band * (rows.max() - rows.min())
        xs, ys = blob.xy(rows[keep], cols[keep])
        return np.stack([xs, ys], axis=1)

    def _confirm(self, t, rule, key, p, foot, v, depth, out, line) -> Optional[str]:
        """How this close pair is confirmed, or None while it is not: depth pending or saying "not at the same depth",
        or hidden feet in a zone that suppresses them; the depth check's state goes on the pair's `line`. The last depth verdict stands while a fresher one is asked for;
        a pair the depth check has never answered for persist_s + DEPTH_WAIT_S since its first request goes without it
        (RESEARCH 5.9: with no second signal, the zone's own choice applies)."""
        if rule.depth_check and depth is not None:
            got = depth(key)
            if got is not None and got[0] > t:
                got = None                          # measured before the clock went back
            if (got is None or t - got[0] >= self.depth_refresh_s) and p.shape.blob is not None:
                # never from a box: its lowest band holds whatever hides the feet and would read "same depth"
                out.depth_requests.append(DepthRequest(key, p.shape.blob, p.box, v.shape.blob, v.box))
            if got is not None:
                same = got[1] <= rule.depth_max_gap
                line.depth = ("same" if same else "apart", got[1])
                return "depth" if same else None
            if t - self._asked.setdefault(key, t) < rule.persist_s + DEPTH_WAIT_S:
                line.depth = ("pending", None)
                return None
            line.depth = ("timeout", None)
        if foot.hidden:
            return None if rule.hidden_feet == "suppress" else "none"
        return "image"
