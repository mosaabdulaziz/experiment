"""Persistence and duplicate suppression (RESEARCH 5.5): a pair alerts once it has stayed close for `persist_s`, and
there is one alert per vehicle episode, however often the tracker switches ids -- both keyed on position, since ids
still break (RESEARCH 4.4)."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

PERSON_RADIUS = 1.5     # person heights: a pair whose person stands this close to an earlier pair's person ...
VEHICLE_RADIUS = 2.0    # ... (or, for episodes, whose vehicle is this close to its vehicle) continues it
SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _near(a: tuple, b: tuple, limit: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= limit


class PairLatch:
    """How long each person-vehicle pair has been close. A pair is latched once it has been close for `persist_s`,
    gaps of up to `grace_s` (detector flicker) not breaking it. Keeps the smallest gap of the current span. A pair that
    is new because the tracker gave its person or vehicle a new id continues the span of the pair it was: one still
    within its grace, of the same zone, whose person stood within PERSON_RADIUS person heights and whose vehicle was
    within VEHICLE_RADIUS (`where` = (foot, vehicle reference point, person height px); keys are (zone, person, vehicle))."""

    def __init__(self) -> None:
        self._spans: dict = {}      # key -> [since, last, min_gap, foot, vref, when min_gap was seen]

    def hot(self, key, t: float, gap: float, persist_s: float, grace_s: float,
            where: Optional[tuple] = None) -> Optional[tuple]:
        """Record that the pair is close at `t`: (how long it has been close, the smallest gap since) once that is at
        least `persist_s`, else None."""
        foot, vref = (where[0], where[1]) if where else (None, None)
        span = self._spans.get(key)
        if span is None or t - span[1] > grace_s:
            span = (where and self._continued(key, t, grace_s, *where)) or [t, t, gap, foot, vref, t]
            self._spans[key] = span
        if gap < span[2]:
            span[5] = t
        span[1], span[2], span[3], span[4] = t, min(span[2], gap), foot, vref
        held = t - span[0]
        return (held, span[2]) if held >= persist_s else None

    def _continued(self, key, t, grace_s, foot, vref, ph) -> Optional[list]:
        for other, s in self._spans.items():
            if (other != key and other[0] == key[0] and 0 < t - s[1] <= grace_s and s[3] is not None
                    and _near(foot, s[3], PERSON_RADIUS * ph) and _near(vref, s[4], VEHICLE_RADIUS * ph)):
                return list(s)
        return None

    def min_time(self, key) -> Optional[float]:
        """When the pair's current span saw its smallest gap, or None for a pair with no span."""
        span = self._spans.get(key)
        return None if span is None else span[5]

    def drop(self, zone_id: str, person_ids, vehicle_ids) -> None:
        """Forget the spans of an episode's pairs (it ended): a pair still close starts again from its own frames."""
        for pid in person_ids:
            for vid in vehicle_ids:
                self._spans.pop((zone_id, pid, vid), None)

    def forget(self, t: float, older_than: float) -> None:
        for key in [k for k, s in self._spans.items() if t - s[1] > older_than]:
            del self._spans[key]


@dataclass
class Episode:
    zone_id: str
    start: float
    end: float                      # the last time a latched pair was seen
    hold_s: float                   # it stays open this long after `end` (grace + cooldown)
    foot: tuple                     # the last person position (mux px)
    vref: tuple                     # the last vehicle reference point (mux px)
    min_gap: float                  # person heights
    severity: str                   # the highest alerted so far
    person_ids: list = field(default_factory=list)
    vehicle_ids: list = field(default_factory=list)
    min_gap_t: float = 0.0          # when min_gap was seen (frame time)
    incident: str = field(default_factory=lambda: uuid.uuid4().hex[:12])   # the app's handle on this episode


class EpisodeBook:
    """The open episodes of one camera. A latched pair joins an episode of its zone still open (its last pair within
    `hold_s`) when its person stands within PERSON_RADIUS person heights of the episode's last person, or its vehicle is
    within VEHICLE_RADIUS of the episode's last vehicle -- by position, not by id, so an id switch does not raise a
    second alert. A pair that joins none opens a new episode, which alerts; a joining pair more severe than anything
    the episode alerted escalates it, which alerts once more (an unconfirmed episode that turns critical must not stay
    filed as info)."""

    def __init__(self) -> None:
        self.open: list = []

    def add(self, zone_id: str, t: float, start: float, pid: int, vid: int, foot: tuple, vref: tuple, gap: float,
            ph: float, hold_s: float, severity: str, min_gap_t: Optional[float] = None) -> tuple:
        """(the episode, "opened" | "escalated" | "joined"). min_gap_t: when the pair's gap was smallest
        (PairLatch.min_time)."""
        e = self.covering(zone_id, t, foot, vref, ph)
        if e is not None:
            e.end, e.foot, e.vref, e.hold_s = t, foot, vref, hold_s
            if gap < e.min_gap:
                e.min_gap, e.min_gap_t = gap, t if min_gap_t is None else min_gap_t
            if pid not in e.person_ids:
                e.person_ids.append(pid)
            if vid not in e.vehicle_ids:
                e.vehicle_ids.append(vid)
            if SEVERITY_RANK[severity] > SEVERITY_RANK[e.severity]:
                e.severity = severity
                return e, "escalated"
            return e, "joined"
        e = Episode(zone_id, start, t, hold_s, foot, vref, gap, severity, [pid], [vid],
                    min_gap_t=t if min_gap_t is None else min_gap_t)
        self.open.append(e)
        return e, "opened"

    def covering(self, zone_id: str, t: float, foot: tuple, vref: tuple, ph: float):
        """The open episode a pair at (foot, vref) would join -- or None -- without adding it (Ticket 6: a pair the
        depth check holds back is not recorded quietly when an episode that alerted covers it)."""
        for e in self.open:
            if e.zone_id == zone_id and t - e.end <= e.hold_s and (
                    _near(foot, e.foot, PERSON_RADIUS * ph) or _near(vref, e.vref, VEHICLE_RADIUS * ph)):
                return e
        return None

    def expire(self, t: float) -> list:
        """Close and return the episodes with no latched pair for longer than their hold."""
        closed = [e for e in self.open if t - e.end > e.hold_s]
        self.open = [e for e in self.open if t - e.end <= e.hold_s]
        return closed
