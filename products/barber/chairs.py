"""Chair occupancy, decided by each chair's seat in the camera that owns it.

A body that may sit (tracklets.seat_ok: the vote may still be undecided),
still, inside the seat, is sitting; after seat_dwell_s the chair reports a start,
backdated to when the stillness began. This start is reported on every
observe_seats call while the chair is free and the body keeps sitting;
callers must claim the chair with begin() or ignore the start. Out of the
seat (or moving) for longer than seat_jitter_s resets the sit. A running
haircut stays alive while its customer shows up at the chair: still in the
seat in the owner camera, or anywhere inside the chair's zone in any camera
(note_presence -- the paired view fills in while a barber hides him). It
ends after chair_gap_s with no such evidence, dated to the last evidence.
An unbound body sitting still in that seat is a claim (claims): the caller
may bind it to the customer, which makes it evidence. Each chair remembers
its latest ended haircut (recent_end) for the engine's chair resume."""
from __future__ import annotations

from dataclasses import dataclass, field

from tracklets import Tracklet, seat_ok


@dataclass
class ChairState:
    area: str
    person: str | None = None          # customer of the running haircut
    session_id: str | None = None
    last_evidence: float | None = None
    last_end: ChairEnd | None = None  # the latest haircut to end here (chair resume)
    sitting: dict = field(default_factory=dict)        # tracklet key -> sitting since
    last_in_seat: dict = field(default_factory=dict)   # tracklet key -> last seen still in the seat


@dataclass(frozen=True)
class ChairStart:
    chair: str
    tracklet: Tracklet
    since: float


@dataclass(frozen=True)
class ChairEnd:
    chair: str
    person: str
    session_id: str
    at: float


class ChairBook:
    def __init__(self, areas: list[str], owner: dict[str, str] | None = None):
        self.chairs: dict[str, ChairState] = {}
        self.set_chairs(areas, owner)

    def set_chairs(self, areas: list[str], owner: dict[str, str] | None = None) -> None:
        """owner: chair area -> the camera whose seat decides it (zones.owner_cameras);
        a chair not in it is decided by whichever camera draws its seat."""
        self.owner = dict(owner or {})
        for a in areas:
            self.chairs.setdefault(a, ChairState(a))
        for a in [a for a, c in self.chairs.items() if a not in areas and c.person is None]:
            del self.chairs[a]

    def observe_seats(self, camera: str, tracklets: list[Tracklet], ts: float, policy: dict) -> list[ChairStart]:
        """tracklets: every live tracklet of `camera`; only those seen at `ts` count.
        Reports a start on every call while the chair is free and the body keeps sitting,
        until the caller claims the chair with begin()."""
        live = {t.key: t for t in tracklets}
        for t in tracklets:
            c = self.chairs.get(t.seat) if t.seat else None
            if c is None or t.last_seen != ts or self.owner.get(c.area, t.camera) != t.camera:
                continue                # 2026-09-26 (Task 5a): a non-owner camera's seat starts nothing
            if not seat_ok(t, policy):
                c.sitting.pop(t.key, None)
                c.last_in_seat.pop(t.key, None)
                continue
            if not t.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"]):
                continue
            c.last_in_seat[t.key] = ts
            c.sitting.setdefault(t.key, ts - policy["still_window_s"])
            if c.person is not None and t.person == c.person:
                c.last_evidence = ts
        starts = []
        for c in self.chairs.values():
            stale = [k for k, last in c.last_in_seat.items()
                     if k[0] == camera and (k not in live or ts - last > policy["seat_jitter_s"])]
            for k in stale:
                c.last_in_seat.pop(k, None)
                c.sitting.pop(k, None)
            if c.person is not None:
                continue
            ready = sorted((since, k) for k, since in c.sitting.items()
                           if k[0] == camera and ts - since >= policy["seat_dwell_s"])
            if ready:
                since, k = ready[0]
                starts.append(ChairStart(c.area, live[k], since))
        return starts

    def claims(self, camera: str, tracklets: list[Tracklet], ts: float,
               policy: dict) -> list[tuple[Tracklet, str, str]]:
        """(tracklet, chair, customer) candidates: unbound bodies of `camera` seen
        at `ts` that may sit (seat_ok), sitting still in the seat -- in the camera
        that owns the chair -- of a chair whose
        haircut runs -- the tracker often gives a seated customer a new track
        (Task 16 tune). The caller decides, binds, and notes the presence."""
        out = []
        for t in tracklets:
            c = self.chairs.get(t.seat) if t.seat else None
            if c is None or c.person is None or t.camera != camera or t.last_seen != ts:
                continue
            if self.owner.get(c.area, t.camera) != t.camera:
                continue                # 2026-09-26 (final B review I2): as observe_seats, owner only
            if t.person is not None or t.retired or not seat_ok(t, policy):
                continue
            if not t.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"]) \
                    and not (int(policy.get("seat_claim_mode", 0)) >= 1 and t.seat_since is not None
                             and ts - t.seat_since >= policy["still_window_s"]):
                continue                # mode 1: in the seat that long is enough (6.6)
            out.append((t, c.area, c.person))
        return out

    def note_presence(self, person_id: str, areas, ts: float) -> None:
        for a in areas:
            c = self.chairs.get(a)
            if c is not None and c.person == person_id:
                c.last_evidence = ts

    def begin(self, chair: str, person_id: str, session_id: str, ts: float) -> None:
        c = self.chairs[chair]
        c.person, c.session_id, c.last_evidence = person_id, session_id, ts

    def expire(self, ts: float, policy: dict) -> list[ChairEnd]:
        ends = []
        for c in self.chairs.values():
            if c.person is not None and ts - c.last_evidence > policy["chair_gap_s"]:
                ends.append(self.release(c.area))
        return ends

    def release(self, chair: str) -> ChairEnd | None:
        c = self.chairs.get(chair)
        if c is None or c.person is None:
            return None
        end = ChairEnd(c.area, c.person, c.session_id, c.last_evidence)
        c.person = c.session_id = c.last_evidence = None
        c.last_end = end
        return end

    def recent_end(self, chair: str, ts: float, within_s: float) -> ChairEnd | None:
        """The haircut that ended in `chair` no more than within_s before ts, if any."""
        c = self.chairs.get(chair)
        end = c.last_end if c is not None else None
        return end if end is not None and ts - end.at <= within_s else None

    def of_person(self, person_id: str) -> str | None:
        return next((c.area for c in self.chairs.values() if c.person == person_id), None)
