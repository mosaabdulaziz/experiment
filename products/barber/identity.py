"""Global person ids across the three cameras.

The engine creates a person from one tracklet (a door entry, a missed entry
in the waiting area, or a seated body nobody claimed). Other tracklets join
it by:
  * handoff -- entering a paired zone while one of the person's tracks in
    another camera is inside the same pair. Candidates whose entry lies
    within sync_window_s rank first (the user's "entering chair3 in both
    views at once"); a candidate already inside for longer still counts,
    which lets the side camera pick up a seated customer after its own track
    churned. Body re-ID vetoes a clear mismatch (reid_veto) and breaks ties;
    two candidates it can't tell apart (reid_margin) mean "wait".
  * re-attach -- a tracklet whose body embedding matches (reid_match_min) a
    person without a track in its camera, whose track there ended within
    reattach_s, or who is lost entirely and was last seen in (or in an area
    drawn in) this camera. A tracklet in a zone never re-attaches to someone
    seen right now in a different area, nor to someone lost in an area not
    next to its own (next_to: sofa <-> chair <-> wash, never chair <-> chair);
    a tracklet in no zone never re-attaches to someone seen right now in a
    chair or at the wash (it retries each frame, so it joins once it walks
    into their area).
A person holds at most one track per camera.

2026-09-27 (39-min testbed spec, section 4): the re-ID reference a track is
compared with follows policy reid_gallery (reference()): the blended mean,
the mean of the track's own camera, or the mean of owned-area samples. A
person with no reference there is no re-attach, and no veto or tie-break.

2026-09-28 (39-min testbed spec, 6.3): handoff_rule tightens the handoff
above. 1 = drop the "already inside for minutes" candidates -- a partner
qualifies only within sync_window_s of the new track's entry; 2 = rule 1,
plus in a chair area the new track must itself be still to take the
handoff."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tracklets import Tracklet, cosine, fold, merge


@dataclass(eq=False)
class Person:
    id: str
    tracks: dict = field(default_factory=dict)          # camera -> live Tracklet
    reid: tuple | None = None
    reid_n: int = 0
    last_seen: float = 0.0
    last_cameras: frozenset = frozenset()
    last_areas: frozenset = frozenset()
    gone_from: dict = field(default_factory=dict)       # camera -> when its track there ended
    lost_since: float | None = None                     # set while no track is left anywhere
    washing_seen: float | None = None                   # last time a track stood in a washing area
    reid_by_cam: dict = field(default_factory=dict)     # camera -> (mean, n): reid_gallery 1
    reid_owned: tuple | None = None                     # samples taken in an owned area: reid_gallery 2
    reid_owned_n: int = 0
    origin: tuple | None = None                         # (camera, track_id) of the track that created him
    origin_since: float | None = None                   # ...and its first_seen (a reused track id is another body)
    other_track: bool = False                           # any other track was ever bound to him

    def only_ever(self, t: Tracklet) -> bool:
        """Was t the track that created him, and his only track ever (staff_release_mode 1)?"""
        return not self.other_track and (t.key, t.first_seen) == (self.origin, self.origin_since)

    @property
    def areas(self) -> frozenset:
        """Where his tracks are evidence for him (Tracklet.evidence: identity_model)."""
        out: frozenset = frozenset()
        for t in self.tracks.values():
            out |= t.evidence
        return out


def reference(p: Person, camera: str, policy: dict) -> tuple | None:
    """The re-ID mean a track of `camera` is compared with (policy reid_gallery)."""
    gallery = int(policy.get("reid_gallery", 0))
    if gallery == 1:
        return p.reid_by_cam.get(camera, (None, 0))[0]
    if gallery == 2:
        return p.reid_owned
    return p.reid


@dataclass(frozen=True)
class Match:
    person: Person
    kind: str                        # "handoff" | "reattach"
    area: str | None = None
    other: Tracklet | None = None    # the partner track of a handoff
    offset_s: float | None = None    # this track's entry minus the partner's
    reid_sim: float | None = None


# Areas a lost customer may turn up in next, by type: the sofa, a chair and
# the wash lead to one another; the chair next door does not (review round 1).
NEXT_TO = {("waiting", "chair"), ("chair", "waiting"), ("chair", "washing"), ("washing", "chair")}
# 2026-09-26 (Task 5a): someone seen right now in an area of these types is
# not a body in no zone of another camera -- the paired zone would show them
# (the raw1 replay merged the L2 customer with a centre track in no zone).
PLACED = ("chair", "washing")


def next_to(a: str, b: str, area_type: dict) -> bool:
    return a == b or (area_type.get(a), area_type.get(b)) in NEXT_TO


def may_claim_seat(t: Tracklet, p: Person, chair: str, policy: dict, now: float | None = None) -> bool:
    """May t, an unbound body sitting still in `chair`'s seat, join p, the
    customer of the haircut running there (the tracker churned his track)?
    Only if p has no track in t's camera, every live track p has is inside
    that chair (none at all: lost), and body re-ID, when both sides have an
    embedding, doesn't veto it -- else the claim would bind the next customer,
    or anyone else in the seat, to the old visit (review #2). With `now` (the
    engine clock), a track unseen for track_gone_s -- a stalled camera's
    frozen one -- doesn't count (final B review C1b)."""
    live = [o for o in p.tracks.values() if now is None or now - o.last_seen <= policy["track_gone_s"]]
    if any(o.camera == t.camera for o in live) or any(chair not in o.areas for o in live):
        return False
    ref = reference(p, t.camera, policy)
    return t.reid is None or ref is None or cosine(t.reid, ref) >= policy["reid_veto"]


def seat_claim_displaces(t: Tracklet, p: Person, chair: str, policy: dict, now: float | None = None):
    """seat_claim_mode 2 (39-min testbed spec, 6.6): p's live track o in t's camera, which blocks
    may_claim_seat, gives way to t -- returned, for the caller to release -- when o is not in
    `chair`'s seat, both have an embedding, and t matches p's reference at least reid_veto and at
    least as well as o does; every other rule of may_claim_seat still holds. Else None."""
    if int(policy.get("seat_claim_mode", 0)) < 2:
        return None
    live = [o for o in p.tracks.values() if now is None or now - o.last_seen <= policy["track_gone_s"]]
    same = [o for o in live if o.camera == t.camera]
    if len(same) != 1 or any(chair not in o.areas for o in live if o.camera != t.camera):
        return None
    o, ref = same[0], reference(p, t.camera, policy)
    if o.seat == chair or t.reid is None or o.reid is None or ref is None:
        return None
    sim = cosine(t.reid, ref)
    return o if sim >= policy["reid_veto"] and sim >= cosine(o.reid, ref) else None


class IdentityBook:
    def __init__(self, new_id: Callable[[], str]):
        self._new_id = new_id
        self.persons: dict[str, Person] = {}

    def create(self, t: Tracklet, ts: float) -> Person:
        p = Person(self._new_id(), origin=t.key, origin_since=t.first_seen)
        self.persons[p.id] = p
        self.bind(t, p, ts)
        return p

    def bind(self, t: Tracklet, p: Person, ts: float) -> None:
        t.person = p.id
        t.retired = False
        if (t.key, t.first_seen) != (p.origin, p.origin_since):
            p.other_track = True
        p.tracks[t.camera] = t
        p.lost_since = None
        p.last_seen = max(p.last_seen, ts)
        p.reid, p.reid_n = merge(p.reid, p.reid_n, t.reid, t.reid_n)
        if t.reid is not None:
            p.reid_by_cam[t.camera] = merge(*p.reid_by_cam.get(t.camera, (None, 0)), t.reid, t.reid_n)
        p.reid_owned, p.reid_owned_n = merge(p.reid_owned, p.reid_owned_n, t.reid_owned, t.reid_owned_n)
        self.note(t)

    def note(self, t: Tracklet) -> None:
        """Remember where the person is: where to look if they're lost."""
        p = self.persons.get(t.person) if t.person else None
        if p is None:
            return
        p.last_seen = max(p.last_seen, t.last_seen)
        p.last_cameras = frozenset(p.tracks)
        if p.areas:                     # stepping out of every zone keeps the last one
            p.last_areas = p.areas

    def add_reid(self, p: Person, v, camera: str | None = None, owned: bool = False) -> None:
        """A sample from one of p's tracks: into the blended mean, its camera's, and
        (taken in an area its camera owns) the owned-only one."""
        p.reid, p.reid_n = fold(p.reid, p.reid_n, v)
        if camera is not None:
            p.reid_by_cam[camera] = fold(*p.reid_by_cam.get(camera, (None, 0)), v)
        if owned:
            p.reid_owned, p.reid_owned_n = fold(p.reid_owned, p.reid_owned_n, v)

    def release(self, t: Tracklet, ts: float) -> Person | None:
        """The track ended or turned out to be staff: take it off its person."""
        p = self.persons.get(t.person) if t.person else None
        t.person = None
        if p is None:
            return None
        if p.tracks.get(t.camera) is t:
            del p.tracks[t.camera]
            p.gone_from[t.camera] = ts
            if not p.tracks and p.lost_since is None:   # already lost (identity_model 2): keep when
                p.lost_since = ts
        return p

    def drop(self, person_id: str) -> None:
        """The visit closed: forget the person; its live tracks never join anyone again."""
        p = self.persons.pop(person_id, None)
        if p is None:
            return
        for t in p.tracks.values():
            t.person = None
            t.retired = True

    def find_handoff(self, t: Tracklet, area: str, partners: list[Tracklet], policy: dict,
                      still: bool | None = None, area_type: str | None = None) -> Match | None:
        rule = int(policy.get("handoff_rule", 0))
        if area_type == "chair" and rule == 4:
            # 2026-09-28: in that chair's seat for still_window_s, jitter allowed -- a customer being
            # worked on isn't still, and a still barber by the chair isn't in the seat that long
            if t.seat != area or t.seat_since is None or t.last_seen - t.seat_since < policy["still_window_s"]:
                return None
        elif rule >= 2 and area_type == "chair" and not still:
            return None                      # a walking new track doesn't take a chair handoff
        elif rule == 3 and area_type == "chair" and t.seat != area:
            return None                      # ...nor one standing by the chair: a barber (2026-09-28)
        cands = []
        for o in partners:
            if o.camera == t.camera or o.person is None or area not in o.areas or area not in t.entered:
                continue
            p = self.persons.get(o.person)
            if p is None or t.camera in p.tracks:
                continue
            offset = t.entered[area] - o.entered[area]
            if rule >= 1 and abs(offset) > policy["sync_window_s"]:
                continue                      # rule 1: no "already inside for minutes" partners
            ref = reference(p, t.camera, policy)
            sim = cosine(t.reid, ref) if t.reid is not None and ref is not None else None
            if sim is not None and sim < policy["reid_veto"]:
                continue
            cands.append((abs(offset) <= policy["sync_window_s"], sim, offset, o, p))
        if not cands:
            return None
        cands.sort(key=lambda c: (c[0], -1.0 if c[1] is None else c[1]), reverse=True)
        within, sim, offset, o, p = cands[0]
        if len(cands) > 1 and cands[1][0] == within:
            other_sim = cands[1][1]
            if sim is None or other_sim is None or sim - other_sim < policy["reid_margin"]:
                return None
        return Match(p, "handoff", area, o, offset, sim)

    def find_reattach(self, t: Tracklet, ts: float, cameras_of_area: dict, policy: dict,
                      area_type: dict | None = None, chairs: dict | None = None) -> Match | None:
        """chairs: person id -> the chair of his running haircut (2026-09-29, reattach_anchor 1: only a
        track in that chair may take him; 2: no track in another chair may)."""
        if t.reid is None:
            return None
        anchor = int(policy.get("reattach_anchor", 0))
        best, best_sim = None, policy["reid_match_min"]
        for p in self.persons.values():
            ref = reference(p, t.camera, policy)
            if t.camera in p.tracks or ref is None:
                continue
            if anchor and p.id in (chairs or {}) and chairs[p.id] not in t.areas and (
                    anchor == 1 or any((area_type or {}).get(a) == "chair" for a in t.areas)):
                continue                # 1: his haircut runs in a chair this track isn't in; 2: ...and it's in another chair
            if p.tracks:
                known = p.areas or p.last_areas
                if t.areas and known and not (t.areas & known):
                    continue            # seen somewhere else right now
                if not t.areas and any((area_type or {}).get(a) in PLACED for a in p.areas):
                    continue            # seen in a chair or at the wash right now: not a body in no zone
            elif t.areas and p.last_areas and not any(next_to(a, b, area_type or {})
                                                      for a in t.areas for b in p.last_areas):
                continue                # lost, but not next to where they were last
            gone = p.gone_from.get(t.camera)
            recent_here = gone is not None and ts - gone <= policy["reattach_s"]
            where = set(p.last_cameras).union(*(cameras_of_area.get(a, ()) for a in p.last_areas))
            lost_near = (not p.tracks and p.lost_since is not None
                         and ts - p.lost_since <= policy["reattach_s"] and t.camera in where)
            if not (recent_here or lost_near):
                continue
            sim = cosine(t.reid, ref)
            if sim >= best_sim:
                best, best_sim = p, sim
        return None if best is None else Match(best, "reattach", reid_sim=best_sim)
