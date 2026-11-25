"""ShopEngine: the three cameras' frames and sampled appearances in,
customer records out (storage.apply_update persists them).

Per frame, in order:
  1. a bound track the staff vote calls staff is released and its pooled
     face/body candidates are dropped; a person left
     with no track has its visit voided if it can be (no haircut yet).
     staff_release_mode 1 voids only a person that track created and who
     never had another; anyone else is lost, and a still unbound body in
     his running haircut's seat joins him at once (_staff_recover);
  2. door: an unbound, not-staff track crossing inward creates a person and
     an unconfirmed visit; a bound one crossing out marks a pending exit
     (crossing back in cancels it; being seen in a shop area defers it);
  3. an unbound, not-staff track tries a paired-zone handoff for each paired
     area it's in, then a re-ID re-attach;
  4. missed entry: an unbound track the vote says is NOT staff, in a waiting
     area its camera owns for waiting_debounce_s, creates a confirmed visit
     backdated to its entry there (create_in_owned_areas 1: an owned chair
     or washing area too);
  5. every bound track notes where its person is, is presence evidence for
     their chair, and marks washing_seen inside a washing area -- only in the
     areas it counts for (Tracklet.counted, below);
  6. this camera's seats: an unbound body sitting in a running haircut's
     seat joins its customer only if identity.may_claim_seat allows it (the
     customer is lost or seen only in that chair, and re-ID doesn't veto);
     a declined claim is no evidence. A start begins or resumes a haircut;
     an unbound sitter's start may first be a chair resume (_chair_resume,
     policy chair_resume_s): the chair's previous customer back in his seat.
Then tracks unseen for track_gone_s end, and the time rules run (_tick).

The engine clock is the newest frame of any camera. A track older than
track_gone_s on it (its camera stalled or died) is no handoff partner, no
bar to a seat claim, and no camera of its person in live(); a camera
camera_dead_s behind it is dead and its tracks end (so its customers can be
lost). A track id seen again after track_gone_s is a new body: the
tracker's ids restart with the sidecar.

2026-09-27 (39-min testbed spec, section 4), policy identity_model: each
frame every track gets `owned` (its areas its camera owns, or with no owner)
and `counted` (model 0: all its areas; 1 and 2: owned). Evidence -- chair
presence, washing, the person's areas, the exit check -- uses counted;
binding (handoff partners, seat claims) still uses the raw areas. Under
model 2 only a track with owned areas, or a centre-camera track outside
every zone, keeps its person from being lost."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Callable

from candidates import VisitPool
from chairs import ChairBook, ChairStart
from customers import CustomerBook
from domain import (EXIT_DOOR, EXIT_LOST, WAITING, Appearance, Binding, BodyCandidate, EngineUpdate, FaceCandidate,
                    Frame, Handoff, Registration, dt)
from identity import IdentityBook, may_claim_seat, reference, seat_claim_displaces
from shopday import TicketCounter
from tracklets import Tracklet, TrackletBook, cosine, seat_ok, staff_vote
from zones import (AREA_TYPES, CHAIR, ENTRANCE_LINE, WAITING as WAITING_ZONE, WASHING, FrameGeometry, area_of,
                   owner_cameras)

logger = logging.getLogger("barber.engine")


def _iso(d: datetime | None) -> str | None:
    return d.isoformat(timespec="seconds") if d is not None else None


class ShopEngine:
    def __init__(self, zones_by_cam: dict[str, list[dict]], geometry: FrameGeometry,
                 new_id: Callable[[], str] | None = None,
                 barber_for: Callable[[str, datetime], str | None] | None = None,
                 tickets: TicketCounter | None = None):
        self._new_id = new_id or (lambda: uuid.uuid4().hex[:12])
        self._barber_for = barber_for or (lambda chair, at: None)
        self.tickets = tickets or TicketCounter()
        self.geometry = geometry
        self._zones: dict[str, list[dict]] = {}
        self.books: dict[str, TrackletBook] = {}
        self.identity = IdentityBook(self._new_id)
        self.customers = CustomerBook(self._new_id, self._barber_for, self.tickets.next)
        self.chairs = ChairBook([])
        self._unknown_cameras: set[str] = set()
        self._exit_track: dict[str, tuple[str, int]] = {}   # person -> the track that crossed out
        self._seat_after_exit: dict[str, tuple[float, float]] = {}   # person -> (start, last) in his seat since
        self.pools: dict[str, VisitPool] = {}
        self._registered: set[str] = set()      # visits registered at their haircut's end (register_at 1)
        self.now_ts: float | None = None       # the engine clock: the newest frame of any camera
        self._track_gone_s = 2.0               # the last policy's, for live()
        self._policy: dict | None = None       # the last policy, for live()'s staff vote
        self._model = 0                        # the last policy's identity_model, for set_zones()
        self.set_zones(zones_by_cam)

    # ── configuration ───────────────────────────────────────────────────────

    def set_zones(self, zones_by_cam: dict[str, list[dict]]) -> None:
        self._zones = zones_by_cam
        for cam, zones in zones_by_cam.items():
            if cam in self.books:
                self.books[cam].set_zones(zones)
            else:
                self.books[cam] = TrackletBook(cam, zones, self.geometry)
        self.area_type: dict[str, str] = {}
        self.cameras_of_area: dict[str, set[str]] = {}
        for cam, zones in zones_by_cam.items():
            for z in zones:
                if z["type"] in AREA_TYPES:
                    self.area_type[area_of(z)] = z["type"]
                    self.cameras_of_area.setdefault(area_of(z), set()).add(cam)
        self.owner = owner_cameras(zones_by_cam)
        self.paired = {a for a, cams in self.cameras_of_area.items() if len(cams) > 1}
        self.waiting_areas = {a for a, ty in self.area_type.items() if ty == WAITING_ZONE}
        self.washing_areas = {a for a, ty in self.area_type.items() if ty == WASHING}
        # the camera that draws the door: its tracks outside every zone keep a person alive (model 2)
        self.centre = next((cam for cam, zones in zones_by_cam.items()
                            if any(z["type"] == ENTRANCE_LINE for z in zones)), None)
        self.chairs.set_chairs([a for a, ty in self.area_type.items() if ty == CHAIR], self.owner)
        for book in self.books.values():
            for t in book.tracks.values():
                self._count(t)

    def clear(self) -> list[str]:
        """Forget everyone (a test clip looped); keep the zones. Returns the
        crop paths of everything pooled, for the caller to delete."""
        paths = [p for pool in self.pools.values() for p in pool.paths()]
        zones = self._zones
        self.books = {}
        self.identity = IdentityBook(self._new_id)
        self.customers = CustomerBook(self._new_id, self._barber_for, self.tickets.next)
        self.chairs = ChairBook([])
        self._exit_track = {}
        self._seat_after_exit = {}
        self.pools = {}
        self._registered = set()
        self.now_ts = None
        self.set_zones(zones)
        return paths

    # ── one cycle ───────────────────────────────────────────────────────────

    def observe(self, frames: list[Frame], appearances: list[Appearance], policy: dict,
                faces: tuple[FaceCandidate, ...] = ()) -> EngineUpdate:
        self.tickets.hour = int(policy.get("day_start_hour", 5))
        self._track_gone_s = policy["track_gone_s"]
        self._policy = policy
        self._model = int(policy.get("identity_model", 0))
        unknown = {f.camera for f in frames if f.camera not in self.books} - self._unknown_cameras
        for cam in unknown:
            logger.warning("frames from a camera with no configured zones: %r", cam)
        self._unknown_cameras |= unknown
        items = ([(f.ts, 1, f) for f in frames if f.camera in self.books] + [(a.ts, 0, a) for a in appearances]
                  + [(c.ts, 0, c) for c in faces])
        for _, is_frame, item in sorted(items, key=lambda x: (x[0], x[1])):
            if is_frame:
                self._frame(item, policy)
            elif isinstance(item, FaceCandidate):
                self._face(item, policy)
            else:
                self._appearance(item, policy)
        latest = [b.latest for b in self.books.values() if b.latest is not None]
        if latest:
            for book in self.books.values():
                for t in book.expire(policy["track_gone_s"]):
                    self.identity.release(t, t.last_seen)
            self._bury_dead_cameras(max(latest), policy)
            self._tick(max(latest), policy)
        if int(policy.get("register_at", 0)) >= 1:
            self._register_early(policy)
        return self.customers.take_update()

    def _bury_dead_cameras(self, now_ts: float, policy: dict) -> None:
        """2026-09-26 (final B review C1b): a camera whose newest frame lags the
        engine clock by camera_dead_s is dead -- end its tracks, else its
        frozen tracks keep their customers open forever. A person left with
        no track is lost from when anyone last saw him."""
        for book in self.books.values():
            if book.latest is None or now_ts - book.latest <= policy["camera_dead_s"] or not book.tracks:
                continue
            logger.warning("camera %s sent no frame for %.0f s: ending its %d track(s)",
                           book.camera, now_ts - book.latest, len(book.tracks))
            for t in book.expire(-1.0):
                was = self.identity.persons[t.person].lost_since if t.person in self.identity.persons else None
                p = self.identity.release(t, t.last_seen)
                if p is not None and not p.tracks and was is None:   # lost earlier (model 2): that time stands
                    p.lost_since = max(p.lost_since, p.last_seen)

    def _count(self, t: Tracklet) -> None:
        """Where t is evidence (identity_model): owned = its areas its camera owns
        or that have no owner; counted = all its areas under model 0, else owned."""
        t.owned = frozenset(a for a in t.areas if self.owner.get(a, t.camera) == t.camera)
        t.counted = t.areas if self._model == 0 else t.owned

    def _keeps_alive(self, t: Tracklet) -> bool:
        """Under identity_model 2, may t keep its person from being lost?"""
        return self._model < 2 or bool(t.owned) or (t.camera == self.centre and not t.areas)

    def _bind(self, t: Tracklet, p, ts: float) -> None:
        """identity.bind, except that under model 2 a track that can't keep him
        alive doesn't clear his lost time (fix round 1)."""
        lost = p.lost_since
        self.identity.bind(t, p, ts)
        if not self._keeps_alive(t):
            p.lost_since = lost

    def _fresh(self, t: Tracklet) -> bool:
        """Seen within track_gone_s on the engine clock (not a stalled camera's frozen track)."""
        return self.now_ts is None or self.now_ts - t.last_seen <= self._track_gone_s

    def _appearance(self, app: Appearance, policy: dict) -> None:
        book = self.books.get(app.camera)
        t = book.add_appearance(app) if book is not None else None
        if t is not None and t.person is not None and app.reid is not None:
            self.identity.add_reid(self.identity.persons[t.person], app.reid, app.camera, bool(t.owned))
        staff = t is not None and self._staff(t, policy) is True
        if staff:
            self._purge_track(t)            # step 1 releases it at its next frame
        if app.crop_path:
            v = self._open_visit_of_track(app.camera, app.track_id)
            seated = t is not None and t.seat is not None and t.seat == self.chairs.of_person(t.person)
            if v is None or staff or app.staff_p is None or app.staff_p >= policy["staff_threshold"] \
                    or (int(policy.get("body_from_seat", 0)) >= 1 and not seated):
                # body_from_seat 1 (2026-09-29): only a body sitting in his own running haircut's seat --
                # body re-ID can't tell caped customers apart, so where the crop came from decides
                self.customers.update.discard_crops.append(app.crop_path)
            else:
                # Ranked by how unlike staff the crop looks, not by its height:
                # a standing barber's crop is taller than a seated customer's.
                quality = 1.0 - app.staff_p
                body = BodyCandidate(app.camera, app.track_id, app.ts, app.crop_path, app.reid, round(quality, 4))
                self.customers.update.discard_crops += self._pool(v, policy).add_body(body)

    def _purge_track(self, t: Tracklet) -> None:
        """The track is staff: none of its candidates may reach the register --
        only this track's, not an earlier body's that had the same id."""
        for pool in self.pools.values():
            self.customers.update.discard_crops += pool.drop_track(t.camera, t.track_id, since=t.first_seen)

    def _looks_like_staff(self, t: Tracklet | None, policy: dict) -> bool:
        """The face gate: the track's staff vote, or else the mean of its
        last staff_min_samples samples. The mean is what catches a latched
        customer track (never staff by vote) that a tracker id swap handed
        to the barber."""
        if t is None:
            return False
        if self._staff(t, policy) is True:
            return True
        recent = list(t.staff_samples)[-int(policy["staff_min_samples"]):]
        return bool(recent) and sum(recent) / len(recent) >= policy["staff_threshold"]

    def _open_visit_of_track(self, camera: str, track_id: int):
        book = self.books.get(camera)
        t = book.tracks.get(track_id) if book is not None else None
        if t is None or t.person is None:
            return None
        return self.customers.visits.get(t.person)

    def _pool(self, v, policy: dict) -> VisitPool:
        pool = self.pools.get(v.id)
        if pool is None:
            pool = self.pools[v.id] = VisitPool(int(policy["visit_max_faces"]), int(policy["visit_max_bodies"]))
        return pool

    def _face(self, c: FaceCandidate, policy: dict) -> None:
        v = self._open_visit_of_track(c.camera, c.track_id)
        book = self.books.get(c.camera)
        if v is None or self._looks_like_staff(book.tracks.get(c.track_id) if book else None, policy) \
                or c.quality < policy.get("face_min_quality", 0.0):     # 2026-09-29: a poor face's embedding is noise
            self.customers.update.discard_crops.append(c.crop_path)
            return
        self.customers.update.discard_crops += self._pool(v, policy).add_face(c)

    def _close_pool(self, v, register: bool) -> None:
        pool = self.pools.pop(v.id, None)
        self._registered.discard(v.id)
        if register:
            faces, bodies = pool.take_unsent() if pool else ((), ())
            self.customers.update.registrations.append(Registration(v.id, v.exited_at, faces, bodies))
        elif pool is not None:
            self.customers.update.discard_crops += pool.paths()

    def _register_early(self, policy: dict) -> None:
        """register_at 1 (2026-09-29, user): a visit is registered as soon as its first haircut
        completes -- the customers page then follows the completed haircuts -- and topped up with
        the later crops when it ends (_close_pool)."""
        for v in self.customers.visits.values():
            if v.haircuts > 0 and v.confirmed and v.id not in self._registered:
                self._registered.add(v.id)
                faces, bodies = self._pool(v, policy).take_unsent()
                at = self.customers.last_closed.get(v.person_id)
                self.customers.update.registrations.append(
                    Registration(v.id, at.ended_at if at is not None else dt(self.now_ts), faces, bodies))

    def _staff(self, t: Tracklet, policy: dict) -> bool | None:
        return staff_vote(t, policy)

    def _bound(self, kind: str, t: Tracklet, pid: str, ts: float, area: str | None = None,
               partner: Tracklet | None = None, reid_sim: float | None = None, offset_s: float | None = None) -> None:
        self.customers.note_binding(Binding(kind, t.camera, t.track_id, pid, area,
                                            partner.camera if partner else None, partner.track_id if partner else None,
                                            None if reid_sim is None else round(reid_sim, 4),
                                            None if offset_s is None else round(offset_s, 2), dt(ts)))

    def _frame(self, frame: Frame, policy: dict) -> None:
        book = self.books[frame.camera]
        ev = book.observe(frame, policy)
        for t in book.tracks.values():
            self._count(t)
        ts, at = frame.ts, dt(frame.ts)
        self.now_ts = ts if self.now_ts is None else max(self.now_ts, ts)
        for t in ev.ended:                  # over after a gap in this camera's frames
            self.identity.release(t, t.last_seen)
        seen = [book.tracks[o.track_id] for o in frame.tracks]
        # 1. staff never carry a customer
        for t in seen:
            if t.person is not None and self._staff(t, policy) is True:
                self._purge_track(t)
                p = self.identity.release(t, ts)
                if p is not None and policy.get("staff_release_mode", 0) >= 1 and not p.only_ever(t):
                    self._staff_recover(t, p, ts, policy)   # lost (release set lost_since), not voided
                    continue
                if p is not None and not p.tracks:
                    v = self.customers.visits.get(p.id)
                    if self.customers.void(p.id, at):
                        self.identity.drop(p.id)
                        if v is not None:
                            self._close_pool(v, register=False)
        # 2. the door
        for t in ev.crossed_in:
            if t.person is None and self._staff(t, policy) is not True:
                p = self.identity.create(t, ts)
                self.customers.open(p.id, at, True, policy)
                self._bound("door", t, p.id, ts)
            elif t.person is not None:
                self.customers.cancel_exit(t.person)
                self._exit_track.pop(t.person, None)
                self._seat_after_exit.pop(t.person, None)
        for t in ev.crossed_out:
            if t.person is not None:
                self.customers.exit_crossed(t.person, at)
                self._exit_track.setdefault(t.person, t.key)
        # 3. unbound bodies join a person
        for t in seen:
            if t.person is None and not t.retired and self._staff(t, policy) is not True:
                self._join(t, ts, policy)
        # 4. missed entries in an area this camera owns
        creating = {WAITING_ZONE, CHAIR, WASHING} if policy.get("create_in_owned_areas", 0) >= 1 else {WAITING_ZONE}
        for t in seen:
            if t.person is not None or t.retired or self._staff(t, policy) is not False:
                continue
            waited = [(a, t.entered[a]) for a in t.areas
                      if self.area_type.get(a) in creating and self.owner.get(a) == t.camera
                      and ts - t.entered[a] >= policy["waiting_debounce_s"]]
            if waited:
                area, first = min(waited, key=lambda aw: aw[1])
                p = self.identity.create(t, ts)
                self.customers.open(p.id, dt(first), t.appeared_at_door, policy, confirmed=True)
                self._bound("waiting" if area in self.waiting_areas else "owned_area", t, p.id, ts, area=area)
        # 5. evidence and last-known places
        for t in seen:
            if t.person is None:
                continue
            p = self.identity.persons[t.person]
            self.identity.note(t)
            self.chairs.note_presence(p.id, t.evidence, ts)
            if t.seat is not None and t.seat == self.chairs.of_person(p.id) and p.id in self._exit_track \
                    and t.key != self._exit_track[p.id]:
                # in his own seat after the crossing, for seat_dwell_s (out of it no longer than
                # seat_jitter_s): he never left. A moment of it is a side view lagging a real exit.
                start, last = self._seat_after_exit.get(p.id, (ts, ts))
                if ts - last > policy["seat_jitter_s"]:
                    start = ts
                self._seat_after_exit[p.id] = (start, ts)
                if ts - start >= policy["seat_dwell_s"]:
                    self.customers.cancel_exit(p.id)
                    self._exit_track.pop(p.id, None)
                    self._seat_after_exit.pop(p.id, None)
            if t.evidence & self.washing_areas:
                p.washing_seen = ts
        # 6. seats seen by this camera: a new track of a seated customer rejoins them
        for t, chair, pid in self.chairs.claims(frame.camera, seen, ts, policy):
            p = self.identity.persons.get(pid)
            if p is None:
                continue
            kind = "seat_claim"
            if not may_claim_seat(t, p, chair, policy, self.now_ts):
                old = seat_claim_displaces(t, p, chair, policy, self.now_ts)
                if old is None:
                    continue                # declined: no bind, and no evidence for the chair
                self.identity.release(old, ts)   # 6.6: the body in the seat is him, not that one
                kind = "seat_swap"
            self._bind(t, p, ts)
            self.chairs.note_presence(pid, {chair}, ts)
            ref = reference(p, t.camera, policy)
            sim = cosine(t.reid, ref) if t.reid is not None and ref is not None else None
            self._bound(kind, t, pid, ts, area=chair, reid_sim=sim)
        if int(policy.get("barber_cover", 0)) >= 1:
            self._barber_cover(frame.camera, book, ts, policy)
        for start in self.chairs.observe_seats(frame.camera, list(book.tracks.values()), ts, policy):
            self._start(start, policy)

    def _barber_cover(self, camera: str, book, ts: float, policy: dict) -> None:
        """2026-09-28 (user, barber_cover 1): the barber leaning over a seated customer hides him. A
        running haircut whose customer no camera sees (no fresh track) is kept going while its owner
        camera sees a body voted staff in the chair's seat."""
        for c in self.chairs.chairs.values():
            if c.person is None or self.owner.get(c.area, camera) != camera:
                continue
            p = self.identity.persons.get(c.person)
            if p is not None and any(self._fresh(o) for o in p.tracks.values()):
                continue
            if any(t.seat == c.area and t.last_seen == ts and self._staff(t, policy) is True
                   for t in book.tracks.values()):
                c.last_evidence = ts

    def _staff_recover(self, released: Tracklet, p, ts: float, policy: dict) -> None:
        """2026-09-28 (39-min testbed spec, 6.2, staff_release_mode 1): the
        released track stood inside the chair of p's running haircut -- bind
        the still, non-staff, unbound body in that chair's seat on its owner
        camera to p at once. Only if exactly one such body is there, p has
        no live track on that camera (one track per camera), and body re-ID
        against p's owner-camera reference doesn't veto it (reid_veto, as the
        seat claim)."""
        chair = self.chairs.of_person(p.id)
        if chair is None or chair not in released.areas:
            return
        book = self.books.get(self.owner.get(chair, released.camera))
        if book is None or any(o.camera == book.camera and self._fresh(o) for o in p.tracks.values()):
            return
        bodies = [o for o in book.tracks.values()
                  if o is not released and o.seat == chair and o.person is None and not o.retired
                  and self._fresh(o) and self._staff(o, policy) is not True
                  and o.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"])]
        if len(bodies) != 1:
            return
        (t,) = bodies
        ref = reference(p, t.camera, policy)    # the owner camera's: the seat claim's veto
        sim = cosine(t.reid, ref) if t.reid is not None and ref is not None else None
        if sim is not None and sim < policy["reid_veto"]:
            return                              # fix round 1: a sitter re-ID rules out stays unbound
        self._bind(t, p, ts)
        self.chairs.note_presence(p.id, {chair}, ts)
        self._bound("staff_recover", t, p.id, ts, area=chair, reid_sim=sim)

    def _join(self, t: Tracklet, ts: float, policy: dict) -> None:
        still = t.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"])
        for area in sorted(t.areas & self.paired):
            partners = [o for cam, b in self.books.items() if cam != t.camera
                        for o in b.tracks.values() if o.person is not None and area in o.areas and self._fresh(o)]
            m = self.identity.find_handoff(t, area, partners, policy, still, self.area_type.get(area))
            if m is not None:
                self._bind(t, m.person, ts)
                self.customers.note_handoff(Handoff(area, m.other.camera, m.other.track_id, t.camera, t.track_id,
                                                    m.offset_s, m.reid_sim, dt(ts)))
                self._bound("handoff", t, m.person.id, ts, area=area, partner=m.other,
                            reid_sim=m.reid_sim, offset_s=m.offset_s)
                return
        running = {c.person: c.area for c in self.chairs.chairs.values() if c.person is not None}
        m = self.identity.find_reattach(t, ts, self.cameras_of_area, policy, self.area_type, running)
        if m is not None:
            self._bind(t, m.person, ts)
            self._bound("reattach", t, m.person.id, ts, reid_sim=m.reid_sim)

    def _start(self, start: ChairStart, policy: dict) -> None:
        t = start.tracklet
        if t.retired:
            return
        if self._staff(t, policy) is not False:
            return                      # undecided or staff: wait for the vote
        if t.person is None and self._chair_resume(start, policy):
            return
        if t.person is None:
            p = self.identity.create(t, start.since)
            self.customers.open(p.id, dt(t.first_seen), t.appeared_at_door, policy, confirmed=True).wait_unseen = True
            self._bound("seat_start", t, p.id, start.since, area=start.chair)
        pid = t.person
        if pid not in self.customers.visits:
            return
        old = self.chairs.of_person(pid)
        if old is not None and old != start.chair and any(
                o is not t and old in o.evidence and t.last_seen - o.last_seen <= policy["track_gone_s"]
                for o in self.identity.persons[pid].tracks.values()):
            # still seen in his own chair: this seated body is someone else, not a move
            self._purge_track(t)            # its crops went to the wrong customer's pool
            self.identity.release(t, t.last_seen)
            p = self.identity.create(t, start.since)
            self.customers.open(p.id, dt(t.first_seen), t.appeared_at_door, policy, confirmed=True).wait_unseen = True
            self._bound("seat_start", t, p.id, start.since, area=start.chair)
            pid, old = p.id, None
        if old is not None:                 # moved to another chair
            end = self.chairs.release(old)
            self.customers.end_haircut(end.session_id, dt(end.at), False, policy)
        s = self.customers.start_haircut(pid, start.chair, dt(start.since), policy)
        self.chairs.begin(start.chair, pid, s.id, t.last_seen)

    def _chair_resume(self, start: ChairStart, policy: dict) -> bool:
        """2026-09-28 (39-min testbed spec, 6.1): an unbound body starting a
        haircut in chair X within chair_resume_s of X's last haircut ending is
        that haircut's customer P coming back -- bind him to P and reopen P's
        session (suspended at the wash: resume it by the wash-return rule) --
        unless P is seen elsewhere (a live track in a waiting or washing area,
        a live track of his own in this camera, or a door crossing outward
        since that end), or the body's embedding is clearly
        not P's reference from X's owner camera (<= resume_sim_diff; between
        that and resume_sim_same, or no embedding, is still P)."""
        window = policy.get("chair_resume_s", 0.0)
        end = self.chairs.recent_end(start.chair, start.since, window) if window > 0 else None
        if end is None:
            return False
        t, pid = start.tracklet, end.person
        p, v = self.identity.persons.get(pid), self.customers.visits.get(pid)
        if p is None or v is None or self.chairs.of_person(pid) is not None:
            return False                # gone, or already in another chair
        if v.session_id is not None:    # fix round 1 (ruling a): suspended at the wash counts as an end
            s = self.customers.sessions.get(v.session_id)
            washing = s is not None and s.washing_since is not None and s.id == end.session_id
            if not washing or s.chair_id != start.chair:
                return False
        else:
            washing = False
            closed = self.customers.last_closed.get(pid)
            if closed is None or closed.id != end.session_id:
                return False
        elsewhere = self.waiting_areas | self.washing_areas
        if any(self._fresh(o) and (o.camera == t.camera or o.evidence & elsewhere) for o in p.tracks.values()):
            return False
        if v.exit_crossed_at is not None and v.exit_crossed_at >= dt(end.at):
            return False
        ref = reference(p, self.owner.get(start.chair, t.camera), policy)
        sim = cosine(t.reid, ref) if t.reid is not None and ref is not None else None
        if sim is not None and sim < policy["resume_sim_same"] and sim <= policy["resume_sim_diff"]:
            return False
        if washing:                     # back from the wash: the existing wash-return path
            s = self.customers.start_haircut(pid, start.chair, dt(start.since), policy)
        else:
            s = self.customers.reopen_haircut(pid, end.session_id, dt(start.since), policy)
            if s is None:
                return False
        self._bind(t, p, t.last_seen)
        self.chairs.begin(start.chair, pid, s.id, t.last_seen)
        self._bound("chair_resume", t, pid, start.since, area=start.chair, reid_sim=sim)
        return True

    def _tick(self, now_ts: float, policy: dict) -> None:
        now = dt(now_ts)
        wash = int(policy.get("washing_state", 1)) >= 1
        if not wash:                        # 2026-09-29 (user): no washing state for now -- end any in progress
            self.customers.expire_washing(now, {**policy, "wash_return_s": 0.0})
        for end in self.chairs.expire(now_ts, policy):
            p = self.identity.persons.get(end.person)
            washing = wash and p is not None and p.washing_seen is not None and p.washing_seen >= end.at
            self.customers.end_haircut(end.session_id, dt(end.at), washing, policy)
        model = self._model
        for pid, v in list(self.customers.visits.items()):
            p = self.identity.persons.get(pid)
            if model >= 2 and p is not None and p.tracks:
                # full transfer: a track only a non-owner camera has doesn't keep him (spec section 4)
                if any(self._keeps_alive(t) for t in p.tracks.values()):
                    p.lost_since = None
                elif p.lost_since is None:
                    p.lost_since = now_ts
            if not v.confirmed:
                if v.exit_crossed_at is not None:
                    self._finish(pid, v.exit_crossed_at, EXIT_DOOR, policy)
                    continue
                if p is not None and p.tracks and (now - v.entered_at).total_seconds() >= policy["min_visit_s"]:
                    self.customers.confirm(pid)
            if v.exit_crossed_at is not None and not self._seen_in_area(p, now_ts, policy):
                out_of_view = p is None or not p.tracks
                if out_of_view or (now - v.exit_crossed_at).total_seconds() >= policy["exit_confirm_s"]:
                    self._finish(pid, v.exit_crossed_at, EXIT_DOOR, policy)
                    continue
            if p is not None and p.lost_since is not None and (not p.tracks or model >= 2) \
                    and now_ts - p.lost_since >= policy["lost_timeout_s"]:
                self._finish(pid, dt(p.lost_since), EXIT_LOST, policy)
        self.customers.expire_washing(now, policy)
        self.customers.check_alerts(now)

    def _seen_in_area(self, p, now_ts: float, policy: dict) -> bool:
        """Is the person, who crossed the door outward, still seen in a shop
        area? Only by tracks seen lately on the engine's clock (a frozen
        camera's last track doesn't count), and not by the crossing track
        itself: in the centre view the door line runs next to chairs L1/R1."""
        if p is None:
            return False
        crossing = self._exit_track.get(p.id)
        return any(o.evidence for o in p.tracks.values()
                   if o.key != crossing and now_ts - o.last_seen <= policy["track_gone_s"])

    def _finish(self, pid: str, at: datetime, reason: str, policy: dict) -> None:
        self._exit_track.pop(pid, None)
        self._seat_after_exit.pop(pid, None)
        chair = self.chairs.of_person(pid)
        if chair is not None:
            end = self.chairs.release(chair)
            self.customers.end_haircut(end.session_id, min(at, dt(end.at)), False, policy)
        v = self.customers.visits.get(pid)
        self.customers.finish(pid, at, reason, policy)
        if v is not None:
            self._close_pool(v, register=v.confirmed and v.haircuts > 0)
        self.identity.drop(pid)

    # ── read side ───────────────────────────────────────────────────────────

    def live(self) -> dict:
        customers = []
        for pid, v in self.customers.visits.items():
            if not v.confirmed:
                continue
            p = self.identity.persons.get(pid)
            s = self.customers.sessions.get(v.session_id) if v.session_id else None
            customers.append({
                "visit_id": v.id, "person_id": pid, "state": v.state,
                "entered_at": _iso(v.entered_at), "entry_observed": v.entry_observed,
                "waiting_started_at": _iso(v.waiting_started_at), "service_started_at": _iso(v.service_started_at),
                "wait_target_s": v.wait_target_s,
                "chair_id": s.chair_id if s else None, "session_id": s.id if s else None,
                "haircut_started_at": _iso(s.started_at) if s else None,
                "barber_id": s.barber_id if s else None,
                "ticket": v.ticket, "state_since": _iso(v.state_since),
                "cameras": sorted(c for c, t in p.tracks.items() if self._fresh(t)) if p else []})
        chairs = {a: {"person_id": c.person, "session_id": c.session_id, "sitting": len(c.sitting)}
                  for a, c in self.chairs.chairs.items()}
        customers.sort(key=lambda c: c["entered_at"])
        # 2026-09-26 (final B review M5): ages on the engine clock, not the wall clock -- frame
        # times are the sidecar's, and replays and tests have no wall clock. A stall of every
        # camera reads 0 for the newest one (the sidecar's fps gauge shows that).
        cameras = {cam: {"last_frame_age_s": None if b.latest is None or self.now_ts is None
                         else round(self.now_ts - b.latest, 1)} for cam, b in self.books.items()}
        return {"customers": customers, "chairs": chairs, "cameras": cameras,
                # 2026-09-28 (user): bodies the centre camera sees in a waiting area right now, not the
                # ones its staff vote says are barbers -- nothing more
                "customers_waiting": sum(1 for t in self.books[self.centre].tracks.values()
                                         if self._fresh(t) and t.areas & self.waiting_areas
                                         and not (self._policy and self._staff(t, self._policy) is True))
                if self.centre in self.books else 0}

    def _occupancy(self, t: Tracklet, policy: dict) -> dict:
        """Compute occupancy reasons for a tracklet in its chair zone, if any."""
        chair = t.seat or next((a for a in sorted(t.areas) if self.area_type.get(a) == CHAIR), None)
        if chair is None:
            return {"chair": None, "why": None}
        if t.seat is None:
            return {"chair": chair, "why": "in_chair_zone"}
        if self.owner.get(chair, t.camera) != t.camera:
            return {"chair": chair, "why": "not_owner"}
        if not seat_ok(t, policy):
            return {"chair": chair, "why": "cannot_sit"}
        if not t.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"]):
            return {"chair": chair, "why": "not_still"}
        return {"chair": chair, "why": "counted"}

    def scene(self, policy: dict) -> dict[str, list[dict]]:
        """Each camera's bodies in its latest frame, normalized boxes; none for a
        camera whose latest frame is older than track_gone_s on the engine clock."""
        w, h = self.geometry.width, self.geometry.height
        r = lambda p: [round(p[0], 4), round(p[1], 4)]
        out = {}
        for cam, book in self.books.items():
            result = []
            for t in book.tracks.values():
                if t.last_seen != book.latest or not self._fresh(t):
                    continue
                sp = self.geometry.seat_points(t.box)
                result.append({
                    "track_id": t.track_id,
                    "box": [round(t.box[0] / w, 4), round(t.box[1] / h, 4), round(t.box[2] / w, 4), round(t.box[3] / h, 4)],
                    "person_id": t.person,
                    "state": self.customers.visits[t.person].state if t.person in self.customers.visits else None,
                    "ticket": self.customers.visits[t.person].ticket if t.person in self.customers.visits else None,
                    "staff": self._staff(t, policy),
                    "areas": sorted(t.areas), "seat": t.seat,
                    "still": t.still(policy["still_window_s"], policy["still_radius"], policy["still_box_frac"]),
                    "anchor": r(self.geometry.anchor(t.box)),
                    "foot": r(self.geometry.foot(t.box)),
                    "seat_line": [r(sp[0]), r(sp[-1])],
                    "occupancy": self._occupancy(t, policy),
                })
            out[cam] = result
        return out
