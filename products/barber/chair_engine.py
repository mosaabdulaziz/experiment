"""The chair-first engine (project 3): ShopEngine's interface over the seat rule (seat.py), chair
states (chairs2.py), haircuts (haircuts.py), door crossings (door.py) and customer waits
(waits.py, project 4 A + B). Each side camera runs
its own chairs; nothing is shared across cameras; decisions read the place and the current
detection. Engine time is the frames' time (Redis entry ids live, the recording's clock offline)."""
from __future__ import annotations

import itertools
from collections import deque
import logging
import uuid
from dataclasses import replace
from datetime import datetime

from alerts import EXCESSIVE_WAIT
from chairs2 import EMPTY, SITTING, ChairStates, Reseat, SitEnd, SitStart
from door import DoorCounter
from entry_faces import EntryFaces
from extra_waits import ExtraWaits
from domain import Appearance, EngineUpdate, FaceCandidate, Frame, dt
from masks import GRID
from photo_plan import PhotoPlanner, outline_boxes
from haircuts import HaircutBook, ResumeDecider, pedicure_policy
from policy import DEFAULT_POLICY, policy_for
from seat import chair_geoms, evaluate_detail
from waiting import WaitingCounter, lower_share_in, zones_of_type
from waits import WaitBook
from zones import (ENTRANCE_LINE, FACE_AREA, PEDICURE, WASHING, DoorLine, FrameGeometry, point_in_polygon,
                   rect_overlaps_polygon)

logger = logging.getLogger("barber.chair_engine")

# 2026-10-04 (final review I-5): staff scores of tracks unseen this long on their camera are dropped
# (thousands of tracker ids an hour otherwise pile up in a 24/7 service).
TRACK_TTL_S = 600.0
PRUNE_EVERY_S = 60.0
STAFF_BEFORE_S, STAFF_AFTER_S = 60.0, 120.0   # ce_staff_cut_check: before he sat down, after he got up
STAFF_HIST_N = 400      # a track's staff samples kept (about 1 a second: the 2-min after window and more)
NO_MASK_WARN_S = 60.0

SITTING_MARGIN_S = 1.0     # _sitting_now: his chair occupied at least this long after the dwell began


def _iso(d: datetime | None) -> str | None:
    return d.isoformat(timespec="seconds") if d is not None else None


class ChairEngine:
    def __init__(self, zones_by_cam: dict[str, list[dict]], geometry: FrameGeometry, barber_for=None,
                 tickets=None, decider: ResumeDecider | None = None, wait_camera: str | None = None):
        self.geometry = geometry
        self._barber_for = barber_for or (lambda chair, at: None)
        self.tickets = tickets
        self._decider = decider
        self._wait_camera = wait_camera
        self._ids = itertools.count(1)
        self._policy = dict(DEFAULT_POLICY)
        self.set_zones(zones_by_cam)
        self.clear()

    # ── zones / reset ───────────────────────────────────────────────────────
    def set_zones(self, zones_by_cam: dict[str, list[dict]]) -> None:
        geoms = chair_geoms(zones_by_cam)
        camera_of = {g.chair: cam for cam, gs in geoms.items() for g in gs if g.owner}
        kind_of = {g.chair: g.kind for gs in geoms.values() for g in gs if g.owner}
        if hasattr(self, "haircuts"):
            self._close_orphans(camera_of, kind_of)
        self.geoms, self.camera_of = geoms, camera_of
        self.kind_of = kind_of                          # chair -> "haircut" | PEDICURE (2026-11-10)
        self.pedicure_chairs = {c for c, k in kind_of.items() if k == PEDICURE}
        self.outline_of = outline_boxes(zones_by_cam, camera_of)      # haircut photos' close-ups (photo_plan.py)
        self._zones_by_cam = zones_by_cam
        self.face_areas = {cam: [(z["chair"], z["points"]) for z in zs if z.get("type") == FACE_AREA]
                           for cam, zs in zones_by_cam.items()}
        line = next((z for zs in zones_by_cam.values() for z in zs if z.get("type") == ENTRANCE_LINE), None)
        self.door_cam = next((cam for cam, zs in zones_by_cam.items()
                              if any(z.get("type") == ENTRANCE_LINE for z in zs)), None)
        self.wait_cam = self._wait_camera or self.door_cam
        self.door = DoorCounter(DoorLine.from_zone(line) if line else None, self.geometry)
        if hasattr(self, "entries"):
            self.entries.set_camera(self.door_cam)
        if hasattr(self, "chairs"):
            self.chairs.set_chairs(self.camera_of)
        if hasattr(self, "waiting"):
            self.waiting.set_zones(zones_by_cam, self.wait_cam)
        if hasattr(self, "waits"):
            self.waits.set_zones(zones_by_cam, self.wait_cam)
            self.extra.set_zones(zones_by_cam, self.wait_cam)

    def _close_orphans(self, camera_of: dict[str, str], kind_of: dict[str, str]) -> None:
        """Haircuts / pedicures on chairs that no longer exist, or no longer have that kind, are closed into the next
        observe()'s update (2026-11-10: a chair ticked or unticked Pedicure mid-session)."""
        for book, kind, pol in ((self.haircuts, "haircut", self._policy),
                                (self.pedicures, PEDICURE, pedicure_policy(self._policy))):
            keep = {c for c in camera_of if kind_of.get(c, "haircut") == kind}
            for chair, h in list(book.open.items()):
                if chair in keep:
                    continue
                if h.last_end is None:
                    cam = self.camera_of.get(chair)
                    ts = self.latest[cam][0] if cam in self.latest else self.now_ts
                    book.on_end(SitEnd(chair, ts if ts is not None else h.sittings[-1][0]), self._pending, pol)
                book._close(h, self._pending, pol)
            book.forget(set(self.camera_of) - keep)

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _next_ticket(self, at: datetime) -> int:
        return self.tickets.next(at) if self.tickets is not None else next(self._ids)

    def _book(self, chair: str | None) -> HaircutBook:
        """The book of the chair's kind: pedicure chairs' sittings are pedicures (2026-11-10)."""
        return self.pedicures if chair in self.pedicure_chairs else self.haircuts

    def _pol(self, chair: str | None, policy: dict) -> dict:
        return pedicure_policy(policy) if chair in self.pedicure_chairs else policy

    def clear(self) -> list[str]:
        paths = self.haircuts.clear() if hasattr(self, "haircuts") else []
        paths += self.pedicures.clear() if hasattr(self, "pedicures") else []
        paths += self.entries.clear() if hasattr(self, "entries") else []
        self.entries = EntryFaces(self.door_cam)
        self.chairs = ChairStates(self.camera_of)
        self.haircuts = HaircutBook(self._new_id, self._barber_for, self._next_ticket, self._decider)
        self.pedicures = HaircutBook(self._new_id, self._barber_for, self._next_ticket, kind=PEDICURE)
        self.photos = PhotoPlanner()
        self.pedicure_photos = PhotoPlanner()
        # chair -> (ts, normalised x0, y0, x1, y1) of its sitter, while its camera's newest frame has one on the seat
        self.sitter_box: dict[str, tuple[float, tuple]] = {}
        self.door = DoorCounter(self.door.line, self.geometry)
        self.waiting = WaitingCounter(self._zones_by_cam, self.wait_cam)
        self.waits = WaitBook(self._zones_by_cam, self.wait_cam, self.geometry, self._new_id, self._next_ticket,
                              sitting=self._sitting_now)
        self.extra = ExtraWaits(self._zones_by_cam, self.wait_cam, self._new_id, self._next_ticket)
        self._waiting_tick: float | None = None
        self.staff: dict[tuple[str, int], float] = {}
        self.staff_run: dict[tuple[str, int], int] = {}   # (camera, track) -> latest staff samples in a row
        self.staff_hist: dict[tuple[str, int], deque] = {}  # (camera, track) -> its last (ts, staff_p) (ce_staff_cut_check)
        self._sat_staff: dict[tuple[str, float], tuple[int, float]] = {}   # (chair, start) -> (n, staff share) before it
        self.seat_staff: dict[tuple[str, int], float] = {}  # the staff score the seat rule reads (ce_staff_release_n)
        self.nonstaff_run: dict[tuple[str, int], int] = {}  # (camera, track) -> latest non-staff samples in a row
        self.seen: dict[tuple[str, int], float] = {}      # (camera, track) -> last frame/appearance ts
        self.boxes: dict[tuple[str, int], tuple] = {}     # (camera, track) -> newest box (register_source 2)
        self._pruned_at: float | None = None
        self._no_mask_warned: dict[str, float] = {}       # camera -> engine ts of the last warning
        self.sitter: dict[tuple[str, int], str] = {}     # (camera, track) -> chair it occupies now
        self.verdicts: dict[str, dict] = {}              # camera -> track -> seat.Verdict of its newest frame
        self.signals: dict[str, object] = {}             # chair -> ChairSignal of its camera's newest frame
        self.latest: dict[str, tuple[float, tuple]] = {}  # camera -> (ts, tracks) of its newest frame
        self.now_ts: float | None = None
        self._pending = EngineUpdate()
        return paths

    # ── one cycle ───────────────────────────────────────────────────────────
    def observe(self, frames: list[Frame], appearances: list[Appearance], policy: dict,
                faces: tuple[FaceCandidate, ...] = ()) -> EngineUpdate:
        self._policy = policy
        upd, self._pending = self._pending, EngineUpdate()
        src = int(policy["register_source"])
        for c in faces:
            if src == 0:
                upd.discard_crops.append(c.crop_path)
            elif src == 1:
                self.entries.add(c)
            else:
                self._seated_face(c, policy, upd)
        items = sorted([(f.ts, 1, f) for f in frames] + [(a.ts, 0, a) for a in appearances], key=lambda x: x[:2])
        for ts, kind, item in items:
            self.now_ts = ts if self.now_ts is None else max(self.now_ts, ts)
            if kind == 0:
                self._appearance(item, policy, upd)
            else:
                self._frame(item, policy, upd)
        if self.now_ts is not None:
            # a STARTING chair counts as taken: its start will be backdated and may resume the haircut (M-1);
            # so does a live start chain (2026-10-05: a return flickering empty <-> starting behind the barber)
            now = self.now_ts
            taken = lambda ch: ch in self.camera_of and (self.chairs.state(ch) != EMPTY
                                                          or self.chairs.chain_live(ch, now, policy))
            self.haircuts.tick(now, taken, upd, policy,
                               lambda ch: self.chairs.occupied_until(ch) if ch in self.camera_of else None)
            self.photos.step(now, self.haircuts.open, self.haircuts.recent, self.camera_of, self.outline_of,
                             self.sitter_box, policy, upd, pending=self.haircuts.pending)
            occ_until = lambda ch: self.chairs.occupied_until(ch) if ch in self.camera_of else None
            self.pedicures.tick(now, taken, upd, pedicure_policy(policy), occ_until)
            self.pedicure_photos.step(now, self.pedicures.open, self.pedicures.recent, self.camera_of, self.outline_of,
                                      self.sitter_box, policy, upd, pending=self.pedicures.pending)
            self._staff_cuts(now, policy, upd)
            self.waits.tick(policy, upd)
            self.waits.stale(now, policy, upd)
            self.extra.tick(now, self.waits.open_in, self.waits.cam_ts, policy, upd)
            upd.registrations += self.entries.due(now, policy)
            self._attach_subjects(upd)
            self._prune()
            if self._waiting_tick is None or self.now_ts - self._waiting_tick >= 1.0:
                self._waiting_tick = self.now_ts
                self.waiting.tick(self.now_ts, policy)
        upd.discard_crops += self.entries.drain_discards()
        for s in upd.sessions.values():
            self.waits.note_haircut(s.id, s.started_at.timestamp())
        for sid in upd.deleted_sessions:
            self.waits.drop_haircut(sid)
        return upd

    def _staff_share(self, cam: str, tid: int, t0: float, t1: float, policy: dict) -> tuple[int, float]:
        """(samples, share at or above staff_threshold) of that track's staff scores in [t0, t1]."""
        v = [p for t, p in self.staff_hist.get((cam, tid), ()) if t0 <= t <= t1]
        return len(v), (sum(p >= policy["staff_threshold"] for p in v) / len(v) if v else 0.0)

    def _staff_cuts(self, now: float, policy: dict, upd: EngineUpdate) -> None:
        """ce_staff_cut_check (2026-11-10): STAFF_AFTER_S after a confirmed haircut's sitting ended, the sitter who got up
        and the one who sat down are judged; staff both times (>= ce_staff_cut_share of >= 3 samples each) = a staff
        member's own haircut, voided."""
        for chair, h in list(self.haircuts.open.items()):
            pc = policy_for(policy, chair)
            if (int(pc["ce_staff_cut_check"]) != 1 or not h.confirmed or h.last_end is None
                    or h.staff_checked_end == h.last_end or now < h.last_end + STAFF_AFTER_S):
                continue
            h.staff_checked_end = h.last_end
            nb, sb = self._sat_staff.get((chair, h.start), (0, 0.0))
            cam = self.camera_of.get(chair)
            na, sa = (self._staff_share(cam, h.last_track, h.last_end + 3.0, h.last_end + STAFF_AFTER_S, policy)
                      if cam is not None and h.last_track is not None else (0, 0.0))
            need = pc["ce_staff_cut_share"]
            if nb >= 3 and na >= 3 and sb >= need and sa >= need:
                logger.info("chair %s: haircut %s from %s is a staff member's own (staff %.2f of %d before, %.2f of %d "
                            "after) -- voided", chair, h.id, dt(h.start).time(), sb, nb, sa, na)
                self.haircuts.void_staff(h, upd)
        if len(self._sat_staff) > 500:                  # starts long gone
            for k in sorted(self._sat_staff, key=lambda k: k[1])[:250]:
                del self._sat_staff[k]

    def _sitting_now(self, session_id: str, since: float) -> bool:
        """That haircut's customer is still in the chair at `since` (a sofa dwell's start): the haircut is open, its
        last sitting hasn't ended, and its chair was occupied after `since`. A sitting's end is confirmed only
        ce_end_gap_s after the last occupied frame, so the unended flag alone lags the stand-up by a minute or more
        (final re-review N1, 2026-10-15)."""
        for chair, h in self.haircuts.open.items():
            if h.id == session_id and h.last_end is None:
                occ = self.chairs.occupied_until(chair) if chair in self.camera_of else None
                return occ is not None and occ > since + SITTING_MARGIN_S
        return False

    def _seated_face(self, c: FaceCandidate, policy: dict, upd: EngineUpdate) -> None:
        """register_source 2 (2026-10-14, branch 2): a face in chair X's face area, while X's haircut runs and the
        person isn't staff, is X's customer's. "In" = the face's own box centre inside the area; a face without
        its box (an older sidecar, an old recording) falls back to its person's box (that camera's newest frame)
        overlapping the area -- the sidecar's own face gate (2026-10-15, final review I1: face areas are drawn
        around faces, so the person box's centre, his chest, is mostly below them). Ambiguous (two running
        chairs) or anything else: dropped. No identity across cameras."""
        key = (c.camera, c.track_id)
        sp = self.staff.get(key)
        hit = None
        if not (sp is not None and sp >= policy["staff_threshold"]):
            if c.face_box is not None:
                x, y = self.geometry.middle(c.face_box)
                hit = lambda pts: point_in_polygon(x, y, pts)
            elif key in self.boxes:
                rect = self.geometry.rect(self.boxes[key])
                hit = lambda pts: rect_overlaps_polygon(*rect, pts)
        running = []
        if hit is not None:
            for chair, pts in self.face_areas.get(c.camera, []):
                hc = self.haircuts.open.get(chair)
                if hc is not None and hc.last_end is None and hit(pts):
                    running.append(chair)
        if len(set(running)) != 1:
            upd.discard_crops.append(c.crop_path)
            return
        self.haircuts.add_face(running[0], c, policy, upd)

    def _prune(self) -> None:
        if self._pruned_at is not None and self.now_ts - self._pruned_at < PRUNE_EVERY_S:
            return
        self._pruned_at = self.now_ts
        stale = [k for k, ts in self.seen.items()
                 if (self.latest[k[0]][0] if k[0] in self.latest else self.now_ts) - ts > TRACK_TTL_S]
        for k in stale:
            del self.seen[k]
            self.staff.pop(k, None)
            self.staff_run.pop(k, None)
            self.staff_hist.pop(k, None)
            self.seat_staff.pop(k, None)
            self.nonstaff_run.pop(k, None)
            self.boxes.pop(k, None)

    def _warn_no_masks(self, f: Frame) -> None:
        if not f.tracks or any(t.conf is not None for t in f.tracks):
            return
        last = self._no_mask_warned.get(f.camera)
        if last is None or f.ts - last >= NO_MASK_WARN_S:
            self._no_mask_warned[f.camera] = f.ts
            logger.warning("%s: persons but no person_masks in its frames -- the chair engine sees no one seated "
                           "(is the sidecar the seg build?) (logged at most once a minute)", f.camera)

    def _appearance(self, app: Appearance, policy: dict, upd: EngineUpdate) -> None:
        self.seen[(app.camera, app.track_id)] = app.ts
        if app.staff_p is not None:
            key = (app.camera, app.track_id)
            self.staff[key] = app.staff_p
            self.staff_hist.setdefault(key, deque(maxlen=STAFF_HIST_N)).append((app.ts, app.staff_p))
            thr = policy["staff_threshold"]
            self.staff_run[key] = self.staff_run.get(key, 0) + 1 if app.staff_p >= thr else 0
            # 2026-10-15 (Palm Hills, L7 on ph-5): on a seat, a staff track turns non-staff only after
            # ce_staff_release_n non-staff samples in a row (one dip made a resting barber a sitter)
            run = self.nonstaff_run[key] = 0 if app.staff_p >= thr else self.nonstaff_run.get(key, 0) + 1
            was = self.seat_staff.get(key)
            if was is None or was < thr or app.staff_p >= thr or run >= policy["ce_staff_release_n"]:
                self.seat_staff[key] = app.staff_p
        chair = self.sitter.get((app.camera, app.track_id))
        self._book(chair).on_appearance(app, chair, self._pol(chair, policy), upd)
        self.waits.on_appearance(app)

    def _frame(self, f: Frame, policy: dict, upd: EngineUpdate) -> None:
        self.latest[f.camera] = (f.ts, f.tracks)
        for t in f.tracks:
            self.seen[(f.camera, t.track_id)] = f.ts
            self.boxes[(f.camera, t.track_id)] = t.box
        staff_of = lambda tid: self.staff.get((f.camera, tid))
        crossings = self.door.observe(f, staff_of, policy) if f.camera == self.door_cam else []
        # an "in" by someone holding a wait was inside already (a step back over the line, the sofa's edge)
        crossings = [c for c in crossings if not (c.direction == "in" and self.waits.holder(c.track_id) is not None)]
        if int(policy["register_source"]) == 1:        # door faces; 0 = chair crops, no door entries
            crossings = [replace(c, entry_id=self._new_id()) if c.direction == "in" and not c.staff else c
                         for c in crossings]
        for c in crossings:
            if c.entry_id:
                self.entries.entry(c.entry_id, c.track_id, c.at.timestamp())
        if f.camera == self.door_cam:
            thr = policy["staff_threshold"]
            self.entries.observe(f, self.geometry.foot, lambda tid: (staff_of(tid) or 0.0) >= thr)
        upd.door_crossings += crossings
        geoms = self.geoms.get(f.camera)
        if geoms:
            self._warn_no_masks(f)
            signals, self.verdicts[f.camera] = evaluate_detail(f.tracks, geoms,
                                                               lambda tid: self.seat_staff.get((f.camera, tid)), policy)
        else:
            signals = {}
        seated = {s.track_id for s in signals.values() if s.occupied and s.track_id is not None}
        self.waiting.observe(f, staff_of, seated, policy)
        self.waits.observe(f, staff_of, seated, crossings, policy, upd,
                           staff_run=lambda tid: self.staff_run.get((f.camera, tid), 0))
        self.extra.observe(f, staff_of, seated, policy)
        if not geoms:
            return
        self.signals.update(signals)
        for key in [k for k in self.sitter if k[0] == f.camera]:
            del self.sitter[key]
        for chair, sig in signals.items():
            book, pol = self._book(chair), self._pol(chair, policy)
            if sig.occupied and sig.track_id is not None:
                self.sitter[(f.camera, sig.track_id)] = chair
                b = self.boxes.get((f.camera, sig.track_id))
                if b is not None and self.camera_of.get(chair) == f.camera:   # the haircut photo's close-up
                    g = self.geometry
                    self.sitter_box[chair] = (f.ts, (b[0] / g.width, b[1] / g.height,
                                                     (b[0] + b[2]) / g.width, (b[1] + b[3]) / g.height))
                h = book.open.get(chair)
                if (h is not None and h.last_end is None and self.chairs.state(chair) == SITTING
                        and chair not in book.pending):                                           # not while re-ID decides
                    h.track_ids.add(sig.track_id)       # the tracker may re-id the sitter mid-haircut
                    h.last_track = sig.track_id         # ce_staff_cut_check: who got up at the end
                    box = self.boxes.get((f.camera, sig.track_id))
                    if box is not None:                 # ce_object_spread: a bag on the seat never moves
                        g = self.geometry               # boxes are mux pixels; the spread is a fraction of the frame
                        conf = next((t.conf for t in f.tracks if t.track_id == sig.track_id), None)
                        book.note_box(chair, (box[0] / g.width, box[1] / g.height), conf)   # ce_object_conf
            if not (sig.occupied and sig.track_id is not None) and self.camera_of.get(chair) == f.camera:
                self.sitter_box.pop(chair, None)    # 2026-10-22 (user): nobody on the seat in this frame -- no photo now
            book.note_frame(chair, f.ts, sig.occupied, pol)       # ce_low_occupancy
            for ev in self.chairs.step(chair, f.ts, sig, policy):
                if isinstance(ev, SitStart):
                    if (chair not in self.pedicure_chairs and int(policy_for(policy, chair)["ce_staff_cut_check"]) == 1
                            and ev.track_id is not None):
                        self._sat_staff[(chair, ev.ts)] = self._staff_share(f.camera, ev.track_id, ev.ts - STAFF_BEFORE_S,
                                                                            ev.ts - 3.0, policy)
                    book.on_start(ev, upd, pol, now_ts=f.ts)
                elif isinstance(ev, Reseat):            # seen again after cover only: is it him? (ce_reseat_check_s)
                    book.on_reseat(ev, upd, pol, f.ts)
                else:
                    book.on_end(ev, upd, pol)

    # ── read side ───────────────────────────────────────────────────────────
    def occupancy(self) -> dict:
        """Who is inside now, for the visit baseline: open confirmed haircuts (away = chair empty), open waits, and
        non-staff people of the entrance-line camera standing in its washing zone(s)."""
        policy = self._policy
        cuts = [(chair, h) for chair, h in self.haircuts.open.items() if h.confirmed]
        away = sum(1 for chair, _ in cuts if self.chairs.state(chair) == EMPTY
                   and not (self.now_ts is not None and self.chairs.no_view(chair, self.now_ts, policy)))
        washing = 0
        zones = zones_of_type(self._zones_by_cam, WASHING).get(self.door_cam) if self.door_cam else None
        frame = self.latest.get(self.door_cam) if self.door_cam else None
        if zones and frame is not None:
            for t in frame[1]:
                sp = self.staff.get((self.door_cam, t.track_id))
                if (sp is not None and sp >= policy["staff_threshold"]) or t.conf is None \
                        or t.conf < policy["ce_conf_min"]:
                    continue
                if lower_share_in(t.outline, zones, policy["wc_lower_frac"], policy["wc_zone_min"]):
                    washing += 1
        return {"haircuts": len(cuts), "away": away, "waiting": len(self.waits.open) + len(self.extra.open()),
                "washing_people": washing}

    def live(self) -> dict:
        chairs, customers = {}, []
        for chair in self.camera_of:
            book = self._book(chair)
            h = book.open.get(chair)
            pend = book.pending.get(chair)             # sitting, re-ID still deciding: new customer or his back
            check = pend                               # either check shows as checking ...
            if pend is not None and pend.reseat_from is not None:
                pend = None                            # ... but under a reseat check his haircut runs on, shown
            prev = h if h is not None else book.recent.get(chair)
            state = self.chairs.state(chair)
            sig = self.signals.get(chair)
            if self.now_ts is not None and self.chairs.no_view(chair, self.now_ts, self._policy):
                state = "no_view"
            elif state == "empty" and h is not None:
                state = "away"
            since = (self.chairs.since(chair) if state in ("sitting", "starting")
                     else h.last_end if state == "away" and h else None)
            shown = h is not None and h.confirmed and pend is None
            d = book.decisions.get(chair)              # the re-ID check's answer, while the sitting it decided runs
            if not (d and pend is None and h is not None and h.id == d.haircut_id
                    and len(h.sittings) - 1 == d.sitting_n and h.last_end is None):
                d = None
            chairs[chair] = {"kind": self.kind_of.get(chair, "haircut"),     # 2026-11-10: pedicure chairs
                             "person_id": h.id if shown else None,
                             "session_id": h.id if shown else None,
                             "sitting": 1 if state == "sitting" else 0, "state": state,
                             "since": _iso(dt(since)) if since is not None else None,
                             # the haircut's own start (2026-10-18): a resumed haircut's clock goes on from it, while
                             # since restarts with each sitting
                             "haircut_since": _iso(dt(h.start)) if shown else None,
                             "camera": self.camera_of[chair], "occupant": sig.track_id if sig and sig.occupied else None,
                             "cover": bool(sig and sig.cover),
                             "resume_until": _iso(dt(h.last_end + self._policy["ce_resume_window_s"]))
                             if state == "away" and h is not None and h.last_end is not None else None,
                             "checking": _iso(dt(check.start.ts)) if check is not None else None,
                             "check_ticket": prev.ticket if check is not None and prev is not None else None,
                             # 2026-10-18 (user): how the check decided -- the dashboard says it with the score
                             "resume": None if d is None else {
                                 "resumed": d.resumed, "sim": None if d.sim is None else round(d.sim, 2),
                                 "threshold": d.threshold, "gap_s": round(d.gap_s), "at": _iso(dt(d.at)),
                                 "ticket": d.prev_ticket, "kind": d.kind}}
            if h is not None and h.confirmed and chair not in self.pedicure_chairs:    # a pedicure is never a customer
                sitting_now = state == "sitting" and pend is None
                customers.append({
                    "visit_id": h.id, "person_id": h.id, "state": "haircut" if sitting_now else "washing",
                    "entered_at": _iso(dt(h.start)), "entry_observed": False, "waiting_started_at": None,
                    "service_started_at": _iso(dt(h.start)), "wait_target_s": None, "chair_id": chair,
                    "session_id": h.id, "haircut_started_at": _iso(dt(h.start)), "barber_id": h.barber_id,
                    "ticket": h.ticket, "cameras": [],
                    "state_since": _iso(dt(h.last_end)) if pend is not None and h.last_end is not None
                    else chairs[chair]["since"]})
        for w in self.waits.open.values():
            started = _iso(dt(w.start))
            customers.append({
                "visit_id": w.id, "person_id": w.id, "state": "waiting", "entered_at": started,
                "entry_observed": False, "waiting_started_at": started, "service_started_at": None,
                "wait_target_s": w.target_s, "chair_id": None, "session_id": None, "haircut_started_at": None,
                "barber_id": None, "ticket": w.ticket, "state_since": started, "cameras": [self.waits.camera]})
        for e in self.extra.open():
            started = _iso(dt(e.start))
            customers.append({
                "visit_id": e.id, "person_id": e.id, "state": "waiting", "entered_at": started,
                "entry_observed": False, "waiting_started_at": started, "service_started_at": None,
                "wait_target_s": e.target_s, "chair_id": None, "session_id": None, "haircut_started_at": None,
                "barber_id": None, "ticket": e.ticket, "state_since": started, "cameras": []})
        cameras = {cam: {"last_frame_age_s": None if self.now_ts is None else round(self.now_ts - ts, 1)}
                   for cam, (ts, _) in self.latest.items()}
        customers.sort(key=lambda c: c["entered_at"])
        return {"customers": customers, "chairs": chairs, "cameras": cameras,
                "customers_waiting": self.waiting.shown,
                "waits": [{"id": w.id, "track_id": w.track, "started_at": _iso(dt(w.start)), "on_hold": w.hold}
                          for w in self.waits.open.values()]
                + [{"id": e.id, "track_id": None, "started_at": _iso(dt(e.start)), "on_hold": None}
                   for e in self.extra.open()],
                "waiting": {"cameras": self.waiting.counts, "raw": self.waiting.raw}}

    def scene(self, policy: dict) -> dict[str, list[dict]]:
        w, h = self.geometry.width, self.geometry.height
        frac = policy["ce_lower_frac"]
        out = {}
        for cam, (_, tracks) in self.latest.items():
            verdicts = self.verdicts.get(cam) if self.geoms.get(cam) else None
            rows = []
            for t in tracks:
                row = {"track_id": t.track_id,
                       "box": [round(t.box[0] / w, 4), round(t.box[1] / h, 4), round(t.box[2] / w, 4),
                               round(t.box[3] / h, 4)],
                       "person_id": None, "state": None, "ticket": None,
                       "staff": (lambda sp: None if sp is None else sp >= policy["staff_threshold"])(
                           self.staff.get((cam, t.track_id))),
                       "areas": [], "seat": self.sitter.get((cam, t.track_id)), "still": None,
                       "outline": list(t.outline) if t.outline else None,
                       "lower_y": _lower_y(t.outline, frac)}
                if cam == self.waits.camera:
                    wait = self.waits.holder(t.track_id)
                    if wait is not None:
                        row.update(person_id=wait.id, state="waiting", ticket=wait.ticket)
                if verdicts is not None:
                    v = verdicts.get(t.track_id)
                    row["rule"] = ({"chair": v.chair, "share": v.share, "why": v.why} if v is not None
                                   else {"chair": None, "share": None, "why": "no_mask"})
                rows.append(row)
            out[cam] = rows
        return out

    def _attach_subjects(self, upd: EngineUpdate) -> None:
        """New open alerts get the camera and box their snapshot shows (dashboard spec 2026-10-05)."""
        for a in upd.alerts:
            if not a.open or a.subject_cam is not None:
                continue
            if a.kind == EXCESSIVE_WAIT:
                w = self.waits.open.get(a.visit_id)
                if w is not None and w.box is not None:
                    a.subject_cam, a.subject_box = self.waits.camera, w.box
            else:
                h = next((h for h in self.haircuts.open.values() if h.id == a.session_id), None)
                g = next((g for gs in self.geoms.values() for g in gs if h is not None and g.owner and g.chair == h.chair),
                         None)
                if g is not None:
                    gh, gw = g.outline.mask.shape
                    a.subject_cam = g.camera
                    a.subject_box = (round(g.outline.x0 / GRID[0], 4), round(g.outline.y0 / GRID[1], 4),
                                     round(gw / GRID[0], 4), round(gh / GRID[1], 4))
            if a.subject_cam is None:
                logger.warning("alert %s (%s): no subject, no snapshot", a.id, a.kind)


def _lower_y(outline, frac: float) -> float | None:
    """Normalised y of the top of the lower `frac` of the mask's vertical extent (the overlay's cut line)."""
    if not outline or len(outline) < 6:
        return None
    ys = outline[1::2]
    top, bottom = min(ys), max(ys)
    return round(bottom - frac * (bottom - top), 4)
