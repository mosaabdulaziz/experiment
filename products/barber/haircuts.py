"""Haircuts and customers for the chair-first engine (project 3, spec 2c).

A haircut = one or more sittings of one customer in one chair; customers = haircut starts. A sitting
starting in a chair whose last haircut is still open (its customer away) asks the ResumeDecider:
resume (same haircut, the gap is wash time) or new (the old one closes now). A haircut is
persisted only once its sitting time reaches ce_min_haircut_s (confirmed). Sitting time runs to each
sitting's last occupied frame, never into the end gap, so the rule is the one checked at close: a sit
shorter than ce_min_haircut_s is never confirmed, while a longer one is confirmed (and shown) as soon as
it gets there. It closes when its chair stays empty for ce_resume_window_s. A confirmed haircut running
longer than service_target_s (snapshotted at confirm) opens one excessive_service alert, closed with it.
register_source (2026-10-05, temporary): 0 = the customer card's pictures are the sitter's own body
crops (appearance samples of the occupying track, staff_p below staff_threshold) from the first
ce_picture_window_s, registered when the haircut closes; 1 = the face register at the door
(entry_faces.py) holds the customers and every crop handed in is discarded at once;
2 = the seated customer's faces from face_area zones (chair_engine) plus his body crops, registered when the
haircut closes (2026-10-14, branch 2). Other crops are always discarded at once.
ce_resume_check (2026-10-07, resume_check.py): 1 = a start within ce_resume_max_s of the chair's last confirmed
haircut waits (pending) for the new sitter's re-ID and resume_check.decide answers; a haircut closed at
ce_resume_window_s stays reopenable (recent) until ce_resume_max_s or until another haircut in that chair is
confirmed (a short stray sitter judged "new" must not lock the real customer out). The new sitter's samples
start with those taken while the chair was still starting (seat_samples), so a backdated start is not decided
on nothing. 0 = the decider (time rule). Each answer is kept per chair (decisions, 2026-10-18, user: the dashboard
says "resumed" / "new customer" with the score instead of leaving it to the timer) and goes to storage (resume_check:
every score, to tune the thresholds per chair); ce_resume_sim_low / _high are read per chair. A haircut a "new customer"
answer opened needs ce_min_after_new_s of sitting (if longer) to be confirmed (2026-11-10: returns re-ID missed).
Reseat (2026-10-18, ce_reseat_check_s): a running sitting's seat taken again after a stretch with no one seen in it
(the barber's cover held the sitting) asks the same question; the sitter before the stretch is the reference. The same
customer: nothing changes. A new one: the haircut ends at the last frame its customer was seen, a new one starts.
kind PEDICURE (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md): the book of pedicure chairs -- driven with
pedicure_policy(), it writes PedicureRecords and nothing else (no visit, session, ticket, alert, card).
The excessive_service clock runs to the chair's last occupied frame, not to now (2026-10-07, review finding 2:
a haircut whose customer had left got an alert from the away / end-gap time); a reopened haircut's alert
starts no earlier than the reopen.
Low occupancy (2026-11-14, ce_low_occupancy > 0, per chair): note_frame counts each frame of a haircut's running sittings
and those with the sitter on the seat; a haircut whose share is below it (below ce_low_occupancy_short while it has
sat less than ce_low_occupancy_short_s) stays a haircut, marked low_occupancy on its session (a standing staff member cut by the frame edge, a child's leftovers: on the seat in a fifth of the frames, a
real customer in more than a third even seen from behind)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol

import numpy as np

from alerts import EXCESSIVE_SERVICE, AlertBook
from chairs2 import Reseat, SitEnd, SitStart
from domain import (COMPLETED, FINISHED, HAIRCUT, SHOP, VOID_STAFF, WASHING, Appearance, BodyCandidate, EngineUpdate,
                    FaceCandidate, PedicureRecord, Registration, ResumeCheckRecord, ServiceSession, SittingRecord, Visit,
                    dt)
from policy import policy_for
from zones import PEDICURE
from resume_check import KEEP_S, WAIT_EXTRA_S, Sample, decide, ref_window, similarity, threshold

logger = logging.getLogger("barber.haircuts")


@dataclass
class Haircut:
    id: str
    chair: str
    sittings: list[list]                    # [start_ts, end_ts | None]
    track_ids: set[int] = field(default_factory=set)
    confirmed: bool = False
    ticket: int | None = None
    barber_id: str | None = None
    service_target_s: float | None = None
    excessive: bool = False
    bodies: list[BodyCandidate] = field(default_factory=list)
    faces: list[FaceCandidate] = field(default_factory=list)   # register_source 2: the seated customer's faces
    reid: list[Sample] = field(default_factory=list)   # the sitter's last ce_resume_ref_s (ce_resume_check)
    registered: bool = False                           # its card was handed in (a reopened haircut: once)
    alert_from: float | None = None                    # reopened at: its excessive alert starts no earlier
    boxes: list[tuple[float, float, float]] = field(default_factory=list)   # the occupant's (left, top, conf) until confirmed
    after_new: bool = False                            # opened by a re-ID "new customer" verdict (ce_min_after_new_s)
    last_track: int | None = None                      # the track last seen on its seat (ce_staff_cut_check)
    staff_checked_end: float | None = None             # the sitting end the staff-haircut check last looked after
    frames: int = 0                                    # frames of its sittings (ce_low_occupancy) ...
    occ_frames: int = 0                                # ... those with the sitter on the seat ...
    frames_occ: int = 0                                # ... and the frame count at the last of them (the tail is the end gap)
    seen_until: float | None = None                    # the running sitting's last occupied frame, as note_frame saw it
    low: tuple[float, float, float] = (0.0, 0.0, 0.0)  # its chair's ce_low_occupancy, _short, _short_s

    @property
    def occupancy(self) -> float | None:
        """The share of its sittings' frames, up to the last occupied one, with the sitter on the seat."""
        return self.occ_frames / self.frames_occ if self.frames_occ else None

    @property
    def low_occupancy(self) -> bool:
        """Its occupancy is below ce_low_occupancy -- or below ce_low_occupancy_short while its sitting time (to the
        last occupied frame) is under ce_low_occupancy_short_s."""
        share, short, short_s = self.low
        if not self.frames_occ:
            return False
        sat = self.sitting_s(self.seen_until if self.seen_until is not None else self.sittings[-1][0])
        need = max(share, short) if short > 0 and sat < short_s else share
        return need > 0 and self.occ_frames < need * self.frames_occ

    @property
    def start(self) -> float:
        return self.sittings[0][0]

    @property
    def last_end(self) -> float | None:
        return self.sittings[-1][1]

    def sitting_s(self, now_ts: float) -> float:
        return sum((e if e is not None else now_ts) - s for s, e in self.sittings)

    def wash_s(self) -> float:
        return sum(self.sittings[i + 1][0] - self.sittings[i][1] for i in range(len(self.sittings) - 1))


@dataclass
class Pending:
    """A start waiting for the new sitter's re-ID (ce_resume_check)."""
    start: SitStart
    created: float                                     # when the start was seen (its ts may be backdated)
    samples: list[Sample] = field(default_factory=list)
    end: float | None = None
    reseat_from: float | None = None                   # a reseat check: the last frame the sitter was seen before


@dataclass(frozen=True)
class ResumeDecision:
    """A re-ID check's answer: the sitter at sittings[sitting_n] of haircut_id is the previous customer back
    (resumed) or a new one. sim None: no re-ID on one side, the time rule answered."""
    haircut_id: str
    sitting_n: int
    at: float                                          # when it was decided
    gap_s: float                                       # how long the chair had been left
    sim: float | None
    threshold: float | None                            # what sim had to reach
    resumed: bool
    prev_ticket: int | None                            # the previous customer's ticket
    kind: str = "return"                               # return | reseat


class ResumeDecider(Protocol):
    def resume(self, chair: str, last: Haircut, start_ts: float, policy: dict) -> bool: ...


class TimeRule:
    def resume(self, chair: str, last: Haircut, start_ts: float, policy: dict) -> bool:
        return last.last_end is not None and start_ts - last.last_end <= policy["ce_resume_window_s"]


def pedicure_policy(policy: dict) -> dict:
    """The policy a pedicure book runs on (2026-11-10): its own minimum (pd_min_s), a return within ce_resume_window_s
    continues the session by time (no re-ID, no reseat check)."""
    return {**policy, "ce_resume_check": 0.0, "ce_reseat_check_s": 0.0, "ce_min_haircut_s": policy["pd_min_s"],
            "ce_min_after_new_s": 0.0}


class HaircutBook:
    def __init__(self, new_id: Callable[[], str], barber_for: Callable[[str, datetime], str | None],
                 next_ticket: Callable[[datetime], int], decider: ResumeDecider | None = None, kind: str = "haircut"):
        self.new_id, self.barber_for, self.next_ticket = new_id, barber_for, next_ticket
        self.kind = kind                            # "haircut" | zones.PEDICURE (2026-11-10)
        self.decider = decider or TimeRule()
        self.open: dict[str, Haircut] = {}
        self.pending: dict[str, Pending] = {}       # chair -> start waiting for re-ID
        self.recent: dict[str, Haircut] = {}        # chair -> its last haircut, closed, still reopenable
        self.seat_samples: dict[str, list[Sample]] = {}   # chair -> sitter re-ID while no sitting runs there
        self.decisions: dict[str, ResumeDecision] = {}    # chair -> its last re-ID check's answer
        self.alerts = AlertBook(new_id)

    # ── events ──────────────────────────────────────────────────────────────
    def on_start(self, ev: SitStart, upd: EngineUpdate, policy: dict, now_ts: float | None = None) -> None:
        """now_ts: the frame that confirmed the start (ev.ts may be backdated by the start chain)."""
        if int(policy["ce_resume_check"]) == 1:
            if ev.chair in self.pending:
                self._resolve(ev.chair, upd, policy, ev.ts if now_ts is None else max(now_ts, ev.ts))
            seed = [r for r in self.seat_samples.pop(ev.chair, []) if r[0] >= ev.ts]
            last = self._last(ev.chair)
            if last is not None and ev.ts - last.last_end <= policy["ce_resume_max_s"]:
                self.pending[ev.chair] = Pending(ev, ev.ts if now_ts is None else max(now_ts, ev.ts), seed)
            else:
                self.decisions.pop(ev.chair, None)
                self._new(ev, upd, policy, seed)
            return
        h = self.open.get(ev.chair)
        if h is not None and h.last_end is not None and self.decider.resume(ev.chair, h, ev.ts, policy):
            h.sittings.append([ev.ts, None])
        else:
            if h is not None:
                self._close(h, upd, policy)
            h = self.open[ev.chair] = Haircut(self.new_id(), ev.chair, [[ev.ts, None]])
        if ev.track_id is not None:
            h.track_ids.add(ev.track_id)
        self._emit(h, upd, HAIRCUT)

    def on_end(self, ev: SitEnd, upd: EngineUpdate, policy: dict) -> None:
        if ev.chair in self.pending:                # ended before re-ID could answer: decide on what there is
            self.pending[ev.chair].end = ev.ts
            self._resolve(ev.chair, upd, policy, ev.ts)
            return
        self._end(self.open.get(ev.chair), ev.ts, upd, policy)

    def on_reseat(self, ev: Reseat, upd: EngineUpdate, policy: dict, now_ts: float) -> None:
        """The seat of a running sitting taken again after ev.ts - ev.last_occ with only cover: is it him?"""
        h = self.open.get(ev.chair)
        if (int(policy["ce_resume_check"]) != 1 or ev.chair in self.pending or h is None or h.last_end is not None):
            return
        new = [r for r in h.reid if r[0] > ev.last_occ]           # the sitter's samples since the stretch: not his
        h.reid = [r for r in h.reid if r[0] <= ev.last_occ]
        self.pending[ev.chair] = Pending(SitStart(ev.chair, ev.ts, ev.track_id), max(now_ts, ev.ts), new,
                                         reseat_from=ev.last_occ)

    def _end(self, h: Haircut | None, ts: float, upd: EngineUpdate, policy: dict) -> None:
        if h is None or h.last_end is not None:
            return
        h.sittings[-1][1] = ts
        h.frames = h.frames_occ                     # the frames after the last occupied one were the end gap
        h.seen_until = None
        if not h.confirmed and h.sitting_s(ts) >= self._min_s(h, policy) and self._a_person(h, policy):
            self._confirm(h, policy)
        self._emit(h, upd, WASHING)

    # ── ce_object_spread (2026-10-18), ce_object_conf (2026-10-22) ──────────
    BOXES_KEPT = 4000                       # an unconfirmed haircut's occupant boxes (minutes at the frame rate)

    def note_box(self, chair: str, box: tuple, conf: float | None = None) -> None:
        """The chair's occupant box (left, top, ...) in this frame as fractions of the frame, and its detection conf,
        for its open haircut until it is confirmed."""
        h = self.open.get(chair)
        if h is not None and not h.confirmed and h.last_end is None:
            h.boxes.append((float(box[0]), float(box[1]), float("nan") if conf is None else float(conf)))
            if len(h.boxes) > self.BOXES_KEPT:
                del h.boxes[:len(h.boxes) - self.BOXES_KEPT // 2]

    def note_frame(self, chair: str, ts: float, occupied: bool, policy: dict) -> None:
        """One frame of the chair: counted for its open haircut while a sitting runs (ce_low_occupancy)."""
        h = self.open.get(chair)
        if h is None or h.last_end is not None or chair in self.pending:
            return
        pc = policy_for(policy, chair)
        h.low = (pc["ce_low_occupancy"], pc["ce_low_occupancy_short"], pc["ce_low_occupancy_short_s"])
        h.frames += 1
        if occupied:
            h.occ_frames += 1
            h.frames_occ, h.seen_until = h.frames, ts

    @staticmethod
    def _min_s(h: Haircut, policy: dict) -> float:
        """The sitting time that makes h a haircut: ce_min_haircut_s, or ce_min_after_new_s if longer when re-ID called
        its sitter a new customer (a returning customer it failed to recognise must sit that long to count twice)."""
        m = policy["ce_min_haircut_s"]
        return max(m, policy["ce_min_after_new_s"]) if h.after_new else m

    @staticmethod
    def _a_person(h: Haircut, policy: dict) -> bool:
        """A person moves; a bag on the seat does not: the occupant's top or left edge spread (p90 - p10) above
        ce_object_spread. And the detector is sure of a person, of a pile of clothes only half: the median conf of
        his boxes at least ce_object_conf. Each off (0), or nothing seen to judge by (an older feed): a person, as
        before."""
        if len(h.boxes) < 2:
            return True
        pc = policy_for(policy, h.chair)
        a = np.asarray(h.boxes, dtype=float)
        lim = pc["ce_object_spread"]
        if lim > 0 and not (np.percentile(a[:, :2], 90, axis=0) - np.percentile(a[:, :2], 10, axis=0) > lim).any():
            return False
        conf = a[:, 2][~np.isnan(a[:, 2])]
        return pc["ce_object_conf"] <= 0 or not len(conf) or float(np.median(conf)) >= pc["ce_object_conf"]

    # ── ce_resume_check ─────────────────────────────────────────────────────
    def _last(self, chair: str) -> Haircut | None:
        """The chair's previous customer: its open confirmed haircut while he is away, else its recently
        closed one, else an open unconfirmed one (a sit too short to be a haircut yet)."""
        h = self.open.get(chair)
        if h is not None and h.last_end is None:
            return None
        if h is not None and h.confirmed:
            return h
        return self.recent.get(chair) or h

    def _new(self, ev: SitStart, upd: EngineUpdate, policy: dict, samples: list[Sample] | None = None) -> Haircut:
        old = self.open.get(ev.chair)
        if old is not None:
            self._close(old, upd, policy)
        h = self.open[ev.chair] = Haircut(self.new_id(), ev.chair, [[ev.ts, None]], reid=list(samples or []))
        if ev.track_id is not None:
            h.track_ids.add(ev.track_id)
        self._emit(h, upd, HAIRCUT)
        return h

    def _resolve(self, chair: str, upd: EngineUpdate, policy: dict, at: float) -> None:
        p = self.pending.pop(chair)
        pc = policy_for(policy, chair)
        if p.reseat_from is not None:
            h = self._reseat(chair, p, upd, pc, at)
            if h is not None and p.end is not None:
                self._end(h, p.end, upd, policy)
            return
        last = self._last(chair)
        if last is None:                            # closed under it (clear): nothing to resume
            self.decisions.pop(chair, None)
            h = self._new(p.start, upd, policy, p.samples)
        else:
            gap = p.start.ts - last.last_end
            ref = ref_window(last.reid, last.sittings[-1][0], last.last_end, policy["ce_resume_ref_s"])
            sim = similarity(ref, p.samples)
            same = decide(gap, sim, pc)
            logger.info("chair %s: sitter at %s after %.0f s, re-ID %s vs %s (%d/%d samples) -> %s", chair,
                        _hms(p.start.ts), gap, "none" if sim is None else f"{sim:.2f}", last.id, len(ref),
                        len(p.samples), "resume" if same else "new customer")
            if same:
                h = last
                if self.recent.get(chair) is h:     # closed at ce_resume_window_s: reopen it
                    stray = self.open.get(chair)
                    if stray is not None:           # a sit too short to be a haircut came in between
                        self._close(stray, upd, policy)
                    del self.recent[chair]
                    self.open[chair] = h
                    h.alert_from = p.start.ts
                h.sittings.append([p.start.ts, None])
                h.reid += p.samples
                if p.start.track_id is not None:
                    h.track_ids.add(p.start.track_id)
                self._emit(h, upd, HAIRCUT)
            else:
                h = self._new(p.start, upd, policy, p.samples)
                h.after_new = True
            self._record(chair, ResumeDecision(h.id, len(h.sittings) - 1, at, gap, sim, threshold(gap, pc), same,
                                               last.ticket), p, last, len(ref), upd)
        if p.end is not None:
            self._end(h, p.end, upd, policy)

    def _reseat(self, chair: str, p: Pending, upd: EngineUpdate, pc: dict, at: float) -> Haircut | None:
        h = self.open.get(chair)
        if h is None or h.last_end is not None:     # ended under it: nothing to split
            return h
        gap = p.start.ts - p.reseat_from
        ref = ref_window(h.reid, h.sittings[-1][0], p.reseat_from, pc["ce_resume_ref_s"])
        sim = similarity(ref, p.samples)
        same = decide(gap, sim, pc)
        logger.info("chair %s: seat taken again at %s after %.0f s of cover only, re-ID %s vs %s (%d/%d samples) -> %s",
                    chair, _hms(p.start.ts), gap, "none" if sim is None else f"{sim:.2f}", h.id, len(ref),
                    len(p.samples), "same customer" if same else "new customer")
        prev = h
        if same:
            h.reid += p.samples
        else:                                       # his haircut ended when he was last seen; the sitter's starts now
            self._end(h, p.reseat_from, upd, pc)
            h = self._new(p.start, upd, pc, p.samples)
            h.after_new = True
        self._record(chair, ResumeDecision(h.id, len(h.sittings) - 1, at, gap, sim, threshold(gap, pc), same,
                                           prev.ticket, kind="reseat"), p, prev, len(ref), upd)
        return h

    def _record(self, chair: str, d: ResumeDecision, p: Pending, prev: Haircut, n_ref: int, upd: EngineUpdate) -> None:
        self.decisions[chair] = d
        upd.resume_checks.append(ResumeCheckRecord(chair, d.kind, dt(p.start.ts), dt(d.at), d.gap_s, d.sim,
                                                   d.threshold, d.resumed, d.haircut_id, prev.id, n_ref,
                                                   len(p.samples)))

    def tick(self, now_ts: float, sitting_now: Callable[[str], bool], upd: EngineUpdate, policy: dict,
             occupied_until: Callable[[str], float | None]) -> None:
        """sitting_now(chair): a sitting is running or starting there (a starting one may resume the
        haircut, so the resume window must not close it under it). occupied_until(chair): the chair's
        last occupied frame while a sitting runs (None: none running) -- the open sitting's time runs
        to it, not to now (2026-10-04, final review I-1: the end gap counted as sitting time)."""
        for chair, p in list(self.pending.items()):
            new_s = policy["ce_resume_new_s"]
            if (len(p.samples) >= 2 and now_ts >= p.start.ts + new_s) or now_ts >= p.created + new_s + WAIT_EXTRA_S:
                self._resolve(chair, upd, policy, now_ts)
        for chair, h in list(self.recent.items()):
            if chair not in self.pending and now_ts - h.last_end > policy["ce_resume_max_s"]:
                del self.recent[chair]
        for chair, h in list(self.open.items()):
            if not h.confirmed:
                until = occupied_until(chair) if h.last_end is None else None
                if (h.sitting_s(until if until is not None else h.sittings[-1][0]) >= self._min_s(h, policy)
                        and self._a_person(h, policy)):
                    self._confirm(h, policy)
                    self._emit(h, upd, HAIRCUT if h.last_end is None else WASHING)
            seen = occupied_until(chair) if h.confirmed and h.last_end is None else None
            if seen is not None and h.service_target_s is not None:     # in the chair and seen there: the clock runs
                since = h.start if h.alert_from is None else max(h.start, h.alert_from - h.service_target_s)
                a = self.alerts.check(EXCESSIVE_SERVICE, h.id, camera=SHOP, visit_id=h.id, session_id=h.id,
                                      since=dt(since), target_s=h.service_target_s, now=dt(seen))
                if a is not None:
                    h.excessive = True
                    upd.alerts.append(a)
                    self._emit(h, upd, HAIRCUT if h.last_end is None else WASHING)
            if h.last_end is not None and not sitting_now(chair) and now_ts - h.last_end > policy["ce_resume_window_s"]:
                self._close(h, upd, policy)

    def on_appearance(self, app: Appearance, chair: str | None, policy: dict, upd: EngineUpdate) -> None:
        """chair: the chair this appearance's track occupies right now (None: not a sitter)."""
        if self.kind == PEDICURE:                   # a pedicure keeps no re-ID samples and no pictures
            if app.crop_path is not None:
                upd.discard_crops.append(app.crop_path)
            return
        if (int(policy["ce_resume_check"]) == 1 and chair is not None and app.reid is not None
                and (app.staff_p is None or app.staff_p < policy["staff_threshold"])):
            p = self.pending.get(chair)
            h = self.open.get(chair)
            keep = max(KEEP_S, policy["ce_resume_ref_s"])
            buf = (p.samples if p is not None else h.reid if h is not None and h.last_end is None
                   else self.seat_samples.setdefault(chair, []))        # starting: seeds the next pending
            buf.append((app.ts, app.reid))
            if p is None:
                while buf and buf[0][0] < app.ts - keep:
                    buf.pop(0)
        if app.crop_path is None:
            return
        h = self.open.get(chair) if chair is not None else None
        ok = (int(policy["register_source"]) in (0, 2) and h is not None and app.track_id in h.track_ids
              and app.ts - h.start <= policy["ce_picture_window_s"]
              and (app.staff_p is None or app.staff_p < policy["staff_threshold"]))
        if not ok:
            upd.discard_crops.append(app.crop_path)
            return
        h.bodies.append(BodyCandidate(app.camera, app.track_id, app.ts, app.crop_path, app.reid,
                                      float(app.crop_quality or 0.0)))
        keep = int(policy["customer_max_bodies"])
        if len(h.bodies) > keep:
            h.bodies.sort(key=lambda b: b.quality, reverse=True)
            upd.discard_crops += [b.crop_path for b in h.bodies[keep:]]
            del h.bodies[keep:]

    def add_face(self, chair: str, face: FaceCandidate, policy: dict, upd: EngineUpdate) -> None:
        """register_source 2: a face of chair's seated customer (chosen by place in chair_engine) -- the best
        visit_max_faces are kept for his card."""
        h = self.open.get(chair)
        if h is None or h.registered:       # none running, or its card is in already: no top-up
            upd.discard_crops.append(face.crop_path)
            return
        h.faces.append(face)
        keep = int(policy["visit_max_faces"])
        if len(h.faces) > keep:
            h.faces.sort(key=lambda f: f.quality, reverse=True)
            upd.discard_crops += [f.crop_path for f in h.faces[keep:]]
            del h.faces[keep:]

    def void_staff(self, h: Haircut, upd: EngineUpdate) -> None:
        """A staff member's own haircut (2026-11-10, ce_staff_cut_check): not a customer. Its session is deleted, its visit
        voided as staff, its photos dropped, its pictures discarded, its alert closed; it is never registered."""
        self.open.pop(h.chair, None)
        if self.recent.get(h.chair) is h:
            del self.recent[h.chair]
        end = dt(h.last_end if h.last_end is not None else h.sittings[-1][0])
        upd.sessions.pop(h.id, None)
        upd.deleted_sessions.add(h.id)
        v = Visit(id=h.id, person_id=h.id, entered_at=dt(h.start), entry_observed=False, state=FINISHED, confirmed=True,
                  service_started_at=dt(h.start), haircuts=0, ticket=h.ticket, wait_unseen=True)
        v.exited_at, v.outcome, v.open = end, VOID_STAFF, False
        upd.visits[h.id] = v
        a = self.alerts.close(EXCESSIVE_SERVICE, h.id, end)
        if a is not None:
            upd.alerts.append(a)
        upd.dropped_photos.add(h.id)
        upd.discard_crops += [b.crop_path for b in h.bodies] + [f.crop_path for f in h.faces]
        h.bodies.clear()
        h.faces.clear()

    def clear(self) -> list[str]:
        paths = [b.crop_path for h in self.open.values() for b in h.bodies]
        paths += [f.crop_path for h in self.open.values() for f in h.faces]
        self.open, self.pending, self.recent, self.seat_samples, self.decisions = {}, {}, {}, {}, {}
        self.alerts.clear()
        return paths

    def forget(self, chairs: set[str]) -> None:
        """Chairs removed from the zones: drop their pending checks, reopenable haircuts, buffers and decisions."""
        for d in (self.pending, self.recent, self.seat_samples, self.decisions):
            for chair in [c for c in d if c in chairs]:
                del d[chair]

    # ── records ─────────────────────────────────────────────────────────────
    def _confirm(self, h: Haircut, policy: dict) -> None:
        if self.recent.get(h.chair) is not None and self.recent[h.chair] is not h:
            del self.recent[h.chair]            # a new haircut in that chair: the previous one is not his any more
        h.confirmed = True
        h.boxes = []                            # ce_object_spread / ce_object_conf: no longer needed
        if self.kind == PEDICURE:                   # no ticket, no barber, no excessive-haircut clock
            return
        h.service_target_s = policy["service_target_s"]
        h.ticket = self.next_ticket(dt(h.start))
        h.barber_id = self.barber_for(h.chair, dt(h.start))

    def _emit(self, h: Haircut, upd: EngineUpdate, state: str) -> None:
        if not h.confirmed:
            return
        if self.kind == PEDICURE:
            upd.pedicures[h.id] = PedicureRecord(h.id, h.chair, dt(h.start))
            return
        start = dt(h.start)
        upd.visits[h.id] = Visit(id=h.id, person_id=h.id, entered_at=start, entry_observed=False, state=state,
                                 confirmed=True, service_started_at=start, haircuts=1, ticket=h.ticket,
                                 wait_unseen=True)
        upd.sessions[h.id] = ServiceSession(id=h.id, visit_id=h.id, chair_id=h.chair, started_at=start,
                                            wash_s=h.wash_s(), barber_id=h.barber_id,
                                            service_target_s=h.service_target_s, excessive_service=h.excessive,
                                            occupancy=h.occupancy, low_occupancy=h.low_occupancy)
        upd.sittings += [SittingRecord(h.id, i, h.chair, dt(s), dt(e) if e is not None else None)
                         for i, (s, e) in enumerate(h.sittings)]

    def _close(self, h: Haircut, upd: EngineUpdate, policy: dict) -> None:
        self.open.pop(h.chair, None)
        if not h.confirmed:
            upd.discard_crops += [b.crop_path for b in h.bodies] + [f.crop_path for f in h.faces]
            return
        end_ts = h.last_end if h.last_end is not None else h.sittings[-1][0]
        if self.kind == PEDICURE:
            upd.pedicures[h.id] = PedicureRecord(h.id, h.chair, dt(h.start), dt(end_ts), h.sitting_s(end_ts), False)
            return
        self._emit(h, upd, FINISHED)
        v, s = upd.visits[h.id], upd.sessions[h.id]
        v.exited_at, v.outcome, v.open = dt(end_ts), COMPLETED, False
        s.ended_at, s.service_s, s.wash_s, s.open = dt(end_ts), h.sitting_s(end_ts), h.wash_s(), False
        a = self.alerts.close(EXCESSIVE_SERVICE, h.id, dt(end_ts))
        if a is not None:
            upd.alerts.append(a)
        if int(policy["ce_resume_check"]) == 1:
            self.recent[h.chair] = h            # reopenable until ce_resume_max_s
        if h.registered:                        # reopened (ce_resume_check): its card is in already
            pass
        elif int(policy["register_source"]) in (0, 2):
            h.registered = True
            upd.registrations.append(Registration(h.id, dt(end_ts), tuple(h.faces), tuple(h.bodies)))
            h.faces.clear()                 # the registration owns those crops now
        else:                           # collected under 0/2, closed after the switch went to 1: no card, no stray files
            upd.discard_crops += [b.crop_path for b in h.bodies] + [f.crop_path for f in h.faces]
            h.bodies.clear()
            h.faces.clear()


def _hms(ts: float) -> str:
    return dt(ts).strftime("%H:%M:%S")
