"""Customer waits for the chair-first engine (project 4 A + B, 2026-10-05; spec
docs/superpowers/specs/2026-10-05-barber-waits-design.md).

Centre camera only (the one that draws the entrance line). A non-staff track whose lower mask has stayed in a waiting
zone wt_start_s (the "Waiting now" test; gaps <= wc_dwell_gap_s) starts a wait dated back to its first frame in the
area, if the median detection confidence over that dwell reaches wt_conf_min (a bag on the sofa is not a customer).
The wait then follows that track anywhere in view and ends:
  * chair -- the track's lower mask holds a `row` zone (>= wt_row_min inside) for wt_row_s: dated to its first frame there;
  * door  -- the door counter confirms a non-staff `out` crossing of the track: at the crossing;
  * lost  -- the track was unseen wt_lost_s, or turned staff (an id swap must not drop a customer's wait: user,
             2026-10-05), so the wait went on hold, and neither the track came back nor a track first seen after
             about that moment (from wt_lost_s before the last sighting) near the last sighting (foot point within
             wt_reconnect_dist of the frame width) took it over within wt_reconnect_s: at the last sighting.
Every wait counts in the average (user, 2026-10-05: companions included); waits are not paired with haircuts.
2026-10-07 (review finding 3: 49 waits for 30 visitors on one lap, 43 ended lost, 22 under a minute):
  * a holder turns staff only after wt_staff_samples staff samples in a row (one 0.98 among 0.01s was enough);
  * a wait ended lost is rejoined -- same id, ticket and start -- by one of its own tracks back within wt_rejoin_s;
  * a track first seen on the street side of the entrance line never takes over a held wait (a newcomer);
  * no excessive_wait alert while the wait is on hold; a rejoined wait's alert starts no earlier than the rejoin;
  * the centre camera silent wt_stale_s (engine clock): open waits end lost at their last sighting.
2026-10-14 (Palm Hills, no door camera): wt_end_mode 1 ends a wait at wt_out_s outside every waiting zone (`left`,
dated to the first frame outside); a return reopens it -- his own track, or a track whose mean re-ID reaches
wt_return_sim -- and a return after a served wait (a haircut started within wt_serve_s, its customer out of the chair
again) is no wait at all. Waits run on `wait_camera` when the site config sets one.
2026-10-15 (final review I2, mode 1): a match whose linking haircut is still sitting reopens the wait (he can't be
on the sofa and in that chair: it was someone else's haircut), and the waits a wait-camera stall ends (`stall`) are
never served.
Positions are box bottom-centres in frame-width units. Time is the centre camera's own frame clock, so a stalled
centre camera holds, ends and alerts nothing."""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from alerts import EXCESSIVE_WAIT, AlertBook
from domain import SHOP, Appearance, DoorCrossing, EngineUpdate, Frame, TrackObs, WaitRecord, dt
from masks import raster
from resume_check import mean_unit
from wait_links import STALL, link_waits
from waiting import lower_share_in, zones_of_type
from zones import ENTRANCE_LINE, ROW, WAITING, WASHING, DoorLine, FrameGeometry, area_of

log = logging.getLogger("barber.waits")

CHAIR, DOOR, LEFT, LOST, RESTART, WASH = "chair", "door", "left", "lost", "restart", "wash"
TRACK_TTL_S = 600.0     # per-track state unseen this long is dropped
PRUNE_EVERY_S = 60.0
REID_KEEP_S = 600.0     # a track's re-ID samples kept this long (a dwell is far shorter)
REFRESH_S = 30.0        # an open wait's record (last_seen_at) is re-sent this often


@dataclass
class _Dwell:
    since: float
    last_in: float
    confs: list[tuple[float, float]] = field(default_factory=list)    # (ts, conf) of the last wt_start_s of the run


@dataclass
class _Seen:
    first: float
    first_pos: tuple[float, float]
    last: float
    first_out: bool = False         # first seen on the street side of the entrance line


@dataclass
class Wait:
    id: str
    ticket: int
    first_track: int
    track: int                      # the track holding it (the last one, while on hold)
    start: float
    target_s: float
    last_seen: float
    pos: tuple[float, float]        # foot point at the last sighting
    hold: str | None = None         # None | "lost" | "staff"
    tracks: list[int] = field(default_factory=list)    # every track that held it, oldest first
    row_since: float | None = None
    row_last: float | None = None
    written: float = 0.0            # centre-camera ts of the last record sent
    box: tuple[float, float, float, float] | None = None    # holder's latest normalised x, y, w, h
    alert_from: float | None = None  # rejoined at: its excessive alert starts no earlier
    out_since: float | None = None   # wt_end_mode 1: first frame of the current run outside every waiting zone
    reid: list[tuple[float, tuple[float, ...]]] = field(default_factory=list)    # (ts, embedding) samples
    area: str | None = None          # the waiting zone (area id) it started in


@dataclass
class _Gone:
    """wt_end_mode 1: a wait that ended left or lost, which a returning customer may reopen."""
    wait: Wait
    end: float


class WaitBook:
    def __init__(self, zones_by_cam: dict[str, list[dict]], camera: str | None, geometry: FrameGeometry,
                 new_id: Callable[[], str], next_ticket: Callable[[datetime], int],
                 sitting: Callable[[str], bool] | None = None):
        """sitting(session id): that haircut's customer is still in the chair (its last sitting hasn't ended);
        None = never (a book that sees no haircuts)."""
        self.geometry = geometry
        self._new_id, self._next_ticket = new_id, next_ticket
        self._sitting = sitting or (lambda session_id, since: False)
        self.set_zones(zones_by_cam, camera)
        self.clear()

    def set_zones(self, zones_by_cam: dict[str, list[dict]], camera: str | None) -> None:
        """New zones (a dashboard save): open waits carry on."""
        self.camera = camera
        wz = [(area_of(z), raster(z.get("points"))) for z in zones_by_cam.get(camera, []) if z.get("type") == WAITING] \
            if camera else []
        wz = [(a, r) for a, r in wz if r is not None and r.area > 0]
        self.zones, self.zone_areas = [r for _, r in wz], [a for a, _ in wz]
        self.rows = zones_of_type(zones_by_cam, ROW).get(camera, []) if camera else []
        self.washes = zones_of_type(zones_by_cam, WASHING).get(camera, []) if camera else []
        line = next((z for z in zones_by_cam.get(camera, []) if z.get("type") == ENTRANCE_LINE), None) if camera else None
        self.line = DoorLine.from_zone(line) if line else None
        no_row = camera is not None and self.line is not None and not self.rows
        if no_row and not getattr(self, "_warned_no_row", False):
            log.warning("no row zone on %s: waits can end only by the door or loss", camera)
        self._warned_no_row = no_row

    def clear(self) -> None:
        """Forget everything (a replay lap; its rows are cleared with it, a restart closes leftovers)."""
        self.open: dict[str, Wait] = {}
        self._dwell: dict[int, _Dwell] = {}
        self._seen: dict[int, _Seen] = {}
        self._ended: dict[int, tuple[Wait, float]] = {}    # track -> (its wait ended lost, at): rejoinable
        self.alerts = AlertBook(self._new_id)
        self.cam_ts: float | None = None
        self._pruned_at: float | None = None
        self._gone: dict[str, _Gone] = {}
        self._ends: dict[str, float] = {}                  # wait id -> end ts, every ended wait
        self._edge_mode = False
        self._reid: dict[int, list[tuple[float, tuple[float, ...]]]] = {}    # wait-camera track -> re-ID samples
        self._starts: dict[str, float] = {}                # haircut session id -> start ts
        self._after: dict[int, float] = {}                 # tracks judged back after their haircut -> last judged

    def on_appearance(self, app: Appearance) -> None:
        """Re-ID samples of the wait camera's tracks (wt_end_mode 1 returns); a holder's join its wait."""
        if not self._edge_mode or app.camera != self.camera or app.reid is None:
            return                                      # mode 0 never reads them
        buf = self._reid.setdefault(app.track_id, [])
        buf.append((app.ts, app.reid))
        while buf and buf[0][0] < app.ts - REID_KEEP_S:
            buf.pop(0)
        w = self.holder(app.track_id)
        if w is not None:
            w.reid.append((app.ts, app.reid))
            while w.reid[0][0] < app.ts - REID_KEEP_S:
                w.reid.pop(0)

    def note_haircut(self, session_id: str, start_ts: float) -> None:
        self._starts[session_id] = start_ts

    def drop_haircut(self, session_id: str) -> None:
        self._starts.pop(session_id, None)

    def _served_by(self, wait_id: str, policy: dict) -> str | None:
        return link_waits(self._ends.items(), self._starts.items(), policy["wt_serve_s"]).get(wait_id)

    def _best_return(self, samples, ts: float, policy: dict) -> tuple[_Gone | None, float | None]:
        mine = mean_unit(samples)
        best, best_sim = None, None
        if mine is None:
            return None, None
        for g in self._gone.values():
            if ts - g.end > policy["wt_return_max_s"]:
                continue
            ref = mean_unit(g.wait.reid)
            if ref is None:
                continue
            sim = float(mine @ ref)
            if sim >= policy["wt_return_sim"] and (best_sim is None or sim > best_sim):
                best, best_sim = g, sim
        return best, best_sim

    def _start_or_return(self, t: TrackObs, d: _Dwell, ts: float, policy: dict, upd: EngineUpdate) -> None:
        """wt_end_mode 1: a dwell long enough to start a wait is first matched against the waits that ended (left or
        lost) within wt_return_max_s -- by its own track, else by the mean re-ID of the dwell >= wt_return_sim (strict;
        the best match). A match reopens that wait (same id, ticket and start: the time away counts), unless that wait
        was served (a haircut started within wt_serve_s of its end) and that haircut's sitting is over: then he is
        back after his haircut -- no wait. A haircut still sitting is someone else's: the wait reopens (and so
        drops out of the links).
        Too few re-ID samples: wait up to wt_return_extra_s more; still too few = a new wait. Nothing to compare
        against (no ended wait with re-ID samples): start at once."""
        tid = t.track_id
        match = next((g for g in self._gone.values()
                      if tid in g.wait.tracks and ts - g.end <= policy["wt_return_max_s"]), None)
        sim = None
        samples = [s for s in self._reid.get(tid, []) if s[0] >= d.since]
        if match is None and not any(g.wait.reid and ts - g.end <= policy["wt_return_max_s"]
                                     for g in self._gone.values()):
            self._start(t, d.since, ts, policy, upd, samples)
            return
        if match is None:
            if len(samples) < int(policy["wt_return_min_samples"]):
                if ts - d.since < policy["wt_start_s"] + policy["wt_return_extra_s"]:
                    return                              # undecided: the start stays dated to d.since
                self._start(t, d.since, ts, policy, upd, samples)    # still too few: a new wait, no comparison
                return
            match, sim = self._best_return(samples, ts, policy)
        if match is None:
            self._start(t, d.since, ts, policy, upd, samples)
            return
        how = "own track" if sim is None else f"re-ID {sim:.2f}"
        sid = self._served_by(match.wait.id, policy)
        if sid is not None and not self._sitting(sid, d.since):
            self._dwell.pop(tid, None)
            self._after[tid] = ts
            log.info("track %d is wait %s (#%s) back after his haircut (%s): no new wait", tid, match.wait.id,
                     match.wait.ticket, how)
            return
        if sid is not None:
            how += f"; haircut {sid} still sitting, not his"
        del self._gone[match.wait.id]
        match.wait.reid += samples
        self._rejoin(match.wait, t, ts, upd, f" (reopened, {how})")

    def open_in(self, area: str) -> int:
        """Open waits (on hold included) that started in that waiting area."""
        return sum(1 for w in self.open.values() if w.area == area)

    def holder(self, track_id: int) -> Wait | None:
        return next((w for w in self.open.values() if w.track == track_id and w.hold is None), None)

    def _owner(self, track_id: int) -> Wait | None:
        return next((w for w in self.open.values() if w.track == track_id), None)

    def _former(self, track_id: int) -> Wait | None:
        """An open wait this track holds or held before (not one it swapped onto staff from)."""
        return next((w for w in self.open.values() if track_id in w.tracks), None)

    def _foot(self, t: TrackObs) -> tuple[float, float]:
        l, top, w, h = t.box
        return ((l + w / 2) / self.geometry.width, (top + h) / self.geometry.width)

    def _nbox(self, t: TrackObs) -> tuple[float, float, float, float]:
        l, top, w, h = t.box
        W, H = self.geometry.width, self.geometry.height
        return (round(l / W, 4), round(top / H, 4), round(w / W, 4), round(h / H, 4))

    # ── frames ──────────────────────────────────────────────────────────────
    def observe(self, frame: Frame, staff_of: Callable[[int], float | None], seated: set[int],
                crossings: list[DoorCrossing], policy: dict, upd: EngineUpdate,
                staff_run: Callable[[int], int] | None = None) -> None:
        """staff_run(track): its latest staff samples >= staff_threshold in a row (None: the latest one only)."""
        if frame.camera != self.camera:
            return
        ts = self.cam_ts = frame.ts
        edge = self._edge_mode = int(policy["wt_end_mode"]) == 1
        thr = policy["staff_threshold"]
        n_staff = int(policy["wt_staff_samples"])

        def is_staff(tid: int) -> bool:
            sp = staff_of(tid)
            return sp is not None and sp >= thr

        run = staff_run if staff_run is not None else (lambda tid: n_staff if is_staff(tid) else 0)

        for t in frame.tracks:
            s = self._seen.get(t.track_id)
            if s is None:
                out = self.line is not None and self.line.signed(self.geometry.foot(t.box)) < 0
                self._seen[t.track_id] = _Seen(ts, self._foot(t), ts, out)
            else:
                s.last = ts
        self._reconnect(frame, is_staff, policy)
        for t in frame.tracks:
            tid = t.track_id
            w = self._owner(tid)
            if w is not None:
                if w.hold == "staff":
                    continue                        # the track that swapped onto staff never holds it again
                if run(tid) >= n_staff:         # a swap onto staff, not one odd sample
                    w.hold = "staff"                # on hold from its last non-staff frame (last_seen unchanged)
                    if tid in w.tracks:
                        w.tracks.remove(tid)    # and it never resumes the wait
                    continue
                w.hold, w.last_seen, w.pos = None, ts, self._foot(t)
                w.box = self._nbox(t)
                if edge:
                    self._edge(w, t, ts, policy, upd)
                else:
                    self._row(w, t, ts, policy, upd)
                continue
            back = self._ended.get(tid)
            if back is not None and ts - back[1] <= policy["wt_rejoin_s"] and not is_staff(tid) and tid not in seated:
                self._rejoin(back[0], t, ts, upd)
                continue
            if self._former(tid) is not None or tid in self._after:
                continue                            # an earlier holder of an open wait never starts another
            if is_staff(tid) or tid in seated or t.conf is None or t.conf < policy["ce_conf_min"] \
                    or not lower_share_in(t.outline, self.zones, policy["wc_lower_frac"], policy["wc_zone_min"]):
                continue
            d = self._dwell.get(tid)
            if d is None or ts - d.last_in > policy["wc_dwell_gap_s"]:
                d = self._dwell[tid] = _Dwell(ts, ts)
            d.last_in = ts
            d.confs.append((ts, t.conf))
            cut = ts - policy["wt_start_s"]
            while d.confs[0][0] < cut:
                d.confs.pop(0)
            if ts - d.since >= policy["wt_start_s"]:
                if statistics.median(c for _, c in d.confs) >= policy["wt_conf_min"]:
                    if edge:
                        self._start_or_return(t, d, ts, policy, upd)
                    else:
                        self._start(t, d.since, ts, policy, upd)
                else:
                    d.since = d.confs[0][0]     # not confident yet: date a later start from the window, not the whole dwell
        for c in crossings:
            if c.camera == self.camera and c.direction == "out" and not c.staff:
                w = self._former(c.track_id)
                if w is not None:
                    self._end(w, c.at.timestamp(), DOOR, upd)

    def _reconnect(self, frame: Frame, is_staff, policy: dict) -> None:
        ts = frame.ts
        present = {t.track_id for t in frame.tracks if not is_staff(t.track_id)}
        held = [w for w in self.open.values() if w.hold is not None]
        busy = {w.track for w in self.open.values() if w.hold is None}
        # an earlier holder coming back beats any newcomer (only while the wait is on hold)
        for w in held:
            back = [tid for tid in w.tracks if tid in present and tid not in busy]
            if not back:
                continue
            tid = w.track if w.track in back else back[-1]
            w.track, w.hold = tid, None
            w.tracks.remove(tid)
            w.tracks.append(tid)
            w.row_since = w.row_last = None
            busy.add(tid)
        held = [w for w in self.open.values() if w.hold is not None]
        if not held:
            return
        cands = []
        for t in frame.tracks:
            if t.track_id in busy or is_staff(t.track_id) or self._former(t.track_id) is not None:
                continue
            s = self._seen[t.track_id]
            if s.first_out:                         # came in from the street: a newcomer, not the lost customer
                continue
            for w in held:
                if s.first < w.last_seen - policy["wt_lost_s"] or ts - w.last_seen > policy["wt_reconnect_s"]:
                    continue
                d = math.dist(s.first_pos, w.pos)
                if d <= policy["wt_reconnect_dist"]:
                    cands.append((d, t.track_id, w))
        used_t, used_w = set(), set()
        for d, tid, w in sorted(cands, key=lambda c: c[0]):
            if tid in used_t or w.id in used_w:
                continue
            used_t.add(tid)
            used_w.add(w.id)
            w.track, w.hold = tid, None             # the main loop refreshes last_seen / pos this frame
            w.tracks.append(tid)
            w.row_since = w.row_last = None
            self._dwell.pop(tid, None)

    def _edge(self, w: Wait, t: TrackObs, ts: float, policy: dict, upd: EngineUpdate) -> None:
        """wt_end_mode 1 (2026-10-14, user): out of every waiting zone for wt_out_s ends the wait, dated to his first
        frame outside; any frame back inside restarts the count (shifting on the sofa's edge never ends it)."""
        if lower_share_in(t.outline, self.zones, policy["wc_lower_frac"], policy["wc_zone_min"]):
            w.out_since = None
            return
        if w.out_since is None:
            w.out_since = ts
        if ts - w.out_since >= policy["wt_out_s"]:
            self._end(w, w.out_since, LEFT, upd)

    def _row(self, w: Wait, t: TrackObs, ts: float, policy: dict, upd: EngineUpdate) -> None:
        """A row zone, or the wash (2026-10-12, user: some haircuts start with a wash) -- but not the wash
        while he is still in a waiting zone: a wash zone drawn over the sofa must not end a seated wait."""
        frac, need = policy["wc_lower_frac"], policy["wt_row_min"]
        if lower_share_in(t.outline, self.rows, frac, need):
            why = CHAIR
        elif (lower_share_in(t.outline, self.washes, frac, need)
              and not lower_share_in(t.outline, self.zones, frac, need)):
            why = WASH
        else:
            return
        if w.row_since is None or ts - w.row_last > policy["wc_dwell_gap_s"]:
            w.row_since = ts
        w.row_last = ts
        if ts - w.row_since >= policy["wt_row_s"]:
            self._end(w, w.row_since, why, upd)

    # ── clock ───────────────────────────────────────────────────────────────
    def tick(self, policy: dict, upd: EngineUpdate) -> None:
        ts = self.cam_ts
        if ts is None:
            return
        self._edge_mode = int(policy["wt_end_mode"]) == 1
        for w in list(self.open.values()):
            if w.hold is None and ts - w.last_seen >= policy["wt_lost_s"]:
                w.hold = "lost"
            if w.hold is not None and ts - w.last_seen > policy["wt_reconnect_s"]:
                self._end(w, w.last_seen, LOST, upd)
                continue
            if w.hold is None:                      # on hold: not seen waiting, no alert from it
                since = w.start if w.alert_from is None else max(w.start, w.alert_from - w.target_s)
                a = self.alerts.check(EXCESSIVE_WAIT, w.id, camera=SHOP, visit_id=w.id, session_id=None,
                                      since=dt(since), target_s=w.target_s, now=dt(ts))
                if a is not None:
                    upd.alerts.append(a)
            if ts - w.written >= REFRESH_S:
                self._write(w, upd, ts)
        if self._pruned_at is None or ts - self._pruned_at >= PRUNE_EVERY_S:
            self._pruned_at = ts
            for tid in [k for k, d in self._dwell.items() if ts - d.last_in > TRACK_TTL_S]:
                del self._dwell[tid]
            for tid in [k for k, s in self._seen.items() if ts - s.last > TRACK_TTL_S]:
                del self._seen[tid]
            for tid in [k for k, (_, at) in self._ended.items() if ts - at > policy["wt_rejoin_s"]]:
                del self._ended[tid]
            for wid in [k for k, g in self._gone.items() if ts - g.end > policy["wt_return_max_s"]]:
                del self._gone[wid]
            for tid in [k for k, b in self._reid.items() if not b or ts - b[-1][0] > TRACK_TTL_S]:
                del self._reid[tid]
            for tid in [k for k in self._after if k not in self._seen]:
                del self._after[tid]
            horizon = policy["wt_return_max_s"] + policy["wt_serve_s"] + 3600
            for sid in [k for k, at in self._starts.items() if ts - at > horizon]:
                del self._starts[sid]
            for wid in [k for k, at in self._ends.items() if ts - at > horizon]:
                del self._ends[wid]

    def stale(self, now_ts: float, policy: dict, upd: EngineUpdate) -> None:
        """The centre camera silent wt_stale_s by the engine's clock: nobody can be seen waiting -- end them
        at their last sighting (rejoinable if it comes back in time) instead of aging them on the dashboard.
        Mode 0 ends them `lost`; mode 1 `stall` (reopenable, never served: wait_links)."""
        if self.cam_ts is None or now_ts - self.cam_ts <= policy["wt_stale_s"]:
            return
        self._edge_mode = int(policy["wt_end_mode"]) == 1
        for w in list(self.open.values()):
            log.info("wait %s (#%s): centre camera silent %.0f s -- ended at its last sighting", w.id, w.ticket,
                     now_ts - self.cam_ts)
            self._end(w, w.last_seen, STALL if self._edge_mode else LOST, upd)

    # ── records ─────────────────────────────────────────────────────────────
    def _start(self, t: TrackObs, since: float, ts: float, policy: dict, upd: EngineUpdate, samples=()) -> None:
        self._dwell.pop(t.track_id, None)
        w = Wait(self._new_id(), self._next_ticket(dt(since)), t.track_id, t.track_id, since,
                 float(policy["wait_target_s"]), ts, self._foot(t), tracks=[t.track_id])
        w.reid = list(samples)
        w.box = self._nbox(t)
        w.area = next((a for a, r in zip(self.zone_areas, self.zones)
                       if lower_share_in(t.outline, [r], policy["wc_lower_frac"], policy["wc_zone_min"])), None)
        self.open[w.id] = w
        self._write(w, upd, ts)

    def _record(self, w: Wait, end: float | None = None, why: str | None = None) -> WaitRecord:
        return WaitRecord(w.id, dt(w.start), dt(end) if end is not None else None,
                          round(end - w.start, 1) if end is not None else None, why, w.target_s, self.camera,
                          w.first_track, w.track, w.ticket, dt(w.last_seen))

    def _write(self, w: Wait, upd: EngineUpdate, ts: float) -> None:
        w.written = ts
        upd.waits[w.id] = self._record(w)

    def _rejoin(self, w: Wait, t: TrackObs, ts: float, upd: EngineUpdate, note: str = "") -> None:
        self._ends.pop(w.id, None)                      # open again: it is no ended wait for the haircut links
        for k in [k for k, (x, _) in self._ended.items() if x is w]:
            del self._ended[k]
        if t.track_id in w.tracks:
            w.tracks.remove(t.track_id)
        w.tracks.append(t.track_id)
        w.track, w.hold, w.last_seen, w.pos, w.box = t.track_id, None, ts, self._foot(t), self._nbox(t)
        w.row_since = w.row_last = None
        w.out_since = None
        w.alert_from = ts
        self._dwell.pop(t.track_id, None)
        self.open[w.id] = w
        self._write(w, upd, ts)
        log.info("wait %s (#%s) rejoined by track %d%s", w.id, w.ticket, t.track_id, note)

    def _end(self, w: Wait, at: float, why: str, upd: EngineUpdate) -> None:
        at = max(at, w.start)
        del self.open[w.id]
        if why != STALL:                                # a stalled wait is never served (wait_links)
            self._ends[w.id] = at
        if why == LOST and not self._edge_mode:
            for tid in w.tracks:
                self._ended[tid] = (w, w.last_seen)
        if self._edge_mode and why in (LEFT, LOST, STALL):
            w.out_since = None
            self._gone[w.id] = _Gone(w, at)
        upd.waits[w.id] = self._record(w, at, why)
        a = self.alerts.close(EXCESSIVE_WAIT, w.id, dt(at))
        if a is not None:
            upd.alerts.append(a)
