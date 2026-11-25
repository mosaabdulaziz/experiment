"""Extra waits from a waiting area's other views (2026-10-15, user; Palm Hills side sofa, GT test hour).

A waiting zone of the wait camera paired with waiting zones of other cameras (the same pair id; each of those draws
the part of the sofa that only it should count) is one waiting area. The wait camera may see two people as one (two
teens sitting close, one leaning across the other: one wait for both); the other views may see them apart.

With wt_extra_views 1, once a second the area's count is the higher of the wait camera's own count and the other
views' counts added up -- each the non-staff, unseated people whose lower mask has lain in its zone wc_dwell_s
(gaps <= wc_dwell_gap_s), from a camera whose newest frame is at most wc_stale_s old. When that count exceeds the
area's open waits (the wait camera's track waits that started in the area, on hold included, plus its extra waits)
for wt_start_s, an extra wait opens, dated to the first second of the excess. When it stays below them for
wt_extra_end_s, the newest extra wait ends `left`, dated to the first second of the deficit (a view losing someone
for a few seconds ends nothing). An extra wait has no track (track id -1): it is counted, averaged and linked to a
haircut by time like any other wait, without alert or return. A silent wait camera ends them `stall`.
Why not follow the other views' tracks: no identity across cameras (lap 3: the higher of ph-3 and the ph-1 half,
exact 96.7 % of the hour's seconds against 90.6 % for ph-3 alone)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from domain import EngineUpdate, Frame, WaitRecord, dt
from masks import Raster, raster
from waiting import lower_share_in
from wait_links import STALL
from zones import WAITING

log = logging.getLogger("barber.extra_waits")

LEFT = "left"
REFRESH_S = 30.0        # an open extra wait's record (last_seen_at) is re-sent this often
TRACK_TTL_S = 600.0


@dataclass
class _Extra:
    id: str
    ticket: int
    start: float
    target_s: float
    written: float = 0.0


@dataclass
class _Area:
    main: Raster
    views: dict[str, Raster]                         # other camera -> its part of the area
    dwell: dict[tuple[str, int], list[float]] = field(default_factory=dict)   # (camera, track) -> [since, last_in]
    counts: dict[str, tuple[float, int]] = field(default_factory=dict)       # camera -> (frame ts, count)
    extras: list[_Extra] = field(default_factory=list)
    excess_since: float | None = None
    deficit_since: float | None = None


class ExtraWaits:
    def __init__(self, zones_by_cam: dict[str, list[dict]], camera: str | None,
                 new_id: Callable[[], str], next_ticket: Callable):
        self._new_id, self._next_ticket = new_id, next_ticket
        self.areas: dict[str, _Area] = {}
        self.set_zones(zones_by_cam, camera)

    def set_zones(self, zones_by_cam: dict[str, list[dict]], camera: str | None) -> None:
        """New zones: an area still drawn keeps its open extra waits; one no longer drawn ends nothing here (the
        engine's restart rule closes leftovers)."""
        self.camera = camera
        old, self.areas = self.areas, {}
        if camera is None:
            return
        mains = {z["pair"]: z for z in zones_by_cam.get(camera, []) if z.get("type") == WAITING and z.get("pair")}
        for pair, z in mains.items():
            m = raster(z.get("points"))
            views = {cam: r for cam, zs in zones_by_cam.items() if cam != camera
                     for v in zs if v.get("type") == WAITING and v.get("pair") == pair
                     for r in [raster(v.get("points"))] if r is not None and r.area > 0}
            if m is None or m.area == 0 or not views:
                continue
            a = self.areas[pair] = _Area(m, views)
            if pair in old:
                a.extras = old[pair].extras

    def open(self) -> list[_Extra]:
        return [e for a in self.areas.values() for e in a.extras]

    def observe(self, frame: Frame, staff_of: Callable[[int], float | None], seated: set[int], policy: dict) -> None:
        if int(policy["wt_extra_views"]) != 1:
            return
        thr, ts = policy["staff_threshold"], frame.ts
        for a in self.areas.values():
            zone = a.main if frame.camera == self.camera else a.views.get(frame.camera)
            if zone is None:
                continue
            n = 0
            for t in frame.tracks:
                sp = staff_of(t.track_id)
                if (sp is not None and sp >= thr) or t.track_id in seated or t.conf is None \
                        or t.conf < policy["ce_conf_min"] \
                        or not lower_share_in(t.outline, [zone], policy["wc_lower_frac"], policy["wc_zone_min"]):
                    continue
                d = a.dwell.get((frame.camera, t.track_id))
                if d is None or ts - d[1] > policy["wc_dwell_gap_s"]:
                    d = a.dwell[(frame.camera, t.track_id)] = [ts, ts]
                d[1] = ts
                n += ts - d[0] >= policy["wc_dwell_s"]
            a.counts[frame.camera] = (ts, n)
            for k in [k for k, d in a.dwell.items() if k[0] == frame.camera and ts - d[1] > TRACK_TTL_S]:
                del a.dwell[k]

    def tick(self, now: float, track_waits: Callable[[str], int], wait_cam_ts: float | None, policy: dict,
             upd: EngineUpdate) -> None:
        """track_waits(area): the wait camera's open waits that started in that area (on hold included)."""
        if not self.areas:
            return
        on = int(policy["wt_extra_views"]) == 1
        silent = wait_cam_ts is None or now - wait_cam_ts > policy["wt_stale_s"]
        for area, a in self.areas.items():
            if not on or silent:
                for e in list(a.extras):
                    self._end(a, e, now if not on else (wait_cam_ts or now), LEFT if not on else STALL, upd)
                a.excess_since = a.deficit_since = None
                continue
            fresh = lambda cam: cam in a.counts and now - a.counts[cam][0] <= policy["wc_stale_s"]
            main = a.counts[self.camera][1] if fresh(self.camera) else 0
            n = max(main, sum(a.counts[cam][1] for cam in a.views if fresh(cam)))
            gap = n - track_waits(area) - len(a.extras)
            if gap > 0:
                a.excess_since = now if a.excess_since is None else a.excess_since
                if now - a.excess_since >= policy["wt_start_s"]:
                    e = _Extra(self._new_id(), self._next_ticket(dt(a.excess_since)), a.excess_since,
                               float(policy["wait_target_s"]))
                    a.extras.append(e)
                    self._write(e, upd, now)
                    log.info("extra wait %s (#%s) in %s: %d seen, %d open waits", e.id, e.ticket, area, n,
                             track_waits(area) + len(a.extras) - 1)
                    a.excess_since = None
            else:
                a.excess_since = None
            if gap < 0 and a.extras:
                a.deficit_since = now if a.deficit_since is None else a.deficit_since
                if now - a.deficit_since >= policy["wt_extra_end_s"]:
                    self._end(a, a.extras[-1], a.deficit_since, LEFT, upd)
                    a.deficit_since = None
            else:
                a.deficit_since = None
            for e in a.extras:
                if now - e.written >= REFRESH_S:
                    self._write(e, upd, now)

    def _record(self, e: _Extra, last_seen: float, end: float | None = None, why: str | None = None) -> WaitRecord:
        return WaitRecord(e.id, dt(e.start), dt(end) if end is not None else None,
                          round(end - e.start, 1) if end is not None else None, why, e.target_s, self.camera,
                          -1, -1, e.ticket, dt(last_seen))

    def _write(self, e: _Extra, upd: EngineUpdate, now: float) -> None:
        e.written = now
        upd.waits[e.id] = self._record(e, now)

    def _end(self, a: _Area, e: _Extra, at: float, why: str, upd: EngineUpdate) -> None:
        at = max(at, e.start)
        a.extras.remove(e)
        upd.waits[e.id] = self._record(e, at, at, why)
        log.info("extra wait %s (#%s) ended %s after %.0f s", e.id, e.ticket, why, at - e.start)
