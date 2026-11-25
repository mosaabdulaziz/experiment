"""'Waiting now' for the chair-first engine (project 4 C, 2026-10-04; centre camera only, user decision, see
dev/chairrun/WAITING.md: it set the count 92 % of the time, MAE 0.08 vs 0.13 for the highest of three).

Each camera counts non-staff people whose lower mask (wc_lower_frac) lies at least wc_zone_min inside one of its
waiting zones, once they have been in the area wc_dwell_s (walking through doesn't count; out longer than
wc_dwell_gap_s restarts the dwell); a chair-seat occupant never counts. Once a second the raw number is the latest count of the
centre camera (the one that draws the entrance line, passed as `camera`) if its newest frame is at most wc_stale_s old,
else None; side cameras are still counted (counts, for diagnosis) but never set the number. If no camera draws an
entrance line (camera=None) it falls back to the highest latest count among fresh cameras. And the shown number rises
when a higher raw value has held wc_up_s and falls when a lower one has held wc_down_s. No identity across cameras."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from domain import Frame
from masks import Raster, raster
from zones import WAITING

TRACK_TTL_S = 600.0


def zones_of_type(zones_by_cam: dict[str, list[dict]], zone_type: str) -> dict[str, list[Raster]]:
    out: dict[str, list[Raster]] = {}
    for cam, zones in zones_by_cam.items():
        rs = [r for z in zones if z.get("type") == zone_type for r in [raster(z.get("points"))]
              if r is not None and r.area > 0]
        if rs:
            out[cam] = rs
    return out


def waiting_zones(zones_by_cam: dict[str, list[dict]]) -> dict[str, list[Raster]]:
    return zones_of_type(zones_by_cam, WAITING)


def lower_share_in(outline, zones: list[Raster], frac: float, need: float) -> bool:
    """The lower `frac` of the mask outline lies at least `need` inside one of the zones."""
    if not zones or outline is None:
        return False
    m = raster(outline)
    if m is None or m.area == 0:
        return False
    low = m.lower(frac)
    return low.area > 0 and any(low.overlap(z) / low.area >= need for z in zones)


@dataclass
class _Track:
    since: float
    last_in: float


class WaitingCounter:
    def __init__(self, zones_by_cam: dict[str, list[dict]], camera: str | None = None):
        self.set_zones(zones_by_cam, camera)
        self.clear()

    def set_zones(self, zones_by_cam: dict[str, list[dict]], camera: str | None = None) -> None:
        self.zones = waiting_zones(zones_by_cam)
        self.camera = camera

    def clear(self) -> None:
        self._tracks: dict[tuple[str, int], _Track] = {}
        self._latest: dict[str, tuple[float, int]] = {}
        self._hist: deque = deque()
        self.shown: int | None = None
        self.raw: int | None = None

    @property
    def counts(self) -> dict[str, int]:
        return {cam: n for cam, (_, n) in self._latest.items()}

    def _inside(self, t, zones: list[Raster], policy: dict) -> bool:
        if t.conf is None or t.conf < policy["ce_conf_min"]:
            return False
        return lower_share_in(t.outline, zones, policy["wc_lower_frac"], policy["wc_zone_min"])

    def observe(self, frame: Frame, staff_of: Callable[[int], float | None], seated: set[int], policy: dict) -> None:
        zones = self.zones.get(frame.camera)
        if not zones:
            return
        n, ts = 0, frame.ts
        for t in frame.tracks:
            sp = staff_of(t.track_id)
            if t.track_id in seated or (sp is not None and sp >= policy["staff_threshold"]) \
                    or not self._inside(t, zones, policy):
                continue
            key = (frame.camera, t.track_id)
            st = self._tracks.get(key)
            if st is None or ts - st.last_in > policy["wc_dwell_gap_s"]:
                st = self._tracks[key] = _Track(ts, ts)
            st.last_in = ts
            if ts - st.since >= policy["wc_dwell_s"]:
                n += 1
        self._latest[frame.camera] = (ts, n)

    def tick(self, now_ts: float, policy: dict) -> int | None:
        cams = self._latest if self.camera is None else {c: v for c, v in self._latest.items() if c == self.camera}
        fresh = [n for ts, n in cams.values() if now_ts - ts <= policy["wc_stale_s"]]
        self.raw = max(fresh) if fresh else None
        if self.raw is None:
            self.shown, self._hist = None, deque()
            return None
        self._hist.append((now_ts, self.raw))
        keep = max(policy["wc_up_s"], policy["wc_down_s"]) + 1.0
        while self._hist and now_ts - self._hist[0][0] > keep:
            self._hist.popleft()
        if self.shown is None:
            self.shown = self.raw
        elif self.raw > self.shown and self._held(now_ts, policy["wc_up_s"], lambda v: v > self.shown):
            self.shown = min(v for t, v in self._hist if now_ts - t <= policy["wc_up_s"])
        elif self.raw < self.shown and self._held(now_ts, policy["wc_down_s"], lambda v: v < self.shown):
            self.shown = max(v for t, v in self._hist if now_ts - t <= policy["wc_down_s"])
        for key in [k for k, st in self._tracks.items() if now_ts - st.last_in > TRACK_TTL_S]:
            del self._tracks[key]
        return self.shown

    def _held(self, now_ts: float, hold: float, pred) -> bool:
        window = [(t, v) for t, v in self._hist if now_ts - t <= hold]
        return (bool(window) and all(pred(v) for _, v in window)
                and (hold <= 0 or self._hist[0][0] <= now_ts - hold))
