"""Door crossings for the chair-first engine (project 3, spec 2c): on the camera that draws the
entrance line, a track's box bottom-centre moving from one side of the line to the other between
two of its frames, within the doorway's extent, is a crossing (direction from the line's inside
point), confirmed as DoorCounter says. Exactly on the line counts as no side. Staff = the track's latest staff_p >=
staff_threshold. 2026-10-07 (review finding 4): a dead band around the line was measured and dropped -- on the side
test it lost exits (a leaving track vanishes right past the line: live-lap outs found 8 -> 4), on the hold alone it
lost 2 real crossings on the 39-min lap for 1 false one less on a 50-min lap."""
from __future__ import annotations

from typing import Callable

from domain import DoorCrossing, Frame, dt
from zones import DoorLine, FrameGeometry


class DoorCounter:
    """A side change is PENDING, dated to the frame where it happened. Dropped if the foot returns to
    the old side; emitted once the foot has stayed >= ce_door_confirm_s on the new side, or the track
    is unseen > ce_door_lost_s, or the foot leaves the doorway extent on the new side (on the old side the pending is dropped). Each frame's ts is the clock
    that flushes pendings of tracks absent from the frame."""

    _PRUNE_S = 60.0     # state of a track unseen this long is dropped

    def __init__(self, line: DoorLine | None, geometry: FrameGeometry):
        self.line, self.geometry = line, geometry
        self._side: dict[int, int] = {}
        self._pend: dict[int, tuple[float, int]] = {}     # track -> (ts of the flip, new side)
        self._seen: dict[int, float] = {}

    def forget(self, track_ids: set[int]) -> None:
        for t in track_ids:
            self._side.pop(t, None)
            self._pend.pop(t, None)
            self._seen.pop(t, None)

    def _commit(self, tid: int, camera: str, staff_of, policy: dict) -> DoorCrossing:
        t0, s = self._pend.pop(tid)
        self._side[tid] = s
        sp = staff_of(tid)
        return DoorCrossing(dt(t0), "in" if s > 0 else "out", camera, tid,
                            sp is not None and sp >= policy["staff_threshold"])

    def observe(self, frame: Frame, staff_of: Callable[[int], float | None], policy: dict) -> list[DoorCrossing]:
        if self.line is None:
            return []
        out = []
        lost, confirm = policy["ce_door_lost_s"], policy["ce_door_confirm_s"]
        here = {t.track_id for t in frame.tracks}
        for t in [t for t in self._pend if t not in here and frame.ts - self._seen[t] > lost]:
            out.append(self._commit(t, frame.camera, staff_of, policy))    # gone: not a camera's frame gap
        for t in [t for t, ts in self._seen.items() if frame.ts - ts > self._PRUNE_S]:
            self.forget({t})
        for t in frame.tracks:
            tid = t.track_id
            self._seen[tid] = frame.ts
            foot = self.geometry.foot(t.box)
            if not self.line.within_extent(foot):
                if tid in self._pend:      # it left the doorway: counts only if it is on the pending side
                    d = self.line.signed(foot)
                    if (d > 0) == (self._pend[tid][1] > 0) and d != 0:
                        out.append(self._commit(tid, frame.camera, staff_of, policy))
                    else:
                        del self._pend[tid]
                continue
            d = self.line.signed(foot)
            side = 1 if d > 0 else -1 if d < 0 else 0
            if side == 0:
                continue
            if tid not in self._side:
                self._side[tid] = side
            elif tid in self._pend:
                if side == self._side[tid]:
                    del self._pend[tid]
                elif frame.ts - self._pend[tid][0] >= confirm:
                    out.append(self._commit(tid, frame.camera, staff_of, policy))
            elif side != self._side[tid]:
                self._pend[tid] = (frame.ts, side)
        return out
