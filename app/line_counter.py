"""Entrance/exit counting on a camera: a line, and which way across it is "in".

The factory entrance is also its exit, so one line per camera is drawn across the
gate and one side-to-side direction is marked as entering; the opposite direction
is leaving. Vehicles are the detections the tracking service already reads from
iris:meta:<cam> every frame (local object id + box). A vehicle is counted when the
bottom-centre of its box goes from one side of the line to the other, through the
line segment itself.

The line is stored normalised (0..1 of the frame) so it doesn't depend on the
pipeline's muxer resolution. Totals and the recent crossings live in Redis and are
reset with the rest of the run's data.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger("iris.linecount")

# A vehicle must be at least this far (px) from the line to establish/flip its side;
# this ignores box jitter right on the line.
SIDE_MARGIN_PX = 3.0
# The crossing has to happen within the segment, widened a little at both ends.
SEGMENT_SLACK = 0.08
# A vehicle first detected this close to the line (px) may have crossed it just before it
# was seen: detection only starts once a vehicle is already big and near, so the gate line
# is often within a step of where tracks begin. Its motion away from the line decides.
NEAR_FIRST_PX = 45.0
# ...but it must actually be moving away from the line, by at least this much (px) between
# two detections, so a parked vehicle next to the line isn't counted.
MOVE_AWAY_PX = 5.0
# The same vehicle (global id, else local id) isn't counted twice in a direction this fast.
DEBOUNCE_S = 20.0
RECENT_MAX = 50


@dataclass
class CountLine:
    a: tuple[float, float]   # normalised x, y
    b: tuple[float, float]
    enter_positive: bool     # moving towards the positive side of a->b means entering

    def to_dict(self) -> dict:
        return {"a": list(self.a), "b": list(self.b), "enter_positive": self.enter_positive}

    @staticmethod
    def from_dict(d: dict) -> "CountLine":
        a, b = d["a"], d["b"]
        for v in (*a, *b):
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError("line points must be normalised to 0..1")
        if abs(a[0] - b[0]) < 1e-4 and abs(a[1] - b[1]) < 1e-4:
            raise ValueError("line needs two different points")
        return CountLine((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), bool(d["enter_positive"]))


def side_of(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    """Signed distance-like value: >0 on the side the normal (-(by-ay), bx-ax) points to."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing_param(a, b, p0, p1) -> float | None:
    """Where along a->b (0 = a, 1 = b) the move p0->p1 crosses the line through a, b.
    None if the move doesn't cross it (or runs parallel)."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-9:
        return None
    t = ((p0[0] - a[0]) * ry - (p0[1] - a[1]) * rx) / den    # along the move
    u = ((p0[0] - a[0]) * sy - (p0[1] - a[1]) * sx) / den    # along the line
    return u if 0.0 <= t <= 1.0 else None


class LineCounter:
    def __init__(self, path: str, get_redis, frame_size=(1280, 720)):
        self.path = path
        self._get_redis = get_redis
        self.frame_w, self.frame_h = frame_size
        self._lines: dict[str, CountLine] = {}
        self._mtime = 0.0
        # (cam, object id) -> [side, (x, y), first_dist]: last confirmed side and position;
        # first_dist is the signed distance of a track's FIRST detection while it still
        # counts as 'near the line' and its direction hasn't been confirmed, else None
        self._state: dict[tuple[str, int], list] = {}
        self._last_count: dict[tuple[str, str, str], float] = {}
        self._load()

    # -- config ----------------------------------------------------------------
    def _load(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._lines, self._mtime = {}, 0.0
            return
        if mtime == self._mtime:
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            self._lines = {cam: CountLine.from_dict(d) for cam, d in raw.items()}
            self._mtime = mtime
        except Exception as exc:
            logger.warning("[linecount] cannot read %s: %s", self.path, exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({c: l.to_dict() for c, l in self._lines.items()}, f)
        os.replace(tmp, self.path)
        self._mtime = os.path.getmtime(self.path)

    def cameras(self) -> list[str]:
        self._load()
        return sorted(self._lines)

    def get_line(self, cam_id: str) -> dict | None:
        self._load()
        line = self._lines.get(cam_id)
        return line.to_dict() if line else None

    def set_line(self, cam_id: str, data: dict) -> dict:
        line = CountLine.from_dict(data)
        self._lines[cam_id] = line
        self._save()
        # a new line makes every remembered side meaningless
        for k in [k for k in self._state if k[0] == cam_id]:
            del self._state[k]
        return line.to_dict()

    def delete_line(self, cam_id: str) -> None:
        self._lines.pop(cam_id, None)
        self._save()
        for k in [k for k in self._state if k[0] == cam_id]:
            del self._state[k]

    # -- counting --------------------------------------------------------------
    def process(self, cam_id: str, dets: list, global_ids: dict, wall: float | None = None) -> list[dict]:
        """Feed one frame's detections; returns the crossings counted in it."""
        self._load()
        line = self._lines.get(cam_id)
        if line is None:
            return []
        wall = time.time() if wall is None else wall
        w, h = self.frame_w, self.frame_h
        a = (line.a[0] * w, line.a[1] * h)
        b = (line.b[0] * w, line.b[1] * h)
        length = max(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5, 1e-6)
        events = []
        seen = set()
        for det in dets:
            if len(det) < 7:
                continue
            try:
                oid = int(det[2])
                left, top, bw, bh = float(det[3]), float(det[4]), float(det[5]), float(det[6])
            except (TypeError, ValueError):
                continue
            p = (left + bw / 2.0, top + bh)            # bottom-centre: where it meets the ground
            key = (cam_id, oid)
            seen.add(key)
            s = side_of(a, b, p)
            d = s / length                               # signed distance to the line, px
            prev = self._state.get(key)
            if abs(d) < SIDE_MARGIN_PX:                  # on the line
                if prev is None:
                    self._state[key] = [0, p, d]         # first seen on it: side unknown yet
                continue
            side = 1 if d > 0 else -1
            u = None
            if prev is None:
                # first detection of the track
                self._state[key] = [side, p, d if abs(d) <= NEAR_FIRST_PX else None]
                continue
            if prev[0] == 0 or (prev[0] == side and prev[2] is not None):
                # the track began at/near the line; it counts as a crossing towards `side`
                # only if it is now moving away from the line on that side
                first_d = prev[2]
                origin = prev[1]
                self._state[key] = [side, p, None]
                if prev[0] != 0 and abs(d) - abs(first_d) < MOVE_AWAY_PX:
                    continue
                if prev[0] == 0 and abs(d) < MOVE_AWAY_PX:
                    continue
                u = crossing_param(a, b, origin, p)
                if u is None:
                    # the start was (almost) on the line, so the move may not straddle it
                    # numerically: fall back to where the start projects onto the line
                    u = ((origin[0] - a[0]) * (b[0] - a[0]) + (origin[1] - a[1]) * (b[1] - a[1])) / length ** 2
            else:
                self._state[key] = [side, p, None]
                if prev[0] == side:
                    continue
                u = crossing_param(a, b, prev[1], p)
            if u is None or not (-SEGMENT_SLACK <= u <= 1 + SEGMENT_SLACK):
                continue   # changed side around the end of the line, not through the gate
            toward_positive = side > 0
            direction = "enter" if toward_positive == line.enter_positive else "exit"
            gid = global_ids.get(str(oid), global_ids.get(oid)) if global_ids else None
            who = f"g{gid}" if gid is not None and int(gid) >= 0 else f"l{oid}"
            dkey = (cam_id, who, direction)
            if wall - self._last_count.get(dkey, 0.0) < DEBOUNCE_S:
                continue
            self._last_count[dkey] = wall
            ev = {"cam_id": cam_id, "direction": direction, "at": round(wall, 3),
                  "object_id": oid, "global_id": int(gid) if gid is not None else None}
            events.append(ev)
            self._record(ev)
        # forget objects that vanished, so a recycled local id starts fresh
        if len(self._state) > 2000:
            for k in [k for k in self._state if k[0] == cam_id and k not in seen][:1000]:
                del self._state[k]
        return events

    def _record(self, ev: dict) -> None:
        try:
            r = self._get_redis()
            if r is None:
                return
            cam = ev["cam_id"]
            r.hincrby(f"iris:count:totals:{cam}", ev["direction"], 1)
            r.lpush(f"iris:count:recent:{cam}", json.dumps(ev))
            r.ltrim(f"iris:count:recent:{cam}", 0, RECENT_MAX - 1)
            logger.info("[linecount] %s %s (object %s, gid %s)", cam, ev["direction"].upper(),
                        ev["object_id"], ev["global_id"])
        except Exception as exc:
            logger.warning("[linecount] could not record crossing: %s", exc)

    def totals(self, cam_id: str) -> dict:
        out = {"enter": 0, "exit": 0, "recent": []}
        try:
            r = self._get_redis()
            if r is None:
                return out
            h = r.hgetall(f"iris:count:totals:{cam_id}")
            for k, v in h.items():
                k = k.decode() if isinstance(k, bytes) else k
                if k in out:
                    out[k] = int(v)
            out["recent"] = [json.loads(x) for x in r.lrange(f"iris:count:recent:{cam_id}", 0, 19)]
        except Exception as exc:
            logger.warning("[linecount] could not read totals: %s", exc)
        return out

    def reset(self, cam_id: str) -> None:
        try:
            r = self._get_redis()
            if r is not None:
                r.delete(f"iris:count:totals:{cam_id}", f"iris:count:recent:{cam_id}")
        except Exception:
            pass
        self._last_count = {k: v for k, v in self._last_count.items() if k[0] != cam_id}
