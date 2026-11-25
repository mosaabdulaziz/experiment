"""Restricted areas: a vehicle that stays in one longer than the zone's threshold raises an alert.

Zones are drawn per camera in the platform's Live page and kept in data/restricted_zones.json as
{camera: {"zones": [{"id", "name", "threshold_s", "points": [[x, y], ...]}]}} with points normalised (0..1)
to the frame. This needs nothing but the camera itself: a vehicle is "in" a zone while the bottom-centre of its
box (where it stands on the ground) is inside the polygon, and it is followed by its tracker id on that camera,
whether or not it was ever matched to a global id or came through the entrance. When it has a global id (it is
tracked on several cameras) the clock follows that instead, and the alert is enriched with the truck's identity
and plate. Gaps of up to `grace_s` (detector dropouts, a pass behind something) do not stop the clock. One alert
per stay; the vehicle has to leave (out of the zone for more than `grace_s`) and come back to fire again, or stay
a further `cooldown_s` to be reported again.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def load_zones(path: str) -> dict[str, list[dict]]:
    """{camera: [zone, ...]} from the zones file; {} when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out: dict[str, list[dict]] = {}
    for cam, entry in raw.items():
        zones = []
        for i, z in enumerate((entry or {}).get("zones", [])):
            pts = [(float(p[0]), float(p[1])) for p in z.get("points", []) if len(p) == 2]
            if len(pts) < 3:
                continue
            zones.append({"id": str(z.get("id") or f"{cam}-r{i}"), "name": str(z.get("name") or f"Zone {i + 1}"),
                          "threshold_s": max(1.0, float(z.get("threshold_s", 60))), "points": pts})
        if zones:
            out[cam] = zones
    return out


def save_zones(path: str, cam: str, zones: list[dict]) -> None:
    """Replace one camera's zones (an empty list removes them)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    stored = [{"id": z["id"], "name": z["name"], "threshold_s": z["threshold_s"],
               "points": [[round(float(x), 5), round(float(y), 5)] for x, y in z["points"]]}
              for z in zones if len(z.get("points", [])) >= 3]
    if stored:
        data[cam] = {"zones": stored}
    else:
        data.pop(cam, None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass
class _Stay:
    cam: str
    obj_id: int
    gid: int | None = None
    since: float | None = None      # when the vehicle was first inside this stay
    last_in: float = 0.0            # last time it was inside
    alerted_at: float | None = None
    key: str = ""                   # dedup key of the alert raised for this stay
    told_gid: int | None = None     # the global id the alert was last enriched with
    fired: int = 0


class RestrictedZoneMonitor:
    CLASSES = frozenset({"vehicle", "car", "truck", "bus"})

    def __init__(self, path: str, mux_w: int = 1280, mux_h: int = 720, grace_s: float = 5.0,
                 cooldown_s: float = 600.0, classes=None):
        self.path, self.mux_w, self.mux_h = path, mux_w, mux_h
        self.grace_s, self.cooldown_s = grace_s, cooldown_s
        self.classes = frozenset(classes) if classes else self.CLASSES
        self._zones: dict[str, list[dict]] = {}
        self._mtime = None
        self._stays: dict[tuple, _Stay] = {}       # (zone id, subject) -> stay; subject = ("g", gid) | ("o", cam, obj)
        self._lock = threading.Lock()

    def _reload(self) -> None:
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        self._mtime = mtime
        self._zones = load_zones(self.path) if mtime is not None else {}
        live = {z["id"] for zs in self._zones.values() for z in zs}
        self._stays = {k: v for k, v in self._stays.items() if k[0] in live}   # a deleted zone forgets its stays

    def observe(self, cam: str, dets: list, global_ids: dict, now: float) -> list[dict]:
        """One frame of a camera: every detection with [label, conf, object_id, left, top, width, height] (mux
        pixels) and the global ids the sidecar has matched so far. Returns the alerts that fall due now (or the
        enrichments of one already raised, when the vehicle's global id becomes known)."""
        with self._lock:
            self._reload()
            zones = self._zones.get(cam)
            if not zones:
                return []
            out = []
            for det in dets:
                if len(det) < 7 or det[0] not in self.classes:
                    continue
                obj = int(det[2])
                gid = global_ids.get(str(obj))
                if gid is None:
                    gid = global_ids.get(obj)
                gid = int(gid) if gid is not None and int(gid) >= 0 else None
                px = (det[3] + det[5] / 2.0) / self.mux_w
                py = (det[4] + det[6]) / self.mux_h
                for z in zones:
                    inside = point_in_polygon(px, py, z["points"])
                    st = self._stay(z["id"], cam, obj, gid)
                    if not inside:
                        continue
                    if st is None:
                        st = self._stays[(z["id"], ("g", gid) if gid is not None else ("o", cam, obj))] = _Stay(cam, obj, gid)
                    if st.since is None:
                        st.since = now
                    st.last_in = now
                    st.obj_id = obj
                    if gid is not None:
                        st.gid = gid
                    dwell = now - st.since
                    if dwell >= z["threshold_s"] and (st.alerted_at is None or now - st.alerted_at >= self.cooldown_s):
                        st.alerted_at = now
                        st.fired += 1
                        st.key = f"restricted:{z['id']}:{cam}:{obj}:{int(st.since)}:{st.fired}"
                        st.told_gid = st.gid
                        out.append(self._alert(st, z, dwell, now, update=False))
                    elif st.alerted_at is not None and st.gid is not None and st.gid != st.told_gid:
                        st.told_gid = st.gid              # it is now known to be a tracked truck: say who
                        out.append(self._alert(st, z, dwell, now, update=True))
            return out

    def _stay(self, zone_id: str, cam: str, obj: int, gid: int | None):
        """The stay of this vehicle in this zone, if there is one; a stay that began on the camera's tracker id
        moves under the global id once the vehicle has one (the earliest start is kept)."""
        by_obj = self._stays.get((zone_id, ("o", cam, obj)))
        if gid is None:
            return by_obj
        by_gid = self._stays.get((zone_id, ("g", gid)))
        if by_obj is not None:
            del self._stays[(zone_id, ("o", cam, obj))]
            if by_gid is None:
                self._stays[(zone_id, ("g", gid))] = by_obj
                by_obj.gid = gid
                return by_obj
            if by_obj.since is not None and (by_gid.since is None or by_obj.since < by_gid.since):
                by_gid.since = by_obj.since
            by_gid.alerted_at = max(filter(None, [by_gid.alerted_at, by_obj.alerted_at]), default=None)
            if not by_gid.key:
                by_gid.key, by_gid.fired, by_gid.told_gid = by_obj.key, by_obj.fired, by_obj.told_gid
            return by_gid
        return by_gid

    @staticmethod
    def _alert(st: _Stay, z: dict, dwell: float, now: float, update: bool) -> dict:
        return {"kind": "restricted_zone", "gid": st.gid, "obj_id": st.obj_id, "cam_id": st.cam,
                "zone_id": z["id"], "zone_name": z["name"], "dwell_s": round(dwell, 1),
                "threshold_s": z["threshold_s"], "since": st.since, "at": now, "key": st.key, "update": update}

    def expire(self, now: float) -> None:
        """A vehicle out of the zone (or not reported at all) for longer than the grace has left: its next stay
        starts a new clock and can alert again."""
        with self._lock:
            for key in [k for k, s in self._stays.items() if now - s.last_in > self.grace_s]:
                del self._stays[key]
