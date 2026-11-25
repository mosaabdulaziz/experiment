"""Bridge from the multi-camera tracker to the truck-tracking platform.

The truck-tracking UI consumes "camera feed" batches from a Redis list
(``iris:camera-feed`` by default): every batch says which plates each camera IP
currently sees. Our identity is the VLM-validated global id, not a plate, so a
vehicle is published as a synthetic plate encoding its global id (see
``gid_to_plate``, e.g. gid 19 -> ``ا ا ا 0019``) until plate OCR can supply
the real text.

Each tick publishes one batch containing every mapped camera, with an empty
``plates`` list where nothing is visible (the platform treats that as "feed is
alive"). A vehicle is "currently visible" on a camera while it was seen there
within ``visible_window_s`` of wall-clock time.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app import plate_reads

logger = logging.getLogger("iris.truck_feed")


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


# The platform's plate parser needs Arabic letters (2-3) plus a 3-4 digit run, so a
# plain "GV-22" is dropped. A global id is encoded as 3 letters + 4 digits:
# number = gid % 10000, letters = base-18 digits of gid // 10000. Unique per id,
# and the letters stay "ا ا ا" until ids pass 9999. No 'أ' so normalisation
# (أ -> ا) can't merge two ids.
_LETTERS = "ابجدرسصطعفقكلمنهوي"


def gid_to_plate(gid: int) -> str:
    hi, lo = divmod(int(gid), 10000)
    letters = []
    for _ in range(3):
        hi, d = divmod(hi, len(_LETTERS))
        letters.append(_LETTERS[d])
    return f"{' '.join(reversed(letters))} {lo:04d}"


def plate_to_gid(plate: str) -> int:
    letters = [c for c in plate if c in _LETTERS]
    hi = 0
    for c in letters:
        hi = hi * len(_LETTERS) + _LETTERS.index(c)
    return hi * 10000 + int("".join(c for c in plate if c.isdigit()))


_NAMESPACE = uuid.UUID("6f1c2a54-0d0e-4b1e-9a55-1f6b7f0c1a10")


def tenant_uuid(value: str) -> str:
    """The platform requires UUID tenant ids. Real UUIDs pass through; any other
    label (e.g. "El-Mohandes") maps to a stable UUIDv5, so the same placeholder
    always yields the same id and can be registered on the platform later."""
    value = value.strip()
    if not value:
        return ""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(_NAMESPACE, value))


class TruckFeedBridge:
    """Builds and pushes camera-feed batches for the truck-tracking platform."""

    def __init__(self, cfg: dict, default_redis_url: str, camera_groups: dict | list | None = None):
        self.enabled = bool(cfg.get("enabled", False))
        # TRUCK_FEED_ENABLED=0 turns the bridge off whatever the config says: a stack that only browses an
        # imported run must never send its replayed history into the platform's database.
        if os.environ.get("TRUCK_FEED_ENABLED", "").strip().lower() in ("0", "false", "no", "off"):
            self.enabled = False
        self.redis_url = str(cfg.get("redis_url") or default_redis_url)
        self.queue_key = str(cfg.get("queue_key", "iris:camera-feed"))
        self.feed_id = str(cfg.get("feed_id", "iris-vlm-mtmc"))
        self.company_id = tenant_uuid(str(
            cfg.get("company_id") or os.environ.get("TRUCK_FEED_COMPANY_ID", "")
        ))
        self.branch_id = tenant_uuid(str(
            cfg.get("branch_id") or os.environ.get("TRUCK_FEED_BRANCH_ID", "")
        ))
        self.plates_dir = str(cfg.get("plates_dir", "data/plates"))
        self.interval_s = float(cfg.get("interval_s", 2.0))
        self.visible_window_s = float(cfg.get("visible_window_s", 5.0))
        # A vehicle is reported as having left only once no camera has seen it for this long: the direction
        # check says "leaving" as it reaches the gate, and a truck that waits there is still on site.
        self.exit_hold_s = float(cfg.get("exit_hold_s", 30.0))
        # our camera id -> the IP registered for that stop in truck-tracking
        self.camera_ips: dict[str, str] = {
            str(k): str(v) for k, v in (cfg.get("camera_ips") or {}).items()
        }
        # Cameras that see the same area are one place: a vehicle there is reported once, under the
        # camera that first saw it, however many of the group's cameras see it. (Otherwise the
        # platform reads every switch of view as a move between stops and splits the visit.)
        named = camera_groups if isinstance(camera_groups, dict) else {
            str(i): g for i, g in enumerate(camera_groups or [])}
        self._group_of: dict[str, str] = {str(c): str(n) for n, g in named.items() for c in g}
        self._home: dict[tuple[str, int], str] = {}   # (group, gid) -> the camera it is reported under
        self._r = None
        self._last_push = 0.0
        self._seq = 0
        # gid -> [camera, time it left, reported to the platform yet]
        self._exits: dict[int, list] = {}
        self._replayed: list[tuple[int, str, float]] = []
        self._last_wall: dict[int, float] = {}        # gid -> when any camera last saw it
        self._exit_marks: list[int] = []
        # restricted-zone alerts: key -> alert; what was last sent for each (gid, reading); and the batch being built
        self._alert_live: dict[str, dict] = {}
        self._alert_told: dict[str, tuple] = {}
        self._alert_marks: dict[str, tuple] = {}
        self._sent: dict[tuple[str, int], float] = {}
        # gid -> the reading last sent for it, and the ones in the batch being built
        self._told: dict[int, str] = {}
        self._told_marks: dict[int, str] = {}
        self._born = time.time()
        # what the monitoring endpoint reports
        self.stats = {"batches": 0, "push_failures": 0, "sightings_sent": 0,
                      "stale_dropped": 0, "exits_sent": 0, "last_push_ok": 0.0}
        self._marks: dict[tuple[str, int], float] = {}
        self._run_id = uuid.uuid4().hex[:8]

        if self.enabled and not self.company_id:
            # The platform rejects Redis batches without a tenant; fail loudly
            # here rather than pushing batches that are silently dropped.
            logger.error(
                "[truck_feed] enabled but company_id is empty "
                "(set truck_feed.company_id or TRUCK_FEED_COMPANY_ID) - disabling"
            )
            self.enabled = False
        if self.enabled and not self.camera_ips:
            logger.error("[truck_feed] enabled but camera_ips is empty - disabling")
            self.enabled = False
        if self.enabled:
            logger.info(
                "[truck_feed] publishing to %s key=%s company=%s branch=%s cameras=%s",
                self.redis_url, self.queue_key, self.company_id, self.branch_id,
                sorted(self.camera_ips),
            )

    def _plate_entry(self, gid: int, first_seen_at: str, **extra) -> dict:
        """The platform identifies the vehicle by the plate it is sent under (our id, encoded as a
        plate, so it never changes); when the plate reader has read it, the reading goes with it as
        `read_plate` and is what the platform shows as the truck's plate."""
        entry = {"plate": gid_to_plate(gid), "first_seen_at": first_seen_at, **extra}
        best = plate_reads.consensus(plate_reads.readings(self.plates_dir, gid, "_reads"))
        if best is not None:
            entry["read_plate"] = plate_reads.logical(best["text"])
            entry["confidence"] = round(float(best["ocr_conf"]), 3)
            self._told_marks[gid] = entry["read_plate"]
        return entry

    def _reading_updates(self, already: set[int]) -> list[dict]:
        """A reading can improve after the vehicle was last seen (the plate is read a moment later, or
        on a better crop), when no sighting is due any more: resend its last sighting with the new
        reading, so the platform's plate follows the vote and not only what it was told first."""
        out = []
        on_disk = set()
        try:
            for name in os.listdir(self.plates_dir):
                if name.endswith("_reads.json") and name[:-len("_reads.json")].isdigit():
                    on_disk.add(int(name[:-len("_reads.json")]))
        except OSError:
            pass
        for gid in sorted({g for (_c, g) in self._sent} | on_disk):
            ex = self._exits.get(gid)
            if gid in already or (ex is not None and ex[2] and not (len(ex) > 3 and ex[3])):
                continue            # one that has left must not be brought back by a plate reading
            best = plate_reads.consensus(plate_reads.readings(self.plates_dir, gid, "_reads"))
            if best is None or plate_reads.logical(best["text"]) == self._told.get(gid):
                continue
            cams = [(t, c) for (c, g), t in self._sent.items() if g == gid and c in self.camera_ips]
            if not cams:       # not sent in this process (the app restarted): where and when its reading was made
                cams = [(float(r.get("at", 0)), r["cam"]) for r in plate_reads.readings(self.plates_dir, gid)
                        if r.get("cam") in self.camera_ips]
            if not cams:
                continue
            at, cam = max(cams)
            out.append({"ip": self.camera_ips[cam], "observed_at": _iso(at),
                        "plates": [self._plate_entry(gid, _iso(at))]})
        return out

    def note_alert(self, alert: dict) -> None:
        """A vehicle stayed in a restricted zone past its threshold (see restricted_zones.py), or the global id of one
        already reported became known (`update`)."""
        key = alert["key"]
        live = self._alert_live.get(key)
        if live is None:
            self._alert_live[key] = dict(alert)
        elif alert.get("gid") is not None:
            live["gid"] = alert["gid"]

    def _resolve_replays(self) -> None:
        """Exits read back after a restart: the ones already reported only silence the vehicle, the others are due."""
        if not self._replayed:
            return
        sent = {int(x) for x in self._r.smembers(self.EXITS_SENT_KEY)}
        for gid, cam_id, at in self._replayed:
            if gid not in self._exits:
                self._exits[gid] = [cam_id, at, gid in sent, False]
        self._replayed.clear()

    def _read_of(self, gid: int) -> str | None:
        best = plate_reads.consensus(plate_reads.readings(self.plates_dir, gid, "_reads"))
        return plate_reads.logical(best["text"]) if best is not None else None

    def _alert_entries(self, now: float) -> list[dict]:
        """Restricted-zone alerts to send: new ones, and ones whose truck became known (or whose plate was read)
        since they were last sent, under the same dedup key so the platform enriches the alert it has."""
        out = []
        self._alert_marks = {}
        for key, a in list(self._alert_live.items()):
            if now - a["at"] > 1800.0:
                del self._alert_live[key]
                continue
            ip = self.camera_ips.get(a["cam_id"])
            if ip is None:
                continue
            gid = a.get("gid")
            read = self._read_of(int(gid)) if gid is not None else None
            sig = (gid, read)
            if self._alert_told.get(key) == sig:
                continue
            entry = {
                "kind": "restricted_zone", "ip": ip, "observed_at": _iso(a["at"]),
                "zone_id": a["zone_id"], "zone_name": a["zone_name"],
                "dwell_seconds": round(float(a["dwell_s"])), "threshold_seconds": round(float(a["threshold_s"])),
                "dedup_key": key,
            }
            if gid is not None:
                entry["plate"] = gid_to_plate(int(gid))
                if read:
                    entry["read_plate"] = read
            out.append(entry)
            self._alert_marks[key] = sig
        return out

    EXITS_SENT_KEY = "iris:truckfeed:exits_sent"

    def note_exit(self, gid: int, cam_id: str, at: float, reopens: bool = False, replay: bool = False) -> None:
        """A vehicle left the premises (crossed the gate outwards). It is reported to the
        platform once as an exit, and nothing more is reported for it afterwards.

        `replay`: the exit is read back from the sidecar's history after the app restarted. Whether it was
        reported before is kept in Redis; a reported one only silences the vehicle again (otherwise the
        sightings of a vehicle that left would open new journeys for it), an unreported one is still due."""
        if replay:
            self._replayed.append((gid, cam_id, at))
            return
        if gid not in self._exits:
            # reopens: the visit is complete but the vehicle returns later; a sighting after the exit
            # starts a new visit instead of being ignored like the sightings of a vehicle that left
            self._exits[gid] = [cam_id, at, False, reopens]

    def build_batch(
        self, vehicle_states: dict[str, dict[int, dict]], now: float | None = None
    ) -> dict[str, Any]:
        """One batch from ``CameraAggregator.get_vehicle_states()``."""
        now = time.time() if now is None else now
        # Every (camera, vehicle) sighting is reported once per advance of its real
        # last-seen time, oldest first. The platform orders by observed_at, so a
        # sighting confirmed late (older than the vehicle's latest) fills in its path
        # instead of moving it back, and the newest entry is the one that decides
        # where the vehicle is now.
        entries = []
        self._marks = {}
        self._told_marks = {}
        # (camera, gid) -> its state, for what is news this push
        news: dict[tuple[str, int], dict] = {}
        for cam_id, ip in self.camera_ips.items():
            for gid, v in sorted(vehicle_states.get(cam_id, {}).items()):
                if gid >= 0 and v["wall_last"] > self._last_wall.get(gid, 0.0):
                    self._last_wall[gid] = v["wall_last"]
                ex = self._exits.get(gid)
                if ex is not None:
                    reopens = len(ex) > 3 and ex[3]
                    if reopens and ex[2] and v["wall_last"] > ex[1] + 1.0:
                        del self._exits[gid]    # it came back after its reported exit: a new visit begins
                        if self._r is not None:
                            try:
                                self._r.srem(self.EXITS_SENT_KEY, gid)
                            except Exception:
                                pass
                    elif ex[2] or reopens:
                        continue   # it has left; whatever the cameras still see of it isn't news
                    elif now - self._last_wall.get(gid, 0.0) >= self.exit_hold_s:
                        continue   # out of sight long enough: the exit goes out in this batch instead
                    # else: the exit is still waiting for the vehicle to be out of sight, so it is still on site
                if gid < 0 or v["wall_last"] <= self._sent.get((cam_id, gid), 0.0):
                    continue
                # A sighting confirmed late (the VLM's verdict, a lagging poll) is reported once
                # even when it is older than the window: the vehicle may already have left the
                # camera. Only what predates this run (a replay after a restart) is not news.
                if (now - v["wall_last"] > self.visible_window_s and not v.get("pending")
                        and v["wall_last"] < self._born - 30.0):
                    self.stats["stale_dropped"] += 1
                    continue
                news[(cam_id, gid)] = v
        # one entry per (place, vehicle): the cameras of a group are merged under the home camera
        placed: dict[tuple[str, int], list[tuple[str, dict]]] = {}
        for (cam_id, gid), v in news.items():
            place = self._group_of.get(cam_id, cam_id)
            placed.setdefault((place, gid), []).append((cam_id, v))
            self._marks[(cam_id, gid)] = v["wall_last"]
        for (place, gid), members in sorted(placed.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            home = self._home.get((place, gid))
            if home is None or home not in self.camera_ips:
                home = min(members, key=lambda m: m[1]["wall_first"])[0]
                self._home[(place, gid)] = home
            newest = max(members, key=lambda m: m[1]["wall_last"])[1]
            entries.append({
                "ip": self.camera_ips[home],
                "observed_at": _iso(newest["wall_last"]),
                "plates": [self._plate_entry(gid, _iso(min(m[1]["wall_first"] for m in members)))],
            })
        # Exits, reported once the vehicle has been out of sight for exit_hold_s and stamped with when it was
        # last seen (not when the direction check called it).
        self._exit_marks = []
        for gid, ex in self._exits.items():
            cam_id, at, sent = ex[0], ex[1], ex[2]
            ip = self.camera_ips.get(cam_id)
            if sent or ip is None:
                continue
            if not (len(ex) > 3 and ex[3]):
                seen = self._last_wall.get(gid, at)
                if now - seen < self.exit_hold_s:
                    continue            # still in view of a camera: not gone yet
                at = max(at, seen)      # it left when it was last seen
            entries.append({
                "ip": ip,
                "observed_at": _iso(at),
                "plates": [self._plate_entry(gid, _iso(at), exited=True)],
            })
            self._exit_marks.append(gid)
        entries += self._reading_updates({plate_to_gid(e["plates"][0]["plate"]) for e in entries})
        entries.sort(key=lambda e: e["observed_at"])
        # A camera with nothing new still sends an empty entry: that is its heartbeat.
        busy = {e["ip"] for e in entries}
        cameras = [{"ip": ip, "plates": []} for ip in self.camera_ips.values()
                   if ip not in busy] + entries
        self._seq += 1
        batch: dict[str, Any] = {
            "schema_version": "1.0",
            "feed_id": self.feed_id,
            "company_id": self.company_id,
            "batch_id": f"{self.feed_id}-{self._run_id}-{self._seq:08d}",
            "observed_at": _iso(now),
            "cameras": cameras,
        }
        if self.branch_id:
            batch["branch_id"] = self.branch_id
        alerts = self._alert_entries(now)
        if alerts:
            batch["alerts"] = alerts
        return batch

    def due(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.enabled and now - self._last_push >= self.interval_s

    def push(self, vehicle_states: dict[str, dict[int, dict]]) -> bool:
        """Build and LPUSH one batch. Blocking; call from an executor."""
        import json

        try:
            if self._r is None:
                import redis as redis_mod
                self._r = redis_mod.Redis.from_url(self.redis_url)
            self._resolve_replays()
            batch = self.build_batch(vehicle_states)
            self._r.lpush(self.queue_key, json.dumps(batch, ensure_ascii=False))
            self._sent.update(self._marks)
            self._told.update(self._told_marks)
            self._alert_told.update(self._alert_marks)
            self.stats["batches"] += 1
            self.stats["sightings_sent"] += len(self._marks)
            self.stats["exits_sent"] += len(self._exit_marks)
            self.stats["last_push_ok"] = time.time()
            for gid in self._exit_marks:
                self._exits[gid][2] = True
            if self._exit_marks:
                self._r.sadd(self.EXITS_SENT_KEY, *self._exit_marks)
            for per_cam in vehicle_states.values():
                for v in per_cam.values():
                    v.pop("pending", None)
            self._last_push = time.time()
            return True
        except Exception as exc:
            self._r = None  # reconnect next tick
            self.stats["push_failures"] += 1
            self._last_push = time.time()
            logger.warning("[truck_feed] push failed: %s", exc)
            return False
