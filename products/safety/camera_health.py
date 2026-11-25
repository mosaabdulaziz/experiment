"""Camera health for the safety product: incidents for a camera that sends nothing, a feature that is not running on
a camera, and a camera that sends too slowly -- read every few seconds from what each role sidecar publishes in Redis.

Per stream (`<camera>@<role>`): `down` (no iris:status key: the role's sidecar is not running it), `silent` (its newest
iris:meta entry is older than silent_s), `slow` (fewer than min_fps meta entries a second since the previous pass),
else ok -- read from the meta stream's counters, never its entries (they carry JPEGs). Per camera: every stream down or silent is one `offline` incident; otherwise each down/silent stream is a
`feature_down` incident and each slow one a `slow` incident. A problem opens an incident only after confirm_s, and the
incident resolves when the problem is gone. A role the controller is applying a change to (iris:safety:applying:<role>,
set by tools/safety_controller.py while it restarts it) is left out, so an Apply never pages anyone.

The incidents are ordinary events: tag camera_health (feature "health"), the camera id as their camera, category
operations, with {"problem", "role"} in details; they go to the notifier like any other incident.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable

from products.safety.config import APPLYING_PREFIX

from app.features import camera_of
from perception.engine.sources import split_stream_key_id

logger = logging.getLogger("iris.health")

# A role as its messages name it.
FEATURE_WORDS = {"proximity": "vehicle proximity", "zones": "zone monitoring", "ppe": "PPE detection",
                 "fall": "fall detection", "fire": "fire detection"}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_streams(sidecar_cfg: dict) -> dict:
    """{camera: [stream ids]} the running role sidecars should publish (the merged role configs' sources)."""
    out: dict = {}
    for source in sidecar_cfg.get("sources") or []:
        sid = str(source.get("id") or "")
        if sid:
            out.setdefault(camera_of(sid), []).append(sid)
    return out


class CameraHealth:
    def __init__(self, store, conn, redis_client, expected: Callable[[], dict], notifier=None,
                 confirm_s: float = 15.0, silent_s: float = 30.0, min_fps: float = 1.0, interval_s: float = 5.0,
                 now: Callable[[], float] = time.time, names: Callable[[], dict] = dict):
        self._store, self._conn, self._redis = store, conn, redis_client
        self._expected = expected
        self._names = names             # () -> {camera id: its name}, for the messages
        self.notifier = notifier
        self.confirm_s, self.silent_s, self.min_fps = confirm_s, silent_s, min_fps
        self.interval_s = interval_s
        self._now = now
        self._first_seen: dict = {}     # (camera, problem, role) -> when first seen
        self._counts: dict = {}         # stream -> (when, its meta stream's entries-added) at the last pass
        self._states: dict = {}         # stream -> "ok" / "slow" at the last pass that measured its rate
        self.open: dict = {}            # (camera, problem, role) -> event id
        self.fps: dict = {}             # stream -> its frames a second at the last pass that measured it (Health)

    # ── one stream ────────────────────────────────────────────────────────────────────────────────────────────
    def _state(self, sid: str, now: float) -> "tuple[str, float | None]":
        """(state, frames/s) of one stream, from its status key and its meta stream's counters only (XINFO STREAM:
        the last id written and how many were ever added) -- never its entries, which carry JPEGs. The rate is the
        count's change since the previous pass; None until there are two readings."""
        if not self._redis.exists(f"iris:status:{sid}"):
            return "down", None
        try:
            info = self._redis.xinfo_stream(f"iris:meta:{sid}")
        except Exception:  # noqa: BLE001 -- no meta stream yet
            return "silent", None
        last = info.get("last-generated-id") or b"0-0"
        last = last.decode() if isinstance(last, bytes) else str(last)
        if now - int(last.split("-")[0]) / 1000.0 > self.silent_s:
            return "silent", None
        added = int(info.get("entries-added") or 0)
        previous = self._counts.get(sid)
        if previous is not None and added == previous[1]:
            # nothing new since the last reading: a camera that stopped is "silent" once silent_s passes, not "slow"
            # first; a very slow one keeps its state, and the next reading measures over the longer span
            return self._states.get(sid, "ok"), None
        self._counts[sid] = (now, added)
        if previous is None or now - previous[0] < 1.0:
            return "ok", None
        fps = (added - previous[1]) / (now - previous[0])
        self._states[sid] = "slow" if fps < self.min_fps else "ok"
        return self._states[sid], fps

    def _held(self, role: "str | None") -> bool:
        return bool(role) and bool(self._redis.exists(f"{APPLYING_PREFIX}{role}"))

    # ── one pass ──────────────────────────────────────────────────────────────────────────────────────────────
    def problems(self, now: float) -> "tuple[dict, set]":
        """({(camera, problem, role): description} found now, {(camera, role)} held for an Apply). A held stream is
        not judged, and a camera with one is never called offline: its other streams are judged one by one."""
        found, held = {}, set()
        try:
            names = self._names() or {}
        except Exception:  # noqa: BLE001 -- the cameras file mid-write: ids will do for this pass
            names = {}
        for camera, sids in self._expected().items():
            name = names.get(camera) or camera
            states = {}
            for sid in sids:
                role = split_stream_key_id(sid)[1]
                if self._held(role):
                    held.add((camera, role or ""))
                else:
                    states[sid] = (role,) + self._state(sid, now)
                    if states[sid][1] in ("down", "silent"):
                        self.fps.pop(sid, None)
                    elif states[sid][2] is not None:
                        self.fps[sid] = states[sid][2]
            if not states:
                continue
            if not any(c == camera for c, _r in held) and all(
                    state in ("down", "silent") for _role, state, _fps in states.values()):
                found[(camera, "offline", "")] = f"No video from {name}"
                continue
            for sid, (role, state, fps) in states.items():
                words = FEATURE_WORDS.get(role or "", role or "its feature")
                if state in ("down", "silent"):
                    found[(camera, "feature_down", role or "")] = f"{words[0].upper()}{words[1:]} is not running on {name}"
                elif state == "slow":
                    rate = fps if fps is not None else self.fps.get(sid, 0.0)      # a pass with no new frame
                    found[(camera, "slow", role or "")] = f"{name} is slow on {words} ({rate:.1f} frames/s)"
        return found, held

    async def check(self) -> None:
        now = self._now()
        # Redis is the app's synchronous client: off the event loop.
        found, held = await asyncio.to_thread(self.problems, now)
        for key in [k for k in self._first_seen if k not in found]:
            camera, problem, role = key
            if (camera, role) in held or (problem == "offline" and any(c == camera for c, _r in held)):
                # held for an Apply: an open incident is neither resolved nor re-confirmed until it is judged; one not
                # open yet starts its confirmation afresh once it can be judged (not from before the hold)
                if key not in self.open:
                    self._first_seen.pop(key)
                continue
            self._first_seen.pop(key)
            if key in self.open:
                await self._resolve(key, now)
        for key, message in found.items():
            first = self._first_seen.setdefault(key, now)
            if key not in self.open and now - first >= self.confirm_s:
                await self._open(key, message, first)

    async def _open(self, key: tuple, message: str, since: float) -> None:
        camera, problem, role = key
        record = await self._store.save(
            "alert", message, ["camera_health"], camera=camera, severity="warning", category="operations",
            incident_status="ongoing", incident_started_at=_iso(since), details={"problem": problem, "role": role})
        self.open[key] = record["id"]
        logger.warning("[%s] health: %s", camera, message)
        await self._conn.broadcast({"type": "event", "event": record, "camera": camera})
        if self.notifier is not None:
            self.notifier.notify(record, "opened")

    async def _resolve(self, key: tuple, now: float) -> None:
        event_id = self.open.pop(key)
        record = await self._store.get_event(event_id)
        started = record.get("incident_started_at") if record else None
        duration = (now - datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                    if started else None)
        record = await self._store.update_incident(event_id, "resolved", duration)
        logger.info("[%s] health: %s resolved", key[0], key[1])
        if record is not None:
            await self._conn.broadcast({"type": "event_update", "id": event_id, "event": record, "camera": key[0]})

    async def run(self) -> None:
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- keep watching
                logger.exception("camera health check failed")
            await asyncio.sleep(self.interval_s)
