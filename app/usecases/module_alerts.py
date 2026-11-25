"""Alerts raised by pipeline modules through the one alert contract (`iris:alert:<camera>`).

A module calls `EngineServices.alert(...)`; this reads those entries for every camera and turns each into an event
the way every other alert becomes one (AlertEvents.emit): saved, broadcast, clip queued. The entry says what it is
-- `kind`, `severity`, `category`, optionally the `message`, `zone`, `count` -- so a new module's alerts need no code
here. Zone intrusion/overcrowding and fire/smoke keep their own streams (they open incidents and name zones).
An entry may carry an `incident` id and `incident_state` (opened / escalated / resolved): the incident is one event,
raised in place and closed when it ends (`ModuleAlerts.open`); its other fields are kept as the event's details."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from ..redis_streams import read_streams
from ..severity import normalise_category, normalise_severity

logger = logging.getLogger("iris.stream")

ALERT_PREFIX = "iris:alert:"


def alert_message(sid: str, kind: str, message: "str | None" = None) -> str:
    """The text of the event: the module's own `message`, or one made from its kind ("proximity_warning" ->
    "Proximity warning on cam-1")."""
    return message or f"{kind.replace('_', ' ').capitalize()} on {sid}"


STANDARD_FIELDS = ("v", "kind", "severity", "category", "message", "zone", "pts", "count", "incident",
                   "incident_state")


def _num(raw) -> "float | None":
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _ids(raw) -> list:
    out = []
    for part in (raw or "").split(","):
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _started_at(duration_raw) -> str:
    """When the pair first got close (wall clock, ISO): now minus how long it has been close."""
    held = _num(duration_raw) or 0.0
    return (datetime.now(timezone.utc) - timedelta(seconds=held)).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModuleAlerts:
    def __init__(self, redis_client, alerts, stream_ids: Callable[[], list], running: Callable[[], bool],
                 depth_images=None, before=None):
        self._redis = redis_client
        # async (sid) -> None, awaited before each entry: reconciles a restart of the sidecar that wrote it, so the new
        # boot's incidents are never closed with the old boot's (StreamManager: FireIncidents.check_sidecar_boot_id)
        self._before = before
        self._depth_images = depth_images         # app/depth_images.DepthImages: proximity's depth images, by event
        self._alerts = alerts
        self._stream_ids = stream_ids
        self._running = running
        # (camera, incident id) -> {"event_id", "min_gap_pts", "persons", "vehicles"} for incidents still open.
        # In memory only, like fire's: an app restart leaves them to EventStore.resolve_stale_pending ("unknown").
        self.open: dict = {}

    async def run(self) -> None:
        """One reader over every camera's alert stream. A bad entry is logged and skipped, never fatal."""
        try:
            async for key, f in read_streams(
                    self._redis, lambda: [ALERT_PREFIX + s for s in self._stream_ids()], self._running):
                sid = key[len(ALERT_PREFIX):]
                try:
                    await self.handle(sid, f)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error("[%s] module alert entry failed", sid, exc_info=True)
        except asyncio.CancelledError:
            pass

    async def handle(self, sid: str, f: dict) -> None:
        """One alert entry: an event, or for an incident, its opening, escalation or end."""
        if self._before is not None:
            await self._before(sid)
        kind = f.get("kind") or "alert"
        incident, state = f.get("incident") or None, f.get("incident_state") or None
        extra = {k: v for k, v in f.items() if k not in STANDARD_FIELDS}
        image = extra.pop("depth_image", None)               # kept apart: the card fetches it by event id
        if image and self._depth_images is not None:
            extra["depth_image"] = "1"
        if incident and state == "resolved":
            await self._resolved(sid, incident, extra)
            return
        severity = normalise_severity(f.get("severity"), alert_hint=True)
        message = alert_message(sid, kind, f.get("message"))
        pts, zone_id = _num(f.get("pts")), f.get("zone") or None
        key = (sid, incident)
        if incident and state == "escalated" and key in self.open:
            event_id = self.open[key]["event_id"]
            await self._alerts.escalate(event_id, sid, pts, severity, message, extra, zone_id, kind)
            self._keep_depth(event_id, image)
            self._remember(key, event_id, extra)
            return
        try:
            count = int(f.get("count", ""))
        except ValueError:
            count = None
        quiet = extra.get("suppressed") == "depth"     # a pair the depth check held back: kept, never shown live
        event_id = await self._alerts.emit(
            sid, message, [kind, "depth-suppressed"] if quiet else [kind], pts, False, kind, count, zone_id=zone_id,
            severity=severity, category=normalise_category(f.get("category")), details=extra or None,
            incident_started_at=_started_at(extra.get("duration_s")) if incident else None,
            **({"broadcast": False} if quiet else {}))
        if event_id is not None:
            self._keep_depth(event_id, image)
        if incident and event_id is not None:
            self._remember(key, event_id, extra)

    def _keep_depth(self, event_id: int, image) -> None:
        if image and self._depth_images is not None:
            self._depth_images.put(event_id, image)          # refused with a warning when malformed; the event stays

    def _remember(self, key: tuple, event_id: int, extra: dict) -> None:
        old = self.open.get(key, {})
        self.open[key] = {"event_id": event_id,
                          "min_gap_pts": _num(extra.get("min_gap_pts")) if "min_gap_pts" in extra
                          else old.get("min_gap_pts"),
                          "persons": _ids(extra.get("person_ids")) or old.get("persons", []),
                          "vehicles": _ids(extra.get("vehicle_ids")) or old.get("vehicles", [])}

    async def _resolved(self, sid: str, incident: str, extra: dict) -> None:
        entry = self.open.pop((sid, incident), None)
        if entry is None:
            logger.info("[%s] incident %s ended, but this app never saw it open (it restarted?)", sid, incident)
            return
        await self._alerts.resolve_incident(entry["event_id"], sid, _num(extra.get("duration_s")), extra or None)
        at = _num(extra.get("min_gap_pts"))
        await self._alerts.snapshot(entry["event_id"], sid, entry["min_gap_pts"] if at is None else at,
                                    _ids(extra.get("person_ids")) or entry["persons"],
                                    _ids(extra.get("vehicle_ids")) or entry["vehicles"])

    async def sidecar_restarted(self, sid: str) -> None:
        """The camera's pipeline restarted: its open incidents will never resolve (FireIncidents closes their rows as
        "unknown"); each still gets its snapshot from the last minimum gap it reported."""
        for key in [k for k in self.open if k[0] == sid]:
            entry = self.open.pop(key)
            await self._alerts.snapshot(entry["event_id"], sid, entry["min_gap_pts"], entry["persons"],
                                        entry["vehicles"])
