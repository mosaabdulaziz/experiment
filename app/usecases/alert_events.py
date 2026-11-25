"""Saving and announcing one alert -- shared by zone alerts and count alerts
(overcrowding, fire/smoke): store it, broadcast it, queue its clip, and start
its description when one is wanted; and, for a module's incident, raising it in
place, closing it and queuing its snapshot."""
from __future__ import annotations

import logging

import time

from ..overlay_track import incident_pair
from ..severity import zone_alert_defaults
from ..zones import zone_by_id

logger = logging.getLogger("iris.stream")


def _rank(severity: "str | None") -> int:
    return {"info": 0, "warning": 1, "critical": 2}.get(severity or "", -1)


class AlertEvents:
    def __init__(self, config: dict, store, conn_manager, clip_service, recordings, descriptions):
        self._config = config
        self._store = store
        self._conn = conn_manager
        self._clip_service = clip_service
        self._recordings = recordings
        self._descriptions = descriptions
        # Telegram (products/safety/notifier.py) in the safety product: told when an incident opens and when it
        # escalates; it only queues, so it never delays the alert. None elsewhere.
        self.notifier = None

    async def emit(self, sid: str, message: str, tags: list, pts: "float | None",
                               describe_enabled: bool, prompt_kind: str, count: "int | None" = None,
                               zone_id: "str | None" = None,
                               incident_started_at: "str | None" = None,
                               severity: "str | None" = None,
                               category: "str | None" = None,
                               details: "dict | None" = None,
                               broadcast: bool = True) -> "int | None":
        """Shared tail for both the zone-intrusion and overcrowding alert samplers: save +
        broadcast the instant alert, then -- if describe_enabled -- spawn a description job
        tagged with which Analyzer prompt to use (prompt_kind) and, for overcrowding, the
        triggering count. Resolves immediately to the fallback text if pts couldn't be
        parsed (no frame position to center the description window on). Returns the saved
        event's id (fire/smoke's caller needs it to track the incident to its later
        resolution; every other caller ignores it). broadcast=False keeps a quiet event (a pair the
        proximity depth check held back) out of the live view: saved and clipped, never sent."""
        description_status = "pending" if describe_enabled else None
        clip_status = "pending" if self._recordings.clips_enabled() else None
        # Deterministic alerts get a fixed severity/category per behaviour; prompt_kind
        # is already the behaviour name on this path.
        # A module's own alert (ModuleAlerts) states its severity and category; the rest take the defaults.
        default_severity, default_category = zone_alert_defaults(prompt_kind)
        severity, category = severity or default_severity, category or default_category
        incident_status = "ongoing" if incident_started_at is not None else None
        zone = zone_by_id(self._config, sid, zone_id) if zone_id else None
        if details is not None and zone is not None:
            details = {**details, "zone_name": zone["name"]}        # as it was named when the alert fired
        record = await self._store.save(
            "alert", message, tags, camera=sid, description_status=description_status,
            clip_status=clip_status, severity=severity, category=category,
            incident_status=incident_status, incident_started_at=incident_started_at, details=details)
        if broadcast:
            await self._conn.broadcast({"type": "event", "event": record, "camera": sid})
            if self.notifier is not None:
                self.notifier.notify(record, "opened")
        if clip_status == "pending":
            # After save+broadcast: the alert's own latency must not wait on footage.
            # The zone was resolved above, not at build time, so the clip draws the
            # geometry that was live when the alert fired.
            await self._clip_service.submit(record["id"], sid, pts, zone, wall_time=time.time(),
                                            pair=incident_pair(details), kind=prompt_kind)
        if description_status == "pending":
            if pts is not None:
                self._descriptions.start(sid, record["id"], pts, prompt_kind, count, zone_id)
            else:
                await self._descriptions.finish_with_fallback(record["id"])
        return record["id"]

    async def escalate(self, event_id: int, sid: str, pts: "float | None", severity: str, message: str,
                       details: "dict | None", zone_id: "str | None" = None, kind: "str | None" = None) -> bool:
        """An open incident got more severe (or, for a zone incident, another entry): the same event, raised in place,
        and a follow-up clip of this moment. Notified only when the severity actually rose. False when the event's row
        is gone (deleted, or past retention): nothing was raised."""
        before = await self._store.get_event(event_id)
        record = await self._store.update_alert(event_id, severity=severity, message=message, details=details)
        if record is None:
            return False
        await self._conn.broadcast({"type": "event_update", "id": event_id, "event": record, "camera": sid})
        if self.notifier is not None and _rank(severity) > _rank((before or {}).get("severity")):
            self.notifier.notify(record, "escalated")
        if pts is not None and self._recordings.clips_enabled():
            zone = zone_by_id(self._config, sid, zone_id) if zone_id else None
            await self._clip_service.submit(event_id, sid, pts, zone, wall_time=time.time(), append=True,
                                            pair=incident_pair(details), kind=kind)
        return True

    async def resolve_incident(self, event_id: int, sid: str, duration_s: "float | None",
                               details: "dict | None") -> None:
        """An incident ended: resolved, with how long the pair was close and its final numbers."""
        if await self._store.update_incident(event_id, "resolved", duration_s) is None:
            return
        record = await self._store.update_alert(event_id, details=details)
        if record is not None:
            await self._conn.broadcast({"type": "event_update", "id": event_id, "event": record, "camera": sid})

    async def snapshot(self, event_id: int, sid: str, at_pts: "float | None", person_ids: list,
                       vehicle_ids: list) -> None:
        """Queue the incident's snapshot at its minimum gap (taken once its clips exist)."""
        if at_pts is not None and self._recordings.clips_enabled():
            self._clip_service.snapshot(event_id, sid, at_pts, person_ids, vehicle_ids)
