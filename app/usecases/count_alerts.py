"""Count alerts: the pipeline's count-based alert stream, which carries both
overcrowding and fire/smoke. Overcrowding becomes an event; a fire/smoke
"started" opens an incident and "resolved" closes it (see fire_incidents)."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from ..redis_streams import read_stream
from ..zones import zone_by_id

logger = logging.getLogger("iris.stream")


def fire_sampler_needed(analytics_enabled: bool, fire_alert_enabled: bool) -> bool:
    """Whether the count-based alert sampler must run for this deployment.

    Fire/smoke alerting uses no nvdsanalytics ROI, so it must NOT be gated on
    analytics_enabled -- if it were, the sidecar would XADD fire alerts that nothing
    ever reads, with every component reporting healthy."""
    return bool(analytics_enabled or fire_alert_enabled)


_FIRE_LABELS = {"fire": "Fire", "smoke": "Smoke"}
# The zones module's end marker (Settings.end_markers): the zone's overcrowding latch released.
OVERCROWDING_RESOLVED = "overcrowding_resolved"


def count_alert_message(sid, zone, kind=None, cls=None, count=None):
    """(message, tags, prompt_kind) for one entry on the count-based alert stream.

    `kind` is absent on entries written by a sidecar older than this app; it defaults to
    overcrowding, which was the stream's only kind before fire/smoke existed -- the same
    version-skew tolerance zones' zone_alert_message applies to object_label/object_id."""
    if kind == "fire_smoke":
        subject = _FIRE_LABELS.get(cls, "Fire or smoke")
        return f"{subject} detected on {sid}", ["fire_smoke"], "fire_smoke"

    name = zone.get("name") if zone else None
    where = f" — {name}" if name else ""
    count_txt = f" ({count} people)" if count is not None else ""
    return (f"Overcrowding detected on {sid}{where}{count_txt}",
            ["zone_overcrowding"], "overcrowding")


class CountAlerts:
    def __init__(self, config: dict, redis_client, alerts, descriptions, fire,
                 running: Callable[[], bool], incidents=None, before=None):
        self._config = config
        # async (sid) -> None, awaited before each entry: reconciles a restart of the sidecar that wrote it
        self._before = before
        self._redis = redis_client
        self._alerts = alerts
        self._descriptions = descriptions
        self._fire = fire
        self._running = running
        # ZoneIncidents when the zones module sends end markers: overcrowding is an incident until the zone's latch
        # releases (overcrowding_resolved). None: one alert per crossing, as before.
        self._incidents = incidents

    async def run_for(self, sid: str) -> None:
        """Watches this stream's iris:overcrowd_alert:<id> Redis stream -- written by the
        sidecar's nvdsanalytics probe via OvercrowdingLatch on each debounced overcrowding
        state transition -- and turns each new entry into an event. Independent read loop,
        same reasoning as ZoneAlerts.run_for: no VLM call in the instant alert path."""
        try:
            async for f in read_stream(self._redis, f"iris:overcrowd_alert:{sid}", self._running):
                # See ZoneAlerts.run_for's matching comment: a non-cancellation
                # exception must not permanently kill this sampler.
                try:
                    await self.handle(sid, f)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error("[%s] alert sampler entry failed", sid, exc_info=True)
        except asyncio.CancelledError:
            pass

    async def handle(self, sid: str, f: dict) -> None:
        """One entry: a fire/smoke start or end, an overcrowding crossing or end. A restart of the sidecar that wrote it
        is reconciled first (`before`: FireIncidents.check_sidecar_boot_id), so the new boot's incident is never closed
        with the old boot's."""
        if self._before is not None:
            await self._before(sid)
        kind = f.get("kind")
        cls = f.get("class")
        if kind == "fire_smoke_resolved":
            await self._fire.resolve(sid, cls, f.get("duration"))
            return
        if kind == OVERCROWDING_RESOLVED:
            # an end marker (zones `end_markers`), never a new overcrowding
            if self._incidents is not None:
                await self._incidents.ended(sid, "overcrowding", f.get("zone"), f.get("duration"))
            return
        try:
            pts = float(f.get("pts", ""))
        except ValueError:
            pts = None
        try:
            count = int(f.get("count", ""))
        except ValueError:
            count = None
        zone_id = f.get("zone")
        zone = zone_by_id(self._config, sid, zone_id) if zone_id else None
        message, tags, prompt_kind = count_alert_message(
            sid, zone, kind=kind, cls=cls, count=count)
        describe = (self._descriptions.describe_fire if prompt_kind == "fire_smoke"
                    else self._descriptions.describe_overcrowding)
        if prompt_kind != "fire_smoke" and self._incidents is not None:
            await self._incidents.alert(
                sid, "overcrowding", zone_id, pts, prompt_kind,
                emit=lambda started: self._alerts.emit(
                    sid, message, tags, pts, describe, prompt_kind, count,
                    zone_id=zone_id, incident_started_at=started),
                again_message=lambda _entries: message, details={"count": count})
            return
        incident_started_at = None
        if prompt_kind == "fire_smoke":
            incident_started_at = self._fire.started_at(f.get("onset_age"))
        event_id = await self._alerts.emit(
            sid, message, tags, pts, describe, prompt_kind, count,
            zone_id=zone_id, incident_started_at=incident_started_at,
            # fire or smoke, as data: a console words the alert in its own language
            details={"class": cls} if prompt_kind == "fire_smoke" and cls else None)
        if incident_started_at is not None and event_id is not None:
            self._fire.track(sid, cls, event_id)
