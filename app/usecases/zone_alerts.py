"""Zone alerts: a person entering a restricted zone, or leaving a valid one.
The pipeline decides each crossing; this turns its alert into an event."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from ..redis_streams import read_stream
from ..zones import zone_by_id

logger = logging.getLogger("iris.stream")


# DeepStream's UNTRACKED_OBJECT_ID — what int(obj_meta.object_id) yields when the tracker
# never assigned an id. Rendered verbatim it reads "Person #18446744073709551615".
_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

# The zones module's end marker (Settings.end_markers): the zone has stayed clear since its intrusion.
INTRUSION_CLEARED = "intrusion_cleared"


def _alert_subject(object_label, object_id):
    """'Person #17' when both parts are trustworthy, else None.

    None tells the caller to fall back to the pre-ID wording: a half-named subject
    ("Object #17") is worse copy than the old message.
    """
    if not object_label:
        return None
    try:
        oid = int(object_id)
    except (TypeError, ValueError):
        return None
    if oid < 0 or oid == _UNTRACKED_OBJECT_ID:
        return None
    return f"{object_label.capitalize()} #{oid}"


def zone_alert_details(object_id, object_label=None) -> "dict | None":
    """The intrusion's details: its tracked object, so its clip draws that object (app/overlay_track.py incident_pair),
    and what it is (`object_label`), so a console can word the alert in its own language; None with neither."""
    details = {"object_label": str(object_label)} if object_label else {}
    try:
        oid = int(object_id)
    except (TypeError, ValueError):
        oid = None
    if oid is not None and 0 <= oid != _UNTRACKED_OBJECT_ID:
        details = {"object_id": str(oid), **details}
    return details or None


def zone_alert_message(sid, zone, object_label=None, object_id=None):
    """(message, tags, prompt_kind) for one fired zone alert.

    `zone` is the resolved zone dict from zone_by_id, or None when the alert carried no
    resolvable zone id (which takes the intrusion branch, as it always has). Names the
    violating tracked object when the sidecar reported both its class and its tracker id
    -- the same '#<id>' the probe draws on the box in _render_jpeg -- so an operator can
    tie a log line to someone on the video wall. Falls back to the pre-ID wording
    otherwise, which is also what an app newer than its sidecar always gets.
    """
    name = zone.get("name") if zone else None
    subject = _alert_subject(object_label, object_id)

    if zone and zone["intrusion"]["mode"] == "valid":
        if subject:
            message = (f"{subject} left zone {name} on {sid}" if name
                       else f"{subject} left the valid zone on {sid}")
        else:
            message = (f"Person left valid zone {name} on {sid}" if name
                       else f"Person left the valid zone on {sid}")
        return message, ["zone_exit"], "exit"

    if subject:
        message = (f"{subject} entered zone {name} on {sid}" if name
                   else f"{subject} entered a restricted zone on {sid}")
    else:
        message = (f"Zone intrusion detected on {sid} — {name}" if name
                   else f"Zone intrusion detected on {sid}")
    return message, ["zone_intrusion"], "intrusion"


class ZoneAlerts:
    def __init__(self, config: dict, redis_client, alerts, descriptions,
                 running: Callable[[], bool], incidents=None, before=None):
        self._config = config
        # async (sid) -> None, awaited before each entry: reconciles a restart of the sidecar that wrote it
        self._before = before
        self._redis = redis_client
        self._alerts = alerts
        self._descriptions = descriptions
        self._running = running
        # ZoneIncidents when the zones module sends end markers: entries become one incident per zone until it
        # clears. None: every entry is its own alert, as before.
        self._incidents = incidents

    async def run_for(self, sid: str) -> None:
        """Watches this stream's iris:zone_alert:<id> Redis stream -- written by the
        sidecar's nvdsanalytics probe on each debounced zone-boundary-crossing
        detection -- and turns each new entry into an event. Message/tags/prompt_kind come
        from zone_alert_message, which reads this stream's saved zone type ('restricted'
        zones alert on entry; 'valid' zones alert on exit) and names the violating object
        from the alert's object_label/object_id. The zone is looked up fresh per alert, not
        cached, so a mid-session type change takes effect on the next alert with no sampler
        restart. Runs on its own read loop, independent of the VLM analysis batch
        cadence, since a zone alert has no VLM call in its path and shouldn't wait on it."""
        try:
            async for f in read_stream(self._redis, f"iris:zone_alert:{sid}", self._running):
                # A non-cancellation exception here (e.g. a transient DB error bubbling
                # up out of AlertEvents.emit under burst load) must not permanently kill
                # this sampler -- caught and logged per-entry so the reader keeps
                # polling instead of the task silently dying.
                try:
                    await self.handle(sid, f)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error("[%s] alert sampler entry failed", sid, exc_info=True)
        except asyncio.CancelledError:
            pass

    async def handle(self, sid: str, f: dict) -> None:
        """One entry: an intrusion, or its zone clearing. A restart of the sidecar that wrote it is reconciled first
        (`before`), as CountAlerts.handle does."""
        if self._before is not None:
            await self._before(sid)
        if f.get("kind") == INTRUSION_CLEARED:
            # an end marker (zones `end_markers`), never a new intrusion
            if self._incidents is not None:
                await self._incidents.ended(sid, "intrusion", f.get("zone"), f.get("duration"))
            return
        try:
            pts = float(f.get("pts", ""))
        except ValueError:
            pts = None
        # The sidecar tags each alert with the zone that fired and the
        # violating object; resolve the zone fresh (not cached) so a
        # mid-session edit takes effect on the next alert.
        zone_id = f.get("zone")
        zone = zone_by_id(self._config, sid, zone_id) if zone_id else None
        message, tags, prompt_kind = zone_alert_message(
            sid, zone, f.get("object_label"), f.get("object_id"))
        details = zone_alert_details(f.get("object_id"), f.get("object_label"))
        if self._incidents is None:
            await self._alerts.emit(
                sid, message, tags, pts, self._descriptions.describe_intrusions, prompt_kind,
                zone_id=zone_id, details=details)
            return
        await self._incidents.alert(
            sid, "intrusion", zone_id, pts, prompt_kind,
            emit=lambda started: self._alerts.emit(
                sid, message, tags, pts, self._descriptions.describe_intrusions, prompt_kind,
                zone_id=zone_id, incident_started_at=started, details=details),
            again_message=lambda entries: f"{message} ({entries} entries since the zone was clear)")
