"""Intrusion and overcrowding as incidents with an end -- when the zones module sends end markers (its
`end_markers` setting; the safety product).

An incident is one event per (stream, kind, zone): the first alert opens it (an ongoing incident), a later one raises
the same event in place (with a follow-up clip of that moment), and the zone's end marker resolves it with its
duration. Open incidents live in memory, like fire's and the modules': a restart of the camera's sidecar leaves its
rows to FireIncidents, which closes them as "unknown"; this forgets them (sidecar_restarted)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..severity import zone_alert_defaults

logger = logging.getLogger("iris.stream")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ZoneIncidents:
    def __init__(self, alerts):
        self._alerts = alerts
        self.open: dict = {}        # (stream, kind, zone) -> {"event_id", "entries"}

    def event_id(self, sid: str, kind: str, zone: "str | None") -> "int | None":
        entry = self.open.get((sid, kind, zone))
        return None if entry is None else entry["event_id"]

    def opened(self, sid: str, kind: str, zone: "str | None", event_id: "int | None") -> None:
        if event_id is not None:
            self.open[(sid, kind, zone)] = {"event_id": event_id, "entries": 1}

    def repeat(self, sid: str, kind: str, zone: "str | None") -> int:
        """Count another alert on the open incident; returns how many it has had."""
        entry = self.open[(sid, kind, zone)]
        entry["entries"] += 1
        return entry["entries"]

    async def alert(self, sid: str, kind: str, zone: "str | None", pts, prompt_kind: str, emit,
                    again_message, details: "dict | None" = None) -> None:
        """An alert in a zone: the first opens the zone's incident (`emit(incident_started_at)` saves it and returns
        its event id); while it is open, another raises it in place -- same severity, `again_message(entries)`, a
        follow-up clip of this moment."""
        event_id = self.event_id(sid, kind, zone)
        if event_id is None:
            self.opened(sid, kind, zone, await emit(now_iso()))
            return
        entries = self.repeat(sid, kind, zone)
        if not await self._alerts.escalate(event_id, sid, pts, zone_alert_defaults(prompt_kind)[0],
                                           again_message(entries), {"entries": entries, **(details or {})},
                                           zone_id=zone, kind=prompt_kind):
            # its row is gone (deleted, or past retention): this alert opens a new incident, never raises a lost one
            self.open.pop((sid, kind, zone), None)
            self.opened(sid, kind, zone, await emit(now_iso()))

    async def ended(self, sid: str, kind: str, zone: "str | None", duration_raw) -> None:
        entry = self.open.pop((sid, kind, zone), None)
        if entry is None:
            logger.info("[%s] %s ended in zone %s with no open incident (a restart, most likely)", sid, kind, zone)
            return
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = None
        await self._alerts.resolve_incident(entry["event_id"], sid, duration, None)

    async def sidecar_restarted(self, sid: str) -> None:
        """The stream's sidecar restarted: its open incidents' rows are closed as "unknown" (FireIncidents)."""
        for key in [k for k in self.open if k[0] == sid]:
            self.open.pop(key)
