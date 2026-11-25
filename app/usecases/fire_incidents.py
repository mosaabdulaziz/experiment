"""Fire and smoke incidents: an incident opens when a "started" alert is saved,
gets follow-up clips while it stays open, and closes with its true duration --
or is force-closed if the pipeline restarts underneath it."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import redis

from perception.engine.sources import split_stream_key_id

from ..redis_streams import STATUS_PREFIX

logger = logging.getLogger("iris.stream")


class FireIncidents:
    def __init__(self, store, conn_manager, clip_service, redis_client):
        self._store = store
        self._conn = conn_manager
        self._clip_service = clip_service
        self._redis = redis_client
        # (sid, cls) -> event id, for fire/smoke incidents currently "ongoing" --
        # set when a "started" alert saves, consumed when its matching
        # "fire_smoke_resolved" entry arrives on the same Redis stream. Purely
        # in-process: a restart mid-incident leaves it un-resolvable (see
        # resolve's warning), same tradeoff the description tasks already make
        # for in-flight VLM jobs.
        self.open_incidents: dict = {}
        # sid -> boot_id last seen on that source's iris:status:<id> -- lets the
        # discovery loop notice a source's owning sidecar process (re)starting
        # mid-run (see reconcile_sidecar_restart) even though the app process
        # itself never restarted and open_incidents alone can't tell.
        self.sidecar_boot_ids: dict = {}
        # Called with the camera id when its sidecar restarted mid-run, before its rows are force-closed: other
        # use cases' in-memory incidents (ModuleAlerts) for that camera go too.
        self.on_restart: list = []

    def track(self, sid: str, cls: "str | None", event_id: int) -> None:
        """A "started" alert for this camera and class was saved as `event_id`."""
        self.open_incidents[(sid, cls)] = event_id

    def read_boot_id(self, sid: str) -> "str | None":
        """The boot_id currently published on this source's iris:status:<id>, or
        None if unreadable/absent (an older sidecar build, a transient Redis
        error, or the key not existing yet) -- never treated as a restart signal
        by the caller, only a change between two successfully-read values is."""
        try:
            raw = self._redis.get(f"{STATUS_PREFIX}{sid}")
        except redis.RedisError:
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw).get("boot_id")
        except (ValueError, AttributeError):
            return None

    async def check_sidecar_boot_id(self, sid: str) -> None:
        """Detect this source's owning sidecar process (re)starting mid-run -- a
        crash-loop restart, most often -- and reconcile anything it left
        stranded. A first sighting of `sid` just records its boot_id: resolving
        anything already stale for it is resolve_stale_pending's job at THIS
        (app) process's own startup (app/events.py), not this loop's. Only a
        CHANGE from a previously-seen boot_id means the owning sidecar restarted
        while this app process kept running -- the sidecar container's
        RestartCount climbing while the app container's own stays 0 is exactly
        the scenario this closes; see reconcile_sidecar_restart."""
        boot_id = self.read_boot_id(sid)
        if boot_id is None:
            return
        previous = self.sidecar_boot_ids.get(sid)
        self.sidecar_boot_ids[sid] = boot_id
        if previous is not None and previous != boot_id:
            logger.warning("[%s] sidecar boot_id changed (%s -> %s): owning process "
                           "restarted mid-run; reconciling any stranded incident(s)",
                           sid, previous, boot_id)
            await self.reconcile_sidecar_restart(sid)

    async def stream_gone(self, sid: str) -> None:
        """A role sidecar's stream left discovery (`<camera>@<role>`, the safety stack: Apply stopped that feature on
        the camera, or its sidecar died) with its status key gone: nothing it had open can be resolved now -- close it
        as a restart does. Every other product's streams, and a stream still publishing its status (it only dropped
        out of one discovery pass), keep their incidents."""
        if not split_stream_key_id(sid)[1]:
            return
        try:
            if self._redis.exists(f"{STATUS_PREFIX}{sid}"):
                return
        except redis.RedisError:
            return
        self.sidecar_boot_ids.pop(sid, None)        # if it comes back, that is a first sighting
        await self.reconcile_sidecar_restart(sid)

    async def reconcile_sidecar_restart(self, sid: str) -> None:
        """A source's owning sidecar process just restarted: its in-memory
        FireAlertState is gone, so any incident it had open on this source can
        never receive a "fire_smoke_resolved" message now -- force-resolve them
        (to 'unknown', resolve_stale_pending's existing terminal-status
        convention) rather than leave them "ongoing" forever."""
        for callback in self.on_restart:
            try:
                await callback(sid)
            except Exception:
                logger.error("[%s] a sidecar-restart callback failed", sid, exc_info=True)
        for cls in [c for (s, c) in self.open_incidents if s == sid]:
            self.open_incidents.pop((sid, cls), None)
        touched_ids = await self._store.resolve_stale_incidents_for_camera(sid)
        for event_id in touched_ids:
            await self._conn.broadcast({
                "type": "event_update", "id": event_id, "incident_status": "unknown"})
        if touched_ids:
            logger.info("[%s] force-resolved %d stranded incident(s) to 'unknown' "
                        "after a sidecar restart: %s", sid, len(touched_ids), touched_ids)

    async def continue_clip(self, sid: str, path: str, recorded_at: "float | None" = None) -> None:
        """Attach a smart-record file that arrived with no queued job to whichever
        fire/smoke incident on this source is still open, and broadcast the
        updated clip list.

        `open_incidents` is keyed by (sid, cls) and popped at resolve time
        (see resolve), so a file that lands after the incident has
        already resolved finds nothing here and is silently dropped -- the tail
        end of a long incident's very last session, at most a few seconds of
        footage largely covered by the previous clip. Not worth chasing further.
        """
        open_classes = [cls for (s, cls) in self.open_incidents if s == sid]
        if not open_classes:
            logger.warning("[%s] smart-record file %s arrived with no queued job and "
                           "no open fire incident -- dropping", sid, path)
            return
        status, clip_id = await self._clip_service.store_standalone(sid, path, recorded_at)
        if status != "ready" or clip_id is None:
            logger.warning("[%s] could not store follow-up clip %s for incident(s) %s",
                           sid, path, open_classes)
            return
        for cls in open_classes:
            event_id = self.open_incidents.get((sid, cls))
            if event_id is None:
                continue
            record = await self._store.append_clip(event_id, clip_id)
            if record is not None:
                await self._conn.broadcast({
                    "type": "event_update", "id": event_id, "clips": record["clips"]})

    def started_at(self, onset_age_raw) -> "str | None":
        """Wall-clock onset for a just-started fire/smoke incident: `onset_age` (how
        many seconds ago, on the sidecar's own pts clock, the condition actually
        began -- see FireIncidentEvent) subtracted from THIS process's current wall
        clock. Approximates "now" as the sidecar's confirming pts moment, which is
        accurate to well under a second since this sampler processes the XADD
        entry essentially as soon as it's written -- the same pts/wall-clock
        bridging approach _maybe_start_svr's age arithmetic already relies on.
        Returns None (no incident timing recorded) if onset_age is missing/malformed,
        e.g. a sidecar older than this app."""
        try:
            onset_age = float(onset_age_raw)
        except (TypeError, ValueError):
            return None
        started = datetime.now(timezone.utc) - timedelta(seconds=onset_age)
        return started.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def resolve(self, sid: str, cls: "str | None", duration_raw) -> None:
        """Handles a "fire_smoke_resolved" stream entry: attaches the incident's true
        duration to whichever "started" event opened it, and broadcasts the update.
        Silently drops entries with no matching open incident -- either this class
        never confirmed on this stream (shouldn't happen -- the sidecar only emits
        "resolved" after a matching "started") or this process restarted mid-incident,
        in which case the row was already marked 'unknown' at startup and nothing here
        could complete it correctly anyway."""
        event_id = self.open_incidents.pop((sid, cls), None)
        if event_id is None:
            logger.warning("[%s] %s resolved with no open incident tracked -- dropping "
                           "(a restart mid-incident, most likely)", sid, cls)
            return
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            logger.warning("[%s] %s resolved with a malformed duration %r", sid, cls, duration_raw)
            return
        record = await self._store.update_incident(event_id, "resolved", duration)
        if record is not None:
            await self._conn.broadcast({
                "type": "event_update", "id": event_id,
                "incident_status": "resolved", "incident_duration_seconds": duration})
