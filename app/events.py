import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import aiosqlite

from .features import camera_of, feature_of, features_where


_EVENT_COLUMNS = ("id, timestamp, event_type, message, tags, stream_offset_start,"
                  " stream_offset_end, camera, description, description_status,"
                  " clip_id, clip_status, severity, category, location,"
                  " recommended_action, incident_status, incident_started_at,"
                  " incident_duration_seconds, clips, acknowledged_at, details, snapshot_id")


def _dumps(value) -> str:
    """JSON for a column, a value JSON cannot hold (bytes, a set, an object) kept as its text rather than refused."""
    return json.dumps(value, default=str)


def _row_to_event(row) -> dict:
    """Map one aiosqlite.Row to the event dict shape the API and WebSocket emit."""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "event_type": row["event_type"],
        "message": row["message"],
        "tags": json.loads(row["tags"]),
        "stream_offset_start": row["stream_offset_start"],
        "stream_offset_end": row["stream_offset_end"],
        "camera": row["camera"],
        "description": row["description"],
        "description_status": row["description_status"],
        "clip_id": row["clip_id"],
        "clip_status": row["clip_status"],
        "severity": row["severity"],
        "category": row["category"],
        "location": row["location"],
        "recommended_action": row["recommended_action"],
        # Fire/smoke incident timing (see app/stream_manager.py's
        # _fire_incident_started_at / _resolve_fire_incident); NULL for every
        # other alert kind. "ended_at" is deliberately not a column: the
        # frontend derives it from started_at + duration_seconds once resolved,
        # and from started_at + (now - started_at) while ongoing -- one fewer
        # place for the two to drift apart.
        "incident_status": row["incident_status"],
        "incident_started_at": row["incident_started_at"],
        "incident_duration_seconds": row["incident_duration_seconds"],
        # Every clip attached to this incident, oldest first -- clips[0] is always
        # the same clip clip_id/clip_status point at, kept as an additive list so a
        # consumer that only reads clip_id/clip_status (the pre-multi-clip shape)
        # keeps working unchanged. Empty for every non-fire alert kind, and for a
        # fire alert with no clip at all.
        "clips": json.loads(row["clips"]) if row["clips"] else [],
        "acknowledged_at": row["acknowledged_at"],
        # A module alert's own fields (strings, as the module sent them: proximity's gap, track ids, ...) and the
        # alert clip its snapshot is stored with; None for every other event.
        "details": json.loads(row["details"]) if row["details"] else None,
        "snapshot_id": row["snapshot_id"],
        **_camera_and_feature(row["camera"], json.loads(row["tags"])),
    }


def _camera_and_feature(camera: "str | None", tags: list) -> dict:
    """`camera` is the stream the event came from -- `<camera>@<role>` beside several role sidecars (the safety
    product). camera_id is the camera it shows, feature the safety feature it belongs to (app/features.py)."""
    return {"camera_id": camera_of(camera), "feature": feature_of(camera, tags)}


def save_kwargs_from_result(result: dict) -> dict:
    """Map one validated analyzer event onto EventStore.save's schema kwargs.

    Shared by both backends so the deepstream and local save paths cannot drift.
    A proactive alert reasons inline, so its description is ready at save time --
    no enrichment worker resolves it later, and an event without one must not
    claim a `pending` status nothing will ever fulfil.
    """
    description = result.get("description")
    return {
        "severity": result.get("severity"),
        "category": result.get("category"),
        "location": result.get("location"),
        "recommended_action": result.get("recommended_action"),
        "description": description,
        "description_status": "ready" if description else None,
    }


class EventStore:
    def __init__(self, config: dict):
        evt_cfg = config.get("events", {})
        self._db_path = evt_cfg.get("db_path", "data/events.db")
        self._retention_hours = evt_cfg.get("retention_hours", 48)

    async def init(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT    NOT NULL,
                    event_type          TEXT    NOT NULL,
                    message             TEXT    NOT NULL,
                    tags                TEXT    NOT NULL DEFAULT '[]',
                    stream_offset_start REAL,
                    stream_offset_end   REAL,
                    camera              TEXT
                )
                """
            )
            for col in ("stream_offset_start REAL", "stream_offset_end REAL", "camera TEXT",
                        "description TEXT", "description_status TEXT",
                        "clip_id TEXT", "clip_status TEXT",
                        "severity TEXT", "category TEXT", "location TEXT",
                        "recommended_action TEXT",
                        "incident_status TEXT", "incident_started_at TEXT",
                        "incident_duration_seconds REAL",
                        "clips TEXT NOT NULL DEFAULT '[]'", "acknowledged_at TEXT",
                        "details TEXT", "snapshot_id TEXT"):
                try:
                    await db.execute(f"ALTER TABLE events ADD COLUMN {col}")
                except Exception:
                    pass  # column already exists
            await db.commit()

    async def save(
        self,
        event_type: str,
        message: str,
        tags: list,
        offset_start: float | None = None,
        offset_end: float | None = None,
        camera: str | None = None,
        description_status: str | None = None,
        clip_status: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        location: str | None = None,
        recommended_action: str | None = None,
        description: str | None = None,
        incident_status: str | None = None,
        incident_started_at: str | None = None,
        details: dict | None = None,
    ) -> dict:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tags_json = json.dumps(tags)
        details_json = None if details is None else _dumps(details)
        if severity is None:
            # Back-compat bridge: callers that predate the severity field pass only
            # event_type. Derive the matching severity so no row is ever inconsistent.
            severity = "info" if event_type == "comment" else "warning"
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "INSERT INTO events (timestamp, event_type, message, tags, stream_offset_start,"
                " stream_offset_end, camera, description, description_status, clip_status,"
                " severity, category, location, recommended_action,"
                " incident_status, incident_started_at, details)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, event_type, message, tags_json, offset_start, offset_end, camera,
                 description, description_status, clip_status,
                 severity, category, location, recommended_action,
                 incident_status, incident_started_at, details_json),
            )
            await db.commit()
            row_id = cur.lastrowid
        record = {
            "id": row_id,
            "timestamp": ts,
            "event_type": event_type,
            "message": message,
            "tags": tags,
            "stream_offset_start": offset_start,
            "stream_offset_end": offset_end,
            "camera": camera,
            "description": description,
            "description_status": description_status,
            "clip_id": None,
            "clip_status": clip_status,
            "severity": severity,
            "category": category,
            "location": location,
            "recommended_action": recommended_action,
            "incident_status": incident_status,
            "incident_started_at": incident_started_at,
            "incident_duration_seconds": None,
            "clips": [],
            "acknowledged_at": None,
            "details": None if details_json is None else json.loads(details_json),
            "snapshot_id": None,
            **_camera_and_feature(camera, tags),
        }
        await self._purge_old()
        return record

    async def update_description(self, event_id: int, status: str,
                                 description: str | None) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                "UPDATE events SET description = ?, description_status = ? WHERE id = ?",
                (description, status, event_id),
            )
            await db.commit()
            async with db.execute(
                f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?",
                (event_id,),
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_event(row)

    async def update_clip(self, event_id: int, status: str,
                          clip_id: str | None) -> dict | None:
        """Attach a built preview clip (or a terminal failure status) to an alert.
        Returns None when the row is gone -- the event was cleared while its clip
        was still building, so the caller should skip its broadcast.

        On a successful attach this also seeds the `clips` list with this same
        clip as its first entry, so a fire incident's later append_clip() calls
        (see stream_manager._continue_fire_clip) have something to append to and
        clip_id/clip_status always mirror clips[0]. Any other alert kind only
        ever gets this one call, so its `clips` list just ends up length 1 --
        harmless, and not read by anything that isn't multi-clip aware. Clips
        appended before this one resolved (an incident's escalation clip) stay
        after it."""
        def change(current):
            if status == "ready" and clip_id is not None:
                clips = [{"clip_id": clip_id, "status": "ready"}] + [
                    c for c in current["clips"] if c.get("clip_id") != clip_id]
                return "clip_id = ?, clip_status = ?, clips = ?", (clip_id, status, json.dumps(clips))
            return "clip_id = ?, clip_status = ?", (clip_id, status)
        return await self._modify(event_id, change)

    async def append_clip(self, event_id: int, clip_id: str) -> dict | None:
        """Add one more clip to an incident that already has at least one (via
        update_clip). Used only for a still-open fire/smoke incident's follow-up
        smart-record sessions -- clip_id/clip_status are left untouched, so they
        keep pointing at the original first clip. Returns None when the row is
        gone, same convention as the other update_* methods."""
        def change(current):
            return "clips = ?", (json.dumps(current["clips"] + [{"clip_id": clip_id, "status": "ready"}]),)
        return await self._modify(event_id, change)

    async def unacknowledged(self, limit: int = 200, features=None) -> list[dict]:
        """Alerts nobody has acknowledged yet, newest first (ended incidents included); only those of `features` when
        given -- filtered in the query, so frequent alerts of other features cannot push them past the limit."""
        where, params = "event_type = 'alert' AND acknowledged_at IS NULL", []
        if features is not None:
            clause, params = features_where(features)
            where += f" AND {clause}"
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"SELECT {_EVENT_COLUMNS} FROM events WHERE {where} ORDER BY id DESC LIMIT ?",
                                  (*params, limit)) as cur:
                rows = await cur.fetchall()
        return [_row_to_event(row) for row in rows]

    async def acknowledge(self, event_id: int) -> dict | None:
        """Set acknowledged_at the first time this is called for an event; a
        repeat call is a no-op (COALESCE keeps the original timestamp) rather
        than an error, since the operator's button stays clickable after use."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                "UPDATE events SET acknowledged_at = COALESCE(acknowledged_at, ?) WHERE id = ?",
                (ts, event_id),
            )
            await db.commit()
            async with db.execute(
                f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_event(row)

    async def update_incident(self, event_id: int, status: str,
                              duration_seconds: "float | None") -> dict | None:
        """Attach a fire/smoke incident's resolution to its "started" event row.
        Returns None when the row is gone (event cleared mid-incident), same
        convention as update_clip -- the caller should skip its broadcast."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                "UPDATE events SET incident_status = ?, incident_duration_seconds = ? WHERE id = ?",
                (status, duration_seconds, event_id),
            )
            await db.commit()
            async with db.execute(
                f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_event(row)

    async def get_event(self, event_id: int) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)) as cur:
                row = await cur.fetchone()
        return None if row is None else _row_to_event(row)

    async def update_alert(self, event_id: int, *, severity: str | None = None, message: str | None = None,
                           details: dict | None = None) -> dict | None:
        """An open incident changed: a new severity (the level it started at stays in details as `escalated_from`),
        a new message, `details` merged into its details. Returns None when the row is gone."""
        def change(current):
            merged = dict(current["details"] or {})
            merged.update(details or {})
            if severity is not None and severity != current["severity"]:
                merged.setdefault("escalated_from", current["severity"])
            return ("severity = ?, message = ?, details = ?",
                    (severity or current["severity"], message or current["message"], _dumps(merged)))
        return await self._modify(event_id, change)

    async def set_snapshot(self, event_id: int, snapshot_id: str, exact: bool) -> dict | None:
        """The incident's snapshot is stored with alert clip `snapshot_id`; `exact` False = the closest frame a clip
        had to the minimum gap. Returns None when the row is gone."""
        def change(current):
            merged = dict(current["details"] or {})
            merged["snapshot_exact"] = "1" if exact else "0"
            return "snapshot_id = ?, details = ?", (snapshot_id, _dumps(merged))
        return await self._modify(event_id, change)

    async def _modify(self, event_id: int, change) -> dict | None:
        """Read one event and write its change in ONE write transaction (BEGIN IMMEDIATE), so updates of one incident
        arriving together (escalation, resolve, snapshot, clips) never lose each other's. `change(record)` returns the
        (SET clause, params) to apply. Returns the updated record, None when the row is gone."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                await db.rollback()
                return None
            sets, params = change(_row_to_event(row))
            await db.execute(f"UPDATE events SET {sets} WHERE id = ?", (*params, event_id))
            await db.commit()
            async with db.execute(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)) as cur:
                row = await cur.fetchone()
        return None if row is None else _row_to_event(row)

    async def resolve_stale_pending(self) -> int:
        """Both enrichment statuses, and a fire/smoke incident's "ongoing" status,
        only ever advance from in-process state (a task, or StreamManager's
        _open_fire_incidents dict) -- so anything still 'pending'/'ongoing' at
        startup was stranded by a restart and will never resolve on its own. Mark
        them terminal so the UI never shows a permanent spinner or a ticking
        "still ongoing" incident for something that in reality resolved
        (unrecorded) while nothing was running. Returns the number of rows touched."""
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "UPDATE events SET description_status = 'unavailable'"
                " WHERE description_status = 'pending'")
            touched = cur.rowcount
            cur = await db.execute(
                "UPDATE events SET clip_status = 'unavailable'"
                " WHERE clip_status = 'pending'")
            touched += cur.rowcount
            cur = await db.execute(
                "UPDATE events SET incident_status = 'unknown'"
                " WHERE incident_status = 'ongoing'")
            touched += cur.rowcount
            await db.commit()
        return touched

    async def resolve_stale_incidents_for_camera(self, camera: str) -> list:
        """Force-close every 'ongoing' incident row for one camera. Used when
        StreamManager detects that camera's owning sidecar process (re)started
        mid-incident (see StreamManager._reconcile_sidecar_restart): its
        in-memory FireAlertState was wiped by the restart, so no
        "fire_smoke_resolved" message for that incident can ever arrive --
        the same 'ongoing'-can-only-advance-from-in-process-state situation
        resolve_stale_pending handles at THIS process's own startup, but here
        triggered by the SEPARATE sidecar process restarting while this one
        keeps running. Returns the touched event ids so the caller can
        broadcast an update per row instead of leaving connected clients
        staring at a stuck "ongoing" card until their next full-list refresh."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id FROM events WHERE camera = ? AND incident_status = 'ongoing'",
                (camera,)
            ) as cur:
                ids = [r["id"] for r in await cur.fetchall()]
            if ids:
                await db.execute(
                    "UPDATE events SET incident_status = 'unknown'"
                    " WHERE camera = ? AND incident_status = 'ongoing'",
                    (camera,))
                await db.commit()
        return ids

    async def get_recent(self, limit: int = 50, types: tuple | None = None,
                         camera: str | None = None) -> list[dict]:
        query = f"SELECT {_EVENT_COLUMNS} FROM events"
        clauses: list = []
        params: list = []
        if types:
            clauses.append(f"event_type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if camera is not None:
            clauses.append("camera = ?")
            params.append(camera)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()

        return [_row_to_event(row) for row in reversed(rows)]  # chronological order

    async def query(self, camera: str | None = None, event_type: str | None = None,
                    tags: list[str] | None = None, since: str | None = None,
                    until: str | None = None, limit: int = 20) -> dict:
        """Filtered event lookup for the chat query_events tool. Returns total
        match count plus up to `limit` rows newest-first."""
        clauses: list = []
        params: list = []
        if camera is not None:
            clauses.append("camera = ?")
            params.append(camera)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if tags:
            tag_clauses = []
            for t in tags:
                tag_clauses.append("tags LIKE ?")
                params.append(f'%"{t}"%')
            clauses.append("(" + " OR ".join(tag_clauses) + ")")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"SELECT COUNT(*) AS c FROM events{where}", params) as cur:
                count = (await cur.fetchone())["c"]
            rows_q = (f"SELECT id, timestamp, event_type, message, tags, stream_offset_start,"
                      f" stream_offset_end, camera FROM events{where} ORDER BY id DESC LIMIT ?")
            async with db.execute(rows_q, params + [limit]) as cur:
                rows = await cur.fetchall()

        events = [{
            "id": r["id"], "timestamp": r["timestamp"], "event_type": r["event_type"],
            "message": r["message"], "tags": json.loads(r["tags"]),
            "stream_offset_start": r["stream_offset_start"],
            "stream_offset_end": r["stream_offset_end"], "camera": r["camera"],
        } for r in rows]
        return {"count": count, "events": events}

    async def analytics_summary(self) -> dict:
        """Deterministic aggregate stats over the event table for the chat
        analytics block. Read-only; no schema change."""
        now = datetime.now(timezone.utc)
        midnight = now.strftime("%Y-%m-%dT00:00:00Z")
        cutoff_24h = datetime.fromtimestamp(now.timestamp() - 24 * 3600, tz=timezone.utc) \
            .strftime("%Y-%m-%dT%H:%M:%SZ")

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            async def scalar(q, p=()):
                async with db.execute(q, p) as cur:
                    return (await cur.fetchone())[0]

            total_alerts = await scalar("SELECT COUNT(*) FROM events WHERE event_type='alert'")
            total_comments = await scalar("SELECT COUNT(*) FROM events WHERE event_type='comment'")
            alerts_today = await scalar(
                "SELECT COUNT(*) FROM events WHERE event_type='alert' AND timestamp>=?", (midnight,))
            alerts_24h = await scalar(
                "SELECT COUNT(*) FROM events WHERE event_type='alert' AND timestamp>=?", (cutoff_24h,))

            async with db.execute(
                "SELECT camera, COUNT(*) AS c FROM events WHERE event_type='alert'"
                " GROUP BY camera ORDER BY c DESC") as cur:
                per_camera = [{"camera": r["camera"], "alerts": r["c"]} for r in await cur.fetchall()]

            async with db.execute(
                "SELECT camera, timestamp, message FROM events WHERE event_type='alert'"
                " ORDER BY id DESC LIMIT 1") as cur:
                row = await cur.fetchone()
            most_recent = ({"camera": row["camera"], "timestamp": row["timestamp"],
                            "message": row["message"]} if row else None)

            async with db.execute("SELECT tags FROM events WHERE event_type='alert'") as cur:
                tag_rows = await cur.fetchall()

        counter: Counter = Counter()
        for r in tag_rows:
            for t in json.loads(r["tags"]):
                counter[t] += 1
        top_tags = [{"tag": t, "count": c} for t, c in counter.most_common(5)]

        return {
            "total_alerts": total_alerts, "total_comments": total_comments,
            "alerts_today": alerts_today, "alerts_24h": alerts_24h,
            "per_camera": per_camera, "most_recent_alert": most_recent, "top_tags": top_tags,
        }

    async def clear_all(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute("DELETE FROM events")
            await db.commit()
            return cur.rowcount

    async def clear_by_tag(self, tag: str) -> int:
        """Delete only events carrying `tag` (e.g. 'fire_smoke') -- the
        scoped counterpart to clear_all(), added 2026-09-06 for the
        attendance dashboard's Fire/Smoke page "Clear" button. That page only
        ever shows fire_smoke-tagged incidents (see attendance-frontend's
        useFireAlerts), but clear_all()/DELETE /api/events wipes every alert
        kind (zone intrusion, overcrowding, ...) plus chat/debug logs and
        alert clips -- unrelated state a fire/smoke-scoped button has no
        business touching. Same `tags LIKE` matching query() already uses."""
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute("DELETE FROM events WHERE tags LIKE ?", (f'%"{tag}"%',))
            await db.commit()
            return cur.rowcount

    async def _purge_old(self):
        cutoff_ts = time.time() - self._retention_hours * 3600
        cutoff = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            await db.commit()
