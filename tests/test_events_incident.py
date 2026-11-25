import asyncio

from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})


def test_save_records_incident_fields_and_defaults_to_none(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        plain = await store.save("alert", "x", [], camera="cam0")
        ongoing = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                                   incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        return plain, ongoing

    plain, ongoing = asyncio.run(run())
    assert plain["incident_status"] is None
    assert plain["incident_started_at"] is None
    assert plain["incident_duration_seconds"] is None
    assert ongoing["incident_status"] == "ongoing"
    assert ongoing["incident_started_at"] == "2026-09-04T10:00:00Z"
    assert ongoing["incident_duration_seconds"] is None


def test_update_incident_resolves_and_returns_full_record(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                               incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        return await store.update_incident(rec["id"], "resolved", 12.5)

    updated = asyncio.run(run())
    assert updated["incident_status"] == "resolved"
    assert updated["incident_duration_seconds"] == 12.5
    assert updated["incident_started_at"] == "2026-09-04T10:00:00Z"   # untouched
    assert updated["message"] == "Fire detected on cam0"              # rest of record intact


def test_update_incident_returns_none_for_a_missing_row(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        # The event was cleared while its incident was still open.
        return await store.update_incident(4242, "resolved", 5.0)

    assert asyncio.run(run()) is None


def test_get_recent_exposes_incident_fields(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "Smoke detected on cam0", ["fire_smoke"], camera="cam0",
                               incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        await store.update_incident(rec["id"], "resolved", 8.0)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[-1]["incident_status"] == "resolved"
    assert rows[-1]["incident_duration_seconds"] == 8.0


def test_resolve_stale_pending_marks_orphaned_ongoing_incidents_unknown(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        stranded = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                                    incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        settled = await store.save("alert", "Smoke detected on cam1", ["fire_smoke"], camera="cam1",
                                   incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        await store.update_incident(settled["id"], "resolved", 4.0)
        n = await store.resolve_stale_pending()
        rows = {r["id"]: r for r in await store.get_recent(limit=10)}
        return n, rows, stranded["id"], settled["id"]

    n, rows, stranded_id, settled_id = asyncio.run(run())
    assert n == 1
    assert rows[stranded_id]["incident_status"] == "unknown"
    assert rows[settled_id]["incident_status"] == "resolved"   # already terminal, untouched


def test_resolve_stale_incidents_for_camera_only_touches_that_camera(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        stranded = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                                    incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        other_cam = await store.save("alert", "Smoke detected on cam1", ["fire_smoke"], camera="cam1",
                                     incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        same_cam_resolved = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                                             incident_status="ongoing",
                                             incident_started_at="2026-09-04T09:00:00Z")
        await store.update_incident(same_cam_resolved["id"], "resolved", 4.0)
        touched_ids = await store.resolve_stale_incidents_for_camera("cam0")
        rows = {r["id"]: r for r in await store.get_recent(limit=10)}
        return touched_ids, rows, stranded["id"], other_cam["id"], same_cam_resolved["id"]

    touched_ids, rows, stranded_id, other_cam_id, resolved_id = asyncio.run(run())
    assert touched_ids == [stranded_id]
    assert rows[stranded_id]["incident_status"] == "unknown"
    assert rows[other_cam_id]["incident_status"] == "ongoing"       # different camera, untouched
    assert rows[resolved_id]["incident_status"] == "resolved"       # already terminal, untouched


def test_resolve_stale_incidents_for_camera_with_nothing_ongoing_returns_empty(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        return await store.resolve_stale_incidents_for_camera("cam0")

    assert asyncio.run(run()) == []
