import asyncio
import aiosqlite
from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "events.db"), "retention_hours": 48}})


def test_save_and_filter_by_camera(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        await store.save("comment", "car on cam0", [], camera="cam0")
        await store.save("comment", "car on cam1", [], camera="cam1")
        all_ev = await store.get_recent(limit=10)
        cam1 = await store.get_recent(limit=10, camera="cam1")
        return all_ev, cam1

    all_ev, cam1 = asyncio.run(run())
    assert {e["camera"] for e in all_ev} == {"cam0", "cam1"}
    assert [e["message"] for e in cam1] == ["car on cam1"]


def test_migration_adds_camera_to_legacy_db(tmp_path):
    db_path = tmp_path / "events.db"

    async def make_legacy():
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,"
                " event_type TEXT NOT NULL, message TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]')"
            )
            await db.execute("INSERT INTO events (timestamp, event_type, message, tags)"
                             " VALUES ('2026-01-01T00:00:00Z','comment','legacy','[]')")
            await db.commit()

    async def run():
        await make_legacy()
        store = EventStore({"events": {"db_path": str(db_path), "retention_hours": 48}})
        await store.init()                              # must ALTER TABLE, not crash
        rows = await store.get_recent(limit=10)
        return rows

    rows = asyncio.run(run())
    assert rows[0]["message"] == "legacy"
    assert rows[0]["camera"] is None                    # legacy row → null camera


def test_save_with_pending_description_status(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        rows = await store.get_recent(limit=10)
        return record, rows

    record, rows = asyncio.run(run())
    assert record["description_status"] == "pending"
    assert record["description"] is None
    assert rows[0]["description_status"] == "pending"


def test_update_description_patches_row_and_returns_full_record(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        saved = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                 camera="cam0", description_status="pending")
        updated = await store.update_description(saved["id"], "ready", "A person climbed the fence.")
        rows = await store.get_recent(limit=10)
        return saved, updated, rows

    saved, updated, rows = asyncio.run(run())
    assert updated["id"] == saved["id"]
    assert updated["description_status"] == "ready"
    assert updated["description"] == "A person climbed the fence."
    assert updated["camera"] == "cam0"          # untouched fields still round-trip
    assert updated["tags"] == ["zone_intrusion"]
    assert rows[0]["description"] == "A person climbed the fence."


def test_update_description_returns_none_for_unknown_id(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        return await store.update_description(999, "ready", "x")

    assert asyncio.run(run()) is None


def test_migration_adds_description_columns_to_legacy_db(tmp_path):
    db_path = tmp_path / "events.db"

    async def make_legacy():
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,"
                " event_type TEXT NOT NULL, message TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]')"
            )
            await db.execute("INSERT INTO events (timestamp, event_type, message, tags)"
                             " VALUES ('2026-01-01T00:00:00Z','comment','legacy','[]')")
            await db.commit()

    async def run():
        await make_legacy()
        store = EventStore({"events": {"db_path": str(db_path), "retention_hours": 48}})
        await store.init()
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description"] is None
    assert rows[0]["description_status"] is None
