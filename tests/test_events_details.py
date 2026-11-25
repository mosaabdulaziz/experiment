"""Event details (a module alert's own fields) and in-place updates of an open incident."""
import asyncio

import aiosqlite

from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})


def test_details_round_trip_and_default_to_none(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        plain = await store.save("alert", "x", [], camera="c")
        rich = await store.save("alert", "y", ["proximity"], camera="c",
                                details={"gap": "0.15", "vehicle_class": "truck"})
        return plain, rich, {e["id"]: e for e in await store.get_recent(limit=10)}

    plain, rich, rows = asyncio.run(run())
    assert plain["details"] is None and rows[plain["id"]]["details"] is None
    assert rich["details"] == rows[rich["id"]]["details"] == {"gap": "0.15", "vehicle_class": "truck"}
    assert rows[rich["id"]]["snapshot_id"] is None


def test_update_alert_raises_severity_merges_details_and_remembers_the_old_level(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m1", ["proximity"], camera="c", severity="warning", details={"gap": "0.40"})
        rec = await store.update_alert(ev["id"], severity="critical", message="m2", details={"gap": "0.12"})
        same = await store.update_alert(ev["id"], details={"duration_s": "3.4"})
        gone = await store.update_alert(999, severity="critical")
        return rec, same, gone

    rec, same, gone = asyncio.run(run())
    assert (rec["severity"], rec["message"]) == ("critical", "m2")
    assert rec["details"] == {"gap": "0.12", "escalated_from": "warning"}
    assert same["severity"] == "critical" and same["details"]["duration_s"] == "3.4"
    assert gone is None


def test_set_snapshot_records_the_clip_and_whether_the_frame_is_exact(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", [], camera="c")
        return await store.set_snapshot(ev["id"], "clipabc", exact=False)

    rec = asyncio.run(run())
    assert rec["snapshot_id"] == "clipabc" and rec["details"] == {"snapshot_exact": "0"}


def test_a_first_clip_finishing_after_an_appended_one_keeps_both(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", [], camera="c", clip_status="pending")
        await store.append_clip(ev["id"], "escalation")
        return await store.update_clip(ev["id"], "ready", "first")

    rec = asyncio.run(run())
    assert rec["clip_id"] == "first"
    assert [c["clip_id"] for c in rec["clips"]] == ["first", "escalation"]


def test_an_old_database_gains_the_new_columns(tmp_path):
    async def run():
        async with aiosqlite.connect(str(tmp_path / "e.db")) as db:
            await db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,"
                             " event_type TEXT NOT NULL, message TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',"
                             " stream_offset_start REAL, stream_offset_end REAL, camera TEXT)")
            await db.commit()
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", [], camera="c", details={"k": "v"})
        return await store.get_event(ev["id"])

    assert asyncio.run(run())["details"] == {"k": "v"}


def test_concurrent_updates_of_one_incident_lose_nothing(tmp_path):
    """An escalation, the resolve and the snapshot can update one incident at the same moment: each update must see
    the others' (read-modify-write in one transaction)."""
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", ["proximity"], camera="c", details={})
        await asyncio.gather(*(store.update_alert(ev["id"], details={f"k{i}": str(i)}) for i in range(20)))
        await asyncio.gather(*(store.append_clip(ev["id"], f"clip{i}") for i in range(10)))
        return await store.get_event(ev["id"])

    row = asyncio.run(run())
    assert {f"k{i}" for i in range(20)} <= set(row["details"])
    assert sorted(c["clip_id"] for c in row["clips"]) == sorted(f"clip{i}" for i in range(10))


def test_a_details_value_json_cannot_hold_is_stored_as_its_text(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", [], camera="c", details={"raw": b"\x01", "ids": {3}})
        return ev, await store.update_alert(ev["id"], details={"when": object})

    saved, updated = asyncio.run(run())
    assert saved["details"]["raw"] == "b'\\x01'" or isinstance(saved["details"]["raw"], str)
    assert isinstance(updated["details"]["ids"], str) and isinstance(updated["details"]["when"], str)


def test_escalated_from_keeps_the_level_the_incident_started_at(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        ev = await store.save("alert", "m", [], camera="c", severity="info")
        await store.update_alert(ev["id"], severity="warning")
        return await store.update_alert(ev["id"], severity="critical")

    assert asyncio.run(run())["details"]["escalated_from"] == "info"

