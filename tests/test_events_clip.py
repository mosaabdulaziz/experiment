import asyncio

from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})


def test_save_records_clip_status_and_defaults_to_none(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        plain = await store.save("alert", "x", [], camera="cam0")
        pending = await store.save("alert", "y", [], camera="cam0", clip_status="pending")
        return plain, pending

    plain, pending = asyncio.run(run())
    assert plain["clip_status"] is None
    assert plain["clip_id"] is None
    assert pending["clip_status"] == "pending"


def test_update_clip_sets_id_and_status_and_returns_full_record(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "intrusion", ["zone_intrusion"], camera="cam0",
                               clip_status="pending")
        return await store.update_clip(rec["id"], "ready", "abc123")

    updated = asyncio.run(run())
    assert updated["clip_status"] == "ready"
    assert updated["clip_id"] == "abc123"
    assert updated["message"] == "intrusion"        # rest of the record intact
    assert updated["tags"] == ["zone_intrusion"]


def test_update_clip_returns_none_for_a_missing_row(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        # The event was cleared while its clip was still building.
        return await store.update_clip(4242, "ready", "abc123")

    assert asyncio.run(run()) is None


def test_get_recent_exposes_clip_fields(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await store.update_clip(rec["id"], "ready", "cid")
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[-1]["clip_id"] == "cid"
    assert rows[-1]["clip_status"] == "ready"


def test_update_clip_seeds_clips_list_on_ready(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "fire detected", ["fire_smoke"], camera="cam-7",
                               clip_status="pending")
        return await store.update_clip(rec["id"], "ready", "cid1")

    updated = asyncio.run(run())
    assert updated["clips"] == [{"clip_id": "cid1", "status": "ready"}]


def test_update_clip_unavailable_leaves_clips_empty(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        return await store.update_clip(rec["id"], "unavailable", None)

    updated = asyncio.run(run())
    assert updated["clips"] == []


def test_append_clip_adds_to_existing_list_without_touching_primary(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "fire detected", ["fire_smoke"], camera="cam-7",
                               clip_status="pending")
        await store.update_clip(rec["id"], "ready", "cid1")
        return await store.append_clip(rec["id"], "cid2")

    updated = asyncio.run(run())
    assert updated["clips"] == [
        {"clip_id": "cid1", "status": "ready"},
        {"clip_id": "cid2", "status": "ready"},
    ]
    assert updated["clip_id"] == "cid1"      # primary untouched by the append


def test_append_clip_returns_none_for_a_missing_row(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        return await store.append_clip(4242, "cid1")

    assert asyncio.run(run()) is None


def test_acknowledge_sets_timestamp_once(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        rec = await store.save("alert", "fire detected", ["fire_smoke"], camera="cam-7")
        first = await store.acknowledge(rec["id"])
        second = await store.acknowledge(rec["id"])   # idempotent
        return first, second

    first, second = asyncio.run(run())
    assert first["acknowledged_at"] is not None
    assert second["acknowledged_at"] == first["acknowledged_at"]


def test_resolve_stale_pending_clears_both_enrichment_statuses(tmp_path):
    async def run():
        store = _store(tmp_path)
        await store.init()
        stranded = await store.save("alert", "x", [], camera="cam0",
                                    description_status="pending", clip_status="pending")
        settled = await store.save("alert", "y", [], camera="cam0", clip_status="pending")
        await store.update_clip(settled["id"], "ready", "cid")
        n = await store.resolve_stale_pending()
        rows = {r["id"]: r for r in await store.get_recent(limit=10)}
        return n, rows, stranded["id"], settled["id"]

    n, rows, stranded_id, settled_id = asyncio.run(run())
    assert n == 2                                          # one description + one clip
    assert rows[stranded_id]["description_status"] == "unavailable"
    assert rows[stranded_id]["clip_status"] == "unavailable"
    assert rows[settled_id]["clip_status"] == "ready"      # already terminal, untouched
