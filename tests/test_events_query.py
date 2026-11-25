import asyncio
from app.events import EventStore


def _store(tmp_path):
    s = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 9999}})
    return s


async def _seed(s):
    await s.init()
    await s.save("alert", "red light runner", ["red-light-violation", "car"], camera="cam1")
    await s.save("comment", "sparse traffic", ["low-traffic"], camera="cam0")
    await s.save("alert", "truck idling", ["truck"], camera="cam1")


def test_query_filters_by_camera_and_type(tmp_path):
    async def go():
        s = _store(tmp_path)
        await _seed(s)
        res = await s.query(camera="cam1", event_type="alert")
        return res
    res = asyncio.run(go())
    assert res["count"] == 2
    assert len(res["events"]) == 2
    assert all(e["camera"] == "cam1" and e["event_type"] == "alert" for e in res["events"])
    assert res["events"][0]["timestamp"] >= res["events"][1]["timestamp"]  # newest-first


def test_query_tag_any_match(tmp_path):
    async def go():
        s = _store(tmp_path)
        await _seed(s)
        return await s.query(tags=["truck", "nonexistent"])
    res = asyncio.run(go())
    assert res["count"] == 1
    assert res["events"][0]["message"] == "truck idling"


def test_query_count_ignores_limit_but_events_capped(tmp_path):
    async def go():
        s = _store(tmp_path)
        await _seed(s)
        return await s.query(limit=1)
    res = asyncio.run(go())
    assert res["count"] == 3
    assert len(res["events"]) == 1


def test_query_since_until(tmp_path):
    async def go():
        s = _store(tmp_path)
        await _seed(s)
        # everything is "now"; an until in the far past returns nothing
        return await s.query(until="2000-01-01T00:00:00Z")
    res = asyncio.run(go())
    assert res["count"] == 0
    assert res["events"] == []


def test_query_empty_db(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        return await s.query()
    res = asyncio.run(go())
    assert res == {"count": 0, "events": []}
