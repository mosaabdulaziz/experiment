import asyncio
from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 9999}})


def test_analytics_empty_db(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        return await s.analytics_summary()
    out = asyncio.run(go())
    assert out["total_alerts"] == 0
    assert out["total_comments"] == 0
    assert out["per_camera"] == []
    assert out["most_recent_alert"] is None
    assert out["top_tags"] == []


def test_analytics_counts_and_breakdown(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        await s.save("alert", "a1", ["red-light-violation", "car"], camera="cam1")
        await s.save("alert", "a2", ["red-light-violation"], camera="cam1")
        await s.save("alert", "a3", ["truck"], camera="cam0")
        await s.save("comment", "c1", ["low-traffic"], camera="cam0")
        return await s.analytics_summary()
    out = asyncio.run(go())
    assert out["total_alerts"] == 3
    assert out["total_comments"] == 1
    assert out["alerts_today"] == 3        # all saved "now" (UTC)
    assert out["alerts_24h"] == 3
    # per_camera ordered desc by alert count
    assert out["per_camera"][0] == {"camera": "cam1", "alerts": 2}
    assert {"camera": "cam0", "alerts": 1} in out["per_camera"]
    assert out["most_recent_alert"]["message"] == "a3"
    assert out["most_recent_alert"]["camera"] == "cam0"
    # top_tags tally over alerts, desc
    top = {t["tag"]: t["count"] for t in out["top_tags"]}
    assert top["red-light-violation"] == 2
    assert top["truck"] == 1
    assert "low-traffic" not in top        # comments excluded from top_tags


def test_top_tags_truncates_to_five(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        for i in range(7):
            await s.save("alert", f"a{i}", [f"tag{i}"], camera="cam0")
        return await s.analytics_summary()
    out = asyncio.run(go())
    assert len(out["top_tags"]) == 5
