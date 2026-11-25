from datetime import datetime, timezone
from app.event_tools import resolve_relative


def test_resolve_today_is_utc_midnight():
    out = resolve_relative("today")
    assert out.endswith("T00:00:00Z")
    assert out[:10] == datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_resolve_relative_deltas_are_iso():
    for kw in ("24h", "7d"):
        out = resolve_relative(kw)
        # parseable ISO ending in Z
        datetime.strptime(out, "%Y-%m-%dT%H:%M:%SZ")


def test_resolve_passthrough_and_none():
    assert resolve_relative("2026-06-01T10:00:00Z") == "2026-06-01T10:00:00Z"
    assert resolve_relative(None) is None


import asyncio
import json
from app.event_tools import QUERY_EVENTS_TOOL, dispatch_tool_call
from app.events import EventStore


def test_tool_schema_shape():
    assert QUERY_EVENTS_TOOL["type"] == "function"
    assert QUERY_EVENTS_TOOL["function"]["name"] == "query_events"
    props = QUERY_EVENTS_TOOL["function"]["parameters"]["properties"]
    assert set(["camera", "event_type", "tags", "since", "until", "limit"]).issubset(props)


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 9999}})


def test_dispatch_happy_path(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        await s.save("alert", "red light runner", ["red-light-violation"], camera="cam1")
        out = await dispatch_tool_call(s, "query_events", {"camera": "cam1", "event_type": "alert"})
        return json.loads(out)
    res = asyncio.run(go())
    assert res["count"] == 1
    assert res["events"][0]["camera"] == "cam1"


def test_dispatch_resolves_relative_since(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        await s.save("alert", "x", ["t"], camera="cam0")
        out = await dispatch_tool_call(s, "query_events", {"since": "today"})
        return json.loads(out)
    res = asyncio.run(go())
    assert res["count"] == 1   # saved today


def test_dispatch_unknown_tool(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        return await dispatch_tool_call(s, "nope", {})
    res = json.loads(asyncio.run(go()))
    assert "error" in res


def test_dispatch_bad_args_no_raise(tmp_path):
    async def go():
        s = _store(tmp_path)
        await s.init()
        # limit must be int-ish; pass a bad type to ensure structured error, not exception
        return await dispatch_tool_call(s, "query_events", {"limit": "not-a-number"})
    res = json.loads(asyncio.run(go()))
    assert "error" in res
