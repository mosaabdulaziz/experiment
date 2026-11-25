import asyncio
import json
from types import SimpleNamespace
from app.analyzer import Analyzer


def _analyzer(tools=False):
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                     "monitoring": {}, "sampling": {},
                     "chat": {"tools_enabled": tools, "max_tool_calls": 3}})


def _stream_response(text):
    """Return a fake create() coroutine yielding openai-style streaming chunks."""
    async def create(*a, **k):
        async def gen():
            for ch in text:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        return gen()
    return create


def _collect(agen):
    async def go():
        items = []
        async for it in agen:
            items.append(it)
        return items
    return asyncio.run(go())


def test_chat_multi_streams_tokens_no_tools():
    az = _analyzer(tools=False)
    az._client.chat.completions.create = _stream_response("hi")
    items = _collect(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None, prior_turns=[]))
    tokens = [it["delta"] for it in items if it["type"] == "token"]
    assert "".join(tokens) == "hi"


def test_chat_multi_strips_think_in_stream():
    az = _analyzer(tools=False)
    az._client.chat.completions.create = _stream_response("<think>x</think>ok")
    items = _collect(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None, prior_turns=[]))
    tokens = "".join(it["delta"] for it in items if it["type"] == "token")
    assert tokens == "ok"


def test_chat_multi_runs_tool_loop_then_streams(tmp_path):
    from app.events import EventStore
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 9999}})

    tool_call = SimpleNamespace(
        id="c1", type="function",
        function=SimpleNamespace(name="query_events", arguments=json.dumps({"event_type": "alert"})))
    first = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[tool_call]))])
    state = {"n": 0}

    async def create(*a, **k):
        if k.get("stream"):
            async def gen():
                for ch in "done":
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
            return gen()
        state["n"] += 1
        return first

    az = _analyzer(tools=True)
    az._client.chat.completions.create = create

    async def go():
        await store.init()
        await store.save("alert", "x", ["t"], camera="cam0")
        items = []
        async for it in az.chat_multi({"cam0": [b"a"]}, "how many alerts?",
                                      store=store, history=[], analytics=None, prior_turns=[]):
            items.append(it)
        return items
    items = asyncio.run(go())
    statuses = [it for it in items if it["type"] == "status"]
    tokens = "".join(it["delta"] for it in items if it["type"] == "token")
    assert any(s["stage"] == "querying_events" for s in statuses)
    assert tokens == "done"
    assert state["n"] >= 1   # at least one non-stream tool round happened


def test_chat_multi_use_tools_false_skips_tool_loop(tmp_path):
    from app.events import EventStore
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 9999}})
    az = _analyzer(tools=True)
    az._client.chat.completions.create = _stream_response("NOT_FOUND")

    async def go():
        await store.init()
        items = []
        async for it in az.chat_multi({"cam0": [b"a"]}, "q", store=store, history=[],
                                      analytics=None, prior_turns=[], use_tools=False, sweep=True):
            items.append(it)
        return items
    items = asyncio.run(go())
    # No tool round => no querying_events status; tokens reconstruct the marker.
    assert not any(it["type"] == "status" for it in items)
    assert "".join(it["delta"] for it in items if it["type"] == "token") == "NOT_FOUND"


def test_chat_multi_sweep_injects_marker_instruction_into_system_prompt():
    az = _analyzer(tools=False)
    captured = {}

    async def create(*a, **k):
        captured["messages"] = k.get("messages")
        async def gen():
            for ch in "NOT_FOUND":
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        return gen()

    az._client.chat.completions.create = create
    _collect(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None,
                           prior_turns=[], use_tools=False, sweep=True))
    sys_msg = captured["messages"][0]
    assert sys_msg["role"] == "system"
    # The sweep suffix is the only source of the NOT_FOUND token in the system prompt.
    assert "NOT_FOUND" in sys_msg["content"]
    assert "FOUND" in sys_msg["content"]


def test_chat_multi_no_sweep_omits_marker_instruction():
    az = _analyzer(tools=False)
    captured = {}

    async def create(*a, **k):
        captured["messages"] = k.get("messages")
        async def gen():
            for ch in "hi":
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        return gen()

    az._client.chat.completions.create = create
    _collect(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None,
                           prior_turns=[], use_tools=False, sweep=False))
    assert "NOT_FOUND" not in captured["messages"][0]["content"]
