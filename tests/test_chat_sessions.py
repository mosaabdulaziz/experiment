import asyncio
from app.chat_sessions import ChatSessionStore


def test_sweep_cursor_roundtrip_and_clear():
    store = ChatSessionStore({"chat": {}})

    async def go():
        s = await store.get_or_create(None)
        assert await store.get_sweep(s.session_id) is None
        await store.set_sweep(s.session_id, history_index=120, since=5.0)
        got = await store.get_sweep(s.session_id)
        assert got == {"history_index": 120, "since": 5.0}
        await store.clear_sweep(s.session_id)
        assert await store.get_sweep(s.session_id) is None
    asyncio.run(go())


def test_get_sweep_unknown_session_returns_none():
    store = ChatSessionStore({"chat": {}})

    async def go():
        assert await store.get_sweep("does-not-exist") is None
    asyncio.run(go())


def test_get_sweep_returns_independent_copy():
    store = ChatSessionStore({"chat": {}})

    async def go():
        s = await store.get_or_create(None)
        await store.set_sweep(s.session_id, history_index=120, since=5.0)
        got = await store.get_sweep(s.session_id)
        got["history_index"] = 999  # mutate the returned copy
        again = await store.get_sweep(s.session_id)
        assert again["history_index"] == 120  # store's copy is unaffected
    asyncio.run(go())
