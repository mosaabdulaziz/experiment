"""Reading a Redis stream the way every alert sampler needs it: from its
current tail, never losing an entry. Shared by the use cases in app/usecases/
(moved out of StreamManager in the MR !25 review, 2026-10-01)."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

import redis

# Each camera's status key, iris:status:<camera>, written by the pipeline. Its
# presence is how cameras are discovered; its boot_id field changes when the
# pipeline process restarts.
STATUS_PREFIX = "iris:status:"


def stream_tail_id(client, key: str) -> str:
    """The newest id in `key`, or "0-0" when the stream does not exist yet."""
    try:
        tail = client.xrevrange(key, count=1)
    except redis.RedisError:
        return "0-0"
    if not tail:
        return "0-0"
    eid = tail[0][0]
    return eid.decode() if isinstance(eid, bytes) else eid


async def read_stream(client, key: str, running: Callable[[], bool]) -> AsyncIterator[dict]:
    """Yield decoded entries from a Redis stream, starting at its current tail,
    for as long as `running()` is true.

    Deliberately never polls with "$". That cursor re-resolves to the tail on
    EVERY xread, so an entry XADDed between two calls is skipped permanently --
    the next read begins after it. The window is microseconds while the loop is
    hot, but each reader parks a ThreadPoolExecutor worker for a full second per
    call, and with several streams running three readers apiece the pool
    saturates; a reader then waits seconds for a worker and silently drops
    everything that landed meanwhile. Seen in the field as smart-record files the
    sidecar announced and the clip service never received. Advancing a CONCRETE id
    makes starvation cost latency instead of data.

    The tail is resolved once, so entries predating this reader are not replayed.
    """
    last_id = await asyncio.to_thread(stream_tail_id, client, key)
    while running():
        try:
            resp = await asyncio.to_thread(client.xread, {key: last_id}, block=1000, count=10)
        except redis.RedisError:
            await asyncio.sleep(1.0)
            continue
        for _stream_key, entries in resp or []:
            for entry_id, fields in entries:
                last_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                yield {(k.decode() if isinstance(k, bytes) else k):
                       (v.decode() if isinstance(v, bytes) else v)
                       for k, v in fields.items()}


async def read_streams(client, keys: Callable[[], list], running: Callable[[], bool]) -> AsyncIterator[tuple]:
    """Yield (key, entry) from SEVERAL Redis streams with one blocking read -- the same guarantees as `read_stream`
    (from each stream's tail when it is first seen, a concrete id per stream, never "$"), but one parked worker
    thread however many streams there are. `keys()` is asked again every round, so a stream that appears later is
    picked up within a second."""
    last: dict = {}
    while running():
        current = list(keys())
        for key in current:
            if key not in last:
                last[key] = await asyncio.to_thread(stream_tail_id, client, key)
        for key in [k for k in last if k not in current]:
            del last[key]
        if not last:
            await asyncio.sleep(1.0)
            continue
        try:
            resp = await asyncio.to_thread(client.xread, dict(last), block=1000, count=50)
        except redis.RedisError:
            await asyncio.sleep(1.0)
            continue
        for stream_key, entries in resp or []:
            key = stream_key.decode() if isinstance(stream_key, bytes) else stream_key
            for entry_id, fields in entries:
                last[key] = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                yield key, {(k.decode() if isinstance(k, bytes) else k):
                            (v.decode() if isinstance(v, bytes) else v)
                            for k, v in fields.items()}
