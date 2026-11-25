"""Fire/smoke incident lifecycle: a "started" alert tracked through to its
"resolved" duration, end to end through StreamManager's alert sampler and
EventStore -- the app-side half of perception/modules/fire_alerts/state.py's FireIncidentEvent.
"""

import asyncio

import fakeredis

from app.events import EventStore
from app.stream_manager import StreamManager


def _cfg():
    return {
        "sampling": {"frames_per_batch": 4, "interval_seconds": 4},
        "monitoring": {"cooldown_seconds": 8, "alert_cooldown_seconds": 4, "max_history": 3},
        "dedup": {"enabled": False, "backend": "local"},
        "perception": {"redis_url": "redis://unused", "queue_depth": 3,
                       "discovery_interval_seconds": 5},
        "zones": {"sidecar_config_path": "/nonexistent/sidecar.yaml"},
    }


class _StubConn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


async def _drain_sampler(sm, sid, entries, settle=0.3):
    """Start the sampler, feed `entries` onto its Redis stream, give it a beat
    to process, then stop it. Mirrors test_stream_manager.py's own pattern."""
    sm._running = True
    task = asyncio.create_task(sm.count_alerts.run_for(sid))
    await asyncio.sleep(0.1)
    for fields in entries:
        sm._redis.xadd(f"iris:overcrowd_alert:{sid}", fields)
        await asyncio.sleep(0.05)
    await asyncio.sleep(settle)
    sm._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_started_entry_records_an_ongoing_incident_with_derived_onset(tmp_path):
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        await _drain_sampler(sm, "cam0", [
            {"frame_num": "5", "pts": "12.500", "kind": "fire_smoke",
             "class": "fire", "count": "1", "onset_age": "3.000"},
        ])
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["incident_status"] == "ongoing"
    assert rows[0]["incident_started_at"] is not None
    assert rows[0]["incident_duration_seconds"] is None
    assert ("cam0", "fire") in sm.fire.open_incidents
    assert rows[0]["details"] == {"class": "fire"}          # fire or smoke, as data: a console words it in its language


def test_resolved_entry_completes_the_matching_started_incident(tmp_path):
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        await _drain_sampler(sm, "cam0", [
            {"frame_num": "5", "pts": "12.500", "kind": "fire_smoke",
             "class": "fire", "count": "1", "onset_age": "3.000"},
            {"pts": "42.000", "kind": "fire_smoke_resolved", "class": "fire", "duration": "31.250"},
        ])
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1                                  # resolve updates, doesn't add a row
    assert rows[0]["incident_status"] == "resolved"
    assert rows[0]["incident_duration_seconds"] == 31.25
    assert sm.fire.open_incidents == {}                   # consumed
    update_msgs = [e for e in conn.events if e.get("type") == "event_update"]
    assert any(m.get("incident_status") == "resolved" for m in update_msgs)


def test_resolved_entry_with_no_open_incident_is_dropped_not_crashed(tmp_path):
    # A restart mid-incident, or (shouldn't happen) version skew -- either way
    # this must not raise and must not touch the store.
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        await _drain_sampler(sm, "cam0", [
            {"pts": "42.000", "kind": "fire_smoke_resolved", "class": "fire", "duration": "5.0"},
        ])
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows == []
    assert conn.events == []


def test_classes_and_streams_track_independent_open_incidents(tmp_path):
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task_a = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        task_b = asyncio.create_task(sm.count_alerts.run_for("cam1"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"pts": "1.0", "kind": "fire_smoke",
                                             "class": "fire", "count": "1", "onset_age": "3.0"})
        r.xadd("iris:overcrowd_alert:cam1", {"pts": "1.0", "kind": "fire_smoke",
                                             "class": "fire", "count": "1", "onset_age": "3.0"})
        await asyncio.sleep(0.3)
        sm._running = False
        for t in (task_a, task_b):
            t.cancel()
        for t in (task_a, task_b):
            try:
                await t
            except asyncio.CancelledError:
                pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 2
    assert {("cam0", "fire"), ("cam1", "fire")} == set(sm.fire.open_incidents.keys())


def test_a_sidecar_restart_runs_the_registered_callbacks_first():
    import asyncio
    from app.usecases.fire_incidents import FireIncidents

    class Store:
        async def resolve_stale_incidents_for_camera(self, sid):
            return []

    class Conn:
        async def broadcast(self, payload):
            pass

    seen = []

    async def callback(sid):
        seen.append(sid)

    fire = FireIncidents(Store(), Conn(), None, None)
    fire.on_restart.append(callback)
    asyncio.run(fire.reconcile_sidecar_restart("cam-1"))
    assert seen == ["cam-1"]
