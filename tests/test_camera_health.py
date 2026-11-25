"""products/safety/camera_health.py: camera-health incidents for the safety product (ticket 2): a camera sending nothing,
a feature's sidecar down on a camera, a camera too slow -- opened after a short confirmation, resolved when it
recovers, and held while the controller is applying a change to that feature."""
import asyncio
import json

import fakeredis

from app.events import EventStore
from products.safety.camera_health import CameraHealth


class _Conn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _setup(tmp_path, expected):
    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    clock, conn = _Clock(), _Conn()
    health = CameraHealth(store, conn, r, expected=lambda: expected, confirm_s=10, silent_s=20, min_fps=1.0,
                          now=clock)
    return r, store, clock, conn, health


def _alive(r, sid, now, fps=5.0, window=10):
    """The stream's status, and `fps` meta entries a second over the last `window` seconds."""
    r.set(f"iris:status:{sid}", json.dumps({"running": True}))
    top = r.xrevrange(f"iris:meta:{sid}", count=1)
    last = int(top[0][0].decode().split("-")[0]) if top else 0
    n = int(fps * window)
    for i in range(n):
        ms = int((now - window + i * window / max(n, 1)) * 1000)
        if ms > last:                                       # a stream only grows
            r.xadd(f"iris:meta:{sid}", {"pts": "1"}, id=f"{ms}-0")


async def _alerts(store):
    return [e for e in await store.get_recent(50) if "camera_health" in e["tags"]]


def test_healthy_cameras_raise_nothing(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        for _ in range(3):
            _alive(r, "bay-a@ppe", clock.t)
            _alive(r, "bay-a@zones", clock.t)
            await health.check()
            clock.t += 10
        return await _alerts(store)

    assert asyncio.run(run()) == []


def test_a_feature_down_on_a_camera_opens_after_confirmation_and_resolves_when_back(tmp_path):
    r, store, clock, conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        _alive(r, "bay-a@zones", clock.t)                     # zones fine, ppe never came up
        await health.check()
        early = await _alerts(store)
        clock.t += 11
        _alive(r, "bay-a@zones", clock.t)
        await health.check()
        opened = await _alerts(store)
        clock.t += 5
        _alive(r, "bay-a@zones", clock.t)
        _alive(r, "bay-a@ppe", clock.t)                       # ppe back
        await health.check()
        return early, opened, await _alerts(store)

    early, opened, after = asyncio.run(run())
    assert early == []
    (incident,) = opened
    assert incident["camera"] == "bay-a" and incident["feature"] == "health"
    assert incident["incident_status"] == "ongoing" and incident["details"]["problem"] == "feature_down"
    assert incident["details"]["role"] == "ppe" and incident["message"] == "PPE detection is not running on bay-a"
    assert after[0]["incident_status"] == "resolved"


def test_a_camera_sending_nothing_on_any_feature_is_one_offline_incident(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        _alive(r, "bay-a@ppe", clock.t - 60)                  # both last sent a minute ago
        _alive(r, "bay-a@zones", clock.t - 60)
        await health.check()
        clock.t += 11
        await health.check()
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["details"]["problem"] == "offline"


def test_a_slow_camera(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"yard": ["yard@ppe"]})

    async def run():
        await store.init()
        for _ in range(3):                      # a rate needs two readings; then the confirmation time
            _alive(r, "yard@ppe", clock.t, fps=0.3)
            await health.check()
            clock.t += 11
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["details"]["problem"] == "slow"


def test_a_camera_that_stops_is_one_no_video_incident_not_first_a_slow_one(tmp_path):
    """2026-11-10 live demo: the store feed stopped; its rate read 0 frames/s at the next pass, so a "slow (0.0
    frames/s)" incident opened 25 s before "No video" -- two incidents and two messages for one outage."""
    r, store, clock, _conn, health = _setup(tmp_path, {"store": ["store@fire"]})

    async def run():
        await store.init()
        _alive(r, "store@fire", clock.t)
        await health.check()
        clock.t += 5
        _alive(r, "store@fire", clock.t)
        await health.check()                                  # a rate: 5 frames/s
        for _ in range(12):                                   # then nothing more arrives, for a minute
            clock.t += 5
            await health.check()
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["details"]["problem"] == "offline"


def test_a_very_slow_camera_stays_slow_through_a_pass_with_no_new_frame(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"yard": ["yard@ppe"]})
    health.silent_s = 60

    async def run():
        await store.init()
        for i in range(8):                                    # 0.2 frames/s, read every 5 s: every other pass is empty
            if i % 2 == 0:
                _alive(r, "yard@ppe", clock.t, fps=0.2)
            await health.check()
            clock.t += 5
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["details"]["problem"] == "slow"


def test_a_feature_the_controller_is_applying_is_held(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        r.set("iris:safety:applying:ppe", "1", ex=300)
        for _ in range(3):
            _alive(r, "bay-a@zones", clock.t)                 # ppe restarting
            await health.check()
            clock.t += 11
        return await _alerts(store)

    assert asyncio.run(run()) == []


def test_a_problem_seen_before_a_hold_is_confirmed_afresh_once_it_can_be_judged_again(tmp_path):
    """At start-up the app sees every camera dark before the controller even starts the roles (and holds them). A
    camera still dark when the hold ends is not opened at once with a start from before the hold: it is confirmed
    again, and starts from when it could be judged."""
    from datetime import datetime, timezone
    r, store, clock, _conn, health = _setup(tmp_path, {"store": ["store@fire"]})

    async def run():
        await store.init()
        await health.check()                                  # dark before the role started
        start = clock.t
        r.set("iris:safety:applying:fire", "1", ex=300)
        for _ in range(4):                                    # held while the controller starts it
            clock.t += 15
            await health.check()
        r.delete("iris:safety:applying:fire")
        judged = clock.t + 5
        clock.t = judged
        await health.check()                                  # judged again: still dark
        assert await _alerts(store) == []                     # not at once
        clock.t += 11
        await health.check()
        (incident,) = await _alerts(store)
        return start, judged, incident

    start, judged, incident = asyncio.run(run())
    began = datetime.strptime(incident["incident_started_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert began.timestamp() >= judged - 1 > start


def test_the_expected_streams_are_the_role_configs_sources_by_camera():
    from products.safety.camera_health import expected_streams
    cfg = {"sources": [{"id": "bay-a@proximity"}, {"id": "bay-a@zones"}, {"id": "office@zones"}]}
    assert expected_streams(cfg) == {"bay-a": ["bay-a@proximity", "bay-a@zones"], "office": ["office@zones"]}



def test_a_hold_keeps_an_open_incident_open_instead_of_resolving_it(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        for _ in range(2):                                    # ppe down: an incident opens
            _alive(r, "bay-a@zones", clock.t)
            await health.check()
            clock.t += 11
        r.set("iris:safety:applying:ppe", "1", ex=300)        # an Apply restarts ppe while it is down
        _alive(r, "bay-a@zones", clock.t)
        await health.check()
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["incident_status"] == "ongoing"


def test_a_camera_with_a_held_feature_is_not_called_offline(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe", "bay-a@zones"]})

    async def run():
        await store.init()
        r.set("iris:safety:applying:ppe", "1", ex=300)
        for _ in range(2):                                    # zones down too, ppe held
            await health.check()
            clock.t += 11
        return await _alerts(store)

    (incident,) = asyncio.run(run())
    assert incident["details"] == {"problem": "feature_down", "role": "zones"}


def test_a_pass_reads_no_frame_metadata(tmp_path):
    """The meta stream's entries carry JPEGs: a pass must read only the stream's counters, never its entries."""
    r, store, clock, _conn, health = _setup(tmp_path, {"bay-a": ["bay-a@ppe"]})
    _alive(r, "bay-a@ppe", clock.t)
    r.xrange = r.xrevrange = lambda *a, **k: (_ for _ in ()).throw(AssertionError("read the entries"))
    asyncio.run(health.check())


def test_each_stream_s_frame_rate_is_kept_for_the_health_page(tmp_path):
    r, store, clock, _conn, health = _setup(tmp_path, {"yard": ["yard@ppe", "yard@fire"]})

    async def run():
        await store.init()
        for _ in range(2):                      # a rate needs two readings
            _alive(r, "yard@ppe", clock.t, fps=5.0)
            await health.check()
            clock.t += 10

    asyncio.run(run())
    assert 4.0 <= health.fps["yard@ppe"] <= 6.0
    assert health.fps.get("yard@fire") is None                # down: no rate


def test_its_messages_name_the_camera_and_the_feature_as_an_operator_knows_them(tmp_path):
    r = fakeredis.FakeStrictRedis()
    health = CameraHealth(None, _Conn(), r, expected=lambda: {}, names=lambda: {"store": "Store room"})
    clock = 1_000_000.0
    health._expected = lambda: {"store": ["store@fire"]}
    assert health.problems(clock)[0] == {("store", "offline", ""): "No video from Store room"}
    health._expected = lambda: {"store": ["store@fire", "store@zones"]}
    _alive(r, "store@zones", clock)
    assert health.problems(clock)[0] == {("store", "feature_down", "fire"): "Fire detection is not running on Store room"}
    r.set("iris:status:store@fire", json.dumps({"running": True}))
    r.xadd("iris:meta:store@fire", {"pts": "1"}, id=f"{int(clock * 1000) - 5}-0")
    health.problems(clock - 10)
    r.xadd("iris:meta:store@fire", {"pts": "1"}, id=f"{int(clock * 1000) - 1}-0")
    assert health.problems(clock)[0][("store", "slow", "fire")] == "Store room is slow on fire detection (0.1 frames/s)"
