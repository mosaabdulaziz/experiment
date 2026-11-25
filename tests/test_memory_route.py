"""GET /api/streams/{id}/memory — the read-only view of a stream's temporal memory.

Three properties carry this route and each has a regression test below.

It reports *durations*, never the `time.monotonic()` stamps the two modules keep
internally: those are arbitrary process-local numbers no client can turn into a
wall clock, so serialising one would guarantee a nonsense rendering downstream.

`stationary_for_s` stays `None` for a track that never carried a bbox — the same
distinction `TrackObserver.render` makes by dropping the state column. `0` would
assert the object is stationary having measured nothing at all.

And it answers with valid JSON in every "no memory" case rather than 404ing or
raising: `app/server.py` serves `static/index.html` with a **200** for an unknown
path, and the frontend client reads a non-JSON 2xx as an error — so a missing or
throwing route surfaces as a confusing parse failure, not as "memory is off".
"""
import time

import fakeredis
import yaml
from fastapi.testclient import TestClient

from app.scene_memory import SceneMemory
from app.server import create_app
from app.track_observer import TrackObserver

W, H = 1920, 1080
_EMPTY = {"situations": [], "areas": [], "entities": [], "scene_notes": None, "tracked": []}


def _det(oid, label="person", cx=500.0, cy=500.0, w=40.0, h=80.0):
    """A parse_dets-shaped (cls, conf, track_id, bbox) tuple centred on (cx, cy)."""
    return (label, 0.9, oid, (cx - w / 2, cy - h / 2, w, h))


def _sit(key, description="something happening", state="ongoing"):
    return {"key": key, "description": description, "state": state}


# --- SceneMemory.snapshot -------------------------------------------------

def test_scene_memory_snapshot_reports_durations_longest_running_first():
    m = SceneMemory()
    m.merge({"situations": [_sit("spill", "a spill by the dock")]}, 100.0)
    m.merge({"situations": [_sit("spill", "a spill by the dock"),
                            _sit("queue", "a queue at the gate")]}, 105.0)
    snap = m.snapshot(110.0)
    assert snap["situations"] == [
        {"key": "spill", "description": "a spill by the dock", "duration_s": 10.0},
        {"key": "queue", "description": "a queue at the gate", "duration_s": 5.0},
    ]


def test_scene_memory_snapshot_reports_areas_entities_and_notes():
    m = SceneMemory()
    m.merge({"areas": [{"zone": "Dock", "condition": "blocked"}],
             "entities": [{"id": 17, "note": "driver in hi-vis"}],
             "scene_notes": "lighting is poor"}, 100.0)
    snap = m.snapshot(163.0)
    assert snap["areas"] == [{"zone": "Dock", "condition": "blocked", "duration_s": 63.0}]
    assert snap["entities"] == [{"id": 17, "note": "driver in hi-vis"}]
    assert snap["scene_notes"] == "lighting is poor"


def test_scene_memory_snapshot_is_empty_and_notes_are_none_before_any_merge():
    assert SceneMemory().snapshot(100.0) == {"situations": [], "areas": [],
                                             "entities": [], "scene_notes": None}


def test_scene_memory_snapshot_never_leaks_a_monotonic_stamp():
    m = SceneMemory()
    m.merge({"situations": [_sit("spill")]}, 100.0)
    assert set(m.snapshot(110.0)["situations"][0]) == {"key", "description", "duration_s"}


def test_scene_memory_snapshot_does_not_disturb_the_rendered_block():
    m = SceneMemory()
    m.merge({"situations": [_sit("spill")], "scene_notes": "quiet"}, 100.0)
    before = m.render(110.0)
    m.snapshot(110.0)
    assert m.render(110.0) == before


# --- TrackObserver.snapshot -----------------------------------------------

def test_track_observer_snapshot_reports_durations_longest_present_first():
    obs = TrackObserver(frame_width=W, frame_height=H)
    obs.observe(100.0, [_det(23, "truck")])
    obs.observe(105.0, [_det(23, "truck"), _det(9, "person")])
    snap = obs.snapshot(110.0)
    assert [t["id"] for t in snap["tracked"]] == [23, 9]
    assert snap["tracked"][0] == {"id": 23, "label": "truck",
                                 "present_for_s": 10.0, "stationary_for_s": 10.0}
    assert snap["tracked"][1]["present_for_s"] == 5.0


def test_stationary_clock_is_independent_of_the_presence_clock():
    obs = TrackObserver(frame_width=W, frame_height=H)
    obs.observe(100.0, [_det(23, "truck", cx=200.0, cy=200.0)])
    obs.observe(105.0, [_det(23, "truck", cx=1500.0, cy=900.0)])   # moved: anchor resets
    snap = obs.snapshot(110.0)
    assert snap["tracked"][0]["present_for_s"] == 10.0
    assert snap["tracked"][0]["stationary_for_s"] == 5.0


def test_stationary_for_s_is_none_not_zero_when_no_position_was_measured():
    """A sidecar too old to publish bboxes lands every track here. `0` would report
    the whole frame as stationary having measured none of it."""
    obs = TrackObserver(frame_width=W, frame_height=H)
    obs.observe(100.0, [("person", 0.9, 4, None)])
    snap = obs.snapshot(110.0)
    assert snap["tracked"] == [{"id": 4, "label": "person",
                                "present_for_s": 10.0, "stationary_for_s": None}]


def test_track_observer_snapshot_is_empty_before_any_detection():
    assert TrackObserver(frame_width=W, frame_height=H).snapshot(100.0) == {"tracked": []}


def test_track_observer_snapshot_does_not_disturb_the_rendered_block():
    obs = TrackObserver(frame_width=W, frame_height=H)
    obs.observe(100.0, [_det(23, "truck")])
    before = obs.render(110.0)
    obs.snapshot(110.0)
    assert obs.render(110.0) == before


# --- the route ------------------------------------------------------------

def _sidecar(tmp_path):
    path = tmp_path / "sidecar.yaml"
    path.write_text(yaml.safe_dump({
        "mux_width": W, "mux_height": H,
        "analytics_config_path": str(tmp_path / "nvdsanalytics_config.txt"),
        "sources": [{"id": "cam0", "uri": "file:///a.mp4"}],
    }))
    return path


def _cfg(tmp_path, *, backend="deepstream", memory=None):
    monitoring = {"mode": "proactive"}
    if memory is not None:
        monitoring["memory"] = memory
    return {
        "perception": {"backend": backend, "redis_url": "redis://unused"},
        "segments": {"enabled": False},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": monitoring,
        "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
        "zones": {"file_path": str(tmp_path / "zones_config.txt"),
                  "sidecar_config_path": str(_sidecar(tmp_path))},
    }


def _deepstream_client(tmp_path, monkeypatch, memory=None):
    """An app with a StreamManager on fakeredis, and NO lifespan.

    Deliberately not `with TestClient(...)`: startup launches the discovery loop,
    which removes every stream fakeredis never announced — including the one these
    tests register by hand. The route is what is under test here, not discovery.
    """
    monkeypatch.setattr("redis.Redis.from_url",
                        lambda url: fakeredis.FakeRedis(server=fakeredis.FakeServer()))
    app = create_app(_cfg(tmp_path, memory=memory))
    return TestClient(app), app.state.stream_manager


def test_route_reports_disabled_on_the_local_backend(tmp_path):
    app = create_app(_cfg(tmp_path, backend="local"))
    with TestClient(app) as client:
        r = client.get("/api/streams/cam0/memory")
        assert r.status_code == 200
        assert r.json() == dict(_EMPTY, enabled=False, stream="cam0")


def test_route_reports_disabled_when_memory_is_off(tmp_path, monkeypatch):
    client, mgr = _deepstream_client(tmp_path, monkeypatch)
    mgr._ensure_stream_state("cam0")           # a live stream, but no observer was built
    r = client.get("/api/streams/cam0/memory")
    assert r.status_code == 200
    assert r.json() == dict(_EMPTY, enabled=False, stream="cam0")


def test_route_reports_disabled_for_an_unknown_stream(tmp_path, monkeypatch):
    client, mgr = _deepstream_client(tmp_path, monkeypatch,
                                     memory={"enabled": True, "scene_delta": True})
    mgr._ensure_stream_state("cam0")
    r = client.get("/api/streams/cam9/memory")
    assert r.status_code == 200
    assert r.json() == dict(_EMPTY, enabled=False, stream="cam9")


def test_route_returns_both_halves_when_memory_is_on(tmp_path, monkeypatch):
    client, mgr = _deepstream_client(tmp_path, monkeypatch,
                                     memory={"enabled": True, "scene_delta": True})
    mgr._ensure_stream_state("cam0")
    then = time.monotonic() - 60.0
    mgr._track_observers["cam0"].observe(then, [_det(23, "truck")])
    mgr._scene_memories["cam0"].merge(
        {"situations": [_sit("unloading", "a truck being unloaded")],
         "areas": [{"zone": "Dock", "condition": "blocked"}],
         "entities": [{"id": 23, "note": "reversing"}],
         "scene_notes": "yard is busy"}, then, live_track_ids={23})

    body = client.get("/api/streams/cam0/memory").json()
    assert body["enabled"] is True and body["stream"] == "cam0"
    assert body["situations"][0]["key"] == "unloading"
    assert body["areas"][0]["zone"] == "Dock"
    assert body["entities"] == [{"id": 23, "note": "reversing"}]
    assert body["scene_notes"] == "yard is busy"
    assert body["tracked"][0]["id"] == 23 and body["tracked"][0]["label"] == "truck"
    # Durations are measured server-side, against the same monotonic clock.
    for seen in (body["situations"][0]["duration_s"], body["tracked"][0]["present_for_s"]):
        assert 60.0 <= seen < 75.0


def test_route_returns_tracked_without_situations_when_scene_delta_is_off(tmp_path, monkeypatch):
    """`enabled: true` with `scene_delta: false` is a supported configuration — the
    measured half is free at inference time, the interpreted half is not."""
    client, mgr = _deepstream_client(tmp_path, monkeypatch, memory={"enabled": True})
    mgr._ensure_stream_state("cam0")
    mgr._track_observers["cam0"].observe(time.monotonic() - 60.0, [_det(23, "truck")])

    body = client.get("/api/streams/cam0/memory").json()
    assert body["enabled"] is True
    assert body["tracked"][0]["id"] == 23
    assert body["situations"] == [] and body["areas"] == []
    assert body["entities"] == [] and body["scene_notes"] is None
