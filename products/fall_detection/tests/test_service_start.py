"""The service's start path: the gate, then the real wiring, and a health reply that is the gate's own.

ACs: AC-F23.1-3, AC-F25.1, AC-N10.1. Design: FALL_DETECTION_DESIGN.md §8.
The bug this file pins: `build_service` passed `redis_client=` to a factory whose parameter is
`redis`, so every start raised TypeError *after* the GPU gate had passed. A start-path test is
the only thing that can catch it -- the parts all passed their own tests.
"""
import dataclasses
import sqlite3

import pytest
import redis as redis_module
from fastapi.testclient import TestClient

import boot
import config
import consumer
import main
import monitor
import server
import storage
from fall_helpers import POLICY


class _FakeClient:
    """The Redis handle the service needs: the ping, the stream reads, and close()."""

    def __init__(self):
        self.closed = False
        self.pings = 0

    def ping(self):
        self.pings += 1

    def xrevrange(self, *a, **k):
        return []

    def xread(self, *a, **k):
        return []

    def close(self):
        self.closed = True


class _Unreachable(_FakeClient):
    def ping(self):
        raise redis_module.ConnectionError("connection refused")


def _cfg(tmp_path) -> config.ServiceConfig:
    return config.ServiceConfig(
        redis_url="redis://localhost:6379/0",
        host="127.0.0.1",
        port=8004,
        data_dir=tmp_path,
        cameras=(config.CameraConfig(id="cam-1"),),
        scoring=POLICY["scoring"],
        classifier={"device": "cuda:0", "weights_dir": str(tmp_path / "weights")},
        suppress_after_fall_s=POLICY["suppress_after_fall_s"],
    )


def _quiet_start(monkeypatch):
    """No threads: the test is about the wiring, not the loops (test_worker covers those).

    The stubs record what was started, so a `build_service` that forgot to start the readers
    still fails the test instead of merely running quieter.
    """
    started = {"workers": [], "reader": 0}
    monkeypatch.setattr(consumer.CameraWorker, "start", lambda self: started["workers"].append(self.camera))
    monkeypatch.setattr(consumer.Consumer, "start", lambda self: started.__setitem__("reader", started["reader"] + 1))
    return started


def _gate(monkeypatch, ready=True, calls=None):
    """boot.boot stubbed with its real signature; `calls` (a list) records each call's (classifier, model_manifest)."""
    state = {"device": "cuda:0", "weights_verified": True, "classifier_ready": ready}

    def fake_boot(classifier, model_manifest=None):
        if calls is not None:
            calls.append((classifier, model_manifest))
        return "CLASSIFIER", state

    monkeypatch.setattr(boot, "boot", fake_boot)
    return state


def test_the_service_starts_and_reports_the_gates_own_state(tmp_path, monkeypatch):
    started = _quiet_start(monkeypatch)
    gate = _gate(monkeypatch, ready=False)          # a degraded gate must show up as degraded
    client = _FakeClient()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: client)
    built = []
    monkeypatch.setattr(monitor, "build_monitor",
                        lambda classifier, **kw: built.append((classifier, kw)) or object())

    service = main.build_service(_cfg(tmp_path))
    try:
        assert client.pings == 1                                  # Redis is checked before work starts
        assert started["workers"] == ["cam-1"]                    # the camera's worker really starts
        assert started["reader"] == 1                             # and so does the reader
        assert len(built) == 1 and built[0][0] == "CLASSIFIER"    # one monitor per camera
        body = TestClient(service.app).get("/status").json()
        assert body == boot.status(gate)                          # the gate's facts, not a hardcoded health
        assert body["classifier_ready"] is False
        assert set(body["paths_live"]) == {"kinematic", "cold"}
    finally:
        service.stop()


def test_the_gate_is_handed_the_configs_model_manifest(tmp_path, monkeypatch):
    """The yaml's model_manifest reaches boot's manifest check: a start path that dropped it would only warn
    ("no manifest") and load whatever checkpoints are on disk."""
    _quiet_start(monkeypatch)
    calls = []
    _gate(monkeypatch, calls=calls)
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: _FakeClient())
    cfg = dataclasses.replace(_cfg(tmp_path), model_manifest="products/fall_detection/models.manifest.json")

    service = main.build_service(cfg)
    try:
        assert calls == [(cfg.classifier, "products/fall_detection/models.manifest.json")]
    finally:
        service.stop()


def test_the_redis_handle_leaves_the_frame_bytes_undecoded(tmp_path, monkeypatch):
    """The meta entry carries the frame's `jpeg` as raw bytes (perception/engine/publisher.py:350).

    A client that decodes every reply as UTF-8 raises UnicodeDecodeError on the first real frame,
    so the reader thread dies while `/status` still says the service is healthy. The ACL turns the
    fields the product reads into text (acl.text), so the connection must not decode for it.
    """
    _quiet_start(monkeypatch)
    _gate(monkeypatch)
    captured = {}
    client = _FakeClient()
    monkeypatch.setattr(redis_module.Redis, "from_url",
                        lambda url, **kw: (captured.update(kw, url=url), client)[1])
    monkeypatch.setattr(monitor, "build_monitor", lambda classifier, **kw: object())

    service = main.build_service(_cfg(tmp_path))
    try:
        assert captured["decode_responses"] is False
    finally:
        service.stop()


def test_a_failed_start_closes_what_it_opened(tmp_path, monkeypatch):
    """A start that dies after the gate must not leak the Redis handle or the SQLite connection."""
    _quiet_start(monkeypatch)
    _gate(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: client)
    monkeypatch.setattr(monitor, "build_monitor", lambda classifier, **kw: object())
    closed = []

    class _RecordingQueue(storage.ReviewQueue):
        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(storage, "ReviewQueue", _RecordingQueue)

    def _no_app(**kw):
        raise RuntimeError("no app")

    monkeypatch.setattr(server, "create_app", _no_app)

    with pytest.raises(RuntimeError):
        main.build_service(_cfg(tmp_path))
    assert client.closed
    assert closed == [True]


def test_stop_keeps_the_handles_open_while_a_thread_still_runs(tmp_path, monkeypatch):
    """Closing the queue under a straggler turns a slow shutdown into an exception in that thread."""
    _quiet_start(monkeypatch)
    _gate(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: client)
    monkeypatch.setattr(monitor, "build_monitor", lambda classifier, **kw: object())

    class _Stuck:
        def stop(self): pass
        def join(self, timeout=None): pass
        @property
        def alive(self): return True

    service = main.build_service(_cfg(tmp_path))
    service.reader = _Stuck()
    try:
        service.stop(timeout=0.01)
        assert not client.closed
        assert service.queue.list_reviews() == []             # the connection is still usable
    finally:
        service.queue.close()
        client.close()


def test_the_redis_handle_carries_the_timeouts_that_keep_a_half_open_socket_from_hanging(tmp_path, monkeypatch):
    _quiet_start(monkeypatch)
    _gate(monkeypatch, ready=True)
    seen = {}
    monkeypatch.setattr(redis_module.Redis, "from_url",
                        lambda url, **kw: seen.update(kw) or _FakeClient())
    monkeypatch.setattr(monitor, "build_monitor", lambda classifier, **kw: object())

    service = main.build_service(_cfg(tmp_path))
    try:
        assert seen["socket_timeout"] == 5.0
        assert seen["socket_connect_timeout"] == 5.0
        assert seen["health_check_interval"] == 30
    finally:
        service.stop()


def test_an_unreachable_redis_refuses_to_start_naming_it(tmp_path, monkeypatch):
    _quiet_start(monkeypatch)
    _gate(monkeypatch, ready=True)
    client = _Unreachable()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: client)
    with pytest.raises(SystemExit) as e:
        main.build_service(_cfg(tmp_path))
    assert "Redis" in str(e.value.code)                       # the message names what is wrong
    assert client.closed                                      # the handle it could not use is closed


def test_stop_closes_the_review_queue_and_the_redis_handle(tmp_path, monkeypatch):
    _quiet_start(monkeypatch)
    _gate(monkeypatch, ready=True)
    client = _FakeClient()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *a, **k: client)
    monkeypatch.setattr(monitor, "build_monitor", lambda classifier, **kw: object())

    service = main.build_service(_cfg(tmp_path))
    service.stop()
    assert client.closed
    with pytest.raises(sqlite3.ProgrammingError):
        service.queue.list_reviews()                              # the connection is closed
