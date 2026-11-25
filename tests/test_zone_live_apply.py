"""Endpoint-level tests for the zone hot-reload handshake: saving zones bumps
iris:ctrl:zones_version, and GET /api/cameras exposes the version/applied pair the
frontend uses to show save -> applying -> active status."""

import fakeredis
import yaml
from fastapi.testclient import TestClient

from app.server import create_app

_TRI = [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}]
_ZONES = {"zones": [{"points": _TRI}]}


def _cfg_deepstream(tmp_path, analytics_enabled=True):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(yaml.safe_dump({
        "mux_width": 1000,
        "mux_height": 500,
        "modules": ({"zones": {"analytics_config_path": str(tmp_path / "nvdsanalytics_config.txt")}}
                    if analytics_enabled else {}),
        "sources": [{"id": "cam0", "uri": "file:///a.mp4"}],
    }))
    return {
        "perception": {"backend": "deepstream", "redis_url": "redis://unused"},
        "segments": {"enabled": False},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {}, "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
        "zones": {
            "file_path": str(tmp_path / "zones_config.txt"),
            "sidecar_config_path": str(sidecar_yaml),
        },
    }


def _shared_redis(monkeypatch):
    """Patch redis so the app and the test share one fakeredis keyspace."""
    server = fakeredis.FakeServer()
    monkeypatch.setattr("redis.Redis.from_url", lambda url: fakeredis.FakeRedis(server=server))
    return fakeredis.FakeRedis(server=server)


def test_post_zones_bumps_version_and_returns_it(tmp_path, monkeypatch):
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg_deepstream(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/api/streams/cam0/zones", json=_ZONES)
        assert resp.status_code == 200
        assert resp.json()["zones_version"] == 1
        assert r.get("iris:ctrl:zones_version") == b"1"


def test_delete_zones_bumps_version_again(tmp_path, monkeypatch):
    _shared_redis(monkeypatch)
    app = create_app(_cfg_deepstream(tmp_path))
    with TestClient(app) as client:
        client.post("/api/streams/cam0/zones", json=_ZONES)
        resp = client.post("/api/streams/cam0/zones", json={"zones": []})
        assert resp.status_code == 200
        assert resp.json() == {"zones": [], "zones_version": 2}


def test_cameras_endpoint_reports_zones_status(tmp_path, monkeypatch):
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg_deepstream(tmp_path))
    with TestClient(app) as client:
        client.post("/api/streams/cam0/zones", json=_ZONES)
        zones = client.get("/api/cameras").json()["zones"]
        assert zones == {"enabled": True, "version": 1, "applied_version": 0,
                         "applied_by_camera": {}}

        r.set("iris:analytics:zones_applied", 1)   # sidecar acks the reload
        zones = client.get("/api/cameras").json()["zones"]
        assert zones == {"enabled": True, "version": 1, "applied_version": 1,
                         "applied_by_camera": {}}       # per camera: tests/test_stream_manager.py


def test_cameras_endpoint_reports_analytics_disabled(tmp_path, monkeypatch):
    _shared_redis(monkeypatch)
    app = create_app(_cfg_deepstream(tmp_path, analytics_enabled=False))
    with TestClient(app) as client:
        zones = client.get("/api/cameras").json()["zones"]
        assert zones["enabled"] is False


def test_legacy_mode_response_has_no_zones_version(tmp_path):
    from tests.test_zone_endpoint import _cfg, _zone
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/api/streams/cam0/zones", json=_ZONES)
        assert resp.status_code == 200
        assert resp.json() == {"zones": [_zone(_TRI)]}
