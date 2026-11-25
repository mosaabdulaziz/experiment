"""Proximity in the zone API: validated, stored sparse against the defaults, kept on every save, applied live."""
import json

import fakeredis
import pytest
import yaml
from fastapi.testclient import TestClient

from app.server import create_app

_TRI = [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}]
_OFF = {"mode": "off", "classes": ["person"]}
_DEFAULTS = {"vehicle_classes": ["car", "truck", "bus", "motorcycle"], "max_gap": 0.75, "stationary_window_s": 2.0,
             "stationary_motion": 0.75, "persist_s": 0.25, "grace_s": 0.5, "cooldown_s": 10.0, "depth_check": True,
             "hidden_feet": "unconfirmed", "depth_max_gap": 0.25}


def _cfg(tmp_path, modules, backend="local"):
    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text(yaml.safe_dump({
        "mux_width": 1000, "mux_height": 500, "modules": modules,
        "relevant_classes": ["person", "car", "truck", "bus", "motorcycle"],
        "sources": [{"id": "cam0", "uri": "file:///a.mp4"}]}))
    cfg = {"perception": {"backend": backend}, "api": {"backend": "cloud", "api_key": "x", "model": "m"},
           "monitoring": {}, "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
           "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
           "zones": {"file_path": str(tmp_path / "zones_config.txt"), "sidecar_config_path": str(sidecar)}}
    if backend == "deepstream":
        cfg["perception"]["redis_url"] = "redis://unused"
        cfg["segments"] = {"enabled": False}
    return cfg


def _stored(tmp_path):
    return json.loads((tmp_path / "zones_config.txt").read_text())["cam0"]["zones"]


def test_a_proximity_only_zone_is_saved_with_only_what_differs_from_the_defaults(tmp_path):
    app = create_app(_cfg(tmp_path, {"proximity": {}}))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {**_DEFAULTS, "enabled": True, "max_gap": 1.0}}]})
        assert r.status_code == 200, r.text
        assert r.json()["zones"][0]["proximity"] == {**_DEFAULTS, "enabled": True, "max_gap": 1.0}
        got = client.get("/api/streams/cam0/zones").json()
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": True, "max_gap": 1.0}
    assert got["zones"][0]["proximity"]["max_gap"] == 1.0
    assert got["proximity"] == {"available": True, "defaults": _DEFAULTS, "depth_model": False}


@pytest.mark.parametrize("bad, message", [
    ({"max_gap": 0}, "max_gap"), ({"grace_s": -1}, "grace_s"), ({"hidden_feet": "maybe"}, "hidden_feet"),
    ({"vehicle_classes": []}, "vehicle class"), ({"vehicle_classes": ["person"]}, "person"),
])
def test_bad_proximity_values_are_refused(tmp_path, bad, message):
    app = create_app(_cfg(tmp_path, {"proximity": {}}))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": True, **bad}}]})
    assert r.status_code == 400 and message in r.json()["error"]


def test_a_zone_with_nothing_on_is_still_refused(tmp_path):
    app = create_app(_cfg(tmp_path, {"proximity": {}}))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": False}}]})
    assert r.status_code == 400 and "enable at least one" in r.json()["error"]


def test_a_stored_block_survives_a_save_on_a_pipeline_without_the_module(tmp_path):
    (tmp_path / "zones_config.txt").write_text(json.dumps({"cam0": {"zones": [{
        "id": "z0", "name": "Dock", "points": [[100, 100], [900, 100], [900, 400]],
        "intrusion": {"mode": "restricted", "classes": ["person"]}, "proximity": {"enabled": True, "persist_s": 2.0}}]}}))
    app = create_app(_cfg(tmp_path, {}))
    with TestClient(app) as client:
        got = client.get("/api/streams/cam0/zones").json()
        assert "proximity" not in got                                   # the module is not running here
        zone = got["zones"][0]
        assert zone["proximity"]["persist_s"] == 2.0
        r = client.post("/api/streams/cam0/zones", json={"zones": [{k: v for k, v in zone.items() if k != "id"}]})
        assert r.status_code == 200, r.text
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": True, "persist_s": 2.0}


def test_a_zone_without_a_block_stays_without_one_when_the_module_is_off(tmp_path):
    app = create_app(_cfg(tmp_path, {}))
    with TestClient(app) as client:
        client.post("/api/streams/cam0/zones", json={"zones": [{"points": _TRI}]})
        got = client.get("/api/streams/cam0/zones").json()
    assert "proximity" not in got["zones"][0] and "proximity" not in _stored(tmp_path)[0]


def test_the_pipelines_own_settings_are_the_defaults_a_zone_is_compared_with(tmp_path):
    app = create_app(_cfg(tmp_path, {"proximity": {"max_gap": 1.0, "depth_model": "/m/depth.onnx"}}))
    with TestClient(app) as client:
        got = client.get("/api/streams/cam0/zones").json()
        client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {**got["proximity"]["defaults"], "enabled": True}}]})
    assert got["proximity"]["defaults"]["max_gap"] == 1.0 and got["proximity"]["depth_model"] is True
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": True}


def _shared_redis(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr("redis.Redis.from_url", lambda url: fakeredis.FakeRedis(server=server))
    return fakeredis.FakeRedis(server=server)


def test_a_save_bumps_the_proximity_version_and_the_status_reports_its_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)              # without a zones module the analytics file goes to ./data: keep it here
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {"proximity": {}}, backend="deepstream"))
    with TestClient(app) as client:
        resp = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": True}}]})
        assert resp.json()["proximity_version"] == 1
        assert client.get("/api/cameras").json()["zones"]["proximity"] == {"version": 1, "applied_version": 0}
        r.set("iris:analytics:proximity_applied", 1)
        assert client.get("/api/cameras").json()["zones"]["proximity"] == {"version": 1, "applied_version": 1}


def test_the_apps_keys_are_the_modules():
    from app import stream_manager as sm
    from perception.modules.proximity import APPLIED_KEY, PREVIEW_KEY, VERSION_KEY
    assert (sm._PROXIMITY_VERSION_KEY, sm._PROXIMITY_APPLIED_KEY) == (VERSION_KEY, APPLIED_KEY)
    assert sm._PROXIMITY_PREVIEW_KEY == PREVIEW_KEY
    from app import overlay_track
    assert overlay_track.PREVIEW_KEY == PREVIEW_KEY


def test_a_pipeline_without_vehicle_classes_still_saves_a_camera_with_a_stored_block(tmp_path):
    cfg = _cfg(tmp_path, {})
    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text(yaml.safe_dump({"mux_width": 1000, "mux_height": 500, "modules": {},
                                       "relevant_classes": ["person"], "sources": [{"id": "cam0", "uri": "x"}]}))
    (tmp_path / "zones_config.txt").write_text(json.dumps({"cam0": {"zones": [{
        "id": "z0", "name": "Dock", "points": [[100, 100], [900, 100], [900, 400]],
        "intrusion": {"mode": "restricted", "classes": ["person"]}, "proximity": {"enabled": False, "persist_s": 2.0}}]}}))
    app = create_app(cfg)
    with TestClient(app) as client:
        zone = client.get("/api/streams/cam0/zones").json()["zones"][0]
        r = client.post("/api/streams/cam0/zones", json={"zones": [{k: v for k, v in zone.items() if k != "id"}]})
        assert r.status_code == 200, r.text
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": False, "persist_s": 2.0}


def test_a_pipeline_running_the_module_refuses_a_vehicle_class_it_does_not_detect(tmp_path):
    cfg = _cfg(tmp_path, {"proximity": {}})
    (tmp_path / "sidecar.yaml").write_text(yaml.safe_dump({"mux_width": 1000, "mux_height": 500,
                                                          "modules": {"proximity": {}},
                                                          "relevant_classes": ["person", "car"],
                                                          "sources": [{"id": "cam0", "uri": "x"}]}))
    app = create_app(cfg)
    with TestClient(app) as client:
        refused = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": True, "vehicle_classes": ["truck"]}}]})
    assert refused.status_code == 400 and "truck" in refused.json()["error"]


def test_an_enabled_narrowed_block_survives_a_save_on_a_pipeline_without_the_module(tmp_path):
    """The Proximity section is hidden there, so the operator could never fix it: the pipeline's classes only matter
    when it runs the module."""
    cfg = _cfg(tmp_path, {})
    (tmp_path / "sidecar.yaml").write_text(yaml.safe_dump({"mux_width": 1000, "mux_height": 500, "modules": {},
                                                          "relevant_classes": ["person"],
                                                          "sources": [{"id": "cam0", "uri": "x"}]}))
    (tmp_path / "zones_config.txt").write_text(json.dumps({"cam0": {"zones": [{
        "id": "z0", "name": "Dock", "points": [[100, 100], [900, 100], [900, 400]],
        "intrusion": {"mode": "restricted", "classes": ["person"]},
        "proximity": {"enabled": True, "vehicle_classes": ["truck"]}}]}}))
    app = create_app(cfg)
    with TestClient(app) as client:
        zone = client.get("/api/streams/cam0/zones").json()["zones"][0]
        r = client.post("/api/streams/cam0/zones", json={"zones": [{k: v for k, v in zone.items() if k != "id"}]})
    assert r.status_code == 200, r.text
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": True, "vehicle_classes": ["truck"]}



def test_the_live_preview_shows_the_pipelines_layers_until_someone_changes_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {"proximity": {"preview_masks": True}}, backend="deepstream"))
    with TestClient(app) as client:
        assert client.get("/api/proximity/preview").json() == {"available": True, "masks": True, "lines": True,
                                                               "boxes": True, "line_range": 2.0, "clip_style": "masks", "depth": False}


def test_a_preview_change_is_stored_for_the_pipeline_and_keeps_the_other_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {"proximity": {}}, backend="deepstream"))
    with TestClient(app) as client:
        assert client.post("/api/proximity/preview", json={"masks": True}).json() == {
            "available": True, "masks": True, "lines": True, "boxes": True, "line_range": 2.0, "clip_style": "masks", "depth": False}
        assert json.loads(r.get("iris:ctrl:proximity_preview")) == {"masks": True, "lines": True, "boxes": True,
                                                                     "line_range": 2.0, "clip_style": "masks", "depth": False}
        client.post("/api/proximity/preview", json={"lines": False, "boxes": False, "line_range": 3.5,
                                                    "clip_style": "boxes"})
        assert client.get("/api/proximity/preview").json() == {"available": True, "masks": True, "lines": False,
                                                               "boxes": False, "line_range": 3.5, "clip_style": "boxes", "depth": False}


@pytest.mark.parametrize("body", [{"masks": "yes"}, {"lines": 1}, {}, {"boxes": "no"}, {"line_range": 0},
                                  {"line_range": 50}, {"line_range": "far"}, {"line_range": True},
                                  {"clip_style": "both"}, {"clip_style": 1}])
def test_a_preview_change_that_is_not_true_or_false_is_refused(tmp_path, monkeypatch, body):
    monkeypatch.chdir(tmp_path)
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {"proximity": {}}, backend="deepstream"))
    with TestClient(app) as client:
        assert client.post("/api/proximity/preview", json=body).status_code in (400, 422)
    assert r.get("iris:ctrl:proximity_preview") is None


def test_the_preview_switches_are_unavailable_without_the_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {}, backend="deepstream"))
    with TestClient(app) as client:
        assert client.get("/api/proximity/preview").json()["available"] is False
        assert client.post("/api/proximity/preview", json={"masks": True}).status_code == 400


def test_the_depth_map_layer_is_switched_like_the_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _shared_redis(monkeypatch)
    app = create_app(_cfg(tmp_path, {"proximity": {}}, backend="deepstream"))
    with TestClient(app) as client:
        assert client.post("/api/proximity/preview", json={"depth": True}).json()["depth"] is True
        assert json.loads(r.get("iris:ctrl:proximity_preview"))["depth"] is True
        assert client.post("/api/proximity/preview", json={"depth": "on"}).status_code in (400, 422)


def test_a_zone_keeps_its_own_depth_cut_off_and_a_bad_one_is_refused(tmp_path):
    # Ticket 6: the depth check's cut-off is a zone setting, tuned per zone in the zone editor
    app = create_app(_cfg(tmp_path, {"proximity": {}}))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": True, "depth_max_gap": 0.35}}]})
        assert r.status_code == 200, r.text
        assert r.json()["zones"][0]["proximity"]["depth_max_gap"] == 0.35
        for bad in (0, -0.1):
            r = client.post("/api/streams/cam0/zones", json={"zones": [
                {"points": _TRI, "intrusion": _OFF, "proximity": {"enabled": True, "depth_max_gap": bad}}]})
            assert r.status_code == 400, bad
    assert _stored(tmp_path)[0]["proximity"] == {"enabled": True, "depth_max_gap": 0.35}


def test_a_zone_written_with_only_a_proximity_block_reads_and_saves_as_no_intrusion(tmp_path):
    (tmp_path / "zones_config.txt").write_text(json.dumps({"cam0": {"zones": [{
        "id": "z0", "name": "Dock", "points": [[100, 100], [900, 100], [900, 400]], "proximity": {"enabled": True}}]}}))
    app = create_app(_cfg(tmp_path, {"proximity": {}}))
    with TestClient(app) as client:
        zone = client.get("/api/streams/cam0/zones").json()["zones"][0]
        assert zone["intrusion"]["mode"] == "off"
        r = client.post("/api/streams/cam0/zones", json={"zones": [{k: v for k, v in zone.items() if k != "id"}]})
        assert r.status_code == 200, r.text
    assert _stored(tmp_path)[0]["intrusion"]["mode"] == "off"
