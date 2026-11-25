import json

import yaml
from fastapi.testclient import TestClient

from app.server import create_app

_DEFAULT_INTRUSION = {"mode": "restricted", "classes": ["person"]}
_DEFAULT_OVERCROWDING = {"enabled": False, "threshold": 5, "classes": ["person"]}
_TRI = [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}]
_TRI2 = [{"x": 0.2, "y": 0.2}, {"x": 0.6, "y": 0.2}, {"x": 0.6, "y": 0.6}]


def _cfg(tmp_path):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(yaml.safe_dump({
        "mux_width": 1000,
        "mux_height": 500,
        "modules": {"zones": {"analytics_config_path": str(tmp_path / "nvdsanalytics_config.txt")}},
        "relevant_classes": ["person", "car", "truck"],
        "sources": [{"id": "cam0", "uri": "file:///a.mp4"}, {"id": "cam1", "uri": "file:///b.mp4"}],
    }))
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive"}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
        "zones": {
            "file_path": str(tmp_path / "zones_config.txt"),
            "sidecar_config_path": str(sidecar_yaml),
        },
    }


def _zone(points, name="Zone 1", zid="z0", intrusion=None, overcrowding=None):
    return {
        "id": zid, "name": name, "points": points,
        "intrusion": intrusion or _DEFAULT_INTRUSION,
        "overcrowding": overcrowding or _DEFAULT_OVERCROWDING,
    }


def test_get_zones_empty_when_unsaved(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/streams/cam0/zones")
        assert r.status_code == 200
        assert r.json() == {"zones": []}


def test_detection_classes_endpoint_lists_relevant_classes(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/detection-classes")
        assert r.status_code == 200
        assert r.json() == {"classes": ["person", "car", "truck"]}


def test_post_then_get_round_trips_defaults_with_assigned_id_and_name(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [{"points": _TRI}]})
        assert r.status_code == 200
        assert r.json() == {"zones": [_zone(_TRI)]}

        r = client.get("/api/streams/cam0/zones")
        assert r.status_code == 200
        assert r.json() == {"zones": [_zone(_TRI)]}


def test_post_preserves_custom_names_and_assigns_positional_ids(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"name": "Dock Door", "points": _TRI,
             "intrusion": {"mode": "restricted", "classes": ["person"]}},
            {"name": "Walkway", "points": _TRI2,
             "intrusion": {"mode": "valid", "classes": ["person", "car"]}},
        ]})
        assert r.status_code == 200
        zones = r.json()["zones"]
        assert [z["id"] for z in zones] == ["z0", "z1"]
        assert [z["name"] for z in zones] == ["Dock Door", "Walkway"]
        assert zones[1]["intrusion"] == {"mode": "valid", "classes": ["person", "car"]}


def test_post_full_config_round_trips_and_stores_pixel_points(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as client:
        payload = {"zones": [{
            "name": "Lobby", "points": _TRI,
            "intrusion": {"mode": "valid", "classes": ["person", "car"]},
            "overcrowding": {"enabled": True, "threshold": 8, "classes": ["person"]},
        }]}
        r = client.post("/api/streams/cam0/zones", json=payload)
        assert r.status_code == 200
        assert r.json()["zones"][0]["overcrowding"] == {
            "enabled": True, "threshold": 8, "classes": ["person"]}

    with open(cfg["zones"]["file_path"]) as f:
        stored = json.load(f)
    zone = stored["cam0"]["zones"][0]
    assert zone["id"] == "z0"
    assert zone["name"] == "Lobby"
    # mux_width=1000, mux_height=500 from the temp sidecar.yaml
    assert zone["points"] == [[100, 100], [900, 100], [900, 400]]
    assert zone["intrusion"]["mode"] == "valid"


def test_empty_name_falls_back_to_positional_default(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [{"name": "  ", "points": _TRI}]})
        assert r.json()["zones"][0]["name"] == "Zone 1"


def test_post_rejects_fewer_than_three_points(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones",
                        json={"zones": [{"points": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}]}]})
        assert r.status_code == 400
        assert "error" in r.json()


def test_post_rejects_out_of_range_coordinates(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": [{"x": -0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}]}]})
        assert r.status_code == 400
        assert "error" in r.json()


def test_post_rejects_no_feature_enabled(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [{
            "points": _TRI, "intrusion": {"mode": "off"}, "overcrowding": {"enabled": False}}]})
        assert r.status_code == 400
        assert "error" in r.json()


def test_post_rejects_unknown_class(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [{
            "points": _TRI, "intrusion": {"mode": "restricted", "classes": ["giraffe"]}}]})
        assert r.status_code == 400
        assert "error" in r.json()


def test_post_rejects_bad_zone_without_persisting_any(tmp_path):
    # One invalid zone in the list fails the whole save (bulk replace is atomic).
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI},
            {"points": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}]},  # too few points
        ]})
        assert r.status_code == 400
        assert client.get("/api/streams/cam0/zones").json() == {"zones": []}


def test_empty_list_deletes_all_zones(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/api/streams/cam0/zones", json={"zones": [{"points": _TRI}]})
        r = client.post("/api/streams/cam0/zones", json={"zones": []})
        assert r.status_code == 200
        assert r.json()["zones"] == []
        assert client.get("/api/streams/cam0/zones").json() == {"zones": []}


def test_saving_one_stream_does_not_affect_another(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/api/streams/cam0/zones", json={"zones": [{"name": "A", "points": _TRI}]})
        client.post("/api/streams/cam1/zones", json={"zones": [{"name": "B", "points": _TRI2}]})

        assert client.get("/api/streams/cam0/zones").json()["zones"][0]["name"] == "A"
        assert client.get("/api/streams/cam1/zones").json()["zones"][0]["name"] == "B"


def test_ppe_fall_and_fire_blocks_save_and_round_trip(tmp_path):
    """The safety product's per-feature zones: a zone with only a `ppe` block on is valid (no intrusion, no
    overcrowding), and its blocks come back as saved, in the response and on read."""
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"name": "Bay", "points": _TRI, "intrusion": {"mode": "off"}, "ppe": {"enabled": True}},
            {"name": "Store", "points": _TRI2, "intrusion": {"mode": "off"},
             "fire": {"enabled": True}, "fall": {"enabled": False}},
        ]})
        assert r.status_code == 200, r.text
        bay, store = r.json()["zones"]
        assert bay["ppe"] == {"enabled": True} and "fire" not in bay
        assert store["fire"] == {"enabled": True} and store["fall"] == {"enabled": False}
        bay, store = client.get("/api/streams/cam0/zones").json()["zones"]
        assert bay["ppe"] == {"enabled": True} and store["fire"] == {"enabled": True}
    saved = json.loads((tmp_path / "zones_config.txt").read_text())["cam0"]["zones"]
    assert saved[0]["ppe"] == {"enabled": True} and saved[1]["fall"] == {"enabled": False}


def test_a_zone_with_every_feature_off_is_still_refused(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": {"mode": "off"}, "ppe": {"enabled": False}}]})
        assert r.status_code == 400 and "enable at least one" in r.json()["error"]


def test_zones_saved_through_a_role_stream_belong_to_its_camera(tmp_path):
    """Zones are per camera: every feature's sidecar on it reads the same ones."""
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        assert client.post("/api/streams/cam0@zones/zones", json={"zones": [{"points": _TRI}]}).status_code == 200
        assert len(client.get("/api/streams/cam0@ppe/zones").json()["zones"]) == 1
    assert list(json.loads((tmp_path / "zones_config.txt").read_text())) == ["cam0"]


def test_filter_zones_saved_through_a_role_stream_belong_to_its_camera(tmp_path):
    """The engine looks filter zones up by camera id: saved under a role stream id they would never apply."""
    from app.zones import save_stream_filter_zones, stream_filter_zones
    cfg = _cfg(tmp_path)
    cfg["zones"]["filter_file_path"] = str(tmp_path / "filter_zones.json")
    save_stream_filter_zones(cfg, "cam0@zones", [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])
    assert list(json.loads((tmp_path / "filter_zones.json").read_text())) == ["cam0"]
    assert len(stream_filter_zones(cfg, "cam0@ppe")) == 1


def _cfg_with_ppe(tmp_path, ppe_settings=None):
    cfg = _cfg(tmp_path)
    sidecar = yaml.safe_load((tmp_path / "sidecar.yaml").read_text())
    sidecar["modules"]["ppe"] = ppe_settings or {}
    (tmp_path / "sidecar.yaml").write_text(yaml.safe_dump(sidecar))
    return cfg


def test_a_ppe_zone_saves_which_equipment_it_checks(tmp_path):
    """The PPE block may name the equipment checked in that zone (hard hat, vest or both); without `items` the zone
    checks everything the PPE module is set to."""
    app = create_app(_cfg_with_ppe(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"name": "Bay", "points": _TRI, "intrusion": {"mode": "off"}, "ppe": {"enabled": True, "items": ["helmet"]}},
            {"name": "Dock", "points": _TRI2, "intrusion": {"mode": "off"}, "ppe": {"enabled": True}},
        ]})
        assert r.status_code == 200, r.text
        bay, dock = r.json()["zones"]
        assert bay["ppe"] == {"enabled": True, "items": ["helmet"]} and dock["ppe"] == {"enabled": True}
        bay, dock = client.get("/api/streams/cam0@ppe/zones").json()["zones"]
        assert bay["ppe"] == {"enabled": True, "items": ["helmet"]} and dock["ppe"] == {"enabled": True}
    saved = json.loads((tmp_path / "zones_config.txt").read_text())["cam0"]["zones"]
    assert saved[0]["ppe"] == {"enabled": True, "items": ["helmet"]}


def test_ppe_items_must_be_equipment_the_ppe_module_checks(tmp_path):
    app = create_app(_cfg_with_ppe(tmp_path))
    with TestClient(app) as client:
        for items in (["mask"], []):
            r = client.post("/api/streams/cam0/zones", json={"zones": [
                {"points": _TRI, "intrusion": {"mode": "off"}, "ppe": {"enabled": True, "items": items}}]})
            assert r.status_code == 400 and "ppe" in r.json()["error"], items


def test_the_zones_payload_says_which_equipment_ppe_can_check(tmp_path):
    with TestClient(create_app(_cfg(tmp_path))) as client:
        assert "ppe" not in client.get("/api/streams/cam0/zones").json()        # no PPE module: nothing to choose
    with TestClient(create_app(_cfg_with_ppe(tmp_path))) as client:
        assert client.get("/api/streams/cam0/zones").json()["ppe"] == {"items": ["helmet", "vest"]}
    with TestClient(create_app(_cfg_with_ppe(tmp_path, {"required_items": ["vest"]}))) as client:
        assert client.get("/api/streams/cam0/zones").json()["ppe"] == {"items": ["vest"]}


def test_a_ppe_block_switched_off_keeps_whatever_items_it_had_without_a_refusal(tmp_path):
    """Untick everything, then switch PPE off: the editor saves it, and so must the app (the picker is hidden while
    PPE is off, so nothing could be fixed there)."""
    app = create_app(_cfg_with_ppe(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/streams/cam0/zones", json={"zones": [
            {"points": _TRI, "intrusion": {"mode": "restricted", "classes": ["person"]},
             "ppe": {"enabled": False, "items": []}},
            {"points": _TRI2, "intrusion": {"mode": "restricted", "classes": ["person"]},
             "ppe": {"enabled": False, "items": ["mask"]}}]})
        assert r.status_code == 200, r.text
