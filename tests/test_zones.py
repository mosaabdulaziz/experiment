from app.zones import stream_zones, zone_by_id, _save_zones


def _config(tmp_path, mux_w=1000, mux_h=500):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(f"mux_width: {mux_w}\nmux_height: {mux_h}\n")
    return {"zones": {
        "file_path": str(tmp_path / "zones_config.txt"),
        "sidecar_config_path": str(sidecar_yaml),
    }}


def _write(config, zones):
    _save_zones(config["zones"]["file_path"], zones)


def test_stream_zones_normalizes_points_and_carries_config(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"zones": [{
        "id": "z0", "name": "Dock Door",
        "points": [[100, 50], [900, 50], [900, 450]],
        "intrusion": {"mode": "restricted", "classes": ["person"]},
        "overcrowding": {"enabled": True, "threshold": 8, "classes": ["person", "car"]},
    }]}})

    zones = stream_zones(config, "cam0")

    assert len(zones) == 1
    z = zones[0]
    assert z["id"] == "z0"
    assert z["name"] == "Dock Door"
    assert z["points"] == [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]
    assert z["intrusion"] == {"mode": "restricted", "classes": ["person"]}
    assert z["overcrowding"] == {"enabled": True, "threshold": 8, "classes": ["person", "car"]}


def test_stream_zones_returns_all_zones_in_order(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"zones": [
        {"id": "z0", "name": "A", "points": [[0, 0], [10, 0], [10, 10]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
        {"id": "z1", "name": "B", "points": [[0, 0], [20, 0], [20, 20]],
         "intrusion": {"mode": "valid", "classes": ["car"]}},
    ]}})

    zones = stream_zones(config, "cam0")

    assert [z["id"] for z in zones] == ["z0", "z1"]
    assert [z["name"] for z in zones] == ["A", "B"]
    assert zones[1]["intrusion"] == {"mode": "valid", "classes": ["car"]}


def test_stream_zones_empty_for_unsaved_stream(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam1": {"zones": [{"points": [[1, 1], [2, 2], [3, 3]]}]}})
    assert stream_zones(config, "cam0") == []


def test_stream_zones_empty_when_file_missing(tmp_path):
    config = {"zones": {"file_path": str(tmp_path / "missing.txt")}}
    assert stream_zones(config, "cam0") == []


def test_stream_zones_defaults_ids_and_names_when_absent(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"zones": [
        {"points": [[0, 0], [10, 0], [10, 10]]},
        {"points": [[0, 0], [20, 0], [20, 20]]},
    ]}})

    zones = stream_zones(config, "cam0")

    assert [z["id"] for z in zones] == ["z0", "z1"]
    assert [z["name"] for z in zones] == ["Zone 1", "Zone 2"]


def test_stream_zones_legacy_single_zone_entry_wrapped(tmp_path):
    config = _config(tmp_path)
    # Pre-multi-zone on-disk shape: one zone object keyed directly under the stream id.
    _write(config, {"cam0": {
        "points": [[100, 50], [900, 50], [900, 450]],
        "intrusion": {"mode": "valid", "classes": ["person"]},
        "overcrowding": {"enabled": False, "threshold": 5, "classes": ["person"]},
    }})

    zones = stream_zones(config, "cam0")

    assert len(zones) == 1
    assert zones[0]["id"] == "z0"
    assert zones[0]["name"] == "Zone 1"
    assert zones[0]["intrusion"] == {"mode": "valid", "classes": ["person"]}


def test_stream_zones_legacy_type_becomes_intrusion_mode(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"points": [[1, 1], [2, 2], [3, 3]], "type": "valid"}})

    zones = stream_zones(config, "cam0")

    assert zones[0]["intrusion"] == {"mode": "valid", "classes": ["person"]}
    assert zones[0]["overcrowding"]["enabled"] is False


def test_zone_by_id_returns_matching_zone(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"zones": [
        {"id": "z0", "name": "A", "points": [[0, 0], [10, 0], [10, 10]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
        {"id": "z1", "name": "B", "points": [[0, 0], [20, 0], [20, 20]],
         "intrusion": {"mode": "valid", "classes": ["person"]}},
    ]}})

    assert zone_by_id(config, "cam0", "z1")["name"] == "B"
    assert zone_by_id(config, "cam0", "z1")["intrusion"]["mode"] == "valid"


def test_zone_by_id_none_when_absent(tmp_path):
    config = _config(tmp_path)
    _write(config, {"cam0": {"zones": [{"id": "z0", "name": "A",
                                        "points": [[0, 0], [10, 0], [10, 10]]}]}})
    assert zone_by_id(config, "cam0", "z9") is None
    assert zone_by_id(config, "cam0", None) is None
    assert zone_by_id(config, "nope", "z0") is None
