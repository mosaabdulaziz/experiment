"""sidecar.zone_config.load_stream_detection_configs — per-stream zone-list load.

Pure Python (no gi/pyds), importable off-GPU (same convention as test_zone_dedup.py)."""
import json

from perception.modules.zones.config import load_stream_detection_configs

_DEF_I = ["person"]
_DEF_O = ["person"]
_DEF_T = 5


def _write(path, zones):
    path.write_text(json.dumps(zones))
    return str(path)


def test_missing_file_returns_empty(tmp_path):
    assert load_stream_detection_configs(str(tmp_path / "nope.txt"), _DEF_I, _DEF_O, _DEF_T) == {}


def test_new_schema_single_zone(tmp_path):
    path = _write(tmp_path / "z.txt", {
        "cam0": {"zones": [{
            "id": "z0", "name": "Dock",
            "points": [[1, 1], [2, 2], [3, 3]],
            "intrusion": {"mode": "valid", "classes": ["person", "car"]},
            "overcrowding": {"enabled": True, "threshold": 9, "classes": ["person"]},
        }]}
    })
    out = load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)
    assert out["cam0"] == [{
        "label": "z0",
        "name": "Dock",
        "mode": "valid",
        "intrusion_classes": {"person", "car"},
        "overcrowding_enabled": True,
        "overcrowding_threshold": 9,
        "overcrowding_classes": {"person"},
    }]


def test_multiple_zones_preserve_order_and_labels(tmp_path):
    path = _write(tmp_path / "z.txt", {
        "cam0": {"zones": [
            {"id": "z0", "name": "A", "points": [[1, 1], [2, 2], [3, 3]],
             "intrusion": {"mode": "restricted", "classes": ["person"]}},
            {"id": "z1", "name": "B", "points": [[4, 4], [5, 5], [6, 6]],
             "intrusion": {"mode": "off"},
             "overcrowding": {"enabled": True, "threshold": 3, "classes": ["car"]}},
        ]}
    })
    out = load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)
    assert [z["label"] for z in out["cam0"]] == ["z0", "z1"]
    assert out["cam0"][0]["mode"] == "restricted"
    assert out["cam0"][1]["mode"] == "off"
    assert out["cam0"][1]["overcrowding_enabled"] is True
    assert out["cam0"][1]["overcrowding_classes"] == {"car"}


def test_default_labels_and_names_when_absent(tmp_path):
    path = _write(tmp_path / "z.txt", {
        "cam0": {"zones": [
            {"points": [[1, 1], [2, 2], [3, 3]], "intrusion": {"mode": "restricted"}},
            {"points": [[4, 4], [5, 5], [6, 6]], "intrusion": {"mode": "valid"}},
        ]}
    })
    out = load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)
    assert [z["label"] for z in out["cam0"]] == ["z0", "z1"]
    assert [z["name"] for z in out["cam0"]] == ["Zone 1", "Zone 2"]


def test_legacy_single_zone_entry_wrapped(tmp_path):
    path = _write(tmp_path / "z.txt", {
        "cam0": {"points": [[1, 1], [2, 2], [3, 3]],
                 "intrusion": {"mode": "restricted", "classes": ["person"]}}
    })
    out = load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)
    assert len(out["cam0"]) == 1
    assert out["cam0"][0]["label"] == "z0"
    assert out["cam0"][0]["mode"] == "restricted"


def test_legacy_type_entry_maps_to_mode(tmp_path):
    path = _write(tmp_path / "z.txt", {"cam0": {"points": [[1, 1], [2, 2], [3, 3]], "type": "valid"}})
    out = load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)
    assert out["cam0"][0]["mode"] == "valid"
    assert out["cam0"][0]["intrusion_classes"] == {"person"}
    assert out["cam0"][0]["overcrowding_enabled"] is False
    assert out["cam0"][0]["overcrowding_threshold"] == 5


def test_defaults_used_when_classes_absent(tmp_path):
    path = _write(tmp_path / "z.txt", {
        "cam0": {"zones": [{"points": [[1, 1], [2, 2], [3, 3]],
                            "intrusion": {"mode": "restricted"},
                            "overcrowding": {"enabled": True}}]}
    })
    out = load_stream_detection_configs(path, ["person"], ["person"], 4)
    assert out["cam0"][0]["intrusion_classes"] == {"person"}
    assert out["cam0"][0]["overcrowding_classes"] == {"person"}
    assert out["cam0"][0]["overcrowding_threshold"] == 4


def test_stream_with_no_zones_is_absent(tmp_path):
    path = _write(tmp_path / "z.txt", {"cam0": {"zones": []}})
    assert load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T) == {}


def test_a_zone_with_only_a_proximity_block_is_not_an_intrusion_zone(tmp_path):
    # the proximity module and multicam read the same zones file and camera ids: a zone written for proximity alone
    # must not raise zone_intrusion in multicam
    path = _write(tmp_path / "z.txt", {
        "cam0": {"zones": [{"id": "z0", "points": [[1, 1], [2, 2], [3, 3]], "proximity": {"enabled": True}}]}
    })
    assert load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)["cam0"][0]["mode"] == "off"


def test_a_legacy_zone_with_only_points_is_still_restricted(tmp_path):
    path = _write(tmp_path / "z.txt", {"cam0": {"points": [[1, 1], [2, 2], [3, 3]]}})
    assert load_stream_detection_configs(path, _DEF_I, _DEF_O, _DEF_T)["cam0"][0]["mode"] == "restricted"
