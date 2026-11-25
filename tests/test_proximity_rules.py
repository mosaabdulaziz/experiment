"""proximity.rules: the zones with proximity on, from the app's zones file."""
import json
import logging
import types

from perception.modules.proximity.rules import load_zone_rules

DEFAULTS = types.SimpleNamespace(
    person_classes=["person"], vehicle_classes=["car", "truck"], max_gap=0.75, stationary_window_s=2.0,
    stationary_motion=0.5, persist_s=1.0, grace_s=0.5, cooldown_s=10.0, depth_max_gap=0.25, hidden_feet="unconfirmed")
SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def _write(tmp_path, data):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_only_zones_with_proximity_enabled_become_rules(tmp_path):
    path = _write(tmp_path, {"cam-1": {"zones": [
        {"id": "a", "name": "Yard", "points": SQUARE, "proximity": {"enabled": True}},
        {"id": "b", "points": SQUARE, "proximity": {"enabled": False}},
        {"id": "c", "points": SQUARE, "intrusion": {"mode": "restricted"}}]}})
    (rule,) = load_zone_rules(path, DEFAULTS)["cam-1"]
    assert (rule.zone_id, rule.name, rule.max_gap, rule.persist_s, rule.depth_check, rule.depth_max_gap,
            rule.hidden_feet) == ("a", "Yard", 0.75, 1.0, True, 0.25, "unconfirmed")   # depth on by default (Ticket 6)
    assert rule.vehicle_classes == frozenset({"car", "truck"}) and rule.person_classes == frozenset({"person"})


def test_a_zone_overrides_the_defaults(tmp_path):
    path = _write(tmp_path, {"cam-1": {"zones": [{"id": "a", "points": SQUARE, "proximity": {
        "enabled": True, "max_gap": 0.5, "vehicle_classes": ["truck"], "depth_check": True,
        "hidden_feet": "suppress", "cooldown_s": 0}}]}})
    (rule,) = load_zone_rules(path, DEFAULTS)["cam-1"]
    assert (rule.max_gap, rule.vehicle_classes, rule.depth_check, rule.hidden_feet, rule.cooldown_s) == \
        (0.5, frozenset({"truck"}), True, "suppress", 0.0)


def test_membership_is_the_polygon_in_mux_pixels(tmp_path):
    path = _write(tmp_path, {"cam-1": {"zones": [{"id": "a", "points": SQUARE, "proximity": {"enabled": True}}]}})
    (rule,) = load_zone_rules(path, DEFAULTS)["cam-1"]
    assert rule.contains(50, 50) and rule.contains(100, 50) and not rule.contains(150, 50)


def test_a_bad_block_skips_that_zone_only(tmp_path, caplog):
    path = _write(tmp_path, {"cam-1": {"zones": [
        {"id": "neg", "points": SQUARE, "proximity": {"enabled": True, "max_gap": -1}},
        {"id": "ok", "points": SQUARE, "proximity": {"enabled": True}},
        {"id": "word", "points": SQUARE, "proximity": {"enabled": True, "persist_s": "long"}},
        {"id": "feet", "points": SQUARE, "proximity": {"enabled": True, "hidden_feet": "maybe"}},
        {"id": "line", "points": [[0, 0], [10, 10]], "proximity": {"enabled": True}}]}})
    with caplog.at_level(logging.ERROR, logger="iris.sidecar"):
        rules = load_zone_rules(path, DEFAULTS)
    assert [r.zone_id for r in rules["cam-1"]] == ["ok"]
    assert caplog.text.count("zone skipped") == 4


def test_a_missing_or_broken_file_is_no_zones(tmp_path):
    assert load_zone_rules(str(tmp_path / "nope.json"), DEFAULTS) == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_zone_rules(str(broken), DEFAULTS) == {}


def test_a_legacy_single_zone_entry_gets_the_default_id_and_name(tmp_path):
    path = _write(tmp_path, {"cam-2": {"points": SQUARE, "proximity": {"enabled": True}}})
    (rule,) = load_zone_rules(path, DEFAULTS)["cam-2"]
    assert (rule.zone_id, rule.name) == ("z0", "Zone 1")
