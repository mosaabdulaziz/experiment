"""perception/lib/feature_zones: per-feature active zones (`ppe` / `fire` / `fall` blocks on the shared zones file).

A camera with at least one zone whose block for the feature is on covers only inside those zones; a camera with none
covers the whole frame (safety integration, decision D19)."""
import json

from perception.lib.feature_zones import FeatureZones

SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


def _file(tmp_path, data):
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps(data))
    return str(path)


def test_a_camera_without_feature_zones_is_covered_everywhere(tmp_path):
    zones = FeatureZones.load(_file(tmp_path, {"cam-1": {"zones": [
        {"id": "z0", "name": "Door", "points": SQUARE, "intrusion": {"mode": "restricted"}}]}}), "ppe")
    assert zones.covering("cam-1", 900, 600) == (True, "")
    assert zones.covering("cam-9", 5, 5) == (True, "")


def test_a_camera_with_feature_zones_is_covered_only_inside_them(tmp_path):
    zones = FeatureZones.load(_file(tmp_path, {"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": SQUARE, "ppe": {"enabled": True}}]}}), "ppe")
    assert zones.covering("cam-1", 200, 200) == (True, "bay-a")
    assert zones.covering("cam-1", 900, 600) == (False, "")


def test_a_block_switched_off_or_for_another_feature_does_not_count(tmp_path):
    path = _file(tmp_path, {"cam-1": {"zones": [
        {"id": "z0", "points": SQUARE, "ppe": {"enabled": False}},
        {"id": "z1", "points": SQUARE, "fire": {"enabled": True}}]}})
    assert FeatureZones.load(path, "ppe").covering("cam-1", 900, 600) == (True, "")
    assert FeatureZones.load(path, "fire").covering("cam-1", 900, 600) == (False, "")


def test_a_missing_or_broken_file_means_whole_frame(tmp_path):
    assert FeatureZones.load(str(tmp_path / "absent.txt"), "ppe").covering("cam-1", 1, 1) == (True, "")
    bad = tmp_path / "bad.txt"
    bad.write_text("{not json")
    assert FeatureZones.load(str(bad), "ppe").covering("cam-1", 1, 1) == (True, "")


def test_a_zone_with_too_few_points_is_skipped(tmp_path):
    zones = FeatureZones.load(_file(tmp_path, {"cam-1": {"zones": [
        {"id": "z0", "points": [[0, 0], [5, 5]], "fall": {"enabled": True}}]}}), "fall")
    assert zones.covering("cam-1", 900, 600) == (True, "")


def test_a_zone_with_malformed_points_is_skipped_and_the_others_still_load(tmp_path):
    bad = [[[0, 0], [5], [9, 9]], [[0, 0], ["a", 1], [9, 9]], [[0, 0], None, [9, 9]], [[0, 0], [float("nan"), 1], [9, 9]],
           "abc", {"x": 1}]
    zones = FeatureZones.load(_file(tmp_path, {
        "cam-1": {"zones": [{"id": f"bad-{i}", "points": p, "ppe": {"enabled": True}} for i, p in enumerate(bad)]
                  + [{"id": "bay-a", "points": SQUARE, "ppe": {"enabled": True}}]},
        "cam-2": {"zones": {"id": "not-a-list"}}}), "ppe")
    assert zones.covering("cam-1", 200, 200) == (True, "bay-a")
    assert zones.covering("cam-1", 900, 600) == (False, "")
    assert zones.covering("cam-2", 1, 1) == (True, "")


def test_the_legacy_single_zone_entry(tmp_path):
    zones = FeatureZones.load(_file(tmp_path, {"cam-1": {"points": SQUARE, "fire": {"enabled": True}}}), "fire")
    assert zones.covering("cam-1", 200, 200) == (True, "z0")
    assert zones.covering("cam-1", 900, 600) == (False, "")


def test_a_zone_can_name_what_it_checks_and_zone_at_hands_it_over(tmp_path):
    """The PPE block may carry `items` (which equipment that zone checks); zone_at gives the zone itself, None for a
    whole-frame camera, and (False, None) outside every zone."""
    zones = FeatureZones.load(_file(tmp_path, {"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": SQUARE, "ppe": {"enabled": True, "items": ["helmet"]}},
        {"id": "dock", "name": "Dock", "points": [[400, 100], [600, 100], [600, 300], [400, 300]],
         "ppe": {"enabled": True}},
        {"id": "odd", "name": "Odd", "points": [[700, 100], [900, 100], [900, 300], [700, 300]],
         "ppe": {"enabled": True, "items": "helmet"}}]}}), "ppe")
    covered, zone = zones.zone_at("cam-1", 200, 200)
    assert covered and (zone.zone_id, zone.items) == ("bay-a", ("helmet",))
    assert zones.zone_at("cam-1", 500, 200)[1].items is None              # no list: the module's own items
    assert zones.zone_at("cam-1", 800, 200)[1].items is None              # not a list of names: ignored
    assert zones.zone_at("cam-1", 50, 50) == (False, None)
    assert zones.zone_at("cam-9", 50, 50) == (True, None)
    assert zones.covering("cam-1", 200, 200) == (True, "bay-a")            # fire and fall keep asking covering
