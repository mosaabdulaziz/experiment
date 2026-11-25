"""The per-zone proximity defaults live in one file that the module and the app both read."""
import json
from pathlib import Path

from perception.modules.proximity import Settings, rules, zone_defaults

FILE = Path(__file__).resolve().parents[1] / "perception" / "modules" / "proximity" / "zone_defaults.json"


def test_the_module_defaults_are_the_files():
    raw = json.loads(FILE.read_text())["defaults"]
    s = Settings()
    for key, value in raw.items():
        if key != "depth_check":                    # a zone setting only: the module has no default of its own
            assert getattr(s, key) == value, key
    assert zone_defaults.DEFAULTS == raw


def test_zone_rules_take_their_ranges_and_choices_from_the_file():
    raw = json.loads(FILE.read_text())
    assert rules.HIDDEN_FEET == tuple(raw["hidden_feet"])
    assert set(zone_defaults.POSITIVE) | set(zone_defaults.NON_NEGATIVE) == {
        "max_gap", "stationary_window_s", "stationary_motion", "persist_s", "grace_s", "cooldown_s", "depth_max_gap"}


def test_a_zone_without_settings_gets_the_file_defaults():
    zone = {"id": "z0", "points": [[0, 0], [10, 0], [10, 10]], "proximity": {"enabled": True}}
    rule = rules.zone_rule(zone, 0, Settings())
    # Ticket 6: the depth check is on by default, at a relative depth gap of 0.25
    assert (rule.max_gap, rule.depth_check, rule.depth_max_gap, rule.hidden_feet) == (0.75, True, 0.25, "unconfirmed")
