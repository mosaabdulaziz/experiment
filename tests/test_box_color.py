"""box_color: live-feed detection-box color by zone-violation status.

The sidecar itself needs gi/pyds (GPU container only), but box_color is pure
Python and importable off-GPU (same convention as test_zone_dedup.py)."""

from perception.modules.zones.colors import box_color, BOX_GREEN, BOX_RED, BOX_GRAY

_PERSON_CLASSES = {"person"}


def test_violating_person_is_red():
    assert box_color(7, "person", {7}, True, _PERSON_CLASSES) == BOX_RED


def test_compliant_person_is_green():
    assert box_color(7, "person", set(), True, _PERSON_CLASSES) == BOX_GREEN


def test_non_person_is_gray():
    assert box_color(3, "car", set(), True, _PERSON_CLASSES) == BOX_GRAY


def test_violating_id_wins_over_label():
    # violating_ids only ever contains zone_alert_classes members in practice, but the
    # id check comes first by design: a violating id is a violation regardless of label.
    assert box_color(3, "car", {3}, True, _PERSON_CLASSES) == BOX_RED


def test_analytics_off_is_all_green_regardless():
    assert box_color(7, "person", {7}, False, _PERSON_CLASSES) == BOX_GREEN
    assert box_color(3, "car", set(), False, _PERSON_CLASSES) == BOX_GREEN
