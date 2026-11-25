"""Where the attendance access check applies (products/attendance/zones.py effective_restricted_zones).

A camera listed in the policy's `access_control` is checked inside the restricted zones drawn on it, or on the whole
stream when none is drawn. A camera that is not listed keeps the old behaviour: only inside restricted zones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "products" / "attendance"))

from zones import WHOLE_STREAM, effective_restricted_zones, role_allowed, zone_hit  # noqa: E402

ACCESS = {"cam-22": {"allowed_roles": ["passenger"]}}
SEAT = {"id": "seat-1", "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]}
DOOR = {"id": "door", "type": "restricted", "allowed_roles": ["staff"],
        "points": [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]]}


def test_listed_camera_without_a_drawn_zone_is_checked_on_the_whole_stream():
    zs = effective_restricted_zones("cam-22", [], ACCESS)
    assert [z["id"] for z in zs] == [WHOLE_STREAM]
    assert zs[0]["allowed_roles"] == ["passenger"]


def test_seat_zones_do_not_count_as_drawn_restricted_zones():
    zs = effective_restricted_zones("cam-22", [SEAT], ACCESS)
    assert [z["id"] for z in zs] == [WHOLE_STREAM]


def test_a_drawn_restricted_zone_narrows_the_check_to_itself_with_its_own_roles():
    zs = effective_restricted_zones("cam-22", [SEAT, DOOR], ACCESS)
    assert [z["id"] for z in zs] == ["door"]
    assert zs[0]["allowed_roles"] == ["staff"]          # not the camera's roles


def test_an_unlisted_camera_is_not_checked_without_a_zone():
    assert effective_restricted_zones("cam-1", [], ACCESS) == []
    assert effective_restricted_zones("cam-1", [], None) == []
    assert effective_restricted_zones("cam-1", [], {}) == []


def test_an_unlisted_camera_with_a_drawn_zone_behaves_as_before():
    assert [z["id"] for z in effective_restricted_zones("cam-1", [DOOR], ACCESS)] == ["door"]


def test_whole_stream_covers_the_frame_edges_even_when_the_box_anchor_leaves_the_frame():
    zs = effective_restricted_zones("cam-22", [], ACCESS)
    for box in [(0, 0, 50, 100), (1000, 600, 120, 200), (1040, 500, 200, 300), (-30, 100, 60, 300)]:
        assert zone_hit(box, zs) == WHOLE_STREAM


def test_a_listed_camera_with_no_roles_allows_nobody():
    zs = effective_restricted_zones("cam-22", [], {"cam-22": {}})
    assert zs[0]["allowed_roles"] == [] and not role_allowed("passenger", zs[0]) and not role_allowed(None, zs[0])
