import pytest

from zones import (CHAIR, ENTRANCE_LINE, SEAT, WASHING, DoorLine, FrameGeometry, ZoneValidationError,
                   area_of, centroid, expand, owner_cameras, point_in_polygon, rect_overlaps_polygon,
                   validate_zones, zone_hit, zone_set_problems)

SQUARE = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]
# vertical doorway at x=0.85 from y=0.3 to 0.7; the shop is to its left
DOOR = {"id": "door", "type": ENTRANCE_LINE, "points": [[0.85, 0.3], [0.85, 0.7]], "inside": [0.5, 0.5]}


def test_point_in_polygon_inside_and_outside():
    assert point_in_polygon(0.3, 0.3, SQUARE)
    assert not point_in_polygon(0.5, 0.3, SQUARE)
    assert not point_in_polygon(0.3, 0.3, SQUARE[:2])  # degenerate


def test_rect_overlaps_polygon():
    # the sidecar face gate's cases (tests/test_face_regions.py), on a 0.4-0.6 square
    sq = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    assert rect_overlaps_polygon(0.45, 0.45, 0.55, 0.55, sq)          # inside
    assert rect_overlaps_polygon(0.0, 0.0, 1.0, 1.0, sq)              # around
    assert rect_overlaps_polygon(0.55, 0.0, 0.65, 0.45, sq)           # a corner in
    assert rect_overlaps_polygon(0.3, 0.45, 0.7, 0.55, sq)            # crossing, no corner inside either way
    assert not rect_overlaps_polygon(0.7, 0.7, 0.9, 0.9, sq)
    tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert not rect_overlaps_polygon(0.6, 0.6, 0.9, 0.9, tri)         # in the triangle's bounding box only
    assert not rect_overlaps_polygon(0.0, 0.0, 1.0, 1.0, sq[:2])      # degenerate


def test_anchor_and_foot_normalized():
    geo = FrameGeometry(1000, 500)
    assert geo.anchor((100, 100, 200, 100)) == pytest.approx((0.2, 0.36))
    assert geo.foot((100, 100, 200, 100)) == pytest.approx((0.2, 0.4))
    assert geo.rect((100, 100, 200, 100)) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_seat_points_run_down_the_box_centre_line_from_its_middle_to_the_anchor():
    geo = FrameGeometry(1000, 500)
    pts = geo.seat_points((100, 100, 200, 100))
    assert [p for pt in pts for p in pt] == pytest.approx([0.2, 0.3, 0.2, 0.33, 0.2, 0.36])


def test_zone_hit_returns_first_containing_zone():
    zones = [{"id": "a", "points": SQUARE}, {"id": "b", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    assert zone_hit((0.3, 0.3), zones) == "a"
    assert zone_hit((0.9, 0.9), zones) == "b"
    assert zone_hit((0.3, 0.3), []) is None


def test_expand_grows_about_centroid():
    grown = expand(SQUARE, 1.0)  # twice the size
    assert centroid(grown) == pytest.approx(centroid(SQUARE))
    assert grown[0] == pytest.approx([0.1, 0.1])
    assert point_in_polygon(0.45, 0.3, grown) and not point_in_polygon(0.45, 0.3, SQUARE)


def test_door_line_side_is_set_by_the_inside_point_not_the_drawing_direction():
    for zone in (DOOR, {**DOOR, "points": DOOR["points"][::-1]}):
        line = DoorLine.from_zone(zone)
        assert line.signed((0.8, 0.5)) == pytest.approx(0.05)     # shop side: positive
        assert line.signed((0.95, 0.5)) == pytest.approx(-0.1)    # street side: negative
    flipped = DoorLine.from_zone({**DOOR, "inside": [0.95, 0.5]})
    assert flipped.signed((0.8, 0.5)) == pytest.approx(-0.05)


def test_door_line_extent_distance_and_just_inside():
    line = DoorLine.from_zone(DOOR)
    assert line.along((0.8, 0.3)) == pytest.approx(0.0) and line.along((0.8, 0.7)) == pytest.approx(1.0)
    assert line.within_extent((0.8, 0.75)) and not line.within_extent((0.8, 0.95))
    assert line.distance_to((0.85, 0.9)) == pytest.approx(0.2)
    assert line.just_inside((0.8, 0.5), band=0.08)
    assert not line.just_inside((0.6, 0.5), band=0.08)   # too deep inside
    assert not line.just_inside((0.9, 0.5), band=0.08)   # outside


def test_validate_cleans_and_keeps_client_ids():
    out = validate_zones([
        {"id": "chair-1", "name": "Chair 1", "points": [[0.1, 0.1], [1.2, 0.1], [0.5, -0.3]],
         "type": "chair", "pair": "L3", "owner": True, "junk": 1},
        {"points": SQUARE, "type": "waiting"},
        {**DOOR, "points": [[0.85, 0.3], [0.85, 1.4]]},
        {"id": "s1", "points": SQUARE, "type": "seat", "seat_of": "chair-1"},
    ])
    assert out[0] == {"id": "chair-1", "name": "Chair 1", "type": CHAIR, "pair": "L3", "owner": True,
                      "points": [[0.1, 0.1], [1.0, 0.1], [0.5, 0.0]]}
    assert out[1]["id"].startswith("zone-") and out[1]["name"] == out[1]["id"]
    assert out[2] == {"id": "door", "name": "door", "type": ENTRANCE_LINE,
                      "points": [[0.85, 0.3], [0.85, 1.0]], "inside": [0.5, 0.5]}
    assert out[3]["seat_of"] == "chair-1" and "pair" not in out[3]


def test_validate_ignores_margin_and_false_owner():
    out = validate_zones([{"id": "w", "points": SQUARE, "type": "waiting", "margin": 2, "owner": False}])
    assert "margin" not in out[0] and "owner" not in out[0]


def test_area_of_is_the_pair_else_the_zone():
    assert area_of({"id": "z1", "pair": "L3"}) == "L3"
    assert area_of({"id": "z1"}) == "z1"


def chair(zid, pair, owner=False):
    z = {"id": zid, "name": zid, "type": CHAIR, "points": SQUARE, "pair": pair}
    return {**z, "owner": True} if owner else z


def seat(zid, of):
    return {"id": zid, "name": zid, "type": SEAT, "points": SQUARE, "seat_of": of}


def test_owner_cameras_uses_the_owner_flag_or_the_only_camera():
    zones = {"barber-c": [chair("c-L3", "L3"), {"id": "sofa", "type": "waiting", "points": SQUARE}],
             "barber-l": [chair("l-L3", "L3", owner=True), seat("l-s3", "l-L3")]}
    assert owner_cameras(zones) == {"L3": "barber-l", "sofa": "barber-c"}


def test_zone_set_problems_names_every_broken_pair():
    zones = {"barber-c": [chair("c-L3", "L3"), chair("c-L4", "L4", owner=True), chair("c-L5", "L5")],
             "barber-l": [chair("l-L3", "L3", owner=True), chair("l-L4", "L4", owner=True),
                          seat("l-s3", "l-L3")]}
    problems = zone_set_problems(zones)
    assert {"code": "pair_single_camera", "pair": "L5", "camera": "barber-c"} in problems
    assert {"code": "pair_owner_count", "pair": "L5", "count": 0} in problems
    assert {"code": "pair_owner_count", "pair": "L4", "count": 2} in problems
    assert {"code": "chair_without_seat", "camera": "barber-l", "zone": "l-L4"} in problems
    assert {"code": "chair_without_seat", "camera": "barber-c", "zone": "c-L4"} in problems
    assert not any("L3" in str(p) for p in problems)


@pytest.mark.parametrize("bad, message", [
    ([{"id": "x", "points": SQUARE, "type": "entrance"}], "type must be"),
    ([{"id": "x", "points": SQUARE[:2], "type": "chair"}], "at least 3 points"),
    ([{"id": "x", "points": [["a", 1], [0, 0], [1, 1]], "type": "chair"}], "numbers"),
    ([{"id": "x", "points": SQUARE, "type": "chair"}, {"id": "x", "points": SQUARE, "type": "waiting"}],
     "duplicate"),
    ([{**DOOR, "points": SQUARE}], "exactly 2 points"),
    ([{**DOOR, "points": [[0.5, 0.5], [0.501, 0.5]]}], "too short"),
    ([{k: v for k, v in DOOR.items() if k != "inside"}], "needs an `inside` point"),
    ([{**DOOR, "inside": [0.85, 0.5]}], "must not lie on the line"),
    ([{"id": "x", "points": SQUARE, "type": "sofa"}], "type must be"),
    ([{"id": "s", "points": SQUARE, "type": "seat"}], "needs `seat_of`"),
    ([{"id": "s", "points": SQUARE, "type": "seat", "seat_of": "nope"}], "not a chair in this camera"),
    ([{**DOOR, "pair": "p"}], "can't be paired"),
    ({"not": "a list"}, "list"),
])
def test_validate_rejects_bad_zones(bad, message):
    with pytest.raises(ZoneValidationError, match=message):
        validate_zones(bad)


def test_a_row_zone_validates_and_is_not_an_area():
    from zones import AREA_TYPES, ROW, validate_zones
    (z,) = validate_zones([{"id": "c-row-l", "name": "left row", "type": "row",
                            "points": [[0.0, 0.5], [0.3, 0.5], [0.3, 0.9]]}])
    assert z["type"] == ROW == "row" and ROW not in AREA_TYPES


def test_a_door_area_zone_validates_and_is_not_an_area_type():
    from zones import AREA_TYPES
    assert validate_zones([{"id": "d", "type": "door_area", "points": SQUARE}])[0]["type"] == "door_area"
    assert "door_area" not in AREA_TYPES


def test_door_area_without_the_entrance_line_is_a_problem():
    door = {"id": "d", "name": "d", "type": "door_area", "points": SQUARE}
    line = {"id": "e", "name": "e", "type": "entrance_line", "points": [[0, 0], [1, 0]], "inside": [0.5, 0.5]}
    assert {"code": "door_area_ignored", "camera": "barber-l"} in zone_set_problems({"barber-l": [door], "barber-c": [line, door]})
    assert not any(p["code"] == "door_area_ignored" for p in zone_set_problems({"barber-c": [line, door]}))


def _fsq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_a_face_area_needs_its_chair():
    from zones import FACE_AREA, validate_zones
    ok = validate_zones([{"id": "f1", "type": FACE_AREA, "points": _fsq(0.1, 0.1, 0.3, 0.3), "chair": "z-r1"}])
    assert ok[0]["chair"] == "z-r1" and ok[0]["type"] == "face_area"
    with pytest.raises(ZoneValidationError, match="chair"):
        validate_zones([{"id": "f1", "type": FACE_AREA, "points": _fsq(0.1, 0.1, 0.3, 0.3)}])


def test_a_face_area_may_name_a_chair_of_another_camera_but_not_an_unknown_one():
    from zones import zone_set_problems
    chair = {"id": "z-r1", "name": "R1", "type": "chair", "points": _fsq(0.1, 0.1, 0.5, 0.5)}
    seat = {"id": "z-r1s", "name": "R1 seat", "type": "seat", "points": _fsq(0.2, 0.2, 0.4, 0.4), "seat_of": "z-r1"}
    good = {"id": "f1", "name": "R1 face", "type": "face_area", "points": _fsq(0.1, 0.1, 0.3, 0.3), "chair": "z-r1"}
    bad = {**good, "id": "f2", "name": "ghost", "chair": "nope"}
    zs = {"ph-5": [chair, seat], "ph-3": [good, bad]}
    assert zone_set_problems(zs) == [{"code": "face_area_unknown_chair", "camera": "ph-3", "zone": "ghost"}]


def test_a_pair_whose_other_half_is_on_a_parked_camera_is_not_single():
    """2026-10-15: ph-4 left the live Palm run (R3/R4 owned by ph-6); its zones stay stored to switch back. Its pair
    halves are parked, not missing -- but a pair with no half anywhere else still is."""
    zones = {"barber-l": [chair("l-L3", "L3", owner=True), seat("l-s3", "l-L3"),
                          chair("l-L4", "L4", owner=True), seat("l-s4", "l-L4")]}
    parked = {"barber-c": [chair("c-L3", "L3")]}
    problems = zone_set_problems(zones, parked)
    assert not any("L3" in str(p) for p in problems)
    assert {"code": "pair_single_camera", "pair": "L4", "camera": "barber-l"} in problems


# ── pedicure chairs (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md) ──────────────────────────────
from zones import PEDICURE, zone_set_problems  # noqa: E402

PSQ = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]


def test_a_chair_keeps_its_pedicure_kind():
    (z,) = validate_zones([{"id": "p1", "name": "P1", "type": "chair", "points": PSQ, "kind": "pedicure"}])
    assert z["kind"] == PEDICURE


def test_a_chair_without_kind_has_none():
    (z,) = validate_zones([{"id": "c1", "name": "R1", "type": "chair", "points": PSQ}])
    assert "kind" not in z


def test_an_unknown_chair_kind_is_rejected():
    with pytest.raises(ZoneValidationError, match="kind"):
        validate_zones([{"id": "p1", "name": "P1", "type": "chair", "points": PSQ, "kind": "massage"}])


def test_kind_on_a_zone_that_is_not_a_chair_is_dropped():
    (z,) = validate_zones([{"id": "w1", "name": "wash", "type": "washing", "points": PSQ, "kind": "pedicure"}])
    assert "kind" not in z


def test_a_pair_drawn_as_two_kinds_is_a_problem():
    zones = {"a": [{"id": "a1", "name": "P1", "type": "chair", "points": PSQ, "pair": "P1", "owner": True, "kind": "pedicure"},
                   {"id": "a1s", "name": "s", "type": "seat", "points": PSQ, "seat_of": "a1"}],
             "b": [{"id": "b1", "name": "P1", "type": "chair", "points": PSQ, "pair": "P1"}]}
    assert {"code": "pair_kind_mismatch", "pair": "P1"} in zone_set_problems(zones)
