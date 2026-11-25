"""Unit tests for sidecar/face_regions.py -- which persons the face SGIE may see (barber face register,
2026-10-05): only those whose box bottom-centre lies in a door area of their camera; with the {"match": "box"} form
(seated faces, 2026-10-15), those whose box overlaps a face area. Pure stdlib."""
import json

from perception.modules.face_detect.regions import FaceRegions, point_in_polygon, rect_overlaps_polygon

SQUARE = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]


def regions(value, clock=lambda: 0.0):
    return FaceRegions(lambda: value, refresh_s=10.0, clock=clock)


def at(x, y):
    """a 0.1 x 0.2 person box whose bottom-centre is (x, y): (left, top, width, height)"""
    return x - 0.05, y - 0.2, 0.1, 0.2


def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, SQUARE) and not point_in_polygon(0.7, 0.5, SQUARE)


def test_only_the_listed_camera_and_area_are_allowed():
    r = regions(json.dumps({"barber-c": [SQUARE]}))
    assert r.allows("barber-c", *at(0.5, 0.5))
    assert not r.allows("barber-c", *at(0.9, 0.9))
    assert not r.allows("barber-l", *at(0.5, 0.5))
    assert not r.allows(None, *at(0.5, 0.5))


def test_bad_or_missing_regions_allow_nothing():
    for value in (None, "", "not json", json.dumps([1, 2]), json.dumps({"barber-c": "x"})):
        assert not regions(value).allows("barber-c", *at(0.5, 0.5))

    def boom():
        raise ConnectionError("redis down")
    assert not FaceRegions(boom).allows("barber-c", *at(0.5, 0.5))


def test_regions_are_refreshed_every_10_s():
    now = [0.0]
    value = [json.dumps({})]
    r = FaceRegions(lambda: value[0], refresh_s=10.0, clock=lambda: now[0])
    assert not r.allows("barber-c", *at(0.5, 0.5))
    value[0] = json.dumps({"barber-c": [SQUARE]})
    now[0] = 5.0
    assert not r.allows("barber-c", *at(0.5, 0.5))       # not re-read yet
    now[0] = 10.5
    assert r.allows("barber-c", *at(0.5, 0.5))


def test_malformed_polygons_inside_a_valid_object_allow_nothing():
    for bad in ({"barber-c": [[1, 2, 3]]}, {"barber-c": ["abc"]}):
        assert not regions(json.dumps(bad)).allows("barber-c", *at(0.5, 0.5))


def test_failing_fetch_clears_previous_regions_and_empty_removes_them():
    now = [0.0]
    value = [json.dumps({"barber-c": [SQUARE]})]

    def fetch():
        if value[0] is Exception:
            raise ConnectionError("down")
        return value[0]
    r = FaceRegions(fetch, refresh_s=10.0, clock=lambda: now[0])
    assert r.allows("barber-c", *at(0.5, 0.5))
    value[0], now[0] = Exception, 11.0
    assert not r.allows("barber-c", *at(0.5, 0.5))
    value[0], now[0] = json.dumps({"barber-c": [SQUARE]}), 22.0
    assert r.allows("barber-c", *at(0.5, 0.5))
    value[0], now[0] = json.dumps({}), 33.0
    assert not r.allows("barber-c", *at(0.5, 0.5))


def test_rect_overlaps_polygon():
    assert rect_overlaps_polygon(0.45, 0.45, 0.55, 0.55, SQUARE)          # inside
    assert rect_overlaps_polygon(0.0, 0.0, 1.0, 1.0, SQUARE)              # around
    assert rect_overlaps_polygon(0.55, 0.0, 0.65, 0.45, SQUARE)           # a corner in
    assert rect_overlaps_polygon(0.3, 0.45, 0.7, 0.55, SQUARE)            # crossing, no corner inside either way
    assert not rect_overlaps_polygon(0.7, 0.7, 0.9, 0.9, SQUARE)
    tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert not rect_overlaps_polygon(0.6, 0.6, 0.9, 0.9, tri)             # in the triangle's bounding box only


def test_box_match_allows_a_person_whose_box_overlaps_the_area():
    """face areas are drawn where the seated customer's face shows: his feet are below them"""
    face_area = [[0.4, 0.2], [0.6, 0.2], [0.6, 0.35], [0.4, 0.35]]
    r = regions(json.dumps({"match": "box", "regions": {"ph-1": [face_area]}}))
    seated = (0.42, 0.25, 0.16, 0.5)                    # head in the area, bottom-centre (0.5, 0.75) far below it
    assert r.allows("ph-1", *seated)
    assert not r.allows("ph-1", 0.7, 0.25, 0.1, 0.5)    # beside it
    assert not r.allows("ph-2", *seated)


def test_the_plain_form_and_foot_match_test_the_bottom_centre():
    seated = (0.42, 0.25, 0.16, 0.5)
    face_area = [[0.4, 0.2], [0.6, 0.2], [0.6, 0.35], [0.4, 0.35]]
    for value in ({"ph-1": [face_area]}, {"match": "foot", "regions": {"ph-1": [face_area]}}):
        assert not regions(json.dumps(value)).allows("ph-1", *seated)
    assert regions(json.dumps({"match": "foot", "regions": {"ph-1": [SQUARE]}})).allows("ph-1", *at(0.5, 0.5))


def test_an_unknown_match_or_bad_envelope_allows_nothing():
    for value in ({"match": "head", "regions": {"ph-1": [SQUARE]}}, {"match": "box", "regions": [SQUARE]},
                  {"match": "box"}):
        r = regions(json.dumps(value))
        assert not r.allows("ph-1", 0.45, 0.45, 0.1, 0.1)
