import json

from face_regions import face_regions_payload

DOOR = {"id": "c-door", "name": "door", "type": "entrance_line", "points": [[0.4, 0.5], [0.6, 0.5]], "inside": [0.5, 0.8]}
AREA = {"id": "c-door-area-1", "name": "door area", "type": "door_area", "points": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7]]}


def test_only_the_entrance_line_cameras_door_areas_are_published():
    z = {"barber-c": [DOOR, AREA], "barber-l": [dict(AREA, id="l-x")]}
    assert json.loads(face_regions_payload(z)) == {"barber-c": [AREA["points"]]}


def test_no_door_area_publishes_an_empty_object():
    assert json.loads(face_regions_payload({"barber-c": [DOOR]})) == {}
    assert json.loads(face_regions_payload({})) == {}


def test_register_source_2_publishes_every_cameras_face_areas():
    zs = {"ph-5": [{"id": "f1", "type": "face_area", "points": [[0, 0], [1, 0], [1, 1]], "chair": "z-r1"},
                   {"id": "d", "type": "door_area", "points": [[0, 0], [1, 0], [0, 1]]}],
          "ph-2": [{"id": "f2", "type": "face_area", "points": [[0, 0], [1, 0], [0, 1]], "chair": "z-l7"}],
          "ph-3": []}
    assert json.loads(face_regions_payload(zs, register_source=2)) == {"match": "box", "regions": {
        "ph-5": [[[0, 0], [1, 0], [1, 1]]], "ph-2": [[[0, 0], [1, 0], [0, 1]]]}}     # the person box overlapping the area
    assert json.loads(face_regions_payload(zs, register_source=1)) == {}     # no entrance line: no door area


def test_register_source_2_without_face_areas_publishes_no_regions():
    assert json.loads(face_regions_payload({"ph-3": []}, register_source=2)) == {"match": "box", "regions": {}}
