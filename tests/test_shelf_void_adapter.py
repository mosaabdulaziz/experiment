"""modules.shelf_void.adapter: ObjectViews -> the heat map's (x1, y1, x2, y2, conf) void boxes."""
from perception.engine.frame import ObjectView
from perception.modules.shelf_void.adapter import EMPTY_LABEL, SHELF_GIE, void_boxes
from perception.modules.shelf_void.core import DetectorConfig, ShelfVoidDetector

UNTRACKED = 0xFFFFFFFFFFFFFFFF
MUX = (1080, 720)


def box(label, conf, left, top, width, height, gie=SHELF_GIE, oid=UNTRACKED):
    return ObjectView(oid, label, conf, conf, left, top, width, height, gie_name=gie)


def test_box_is_converted_to_corners_in_the_same_pixels_with_its_confidence():
    assert void_boxes([box("empty", 0.03, 100.0, 50.0, 40.0, 20.0)]) == [(100.0, 50.0, 140.0, 70.0, 0.03)]


def test_only_the_shelf_detectors_empty_boxes_count_down_to_its_floor():
    objs = [box("empty", 0.02, 0, 0, 10, 10),                       # the detector's floor
            box("empty", 0.9, 10, 0, 10, 10, gie="pgie"),          # an "empty" label from another GIE
            box("product", 0.9, 20, 0, 10, 10),                     # the model's other class
            box("person", 0.9, 30, 0, 10, 10, gie="pgie", oid=4),   # the COCO pgie
            box("empty", 0.5, 40, 0, 10, 10)]
    assert [b[0] for b in void_boxes(objs)] == [0, 40]


def test_a_shadow_box_is_no_evidence():
    assert void_boxes([box("empty", -0.1, 0, 0, 10, 10)]) == []


def test_another_model_only_needs_its_own_empty_label():
    objs = [box("void", 0.4, 0, 0, 10, 10), box(EMPTY_LABEL, 0.4, 20, 0, 10, 10)]
    assert [b[0] for b in void_boxes(objs, label="void")] == [0]


def test_adapter_boxes_drive_the_heat_map_in_mux_pixels():
    det = ShelfVoidDetector(DetectorConfig(min_area=100))
    objs = [box("empty", 0.05, 100, 100, 80, 40)]
    for _ in range(30):
        regions = det.update_boxes(void_boxes(objs), MUX)
    assert det.heat.shape == (720, 1080)
    (r,) = regions
    x1, y1, x2, y2 = r.box
    assert abs(x1 - 100) <= 2 and abs(y1 - 100) <= 2 and abs(x2 - 180) <= 2 and abs(y2 - 140) <= 2
