"""modules.ppe.adapter: ObjectViews -> the vendored rules' Detections, and track ids back onto their results."""
from perception.engine.frame import ObjectView
from perception.modules.ppe.adapter import UNTRACKED, frame_input, to_bbox
from perception.modules.ppe.compliance.compliance import evaluate_frame
from perception.modules.ppe.compliance.config import ComplianceConfig
from perception.modules.ppe.compliance.types import ItemStatus

MUX = (1080, 720)


def person(oid, left, top, width, height, conf=0.9, gie="pgie", label="person"):
    return ObjectView(oid, label, conf, conf, left, top, width, height, gie_name=gie)


def ppe(label, conf, left, top, width, height, gie="pgie_ppe"):
    return ObjectView(UNTRACKED, label, conf, conf, left, top, width, height, gie_name=gie)


def classes(inp):
    return [(d.class_name, d.confidence) for d in inp.detections]


def test_box_is_converted_to_corners_in_the_same_pixels():
    o = person(1, 100.0, 50.0, 40.0, 120.0)
    assert to_bbox(o) == (100.0, 50.0, 140.0, 170.0)
    inp = frame_input([o], MUX)
    assert inp.detections[0].bbox == (100.0, 50.0, 140.0, 170.0)
    assert inp.image_size == (1080, 720)


def test_persons_come_only_from_the_pgie_and_ppe_only_from_the_ppe_detector():
    objs = [person(1, 100, 100, 60, 200),
            person(UNTRACKED, 300, 100, 60, 200, gie="pgie_ppe", label="Person"),   # VoxDroid's own person class
            person(2, 500, 100, 60, 200, gie="pgie_fire"),                         # a "person" from another GIE
            ppe("Hardhat", 0.8, 110, 90, 30, 30, gie="pgie"),                     # a PPE label from another GIE
            ppe("Hardhat", 0.8, 115, 92, 30, 30),
            ppe("face", 0.9, 120, 100, 20, 20, gie="sgie_face")]
    assert classes(frame_input(objs, MUX)) == [("person", 0.9), ("Hardhat", 0.8)]


def test_every_ppe_label_is_kept_down_to_the_detector_floor_and_other_labels_are_not():
    objs = [ppe("Hardhat", 0.10, 0, 0, 10, 10), ppe("NO-Hardhat", 0.5, 0, 0, 10, 10),
            ppe("Safety Vest", 0.101, 0, 0, 10, 10), ppe("NO-Safety Vest", 0.9, 0, 0, 10, 10),
            ppe("Mask", 0.9, 0, 0, 10, 10), ppe("Safety Cone", 0.9, 0, 0, 10, 10)]
    assert classes(frame_input(objs, MUX)) == [("Hardhat", 0.10), ("NO-Hardhat", 0.5), ("Safety Vest", 0.101),
                                               ("NO-Safety Vest", 0.9)]


def test_shadow_and_untracked_persons_are_left_out():
    objs = [person(1, 100, 100, 60, 200),
            person(2, 300, 100, 60, 200, conf=-0.1),          # shadow track: the detector missed it this frame
            person(UNTRACKED, 500, 100, 60, 200),            # never tracked: no id to smooth over
            ppe("Hardhat", -0.1, 110, 90, 30, 30)]           # a shadow PPE box is no evidence either
    inp = frame_input(objs, MUX)
    assert [(d.class_name, d.bbox[0]) for d in inp.detections] == [("person", 100)]
    assert list(inp.track_ids.values()) == [1]


def test_persons_come_first_then_ppe_each_in_metadata_order():
    objs = [ppe("Safety Vest", 0.7, 0, 0, 10, 10), person(7, 100, 100, 60, 200), ppe("Hardhat", 0.8, 0, 0, 10, 10),
            person(3, 300, 100, 60, 200)]
    inp = frame_input(objs, MUX)
    assert classes(inp) == [("person", 0.9), ("person", 0.9), ("Safety Vest", 0.7), ("Hardhat", 0.8)]
    assert list(inp.track_ids.values()) == [7, 3]


def test_track_ids_go_back_to_the_right_person_through_the_rules():
    # Two workers side by side in mux pixels: the left one with a helmet, the right one with a bare head.
    objs = [person(41, 100, 200, 100, 300), person(17, 400, 200, 100, 300),
            ppe("Hardhat", 0.85, 125, 185, 50, 75), ppe("NO-Hardhat", 0.8, 425, 185, 50, 75)]
    inp = frame_input(objs, MUX)
    results, _ = evaluate_frame(inp.detections, ComplianceConfig(person_classes=["person"]), inp.image_size)
    inp.attach_track_ids(results)
    assert {r.track_id: r.items["helmet"].status for r in results} == {41: ItemStatus.PRESENT, 17: ItemStatus.ABSENT}


def test_truncation_is_judged_against_the_mux_frame():
    # A head cut off by the top of the 1080x720 mux frame: a confident NO-Hardhat must not become a violation.
    objs = [person(5, 400, 0, 100, 300), ppe("NO-Hardhat", 0.9, 425, 0, 50, 60)]
    inp = frame_input(objs, MUX)
    results, _ = evaluate_frame(inp.detections, ComplianceConfig(person_classes=["person"]), inp.image_size)
    (r,) = inp.attach_track_ids(results)
    assert (r.track_id, r.items["helmet"].status, r.items["helmet"].reason) == (5, ItemStatus.UNKNOWN,
                                                                                 "person_truncated")
