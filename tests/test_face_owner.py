"""FaceOwnership: which person box a face belongs to when boxes overlap.

The face network runs on every person box (process-mode=2) and its parent
pointer arrives unset, so the probe has to decide. The old rule -- the
smallest box containing the face centre -- hands a barber's face to the
seated customer she leans over (a seated box is the smaller one), and the
name then bounces between the two tracks. Pure Python, like zone_dedup."""

from perception.modules.face_detect.owner import FaceOwnership, crop_box

# Mux coords. A standing barber and a seated customer whose boxes overlap;
# the customer's box is the smaller one.
BARBER, CUSTOMER = 1, 2
CUSTOMER_BOX = (540.0, 300.0, 160.0, 260.0, CUSTOMER)


def face(left, top, w=40.0, h=50.0, conf=0.8):
    return (left, top, w, h, conf, None)


def owners(result):
    return [f[5] for f in result]


def test_one_box_owns_its_face():
    own = FaceOwnership()
    assert owners(own.assign("cam", [face(560, 315)], [CUSTOMER_BOX], now=0.0)) == [CUSTOMER]


def test_a_face_in_no_box_keeps_no_owner():
    own = FaceOwnership()
    assert owners(own.assign("cam", [face(50, 50)], [CUSTOMER_BOX], now=0.0)) == [None]


def test_a_barber_leaning_over_a_customer_keeps_her_face():
    own = FaceOwnership()
    # She walks in from the left, face in her own box only...
    for step in range(12):
        x = 380.0 + 20.0 * step                    # face slides right 20 px a frame
        barber_box = (x - 40.0, 240.0, 200.0, 400.0, BARBER)
        result = own.assign("cam", [face(x, 330.0)], [barber_box, CUSTOMER_BOX], now=step * 0.04)
        # ...and once it is inside the (smaller) customer box too, it stays hers.
        assert owners(result) == [BARBER], step


def test_the_customers_own_face_stays_the_customers_when_the_barber_covers_it():
    own = FaceOwnership()
    customer_face = face(600, 315)
    own.assign("cam", [customer_face], [CUSTOMER_BOX], now=0.0)
    barber_box = (480.0, 200.0, 260.0, 440.0, BARBER)   # bigger, covering her
    for step in range(1, 10):
        result = own.assign("cam", [customer_face], [barber_box, CUSTOMER_BOX], now=step * 0.04)
        assert owners(result) == [CUSTOMER], step


def test_first_sighting_in_an_overlap_goes_to_the_box_whose_top_is_just_above_it():
    # No history: a head sits at the top of its own body box. The barber leans
    # low beside the customer: her box top is just above the face, the
    # customer's head (his box top) well above it. The old rule picked him.
    barber_low = (470.0, 330.0, 200.0, 290.0, BARBER)
    result = FaceOwnership().assign("cam", [face(600, 345)], [barber_low, CUSTOMER_BOX], now=0.0)
    assert owners(result) == [BARBER]


def test_a_face_sticking_out_above_a_box_is_not_that_boxs():
    # The customer's box cuts through this face: it belongs to a taller body.
    barber_box = (470.0, 250.0, 200.0, 400.0, BARBER)
    result = FaceOwnership().assign("cam", [face(560, 280)], [barber_box, CUSTOMER_BOX], now=0.0)
    assert owners(result) == [BARBER]


def test_a_face_sticking_out_above_every_box_goes_to_the_highest_one():
    # Both box tops cut through the face (the detector clipped the head). The
    # higher top is the body the head sits on -- not the smaller, seated box.
    barber_box = (470.0, 295.0, 200.0, 400.0, BARBER)   # top 295, bigger
    result = FaceOwnership().assign("cam", [face(560, 280)], [barber_box, CUSTOMER_BOX], now=0.0)
    assert owners(result) == [BARBER]


def test_history_expires():
    own = FaceOwnership(memory_s=2.0)
    customer_face = face(600, 345)
    own.assign("cam", [customer_face], [CUSTOMER_BOX], now=0.0)
    barber_low = (470.0, 330.0, 200.0, 290.0, BARBER)
    # Long after: history is stale, so the head rule decides (barber).
    assert owners(own.assign("cam", [customer_face], [barber_low, CUSTOMER_BOX], now=5.0)) == [BARBER]


def test_history_is_per_camera():
    own = FaceOwnership()
    own.assign("cam-a", [face(600, 345)], [CUSTOMER_BOX], now=0.0)
    barber_low = (470.0, 330.0, 200.0, 290.0, BARBER)
    assert owners(own.assign("cam-b", [face(600, 345)], [barber_low, CUSTOMER_BOX], now=0.1)) == [BARBER]


def test_one_face_detected_twice_is_kept_once():
    # Found in both overlapping person boxes: two detections of the same face.
    barber_box = (470.0, 250.0, 200.0, 400.0, BARBER)
    twice = [face(598, 290, 42, 52, conf=0.6), face(600, 292, 40, 50, conf=0.8)]
    result = FaceOwnership().assign("cam", twice, [barber_box, CUSTOMER_BOX], now=0.0)
    assert len(result) == 1 and result[0][4] == 0.8


def test_two_faces_in_one_overlap_get_one_owner_each():
    own = FaceOwnership()
    barber_box = (480.0, 200.0, 260.0, 440.0, BARBER)
    her_face, his_face = face(560, 215), face(600, 315)
    own.assign("cam", [her_face], [barber_box], now=0.0)
    own.assign("cam", [his_face], [CUSTOMER_BOX], now=0.0)
    result = own.assign("cam", [her_face, his_face], [barber_box, CUSTOMER_BOX], now=0.04)
    assert owners(result) == [BARBER, CUSTOMER]


def test_crop_box_is_the_box_that_maps_the_landmarks_into_the_face():
    # Landmarks come in the face network's coords for the box it ran on;
    # only that box maps them back inside the face.
    f = face(600, 345)
    boxes = [(470.0, 330.0, 200.0, 290.0, BARBER), CUSTOMER_BOX]
    to_points = {BARBER: [[0.0, 0.0]] * 5, CUSTOMER: [[620.0, 370.0]] * 5}
    assert crop_box(f, boxes, lambda box: to_points[box[4]]) == CUSTOMER_BOX[:4]
    assert crop_box(face(50, 50), boxes, lambda box: to_points[box[4]]) is None


def test_the_fallback_is_counted_and_so_is_every_time_it_beats_the_smallest_box():
    # 2026-09-21: the highest-top fallback went live bundled with other changes
    # and was never measured on its own. These counters say how often it runs
    # and how often it picks a different box than the old smallest-box rule.
    own = FaceOwnership()
    barber_box = (470.0, 295.0, 200.0, 400.0, BARBER)   # higher top, bigger box
    own.assign("cam", [face(560, 280)], [barber_box, CUSTOMER_BOX], now=0.0)
    assert own.stats == {"contested": 1, "fallback": 1, "fallback_changed": 1}


def test_a_face_the_earlier_rules_settle_never_reaches_the_fallback():
    own = FaceOwnership()
    barber_low = (470.0, 330.0, 200.0, 290.0, BARBER)
    own.assign("cam", [face(600, 345)], [barber_low, CUSTOMER_BOX], now=0.0)   # head-at-top rule decides
    own.assign("cam", [face(50, 50)], [CUSTOMER_BOX], now=0.1)                 # in no box
    own.assign("cam", [face(560, 315)], [CUSTOMER_BOX], now=0.2)               # in one box
    assert own.stats == {"contested": 1, "fallback": 0, "fallback_changed": 0}


def test_the_counters_are_reported_and_reset_once_a_minute(caplog):
    own = FaceOwnership()
    barber_box = (470.0, 295.0, 200.0, 400.0, BARBER)
    with caplog.at_level("INFO", logger="iris.face_owner"):
        own.assign("cam", [face(560, 280)], [barber_box, CUSTOMER_BOX], now=0.0)
        own.assign("cam", [face(560, 280)], [barber_box, CUSTOMER_BOX], now=61.0)
    assert any("fallback" in r.message for r in caplog.records)
    assert own.stats == {"contested": 0, "fallback": 0, "fallback_changed": 0}


# ── The rule is per product (2026-09-22) ──────────────────────────────────────
# `tracked` re-decides every face by position; `legacy` is what attendance's
# face-ID thresholds were calibrated against. Each product declares its own in
# its sidecar yaml, so one product's change cannot move the other's faces.

from pathlib import Path

import pytest
import yaml

from perception.modules.face_detect.owner import assign_legacy
from perception.engine.modules import ModuleConfigError
from perception.modules.face_detect import Settings
from perception.engine.sources import load_sidecar_config

REPO = Path(__file__).resolve().parents[1]
# A tall standing body and the smaller seated one, overlapping and starting at
# nearly the same height. A face whose centre is inside both, and whose top is
# ABOVE both box tops (a clipped head), leaves rule 1 (continuity, no history)
# and rule 2 (head-at-top, no box top near the face top) undecided -- so only
# the fallback separates them.
STANDING_BOX = (500.0, 305.0, 260.0, 560.0, BARBER)     # area 145,600
SEATED_BOX = (540.0, 310.0, 160.0, 260.0, CUSTOMER)     # area  41,600 -- the smaller


def test_legacy_only_fills_in_a_face_the_detector_left_unowned():
    already_owned = (560.0, 315.0, 40.0, 50.0, 0.8, BARBER)   # detector said: barber
    out = assign_legacy([already_owned], [CUSTOMER_BOX])
    assert owners(out) == [BARBER]                            # untouched, though it sits in the customer's box


def test_legacy_picks_the_smallest_containing_box():
    out = assign_legacy([face(560, 315)], [STANDING_BOX, SEATED_BOX])
    assert owners(out) == [CUSTOMER]


def test_legacy_leaves_a_face_in_no_box_unowned():
    assert owners(assign_legacy([face(50, 50)], [CUSTOMER_BOX])) == [None]


def test_tracked_fallback_is_selectable():
    # Same contested frame, two products' fallbacks, two different owners.
    highest = FaceOwnership(fallback="highest-top")
    smallest = FaceOwnership(fallback="smallest")
    f = [face(560, 290)]       # centre (580, 315): inside both, top above both box tops
    assert owners(highest.assign("cam", f, [STANDING_BOX, SEATED_BOX], now=0.0)) == [BARBER]
    assert owners(smallest.assign("cam2", f, [STANDING_BOX, SEATED_BOX], now=0.0)) == [CUSTOMER]


def test_an_unknown_fallback_is_refused():
    with pytest.raises(ValueError):
        FaceOwnership(fallback="whatever")


def _module_settings(product):
    cfg = load_sidecar_config(str(REPO / "products" / product / "sidecar.yaml"))
    return cfg["modules"]["face_detect"]


def test_each_product_declares_its_own_rule():
    # Each product says it in its OWN file; the shared file lists no module at all.
    base = load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))
    assert not base.get("modules")
    assert yaml.safe_load((REPO / "products" / "attendance" / "sidecar.yaml").read_text())["modules"]["face_detect"][
        "owner_mode"] == "legacy"
    assert yaml.safe_load((REPO / "products" / "barber" / "sidecar.yaml").read_text())["modules"]["face_detect"][
        "owner_mode"] == "tracked"                           # barber: leaning over seated customers
    assert _module_settings("attendance")["owner_mode"] == "legacy"      # attendance: as its thresholds were tuned
    assert _module_settings("barber")["owner_fallback"] == "highest-top"


def test_settings_default_to_the_legacy_rule():
    s = Settings()
    assert (s.owner_mode, s.owner_fallback) == ("legacy", "highest-top")


@pytest.mark.parametrize("bad", [{"owner_mode": "trackd"}, {"owner_mode": "tracked", "owner_fallback": "biggest"}])
def test_a_typo_is_refused_not_defaulted(bad):
    # Silently falling back to the other rule would move faces between people
    # with nothing in the log.
    with pytest.raises(ModuleConfigError):
        Settings(**bad)


def test_each_products_file_parses_to_its_rule():
    for product, rule in (("attendance", ("legacy", "highest-top")), ("barber", ("tracked", "highest-top"))):
        s = Settings(**_module_settings(product))
        assert (s.owner_mode, s.owner_fallback) == rule
