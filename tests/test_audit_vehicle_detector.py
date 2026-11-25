"""Unit tests for the plate-anchored vehicle-detector audit helpers.

These cover the pure scoring/aggregation logic only — no model inference, so the
suite stays fast and GPU-free.
"""

import numpy as np
import pytest

pytest.importorskip("ultralytics", reason="the CI test image leaves out ultralytics (the test group, pyproject.toml)")

from mtmc.audit_vehicle_detector import (
    _iou_one,
    anchor_plate,
    sample_frame_indices,
    size_bucket,
    summarize,
)
from mtmc.yolo_detector import _plate_to_vehicle_box


def test_size_bucket_boundaries():
    assert size_bucket(10) == "w<32px"
    assert size_bucket(31.9) == "w<32px"
    assert size_bucket(32) == "w32-64"
    assert size_bucket(64) == "w64-128"
    assert size_bucket(128) == "w>=128"
    assert size_bucket(5000) == "w>=128"


def test_expansion_matches_documented_ratios():
    """Sanity-check the expansion the anchor relies on (2.5x width, 3.5x height)."""
    plate = np.array([100.0, 200.0, 120.0, 210.0])  # 20x10
    box = _plate_to_vehicle_box(plate, 1000, 1000)
    assert box[0] == 85.0 and box[2] == 135.0        # width 20 -> 50, centred at 110
    assert box[3] == 207.0                            # slight downward from plate centre
    assert round(box[3] - box[1], 3) == 35.0          # height 10 -> 35


def test_anchor_hit_when_vehicle_box_matches_expansion():
    plate = np.array([100.0, 200.0, 120.0, 210.0])
    vehicles = np.array([[85.0, 172.0, 135.0, 207.0]])   # 50 px wide -> ratio 0.4
    rec = anchor_plate(plate, vehicles, np.array([0.9]), 1000, 1000)
    assert rec["hit"] is True
    assert rec["contained"] is True
    assert rec["iou_hit"] is True
    assert rec["best_iou"] == 1.0
    assert rec["plate_w_ratio"] == 0.4
    assert rec["bucket"] == "w<32px"
    assert rec["best_vehicle_conf"] == 0.9


def test_anchor_miss_when_no_vehicle_nearby():
    plate = np.array([100.0, 200.0, 120.0, 210.0])
    vehicles = np.array([[300.0, 300.0, 400.0, 400.0]])
    rec = anchor_plate(plate, vehicles, np.array([0.8]), 1000, 1000)
    assert rec["hit"] is False
    assert rec["contained"] is False
    assert rec["best_iou"] == 0.0


def test_scale_guard_rejects_a_box_that_swallows_the_plate():
    """Containment alone would accept a frame-sized box; the width ratio stops it."""
    plate = np.array([100.0, 200.0, 120.0, 210.0])       # 20 px wide
    vehicles = np.array([[0.0, 0.0, 1000.0, 1000.0]])    # contains the centre, absurd size
    rec = anchor_plate(plate, vehicles, np.array([0.7]), 1000, 1000)
    assert rec["contained"] is True
    assert rec["plate_w_ratio"] == 0.02
    assert rec["hit"] is False


def test_scale_guard_rejects_a_box_tighter_than_a_plate():
    """A 20px plate cannot sit on a 30px-wide 'vehicle'."""
    plate = np.array([100.0, 200.0, 120.0, 210.0])
    vehicles = np.array([[95.0, 195.0, 125.0, 215.0]])   # 30 px wide
    rec = anchor_plate(plate, vehicles, np.array([0.7]), 1000, 1000)
    assert rec["contained"] is True
    assert rec["plate_w_ratio"] == 0.6667
    assert rec["hit"] is False


def test_plausible_car_box_counts_as_found():
    """A plate on a correctly scaled vehicle box is a detection."""
    plate = np.array([100.0, 200.0, 120.0, 210.0])       # 20 px wide
    vehicles = np.array([[80.0, 150.0, 140.0, 240.0]])   # 60 px wide -> ratio 0.33
    rec = anchor_plate(plate, vehicles, np.array([0.7]), 1000, 1000)
    assert rec["contained"] is True
    assert rec["hit"] is True


def test_iou_hit_is_recorded_separately_from_containment():
    """The expansion-based IoU is a diagnostic and must not drive the verdict."""
    plate = np.array([100.0, 200.0, 120.0, 210.0])
    # A plausible car box drawn around the plate: much larger than the 2.5x/3.5x
    # estimate, which is exactly why IoU against the estimate is misleading.
    vehicles = np.array([[70.0, 120.0, 150.0, 240.0]])
    rec = anchor_plate(plate, vehicles, np.array([0.9]), 1000, 1000)
    assert rec["contained"] is True
    assert rec["hit"] is True          # containment decides
    assert rec["best_iou"] < 0.30      # expansion under-estimates the box
    assert rec["iou_hit"] is False


def test_anchor_with_no_vehicle_boxes():
    plate = np.array([100.0, 200.0, 120.0, 210.0])
    rec = anchor_plate(plate, np.zeros((0, 4)), np.zeros(0), 1000, 1000)
    assert rec["hit"] is False
    assert rec["contained"] is False
    assert rec["plate_w_ratio"] is None
    assert rec["best_vehicle_conf"] == 0.0


def test_iou_one():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    ious = _iou_one(a, others)
    assert ious[0] == 1.0
    assert ious[1] == 0.0
    assert round(float(ious[2]), 4) == round(50 / 150, 4)
    assert len(_iou_one(a, np.zeros((0, 4)))) == 0


def test_summarize_aggregates_and_buckets():
    records = [
        {"bucket": "w<32px", "hit": True, "contained": True, "iou_hit": False, "best_iou": 0.8},
        {"bucket": "w<32px", "hit": False, "contained": True, "iou_hit": False, "best_iou": 0.1},
        {"bucket": "w32-64", "hit": True, "contained": True, "iou_hit": True, "best_iou": 0.5},
    ]
    out = summarize(records)
    assert out["plates"] == 3
    assert out["hits"] == 2
    assert out["recall"] == 0.6667
    assert out["containment"] == 1.0        # all three are contained by some box
    assert out["iou_recall"] == 0.3333
    assert out["mean_best_iou"] == 0.4667
    assert out["by_bucket"]["w<32px"] == {
        "plates": 2, "hits": 1, "recall": 0.5, "containment": 1.0,
    }
    assert out["by_bucket"]["w32-64"]["recall"] == 1.0
    assert "w64-128" not in out["by_bucket"]   # no plates in that bucket


def test_summarize_empty():
    out = summarize([])
    assert out["plates"] == 0
    assert out["recall"] == 0.0
    assert out["containment"] == 0.0
    assert out["iou_recall"] == 0.0
    assert out["mean_best_iou"] == 0.0
    assert out["by_bucket"] == {}


def test_sample_frame_indices():
    assert sample_frame_indices(200, 20.0, 2.0, 0) == [0, 40, 80, 120, 160]
    assert sample_frame_indices(200, 20.0, 2.0, 3) == [0, 40, 80]
    assert sample_frame_indices(200, 20.0, 0.5, 0) == list(range(0, 200, 10))
    assert sample_frame_indices(200, 20.0, 0.5, 4) == [0, 10, 20, 30]
    assert sample_frame_indices(0, 20.0, 2.0, 0) == []
    assert sample_frame_indices(200, 0.0, 2.0, 0) == []
    assert sample_frame_indices(200, 20.0, 0.0, 0) == []
