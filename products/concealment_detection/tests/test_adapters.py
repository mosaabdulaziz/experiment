"""Unit tests for `adapters.py` (RetailS/PoseLift coordinate swap, mask intervals, event evaluation, FA/h bound)."""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
import pytest

from concealment.types import StageTransition

from adapters import (
    evaluate_clip_predictions,
    extract_mask_events,
    false_alarm_upper_bound_95,
    load_retails_pose_json,
)
from concealment.types import NOSE, L_ANKLE
from test_evidence_and_clock import _sample_candidate


def test_load_retails_pose_json_swaps_yx_and_derives_bbox():
    """CALIBRATION_DATASETS.md §1 & §3: RetailS stores each joint as [y, x, conf].
    After loading with swap_yx=True, nose_y < ankle_y and x is in [0, 1920]."""
    # Build 17 joints where col0 (y) increases from 320 (nose) to 580 (ankle) and col1 (x) is 1400
    raw_kps = []
    for j in range(17):
        y_val = 320.0 + j * 16.0
        x_val = 1400.0 + (j % 3) * 10.0
        raw_kps.extend([y_val, x_val, 0.85])

    raw_json = {
        "1": {
            "0": {"keypoints": raw_kps, "scores": None},
            "2": {"keypoints": raw_kps, "scores": None},
        }
    }
    frames = load_retails_pose_json(raw_json, camera="cam-1", fps=15.0, shape=(1920, 1080), swap_yx=True)
    assert len(frames) == 3  # frames 0, 1 (empty), 2
    f0 = frames[0]
    assert len(f0.tracks) == 1
    obs = f0.tracks[0]
    assert obs.track_id == 1
    nose_x, nose_y, _ = obs.kps[NOSE]
    ank_x, ank_y, _ = obs.kps[L_ANKLE]
    assert math.isclose(nose_x, 1400.0, abs_tol=1e-3)
    assert math.isclose(nose_y, 320.0, abs_tol=1e-3)
    assert ank_y > nose_y
    assert obs.box[2] > obs.box[0]
    assert obs.box[3] > obs.box[1]


def test_extract_mask_events_and_evaluate_clip_predictions():
    mask = np.zeros(60, dtype=int)
    mask[15:45] = 1  # 1.0s to 3.0s at 15 fps
    intervals = extract_mask_events(mask, fps=15.0)
    assert len(intervals) == 1
    assert math.isclose(intervals[0][0], 1.0, abs_tol=1e-3)
    assert math.isclose(intervals[0][1], 3.0, abs_tol=1e-3)

    # Candidate at pts=3.4s (within tolerance_s=2.0 of [1.0, 3.0]) -> TP=1, FP=0
    cand_tp = _sample_candidate("high", "clothing")
    cand_tp = type(cand_tp)(**{**cand_tp.__dict__, "pts": 3.4, "stage_timeline": ()})
    metrics = evaluate_clip_predictions([cand_tp], intervals, duration_s=4.0, tolerance_s=2.0)
    assert metrics["tp_events"] == 1
    assert metrics["fn_events"] == 0
    assert metrics["fp_candidates"] == 0
    assert math.isclose(metrics["recall"], 1.0, abs_tol=1e-5)

    # Candidate on normal clip (no gt intervals) -> FP=1
    metrics_norm = evaluate_clip_predictions([cand_tp], [], duration_s=3600.0, tolerance_s=2.0)
    assert metrics_norm["fp_candidates"] == 1
    assert math.isclose(metrics_norm["false_alarms_per_hour"], 1.0, abs_tol=1e-5)


def test_pose_timestamps_keep_absolute_clip_origin():
    frames = load_retails_pose_json({"1": {"30": [100, 200, .9] * 17}}, fps=15)
    assert frames[0].frame_num == 0
    assert frames[0].tracks == ()
    assert frames[-1].pts == 2.0


def test_duplicate_alerts_are_false_positives():
    cand = replace(_sample_candidate(), pts=2.0, stage_timeline=())
    metrics = evaluate_clip_predictions([cand, cand], [(1, 3)], 4, tolerance_s=0)
    assert (metrics["tp_events"], metrics["fp_candidates"]) == (1, 1)


@pytest.mark.parametrize("reverse", [False, True])
def test_matching_is_maximum_not_first_hit(reverse):
    # First alert fits either event, second only the first: reassign the first.
    candidates = [replace(_sample_candidate(), pts=t, stage_timeline=()) for t in (2.5, 1)]
    if reverse:
        candidates.reverse()
    metrics = evaluate_clip_predictions(candidates, [(0, 2), (3, 5)], 6, tolerance_s=1)
    assert (metrics["tp_events"], metrics["fp_candidates"]) == (2, 0)


@pytest.mark.parametrize("reverse", [False, True])
def test_matched_flags_follow_input_order(reverse):
    times = [2.5, 1.0, 9.0]
    if reverse:
        times.reverse()
    candidates = [replace(_sample_candidate(), pts=t, stage_timeline=()) for t in times]
    metrics = evaluate_clip_predictions(candidates, [(0, 2), (3, 5)], 10, tolerance_s=1)
    expected_gt = {2.5: 1, 1.0: 0, 9.0: None}
    assert metrics["matched_gt"] == [expected_gt[t] for t in times]
    assert metrics["matched"] == [expected_gt[t] is not None for t in times]


def test_actual_sequence_overlap_matches_but_stale_dwell_does_not():
    cand = replace(_sample_candidate(), pts=12, stage_timeline=(
        StageTransition("DWELL", 0), StageTransition("REACH", 8),
        StageTransition("AT_BODY", 9), StageTransition("RETURNED", 12),
    ))
    assert evaluate_clip_predictions([cand], [(8, 9)], 15, 0)["tp_events"] == 1
    assert evaluate_clip_predictions([cand], [(1, 2)], 15, 0)["fp_candidates"] == 1
    # A reset breaks the old sequence; its reach cannot expand the current alert.
    reset = replace(cand, stage_timeline=(StageTransition("REACH", 1),
                    StageTransition("IDLE", 2), StageTransition("AT_BODY", 10)))
    assert evaluate_clip_predictions([reset], [(1, 2)], 15, 0)["fp_candidates"] == 1


def test_false_alarm_upper_bound_95_rule_of_three():
    # With 0 false alarms in 100 camera-hours, 95% upper bound is ~3.0 / 100 = 0.03 / h
    ub0 = false_alarm_upper_bound_95(0, 100.0)
    assert math.isclose(ub0, -math.log(0.05) / 100.0, rel_tol=2e-2)
    assert ub0 > 0.0
    ub2 = false_alarm_upper_bound_95(2, 100.0)
    assert ub2 > ub0
