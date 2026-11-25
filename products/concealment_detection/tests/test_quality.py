"""Unit tests for `concealment/quality.py` (quality metrics, hysteresis gate, camera gate, ReentryGrace)."""
from __future__ import annotations

import math

from concealment.quality import QualityGate, ReentryGrace, compute_track_quality_metrics
from concealment.types import DetectorSettings
from conftest import make_frame, make_skeleton_obs


def test_compute_track_quality_metrics():
    settings = DetectorSettings()
    obs = make_skeleton_obs(shoulder_y=400.0, hip_y=540.0, ankle_y=780.0, conf=0.85)
    core_conf, box_h, torso_px, n_joints = compute_track_quality_metrics(obs, settings)
    assert math.isclose(core_conf, 0.85, abs_tol=1e-4)
    assert math.isclose(box_h, (780.0 + 15.0) - (400.0 - 75.0), abs_tol=1e-4)
    assert math.isclose(torso_px, 140.0, abs_tol=1e-4)
    assert n_joints == 17


def test_reentry_grace_bridges_short_occlusion_id_switch():
    settings = DetectorSettings(reid_max_gap_s=1.5, reid_max_dist_torsos=1.2)
    grace = ReentryGrace(settings)
    obs1 = make_skeleton_obs(track_id=10, center_x=600.0, shoulder_y=400.0, hip_y=540.0)
    cid1 = grace.resolve_id(10, obs1, pts=1.0, active_raw_ids={10})
    assert cid1 == 10
    grace.observe_active(cid1, obs1, pts=1.0, torso_center=(600.0, 470.0), torso_length=140.0)
    grace.mark_lost(cid1, pts=1.1)

    # 0.6s later, new tracker ID 25 appears 40 px away (< 1.2 * 140 = 168 px), while 10 is gone
    obs2 = make_skeleton_obs(track_id=25, center_x=640.0, shoulder_y=400.0, hip_y=540.0)
    cid2 = grace.resolve_id(25, obs2, pts=1.7, active_raw_ids={25})
    assert cid2 == 10

    # A new tracker ID 99 appearing far away (center_x=1500) must NOT inherit ID 10
    obs_far = make_skeleton_obs(track_id=99, center_x=1500.0, shoulder_y=400.0, hip_y=540.0)
    cid_far = grace.resolve_id(99, obs_far, pts=1.8, active_raw_ids={25, 99})
    assert cid_far == 99


def test_quality_gate_camera_level_states():
    settings = DetectorSettings(camera_stale_s=2.0)
    gate = QualityGate(settings)
    f_ok = make_frame(pts=1.0, tracks=[make_skeleton_obs()], observed=True)
    state, reason = gate.evaluate_camera(f_ok)
    assert state == "LIVE"
    assert reason == ""

    # Unobserved frame within camera_stale_s stays LIVE
    f_skip = make_frame(pts=2.0, tracks=[], observed=False)
    state2, _ = gate.evaluate_camera(f_skip)
    assert state2 == "LIVE"

    # Unobserved for > camera_stale_s -> NO_POSE
    f_stale = make_frame(pts=3.5, tracks=[], observed=False)
    state3, reason3 = gate.evaluate_camera(f_stale)
    assert state3 == "NO_POSE"
    assert reason3 != ""

    # Invalid capture shape -> STALE
    f_bad_shape = make_frame(pts=4.0, tracks=[], shape=(0, 0), observed=True)
    state4, _ = gate.evaluate_camera(f_bad_shape)
    assert state4 == "STALE"


def test_quality_gate_track_hysteresis_and_insufficient_reasons():
    settings = DetectorSettings(
        min_track_s=0.4,
        min_keypoints=6,
        min_box_height_px=48.0,
        min_torso_px=25.0,
        min_core_conf_enter=0.30,
        min_core_conf_exit=0.35,
        degrade_max_s=1.0,
        recover_s=0.4,
        lost_s=1.5,
    )
    gate = QualityGate(settings)
    good = make_skeleton_obs(track_id=1, conf=0.85)

    # First frame at t=0.0: UNKNOWN (waiting for min_track_s=0.4)
    q0 = gate.evaluate_track(1, good, pts=0.0)
    assert q0.state == "UNKNOWN"

    # At t=0.5 (>= 0.4s): transitions to OK
    q1 = gate.evaluate_track(1, good, pts=0.5)
    assert q1.state == "OK"

    # Core conf dips to 0.20 at t=0.6 -> DEGRADED
    low_conf_obs = make_skeleton_obs(track_id=1, conf=0.20)
    q2 = gate.evaluate_track(1, low_conf_obs, pts=0.6)
    assert q2.state == "DEGRADED"

    # Stays low beyond degrade_max_s (1.0s) at t=1.7 -> INSUFFICIENT
    q3 = gate.evaluate_track(1, low_conf_obs, pts=1.7)
    assert q3.state == "INSUFFICIENT"
    assert q3.reason != ""

    # Recovery: good observations for >= recover_s (0.4s) from t=2.0 to t=2.5 -> OK
    gate.evaluate_track(1, good, pts=2.0)
    q_rec = gate.evaluate_track(1, good, pts=2.5)
    assert q_rec.state == "OK"

    # Skipped pose frame (frame_observed=False) does not drop or penalize track 1
    missing_on_skip = gate.step_missing_tracks(seen_track_ids=set(), pts=2.6, frame_observed=False)
    assert missing_on_skip == []

    # Track 1 missing on observed frame after lost_s (t=4.2 >= 2.5 + 1.5) -> GONE
    missing_gone = dict(gate.step_missing_tracks(seen_track_ids=set(), pts=4.2, frame_observed=True))
    assert missing_gone[1].state == "GONE"

    # Immediate INSUFFICIENT when a brand-new track has tiny bbox (< 48 px)
    tiny_obs = make_skeleton_obs(track_id=2, box=(100.0, 100.0, 120.0, 130.0))
    q_tiny = gate.evaluate_track(2, tiny_obs, pts=5.0)
    assert q_tiny.state == "INSUFFICIENT"
    assert "box" in q_tiny.reason or "height" in q_tiny.reason or "small" in q_tiny.reason
