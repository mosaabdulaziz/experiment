"""Unit tests for `concealment/features.py` (elbow angle, torso geometry, anatomical body-zone guard, excursion, wrist drop)."""
from __future__ import annotations

import math

from concealment.features import (
    classify_body_zone,
    extract_track_features,
    joint_angle_deg,
    torso_geometry,
)
from concealment.types import CameraZones, DetectorSettings, ZonePolygon
from conftest import make_skeleton_obs


def test_joint_angle_deg_straight_right_and_degenerate():
    assert math.isclose(joint_angle_deg((0.0, 0.0), (0.0, 10.0), (0.0, 20.0)), 180.0, abs_tol=1e-4)
    assert math.isclose(joint_angle_deg((0.0, 0.0), (0.0, 10.0), (10.0, 10.0)), 90.0, abs_tol=1e-4)
    # Degenerate zero-length segment returns 180.0 without ZeroDivisionError
    assert math.isclose(joint_angle_deg((5.0, 5.0), (5.0, 5.0), (10.0, 10.0)), 180.0, abs_tol=1e-4)


def test_torso_geometry_computes_center_length_and_shoulder_width():
    obs = make_skeleton_obs(center_x=960.0, shoulder_y=400.0, hip_y=540.0, shoulder_half_w=45.0)
    center, torso_len, sh_w = torso_geometry(obs)
    assert math.isclose(center[0], 960.0, abs_tol=1e-4)
    assert math.isclose(center[1], 470.0, abs_tol=1e-4)
    assert math.isclose(torso_len, 140.0, abs_tol=1e-4)
    assert math.isclose(sh_w, 90.0, abs_tol=1e-4)


def test_resting_straight_arm_next_to_hip_never_triggers_at_body():
    """Critical anatomical guard (CALIBRATION_DATASETS.md §1 & §4):
    A standing shopper whose straight arm (elbow_deg ~ 177 deg) hangs beside their hip
    must NOT satisfy elbow_flexed_or_medial or in_at_body_pose."""
    settings = DetectorSettings(max_conceal_elbow_deg=135.0)
    obs = make_skeleton_obs(center_x=960.0, shoulder_y=400.0, hip_y=540.0)
    cz = CameraZones(camera="cam-1", zone_free=True)
    feat = extract_track_features(obs, history=[], camera_zones=cz, shape=(1920, 1080), settings=settings)
    assert feat.min_elbow_deg > 160.0
    assert feat.elbow_flexed_or_medial is False
    assert feat.in_at_body_pose is False


def test_flexed_arm_at_waist_chest_under_arm_and_personal_bag():
    settings = DetectorSettings(max_conceal_elbow_deg=135.0)
    # 1. Flexed right elbow (elbow out at +70, +80; wrist tucked to front waist at +5, +135)
    obs_waist = make_skeleton_obs(
        center_x=960.0,
        shoulder_y=400.0,
        hip_y=540.0,
        right_elbow_offset=(75.0, 75.0),
        right_wrist_offset=(5.0, 135.0),
    )
    cz = CameraZones(
        camera="cam-1",
        zone_free=False,
        zones=(ZonePolygon(id="shelf-a", points=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),),
    )
    feat_waist = extract_track_features(obs_waist, history=[], camera_zones=cz, shape=(1920, 1080), settings=settings)
    assert feat_waist.in_shelf_zone is True
    assert feat_waist.zone_id == "shelf-a"
    assert feat_waist.right_elbow_deg < 120.0
    assert feat_waist.elbow_flexed_or_medial is True
    assert feat_waist.in_at_body_pose is True
    assert feat_waist.target_kind == "clothing"
    assert feat_waist.body_zone_name in ("waist", "chest", "under_arm")

    # 2. Personal bag concealment: bag worn near hip, wrist enters bag box
    obs_bag = make_skeleton_obs(
        center_x=960.0,
        shoulder_y=400.0,
        hip_y=540.0,
        right_elbow_offset=(80.0, 70.0),
        right_wrist_offset=(60.0, 130.0),
        has_bag=True,
        bag_box=(995.0, 500.0, 1055.0, 575.0),
    )
    zone_name, target_kind, _, flexed_ok = classify_body_zone(
        wrist_xy=(1020.0, 530.0),
        side="right",
        obs=obs_bag,
        torso_center=(960.0, 470.0),
        torso_length=140.0,
        elbow_deg=95.0,
        settings=settings,
    )
    assert zone_name == "personal_bag"
    assert target_kind == "personal_bag"
    assert flexed_ok is True


def test_reach_excursion_and_wrist_confidence_drop_against_baseline():
    settings = DetectorSettings(
        reach_excursion_torsos=0.35,
        reach_window_s=2.5,
        baseline_window_s=2.0,
        hidden_conf_drop_fraction=0.45,
        keypoint_conf_threshold=0.30,
    )
    cz = CameraZones(camera="cam-1", zone_free=True)
    # Build history: t=0.0 resting, t=0.5 reaching far out (+140 px = 1.0 torso), with high wrist conf 0.90
    obs_rest = make_skeleton_obs(right_wrist_offset=(52.0, 145.0), right_wrist_conf=0.90)
    obs_reach = make_skeleton_obs(
        right_elbow_offset=(95.0, -10.0),
        right_wrist_offset=(165.0, -35.0),
        right_wrist_conf=0.90,
    )
    history = [
        (0.0, obs_rest, 140.0),
        (0.4, obs_reach, 140.0),
        (0.8, obs_reach, 140.0),
    ]
    # Now at t=1.2, hand retracts to waist with flexed elbow and wrist conf drops to 0.22 (< 0.45 * 0.90)
    obs_conceal = make_skeleton_obs(
        right_elbow_offset=(75.0, 75.0),
        right_wrist_offset=(5.0, 135.0),
        right_wrist_conf=0.22,
    )
    feat = extract_track_features(
        obs_conceal,
        history=history,
        camera_zones=cz,
        shape=(1920, 1080),
        settings=settings,
    )
    # The reach is history evidence and stays credited; the dropped wrist is evidence of
    # loss, not of where it entered, so AT_BODY needs the geometry it no longer has.
    assert feat.reach_detected is True
    assert feat.wrist_excursion_torsos >= 0.35
    assert feat.in_at_body_pose is False
    assert feat.wrist_geometry_valid is False
    assert feat.active_hand == "right"
    assert math.isclose(feat.wrist_conf_baseline, 0.90, abs_tol=1e-3)
    assert feat.wrist_hidden_now is True


def test_stationary_hand_near_waist_without_prior_inward_motion_is_rejected():
    """With multi-frame history, a hand resting stationary near the waist must NOT satisfy
    `in_at_body_pose` unless it moved inward toward the body by >= inward_min_torsos."""
    settings = DetectorSettings(inward_min_torsos=0.12)
    cz = CameraZones(camera="cam-1", zone_free=True)
    obs_stationary = make_skeleton_obs(
        right_elbow_offset=(75.0, 75.0),
        right_wrist_offset=(5.0, 135.0),
    )
    history = [
        (0.0, obs_stationary, 140.0),
        (0.2, obs_stationary, 140.0),
        (0.4, obs_stationary, 140.0),
    ]
    feat = extract_track_features(
        obs_stationary,
        history=history,
        camera_zones=cz,
        shape=(1920, 1080),
        settings=settings,
    )
    assert feat.in_at_body_pose is False
