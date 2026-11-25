"""FSM_SCENARIOS.md traces under the deployed `concealment_detection.yaml` detector profile."""
from __future__ import annotations

from config import CONFIG_PATH, load_config
from concealment.state_machine import ConcealmentMonitor
from concealment.types import CameraZones, ZonePolygon
from conftest import make_frame, make_skeleton_obs

REACH = dict(right_elbow_offset=(80.0, -15.0), right_wrist_offset=(140.0, -60.0))
POCKET = dict(right_elbow_offset=(75.0, 75.0), right_wrist_offset=(5.0, 135.0))


def _monitor() -> ConcealmentMonitor:
    zones = CameraZones(camera="cam-1", zone_free=False, zones=(ZonePolygon(
        id="shelf-a", points=((0.10, 0.20), (0.90, 0.20), (0.90, 0.85), (0.10, 0.85)),
        facing=(0.0, -1.0)),))
    return ConcealmentMonitor("cam-1", load_config(CONFIG_PATH, env={}).detector, zones)


def _play(mon: ConcealmentMonitor, segments: list[tuple[int, dict | None]]) -> list:
    """Each segment is (frame count, pose overrides or None for no person) at 10 fps."""
    fired, i = [], 0
    for count, pose in segments:
        for _ in range(count):
            tracks = [] if pose is None else [make_skeleton_obs(track_id=7, **pose)]
            fired.extend(mon.update(make_frame(round(i * 0.1, 2), tracks, frame_num=i)).candidates)
            i += 1
    return fired


def test_deployed_profile_is_the_measured_one():
    s = load_config(CONFIG_PATH, env={}).detector
    assert (s.dwell_min_s, s.short_track_grace_s, s.min_hold_s, s.hidden_s) == (2.0, 1.5, 0.35, 0.40)
    assert s.emit_low_tier is False
    # Every FSM-upgrade knob stays at its legacy default: none passed the Phase 3 adoption gate.
    assert (s.sequence_max_gap_s, s.insufficient_grace_s, s.zone_exit_grace_s) == (0.5, 0.0, 0.0)
    assert (s.reach_expiry_s, s.partial_min_near_body_s, s.hold_flicker_grace_s) == (0.0, 0.0, 0.0)
    assert (s.fast_dropin_min_hold_s, s.spatial_hidden_s, s.exit_min_hidden_frac) == (0.0, 0.0, 0.0)
    assert not (s.zone_persist_across_shelves or s.reach_seeds_dwell or s.hand_transfer)
    assert s.at_body_requires_flex is True and s.cooldown_scope == "track"


def test_full_concealment_after_full_dwell_emits_one_high_returned():
    fired = _play(_monitor(), [(23, {}), (4, REACH), (6, POCKET), (1, {})])
    assert [(c.tier, c.terminal_stage, c.target_kind) for c in fired] == [("high", "RETURNED", "clothing")]


def test_short_track_grace_concealment_emits_medium():
    fired = _play(_monitor(), [(3, {}), (3, REACH), (5, POCKET), (1, {})])
    assert [(c.tier, c.terminal_stage) for c in fired] == [("medium", "RETURNED")]


def test_brief_brush_does_not_alert():
    assert _play(_monitor(), [(23, {}), (4, REACH), (2, POCKET), (10, {})]) == []


def test_standing_with_arms_at_sides_does_not_alert():
    assert _play(_monitor(), [(60, {})]) == []


def test_track_lost_while_hand_at_body_emits_nothing():
    assert _play(_monitor(), [(23, {}), (4, REACH), (6, POCKET), (40, None)]) == []


def test_personal_bag_confidence_drop_emits_high_hidden():
    bag = dict(has_bag=True, bag_box=(990.0, 495.0, 1060.0, 580.0))
    at_bag = dict(right_elbow_offset=(80.0, 70.0), right_wrist_offset=(60.0, 130.0))
    fired = _play(_monitor(), [(23, bag), (4, {**REACH, **bag}),
                              (1, {**at_bag, **bag}),
                              (6, {**at_bag, **bag, "right_wrist_conf": 0.15})])
    assert [(c.tier, c.terminal_stage, c.target_kind) for c in fired] == [
        ("high", "HIDDEN", "personal_bag")]


def test_repeat_within_cooldown_is_suppressed():
    sequence = [(23, {}), (4, REACH), (6, POCKET), (1, {})]
    fired = _play(_monitor(), sequence + sequence)
    assert [(c.tier, c.terminal_stage) for c in fired] == [("high", "RETURNED")]


def test_insufficient_footage_logs_assessment_without_candidate():
    obs = make_skeleton_obs(track_id=7, box=(100.0, 100.0, 120.0, 140.0), **POCKET)
    result = _monitor().update(make_frame(1.0, [obs], frame_num=10))
    assert result.candidates == ()
    assert [(a.track_id, a.state) for a in result.assessments] == [(7, "insufficient_evidence")]
