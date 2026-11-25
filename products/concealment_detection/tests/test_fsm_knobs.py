"""Opt-in FSM knobs: each default reproduces legacy behaviour; these pin the knob-on paths."""
from dataclasses import replace

import pytest

from concealment.features import extract_track_features, torso_geometry
from concealment.state_machine import ConcealmentMonitor, TrackStateMachine
from concealment.types import (
    TERMINAL_STAGES,
    CameraZones,
    DetectorSettings,
    QualityStatus,
    validate_detector_settings,
)
from conftest import make_frame, make_skeleton_obs


SETTINGS = DetectorSettings(min_track_s=0.0, dwell_min_s=0.2, min_hold_s=0.2,
                            hidden_s=0.3, short_track_grace_s=0.0, reach_window_s=0.4)
OK = QualityStatus(state="OK")
BAD = QualityStatus(state="INSUFFICIENT")
ZONES = CameraZones(camera="cam-1", zone_free=True)
R = "right"


def features(settings=SETTINGS, **changes):
    base = extract_track_features(make_skeleton_obs(), [], ZONES, (1920, 1080), settings)
    return replace(base, **changes)


def body(**changes):
    return features(active_hand=R, in_at_body_pose=True, **changes)


def out(**changes):
    changes.setdefault("at_body_geometry", False)
    return features(active_hand=R, in_at_body_pose=False, **changes)


def hidden(**changes):
    return features(active_hand=R, active_wrist_conf=0.0, wrist_hidden_now=True, **changes)


def reaching(settings, zone=None):
    z = {} if zone is None else {"zone_id": zone, "in_shelf_zone": True}
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(**z), OK, settings)
    sm.step(0.1, features(**z), OK, settings)
    sm.step(0.2, features(reach_detected=True, active_hand=R, **z), OK, settings)
    assert sm.stage == "REACH"
    return sm


def at_body(settings, zone=None):
    sm = reaching(settings, zone)
    z = {} if zone is None else {"zone_id": zone, "in_shelf_zone": True}
    sm.step(0.3, body(**z), OK, settings)
    assert sm.stage == "AT_BODY"
    return sm


def knob(**changes):
    return replace(SETTINGS, **changes)


# --- shared infrastructure ---------------------------------------------------------

def test_terminal_stages_cover_every_emission():
    assert set(TERMINAL_STAGES) == {"RETURNED", "HIDDEN", "PARTIAL", "RETURNED_FAST",
                                    "HIDDEN_SPATIAL", "EXIT"}


def test_defaults_are_valid():
    validate_detector_settings(DetectorSettings())
    validate_detector_settings(SETTINGS)


@pytest.mark.parametrize("changes", [
    {"sequence_max_gap_s": 0.3},
    {"sequence_max_gap_s": 1.5},
    {"sequence_max_gap_s": 1.4, "reid_max_gap_s": 1.0},
    {"hold_flicker_grace_s": 0.2},
    {"fast_dropin_min_hold_s": 0.2},
    {"spatial_hidden_s": 0.35},
    {"exit_min_hidden_frac": 1.5},
    {"hand_transfer_meet_torsos": 0.0},
    {"cooldown_scope": "zone"},
])
def test_invalid_knob_combinations_are_rejected(changes):
    with pytest.raises(ValueError):
        validate_detector_settings(knob(**changes))


def test_monitor_rejects_invalid_settings():
    with pytest.raises(ValueError):
        ConcealmentMonitor("cam-1", knob(cooldown_scope="zone"), ZONES)


# --- 2A continuity -----------------------------------------------------------------

def test_sequence_gap_bridges_without_crediting_unseen_time():
    s = knob(sequence_max_gap_s=1.0)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(1.2, body(), OK, s)
    assert sm.stage == "AT_BODY" and sm.hold_accum_s == pytest.approx(0.1)
    sm.step(1.3, body(), OK, s)
    cand = sm.step(1.4, out(), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"


def test_sequence_gap_default_still_resets():
    sm = at_body(SETTINGS)
    sm.step(1.2, body(), OK, SETTINGS)
    assert sm.stage != "AT_BODY"


def test_sequence_gap_never_bridges_dwell():
    s = knob(sequence_max_gap_s=1.0, dwell_min_s=5.0)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, s)
    sm.step(0.4, features(), OK, s)
    sm.step(1.2, features(), OK, s)
    assert sm.dwell_accum_s == 0.0


def test_monitor_keeps_sequence_through_short_disappearance():
    s = knob(sequence_max_gap_s=1.0)
    mon = ConcealmentMonitor("cam-1", s, ZONES)
    for i in range(3):
        mon.update(make_frame(i / 10, [make_skeleton_obs()]))
    mon.update(make_frame(0.3, [make_skeleton_obs(right_elbow_offset=(85.0, -10.0),
                                                  right_wrist_offset=(145.0, -55.0))]))
    mon.update(make_frame(0.4, [make_skeleton_obs(right_elbow_offset=(75.0, 75.0),
                                                  right_wrist_offset=(5.0, 135.0))]))
    assert mon._machines[1].stage == "AT_BODY"
    for i in range(5, 12):
        mon.update(make_frame(i / 10, []))
    assert mon._machines[1].stage == "AT_BODY"
    for i in range(12, 16):
        mon.update(make_frame(i / 10, []))
    assert mon._machines[1].stage != "AT_BODY"


def test_insufficient_grace_freezes_then_resumes():
    s = knob(insufficient_grace_s=0.3)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    assert sm.step(0.5, body(), BAD, s) is None
    assert sm.stage == "AT_BODY"
    sm.step(0.6, body(), OK, s)
    assert sm.hold_accum_s == pytest.approx(0.1)
    sm.step(0.7, body(), OK, s)
    cand = sm.step(0.8, out(), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"


def test_insufficient_grace_expires():
    s = knob(insufficient_grace_s=0.3)
    sm = at_body(s)
    for t in (0.4, 0.5, 0.6, 0.7, 0.9):
        sm.step(t, body(), BAD, s)
    assert sm.stage == "IDLE"


def test_insufficient_grace_does_not_bypass_sequence_gap_limit():
    s = knob(insufficient_grace_s=0.8)
    sm = at_body(s)
    sm.step(0.4, body(), BAD, s)
    assert sm.step(1.0, body(), OK, s) is None
    assert sm.stage == "IDLE"


@pytest.mark.parametrize("recovered_pts, expected_stage", [(0.8, "AT_BODY"), (0.81, "IDLE")])
def test_insufficient_grace_checks_deadline_on_recovery(recovered_pts, expected_stage):
    s = knob(insufficient_grace_s=0.3)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, body(), BAD, s)
    sm.step(0.6, body(), BAD, s)
    assert sm.step(recovered_pts, body(), OK, s) is None
    assert sm.stage == expected_stage
    if expected_stage == "IDLE":
        assert sm.hold_accum_s == 0.0
        assert sm.step(recovered_pts + 0.1, out(), OK, s) is None


def test_insufficient_grace_pauses_reach_expiry():
    s = knob(insufficient_grace_s=0.5, reach_expiry_s=0.4)
    sm = reaching(s)
    sm.step(0.3, out(), BAD, s)
    sm.step(0.4, out(), BAD, s)
    assert sm.step(0.7, body(), OK, s) is None
    assert sm.stage == "AT_BODY"
    assert sm.reach_pts == pytest.approx(0.6)


def test_insufficient_grace_pauses_at_body_timestamps():
    s = knob(insufficient_grace_s=0.3, hold_flicker_grace_s=0.15)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, body(), BAD, s)
    sm.step(0.6, body(), BAD, s)
    assert sm.step(0.7, out(), OK, s) is None
    assert sm.stage == "AT_BODY"
    assert sm.at_body_entered_pts == pytest.approx(0.5)
    assert sm.last_in_body_pts == pytest.approx(0.6)


def test_insufficient_grace_pauses_zone_exit_grace():
    s = knob(insufficient_grace_s=0.3, zone_exit_grace_s=0.2)
    sm = at_body(s, zone="A")
    sm.step(0.4, body(in_shelf_zone=False), OK, s)
    sm.step(0.5, body(in_shelf_zone=False), BAD, s)
    sm.step(0.6, body(in_shelf_zone=False), BAD, s)
    assert sm.step(0.7, body(in_shelf_zone=False), OK, s) is None
    assert sm.stage == "AT_BODY"
    assert sm._zone_exit_since_pts == pytest.approx(0.6)


def test_insufficient_grace_preserves_hidden_evidence():
    s = knob(insufficient_grace_s=0.3)
    sm = at_body(s)
    sm.step(0.4, hidden(), OK, s)
    sm.step(0.5, hidden(), OK, s)
    assert sm.hidden_accum_s == pytest.approx(0.1)
    sm.step(0.6, hidden(), BAD, s)
    sm.step(0.7, hidden(), BAD, s)
    assert sm.step(0.8, hidden(), OK, s) is None
    assert sm.hidden_accum_s == pytest.approx(0.1)


def test_insufficient_grace_preserves_spatial_evidence():
    s = knob(insufficient_grace_s=0.3, spatial_hidden_s=0.5)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, body(), OK, s)
    assert sm.body_run_s == pytest.approx(0.2)
    sm.step(0.6, body(), BAD, s)
    sm.step(0.7, body(), BAD, s)
    assert sm.step(0.8, body(), OK, s) is None
    assert sm.body_run_s == pytest.approx(0.2)


def test_insufficient_grace_never_applies_before_reach():
    s = knob(insufficient_grace_s=0.3)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, s)
    sm.step(0.1, features(), OK, s)
    sm.step(0.2, features(), BAD, s)
    assert sm.dwell_accum_s == 0.0


def test_terminal_reset_clears_observation_gap_clock():
    resets = []
    sm = TrackStateMachine(
        1,
        "cam-1",
        0.0,
        on_reset=lambda pts, track_id, reason, stage: resets.append((reason, stage)),
    )
    sm.step(0.0, features(), OK, SETTINGS)
    sm.step(0.1, features(), BAD, SETTINGS)
    sm.step(1.0, features(), OK, SETTINGS)
    assert resets == [("quality", "IDLE")]


def test_zone_reset_restarts_observation_gap_clock():
    resets = []
    sm = TrackStateMachine(
        1,
        "cam-1",
        0.0,
        on_reset=lambda pts, track_id, reason, stage: resets.append((reason, stage)),
    )
    sm.step(0.0, features(zone_id="A", in_shelf_zone=True), OK, SETTINGS)
    sm.step(0.1, features(zone_id="B", in_shelf_zone=True), OK, SETTINGS)
    sm.step(1.0, features(zone_id="B", in_shelf_zone=True), OK, SETTINGS)
    assert resets == [("zone_change", "IDLE"), ("gap", "IDLE")]
    assert sm.dwell_accum_s == 0.0


def test_cooldown_completion_restarts_observation_gap_clock():
    resets = []
    sm = TrackStateMachine(
        1,
        "cam-1",
        0.0,
        on_reset=lambda pts, track_id, reason, stage: resets.append((reason, stage)),
    )
    sm.stage = "COOLDOWN"
    sm.cooldown_until_pts = 0.0
    sm.step(0.1, features(), OK, SETTINGS)
    sm.step(1.0, features(), OK, SETTINGS)
    assert resets == [("cooldown", "COOLDOWN"), ("gap", "IDLE")]
    assert sm.dwell_accum_s == 0.0


def test_zone_persistence_keeps_reach_zone():
    s = knob(zone_persist_across_shelves=True)
    sm = reaching(s, zone="A")
    sm.step(0.3, body(zone_id="B", in_shelf_zone=True), OK, s)
    assert sm.stage == "AT_BODY"
    sm.step(0.4, body(zone_id="B", in_shelf_zone=True), OK, s)
    sm.step(0.5, body(zone_id="B", in_shelf_zone=True), OK, s)
    cand = sm.step(0.6, out(zone_id="B", in_shelf_zone=True), OK, s)
    assert cand is not None and cand.zone_id == "A"


def test_zone_change_default_resets():
    sm = reaching(SETTINGS, zone="A")
    sm.step(0.3, body(zone_id="B", in_shelf_zone=True), OK, SETTINGS)
    assert sm.stage != "AT_BODY"


def test_zone_exit_grace_bridges_short_exit():
    s = knob(zone_exit_grace_s=0.3)
    sm = at_body(s, zone="A")
    sm.step(0.4, body(zone_id=None, in_shelf_zone=False), OK, s)
    assert sm.stage == "AT_BODY"
    sm.step(0.5, body(zone_id="A", in_shelf_zone=True), OK, s)
    cand = sm.step(0.6, out(zone_id="A", in_shelf_zone=True), OK, s)
    assert cand is not None and cand.zone_id == "A"


def test_zone_exit_grace_expires():
    s = knob(zone_exit_grace_s=0.3)
    sm = at_body(s, zone="A")
    for t in (0.4, 0.5, 0.6, 0.8):
        sm.step(t, body(zone_id=None, in_shelf_zone=False), OK, s)
    assert sm.stage == "IDLE"


def test_zone_exit_grace_never_applies_to_dwell():
    s = knob(zone_exit_grace_s=0.3, dwell_min_s=5.0)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(zone_id="A", in_shelf_zone=True), OK, s)
    sm.step(0.1, features(zone_id="A", in_shelf_zone=True), OK, s)
    sm.step(0.2, features(zone_id=None, in_shelf_zone=False), OK, s)
    assert sm.dwell_accum_s == 0.0


# --- 2B entry and expiry -----------------------------------------------------------

def test_reach_expiry_extends_reach():
    s = knob(reach_expiry_s=1.0)
    sm = reaching(s)
    sm.step(0.5, out(), OK, s)
    sm.step(0.7, body(), OK, s)
    assert sm.stage == "AT_BODY"


def test_partial_requires_near_body_time():
    s = knob(partial_min_near_body_s=0.2)
    sm = reaching(s)
    sm.step(0.5, out(), OK, s)
    assert sm.step(0.7, out(), OK, s) is None
    assert sm.stage == "IDLE"


def test_partial_emits_after_near_body_time():
    s = knob(partial_min_near_body_s=0.2)
    sm = reaching(s)
    for t in (0.3, 0.4, 0.5):
        sm.step(t, out(at_body_geometry=True), OK, s)
    cand = sm.step(0.7, out(), OK, s)
    assert cand is not None and cand.terminal_stage == "PARTIAL" and cand.tier == "low"


def test_reach_seeds_sequence_without_dwell_at_low_tier():
    s = knob(reach_seeds_dwell=True, dwell_min_s=5.0)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, s)
    sm.step(0.1, features(reach_detected=True, active_hand=R, torso_speed_torsos_s=0.1), OK, s)
    assert sm.stage == "REACH"
    for t in (0.2, 0.3, 0.4):
        sm.step(t, body(), OK, s)
    cand = sm.step(0.5, out(), OK, s)
    assert cand is not None and cand.tier == "low"


def test_reach_seed_rejects_walking_torso():
    s = knob(reach_seeds_dwell=True, dwell_min_s=5.0)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, s)
    sm.step(0.1, features(reach_detected=True, active_hand=R, torso_speed_torsos_s=2.0), OK, s)
    assert sm.stage == "IDLE"


def test_reach_seed_off_by_default():
    s = knob(dwell_min_s=5.0)
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, s)
    sm.step(0.1, features(reach_detected=True, active_hand=R, torso_speed_torsos_s=0.0), OK, s)
    assert sm.stage == "IDLE"


def _history(obs_at):
    hist = []
    for i in range(11):
        obs = obs_at(i)
        hist.append((i / 10, obs, torso_geometry(obs)[1]))
    return hist


def test_torso_speed_separates_standing_from_walking():
    standing = _history(lambda i: make_skeleton_obs())
    walking = _history(lambda i: make_skeleton_obs(center_x=600.0 + 60.0 * i))
    f_stand = extract_track_features(standing[-1][1], standing, ZONES, (1920, 1080), SETTINGS)
    f_walk = extract_track_features(walking[-1][1], walking, ZONES, (1920, 1080), SETTINGS)
    assert f_stand.torso_speed_torsos_s == pytest.approx(0.0, abs=1e-6)
    assert f_walk.torso_speed_torsos_s > 2.0


def test_torso_speed_unknown_without_history():
    assert features().torso_speed_torsos_s == float("inf")


# --- 2C at-body and hold -----------------------------------------------------------

def test_straight_elbow_counts_as_body_geometry_when_flex_not_required():
    obs = make_skeleton_obs()
    strict = extract_track_features(obs, [], ZONES, (1920, 1080), SETTINGS)
    loose = extract_track_features(obs, [], ZONES, (1920, 1080),
                                   knob(at_body_requires_flex=False))
    assert strict.at_body_geometry is False and loose.at_body_geometry is True


def test_straight_elbow_entry_caps_tier_at_medium():
    s = knob(at_body_requires_flex=False)
    sm = reaching(s)
    sm.step(0.3, body(elbow_flexed_or_medial=False), OK, s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, body(), OK, s)
    cand = sm.step(0.6, out(), OK, s)
    assert cand is not None and cand.tier == "medium"


def test_flicker_grace_ignores_brief_exit():
    s = knob(hold_flicker_grace_s=0.15)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    assert sm.step(0.5, out(), OK, s) is None and sm.stage == "AT_BODY"
    sm.step(0.6, body(), OK, s)
    assert sm.hold_accum_s == pytest.approx(0.1)
    sm.step(0.7, body(), OK, s)
    cand = sm.step(0.8, out(), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"


def test_flicker_grace_expires():
    s = knob(hold_flicker_grace_s=0.15)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, out(), OK, s)
    sm.step(0.6, out(), OK, s)
    assert sm.stage == "IDLE"


def test_fast_dropin_emits_low_tier():
    s = knob(fast_dropin_min_hold_s=0.1)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    cand = sm.step(0.5, out(), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED_FAST" and cand.tier == "low"


def test_fast_dropin_respects_disabled_low_tier():
    s = knob(fast_dropin_min_hold_s=0.1, emit_low_tier=False)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    assert sm.step(0.5, out(), OK, s) is None


def test_fast_dropin_requires_some_hold():
    s = knob(fast_dropin_min_hold_s=0.1)
    sm = at_body(s)
    assert sm.step(0.4, out(), OK, s) is None


# --- 2D hidden and exit ------------------------------------------------------------

def test_spatial_hidden_emits_medium_after_sustained_visible_contact():
    s = knob(spatial_hidden_s=0.5)
    sm = at_body(s)
    for t in (0.4, 0.5, 0.6, 0.7):
        assert sm.step(t, body(), OK, s) is None
    cand = sm.step(0.8, body(), OK, s)
    assert cand is not None and cand.terminal_stage == "HIDDEN_SPATIAL" and cand.tier == "medium"


def test_spatial_hidden_run_restarts_after_interruption():
    s = knob(spatial_hidden_s=0.5)
    sm = at_body(s)
    for t in (0.4, 0.5, 0.6):
        sm.step(t, body(), OK, s)
    sm.step(0.7, features(active_hand=R, active_wrist_conf=0.1), OK, s)
    for t in (0.8, 0.9, 1.0, 1.1):
        assert sm.step(t, body(), OK, s) is None


def test_spatial_hidden_run_restarts_after_bridged_gap():
    s = knob(spatial_hidden_s=0.5, sequence_max_gap_s=1.0)
    sm = at_body(s)
    for t in (0.4, 0.5, 0.6):
        sm.step(t, body(), OK, s)
    for t in (1.2, 1.3, 1.4):
        assert sm.step(t, body(), OK, s) is None
    assert sm.body_run_s == pytest.approx(0.2)
    assert sm.hold_accum_s == pytest.approx(0.5)


@pytest.mark.parametrize("observed", [True, False])
def test_monitor_spatial_hidden_run_restarts_after_missing_person(observed):
    s = knob(spatial_hidden_s=0.5)
    mon = ConcealmentMonitor("cam-1", s, ZONES)
    resting = make_skeleton_obs()
    reach = make_skeleton_obs(right_elbow_offset=(85.0, -10.0),
                              right_wrist_offset=(145.0, -55.0))
    pocket = make_skeleton_obs(right_elbow_offset=(75.0, 75.0),
                               right_wrist_offset=(5.0, 135.0))
    for i in range(3):
        mon.update(make_frame(i / 10, [resting]))
    mon.update(make_frame(0.3, [reach]))
    for i in range(4, 8):
        assert mon.update(make_frame(i / 10, [pocket])).candidates == ()
    assert mon._machines[1].stage == "AT_BODY"
    mon.update(make_frame(0.8, [], observed=observed))
    for i in range(9, 14):
        assert mon.update(make_frame(i / 10, [pocket])).candidates == ()
    fired = mon.update(make_frame(1.4, [pocket])).candidates
    assert [c.terminal_stage for c in fired] == ["HIDDEN_SPATIAL"]


def test_exit_emits_only_after_hidden_evidence():
    s = knob(exit_min_hidden_frac=0.5)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    for t in (0.5, 0.6, 0.7):
        sm.step(t, hidden(), OK, s)
    cand = sm.flush_on_track_end(0.7, s)
    assert cand is not None and cand.terminal_stage == "EXIT" and cand.tier == "medium"


def test_exit_never_fires_on_hold_alone():
    s = knob(exit_min_hidden_frac=0.5)
    sm = at_body(s)
    for t in (0.4, 0.5, 0.6):
        sm.step(t, body(), OK, s)
    assert sm.flush_on_track_end(0.6, s) is None


def test_exit_needs_enough_hidden_time():
    s = knob(exit_min_hidden_frac=0.5)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, hidden(), OK, s)
    assert sm.flush_on_track_end(0.5, s) is None


def test_exit_off_by_default():
    sm = at_body(SETTINGS)
    for t in (0.4, 0.5, 0.6):
        sm.step(t, hidden(), OK, SETTINGS)
    assert sm.flush_on_track_end(0.6, SETTINGS) is None


# --- 2E hand transfer --------------------------------------------------------------

def _transfer_ready(**changes):
    return features(active_hand=R, other_hand_visible=True, other_hand_in_at_body_pose=True,
                    other_hand_at_body_geometry=True, **changes)


def test_reach_transfers_to_other_hand_that_reached():
    s = knob(hand_transfer=True)
    sm = reaching(s)
    assert sm.step(0.3, _transfer_ready(other_hand_reach_detected=True), OK, s) is None
    assert sm.active_hand == "left" and sm.stage == "REACH"
    assert any(t.stage == "TRANSFER" for t in sm.timeline)
    sm.step(0.4, features(active_hand="left", in_at_body_pose=True), OK, s)
    assert sm.stage == "AT_BODY"


def test_reach_handoff_precedes_expiry():
    s = knob(hand_transfer=True, reach_expiry_s=0.3)
    sm = reaching(s)
    assert sm.step(0.6, _transfer_ready(other_hand_reach_detected=True), OK, s) is None
    assert sm.stage == "REACH" and sm.active_hand == "left"
    assert [t.stage for t in sm.timeline].count("TRANSFER") == 1


def test_reach_handoff_precedes_near_body_evidence():
    s = knob(hand_transfer=True, partial_min_near_body_s=0.2)
    sm = reaching(s)
    sm.step(0.3, features(active_hand=R, at_body_geometry=True), OK, s)
    previous_near_body_s = sm.near_body_s
    assert sm.step(0.4, _transfer_ready(other_hand_reach_detected=True), OK, s) is None
    assert sm.active_hand == "left"
    assert sm.near_body_s == previous_near_body_s


def test_reach_handoff_precedes_current_hand_at_body_entry():
    s = knob(hand_transfer=True)
    sm = reaching(s)
    assert sm.step(
        0.3,
        _transfer_ready(in_at_body_pose=True, other_hand_reach_detected=True),
        OK,
        s,
    ) is None
    assert sm.stage == "REACH" and sm.active_hand == "left"
    assert all(t.stage != "AT_BODY" for t in sm.timeline)


def test_reach_transfer_needs_reach_or_meeting():
    s = knob(hand_transfer=True)
    sm = reaching(s)
    sm.step(0.3, _transfer_ready(), OK, s)
    assert sm.active_hand == R


def test_transfer_happens_once_per_sequence():
    s = knob(hand_transfer=True)
    sm = reaching(s)
    sm.step(0.3, _transfer_ready(other_hand_reach_detected=True), OK, s)
    sm.step(0.4, replace(_transfer_ready(other_hand_reach_detected=True), active_hand="left"),
            OK, s)
    assert sm.active_hand == "left"


def test_transfer_off_by_default():
    sm = reaching(SETTINGS)
    sm.step(0.3, _transfer_ready(other_hand_reach_detected=True, hands_met_recent=True),
            OK, SETTINGS)
    assert sm.active_hand == R


def test_at_body_handoff_continues_sequence():
    s = knob(hand_transfer=True)
    sm = at_body(s)
    assert sm.step(0.4, out(other_hand_visible=True, other_hand_at_body_geometry=True,
                            hands_met_recent=True), OK, s) is None
    assert sm.stage == "AT_BODY" and sm.active_hand == "left"
    for t in (0.5, 0.6, 0.7):
        sm.step(t, features(active_hand="left", in_at_body_pose=True), OK, s)
    cand = sm.step(0.8, features(active_hand="left"), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"


def test_at_body_handoff_accepts_other_hand_reach_without_meeting():
    s = knob(hand_transfer=True)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    assert sm.step(0.5, out(other_hand_visible=True, other_hand_at_body_geometry=True,
                            other_hand_reach_detected=True, hands_met_recent=False), OK, s) is None
    assert sm.stage == "AT_BODY" and sm.active_hand == "left"
    assert sm.hold_accum_s == pytest.approx(0.1)
    for t in (0.6, 0.7):
        sm.step(t, features(active_hand="left", in_at_body_pose=True), OK, s)
    cand = sm.step(0.8, features(active_hand="left", at_body_geometry=False), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"
    assert [t.stage for t in cand.stage_timeline].count("TRANSFER") == 1


def test_at_body_return_after_minimum_hold_precedes_handoff():
    s = knob(hand_transfer=True)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    sm.step(0.5, body(), OK, s)
    assert sm.hold_accum_s == pytest.approx(s.min_hold_s)
    cand = sm.step(0.6, out(other_hand_visible=True, other_hand_at_body_geometry=True,
                            other_hand_reach_detected=True), OK, s)
    assert cand is not None and cand.terminal_stage == "RETURNED"


def test_at_body_active_body_evidence_precedes_handoff():
    s = knob(hand_transfer=True)
    sm = at_body(s)
    sm.step(0.4, body(), OK, s)
    assert sm.step(0.5, body(other_hand_visible=True, other_hand_at_body_geometry=True,
                             other_hand_reach_detected=True), OK, s) is None
    assert sm.active_hand == R
    assert sm.hold_accum_s == pytest.approx(0.2)
    assert sm.body_run_s == pytest.approx(0.2)


def test_at_body_hidden_evidence_precedes_handoff():
    s = knob(hand_transfer=True)
    sm = at_body(s)
    sm.step(0.4, hidden(), OK, s)
    sm.step(0.5, hidden(), OK, s)
    assert sm.hidden_accum_s == pytest.approx(0.1)
    assert sm.step(0.6, hidden(other_hand_visible=True, other_hand_at_body_geometry=True,
                               hands_met_recent=True), OK, s) is None
    assert sm.active_hand == R
    assert sm.hidden_accum_s == pytest.approx(0.2)


def test_hands_met_is_computed_from_history():
    s = knob(hand_transfer=True)
    met = make_skeleton_obs(right_elbow_offset=(-20.0, 70.0), right_wrist_offset=(-5.0, 145.0),
                            left_elbow_offset=(20.0, 70.0), left_wrist_offset=(5.0, 145.0))
    hist = [(0.0, met, torso_geometry(met)[1])]
    feat = extract_track_features(met, hist, ZONES, (1920, 1080), s)
    assert feat.hands_met_recent is True
    apart = extract_track_features(make_skeleton_obs(), [(0.0, make_skeleton_obs(), 140.0)],
                                   ZONES, (1920, 1080), s)
    assert apart.hands_met_recent is False


# --- 2F cooldown scope -------------------------------------------------------------

def _emit_in_zone(s, zone):
    sm = at_body(s, zone=zone)
    z = {"zone_id": zone, "in_shelf_zone": True}
    sm.step(0.4, body(**z), OK, s)
    sm.step(0.5, body(**z), OK, s)
    assert sm.step(0.6, out(**z), OK, s) is not None
    return sm


def test_zone_scoped_cooldown_allows_other_shelf():
    s = knob(cooldown_scope="track_zone")
    sm = _emit_in_zone(s, "A")
    for i in range(7, 12):
        sm.step(i / 10, features(zone_id="B", in_shelf_zone=True), OK, s)
    assert sm.stage == "DWELL"


def test_zone_scoped_cooldown_still_blocks_same_shelf():
    s = knob(cooldown_scope="track_zone")
    sm = _emit_in_zone(s, "A")
    for i in range(7, 12):
        sm.step(i / 10, features(zone_id="A", in_shelf_zone=True), OK, s)
    assert sm.stage == "COOLDOWN"


def test_track_cooldown_blocks_every_shelf_by_default():
    sm = _emit_in_zone(SETTINGS, "A")
    for i in range(7, 12):
        sm.step(i / 10, features(zone_id="B", in_shelf_zone=True), OK, SETTINGS)
    assert sm.stage == "COOLDOWN"


# --- configuration -----------------------------------------------------------------

def _load_with(tmp_path, detector_yaml):
    from config import CONFIG_PATH, load_config
    cfg = tmp_path / "knobs.yaml"
    cfg.write_text(f"extends: '{CONFIG_PATH.as_posix()}'\ndetector:\n{detector_yaml}", encoding="utf-8")
    return load_config(cfg, env={})


def test_config_accepts_bool_and_scope_knobs(tmp_path):
    cfg = _load_with(tmp_path, "  hand_transfer: true\n  cooldown_scope: track_zone\n")
    assert cfg.detector.hand_transfer is True and cfg.detector.cooldown_scope == "track_zone"


@pytest.mark.parametrize("line", [
    "  hand_transfer: 1\n",
    "  cooldown_scope: 3\n",
    "  cooldown_scope: shelf\n",
    "  spatial_hidden_s: 0.1\n",
])
def test_config_rejects_bad_knobs(tmp_path, line):
    with pytest.raises(ValueError):
        _load_with(tmp_path, line)
