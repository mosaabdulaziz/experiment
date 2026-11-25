"""Safety regressions: observations, identity, geometry, and terminal evidence."""
from dataclasses import replace

import pytest

from concealment.features import extract_track_features
from concealment.quality import QualityGate, ReentryGrace
from concealment.state_machine import ConcealmentMonitor, TrackStateMachine
from concealment.types import CameraZones, DetectorSettings, QualityStatus
from concealment.zones import propose_zones_from_dwell
from conftest import make_frame, make_skeleton_obs


SETTINGS = DetectorSettings(min_track_s=0.0, dwell_min_s=0.2, min_hold_s=0.2,
                            hidden_s=0.3, short_track_grace_s=0.0, reach_window_s=0.4)
OK = QualityStatus(state="OK")
ZONES = CameraZones(camera="cam-1", zone_free=True)


def features(**changes):
    base = extract_track_features(make_skeleton_obs(), [], ZONES, (1920, 1080), SETTINGS)
    return replace(base, **changes)


def reaching_machine(settings=SETTINGS):
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(), OK, settings)
    sm.step(0.1, features(), OK, settings)
    sm.step(0.2, features(reach_detected=True, active_hand="right"), OK, settings)
    assert sm.stage == "REACH"
    return sm


def body_machine(settings=SETTINGS):
    sm = reaching_machine(settings)
    sm.step(0.3, features(active_hand="right", in_at_body_pose=True), OK, settings)
    assert sm.stage == "AT_BODY"
    return sm


def body_obs(**kwargs):
    return make_skeleton_obs(right_elbow_offset=(75.0, 75.0),
                             right_wrist_offset=(5.0, 135.0), **kwargs)


def reach_obs(**kwargs):
    return make_skeleton_obs(right_elbow_offset=(85.0, -10.0),
                             right_wrist_offset=(145.0, -55.0), **kwargs)


def body_monitor():
    mon = ConcealmentMonitor("cam-1", SETTINGS, ZONES)
    for i in range(3):
        mon.update(make_frame(i / 10, [make_skeleton_obs()]))
    mon.update(make_frame(0.3, [reach_obs()]))
    mon.update(make_frame(0.4, [body_obs()]))
    assert mon._machines[1].stage == "AT_BODY"
    return mon


def test_reaching_hand_cannot_be_replaced_by_other_body_hand():
    sm = reaching_machine()
    sm.step(0.3, features(active_hand="left", in_at_body_pose=True), OK, SETTINGS)
    assert sm.stage == "REACH"


def test_other_hand_cannot_supply_return_after_valid_entry():
    sm = body_machine()
    sm.step(0.5, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert sm.step(0.6, features(active_hand="left"), OK, SETTINGS) is None


def test_sustained_body_contact_is_not_a_return():
    sm = body_machine()
    for i in range(4, 25):
        assert sm.step(i / 10, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS) is None


def test_disappearance_never_supplies_return_evidence():
    sm = body_machine()
    sm.step(0.5, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert sm.flush_on_track_end(2.0, SETTINGS) is None


def test_low_confidence_exit_is_hidden_only_after_sustained_loss():
    sm = body_machine()
    sm.step(0.5, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    hidden = features(active_hand="right", active_wrist_conf=0.0, wrist_hidden_now=True)
    assert sm.step(0.6, hidden, OK, SETTINGS) is None
    assert sm.step(0.8, hidden, OK, SETTINGS) is None
    cand = sm.step(0.9, hidden, OK, SETTINGS)
    assert cand is not None and cand.terminal_stage == "HIDDEN"


def test_hidden_coordinates_cannot_establish_body_entry():
    sm = reaching_machine()
    sm.step(0.3, features(active_hand="right", in_at_body_pose=True,
                          active_wrist_conf=0.0, wrist_hidden_now=True), OK, SETTINGS)
    assert sm.stage == "REACH"


def test_expired_reach_rejects_late_body_contact_and_emits_low_partial():
    sm = reaching_machine()
    sm.step(0.5, features(active_hand="right"), OK, SETTINGS)
    cand = sm.step(0.7, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert cand is not None and cand.terminal_stage == "PARTIAL" and cand.tier == "low"
    assert all(t.stage != "AT_BODY" for t in cand.stage_timeline)


def test_partial_respects_disabled_low_tier():
    settings = replace(SETTINGS, emit_low_tier=False)
    sm = reaching_machine(settings)
    sm.step(0.5, features(active_hand="right"), OK, settings)
    assert sm.step(0.7, features(active_hand="right", in_at_body_pose=True), OK, settings) is None
    assert sm.stage != "AT_BODY"


def test_hold_does_not_count_unseen_exit_interval():
    sm = body_machine()
    sm.step(0.4, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert sm.step(0.6, features(active_hand="right"), OK, SETTINGS) is None


def test_insufficient_reset_clears_dwell_but_preserves_cooldown():
    sm = body_machine()
    sm.step(0.5, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert sm.step(0.6, features(active_hand="right"), OK, SETTINGS) is not None
    until = sm.cooldown_until_pts
    sm.step(0.7, features(), QualityStatus(state="INSUFFICIENT"), SETTINGS)
    sm.step(0.8, features(reach_detected=True, in_at_body_pose=True), OK, SETTINGS)
    assert sm.stage == "COOLDOWN" and sm.cooldown_until_pts == until
    assert sm.dwell_accum_s == 0.0


def test_dwell_is_fresh_after_quality_reset():
    sm = reaching_machine()
    sm.step(0.3, features(), QualityStatus(state="INSUFFICIENT"), SETTINGS)
    sm.step(0.4, features(), OK, SETTINGS)
    assert sm.stage == "IDLE" and sm.dwell_accum_s == 0.0


def test_dwell_is_fresh_in_a_different_zone():
    sm = TrackStateMachine(1, "cam-1", 0.0)
    sm.step(0.0, features(zone_id="a"), OK, SETTINGS)
    sm.step(0.1, features(zone_id="a"), OK, SETTINGS)
    sm.step(0.2, features(zone_id="b", reach_detected=True), OK, SETTINGS)
    assert sm.stage == "IDLE" and sm.dwell_accum_s == 0.0


@pytest.mark.parametrize("bad_index", [5, 8, 10, 11])
def test_unconfident_geometry_cannot_generate_reach(bad_index):
    rest = make_skeleton_obs()
    bad = reach_obs()
    kps = list(bad.kps)
    x, y, _ = kps[bad_index]
    kps[bad_index] = (x, y, 0.0)
    bad = replace(bad, kps=tuple(kps))
    feat = extract_track_features(rest, [(0.0, rest, 140.0), (0.1, bad, 140.0),
                                         (0.2, rest, 140.0)], ZONES, (1920, 1080), SETTINGS)
    assert feat.reach_detected is False


def test_zero_confidence_zero_coordinates_do_not_supply_reach_extrema():
    rest = make_skeleton_obs()
    kps = list(rest.kps)
    kps[10] = (0.0, 0.0, 0.0)
    bad = replace(rest, kps=tuple(kps))
    feat = extract_track_features(rest, [(0.0, rest, 140.0), (0.1, bad, 140.0),
                                         (0.2, rest, 140.0)], ZONES, (1920, 1080), SETTINGS)
    assert feat.wrist_excursion_torsos == 0.0 and not feat.reach_detected


@pytest.mark.parametrize("raw_order", [(25, 99), (99, 25), (10, 25)])
def test_reid_alias_cannot_merge_simultaneously_visible_people(raw_order):
    grace = ReentryGrace(SETTINGS)
    first = make_skeleton_obs(track_id=10)
    grace.observe_active(10, first, 0.0, (960.0, 470.0), 140.0)
    switched = make_skeleton_obs(track_id=25)
    assert grace.resolve_id(25, switched, 0.1, {25}) == 10
    grace.observe_active(10, switched, 0.1, (960.0, 470.0), 140.0)
    ids = []
    for raw in raw_order:
        obs = make_skeleton_obs(track_id=raw)
        cid = grace.resolve_id(raw, obs, 0.2, set(raw_order))
        grace.observe_active(cid, obs, 0.2, (960.0, 470.0), 140.0)
        ids.append(cid)
    assert len(set(ids)) == 2
    assert ids[raw_order.index(25)] == 10


@pytest.mark.parametrize("failure", ["clock", "size", "startup"])
def test_camera_rejects_discontinuity_and_initial_missing_pose(failure):
    gate = QualityGate(SETTINGS)
    if failure != "startup":
        gate.evaluate_camera(make_frame(10.0, [make_skeleton_obs()]))
    frame = {"clock": make_frame(1.0), "size": make_frame(10.1, shape=(640, 480)),
             "startup": make_frame(0.0, observed=False)}[failure]
    state, reason = gate.evaluate_camera(frame)
    assert state != "LIVE" and reason


@pytest.mark.parametrize("gap_kind", ["silence", "skipped", "stale", "size", "clock"])
def test_monitor_resets_evidence_across_gaps(gap_kind):
    mon = body_monitor()
    if gap_kind == "skipped":
        mon.update(make_frame(0.5, observed=False))
        resumed = 1.5
    elif gap_kind == "stale":
        assert not mon.update(make_frame(4.0, observed=False)).camera_ok
        resumed = 4.1
    elif gap_kind == "size":
        mon.update(make_frame(0.5, shape=(640, 480)))
        resumed = 0.6
    elif gap_kind == "clock":
        mon.update(make_frame(0.0))
        resumed = 0.1
    else:
        resumed = 1.5
    result = mon.update(make_frame(resumed, [make_skeleton_obs()]))
    assert result.candidates == ()
    assert all(t.stage != "AT_BODY" for t in mon._machines[1].timeline)
    assert not mon._histories.get(1) or len(mon._histories[1]) == 1


def test_one_frame_absence_bridges_the_sequence_but_credits_no_time():
    """Under MAX_OBSERVATION_GAP_S a blip is a normal skipped inference, so the reach history
    survives -- but the unseen interval must not be counted as dwell the person never held."""
    mon = ConcealmentMonitor("cam-1", SETTINGS, ZONES)
    rest = make_skeleton_obs()
    mon.update(make_frame(0.0, [rest]))
    mon.update(make_frame(0.1, []))
    mon.update(make_frame(0.2, [rest]))
    assert len(mon._histories[1]) == 2
    assert mon._machines[1].stage == "IDLE"
    assert mon._machines[1].dwell_accum_s < SETTINGS.dwell_min_s


def test_absence_does_not_extend_a_hidden_wrist_interval():
    mon = body_monitor()
    hidden = body_obs(right_wrist_conf=0.0)
    mon.update(make_frame(0.5, [hidden]))
    mon.update(make_frame(0.6, []))
    result = mon.update(make_frame(0.9, [hidden]))
    assert result.candidates == ()
    assert mon._machines[1].hidden_accum_s < SETTINGS.hidden_s


def test_healthy_interval_skip_keeps_valid_sequence():
    mon = body_monitor()
    mon.update(make_frame(0.5, observed=False))
    mon.update(make_frame(0.6, [body_obs()]))
    result = mon.update(make_frame(0.7, [make_skeleton_obs()]))
    assert len(result.candidates) == 1
    assert result.candidates[0].terminal_stage == "RETURNED"
    assert result.candidates[0].hold_duration_s == pytest.approx(0.2)


def test_monitor_hidden_zero_coordinates_uses_locked_hand():
    mon = body_monitor()
    hidden = body_obs(right_wrist_conf=0.0)
    kps = list(hidden.kps)
    kps[10] = (0.0, 0.0, 0.0)
    hidden = replace(hidden, kps=tuple(kps))
    fired = []
    for i in range(5, 10):
        fired.extend(mon.update(make_frame(i / 10, [hidden])).candidates)
    assert len(fired) == 1 and fired[0].terminal_stage == "HIDDEN"


def test_monitor_retains_full_four_second_inward_history():
    mon = ConcealmentMonitor("cam-1", replace(SETTINGS, reach_window_s=2.5), ZONES)
    for i in range(40):
        mon.update(make_frame(i / 10, [reach_obs() if i == 0 else body_obs()]))
    assert mon._histories[1][0][0] == 0.0
    assert len({t for t, _, _ in mon._histories[1]}) == len(mon._histories[1])


def test_monitor_invalidate_clears_evidence_quality_and_keeps_cooldown():
    mon = body_monitor()
    mon.update(make_frame(0.6, [body_obs()]))
    assert mon.update(make_frame(0.7, [make_skeleton_obs()])).candidates
    until = mon._machines[1].cooldown_until_pts
    assert callable(getattr(mon, "invalidate", None)), "backend needs explicit invalidation hook"
    assert mon.invalidate(0.8) is None
    assert not mon._histories and not mon.quality_gate._tracks
    assert mon._machines[1].stage == "COOLDOWN"
    assert mon._machines[1].cooldown_until_pts == until
    assert not mon._machines[1].timeline


def test_expired_tracks_quality_and_aliases_are_removed():
    mon = body_monitor()
    mon.update(make_frame(0.5, []))
    mon.update(make_frame(12.0, []))
    assert not mon._machines and not mon._histories and not mon._last_quality_state
    assert not mon.quality_gate._tracks
    assert not mon.reentry_grace._records and not mon.reentry_grace._alias_to_canonical


def test_cooldown_retained_after_reid_grace_then_expires():
    mon = body_monitor()
    mon.update(make_frame(0.6, [body_obs()]))
    mon.update(make_frame(0.7, [make_skeleton_obs()]))
    until = mon._machines[1].cooldown_until_pts
    mon.update(make_frame(3.0, []))
    assert 1 in mon._machines and mon._machines[1].cooldown_until_pts == until
    mon.update(make_frame(until + 2.0, []))
    assert not mon._machines


@pytest.mark.parametrize("gap_kind", ["missing", "unobserved", "size", "clock"])
def test_zone_proposal_does_not_count_missing_or_reset_footage(gap_kind):
    obs = make_skeleton_obs()
    frames = [make_frame(0.0, [obs])]
    if gap_kind == "missing":
        frames += [make_frame(0.2, []), make_frame(0.4, [obs])]
    elif gap_kind == "unobserved":
        frames += [make_frame(0.2, [obs], observed=False)]
    elif gap_kind == "size":
        frames += [make_frame(0.2, [obs], shape=(1921, 1080))]
    else:
        frames += [make_frame(-0.2, [obs]), make_frame(0.0, [obs])]
    assert not propose_zones_from_dwell(frames, min_dwell_s=0.1).zones


def _logged_machine(log, settings=SETTINGS):
    sm = TrackStateMachine(1, "cam-1", 0.0, on_reset=lambda *event: log.append(event))
    sm.step(0.0, features(), OK, settings)
    sm.step(0.1, features(), OK, settings)
    sm.step(0.2, features(reach_detected=True, active_hand="right"), OK, settings)
    return sm


def test_reset_reports_reach_expiry_with_the_lost_stage():
    log = []
    sm = _logged_machine(log)
    hidden = features(active_hand="right", active_wrist_conf=0.0, wrist_hidden_now=True)
    sm.step(0.3, features(active_hand="right"), OK, SETTINGS)
    sm.step(0.7, hidden, OK, SETTINGS)
    assert (0.7, 1, "reach_expired", "REACH") in log


def test_reset_reports_brief_hold_and_suppressed_low_tier():
    log = []
    sm = _logged_machine(log)
    sm.step(0.3, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    sm.step(0.4, features(active_hand="right"), OK, SETTINGS)
    assert log[-1] == (0.4, 1, "brief_hold", "AT_BODY")
    log.clear()
    no_low = replace(SETTINGS, emit_low_tier=False)
    sm = _logged_machine(log, no_low)
    sm.step(0.5, features(active_hand="right"), OK, no_low)
    sm.step(0.7, features(active_hand="right"), OK, no_low)
    assert log[-1] == (0.7, 1, "low_tier_suppressed", "REACH")


def test_emission_is_not_reported_as_a_lost_sequence():
    log = []
    sm = _logged_machine(log)
    sm.step(0.3, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    sm.step(0.5, features(active_hand="right", in_at_body_pose=True), OK, SETTINGS)
    assert sm.step(0.6, features(active_hand="right"), OK, SETTINGS) is not None
    assert all(stage not in ("REACH", "AT_BODY") for _, _, _, stage in log)


@pytest.mark.parametrize("reason,zone", [("zone_change", "b"), ("zone_exit", None)])
def test_reset_distinguishes_zone_change_from_zone_exit(reason, zone):
    log = []
    sm = _logged_machine(log)
    sm.step(0.3, features(active_hand="right", zone_id=zone, in_shelf_zone=zone is not None),
            OK, SETTINGS)
    assert log[-1] == (0.3, 1, reason, "REACH")


def test_monitor_counts_resets_and_logs_lost_sequences():
    mon = body_monitor()
    mon.update(make_frame(1.5, [make_skeleton_obs()]))
    assert (1.5, 1, "expired_track", "AT_BODY") in mon.sequence_resets
    assert mon.reset_counts[("expired_track", "AT_BODY")] == 1
    assert all(stage in ("REACH", "AT_BODY") for _, _, _, stage in mon.sequence_resets)


def _spy_track_end(monkeypatch, result=None):
    seen = []
    original = TrackStateMachine.flush_on_track_end

    def spy(self, pts, settings):
        seen.append((self.stage, pts))
        original(self, pts, settings)
        return result

    monkeypatch.setattr(TrackStateMachine, "flush_on_track_end", spy)
    return seen


def test_track_end_sees_the_live_sequence_once_at_last_sighting(monkeypatch):
    mon = body_monitor()
    seen = _spy_track_end(monkeypatch)
    for i in range(5, 30):
        mon.update(make_frame(i / 10, []))
    assert seen == [("AT_BODY", pytest.approx(0.4))]
    assert mon.reset_counts[("expired_track", "AT_BODY")] == 1


def test_track_end_candidate_reaches_the_caller(monkeypatch):
    mon = body_monitor()
    sentinel = object()
    _spy_track_end(monkeypatch, result=sentinel)
    emitted = [c for i in range(5, 30) for c in mon.update(make_frame(i / 10, [])).candidates]
    assert emitted == [sentinel]
