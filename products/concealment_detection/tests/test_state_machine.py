"""Unit and scenario tests for `concealment/state_machine.py` (ACs 1, 2, 3, 5, 6)."""
from __future__ import annotations

from concealment.state_machine import ConcealmentMonitor
from concealment.types import CameraZones, DetectorSettings, ZonePolygon
from conftest import make_frame, make_skeleton_obs


def _shelf_zones() -> CameraZones:
    return CameraZones(
        camera="cam-1",
        zone_free=False,
        zones=(
            ZonePolygon(
                id="shelf-a",
                points=((0.10, 0.20), (0.90, 0.20), (0.90, 0.85), (0.10, 0.85)),
                facing=(0.0, -1.0),
            ),
        ),
    )


def test_full_clothing_concealment_sequence_emits_high_tier_and_respects_cooldown():
    """AC-2: Person dwells near shelf, reaches toward shelf, brings hand to clothing (waist pocket)
    with flexed elbow for >= min_hold_s, and returns hand -> emits high-tier REVIEW candidate."""
    settings = DetectorSettings(
        min_track_s=0.3,
        dwell_min_s=1.0,
        min_hold_s=0.35,
        cooldown_s=5.0,
    )
    mon = ConcealmentMonitor("cam-1", settings, _shelf_zones())
    candidates = []

    # 1. Dwell in shelf-a from t=0.0 to t=1.2 (arms resting at sides)
    for i in range(13):
        t = round(i * 0.1, 2)
        res = mon.update(make_frame(t, [make_skeleton_obs(track_id=7)], frame_num=i))
        assert res.candidates == ()

    # 2. Reach toward shelf (upward/outward) from t=1.3 to t=1.6
    for i in range(13, 17):
        t = round(i * 0.1, 2)
        obs_reach = make_skeleton_obs(
            track_id=7,
            right_elbow_offset=(80.0, -15.0),
            right_wrist_offset=(140.0, -60.0),
        )
        res = mon.update(make_frame(t, [obs_reach], frame_num=i))
        assert res.candidates == ()

    # 3. Hand into waist pocket (flexed elbow) from t=1.7 to t=2.2 (0.5s >= min_hold_s=0.35s)
    for i in range(17, 23):
        t = round(i * 0.1, 2)
        obs_pocket = make_skeleton_obs(
            track_id=7,
            right_elbow_offset=(75.0, 75.0),
            right_wrist_offset=(5.0, 135.0),
        )
        res = mon.update(make_frame(t, [obs_pocket], frame_num=i))
        candidates.extend(res.candidates)

    # 4. Hand returns to resting side at t=2.3 -> triggers RETURNED -> CANDIDATE
    res_ret = mon.update(make_frame(2.3, [make_skeleton_obs(track_id=7)], frame_num=23))
    candidates.extend(res_ret.candidates)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.camera == "cam-1"
    assert cand.track_id == 7
    assert cand.tier == "high"
    assert cand.target_kind == "clothing"
    assert cand.zone_id == "shelf-a"
    assert cand.terminal_stage == "RETURNED"
    stage_names = [s.stage for s in cand.stage_timeline]
    assert "DWELL" in stage_names
    assert "REACH" in stage_names
    assert "AT_BODY" in stage_names
    assert "RETURNED" in stage_names

    # 5. Immediate repeat within cooldown_s (5.0s) must be suppressed
    for i in range(24, 34):
        t = round(i * 0.1, 2)
        obs_pocket = make_skeleton_obs(
            track_id=7,
            right_elbow_offset=(75.0, 75.0),
            right_wrist_offset=(5.0, 135.0),
        )
        res_cd = mon.update(make_frame(t, [obs_pocket], frame_num=i))
        assert res_cd.candidates == ()


def test_personal_bag_concealment_with_sustained_hidden_wrist():
    """AC-3 & T6: Concealing an item into a personal bag where wrist confidence stays dropped
    for >= hidden_s triggers AT_BODY -> HIDDEN -> CANDIDATE with target_kind == 'personal_bag'."""
    settings = DetectorSettings(
        min_track_s=0.3,
        dwell_min_s=0.8,
        hidden_s=0.40,
        min_hold_s=2.0,  # set high so HIDDEN is what fires first
    )
    mon = ConcealmentMonitor("cam-1", settings, _shelf_zones())
    bag_box = (990.0, 495.0, 1060.0, 580.0)

    # Dwell 0.0 .. 1.0s with high wrist confidence (0.92)
    for i in range(11):
        t = round(i * 0.1, 2)
        obs = make_skeleton_obs(track_id=3, right_wrist_conf=0.92, has_bag=True, bag_box=bag_box)
        mon.update(make_frame(t, [obs], frame_num=i))

    # Reach 1.1 .. 1.3s
    for i in range(11, 14):
        t = round(i * 0.1, 2)
        obs_reach = make_skeleton_obs(
            track_id=3,
            right_elbow_offset=(85.0, -10.0),
            right_wrist_offset=(145.0, -55.0),
            right_wrist_conf=0.92,
            has_bag=True,
            bag_box=bag_box,
        )
        mon.update(make_frame(t, [obs_reach], frame_num=i))

    # Enter bag at t=1.4 with a 1-frame dip (0.1s < hidden_s=0.40s), then back to high conf -> must NOT fire HIDDEN
    obs_bag_dip = make_skeleton_obs(
        track_id=3,
        right_elbow_offset=(80.0, 70.0),
        right_wrist_offset=(60.0, 130.0),
        right_wrist_conf=0.15,
        has_bag=True,
        bag_box=bag_box,
    )
    assert mon.update(make_frame(1.4, [obs_bag_dip], frame_num=14)).candidates == ()
    obs_bag_visible = make_skeleton_obs(
        track_id=3,
        right_elbow_offset=(80.0, 70.0),
        right_wrist_offset=(60.0, 130.0),
        right_wrist_conf=0.90,
        has_bag=True,
        bag_box=bag_box,
    )
    assert mon.update(make_frame(1.5, [obs_bag_visible], frame_num=15)).candidates == ()

    # Now sustained wrist confidence drop inside bag from t=1.6 to t=2.1 (0.5s >= hidden_s=0.40s)
    fired = []
    for i in range(16, 22):
        t = round(i * 0.1, 2)
        res = mon.update(make_frame(t, [obs_bag_dip], frame_num=i))
        fired.extend(res.candidates)

    assert len(fired) == 1
    assert fired[0].target_kind == "personal_bag"
    assert fired[0].body_zone_name == "personal_bag"
    assert fired[0].terminal_stage == "HIDDEN"
    assert fired[0].tier == "high"


def test_short_track_grace_path_emits_medium_tier_on_trimmed_clip():
    """CALIBRATION_DATASETS.md §4: A track born inside a shelf zone that reaches within short_track_grace_s
    (without full dwell_min_s) emits a 'medium' tier candidate."""
    settings = DetectorSettings(
        min_track_s=0.2,
        dwell_min_s=3.0,
        short_track_grace_s=1.2,
        min_hold_s=0.30,
    )
    mon = ConcealmentMonitor("cam-1", settings, _shelf_zones())
    # t=0.0 .. 0.2: brief appearance
    for i in range(3):
        mon.update(make_frame(round(i * 0.1, 2), [make_skeleton_obs(track_id=11)], frame_num=i))
    # t=0.3 .. 0.5: reach (within short_track_grace_s=1.2s, while dwell < 3.0s)
    for i in range(3, 6):
        obs_reach = make_skeleton_obs(
            track_id=11,
            right_elbow_offset=(85.0, -10.0),
            right_wrist_offset=(145.0, -55.0),
        )
        mon.update(make_frame(round(i * 0.1, 2), [obs_reach], frame_num=i))
    # t=0.6 .. 1.0: hold at chest/waist (0.4s >= 0.30s)
    fired = []
    for i in range(6, 11):
        obs_body = make_skeleton_obs(
            track_id=11,
            right_elbow_offset=(75.0, 75.0),
            right_wrist_offset=(5.0, 135.0),
        )
        fired.extend(mon.update(make_frame(round(i * 0.1, 2), [obs_body], frame_num=i)).candidates)
    # t=1.1: return hand
    fired.extend(mon.update(make_frame(1.1, [make_skeleton_obs(track_id=11)], frame_num=11)).candidates)

    assert len(fired) == 1
    assert fired[0].tier == "medium"


def test_brief_brush_and_normal_standing_do_not_alert():
    """Hard negatives: (a) brief hand brush (< min_hold_s) and (b) standing/walking with arms at sides."""
    settings = DetectorSettings(min_track_s=0.2, dwell_min_s=0.6, min_hold_s=0.50, emit_low_tier=False)
    mon = ConcealmentMonitor("cam-1", settings, _shelf_zones())

    # Dwell + Reach
    for i in range(8):
        mon.update(make_frame(round(i * 0.1, 2), [make_skeleton_obs(track_id=5)], frame_num=i))
    for i in range(8, 11):
        obs_reach = make_skeleton_obs(
            track_id=5,
            right_elbow_offset=(85.0, -10.0),
            right_wrist_offset=(145.0, -55.0),
        )
        mon.update(make_frame(round(i * 0.1, 2), [obs_reach], frame_num=i))

    # Brief brush at waist for only 0.2s (< min_hold_s=0.50s), then return to side
    for i in range(11, 13):
        obs_brush = make_skeleton_obs(
            track_id=5,
            right_elbow_offset=(75.0, 75.0),
            right_wrist_offset=(5.0, 135.0),
        )
        assert mon.update(make_frame(round(i * 0.1, 2), [obs_brush], frame_num=i)).candidates == ()
    for i in range(13, 20):
        assert mon.update(make_frame(round(i * 0.1, 2), [make_skeleton_obs(track_id=5)], frame_num=i)).candidates == ()


def test_insufficient_footage_logs_assessment_and_never_emits_candidate():
    """AC-6: Given the available footage is insufficient to assess the behavior reliably,
    IRIS logs insufficient_evidence and never emits a theft/concealment candidate."""
    settings = DetectorSettings(min_box_height_px=60.0)
    mon = ConcealmentMonitor("cam-1", settings, _shelf_zones())
    tiny_obs = make_skeleton_obs(
        track_id=9,
        right_elbow_offset=(75.0, 75.0),
        right_wrist_offset=(5.0, 135.0),
        box=(100.0, 100.0, 120.0, 140.0),  # height 40 < 60
    )
    res = mon.update(make_frame(1.0, [tiny_obs], frame_num=1))
    assert res.candidates == ()
    assert len(res.assessments) == 1
    assert res.assessments[0].state == "insufficient_evidence"
    assert res.assessments[0].track_id == 9
