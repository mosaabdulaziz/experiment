from collections import deque
from shop_helpers import (C_DOOR, C_SOFA, C_STREET, GEO, L_AISLE, L_EDGE, L_SEAT, POLICY, ZONES, frame, reid_app,
                          staff_app, ts)
from domain import Frame, TrackObs
from tracklets import TrackletBook, cosine, seat_ok, staff_vote, unit


def centre():
    return TrackletBook("barber-c", ZONES["barber-c"], GEO)


def left():
    return TrackletBook("barber-l", ZONES["barber-l"], GEO)


def test_new_track_then_area_entry_and_exit():
    b = centre()
    ev = b.observe(frame("barber-c", 0, {1: C_DOOR}), POLICY)
    assert [t.track_id for t in ev.new] == [1] and ev.entered == []
    ev = b.observe(frame("barber-c", 1, {1: C_SOFA}), POLICY)
    assert [(t.track_id, a) for t, a in ev.entered] == [(1, "sofa")]
    assert b.tracks[1].entered["sofa"] == ts(1)
    ev = b.observe(frame("barber-c", 2, {1: C_DOOR}), POLICY)
    assert [(t.track_id, a) for t, a in ev.left] == [(1, "sofa")] and "sofa" not in b.tracks[1].entered


def test_paired_zone_reports_the_pair_and_the_seat_maps_to_its_chair_area():
    b = left()
    b.observe(frame("barber-l", 0, {7: L_EDGE}), POLICY)
    assert b.tracks[7].areas == {"ch1"} and b.tracks[7].seat is None
    b.observe(frame("barber-l", 1, {7: L_SEAT}), POLICY)
    assert b.tracks[7].seat == "ch1"


def test_a_reclined_body_whose_anchor_falls_below_the_seat_still_sits_in_it():
    # 2026-09-24 (Task 16 tune): seated customers' 80% anchor lands on the knees/footrest,
    # below the seat polygon; the box's middle is inside it.
    b = left()
    b.observe(frame("barber-l", 0, {7: (0.6, 0.83)}), POLICY)   # anchor below the seat, box middle inside
    assert b.tracks[7].seat == "ch1"


def test_door_crossing_must_clear_the_dead_band():
    b = centre()
    b.observe(frame("barber-c", 0, {1: C_STREET}), POLICY)
    ev = b.observe(frame("barber-c", 1, {1: C_DOOR}), POLICY)
    assert ev.crossed_in == [b.tracks[1]]
    ev = b.observe(frame("barber-c", 2, {1: (0.5, 0.075)}), POLICY)   # feet 0.005 inside: dead band
    assert not ev.crossed_out
    ev = b.observe(frame("barber-c", 3, {1: C_STREET}), POLICY)
    assert ev.crossed_out == [b.tracks[1]]


def test_first_seen_just_inside_the_door_is_flagged():
    b = centre()
    b.observe(frame("barber-c", 0, {1: (0.5, 0.1)}), POLICY)   # feet 0.03 inside
    assert b.tracks[1].appeared_at_door


def test_still_needs_the_whole_window_within_the_radius():
    b = left()
    for s in range(3):
        b.observe(frame("barber-l", s, {7: L_SEAT}), POLICY)
    assert not b.tracks[7].still(3, 0.02)          # seen for 2 s only
    b.observe(frame("barber-l", 3, {7: L_SEAT}), POLICY)
    assert b.tracks[7].still(3, 0.02)
    b.observe(frame("barber-l", 4, {7: (0.65, 0.65)}), POLICY)
    assert not b.tracks[7].still(3, 0.02)


def boxes(b, script):
    """script: [(s, (left, top, w, h) normalized)] for track 7 on the left camera."""
    for s, (x, y, w, h) in script:
        b.observe(Frame("barber-l", ts(s), (TrackObs(7, (x * 1000, y * 1000, w * 1000, h * 1000)),)), POLICY)


def test_the_still_tolerance_grows_with_the_box():
    # 2026-09-24 (Task 16 tune): a barber working around a seated customer shakes a big box
    # by more than 0.02 without the customer moving.
    b = left()
    boxes(b, [(0, (0.5, 0.3, 0.2, 0.5)), (1, (0.54, 0.3, 0.2, 0.5)), (2, (0.5, 0.3, 0.2, 0.5)),
              (3, (0.52, 0.3, 0.2, 0.5))])
    assert not b.tracks[7].still(3, 0.02)
    assert b.tracks[7].still(3, 0.02, 0.1)          # 0.1 x box height 0.5 = 0.05


def test_stillness_is_measured_at_the_middle_of_the_box():
    # the box's bottom comes and goes behind the barber: the anchor moves, the middle much less
    b = left()
    boxes(b, [(s, (0.5, 0.3, 0.2, 0.3 if s % 2 else 0.36)) for s in range(4)])
    assert b.tracks[7].still(3, 0.035)              # middle moves 0.03, the anchor 0.048


def test_staff_vote_is_undecided_until_enough_samples():
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    for s, p in enumerate([0.9, 0.8]):
        b.add_appearance(staff_app("barber-l", 7, s, p))
    assert b.tracks[7].staff(3, 0.6) is None
    b.add_appearance(staff_app("barber-l", 7, 2, 0.1))
    assert b.tracks[7].staff(3, 0.6) is True       # mean 0.6
    b.add_appearance(staff_app("barber-l", 7, 3, 0.1))
    assert b.tracks[7].staff(3, 0.6) is False


def test_a_track_seated_seat_dwell_s_latches_as_a_customer_and_keeps_it_standing():
    b = left()
    for s in range(3):
        b.observe(frame("barber-l", s, {7: L_SEAT}), POLICY)
        b.add_appearance(staff_app("barber-l", 7, s, 0.8))
    for s in range(3, 11):
        b.observe(frame("barber-l", s, {7: L_SEAT}), POLICY)
    t = b.tracks[7]
    assert t.customer and staff_vote(t, POLICY) is False
    b.observe(frame("barber-l", 11, {7: L_EDGE}), POLICY)
    assert staff_vote(t, POLICY) is False                # stood up: still a customer


def test_a_vote_decided_outside_a_seat_never_latches():
    b = left()
    b.observe(frame("barber-l", 0, {7: L_EDGE}), POLICY)
    for s in range(3):
        b.add_appearance(staff_app("barber-l", 7, s, 0.75))
    for s in range(1, 22):                              # staff_outside_confirm_s (20 s) outside
        b.observe(frame("barber-l", s, {7: L_EDGE}), POLICY)
    for s in range(22, 40):
        b.observe(frame("barber-l", s, {7: L_SEAT}), POLICY)
    t = b.tracks[7]
    assert t.staff_outside and not t.customer and staff_vote(t, POLICY) is True
    assert not seat_ok(t, POLICY)


def test_a_brief_staff_vote_outside_a_seat_does_not_latch_staff_outside():
    # 2026-09-26 (Task 5a): a caped customer's box misses the seat for a frame while his vote
    # says staff; one frame must not rule him out of the seat for good
    policy = dict(POLICY, staff_outside_confirm_s=3.0)
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT}), policy)
    for s in range(3):
        b.add_appearance(staff_app("barber-l", 7, s, 0.75))
    b.observe(frame("barber-l", 1, {7: L_EDGE}), policy)
    b.observe(frame("barber-l", 3, {7: L_EDGE}), policy)       # 2 s outside: not confirmed
    for s in range(4, 20):
        b.observe(frame("barber-l", s, {7: L_SEAT}), policy)
    t = b.tracks[7]
    assert not t.staff_outside and t.customer


def test_a_staff_vote_outside_a_seat_latches_once_it_lasts_the_confirm_time():
    policy = dict(POLICY, staff_outside_confirm_s=3.0)
    b = left()
    b.observe(frame("barber-l", 0, {7: L_EDGE}), policy)
    for s in range(3):
        b.add_appearance(staff_app("barber-l", 7, s, 0.75))
    for s in (1, 2, 3):
        b.observe(frame("barber-l", s, {7: L_EDGE}), policy)
    assert not b.tracks[7].staff_outside
    b.observe(frame("barber-l", 4, {7: L_EDGE}), policy)       # outside with a staff vote 1 -> 4 s
    assert b.tracks[7].staff_outside


def test_a_seat_frame_restarts_the_staff_outside_count():
    policy = dict(POLICY, staff_outside_confirm_s=3.0)
    b = left()
    b.observe(frame("barber-l", 0, {7: L_EDGE}), policy)
    for s in range(3):
        b.add_appearance(staff_app("barber-l", 7, s, 0.75))
    for s, pt in [(1, L_EDGE), (2, L_EDGE), (3, L_SEAT), (4, L_EDGE), (5, L_EDGE)]:
        b.observe(frame("barber-l", s, {7: pt}), policy)
    assert not b.tracks[7].staff_outside


def test_reid_folds_into_a_unit_mean():
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    b.add_appearance(reid_app("barber-l", 7, 0, (1, 0)))
    b.add_appearance(reid_app("barber-l", 7, 1, (0, 1)))
    assert cosine(b.tracks[7].reid, unit((1, 1))) > 0.999 and b.tracks[7].reid_n == 2


def test_appearance_for_an_unknown_track_is_ignored():
    assert left().add_appearance(staff_app("barber-l", 99, 0, 0.5)) is None


def test_expire_uses_the_cameras_own_latest_frame():
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT, 8: L_EDGE}), POLICY)
    b.observe(frame("barber-l", 1.5, {8: L_EDGE}), POLICY)
    b.observe(frame("barber-l", 3, {8: L_EDGE}), POLICY)
    assert [t.track_id for t in b.expire(2.0)] == [7] and set(b.tracks) == {8}


def test_after_a_gap_in_the_cameras_frames_every_track_is_over_and_an_id_back_is_a_new_track():
    # 2026-09-26 (final B review C1a): NvDCF ids restart with the sidecar (useUniqueID 0)
    b = left()
    for s in range(0, 15):
        b.observe(frame("barber-l", s, {7: L_SEAT, 9: L_AISLE}), POLICY)
    old, other = b.tracks[7], b.tracks[9]
    old.person, old.staff_samples = "p1", deque([0.1, 0.1, 0.1])
    assert old.customer
    ev = b.observe(frame("barber-l", 40, {7: L_EDGE}), POLICY)       # no frames 15..39: a restart
    new = b.tracks[7]
    assert ev.ended == [old, other] and ev.new == [new] and new is not old and set(b.tracks) == {7}
    assert (new.person, new.customer, list(new.staff_samples), new.first_seen) == (None, False, [], ts(40))
    ev = b.observe(frame("barber-l", 41, {7: L_EDGE}), POLICY)       # within track_gone_s: the same track
    assert ev.ended == [] and b.tracks[7] is new


def test_the_staff_vote_with_no_minimum_and_no_samples_is_undecided():
    # 2026-09-26 (final B review I4): staff_min_samples 0 divided by zero every cycle
    t = left().observe(frame("barber-l", 0, {7: L_SEAT}), POLICY).new[0]
    assert t.staff(0, 0.6) is None
    assert seat_ok(t, dict(POLICY, staff_min_samples=0)) is True


def test_a_track_back_after_track_gone_s_in_a_running_stream_is_left_to_expire():
    # the tracker's own reacquisition while frames keep coming: expire() decides, per cycle
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    b.observe(frame("barber-l", 1.5, {}), POLICY)
    ev = b.observe(frame("barber-l", 2.5, {7: L_SEAT}), POLICY)
    assert ev.ended == [] and b.tracks[7].first_seen == ts(0)


def _samples(ps, customer=False):
    b = left()
    b.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    for s, p in enumerate(ps):
        b.add_appearance(staff_app("barber-l", 7, s, p))
    b.tracks[7].customer = customer
    return b.tracks[7]


def test_staff_window_votes_on_the_last_crops_only():
    # 2026-09-29: a tracker id swap hands a customer's track to a barber; with the 15-crop window
    # the customer's old low scores keep the vote False for ~11 s (dev/staff_hold_rules.py)
    t = _samples([0.05] * 10 + [0.95] * 4)
    assert staff_vote(t, POLICY) is False                        # today: mean of all 14 = 0.31
    assert staff_vote(t, {**POLICY, "staff_window": 5}) is True  # last 5: 0.77
    assert staff_vote(_samples([0.95, 0.05]), {**POLICY, "staff_window": 5}) is None   # still too few crops


def test_a_latched_customer_track_votes_staff_only_on_a_run_of_sure_staff_crops():
    lat = {**POLICY, "staff_latched_release": 5}
    t = _samples([0.1] * 5 + [0.95] * 5, customer=True)
    assert staff_vote(t, POLICY) is False                        # today: a latched track is never staff
    assert staff_vote(t, lat) is True
    assert staff_vote(_samples([0.1] * 5 + [0.95] * 4 + [0.85], customer=True), lat) is False   # one below 0.9
    assert staff_vote(_samples([0.95] * 4, customer=True), lat) is False                       # too few


def test_the_window_also_rules_seats_and_the_staff_outside_latch():
    # one vote everywhere: a released track must not look like a customer to the seat rules
    t = _samples([0.05] * 10 + [0.95] * 4)
    assert seat_ok(t, POLICY) is True
    assert seat_ok(t, {**POLICY, "staff_window": 5, "seat_staff_threshold": 0.5}) is False
