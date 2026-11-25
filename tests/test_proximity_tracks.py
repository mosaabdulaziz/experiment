"""proximity.tracks: moving vs stationary vehicles, and putting back hidden feet."""
import pytest

from perception.modules.proximity.scale import PersonScale
from perception.modules.proximity.shapes import PersonShape
import random

from perception.modules.proximity.tracks import FootAnchors, VehicleMotion, co_motion, resolve_foot


def test_a_vehicle_that_holds_still_is_stationary_and_one_that_drives_is_moving():
    m = VehicleMotion()
    for i in range(20):
        t = i * 0.1
        m.observe(1, t, (500 + (i % 2) * 3, 400), 10000)       # parked: 3 px of jitter
        m.observe(2, t, (100 + 20 * i, 400), 10000)             # driving: 200 px/s
    assert m.state(1, 1.9, window_s=2.0, motion_px=50) == "stationary"
    assert m.state(2, 1.9, window_s=2.0, motion_px=50) == "moving"


def test_too_short_a_history_has_no_state():
    m = VehicleMotion()
    for t in (0.0, 0.1, 0.2):
        m.observe(1, t, (100 + 100 * t, 400), 10000)
    assert m.state(1, 0.2, 2.0, 50) is None
    assert m.state(99, 0.2, 2.0, 50) is None


def test_a_mask_merge_does_not_make_a_parked_vehicle_move():
    m = VehicleMotion()
    for i in range(20):
        ref, area = ((700, 400), 25000) if i == 10 else ((500, 400), 10000)   # merged with a passer-by once
        m.observe(1, i * 0.1, ref, area)
    assert m.state(1, 1.9, 2.0, 50) == "stationary"


def test_a_vehicle_not_seen_for_a_while_is_forgotten():
    m = VehicleMotion(keep_s=10.0)
    for i in range(10):
        m.observe(1, i * 0.1, (100 + 30 * i, 400), 10000)
    m.forget(20.0)
    assert m.state(1, 20.0, 30.0, 50) is None


def test_a_one_frame_jump_does_not_make_a_parked_vehicle_move():
    m = VehicleMotion()
    for i in range(20):
        ref, area = ((560, 400), 11000) if i == 10 else ((500, 400), 10000)   # half-hidden once: the point jumps
        m.observe(1, round(i * 0.15, 3), ref, area)
        if i >= 10:
            assert m.state(1, round(i * 0.15, 3), 2.0, 50) == "stationary"


def test_slow_steady_motion_reads_moving():
    m = VehicleMotion()
    for i in range(20):
        m.observe(1, round(i * 0.15, 3), (100 + 40 * i * 0.15, 400), 10000)     # 40 px/s: 0.4 ph/s at 100 px
    assert m.state(1, 2.85, 2.0, 50) == "moving"


def _drive_vehicle(m, vid=1, until=10):
    for i in range(until):
        m.observe(vid, round(i * 0.15, 3), (100 + 150 * i * 0.15, 400), 10000, link_px=100)


def test_a_new_id_where_a_vehicle_was_just_lost_takes_over_its_history():
    m = VehicleMotion()
    _drive_vehicle(m)
    m.observe(2, 1.5, (100 + 150 * 1.5, 400), 10000, link_px=100)          # the tracker re-identified it
    assert m.state(2, 1.5, 2.0, 50) == "moving" and m.state(1, 1.5, 2.0, 50) is None


def test_a_new_id_far_from_any_lost_vehicle_starts_fresh():
    m = VehicleMotion()
    _drive_vehicle(m)
    m.observe(2, 1.5, (900, 100), 10000, link_px=100)
    assert m.state(2, 1.5, 2.0, 50) is None and m.state(1, 1.5, 2.0, 50) == "moving"


# the box must move too (Ticket 3, 2026-11-01): a parked truck whose mask merges with a neighbour now and then moves
# its reference point, not its box (C189)

FRAME = (1280, 720)


def _gated():
    return VehicleMotion(frame_size=FRAME, box_motion=0.5)


def _glitch(m, i, vid=1, box=(300, 250, 400, 250), link_px=0.0):
    """Frame i of a parked truck whose mask merges with its neighbour in 4 frames of every 13: its point jumps 300 px."""
    jump = i % 13 >= 9
    m.observe(vid, round(i * 0.15, 3), (800 if jump else 500, 500), 10000, link_px=link_px, box=box)
    return jump


def test_a_parked_truck_whose_point_jumps_but_whose_box_stays_is_parked():
    plain, gated = VehicleMotion(), _gated()
    for i in range(14):
        _glitch(plain, i)
        _glitch(gated, i)
    assert plain.state(1, 1.95, 2.0, 50) == "moving" and gated.state(1, 1.95, 2.0, 50) == "stationary"


def test_a_merge_that_stretches_one_edge_of_the_box_does_not_move_it():
    m = _gated()
    for i in range(14):
        jump = i % 13 >= 9
        m.observe(1, round(i * 0.15, 3), (800 if jump else 500, 500), 10000, box=(300, 250, 400, 380 if jump else 250))
    assert m.state(1, 1.95, 2.0, 50) == "stationary"


def test_a_truck_driving_sideways_moves_two_edges_and_is_moving():
    m = _gated()
    for i in range(20):
        x = 100 + 150 * i * 0.15
        m.observe(1, round(i * 0.15, 3), (x + 200, 500), 10000, box=(x, 250, 400, 250))
    assert m.state(1, 2.85, 2.0, 50) == "moving"


def test_a_truck_entering_the_picture_moves_one_free_edge_and_is_moving():
    m = _gated()
    for i in range(20):
        right = 100 + 150 * i * 0.15                     # its box cut off at the left border
        m.observe(1, round(i * 0.15, 3), (right / 2, 500), 10000, box=(0, 250, right, 250))
    assert m.state(1, 2.85, 2.0, 50) == "moving"


def test_without_boxes_the_point_alone_decides():
    m = _gated()
    for i in range(14):
        m.observe(1, round(i * 0.15, 3), (800 if i % 13 >= 9 else 500, 500), 10000)
    assert m.state(1, 1.95, 2.0, 50) == "moving"


def test_a_new_id_taking_over_a_parked_truck_takes_its_box_history_too():
    m = _gated()
    for i in range(12):
        _glitch(m, i, link_px=100)
    _glitch(m, 12, vid=2, link_px=100)                  # re-identified where it was lost (a jumped point)
    assert m.state(2, 1.8, 2.0, 50) == "stationary" and m.state(1, 1.8, 2.0, 50) is None


# a steady moving / parked state (Ticket 5, 2026-11-04): on warehouse camera C26 a manoeuvring forklift switched
# between moving, parked and unknown 18 times in 20 s -- grey and blue in the preview, and never alerting meanwhile


def _pause(m, pause_s, vid=1):
    """A vehicle driving 200 px/s that stops for `pause_s` at t = 3 s and drives on: its state after each frame."""
    x, states = 100.0, []
    for i in range(60):
        t = round(i * 0.15, 3)
        if i and not 3.0 <= t < 3.0 + pause_s:
            x += 200 * 0.15
        m.observe(vid, t, (x, 400), 10000)
        states.append(m.state(vid, t, 2.0, 50))
    return states


def test_a_vehicle_that_pauses_for_a_moment_stays_moving():
    plain, held = _pause(VehicleMotion(), 1.5), _pause(VehicleMotion(parked_hold_s=1.0), 1.5)
    assert "stationary" in plain                        # the window alone reads the pause as parked for 0.6 s
    assert "stationary" not in held


def test_a_vehicle_that_stops_reads_parked_once_it_has_looked_parked_for_the_hold():
    plain, held = _pause(VehicleMotion(), 3.0), _pause(VehicleMotion(parked_hold_s=1.0), 3.0)
    first = plain.index("stationary")
    hold_frames = round(1.0 / 0.15)                     # 1 s of stationary verdicts, at 0.15 s a frame
    assert held[first:first + hold_frames] == ["moving"] * hold_frames
    assert held[first + hold_frames + 1] == "stationary"


def test_a_parked_vehicle_that_drives_off_reads_moving_without_waiting():
    def drive_off(m):
        states = []
        for i in range(50):
            t = round(i * 0.15, 3)
            m.observe(1, t, (500 + 200 * max(0.0, t - 3.0), 400), 10000)        # parked 3 s, then 200 px/s
            states.append(m.state(1, t, 2.0, 50))
        return states
    plain, held = drive_off(VehicleMotion()), drive_off(VehicleMotion(parked_hold_s=1.0))
    assert held.index("moving") == plain.index("moving")


def _parked_then_lost(m, vid=1, link_px=0.0):
    for i in range(21):                                 # parked at x = 500 for 3 s, then the detector loses it
        m.observe(vid, round(i * 0.15, 3), (500, 400), 10000, link_px=link_px)
    assert m.state(vid, 3.0, 2.0, 50) == "stationary"


def test_a_vehicle_back_from_a_detector_dropout_where_it_was_keeps_its_state():
    m = VehicleMotion()
    _parked_then_lost(m)
    m.observe(1, 6.0, (510, 400), 10000)                # 3 s later, back in the same place: too little history to tell
    assert m.state(1, 6.0, 2.0, 50) == "stationary"


def test_a_parked_truck_back_from_a_dropout_with_its_point_shifted_stays_parked():
    # C189, Ticket 5 evaluation: a parked truck re-detected partly hidden by workers came back with its ground point
    # elsewhere; reading that as driving made critical false alarms. Until its window can tell, it keeps its state.
    m = VehicleMotion()
    _parked_then_lost(m)
    m.observe(1, 6.0, (800, 400), 10000)
    assert m.state(1, 6.0, 2.0, 50) == "stationary"


def test_a_re_identified_vehicle_keeps_its_state():
    m = VehicleMotion()
    _parked_then_lost(m, link_px=100)
    m.observe(2, 4.9, (505, 400), 10000, link_px=100)   # a new id where it was lost: too few samples in its window
    assert m.state(2, 4.9, 2.0, 50) == "stationary"


def test_co_motion_tells_a_rider_from_a_pedestrian():
    rnd = random.Random(7)
    vehicle = [(round(i * 0.15, 3), 100.0 + 30 * i, 400.0) for i in range(14)]
    driver = [(t, x + 50 + rnd.uniform(-35, 35), 300.0 + rnd.uniform(-35, 35)) for t, x, _y in vehicle]  # jittery head
    still = [(t, 400.0, 300.0) for t, _x, _y in vehicle]
    turning = [(t, 600.0 - 30 * i, 300.0) for i, (t, _x, _y) in enumerate(vehicle)]   # the cab swings the other way
    now = vehicle[-1][0]
    travel, share = co_motion(driver, vehicle, now, 2.0)
    assert 0.8 < travel < 1.2 and share > 0.8
    travel, _share = co_motion(still, vehicle, now, 2.0)
    assert travel < 0.1
    travel, share = co_motion(turning, vehicle, now, 2.0)
    assert travel > 0.8 and share < 0
    assert co_motion(driver[-3:], vehicle, now, 2.0) is None                  # too little history


def test_co_motion_compares_the_same_span_of_both_tracks():
    vehicle = [(round(i * 0.15, 3), 100.0 + 30 * i, 400.0) for i in range(14)]
    rider = [(t, x + 50, 300.0) for t, x, _y in vehicle][-7:]               # a new track id: 0.9 s of history
    travel, share = co_motion(rider, vehicle, vehicle[-1][0], 2.0)
    assert 0.9 < travel < 1.1 and 0.9 < share < 1.1


def test_a_new_id_does_not_take_the_history_of_a_vehicle_still_in_view():
    m = VehicleMotion()
    _drive_vehicle(m)
    # this frame: a new id (the forklift's load) listed before the forklift itself, right next to it
    m.observe(2, 1.5, (100 + 150 * 1.5, 400), 3000, link_px=100, present={1, 2})
    m.observe(1, 1.5, (100 + 150 * 1.5, 400), 10000, link_px=100, present={1, 2})
    assert m.state(1, 1.5, 2.0, 50) == "moving" and m.state(2, 1.5, 2.0, 50) is None


def _scale():
    s = PersonScale(min_samples=50)
    s.restore({"a": 0.5, "b": -50.0, "n": 100})                 # h = 0.5 v - 50
    return s


def _person(head_y, foot_y, x=300.0, touches=False):
    return PersonShape((x, foot_y), (x, head_y), touches)


def test_visible_feet_are_used_as_seen():
    f = resolve_foot(7, 0.0, _person(300, 500), _scale(), FootAnchors())
    assert (f.point, f.hidden, f.source) == ((300.0, 500.0), False, "visible")


def test_hidden_feet_come_back_from_the_persons_height_and_the_heads_column():
    a, s = FootAnchors(), _scale()
    resolve_foot(7, 0.0, _person(300, 500), s, a)
    f = resolve_foot(7, 1.0, _person(310, 420, x=340), s, a)     # 110 px visible of ~210: behind something
    # their height (1.0 of the map's) at the head's row 310: 210 px; the column moved with the head
    assert f.hidden and f.source == "track" and f.point == pytest.approx((340.0, 520.0))


def test_without_a_recent_track_the_head_row_and_scale_map_place_the_feet():
    a, s = FootAnchors(max_age_s=4.0), _scale()
    resolve_foot(7, 0.0, _person(300, 500), s, a)
    f = resolve_foot(7, 40.0, _person(300, 400), s, a)           # last seen 40 s ago: no foot, no height ratio
    assert f.source == "head" and f.point == pytest.approx((300.0, 500.0))


def test_feet_below_the_frame_edge_are_hidden_even_without_a_scale_map():
    f = resolve_foot(1, 0.0, _person(600, 719, touches=True), PersonScale(), FootAnchors())
    assert f.hidden and f.source == "unrecovered" and f.point == (300.0, 719.0)


def test_without_a_scale_map_a_short_person_is_taken_as_seen():
    assert not resolve_foot(1, 0.0, _person(450, 500), PersonScale(), FootAnchors()).hidden


# Ticket 4: a hidden foot from the person's own height; feet hidden with under half the person in view are a guess

def _sighting(a, s, t, head_y, ratio, pid=7, x=300.0):
    """Person `pid` seen with the feet in view, `ratio` of the map's standing height tall."""
    return resolve_foot(pid, t, _person(head_y, head_y + ratio * s.standing_height_from_head(head_y), x=x), s, a)


@pytest.mark.parametrize("heads, hidden_head", [((300.0, 290.0, 280.0), 260.0),      # walking away: the head rises
                                                ((300.0, 310.0, 320.0), 340.0)])     # walking towards the camera
def test_a_hidden_foot_is_placed_by_the_persons_own_height_walking_away_or_towards(heads, hidden_head):
    a, s = FootAnchors(), _scale()
    for t, head_y in zip((0.0, 0.5, 1.0), heads):
        assert not _sighting(a, s, t, head_y, 0.9).hidden
    standing = s.standing_height_from_head(hidden_head)
    f = resolve_foot(7, 2.0, _person(hidden_head, hidden_head + 0.69 * standing), s, a)   # 0.69 of them in view
    assert f.hidden and f.trusted and f.source == "track"
    # the last foot moved with the head would land 18 px too low walking away (422 vs 404), 18 px too high towards
    assert f.point == pytest.approx((300.0, hidden_head + 0.9 * standing))


def test_a_persons_height_is_forgotten_30_s_after_their_feet_were_last_seen():
    a, s = FootAnchors(), _scale()
    _sighting(a, s, 0.0, 300.0, 0.9)
    a.forget(29.0)
    assert resolve_foot(7, 29.0, _person(300.0, 440.0), s, a).source == "track"      # 140 of 200 in view: hidden
    a.forget(31.0)
    assert resolve_foot(7, 31.0, _person(300.0, 440.0), s, a).source == "head"


def test_a_new_track_id_with_hidden_feet_gets_the_head_estimate():
    a, s = FootAnchors(), _scale()
    _sighting(a, s, 0.0, 300.0, 0.9, pid=7)
    f = resolve_foot(8, 1.0, _person(300.0, 440.0), s, a)                             # the tracker switched ids
    assert f.source == "head" and f.trusted and f.point == pytest.approx((300.0, 500.0))


@pytest.mark.parametrize("in_view, trusted", [(0.4, False), (0.6, True)])
def test_hidden_feet_are_trusted_only_with_half_the_person_in_view(in_view, trusted):
    f = resolve_foot(7, 0.0, _person(300.0, 300.0 + in_view * 200.0), _scale(), FootAnchors())  # standing: 200 px
    assert f.hidden and f.trusted is trusted and f.source == "head"


def test_feet_below_the_frame_edge_with_most_of_the_person_in_view_are_trusted():
    f = resolve_foot(7, 0.0, _person(420.0, 719.0, touches=True), _scale(), FootAnchors())   # 299 of 320 in view
    assert f.hidden and f.trusted


def test_without_a_scale_map_hidden_feet_stay_trusted():
    f = resolve_foot(1, 0.0, _person(600, 719, touches=True), PersonScale(), FootAnchors())
    assert f.hidden and f.trusted
