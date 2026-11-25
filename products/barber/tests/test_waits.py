import itertools

import numpy as np

from domain import Appearance, DoorCrossing, EngineUpdate, Frame, TrackObs, dt
from policy import DEFAULT_POLICY
from waits import WaitBook
from zones import FrameGeometry

P = dict(DEFAULT_POLICY)
G = FrameGeometry(1000, 1000)
T = 1_000_000.0


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ZONES = {"barber-c": [{"id": "c-sofa", "name": "sofa", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9)},
                      {"id": "c-row-l", "name": "left row", "type": "row", "points": sq(0.0, 0.5, 0.3, 0.9)}],
         "barber-l": [{"id": "l-waiting-1", "name": "w", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9)}]}
SOFA = (0.6, 0.4, 0.7, 0.8)      # lower part well inside the waiting zone; foot (0.65, 0.8)
NEAR = (0.62, 0.4, 0.72, 0.8)    # foot 0.02 from SOFA's
FAR = (0.75, 0.4, 0.85, 0.8)     # foot 0.15 from SOFA's, still in the waiting zone
ROWL = (0.1, 0.4, 0.2, 0.8)      # lower part inside the left row zone
WALK = (0.35, 0.1, 0.45, 0.4)    # in neither zone


def person(tid, where=SOFA, conf=0.9):
    x0, y0, x1, y1 = where
    return TrackObs(tid, (x0 * 1000, y0 * 1000, (x1 - x0) * 1000, (y1 - y0) * 1000), conf,
                    tuple(v for p in sq(*where) for v in p))


def book():
    ids, tickets = itertools.count(1), itertools.count(1)
    return WaitBook(ZONES, "barber-c", G, lambda: f"w{next(ids)}", lambda at: next(tickets))


def feed(b, t0, secs, tracks, staff=None, seated=(), crossings=(), step=0.5, cam="barber-c", policy=P, upd=None):
    """A frame every `step` s from t0 for `secs`, each followed by tick(); crossings go with the first frame."""
    upd = upd if upd is not None else EngineUpdate()
    staff = staff or {}
    for i in range(int(round(secs / step))):
        b.observe(Frame(cam, t0 + i * step, tuple(tracks)), lambda tid: staff.get(tid), set(seated),
                  list(crossings) if i == 0 else [], policy, upd)
        b.tick(policy, upd)
    return upd


def ended(upd):
    return [r for r in upd.waits.values() if r.ended_at is not None]


def test_a_wait_starts_after_the_threshold_dated_to_the_first_frame():
    b = book()
    feed(b, T, 8.5, [person(1)])
    assert not b.open
    u = feed(b, T + 8.5, 1.0, [person(1)])
    (w,) = b.open.values()
    assert (w.start, w.ticket, w.first_track, w.track) == (T, 1, 1, 1)
    (r,) = u.waits.values()
    assert r.started_at == dt(T) and r.ended_at is None and r.target_s == 900.0 and r.camera == "barber-c"


def test_a_passer_by_never_waits():
    b = book()
    feed(b, T, 6, [person(1)])
    feed(b, T + 6, 10, [person(1, WALK)])
    assert not b.open


def test_a_gap_restarts_the_threshold():
    b = book()
    feed(b, T, 8, [person(1)])
    feed(b, T + 8, 3, [])                       # out of sight > wc_dwell_gap_s
    feed(b, T + 11, 2, [person(1)])
    assert not b.open


def test_staff_low_confidence_and_seat_occupants_never_wait():
    b = book()
    feed(b, T, 12, [person(1), person(2, NEAR, conf=0.1), person(3, FAR)], staff={1: 0.95}, seated={3})
    assert not b.open


def test_other_cameras_are_ignored():
    b = book()
    feed(b, T, 12, [person(1)], cam="barber-l")
    assert not b.open


def test_holding_a_row_zone_ends_the_wait_dated_to_entry():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 1, [person(1, ROWL)])       # 1 s in a row: not yet
    feed(b, T + 11, 5, [person(1)])
    assert len(b.open) == 1
    u = feed(b, T + 16, 4, [person(1, ROWL)])
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at, r.wait_s) == ("chair", dt(T + 16), 16.0)
    assert not b.open


WASHZ = {"barber-c": ZONES["barber-c"] + [
    {"id": "c-wash", "name": "wash", "type": "washing", "points": sq(0.0, 0.0, 0.3, 0.3)},
    {"id": "c-wash-2", "name": "wash", "type": "washing", "points": sq(0.5, 0.7, 0.9, 1.0)}]}   # 2nd: over the sofa
WASH = (0.1, 0.0, 0.2, 0.3)      # lower part inside the first wash zone only


def test_holding_the_wash_ends_the_wait_like_a_row():
    b = WaitBook(WASHZ, "barber-c", G, lambda: "w1", lambda at: 1)
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 1, [person(1, WASH)])       # 1 s at the wash: not yet
    feed(b, T + 11, 5, [person(1)])             # back on the sofa (in the 2nd wash zone too): still waiting
    assert len(b.open) == 1
    u = feed(b, T + 16, 4, [person(1, WASH)])
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at, r.wait_s) == ("wash", dt(T + 16), 16.0)


def test_a_door_out_crossing_ends_the_wait_but_a_staff_crossing_doesnt():
    b = book()
    feed(b, T, 10, [person(1)])
    u = feed(b, T + 10, 0.5, [person(1)], crossings=[DoorCrossing(dt(T + 10), "out", "barber-c", 1, True)])
    assert not ended(u)
    u = feed(b, T + 12, 0.5, [], crossings=[DoorCrossing(dt(T + 12), "out", "barber-c", 1, False)])
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at, r.wait_s) == ("door", dt(T + 12), 12.0)


def test_a_lost_track_reconnects_to_a_new_nearby_one():
    b = book()
    feed(b, T, 10, [person(1)])                 # last seen T + 9.5
    feed(b, T + 10, 3, [])                      # on hold from T + 11.5
    (w,) = b.open.values()
    assert w.hold == "lost"
    u = feed(b, T + 13, 5, [person(2, NEAR)])
    (w,) = b.open.values()
    assert (w.track, w.first_track, w.start, w.hold) == (2, 1, T, None)
    assert not ended(u)


def test_an_overlapping_new_id_still_takes_over():
    b = book()
    feed(b, T, 9, [person(1)])                  # last frame T + 8.5: no wait yet
    feed(b, T + 9, 1, [person(1), person(2, NEAR)])   # the wait starts at T + 9; id 2 first seen T + 9
    feed(b, T + 10, 3, [person(2, NEAR)])       # id 1 gone (last seen T + 9.5); on hold at T + 11.5
    (w,) = b.open.values()
    assert (w.track, w.first_track) == (2, 1)


def test_a_far_new_track_doesnt_take_over_and_the_wait_counts_to_the_last_sighting():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 3, [])
    u = feed(b, T + 13, 12, [person(2, FAR)])
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at, r.wait_s, r.last_track_id) == ("lost", dt(T + 9.5), 9.5, 1)
    (w,) = b.open.values()                      # id 2 waited long enough for its own wait
    assert (w.first_track, w.start) == (2, T + 13)


def test_a_late_new_track_doesnt_take_over():
    b = book()
    feed(b, T, 10, [person(1)])
    u = feed(b, T + 10, 12, [])
    (r,) = ended(u)
    assert r.ended_by == "lost"
    feed(b, T + 22, 3, [person(2, NEAR)])
    assert not b.open


def test_an_already_waiting_track_doesnt_take_over():
    b = book()
    feed(b, T, 10, [person(1), person(2, NEAR)])
    assert len(b.open) == 2
    u = feed(b, T + 10, 15, [person(2, NEAR)])
    (r,) = ended(u)
    assert (r.ended_by, r.last_track_id) == ("lost", 1)
    (w,) = b.open.values()
    assert w.first_track == 2 and w.track == 2


def test_the_same_track_coming_back_resumes_its_wait():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 4, [])
    u = feed(b, T + 14, 3, [person(1)])
    (w,) = b.open.values()
    assert (w.track, w.hold, w.start) == (1, None, T) and not ended(u)


def test_an_id_swap_onto_staff_holds_and_reconnects():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 2, [person(1)], staff={1: 0.95})
    (w,) = b.open.values()
    assert w.hold == "staff" and w.last_seen == T + 9.5
    feed(b, T + 12, 3, [person(1), person(2, NEAR)], staff={1: 0.95})
    (w,) = b.open.values()
    assert (w.track, w.hold, w.start) == (2, None, T)


def test_an_id_swap_onto_staff_without_reconnection_counts_to_the_last_non_staff_frame():
    b = book()
    feed(b, T, 10, [person(1)])
    u = feed(b, T + 10, 12, [person(1)], staff={1: 0.95})
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at, r.wait_s) == ("lost", dt(T + 9.5), 9.5)


def test_the_long_wait_alert_opens_past_the_target_and_closes_at_the_end():
    p = dict(P, wait_target_s=20.0)
    b = book()
    u = feed(b, T, 25, [person(1)], policy=p)
    (a,) = u.alerts
    (w,) = b.open.values()
    assert (a.kind, a.visit_id, a.started_at, a.open) == ("excessive_wait", w.id, dt(T + 20), True)
    u = feed(b, T + 25, 4, [person(1, ROWL)], policy=p)
    (a,) = u.alerts
    assert a.ended_at == dt(T + 25)


def test_an_open_wait_refreshes_its_last_sighting_every_30_s():
    b = book()
    u = feed(b, T, 45, [person(1)])
    (r,) = u.waits.values()
    assert r.ended_at is None and r.last_seen_at == dt(T + 39)


def test_clear_drops_open_waits_and_set_zones_keeps_them():
    b = book()
    feed(b, T, 10, [person(1)])
    b.set_zones(ZONES, "barber-c")
    assert len(b.open) == 1
    b.clear()
    assert not b.open


def test_two_held_waits_reconnect_closest_pairs_first():
    b = book()
    feed(b, T, 10, [person(1, SOFA), person(2, FAR)])
    feed(b, T + 10, 2.5, [])
    feed(b, T + 12.5, 1, [person(3, (0.76, 0.4, 0.86, 0.8)), person(4, NEAR)])
    by_first = {w.first_track: w.track for w in b.open.values()}
    assert by_first == {1: 4, 2: 3}


def test_a_door_out_crossing_ends_a_wait_already_on_hold():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 3, [])
    (w,) = b.open.values()
    assert w.hold == "lost"
    u = feed(b, T + 13, 0.5, [], crossings=[DoorCrossing(dt(T + 12.5), "out", "barber-c", 1, False)])
    (r,) = ended(u)
    assert (r.ended_by, r.ended_at) == ("door", dt(T + 12.5))


def test_the_own_track_beats_a_newcomer_in_the_same_frame():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 2.5, [])
    u = feed(b, T + 12.5, 1, [person(1), person(2, NEAR)])
    (w,) = b.open.values()
    assert (w.track, w.hold) == (1, None) and not ended(u)
    u = feed(b, T + 13.5, 10, [person(1), person(2, NEAR)])
    assert sorted(x.first_track for x in b.open.values()) == [1, 2]
    assert not ended(u)


def test_a_fragment_that_vanishes_leaves_the_wait_to_the_original_track():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 2.5, [])
    feed(b, T + 12.5, 1, [person(2, NEAR)])
    u = feed(b, T + 13.5, 30, [person(1)])
    assert not ended(u)
    (w,) = b.open.values()
    assert (w.first_track, w.track, w.start) == (1, 1, T)


def test_a_staff_swapped_track_coming_back_non_staff_doesnt_resume():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 2, [person(1)], staff={1: 0.95})
    u = feed(b, T + 12, 12, [person(1)])
    (r,) = ended(u)
    assert r.ended_by == "lost" and r.wait_s == 9.5


def test_a_visible_earlier_holder_doesnt_take_an_open_wait_back_when_its_holder_blinks():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 3, [])
    feed(b, T + 13, 1, [person(2, NEAR)])               # 2 takes over
    feed(b, T + 14, 1, [person(1, WALK), person(2, NEAR)])
    feed(b, T + 15, 0.5, [person(1, WALK)])             # 2 misses one frame
    (w,) = b.open.values()
    assert w.track == 2
    feed(b, T + 15.5, 0.5, [person(1, WALK), person(2, NEAR)])
    u = feed(b, T + 16, 4, [person(1, WALK), person(2, ROWL)])
    (r,) = ended(u)
    assert r.ended_by == "chair" and r.last_track_id == 2


def test_an_earlier_holder_resumes_a_wait_held_for_staff():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 3, [])
    feed(b, T + 13, 1, [person(2, NEAR)])               # fragment takes over
    feed(b, T + 14, 1, [person(2, NEAR)], staff={2: 0.95})
    (w,) = b.open.values()
    assert w.hold == "staff"
    feed(b, T + 15, 1, [person(1)], staff={2: 0.95})
    (w,) = b.open.values()
    assert (w.track, w.hold, w.start) == (1, None, T)


def test_a_low_confidence_object_never_starts_a_wait_but_a_confident_track_does():
    b = book()
    feed(b, T, 12, [person(1, conf=0.55)])
    assert not b.open
    b = book()
    feed(b, T, 12, [person(1, conf=0.75)])
    (w,) = b.open.values()
    assert w.start == T


def test_the_start_confidence_is_a_median_not_a_per_frame_rule():
    b = book()
    upd = EngineUpdate()
    for i in range(24):
        conf = 0.3 if i % 4 == 3 else 0.8
        b.observe(Frame("barber-c", T + i * 0.5, (person(1, conf=conf),)), lambda tid: None, set(), [], P, upd)
        b.tick(P, upd)
    (w,) = b.open.values()
    assert w.start == T


def test_wt_conf_min_zero_switches_the_check_off():
    b = book()
    feed(b, T, 12, [person(1, conf=0.55)], policy={**P, "wt_conf_min": 0})
    (w,) = b.open.values()
    assert w.start == T


def test_a_track_whose_confidence_rises_starts_once_the_recent_median_reaches_it():
    b = book()
    feed(b, T, 10, [person(1, conf=0.5)])
    assert not b.open
    feed(b, T + 10, 3, [person(1, conf=0.8)])
    assert not b.open                           # recent median still 0.5 / mixed
    feed(b, T + 13, 3, [person(1, conf=0.8)])
    (w,) = b.open.values()
    assert T + 10 - P["wt_start_s"] <= w.start <= T + 10      # dated into the window that first passed, not to T


def test_a_long_low_confidence_dwell_does_not_date_a_wait_back_to_its_start():
    b = book()
    feed(b, T, 600, [person(1, conf=0.55)])
    assert not b.open
    feed(b, T + 600, 9.5, [person(1, conf=0.8)])
    (w,) = b.open.values()
    assert T + 600 - P["wt_start_s"] <= w.start <= T + 600 + 1


def test_the_dwell_confidences_reset_after_a_gap():
    b = book()
    feed(b, T, 8, [person(1, conf=0.55)])
    feed(b, T + 8, 5, [])                       # gone longer than wc_dwell_gap_s
    assert b._dwell[1].last_in < T + 8
    feed(b, T + 13, 0.5, [person(1, conf=0.8)])
    d = b._dwell[1]
    assert d.since == T + 13 and d.confs == [(T + 13, 0.8)]


def test_no_row_zone_logs_one_warning_per_state_change(caplog):
    import logging
    door = {"id": "c-door", "name": "door", "type": "entrance_line", "points": [[0.4, 0.5], [0.6, 0.5]], "inside": [0.5, 0.8]}
    no_row = {"barber-c": [ZONES["barber-c"][0], door]}    # the warning is for the camera with the door line
    with caplog.at_level(logging.WARNING, logger="barber.waits"):
        b = WaitBook(no_row, "barber-c", G, lambda: "w", lambda at: 1)
        b.set_zones(no_row, "barber-c")
        assert len(caplog.records) == 1
        b.set_zones(ZONES, "barber-c")
        b.set_zones(no_row, "barber-c")
        assert len(caplog.records) == 2
        WaitBook(ZONES, "barber-c", G, lambda: "w", lambda at: 1)
        assert len(caplog.records) == 2


# ── 2026-10-07 review finding 3: fragmentation ───────────────────────────────
def feed_run(b, t0, secs, tracks, staff=None, run=None, seated=(), step=0.5, policy=P, upd=None):
    """feed() with the engine's staff_run (latest staff samples in a row per track)."""
    upd = upd if upd is not None else EngineUpdate()
    staff, run = staff or {}, run or {}
    for i in range(int(round(secs / step))):
        b.observe(Frame("barber-c", t0 + i * step, tuple(tracks)), lambda tid: staff.get(tid), set(seated), [],
                  policy, upd, staff_run=lambda tid: run.get(tid, 0))
        b.tick(policy, upd)
    return upd


def test_one_odd_staff_sample_does_not_end_a_wait():
    b = book()
    feed_run(b, T, 10, [person(1)])
    (w,) = b.open.values()
    u = feed_run(b, T + 10, 1, [person(1)], staff={1: 0.98}, run={1: 1})       # one 0.98 among 0.01s
    u = feed_run(b, T + 11, 20, [person(1)], staff={1: 0.01}, upd=u)
    assert list(b.open.values()) == [w] and w.hold is None and w.tracks == [1] and not ended(u)


def test_staff_samples_in_a_row_still_mean_an_id_swap_onto_staff():
    b = book()
    feed_run(b, T, 10, [person(1)])
    u = feed_run(b, T + 10, 12, [person(1)], staff={1: 0.95}, run={1: 3})
    (r,) = ended(u)
    assert r.ended_by == "lost" and r.wait_s == 9.5


def test_a_lost_wait_is_rejoined_by_its_own_track():
    b = book()
    feed_run(b, T, 30, [person(1)])
    (w,) = b.open.values()
    u = feed_run(b, T + 30, 15, [])                                            # gone 15 s: ended lost
    assert [r.ended_by for r in ended(u)] == ["lost"]
    u = feed_run(b, T + 45, 2, [person(1)])
    (w2,) = b.open.values()
    assert w2 is w and w.hold is None and u.waits[w.id].ended_at is None and u.waits[w.id].ticket == w.ticket


def test_no_rejoin_after_the_window_or_from_a_chair():
    pol = {**P, "wt_rejoin_s": 20.0}
    b = book()
    feed_run(b, T, 30, [person(1)], policy=pol)
    feed_run(b, T + 30, 25, [], policy=pol)
    feed_run(b, T + 55, 12, [person(1)], policy=pol)                           # 25 s after: a new wait
    (w,) = b.open.values()
    assert w.id != "w1"
    b = book()
    feed_run(b, T, 30, [person(1)])
    feed_run(b, T + 30, 15, [])
    feed_run(b, T + 45, 2, [person(1, ROWL)], seated=(1,))                     # seen again, but in a chair
    assert not b.open


def test_a_newcomer_from_the_street_does_not_take_over_a_held_wait():
    zones = {"barber-c": ZONES["barber-c"] + [{"id": "c-door", "name": "door", "type": "entrance_line",
                                                "points": [[0.4, 0.45], [0.9, 0.45]], "inside": [0.65, 0.8]}]}
    ids, tickets = itertools.count(1), itertools.count(1)
    b = WaitBook(zones, "barber-c", G, lambda: f"w{next(ids)}", lambda at: next(tickets))
    feed_run(b, T, 10, [person(1)])
    feed_run(b, T + 10, 1, [person(1, (0.6, 0.1, 0.7, 0.47))])                 # walks to the door, inside
    u = feed_run(b, T + 11, 12, [person(2, (0.6, 0.1, 0.7, 0.43))])            # someone appears outside it
    (r,) = ended(u)
    assert r.ended_by == "lost" and r.last_track_id == 1


def test_a_held_wait_raises_no_alert_and_a_rejoined_one_alerts_from_the_rejoin():
    pol = {**P, "wait_target_s": 12.0}
    b = book()
    feed_run(b, T, 10, [person(1)], policy=pol)
    u = feed_run(b, T + 10, 15, [], policy=pol)                               # on hold past 12 s, then lost
    assert not u.alerts
    u = feed_run(b, T + 25, 2, [person(1)], policy=pol)                       # rejoined at T + 25
    (a,) = u.alerts
    assert a.open and a.started_at.timestamp() == T + 25


def test_a_silent_centre_camera_ends_open_waits_at_their_last_sighting():
    b = book()
    feed_run(b, T, 30, [person(1)])
    u = EngineUpdate()
    b.stale(T + 30 + 30, P, u)
    assert b.open
    b.stale(T + 30 + 61, P, u)
    (r,) = ended(u)
    assert not b.open and r.ended_by == "lost" and r.ended_at.timestamp() == T + 29.5


from waits import LEFT

P1 = {**P, "wt_end_mode": 1.0, "wt_out_s": 3.0}


def test_mode1_a_wait_ends_when_he_steps_out_dated_to_his_first_frame_outside():
    b = book()
    feed(b, T, 10, [person(1)], policy=P1)
    (w,) = b.open.values()
    u = feed(b, T + 10, 4, [person(1, WALK)], policy=P1)
    (r,) = ended(u)
    assert r.ended_by == LEFT and r.ended_at == dt(T + 10) and not b.open


def test_mode1_a_short_step_out_does_not_end_it():
    b = book()
    feed(b, T, 10, [person(1)], policy=P1)
    feed(b, T + 10, 2.5, [person(1, WALK)], policy=P1)
    feed(b, T + 12.5, 1, [person(1)], policy=P1)                # back in: the count restarts
    feed(b, T + 13.5, 2.5, [person(1, WALK)], policy=P1)        # out again, never 3 s continuous
    assert len(b.open) == 1 and not ended(feed(b, T + 16, 0.5, [person(1)], policy=P1))


def test_mode1_flickering_on_the_sofa_edge_never_ends_it():
    b = book()
    feed(b, T, 10, [person(1)], policy=P1)
    for k in range(15):
        feed(b, T + 10 + 2 * k, 1, [person(1, WALK)], policy=P1)
        feed(b, T + 11 + 2 * k, 1, [person(1)], policy=P1)
    assert len(b.open) == 1


def test_mode1_rows_and_the_wash_do_not_end_it_the_edge_does():
    b = book()
    feed(b, T, 10, [person(1)], policy=P1)
    u = feed(b, T + 10, 5, [person(1, ROWL)], policy=P1)
    (r,) = ended(u)
    assert r.ended_by == LEFT                              # not "chair": mode 1 ignores row zones


def test_mode1_a_lost_track_still_holds_and_reconnects():
    b = book()
    feed(b, T, 10, [person(1)], policy=P1)
    feed(b, T + 10, 3, [], policy=P1)                      # unseen 3 s: on hold
    feed(b, T + 13, 2, [person(2, NEAR)], policy=P1)       # a new track next to him takes it over
    (w,) = b.open.values()
    assert w.track == 2


def test_mode0_is_unchanged_by_the_edge():
    b = book()
    feed(b, T, 10, [person(1)])
    feed(b, T + 10, 10, [person(1, WALK)])
    assert len(b.open) == 1                                # mode 0 follows the track anywhere in view


P1R = {**P1, "wt_return_sim": 0.82, "wt_return_max_s": 1800.0, "wt_return_min_samples": 2.0,
       "wt_return_extra_s": 30.0, "wt_serve_s": 150.0}


def unit(*v):
    a = np.asarray(v, np.float32)
    return tuple(float(x) for x in a / np.linalg.norm(a))


ALICE, BOB = unit(1, 0, 0, 0), unit(0, 1, 0, 0)
ALICE2 = unit(0.95, 0.1, 0, 0)          # cosine with ALICE ~0.99


def sampled_many(b, t0, secs, people, policy=P1R):
    """people: [(tid, where, emb)], all in every frame, each with one re-ID sample a second (time only moves on)."""
    upd = EngineUpdate()
    for k in range(int(secs)):
        for tid, _, emb in people:
            b.on_appearance(Appearance("barber-c", tid, t0 + k, emb, 0.01))
        feed(b, t0 + k, 1, [person(tid, where) for tid, where, _ in people], policy=policy, upd=upd)
    return upd


def sampled(b, t0, secs, tid, where, emb, policy=P1R):
    return sampled_many(b, t0, secs, [(tid, where, emb)], policy=policy)


def leave(b, t0, tid, policy=P1R):
    return feed(b, t0, 5, [person(tid, WALK)], policy=policy)


def test_a_returning_customer_reopens_his_wait_by_re_id():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    (w,) = b.open.values()
    leave(b, T + 12, 1)
    feed(b, T + 17, 60, [], policy=P1R)                          # away a minute
    u = sampled(b, T + 77, 12, 7, SOFA, ALICE2)                   # a new track, same person
    (w2,) = b.open.values()
    assert (w2.id, w2.ticket, w2.start) == (w.id, w.ticket, T)    # the same wait, time away counted
    assert u.waits[w.id].ended_at is None


def test_a_different_person_starts_a_new_wait():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    sampled(b, T + 30, 12, 7, SOFA, BOB)
    (w2,) = b.open.values()
    assert w2.first_track == 7 and w2.ticket == 2


def test_his_own_track_back_on_the_sofa_reopens_without_re_id():
    b = book()
    feed(b, T, 12, [person(1)], policy=P1R)                      # no re-ID samples at all
    leave(b, T + 12, 1)
    feed(b, T + 17, 12, [person(1)], policy=P1R)
    (w,) = b.open.values()
    assert w.first_track == 1 and w.ticket == 1 and w.start == T


def test_too_few_samples_waits_wt_return_extra_s_then_starts_new():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    feed(b, T + 30, 20, [person(7)], policy=P1R)                 # no samples: past wt_start_s, still undecided
    assert not b.open
    feed(b, T + 50, 25, [person(7)], policy=P1R)                 # past wt_start_s + wt_return_extra_s
    (w2,) = b.open.values()
    assert w2.first_track == 7 and w2.start == T + 30            # dated back to his first frame on the sofa


def test_back_after_his_haircut_is_no_new_wait():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)                                          # wait ends at T + 12
    b.note_haircut("s1", T + 60)                                 # his haircut starts 48 s later: served
    sampled(b, T + 1500, 12, 9, SOFA, ALICE2)                    # back on the sofa after the haircut
    assert not b.open
    feed(b, T + 1512, 60, [person(9)], policy=P1R)
    assert not b.open                                            # that track never starts a wait


def test_one_reopen_at_a_time_a_second_look_alike_starts_a_new_wait():
    # track 7 wins because it is processed first (frame order), not because it matches better
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    sampled_many(b, T + 30, 12, [(7, SOFA, ALICE2), (8, FAR, ALICE)])    # both decide in the same frame
    assert sorted(w.first_track for w in b.open.values()) == [1, 8]     # 7 reopened wait #1, 8 is new
    assert sorted(w.ticket for w in b.open.values()) == [1, 2]


def test_no_alert_while_away_and_a_reopened_alert_waits_for_the_reopen():
    pol = {**P1R, "wait_target_s": 20.0}
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE, policy=pol)
    u = leave(b, T + 12, 1, policy=pol)
    u2 = feed(b, T + 17, 60, [], policy=pol)
    assert not [a for a in u2.alerts if a.open]
    u3 = sampled(b, T + 77, 12, 7, SOFA, ALICE2, policy=pol)
    opened = [a for a in u3.alerts if a.open]
    assert opened and all(a.started_at >= dt(T + 77) for a in opened)    # fires on the reopen, not backdated


def test_a_reopened_wait_runs_from_his_first_arrival():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    sampled(b, T + 77, 12, 7, SOFA, ALICE2)
    u = leave(b, T + 89, 7)
    (r,) = ended(u)
    assert r.started_at == dt(T) and r.wait_s == 89.0


def test_after_a_wait_camera_stall_the_same_people_reopen_their_waits():
    # Review Focus: stale() ends every wait lost; the same people come back under new track ids
    b = book()
    sampled_many(b, T, 12, [(1, SOFA, ALICE), (2, FAR, BOB)])
    assert sorted(w.ticket for w in b.open.values()) == [1, 2]
    b.stale(T + 12 + P1R["wt_stale_s"] + 1, P1R, EngineUpdate())
    assert not b.open
    t2 = T + 12 + P1R["wt_stale_s"] + 2
    sampled_many(b, t2, 12, [(11, SOFA, ALICE2), (12, FAR, BOB)])
    assert sorted((w.ticket, w.track) for w in b.open.values()) == [(1, 11), (2, 12)]


def test_still_one_sample_after_the_extra_window_is_a_new_wait_not_a_reopen():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    b.on_appearance(Appearance("barber-c", 7, T + 31, ALICE2, 0.01))      # the dwell's only sample
    feed(b, T + 30, 50, [person(7)], policy=P1R)
    (w2,) = b.open.values()
    assert w2.first_track == 7 and w2.ticket == 2 and w2.start == T + 30


def test_a_reopened_wait_does_not_steal_anothers_haircut():
    b = book()
    sampled_many(b, T, 12, [(1, SOFA, ALICE), (2, FAR, BOB)])
    feed(b, T + 12, 5, [person(1, WALK), person(2, FAR)], policy=P1R)     # W (ALICE) leaves: ends T + 12
    feed(b, T + 17, 30, [person(2, WALK)], policy=P1R)                    # V (BOB) leaves: ends T + 17
    sampled(b, T + 30, 12, 7, SOFA, ALICE2)                               # W comes back: reopened
    assert [w.ticket for w in b.open.values()] == [1]
    b.note_haircut("sv", T + 60)                                          # V's haircut; W's old end is not a candidate
    feed(b, T + 42, 1458, [person(7)], policy=P1R)                        # W sits on, his wait stays open
    sampled_many(b, T + 1500, 12, [(7, SOFA, ALICE2), (9, FAR, BOB)])     # V back after his haircut: no wait
    assert [w.ticket for w in b.open.values()] == [1]


def test_mode_0_buffers_no_samples_and_a_waits_samples_are_trimmed():
    b = book()
    b.on_appearance(Appearance("barber-c", 1, T, ALICE, 0.01))
    assert b._reid == {}
    sampled(b, T, 12, 1, SOFA, ALICE)
    (w,) = b.open.values()
    for k in range(0, 3000, 100):
        b.on_appearance(Appearance("barber-c", 1, T + 12 + k, ALICE, 0.01))
    assert w.reid and w.reid[-1][0] - w.reid[0][0] <= 600.0


def test_a_dropped_haircut_no_longer_makes_a_wait_served():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    b.note_haircut("s1", T + 60)
    b.drop_haircut("s1")                                         # deleted: too short to be a haircut
    sampled(b, T + 1500, 12, 9, SOFA, ALICE2)
    (w,) = b.open.values()
    assert w.ticket == 1


# ── final review I2 (2026-10-15): a running haircut is not his; a stall-ended wait is never served ──────────────
def book_sitting(running):
    """a book whose haircuts in `running` still have their customer in the chair"""
    ids, tickets = itertools.count(1), itertools.count(1)
    return WaitBook(ZONES, "barber-c", G, lambda: f"w{next(ids)}", lambda at: next(tickets),
                    sitting=lambda sid, since: sid in running)


def test_a_match_whose_linking_haircut_still_sits_reopens_the_wait():
    running = {"s1"}
    b = book_sitting(running)
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)                                          # his wait ends at T + 12 (a hand-off, say)
    b.note_haircut("s1", T + 60)                                 # someone else's haircut starts 48 s later
    u = sampled(b, T + 300, 12, 9, SOFA, ALICE2)                 # he is on the sofa while s1's customer sits
    (w,) = b.open.values()
    assert (w.ticket, w.start, w.track) == (1, T, 9) and u.waits[w.id].ended_at is None
    assert b._served_by(w.id, P1R) is None                       # open again: out of the haircut links


def test_back_after_his_haircut_once_its_sitting_is_over():
    b = book_sitting(set())                                      # s1's sitting is over (or it closed)
    sampled(b, T, 12, 1, SOFA, ALICE)
    leave(b, T + 12, 1)
    b.note_haircut("s1", T + 60)
    sampled(b, T + 1500, 12, 9, SOFA, ALICE2)
    assert not b.open                                            # back after his haircut: no wait


def test_a_wait_a_stall_ended_is_never_served_its_customer_reopens_it():
    b = book()
    sampled(b, T, 12, 1, SOFA, ALICE)
    u = EngineUpdate()
    b.stale(T + 12 + P1R["wt_stale_s"] + 1, P1R, u)              # the wait camera stalls: ended "stall"
    (r,) = ended(u)
    assert r.ended_by == "stall" and r.ended_at.timestamp() == T + 11.5
    b.note_haircut("s1", T + 60)                                 # a haircut starts inside the window
    sampled(b, T + 1500, 12, 9, SOFA, ALICE2)                    # the camera is back, he is still there
    (w,) = b.open.values()
    assert (w.ticket, w.start) == (1, T)                         # reopened, not "back after his haircut"


def test_mode0_a_stall_still_ends_waits_lost():
    b = book()
    feed_run(b, T, 30, [person(1)])
    u = EngineUpdate()
    b.stale(T + 30 + 61, P, u)
    assert [r.ended_by for r in ended(u)] == ["lost"] and b._ends

