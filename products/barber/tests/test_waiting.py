import pytest

from domain import Frame, TrackObs
from policy import DEFAULT_POLICY, validate_policy_update
from waiting import WaitingCounter

P = dict(DEFAULT_POLICY)


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ZONES = {"barber-c": [{"id": "c-sofa", "name": "sofa", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9)}],
         "barber-l": [{"id": "l-waiting-1", "name": "w", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9)}],
         "barber-r": []}
INSIDE = tuple(v for p in sq(0.6, 0.4, 0.7, 0.8) for v in p)     # lower part well inside the zone
OUTSIDE = tuple(v for p in sq(0.1, 0.1, 0.2, 0.4) for v in p)


def person(tid, outline=INSIDE, conf=0.9):
    return TrackObs(tid, (0, 0, 1, 1), conf, outline)


def feed(wc, cam, t0, secs, tracks, staff=None, seated=(), step=0.5):
    staff = staff or {}
    for i in range(int(secs / step)):
        wc.observe(Frame(cam, t0 + i * step, tuple(tracks)), lambda tid: staff.get(tid), set(seated), P)


def run(wc, t0, secs, per_cam, step=1.0):
    """per_cam: {cam: tracks}; ticks once per second; returns the shown values."""
    out = []
    for i in range(int(secs / step)):
        t = t0 + i * step
        for cam, tracks in per_cam.items():
            wc.observe(Frame(cam, t, tuple(tracks)), lambda tid: None, set(), P)
        out.append(wc.tick(t, P))
    return out


def test_a_person_counts_after_the_dwell():
    wc = WaitingCounter(ZONES)
    feed(wc, "barber-c", 1000.0, 8.5, [person(1)])
    assert wc.counts["barber-c"] == 0
    feed(wc, "barber-c", 1008.5, 1.0, [person(1)])
    assert wc.counts["barber-c"] == 1


def test_walking_through_never_counts():
    wc = WaitingCounter(ZONES)
    feed(wc, "barber-c", 1000.0, 6, [person(1)])
    feed(wc, "barber-c", 1006.0, 10, [person(1, OUTSIDE)])
    assert wc.counts["barber-c"] == 0


def test_staff_low_conf_and_seated_dont_count():
    wc = WaitingCounter(ZONES)
    feed(wc, "barber-c", 1000.0, 12, [person(1), person(2, conf=0.1), person(3)], staff={1: 0.95}, seated={3})
    assert wc.counts["barber-c"] == 0


def test_dwell_restarts_after_a_gap():
    wc = WaitingCounter(ZONES)
    feed(wc, "barber-c", 1000.0, 8, [person(1)])
    feed(wc, "barber-c", 1008.0, 3, [])                  # out of sight > wc_dwell_gap_s
    feed(wc, "barber-c", 1011.0, 2, [person(1)])
    assert wc.counts["barber-c"] == 0


def test_side_cameras_are_ignored_when_a_centre_camera_exists():
    # was test_highest_camera_wins...: the max-of-three rule now applies only without a centre camera
    wc = WaitingCounter(ZONES, camera="barber-c")
    shown = run(wc, 1000.0, 20, {"barber-c": [person(1)], "barber-l": [person(1), person(2)],
                                  "barber-r": [person(5), person(6), person(7)]})
    assert shown[-1] == 1 and wc.raw == 1
    assert wc.counts["barber-l"] == 2                                     # still reported for diagnosis


def test_centre_stale_is_none_even_if_side_cameras_are_fresh():
    wc = WaitingCounter(ZONES, camera="barber-c")
    run(wc, 1000.0, 15, {"barber-c": [person(1)], "barber-l": [person(1)]})
    t = 1015.0 + P["wc_stale_s"] + 1
    wc.observe(Frame("barber-l", t, (person(1),)), lambda tid: None, set(), P)
    assert wc.tick(t, P) is None


def test_without_a_centre_camera_the_highest_camera_wins():
    wc = WaitingCounter(ZONES)
    shown = run(wc, 1000.0, 20, {"barber-c": [person(1)], "barber-l": [person(1), person(2)],
                                  "barber-r": [person(5), person(6), person(7)]})
    assert shown[-1] == 2


def test_camera_without_zone_is_ignored():
    wc = WaitingCounter(ZONES)  # fallback mode (no centre camera)
    shown = run(wc, 1000.0, 15, {"barber-c": [person(1)], "barber-r": [person(9)]})
    assert shown[-1] == 1


def test_rises_fast_falls_after_the_hold():
    wc = WaitingCounter(ZONES)
    run(wc, 1000.0, 15, {"barber-c": [person(1)]})
    up = run(wc, 1015.0, 20, {"barber-c": [person(1), person(2)]})
    assert up.index(2) == 9 + int(P["wc_up_s"])                           # 9 s dwell, then the wc_up_s rise hold
    down = run(wc, 1035.0, 30, {"barber-c": [person(1)]})
    assert down[int(P["wc_down_s"]) - 2] == 2                             # a lower value held < wc_down_s: no drop
    assert down[-1] == 1


def test_a_brief_dip_does_not_lower_the_number():
    wc = WaitingCounter(ZONES)
    run(wc, 1000.0, 15, {"barber-c": [person(1), person(2)]})
    dip = run(wc, 1015.0, 5, {"barber-c": [person(1)]})
    back = run(wc, 1020.0, 5, {"barber-c": [person(1), person(2)]})
    assert set(dip + back) == {2}


def test_all_stale_is_none():
    wc = WaitingCounter(ZONES)
    run(wc, 1000.0, 15, {"barber-c": [person(1)]})
    assert wc.tick(1000.0 + 15 + P["wc_stale_s"] + 1, P) is None


def test_waiting_policy_bounds():
    for key in ("wc_stale_s", "wc_zone_min", "wc_lower_frac"):
        with pytest.raises(ValueError):
            validate_policy_update({key: 0})
    with pytest.raises(ValueError):
        validate_policy_update({"wc_zone_min": 1.5})
    assert validate_policy_update({"wc_down_s": 0})["wc_down_s"] == 0
