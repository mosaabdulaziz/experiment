from domain import Frame, TrackObs, dt
from door import DoorCounter
from policy import DEFAULT_POLICY
from zones import DoorLine, FrameGeometry

P = dict(DEFAULT_POLICY)
GEO = FrameGeometry(1000, 1000)
LINE = DoorLine.from_zone({"id": "c-door", "points": [[0.4, 0.5], [0.6, 0.5]], "inside": [0.5, 0.8]})


def at(tid, x, y):         # a box whose bottom-centre is (x, y) normalised
    return TrackObs(tid, (x * 1000 - 20, y * 1000 - 100, 40, 100), 0.9, None)


def walk(dc, tid, ys, staff=None, x=0.5, t0=100.0):
    out = []
    for i, y in enumerate(ys):
        out += dc.observe(Frame("barber-c", t0 + i, (at(tid, x, y),)), lambda t: staff, P)
    return out


def gone(dc, t, staff=None):    # an empty frame at time t: the clock for tracks that vanished
    return dc.observe(Frame("barber-c", t, ()), lambda t: staff, P)


def test_walking_in_and_out():
    # 2026-10-04 (Task 14b): a crossing is confirmed after ce_door_confirm_s, or when the track is
    # lost; these walks end, so the empty frame afterwards is what emits them.
    dc = DoorCounter(LINE, GEO)
    got = walk(dc, 1, [0.40, 0.45, 0.55, 0.60]) + gone(dc, 110.0)
    (cin,) = got
    assert cin.direction == "in" and not cin.staff and cin.at == dt(102.0)     # dated to the crossing frame
    got = walk(dc, 1, [0.60, 0.45, 0.40], t0=200.0) + gone(dc, 210.0)
    (cout,) = got
    assert cout.direction == "out" and cout.at == dt(201.0)


def test_staff_crossing_is_flagged():
    dc = DoorCounter(LINE, GEO)
    (c,) = walk(dc, 2, [0.4, 0.6]) + gone(dc, 110.0, staff=0.9)
    assert c.staff


def test_clean_walk_in_confirms_after_2s_on_the_new_side():
    dc = DoorCounter(LINE, GEO)
    got = walk(dc, 1, [0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75])    # crossing at t=102, held to 106
    (c,) = got
    assert c.direction == "in" and c.at == dt(104.0 - 2.0)
    assert walk(dc, 1, [0.8, 0.85], t0=107.0) == []               # already emitted, no repeat


def test_out_in_out_along_the_line_within_1s_is_one_crossing():
    dc = DoorCounter(LINE, GEO)
    seq = [(0.0, 0.55), (0.2, 0.45), (0.4, 0.55), (0.6, 0.45), (0.8, 0.55), (1.0, 0.45)]
    got = []
    for t, y in seq:
        got += dc.observe(Frame("barber-c", 100 + t, (at(9, 0.5, y),)), lambda t: None, P)
    got += gone(dc, 105.0)
    (c,) = got
    assert c.direction == "out" and c.at == dt(101.0)             # the last flip, which was never undone


def test_out_in_returning_to_the_start_side_is_no_crossing():
    dc = DoorCounter(LINE, GEO)
    got = []
    for t, y in [(0.0, 0.45), (0.2, 0.55), (0.4, 0.45), (0.6, 0.45)]:
        got += dc.observe(Frame("barber-c", 100 + t, (at(9, 0.5, y),)), lambda t: None, P)
    assert got + gone(dc, 105.0) == []


def test_a_crossing_whose_track_vanishes_is_emitted_after_the_lost_time():
    dc = DoorCounter(LINE, GEO)
    assert walk(dc, 1, [0.4, 0.6]) == []                          # crossing at t=101, last seen t=101
    assert gone(dc, 101.9) == []                                  # unseen < 1 s
    (c,) = gone(dc, 102.2)
    assert c.direction == "in" and c.at == dt(101.0)
    assert gone(dc, 120.0) == []


def test_leaving_the_extent_after_crossing_emits():
    dc = DoorCounter(LINE, GEO)
    assert walk(dc, 1, [0.4, 0.6]) == []
    (c,) = dc.observe(Frame("barber-c", 102.0, (at(1, 0.95, 0.6),)), lambda t: None, P)
    assert c.direction == "in" and c.at == dt(101.0)


def test_forget_drops_a_pending_crossing():
    dc = DoorCounter(LINE, GEO)
    walk(dc, 1, [0.4, 0.6])
    dc.forget({1})
    assert gone(dc, 110.0) == []


def test_standing_on_the_line_is_not_a_crossing():
    dc = DoorCounter(LINE, GEO)
    assert walk(dc, 3, [0.49, 0.51, 0.49, 0.51][:1] + [0.5] * 5) + gone(dc, 120.0) == []


def test_outside_the_doorway_is_ignored():
    dc = DoorCounter(LINE, GEO)
    assert walk(dc, 4, [0.4, 0.6], x=0.95) + gone(dc, 120.0) == []


def test_no_line_counts_nothing():
    assert walk(DoorCounter(None, GEO), 5, [0.4, 0.6]) == []


def test_leaving_the_extent_back_on_the_old_side_drops_the_pending_crossing():
    dc = DoorCounter(LINE, GEO)
    assert walk(dc, 1, [0.4, 0.6]) == []
    assert dc.observe(Frame("barber-c", 102.0, (at(1, 0.95, 0.4),)), lambda t: None, P) == []
    assert gone(dc, 110.0) == []


# ── 2026-10-07 review finding 4 ──────────────────────────────────────────────
def test_a_camera_frame_gap_does_not_commit_a_visible_tracks_pending_flip():
    """The track is in the next frame: a 1.5 s gap between frames is the camera's, not the track lost."""
    dc = DoorCounter(LINE, GEO)
    out = dc.observe(Frame("barber-c", 100.0, (at(1, 0.5, 0.45),)), lambda t: None, P)
    out += dc.observe(Frame("barber-c", 100.5, (at(1, 0.5, 0.56),)), lambda t: None, P)      # flip: pending
    out += dc.observe(Frame("barber-c", 102.0, (at(1, 0.5, 0.45),)), lambda t: None, P)      # 1.5 s later, back
    out += gone(dc, 110.0)
    assert out == []
