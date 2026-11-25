from domain import FaceCandidate
from entry_faces import EntryFaces
from policy import DEFAULT_POLICY

P = {**DEFAULT_POLICY, "fr_after_s": 5.0}   # timings below are for a 5 s settle (the default is 30 since 2026-10-11)
T = 1_000_000.0


def face(track, ts, path=None, cam="barber-c"):
    return FaceCandidate(cam, track, ts, path or f"f/{track}-{ts}.jpg", (1.0,) + (0.0,) * 511, 0.9)


def test_an_entry_registers_its_tracks_faces_in_the_window_after_the_settle():
    b = EntryFaces("barber-c")
    for ts in (T - 10, T - 6, T - 1, T + 3, T + 7):
        b.add(face(7, ts))
    b.add(face(8, T))
    b.entry("e1", 7, T)
    assert b.due(T + 4.9, P) == []
    (reg,) = b.due(T + 5.0, P)
    assert reg.visit_id == "e1" and [f.ts for f in reg.faces] == [T - 6, T - 1, T + 3]
    assert reg.bodies == ()


def test_faces_of_tracks_that_never_cross_in_are_deleted():
    b = EntryFaces("barber-c")
    b.add(face(8, T))
    b.add(face(9, T, cam="barber-l"))               # a side camera's face: deleted at once
    assert b.drain_discards() == [f"f/9-{T}.jpg"]
    b.due(T + P["fr_keep_s"] + 1, P)
    assert b.drain_discards() == [f"f/8-{T}.jpg"]


def test_two_entries_close_together_keep_their_own_faces():
    b = EntryFaces("barber-c")
    b.add(face(7, T - 2))
    b.add(face(8, T - 1))
    b.entry("e1", 7, T)
    b.entry("e2", 8, T + 1)
    regs = {r.visit_id: [f.track_id for f in r.faces] for r in b.due(T + 10, P)}
    assert regs == {"e1": [7], "e2": [8]}


def test_faces_outside_the_window_are_deleted_when_the_entry_registers():
    b = EntryFaces("barber-c")
    b.add(face(7, T - 9))
    b.add(face(7, T - 2))
    b.entry("e1", 7, T)
    (reg,) = b.due(T + 5, P)
    assert [f.ts for f in reg.faces] == [T - 2] and b.drain_discards() == [f"f/7-{T - 9}.jpg"]


def test_clear_returns_pending_face_crops():
    b = EntryFaces("barber-c")
    b.add(face(7, T))
    b.entry("e1", 7, T)
    assert b.clear() == [f"f/7-{T}.jpg"] and b.due(T + 10, P) == []


# ── 2026-10-07 review finding 4: an id switch after the crossing ─────────────
from domain import Frame, TrackObs  # noqa: E402


def frame(ts, *tracks):
    """tracks: (track_id, foot_x, foot_y) -- a 0.1-wide box standing on that foot point (1000 x 1000 frame)."""
    return Frame("barber-c", ts, tuple(TrackObs(tid, ((x - 0.05) * 1000, (y - 0.3) * 1000, 100, 300), 0.9, None)
                                     for tid, x, y in tracks))


def foot(box):
    l, t, w, h = box
    return ((l + w / 2) / 1000, (t + h) / 1000)


def test_an_entry_follows_its_entrant_onto_a_new_track_id():
    b = EntryFaces("barber-c")
    b.observe(frame(T, (7, 0.5, 0.4)), foot, lambda tid: False)
    b.entry("e1", 7, T)
    b.observe(frame(T + 1, (7, 0.5, 0.45)), foot, lambda tid: False)
    b.observe(frame(T + 2.0, (9, 0.52, 0.47)), foot, lambda tid: False)     # 7 gone 1 s, 9 born next to it
    b.observe(frame(T + 3.0, (9, 0.52, 0.5)), foot, lambda tid: False)
    b.add(face(9, T + 3))
    b.add(face(9, T + 4.5))
    (reg,) = b.due(T + 5.0, P)
    assert [f.track_id for f in reg.faces] == [9, 9]


def test_no_successor_far_away_late_or_staff():
    for second, staff in (((9, 0.9, 0.9), False), ((9, 0.52, 0.47), True)):
        b = EntryFaces("barber-c")
        b.observe(frame(T, (7, 0.5, 0.4)), foot, lambda tid: False)
        b.entry("e1", 7, T)
        b.observe(frame(T + 1.0, (7, 0.5, 0.45)), foot, lambda tid: False)
        b.observe(frame(T + 2.0, second), foot, lambda tid: staff)
        b.add(face(9, T + 3))
        (reg,) = b.due(T + 5.0, P)
        assert reg.faces == ()
    b = EntryFaces("barber-c")
    b.observe(frame(T, (7, 0.5, 0.4)), foot, lambda tid: False)
    b.entry("e1", 7, T)
    b.observe(frame(T + 4.5, (9, 0.52, 0.47)), foot, lambda tid: False)     # born 4.5 s after 7 was last seen
    b.add(face(9, T + 4.6))
    (reg,) = b.due(T + 5.0, P)
    assert reg.faces == ()


def test_a_track_still_in_view_hands_nothing_to_a_neighbour():
    b = EntryFaces("barber-c")
    b.observe(frame(T, (7, 0.5, 0.4)), foot, lambda tid: False)
    b.entry("e1", 7, T)
    b.observe(frame(T + 1.0, (7, 0.5, 0.45), (9, 0.55, 0.45)), foot, lambda tid: False)   # two people side by side
    b.add(face(9, T + 2))
    (reg,) = b.due(T + 5.0, P)
    assert reg.faces == ()
