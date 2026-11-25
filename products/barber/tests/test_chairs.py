from chairs import ChairBook
from shop_helpers import C_CHAIR, GEO, L_AISLE, L_EDGE, L_SEAT, POLICY, ZONES, frame, sq, staff_app, ts
from tracklets import TrackletBook, staff_vote


def left():
    return TrackletBook("barber-l", ZONES["barber-l"], GEO)


def play(book, chairs, script, apps=()):
    """script: [(s0, s1, {track: point})] -> [(s, ChairStart)] for every start reported."""
    starts, pending = [], list(apps)
    for s0, s1, tracks in script:
        for s in range(s0, s1):
            book.observe(frame("barber-l", s, tracks), POLICY)
            for a in [a for a in pending if a.ts <= ts(s)]:
                book.add_appearance(a)
                pending.remove(a)
            starts += [(s, st) for st in chairs.observe_seats("barber-l", list(book.tracks.values()), ts(s), POLICY)]
    return starts


def test_still_body_in_the_seat_starts_after_the_dwell_backdated():
    starts = play(left(), ChairBook(["ch1"]), [(0, 20, {7: L_SEAT})])
    s, first = starts[0]
    assert s == 10 and first.chair == "ch1" and first.since == ts(0) and first.tracklet.track_id == 7


def test_a_big_box_shaking_within_its_size_still_sits():
    # 2026-09-24 (Task 16 tune): stillness tolerance scales with the box (still_box_frac)
    from domain import Frame, TrackObs
    book, ch, starts = left(), ChairBook(["ch1"]), []
    for s in range(20):
        dx = 30 if s % 2 else 0                     # 0.03 jitter, box 0.4 high: within 0.1 x 0.4
        book.observe(Frame("barber-l", ts(s), (TrackObs(7, (500 + dx, 450, 150, 400)),)), POLICY)
        starts += ch.observe_seats("barber-l", list(book.tracks.values()), ts(s), POLICY)
    assert starts and starts[0].since == ts(0)


def test_a_barber_decided_outside_the_seat_never_sits_in_it():
    apps = [staff_app("barber-l", 9, x, 0.75) for x in (0, 1, 2)]
    assert play(left(), ChairBook(["ch1"]), [(0, 25, {9: L_EDGE}), (25, 60, {9: L_SEAT})], apps) == []


def test_a_caped_customer_scoring_like_staff_still_sits():
    # 2026-09-24 (Task 16 tune): zero-shot staff_p of seated caped customers runs 0.4-0.85;
    # first seen in the seat, below seat_staff_threshold (0.9), he sits -- and latches as a customer
    apps = [staff_app("barber-l", 7, x, 0.8) for x in (0, 1, 2)]
    starts = play(left(), ChairBook(["ch1"]), [(0, 20, {7: L_SEAT})], apps)
    assert starts and staff_vote(starts[-1][1].tracklet, POLICY) is False


def test_standing_in_the_chair_zone_is_not_sitting():
    assert play(left(), ChairBook(["ch1"]), [(0, 30, {7: L_EDGE})]) == []


def test_leaving_the_seat_longer_than_the_jitter_resets_the_sit():
    starts = play(left(), ChairBook(["ch1"]), [(0, 8, {7: L_SEAT}), (8, 12, {7: L_AISLE}), (12, 30, {7: L_SEAT})])
    s, first = starts[0]
    assert s == 22 and first.since == ts(12)


def test_no_start_while_a_haircut_runs_and_the_customer_seated_is_evidence():
    book, ch = left(), ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    book.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    book.tracks[7].person = "p1"
    assert play(book, ch, [(1, 30, {7: L_SEAT})]) == []
    assert ch.chairs["ch1"].last_evidence == ts(29)


def test_a_new_unbound_track_sitting_in_a_running_haircut_seat_is_a_claim_for_its_customer():
    # 2026-09-24 (Task 16 tune): the tracker churns a seated customer's track; the new one is nobody's.
    # A claim is only a candidate: the caller decides, and notes the evidence if it binds.
    book, ch = left(), ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    for s in range(1, 6):
        book.observe(frame("barber-l", s, {8: L_SEAT}), POLICY)
    assert ch.claims("barber-l", list(book.tracks.values()), ts(5), POLICY) == [(book.tracks[8], "ch1", "p1")]
    assert ch.chairs["ch1"].last_evidence == ts(0)
    book.tracks[8].person = "p2"                           # someone's already: not claimed
    assert ch.claims("barber-l", list(book.tracks.values()), ts(5), POLICY) == []


def test_a_claim_nobody_binds_never_keeps_the_haircut_alive():
    # review #2: the caller declines the claim every frame; the haircut ends on its own evidence
    book, ch = left(), ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    ends = []
    for s in range(1, 30):
        book.observe(frame("barber-l", s, {8: L_SEAT}), POLICY)
        ch.observe_seats("barber-l", list(book.tracks.values()), ts(s), POLICY)
        ch.claims("barber-l", list(book.tracks.values()), ts(s), POLICY)
        ends += ch.expire(ts(s), POLICY)
    (end,) = ends
    assert (end.person, end.at) == ("p1", ts(0))


def test_haircut_ends_after_the_gap_dated_to_the_last_evidence():
    ch = ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    ch.note_presence("p1", {"ch1"}, ts(5))
    ch.note_presence("p2", {"ch1"}, ts(10))               # someone else: not evidence
    assert ch.expire(ts(25), POLICY) == []
    (end,) = ch.expire(ts(25.5), POLICY)
    assert (end.chair, end.person, end.session_id, end.at) == ("ch1", "p1", "s1", ts(5))
    assert ch.of_person("p1") is None


def test_release_hands_back_the_running_haircut():
    ch = ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(3))
    assert ch.of_person("p1") == "ch1"
    assert ch.release("ch1").session_id == "s1" and ch.release("ch1") is None


def test_start_repeats_every_frame_until_claimed():
    book, ch = left(), ChairBook(["ch1"])
    starts = []
    for s in range(0, 13):
        book.observe(frame("barber-l", s, {7: L_SEAT}), POLICY)
        for st in ch.observe_seats("barber-l", list(book.tracks.values()), ts(s), POLICY):
            starts.append((s, st))
    assert len(starts) == 3
    assert all(st.since == ts(0) for _, st in starts)
    assert [s for s, _ in starts] == [10, 11, 12]
    # After begin, frame 13 reports no start
    ch.begin("ch1", "p1", "s1", ts(12))
    book.observe(frame("barber-l", 13, {7: L_SEAT}), POLICY)
    assert ch.observe_seats("barber-l", list(book.tracks.values()), ts(13), POLICY) == []


# 2026-09-26 (Task 5a): the centre camera may draw a seat for a chair another camera owns
# (c-seat-1 for L5); only the owner camera's seat starts a haircut there.
C_SEAT_ZONES = ZONES["barber-c"] + [
    {"id": "c-s1", "name": "seat 1 (centre)", "type": "seat", "points": sq(0.05, 0.3, 0.15, 0.45), "seat_of": "c-ch1"}]


def test_a_seat_in_a_camera_that_does_not_own_the_chair_starts_nothing():
    book, ch = TrackletBook("barber-c", C_SEAT_ZONES, GEO), ChairBook(["ch1"], {"ch1": "barber-l"})
    starts = []
    for s in range(30):
        book.observe(frame("barber-c", s, {7: C_CHAIR}), POLICY)
        starts += ch.observe_seats("barber-c", list(book.tracks.values()), ts(s), POLICY)
    assert book.tracks[7].seat == "ch1" and starts == []


def test_the_owner_cameras_seat_still_starts():
    starts = play(left(), ChairBook(["ch1"], {"ch1": "barber-l"}), [(0, 20, {7: L_SEAT})])
    assert starts and starts[0][1].chair == "ch1"


def test_a_seat_in_a_camera_that_does_not_own_the_chair_claims_nobody():
    # 2026-09-26 (final B review I2): claims follow observe_seats' owner rule
    book, ch = TrackletBook("barber-c", C_SEAT_ZONES, GEO), ChairBook(["ch1"], {"ch1": "barber-l"})
    ch.begin("ch1", "p1", "s1", ts(0))
    for s in range(1, 6):
        book.observe(frame("barber-c", s, {5: C_CHAIR}), POLICY)
    assert book.tracks[5].seat == "ch1" and book.tracks[5].still(3, 0.02, 0.1)
    assert ch.claims("barber-c", list(book.tracks.values()), ts(5), POLICY) == []


# 2026-09-28 (39-min testbed spec, 6.6): seat_claim_mode 1 -- a customer being worked on jitters
# more than still_radius; in the seat for still_window_s is enough for a claim (not for a start)

def test_seat_claim_mode_1_claims_a_body_jittering_in_the_seat():
    book, ch = left(), ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    for s in range(1, 7):
        book.observe(frame("barber-l", s, {8: (0.6 + (0.05 if s % 2 else 0), 0.65)}), POLICY)
    tracks = list(book.tracks.values())
    assert ch.claims("barber-l", tracks, ts(6), POLICY) == []
    assert ch.claims("barber-l", tracks, ts(6), {**POLICY, "seat_claim_mode": 1}) == [(book.tracks[8], "ch1", "p1")]


def test_seat_claim_mode_1_still_needs_the_window_in_the_seat():
    book, ch = left(), ChairBook(["ch1"])
    ch.begin("ch1", "p1", "s1", ts(0))
    book.observe(frame("barber-l", 1, {8: L_EDGE}), POLICY)
    for s in range(2, 4):
        book.observe(frame("barber-l", s, {8: (0.6 + (0.05 if s % 2 else 0), 0.65)}), POLICY)
    assert ch.claims("barber-l", list(book.tracks.values()), ts(3), {**POLICY, "seat_claim_mode": 1}) == []
