"""proximity.episodes: persistence with grace, and one alert per vehicle episode."""
import pytest

from perception.modules.proximity.episodes import EpisodeBook, PairLatch


def test_a_pair_latches_after_persist_and_survives_short_gaps():
    latch = PairLatch()
    assert latch.hot("k", 0.0, 0.6, persist_s=1.0, grace_s=0.5) is None
    assert latch.hot("k", 0.4, 0.5, 1.0, 0.5) is None
    assert latch.hot("k", 0.8, 0.7, 1.0, 0.5) is None          # after a 0.4 s detector gap
    held, min_gap = latch.hot("k", 1.0, 0.7, 1.0, 0.5)
    assert held == pytest.approx(1.0) and min_gap == pytest.approx(0.5)


def test_a_gap_longer_than_grace_restarts_the_clock():
    latch = PairLatch()
    for t in (0.0, 0.4, 0.8):
        latch.hot("k", t, 0.5, 1.0, 0.5)
    assert latch.hot("k", 1.5, 0.5, 1.0, 0.5) is None          # 0.7 s without it: the clock restarts at 1.5
    latch.hot("k", 2.0, 0.5, 1.0, 0.5)
    assert latch.hot("k", 2.5, 0.5, 1.0, 0.5)[0] == pytest.approx(1.0)


def test_forgotten_pairs_start_again():
    latch = PairLatch()
    latch.hot("k", 0.0, 0.5, 1.0, 20.0)                        # a grace so long only forget() can end the span
    latch.forget(10.0, older_than=5.0)
    assert latch.hot("k", 10.5, 0.5, 1.0, 20.0) is None


def test_a_pair_whose_ids_switch_keeps_its_clock():
    latch = PairLatch()
    for t in (0.0, 0.15, 0.3, 0.45, 0.6):
        latch.hot(("z", 1, 10), t, 0.5, 1.0, 0.5, where=((450, 560), (300 + 100 * t, 500), 100))
    # the vehicle comes back as id 11, a little further on; the person is the same
    held, _gap = latch.hot(("z", 1, 11), 1.05, 0.5, 1.0, 0.5, where=((450, 560), (405, 500), 100))
    assert held == pytest.approx(1.05)
    assert latch.hot(("z", 1, 12), 1.05, 0.5, 1.0, 0.5, where=((900, 100), (950, 120), 100)) is None   # elsewhere


def _add(book, t, start, pid, vid, foot, vref, gap=0.5, severity="warning", hold=10.0):
    return book.add("z", t, start, pid, vid, foot, vref, gap, 100, hold, severity)


def test_one_episode_however_the_ids_switch():
    book = EpisodeBook()
    e, what = _add(book, 1.0, 0.0, 1, 10, (100, 500), (150, 520))
    assert what == "opened" and (e.start, e.end) == (0.0, 1.0)
    again, what = _add(book, 2.0, 1.0, 2, 11, (130, 500), (190, 520), gap=0.3)   # both re-identified
    assert again is e and what == "joined"
    assert e.person_ids == [1, 2] and e.vehicle_ids == [10, 11] and e.min_gap == 0.3 and e.end == 2.0


def test_a_pair_elsewhere_in_another_zone_or_after_the_hold_opens_a_new_episode():
    book = EpisodeBook()
    _add(book, 1.0, 0.0, 1, 10, (100, 500), (150, 520))
    assert _add(book, 2.0, 1.0, 3, 12, (600, 300), (650, 320))[1] == "opened"                  # 500 px away
    assert book.add("y", 2.0, 1.0, 1, 10, (100, 500), (150, 520), 0.5, 100, 10.0, "warning")[1] == "opened"
    assert _add(book, 12.5, 11.5, 1, 10, (100, 500), (150, 520))[1] == "opened"                # hold over


def test_a_more_severe_pair_escalates_the_episode_once():
    book = EpisodeBook()
    e, _ = _add(book, 1.0, 0.0, 1, 10, (100, 500), (150, 520), severity="info")
    assert _add(book, 1.5, 0.5, 2, 10, (120, 500), (160, 520), gap=0.1, severity="critical") == (e, "escalated")
    assert _add(book, 2.0, 1.0, 2, 10, (120, 500), (170, 520), gap=0.1, severity="critical") == (e, "joined")
    assert _add(book, 2.5, 1.5, 3, 10, (110, 500), (180, 520), severity="warning") == (e, "joined")


def test_episodes_close_after_their_hold():
    book = EpisodeBook()
    _add(book, 1.0, 0.0, 1, 10, (100, 500), (150, 520), hold=10.0)
    assert book.expire(5.0) == []
    (closed,) = book.expire(11.5)
    assert closed.zone_id == "z" and book.open == []


def test_a_latched_pair_remembers_when_its_gap_was_smallest():
    latch = PairLatch()
    for t, gap in ((0.0, 0.6), (0.4, 0.3), (0.8, 0.5)):
        latch.hot("k", t, gap, 1.0, 0.5)
    assert latch.min_time("k") == 0.4 and latch.min_time("other") is None


def test_an_episode_keeps_one_incident_id_and_when_its_smallest_gap_was_seen():
    book = EpisodeBook()
    e, _ = book.add("z", 1.0, 0.0, 1, 10, (100, 500), (150, 520), 0.5, 100, 10.0, "warning", min_gap_t=0.4)
    assert len(e.incident) == 12 and e.min_gap_t == 0.4
    book.add("z", 2.0, 1.0, 1, 10, (110, 500), (160, 520), 0.3, 100, 10.0, "critical", min_gap_t=1.8)
    book.add("z", 2.5, 1.5, 1, 10, (115, 500), (165, 520), 0.4, 100, 10.0, "critical", min_gap_t=2.5)
    assert (e.min_gap, e.min_gap_t) == (0.3, 1.8)
    other, _ = book.add("z", 2.5, 1.5, 3, 12, (900, 100), (950, 120), 0.5, 100, 10.0, "warning")
    assert other.incident != e.incident and other.min_gap_t == 2.5



def test_the_book_says_which_open_episode_a_pair_would_join_without_adding_it():
    # Ticket 6: a pair the depth check holds back is not recorded quietly when an alerted episode covers it
    book = EpisodeBook()
    e, _ = _add(book, 1.0, 0.5, 1, 10, (100, 100), (200, 100))
    assert book.covering("z", 2.0, (110, 100), (900, 900), 100.0) is e        # its person is near
    assert book.covering("z", 2.0, (900, 900), (210, 100), 100.0) is e        # its vehicle is near
    assert book.covering("z", 2.0, (900, 900), (900, 900), 100.0) is None     # neither
    assert book.covering("other", 2.0, (110, 100), (200, 100), 100.0) is None
    assert book.covering("z", 20.0, (110, 100), (200, 100), 100.0) is None    # its hold ran out
    assert len(book.open) == 1 and e.person_ids == [1]
