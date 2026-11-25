from wait_links import SKEW_S, link_waits


def test_each_haircut_takes_the_earliest_wait_that_ended_within_the_window():
    waits = [("w1", 100.0), ("w2", 130.0)]
    starts = [("s1", 160.0), ("s2", 200.0)]
    assert link_waits(waits, starts, 150.0) == {"w1": "s1", "w2": "s2"}


def test_a_wait_ended_longer_ago_than_the_window_is_not_served():
    assert link_waits([("w1", 100.0)], [("s1", 251.0)], 150.0) == {}
    assert link_waits([("w1", 100.0)], [("s1", 250.0)], 150.0) == {"w1": "s1"}


def test_one_wait_per_haircut_and_one_haircut_per_wait():
    assert link_waits([("w1", 100.0)], [("s1", 110.0), ("s2", 120.0)], 150.0) == {"w1": "s1"}
    assert link_waits([("w1", 100.0), ("w2", 101.0)], [("s1", 110.0)], 150.0) == {"w1": "s1"}


def test_open_waits_never_link():
    assert link_waits([("w1", None)], [("s1", 110.0)], 150.0) == {}


def test_a_start_dated_slightly_before_the_wait_end_still_links():
    # Review Focus: the chair camera dates the start back to the sitting; the wait camera ends the wait at the edge
    assert link_waits([("w1", 100.0)], [("s1", 100.0 - SKEW_S + 1)], 150.0) == {"w1": "s1"}
    assert link_waits([("w1", 100.0)], [("s1", 100.0 - SKEW_S - 1)], 150.0) == {}
