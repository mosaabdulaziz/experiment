from datetime import datetime
from itertools import count

from domain import EngineUpdate, Frame, TrackObs
from extra_waits import ExtraWaits
from policy import DEFAULT_POLICY

P = {**DEFAULT_POLICY, "wt_extra_views": 1.0}
T0 = datetime(2026, 10, 15, 20, 0).timestamp()


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ZONES = {
    "c": [{"id": "c-sofa", "name": "sofa", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9), "pair": "S", "owner": True},
          {"id": "c-other", "name": "other sofa", "type": "waiting", "points": sq(0.0, 0.5, 0.4, 0.9)}],
    "a": [{"id": "a-sofa", "name": "sofa a", "type": "waiting", "points": sq(0.1, 0.1, 0.5, 0.5), "pair": "S"}],
    "b": [{"id": "b-sofa", "name": "sofa b", "type": "waiting", "points": sq(0.5, 0.1, 0.9, 0.5), "pair": "S"}],
}
SPOT = {"c": (0.55, 0.55), "a": (0.15, 0.15), "b": (0.55, 0.15)}


def people(cam, n):
    x, y = SPOT[cam]
    return tuple(TrackObs(i + 1, (0, 0, 1, 1), 0.9, tuple(v for p in sq(x + 0.05 * i, y, x + 0.05 * i + 0.04, y + 0.3)
                                                           for v in p)) for i in range(n))


def run(timeline, track_waits=1, policy=P, book=None, silent_from=None):
    """timeline: [(seconds, {camera: people})] -- each held for that long, one frame per camera a second."""
    ids = count(1)
    book = book or ExtraWaits(ZONES, "c", lambda: f"x{next(ids)}", lambda at: next(ids))
    upd, t = EngineUpdate(), T0
    for secs, seen in timeline:
        for _ in range(secs):
            for cam, n in seen.items():
                if not (cam == "c" and silent_from is not None and t >= silent_from):
                    book.observe(Frame(cam, t, people(cam, n)), lambda tid: None, set(), policy)
            last_c = t if silent_from is None or t < silent_from else silent_from - 1
            book.tick(t, lambda area: track_waits, last_c, policy, upd)
            t += 1
    return book, upd


def ended(upd):
    return [(round(r.started_at.timestamp() - T0), round(r.ended_at.timestamp() - T0), r.ended_by)
            for r in upd.waits.values() if r.ended_at is not None]


def test_two_people_one_view_sees_apart_give_one_extra_wait_dated_back():
    book, upd = run([(60, {"c": 1, "a": 2}), (60, {"c": 0, "a": 0})], track_waits=1)
    assert not book.open()
    # the second person counts after the 9 s dwell (excess from 9 s), an extra wait opens 9 s later dated to 9 s;
    # both gone at 60 s (one track wait still open: the count stays below it 30 s) -> ended at 60
    assert ended(upd) == [(9, 60, "left")]
    (r,) = upd.waits.values()
    assert r.track_id == -1 and r.camera == "c"


def test_the_views_count_added_up():
    book, upd = run([(40, {"c": 1, "a": 1, "b": 1})], track_waits=1)
    assert len(book.open()) == 1


def test_views_seeing_no_more_than_the_open_waits_add_nothing():
    book, upd = run([(120, {"c": 2, "a": 2})], track_waits=2)
    assert not book.open() and not upd.waits


def test_a_short_excess_opens_nothing_and_a_short_loss_ends_nothing():
    book, upd = run([(15, {"c": 1, "a": 2})], track_waits=1)     # excess from 9 s: 6 s < wt_start_s
    assert not book.open()
    book, upd = run([(30, {"c": 1, "a": 2}), (20, {"c": 1, "a": 1}), (30, {"c": 1, "a": 2})], track_waits=1)
    assert len(book.open()) == 1 and not ended(upd)             # 20 s below the open waits < wt_extra_end_s


def test_off_by_default_and_without_paired_views():
    book, upd = run([(60, {"c": 1, "a": 2})], policy=DEFAULT_POLICY)
    assert not book.open() and not upd.waits
    zones = {"c": ZONES["c"], "a": [{**ZONES["a"][0], "pair": "T"}]}
    assert not ExtraWaits(zones, "c", lambda: "x", lambda at: 1).areas


def test_a_silent_wait_camera_ends_them_as_a_stall():
    book, upd = run([(30, {"c": 1, "a": 2}), (90, {"a": 2})], track_waits=1, silent_from=T0 + 30)
    assert not book.open()
    assert ended(upd) == [(9, 29, "stall")]                    # at the wait camera's last frame
