"""What the worker records in the product's history, and what it deliberately does not.

A confirmed fall is never queued for review, so the history is the only place it can live.
The aftermath of an alerted fall is not an event at all (consumer.py:203-212) and must not
become one in the history -- the write is guarded at consumer.py:187-188.
"""
from consumer import CameraWorker
from fall_helpers import (DECISIVE_FALL, DECISIVE_UPRIGHT, POLICY, VENDOR_CFG,
                           FakeRedis, FakeStorage, make_rec)


def _worker(storage):
    return CameraWorker(camera="cam-1", monitor=None, redis=FakeRedis(), storage=storage,
                        policy=POLICY, vendor_cfg=VENDOR_CFG)


def test_a_confirmed_fall_is_recorded_even_though_it_is_never_queued():
    storage = FakeStorage()
    worker = _worker(storage)
    results = worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=1)], t=200.0)
    assert results[0]["severity"] == "critical"
    assert results[0]["review_id"] is None            # the routing queues nothing for it
    assert len(storage.events) == 1                   # but the history keeps it
    (event,) = storage.events
    assert event["evaluation"] == "falling"
    assert event["severity"] == "critical"
    assert event["review_id"] is None
    assert event["camera"] == "cam-1"


def test_a_candidate_below_the_threshold_is_recorded_with_its_review_row():
    storage = FakeStorage()
    worker = _worker(storage)
    results = worker.handle_candidates([make_rec(kind="lying", probs=None, track=2)], t=200.0)
    assert results[0]["review_id"] is not None
    (event,) = storage.events
    assert event["review_id"] == results[0]["review_id"]
    assert event["severity"] == "warning"


def test_the_closed_false_alarm_is_recorded_with_no_severity():
    storage = FakeStorage()
    worker = _worker(storage)
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_UPRIGHT, track=3)], t=200.0)
    (event,) = storage.events
    assert event["evaluation"] == "upright"
    assert event["severity"] is None


def test_the_aftermath_of_an_alerted_fall_is_not_recorded():
    storage = FakeStorage()
    worker = _worker(storage)
    # the worker's clock is the candidate's own t_alert (103.0 by default), not the t argument
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=7)], t=200.0)
    assert len(storage.events) == 1
    results = worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=7)], t=201.0)
    assert results[0]["suppressed"] is True
    assert len(storage.events) == 1                    # the aftermath is not an event
    # ...and past the window, the same track records again
    results = worker.handle_candidates(
        [make_rec(kind="fall", probs=DECISIVE_FALL, track=7, t_alert=200.0)], t=200.0)
    assert results[0]["suppressed"] is False
    assert len(storage.events) == 2
