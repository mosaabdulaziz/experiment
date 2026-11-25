"""One monitor per camera, a bounded queue that drops the oldest, and no latency in the decision path.

ACs: AC-F20.1, AC-F21.1, AC-N3.1 (service-level half; the end-to-end half is the gated replay).
Design: FALL_DETECTION_DESIGN.md §5.1-5.2; plan §The monitor runs on its own thread.
"""
import json
import logging
import time

import pytest

from consumer import CameraWorker, Consumer, FrameQueue, build_workers
from fall_helpers import DECISIVE_FALL, POLICY, VENDOR_CFG, FakeRedis, FakeStorage, make_rec


class _Recording:
    def __init__(self):
        self.calls = []

    def update(self, t, ids, boxes, kps):
        self.calls.append((t, ids))
        return []


class _ShapeRecording(_Recording):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def set_shape(self, shape):
        self.shapes.append(shape)


def _entry(pts, track=1):
    pose = {"shape": [1280, 720], "items": [
        {"id": track, "box": [10.0, 10.0, 90.0, 200.0], "kps": [[50.0, 60.0, 0.9]] * 17}]}
    return {"pts": str(pts), "pose": json.dumps(pose)}


# ── one FallMonitor per camera (AC-F20.1) ──────────────────────────────────────────────────────────────────

def test_two_cameras_never_share_a_monitor_or_cross_talk():
    workers = build_workers(["cam-1", "cam-2"], monitor_factory=lambda cam: _Recording(),
                            redis=FakeRedis(), storage=FakeStorage(),
                            policy=POLICY, vendor_cfg=VENDOR_CFG)
    assert workers["cam-1"].monitor is not workers["cam-2"].monitor
    workers["cam-1"].handle_entry(_entry(1.0, track=42))
    workers["cam-2"].handle_entry(_entry(2.0, track=42))     # the same tracker id on both cameras
    assert workers["cam-1"].monitor.calls == [(1.0, [42])]
    assert workers["cam-2"].monitor.calls == [(2.0, [42])]


# ── a frame without a pose field says nothing; a frame without people says they are gone ───────────────────

def test_a_frame_without_a_pose_field_does_not_tell_the_monitor_that_the_people_are_gone():
    monitor = _Recording()
    worker = CameraWorker(camera="cam-1", monitor=monitor, redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    worker.handle_entry(_entry(1.0, track=42))
    worker.handle_entry({"pts": "2.0"})                        # no pose field at all
    worker.handle_entry({"pts": "3.0", "pose": "not json"})    # unreadable: the same thing
    assert monitor.calls == [(1.0, [42])]                      # the monitor was not updated


def test_a_frame_with_a_pose_field_and_no_people_does_tell_the_monitor():
    monitor = _Recording()
    worker = CameraWorker(camera="cam-1", monitor=monitor, redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    worker.handle_entry({"pts": "1.0", "pose": json.dumps({"shape": [1280, 720], "items": []})})
    assert monitor.calls == [(1.0, [])]                        # the people are gone


def test_a_pose_that_is_not_an_object_is_no_field_not_a_crash():
    monitor = _ShapeRecording()
    worker = CameraWorker(camera="cam-1", monitor=monitor, redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    worker.handle_entry({"pts": "1.0", "pose": "[]"})          # JSON, but not an object
    worker.handle_entry({"pts": "2.0", "pose": json.dumps(
        {"shape": {"w": 1280, "h": 720},                       # a shape the monitor cannot index
         "items": [{"id": 7, "box": [0, 0, 10, 10], "kps": [[1.0, 2.0, 0.9]] * 17}]})})
    assert monitor.shapes == []                                # no shape was ever handed over
    assert monitor.calls == [(2.0, [7])]                       # a readable frame still delivers its people


def test_a_malformed_person_is_dropped_and_the_rest_of_the_frame_survives():
    monitor = _Recording()
    worker = CameraWorker(camera="cam-1", monitor=monitor, redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    worker.handle_entry({"pts": "1.0", "pose": json.dumps({"shape": [1280, 720], "items": [
        "not a person",                                        # no `.get` at all
        {"id": 7, "box": [0, 0, 10, 10], "kps": [[1.0, 2.0, 0.9]] * 17},
        {"id": "not a number", "box": [], "kps": []},
    ]})})
    assert monitor.calls == [(1.0, [7])]


def test_a_malformed_pose_is_said_once_not_once_per_frame(caplog):
    """The warning is for the operator; at camera rate it would bury the log it is written to."""
    worker = CameraWorker(camera="cam-1", monitor=_Recording(), redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    with caplog.at_level(logging.WARNING, logger="fall_detection.consumer"):
        for i in range(30):
            assert worker.handle_entry({"pts": str(float(i)), "pose": "not json"}) == []
    assert sum("not JSON" in r.getMessage() for r in caplog.records) == 1


def test_a_frame_whose_clock_is_not_a_number_is_dropped_and_said_once(caplog):
    worker = CameraWorker(camera="cam-1", monitor=_Recording(), redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    with caplog.at_level(logging.WARNING, logger="fall_detection.consumer"):
        for _ in range(30):
            assert worker.handle_entry({"pts": "no clock", "pose": _POSE}) == []
    assert sum("clock" in r.getMessage() for r in caplog.records) == 1


# ── the bounded queue drops the oldest (AC-F21.1) ──────────────────────────────────────────────────────────

def test_a_full_queue_drops_the_oldest_frame_and_never_grows():
    q = FrameQueue(maxsize=3)
    for i in range(10):
        q.put(_entry(float(i)))
    assert q.qsize() == 3
    kept = [q.get()["pts"] for _ in range(3)]
    assert kept == ["7.0", "8.0", "9.0"]                      # the newest survive


def test_the_worker_keeps_running_after_the_queue_overflowed():
    monitor = _Recording()
    worker = CameraWorker(camera="cam-1", monitor=monitor, redis=FakeRedis(), storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG, queue=FrameQueue(maxsize=2))
    for i in range(20):
        worker.submit(_entry(float(i)))
    worker.drain()                                            # process what survived
    assert worker.alive                                      # the monitor thread/loop is still usable
    assert [c[0] for c in monitor.calls] == [18.0, 19.0]


# ── the decision path adds no latency of its own (AC-N3.1, service-level half) ────────────────────────────

def test_a_hundred_confirmed_falls_are_routed_in_well_under_the_clip_window():
    worker = CameraWorker(camera="cam-1", monitor=_Recording(), redis=FakeRedis(),
                          storage=FakeStorage(), policy=POLICY, vendor_cfg=VENDOR_CFG)
    recs = [make_rec(kind="fall", probs=DECISIVE_FALL, track=i, t_alert=100.0 + i,
                     event_id=f"ev{i}") for i in range(100)]
    started = time.monotonic()
    worker.handle_candidates(recs, t=200.0)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0                                      # the 5 s budget is for the whole chain


# ── the aftermath bookkeeping stays bounded ───────────────────────────────────────────────────────────────

def test_the_aftermath_bookkeeping_forgets_falls_whose_window_has_passed():
    worker = CameraWorker(camera="cam-1", monitor=_Recording(), redis=FakeRedis(),
                          storage=FakeStorage(), policy=POLICY, vendor_cfg=VENDOR_CFG)
    window = float(POLICY["suppress_after_fall_s"])
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=1,
                                       t_alert=100.0, event_id="ev1")])
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=2,
                                       t_alert=100.0 + window + 1.0, event_id="ev2")])
    assert list(worker._alerted_falls) == [2]                 # track 1's window has passed


# ── a Redis outage is said once, retried slowly, and its recovery is said once ────────────────────────────

class _StopAfter:
    """Stands in for Consumer's stop event: the loop runs a set number of passes, then stops."""

    def __init__(self, passes: int):
        self._passes = passes
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return len(self.waits) >= self._passes

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.is_set()


class _Reader:
    """Stands in for acl.MetaStreamReader: each pass pops the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def read(self, camera, count):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return [{"id": "1-0", "fields": {"pts": "1.0", "pose": outcome}}]


_POSE = json.dumps({"shape": [1280, 720], "items": [
    {"id": 7, "box": [10.0, 10.0, 90.0, 200.0], "kps": [[50.0, 60.0, 0.9]] * 17}]})


def _consumer(reader, passes: int) -> tuple[Consumer, _Recording, dict]:
    monitor = _Recording()
    workers = build_workers(["cam-1"], monitor_factory=lambda _c: monitor,
                            redis=FakeRedis(), storage=FakeStorage(),
                            policy=POLICY, vendor_cfg=VENDOR_CFG)
    consumer = Consumer(FakeRedis(), workers, interval=0.05, retry_interval=1.0)
    consumer._reader = reader
    consumer._stop = _StopAfter(passes)
    return consumer, monitor, workers


def test_a_sustained_outage_is_logged_once_and_retried_at_the_slow_interval(caplog):
    import redis
    consumer, monitor, _workers = _consumer(_Reader([redis.ConnectionError("down")] * 4), passes=4)
    with caplog.at_level(logging.WARNING, logger="fall_detection.consumer"):
        consumer.run()
    said = [r.getMessage() for r in caplog.records]
    assert sum("unreachable" in m for m in said) == 1          # not one line per attempt
    assert consumer._stop.waits == [1.0] * 4                   # the slow interval, not 50 ms
    assert monitor.calls == []


def test_the_recovery_is_said_once_and_the_frames_flow_again(caplog):
    import redis
    consumer, monitor, workers = _consumer(
        _Reader([redis.ConnectionError("down"), redis.ConnectionError("down"), _POSE]), passes=3)
    with caplog.at_level(logging.INFO, logger="fall_detection.consumer"):
        consumer.run()
        workers["cam-1"].drain()                               # the worker thread would do this
    said = [r.getMessage() for r in caplog.records]
    assert sum("unreachable" in m for m in said) == 1
    assert sum("readable again" in m for m in said) == 1
    assert consumer._stop.waits == [1.0, 1.0, 0.05]
    assert monitor.calls == [(1.0, [7])]                      # the frame after the outage is handled
