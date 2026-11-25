"""Routes, alert contract, clip request, suppression and the ignored fall_recovered.

ACs: AC-F16.1-4, AC-F17.1, AC-F18.1, AC-F19.1, AC-F30.1, AC-F30.2, AC-F31.1, AC-F31.2,
AC-F36.1, AC-F37.1.
Route table: FALL_DETECTION_DESIGN.md §5.3; rulings A2 (§13.3), A3.
The side effects are asserted on a FakeRedis/FakeStorage — the observable contract, not the wiring.
"""
import pytest

import scoring
from consumer import CameraWorker
from fall_helpers import (DECISIVE_FALL, DECISIVE_LYING, DECISIVE_UPRIGHT, POLICY, VENDOR_CFG,
                          FakeRedis, FakeStorage, detailed, make_rec)


class _NoCandidates:
    def update(self, *a, **k):
        return []


def _worker(redis=None, storage=None, policy=POLICY):
    return CameraWorker(camera="cam-1", monitor=_NoCandidates(), redis=redis or FakeRedis(),
                        storage=storage or FakeStorage(), policy=policy, vendor_cfg=VENDOR_CFG)


def _route(evaluation, combined):
    return scoring.route(evaluation, combined, POLICY["scoring"]["confirm_threshold"])


# ── the route table (AC-F16.1-4, AC-F36.1, AC-F30.1/2) ─────────────────────────────────────────────────────

def test_a_confident_falling_evaluation_alerts_critical_with_no_review():
    assert _route("falling", 0.90) == ("critical", False)


def test_a_confident_immobile_evaluation_alerts_critical_with_no_review():
    assert _route("immobile", 0.90) == ("critical", False)


def test_an_unconfident_candidate_of_any_evaluation_alerts_warning_and_queues_review():
    for evaluation in ("falling", "immobile", "lying", "unstable", "upright"):
        assert _route(evaluation, 0.50) == ("warning", True)


def test_a_confident_upright_evaluation_is_a_closed_false_alarm():
    assert _route("upright", 0.90) == (None, False)


def test_a_confident_lying_evaluation_alerts_warning_with_no_review():
    assert _route("lying", 0.90) == ("warning", False)


def test_unstable_routes_exactly_like_lying():
    assert _route("unstable", 0.90) == _route("lying", 0.90) == ("warning", False)
    assert _route("unstable", 0.50) == _route("lying", 0.50) == ("warning", True)


# ── the side effects (AC-F16.x through the worker, AC-F17.1, AC-F18.1) ────────────────────────────────────

def test_a_confirmed_fall_writes_the_platform_alert_and_requests_the_clip_on_the_frame_clock():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=DECISIVE_FALL, action=None, kinematic=None, cold=None,
                   t_start=1232.5, t_alert=1234.5)
    results = _worker(redis, storage, detailed()).handle_candidates([rec], t=1234.5)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["v"] == "1" and alert["kind"] == "fall" and alert["category"] == "safety"
    assert alert["severity"] == "critical"
    assert alert["pts"] == "1234.5"                          # AC-F19.1: the sidecar's frame clock
    assert alert["message"] == f"Person fell (confidence {round(results[0]['combined'] * 100)}%)"
    # what the message says, as fields too: a console words it in its own language (the safety console's Arabic)
    assert (alert["evaluation"], alert["review"]) == ("falling", "0")
    assert alert["confidence"] == f"{results[0]['combined']:.2f}"
    (svr,) = redis.entries("iris:svr_request:cam-1")
    assert svr["pts"] == "1234.5"                            # AC-F18.1: the same clock
    assert storage.rows == []                                # auto-confirmed: no review


def test_an_unconfident_fall_writes_a_warning_alert_and_a_review_row_carrying_the_evidence():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=None, kinematic=0.9, t_alert=1234.5)   # no window: cold ceiling
    _worker(redis, storage, detailed()).handle_candidates([rec], t=1234.5)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["severity"] == "warning" and alert["kind"] == "fall"
    assert (alert["evaluation"], alert["review"]) == ("falling", "1")
    (row,) = storage.rows
    assert row["evaluation"] == "falling"                     # the detector's own verdict, no reading yet
    assert 0.0 <= row["combined"] < 0.70
    assert row["camera"] == "cam-1" and row["pts"] == 1234.5


def test_without_alert_details_the_alert_is_the_platform_s_as_before():
    """`alert_details` is the safety stack's (products/safety/fall_detection.yaml); a plain fall-up keeps its alert:
    the evaluation and the confidence as the message, and no other fields."""
    redis = FakeRedis()
    rec = make_rec(kind="fall", probs=DECISIVE_FALL, track=7, t_alert=1234.5)
    results = _worker(redis).handle_candidates([rec], t=1234.5)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["message"] == f"falling {results[0]['combined']:.2f}"
    assert not {"person_ids", "evaluation", "confidence", "review"} & set(alert)


def test_a_confirmed_upright_evaluation_writes_no_alert_and_no_review_row():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=DECISIVE_UPRIGHT, t_alert=1234.5)
    results = _worker(redis, storage).handle_candidates([rec], t=1234.5)
    assert redis.added == [] and storage.rows == []
    assert results[0]["evaluation"] == "upright" and results[0]["alert"] is None
    (event,) = storage.events                       # the history keeps what the product saw
    assert event["evaluation"] == "upright" and event["severity"] is None


def test_a_confirmed_lying_candidate_alerts_warning_without_a_review_row():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="lying", probs=DECISIVE_LYING, t_alert=500.0)
    _worker(redis, storage).handle_candidates([rec], t=500.0)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["severity"] == "warning" and alert["kind"] == "fall"
    assert "lying" in alert["message"]
    assert storage.rows == []


def test_an_unconfident_lying_candidate_alerts_warning_and_queues_review():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="lying", probs=None, t_alert=500.0)  # no window: cold ceiling, delegated
    _worker(redis, storage).handle_candidates([rec], t=500.0)
    assert len(redis.entries("iris:alert:cam-1")) == 1
    assert redis.entries("iris:alert:cam-1")[0]["severity"] == "warning"
    assert len(storage.rows) == 1


# ── the frame clock (AC-F19.1) ─────────────────────────────────────────────────────────────────────────────

def test_the_monitor_is_given_the_frame_entrys_pts_and_every_write_carries_it():
    calls = []

    class _Recording:
        def update(self, t, ids, boxes, kps):
            calls.append((t, ids, boxes, kps))
            return []

    redis = FakeRedis()
    worker = CameraWorker(camera="cam-1", monitor=_Recording(), redis=redis, storage=FakeStorage(),
                          policy=POLICY, vendor_cfg=VENDOR_CFG)
    worker.handle_entry({"pts": "1234.5", "pose": '{"shape": [1280, 720], "items": []}'})
    ((t, ids, boxes, kps),) = calls
    assert t == pytest.approx(1234.5) and ids == [] and boxes == [] and kps == []


# ── suppression of the fall's aftermath (AC-F31.1, AC-F31.2) ───────────────────────────────────────────────

def test_a_lying_candidate_right_after_an_alerted_fall_on_the_same_track_is_suppressed_but_still_scored():
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage)
    fall = make_rec(kind="fall", track=3, probs=DECISIVE_FALL, t_start=100.0, t_alert=103.0,
                    event_id="fall-1")
    worker.handle_candidates([fall], t=103.0)
    assert len(redis.entries("iris:alert:cam-1")) == 1
    lying = make_rec(kind="lying", track=3, probs=DECISIVE_LYING, t_start=105.0, t_alert=108.0,
                     event_id="lying-1")
    results = worker.handle_candidates([lying], t=108.0)     # 5 s after the fall alert: inside 10 s
    assert results[0]["suppressed"] is True
    assert results[0]["combined"] > 0.0                       # scored, not skipped
    assert len(redis.entries("iris:alert:cam-1")) == 1        # no second alert
    assert storage.rows == []                                 # not queued


def test_a_lying_candidate_after_the_suppression_window_alerts_normally():
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage)
    worker.handle_candidates([make_rec(kind="fall", track=3, probs=DECISIVE_FALL, t_start=100.0,
                                       t_alert=103.0, event_id="fall-1")], t=103.0)
    late = make_rec(kind="lying", track=3, probs=DECISIVE_LYING, t_start=114.0, t_alert=116.0,
                    event_id="lying-2")
    results = worker.handle_candidates([late], t=116.0)       # 13 s after: outside 10 s
    assert results[0]["suppressed"] is False
    assert len(redis.entries("iris:alert:cam-1")) == 2


def test_suppression_is_per_track_not_a_mute_on_the_kind():
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage)
    worker.handle_candidates([make_rec(kind="fall", track=3, probs=DECISIVE_FALL, t_start=100.0,
                                       t_alert=103.0, event_id="fall-1")], t=103.0)
    other = make_rec(kind="lying", track=8, probs=DECISIVE_LYING, t_start=104.0, t_alert=106.0,
                     event_id="lying-3")
    results = worker.handle_candidates([other], t=106.0)      # another person, inside the window
    assert results[0]["suppressed"] is False
    assert len(redis.entries("iris:alert:cam-1")) == 2


# ── fall_recovered is ignored (AC-F37.1) ───────────────────────────────────────────────────────────────────

def test_a_fall_recovered_candidate_produces_no_alert_no_review_and_no_state():
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage)
    rec = make_rec(kind="fall_recovered", probs=DECISIVE_UPRIGHT, t_alert=200.0)
    results = worker.handle_candidates([rec], t=200.0)
    assert redis.added == [] and storage.rows == []
    assert storage.events == []                                 # not in the history either
    assert results == []                                      # not even an evaluation is reported


# ── the alert floor (safety integration, 2026-11-10) ──────────────────────────────────────────────────────
# The safety stack sets scoring.alert_above: 0.0 -- a candidate the product is 0% sure of ("A person may be lying on the
# floor: needs review (confidence 0%)") is not an incident. The floor is the confidence as the alert states it (two
# decimals), so nothing that reads 0% gets through. The review queue and the history keep the candidate.

def _floor_worker(redis, storage, floor=0.0):
    from fall_helpers import policy_with
    return CameraWorker(camera="cam-1", monitor=_NoCandidates(), redis=redis, storage=storage,
                        policy=detailed(policy_with(alert_above=floor)), vendor_cfg=VENDOR_CFG)


def test_a_candidate_at_zero_confidence_is_no_incident_but_still_queued_for_review():
    for cold in (0.0, 0.01):                                  # combined 0.0 and 0.004: both read 0.00 / 0%
        redis, storage = FakeRedis(), FakeStorage()
        rec = make_rec(kind="fall", probs=None, cold=cold, t_alert=1234.5)
        (result,) = _floor_worker(redis, storage).handle_candidates([rec], t=1234.5)
        assert result["alert"] is None and redis.added == []          # no alert, no clip request
        assert len(storage.rows) == 1 and len(storage.events) == 1     # the review queue and the history keep it


def test_a_candidate_above_the_floor_is_an_incident():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=None, cold=0.3, t_alert=1234.5)  # combined 0.12
    _floor_worker(redis, storage).handle_candidates([rec], t=1234.5)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["confidence"] == "0.12" and alert["severity"] == "warning"


def test_without_a_floor_every_routed_candidate_alerts_as_before():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=None, cold=0.0, t_alert=1234.5)
    _worker(redis, storage).handle_candidates([rec], t=1234.5)
    assert len(redis.entries("iris:alert:cam-1")) == 1
