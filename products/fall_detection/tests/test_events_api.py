"""The events endpoint: the tab's feed of what the product has seen since its cursor.

ACs: AC-F22.1 (the evidence beside the operator's decision). Design: phase-5 §4.3.
"""
from fastapi.testclient import TestClient
import pytest

import boot
import storage
from server import create_app
from storage import ReviewQueue

EVENT = {"camera": "cam-1", "pts": 103.5, "track": 3, "kind": "fall",
         "evaluation": "falling", "combined": 0.81, "severity": "critical",
         "path_scores": {"action": 0.9, "kinematic": None, "cold": None},
         "clip_window": [98.0, 105.5], "review_id": None}


@pytest.fixture
def queue(tmp_path):
    q = ReviewQueue(storage.connect(tmp_path / "events.db"))
    yield q
    q.close()


def _client(queue):
    ready = {"device": "cuda:0", "weights_verified": True, "classifier_ready": True}
    return TestClient(create_app(status=boot.status(ready), storage=queue))


def test_the_events_are_a_bare_array_newest_first(queue):
    queue.add_event(dict(EVENT, event_id="ev-1"))
    queue.add_event(dict(EVENT, event_id="ev-2"))
    body = _client(queue).get("/events").json()
    assert isinstance(body, list)
    assert [r["event_id"] for r in body] == ["ev-2", "ev-1"]
    assert body[0]["severity"] == "critical"


def test_the_cursor_returns_only_what_is_newer(queue):
    first = queue.add_event(dict(EVENT, event_id="ev-1"))
    queue.add_event(dict(EVENT, event_id="ev-2"))
    body = _client(queue).get(f"/events?since={first}").json()
    assert [r["event_id"] for r in body] == ["ev-2"]


def test_the_camera_narrows_the_feed(queue):
    queue.add_event(dict(EVENT, camera="cam-1"))
    queue.add_event(dict(EVENT, camera="cam-2"))
    body = _client(queue).get("/events?camera=cam-2").json()
    assert [r["camera"] for r in body] == ["cam-2"]


def test_the_limit_is_capped(queue):
    for i in range(5):
        queue.add_event(dict(EVENT, event_id=f"ev-{i}"))
    body = _client(queue).get("/events?limit=2").json()
    assert len(body) == 2
