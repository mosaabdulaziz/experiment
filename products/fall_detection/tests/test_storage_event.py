"""The product's own history: every scored candidate, kept with its evidence.

The review queue is the operator's work list; this is the record of what the product saw.
A confirmed fall is never queued, so without this table the tab could never show one.
"""
import pytest

import storage

EVENT = {
    "camera": "cam-1",
    "pts": 103.5,
    "track": 3,
    "event_id": "ev-abc123",
    "kind": "fall",
    "evaluation": "falling",
    "combined": 0.81,
    "severity": "critical",
    "path_scores": {"action": 0.9, "kinematic": None, "cold": None},
    "clip_window": [98.0, 105.5],
    "review_id": None,
}


def test_an_event_survives_a_restart_with_all_of_its_evidence(tmp_path):
    db = tmp_path / "fall.db"
    conn = storage.connect(db)
    event_id = storage.add_event(conn, EVENT)
    conn.close()
    conn2 = storage.connect(db)                              # the restart
    (row,) = [r for r in storage.list_events(conn2) if r["id"] == event_id]
    assert row["camera"] == "cam-1"
    assert row["pts"] == pytest.approx(103.5)
    assert row["evaluation"] == "falling"
    assert row["combined"] == pytest.approx(0.81)
    assert row["severity"] == "critical"
    assert row["path_scores"] == {"action": 0.9, "kinematic": None, "cold": None}
    assert row["clip_window"] == [98.0, 105.5]
    assert row["review_status"] is None


def test_the_history_is_newest_first_and_the_cursor_moves_forward(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    first = storage.add_event(conn, dict(EVENT, event_id="ev-1"))
    second = storage.add_event(conn, dict(EVENT, event_id="ev-2"))
    assert [r["id"] for r in storage.list_events(conn)] == [second, first]
    assert [r["id"] for r in storage.list_events(conn, since=first)] == [second]


def test_the_history_can_be_narrowed_to_one_camera(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    storage.add_event(conn, dict(EVENT, camera="cam-1"))
    storage.add_event(conn, dict(EVENT, camera="cam-2"))
    assert [r["camera"] for r in storage.list_events(conn, camera="cam-2")] == ["cam-2"]


def test_a_queued_candidate_carries_the_review_row_status(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    review_id = storage.add_review(conn, {"camera": "cam-1", "evaluation": "falling", "combined": 0.4})
    storage.add_event(conn, dict(EVENT, review_id=review_id))
    (row,) = storage.list_events(conn)
    assert row["review_id"] == review_id
    assert row["review_status"] == "candidate"
    storage.set_review_status(conn, review_id, "acknowledged")
    (row,) = storage.list_events(conn)
    assert row["review_status"] == "acknowledged"


def test_the_queue_serializes_the_history_the_same_way_it_serializes_the_reviews(tmp_path):
    queue = storage.ReviewQueue(storage.connect(tmp_path / "fall.db"))
    assert queue.add_event(dict(EVENT)) == 1
    assert len(queue.list_events()) == 1
