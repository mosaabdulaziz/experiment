"""The review queue in SQLite: persisted evidence, restart survival, the three operator statuses.

ACs: AC-F22.1, AC-F22.2. Design: FALL_DETECTION_DESIGN.md §5.4 (connect() forked from the barber's:
busy timeout, WAL and the schema on every connect).
"""
import threading

import pytest

import storage
from fall_helpers import DECISIVE_FALL

EVIDENCE = {
    "camera": "cam-1",
    "track": 3,
    "event_id": "ev-abc123",
    "kind": "fall",
    "evaluation": "falling",
    "combined": 0.45,
    "path_scores": {"action": None, "kinematic": 0.9, "cold": None},
    "top_class": 6,
    "clip_window": [98.0, 105.5],
    "pts": 103.5,
    "probs": DECISIVE_FALL,
}


def test_a_review_row_survives_a_restart_with_all_of_its_evidence(tmp_path):
    db = tmp_path / "fall.db"
    conn = storage.connect(db)
    row_id = storage.add_review(conn, EVIDENCE)
    conn.close()
    conn2 = storage.connect(db)                              # the restart
    (row,) = [r for r in storage.list_reviews(conn2) if r["id"] == row_id]
    assert row["evaluation"] == "falling"
    assert row["combined"] == pytest.approx(0.45)
    assert row["path_scores"] == {"action": None, "kinematic": 0.9, "cold": None}
    assert row["top_class"] == 6
    assert row["clip_window"] == [98.0, 105.5]
    assert row["camera"] == "cam-1" and row["pts"] == pytest.approx(103.5)
    assert row["status"] == "candidate"


def test_only_the_three_operator_statuses_are_accepted_and_the_note_is_stored(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    row_id = storage.add_review(conn, EVIDENCE)
    for status in ("acknowledged", "confirmed", "dismissed"):
        storage.set_review_status(conn, row_id, status, note=f"seen by op ({status})")
        (row,) = [r for r in storage.list_reviews(conn) if r["id"] == row_id]
        assert row["status"] == status
        assert row["note"] == f"seen by op ({status})"
    for bad in ("bogus", "resolved", "", "CANDIDATE"):
        with pytest.raises(ValueError):
            storage.set_review_status(conn, row_id, bad)


def test_the_queue_serializes_the_shared_connection_across_the_workers_and_the_api(tmp_path):
    """The workers write from their own threads while the API reads from FastAPI's pool.

    A sqlite3 connection is not safe to share concurrently, so the queue's lock is the one
    thing keeping this legal; without it the inserts collide on the connection.
    """
    queue = storage.ReviewQueue(storage.connect(tmp_path / "fall.db"))
    errors = []

    def writer(track):
        try:
            for i in range(50):
                queue.add_review(dict(EVIDENCE, track=track, event_id=f"ev-{track}-{i}"))
        except Exception as exc:                              # noqa: BLE001 -- reported, not raised
            errors.append(exc)

    def reader():
        try:
            for _ in range(200):
                queue.list_reviews()
        except Exception as exc:                              # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(track,)) for track in range(8)]
    threads.append(threading.Thread(target=reader))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(queue.list_reviews()) == 400                   # every insert landed
