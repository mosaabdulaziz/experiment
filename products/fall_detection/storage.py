"""The review queue: the candidates that were too weak to confirm, kept with their evidence.

connect() follows the repo's product pattern (busy timeout, WAL, the schema on every connect);
the tables are this product's own -- schema.sql.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# The three statuses the operator may set, the same three the vendor's operator API defines.
STATUSES = ("acknowledged", "confirmed", "dismissed")


def connect(db_path) -> sqlite3.Connection:
    """Open (creating if needed) the database and ensure the schema exists.

    The consumer's worker threads and the API's request threads share this connection, so the
    queue serializes their use with a lock (ReviewQueue); WAL also lets another process read
    while this one writes, and timeout=10 gives the writer room to resolve that contention.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def add_review(conn: sqlite3.Connection, evidence: dict) -> int:
    """Store one candidate for review and return its row id."""
    cursor = conn.execute(
        """INSERT INTO fall_review (camera, pts, track, event_id, kind, evaluation, combined,
                                   path_scores, top_class, clip_window, probs, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (evidence.get("camera"), evidence.get("pts"), evidence.get("track"), evidence.get("event_id"),
         evidence.get("kind"), evidence.get("evaluation"), float(evidence.get("combined") or 0.0),
         json.dumps(evidence.get("path_scores") or {}), evidence.get("top_class"),
         json.dumps(evidence.get("clip_window")) if evidence.get("clip_window") is not None else None,
         json.dumps(evidence.get("probs")) if evidence.get("probs") is not None else None,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()
    return int(cursor.lastrowid)


def add_event(conn: sqlite3.Connection, row: dict) -> int:
    """Store one scored candidate in the product's own history and return its row id."""
    cursor = conn.execute(
        """INSERT INTO fall_event (camera, pts, track, event_id, kind, evaluation, combined,
                                   severity, path_scores, clip_window, review_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (row.get("camera"), row.get("pts"), row.get("track"), row.get("event_id"),
         row.get("kind"), row.get("evaluation"), float(row.get("combined") or 0.0),
         row.get("severity"), json.dumps(row.get("path_scores") or {}),
         json.dumps(row.get("clip_window")) if row.get("clip_window") is not None else None,
         row.get("review_id"), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()
    return int(cursor.lastrowid)


def list_events(conn: sqlite3.Connection, camera: str | None = None, since: int = 0,
                limit: int = 50) -> list[dict]:
    """The history, newest first, with the review row's status beside it when it has one."""
    query = ("SELECT e.*, r.status AS review_status "
             "FROM fall_event e LEFT JOIN fall_review r ON r.id = e.review_id "
             "WHERE e.id > ?")
    params: list = [int(since)]
    if camera:
        query += " AND e.camera = ?"
        params.append(camera)
    query += " ORDER BY e.id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 200)))
    return [_as_dict(row) for row in conn.execute(query, params)]


def set_review_status(conn: sqlite3.Connection, row_id: int, status: str, note: str = "") -> None:
    """Set the operator's verdict on a queued candidate."""
    if status not in STATUSES:
        raise ValueError(f"review status must be one of {', '.join(STATUSES)}, not {status!r}")
    cursor = conn.execute("UPDATE fall_review SET status = ?, note = ? WHERE id = ?", (status, note, row_id))
    conn.commit()
    if not cursor.rowcount:
        raise KeyError(f"no review row {row_id}")


def list_reviews(conn: sqlite3.Connection, camera: str | None = None,
                 status: str | None = None) -> list[dict]:
    """The queue, newest first."""
    query = "SELECT * FROM fall_review"
    clauses, params = [], []
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"
    return [_as_dict(row) for row in conn.execute(query, params)]


def prune(conn: sqlite3.Connection, *, keep_events: int = 5000, review_days: int = 30) -> tuple[int, int]:
    """The retention sweep: keep the newest `keep_events` history rows, and let a terminal
    review row go once it is `review_days` old. A 'candidate' stays whatever its age -- it is
    the operator's work list. The stamps are ISO-8601 UTC, so they order lexicographically."""
    events = conn.execute(
        "DELETE FROM fall_event WHERE id NOT IN "
        "(SELECT id FROM fall_event ORDER BY id DESC LIMIT ?)",
        (max(1, int(keep_events)),)).rowcount
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=int(review_days))).isoformat(timespec="seconds")
    reviews = conn.execute(
        "DELETE FROM fall_review WHERE status != 'candidate' AND created_at < ?",
        (cutoff,)).rowcount
    conn.commit()
    return events, reviews


def _as_dict(row: sqlite3.Row) -> dict:
    out = dict(row)
    out["path_scores"] = json.loads(out["path_scores"]) if out.get("path_scores") else {}
    out["clip_window"] = json.loads(out["clip_window"]) if out.get("clip_window") else None
    out["probs"] = json.loads(out["probs"]) if out.get("probs") else None
    return out


class ReviewQueue:
    """The queue over one connection -- the shape the consumer talks to.

    The camera workers write from their own threads while FastAPI serving the review endpoints
    runs on a thread pool, and a sqlite3 connection is not safe to use concurrently: this lock
    is the one place that serializes them.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def add_review(self, row: dict) -> int:
        with self._lock:
            return add_review(self._conn, row)

    def add_event(self, row: dict) -> int:
        with self._lock:
            return add_event(self._conn, row)

    def list_events(self, camera: str | None = None, since: int = 0, limit: int = 50) -> list[dict]:
        with self._lock:
            return list_events(self._conn, camera, since, limit)

    def set_review_status(self, row_id: int, status: str, note: str = "") -> None:
        with self._lock:
            set_review_status(self._conn, row_id, status, note)

    def list_reviews(self, camera: str | None = None, status: str | None = None) -> list[dict]:
        with self._lock:
            return list_reviews(self._conn, camera, status)

    def prune(self, *, keep_events: int = 5000, review_days: int = 30) -> tuple[int, int]:
        """The retention sweep over the queue's lock; (events dropped, reviews dropped)."""
        with self._lock:
            return prune(self._conn, keep_events=keep_events, review_days=review_days)

    def close(self) -> None:
        """Close the connection the queue holds."""
        with self._lock:
            self._conn.close()
