"""SQLite storage and thread-safe review queue for suspected concealment candidates and assessments.

Provenance: adapted from `products/fall_detection/storage.py` (`connect`, `ReviewQueue`).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading

from concealment.types import REVIEW_STATUSES, TARGET_KINDS
from evidence import validate_non_accusatory_text

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
STAFF_DECISION_STATUSES = ("acknowledged", "concern_noted", "dismissed")
ALL_STATUSES = REVIEW_STATUSES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path) -> sqlite3.Connection:
    """Open SQLite connection with WAL mode and ensure `schema.sql` is applied."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    # CREATE TABLE IF NOT EXISTS cannot upgrade databases created by older versions.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(concealment_candidates)")}
    for name, sql_type in (
        ("terminal_stage", "TEXT"),
        ("hold_duration_s", "REAL"),
        ("dwell_duration_s", "REAL"),
        ("thumbnail_path", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE concealment_candidates ADD COLUMN {name} {sql_type}")
    conn.commit()
    return conn


def _row_to_candidate(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["stage_timeline"] = json.loads(d.pop("stage_timeline_json") or "[]")
    d["clip_window"] = json.loads(d.pop("clip_window_json") or "[]")
    d["keyframes_pts"] = json.loads(d.pop("keyframes_pts_json") or "[]")
    d["requires_human_verification"] = bool(d.get("requires_human_verification", 1))
    return d


def add_candidate(conn: sqlite3.Connection, record: dict) -> int:
    """Insert a suspected concealment candidate and return its row ID."""
    status = str(record.get("status") or "candidate")
    if status not in ALL_STATUSES:
        raise ValueError(f"Invalid candidate status '{status}'; 'confirmed' is never permitted")
    target_kind = str(record["target_kind"])
    if target_kind not in TARGET_KINDS:
        raise ValueError(f"Invalid target_kind '{target_kind}'; expected one of {TARGET_KINDS}")
    description = str(record.get("description") or "")
    validate_non_accusatory_text(description)

    cur = conn.execute(
        """
        INSERT INTO concealment_candidates (
            camera, location, track_id, pts, display_time,
            tier, severity, category, kind, score,
            zone_id, target_kind, body_zone_name, description,
            stage_timeline_json, clip_window_json, keyframes_pts_json,
            clip_path, requires_human_verification, status,
            reviewer, reviewed_at, note, created_at,
            terminal_stage, hold_duration_s, dwell_duration_s, thumbnail_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(record["camera"]),
            str(record.get("location") or record["camera"]),
            int(record["track_id"]),
            float(record["pts"]),
            str(record.get("display_time") or _now_iso()),
            str(record["tier"]),
            str(record["severity"]),
            str(record.get("category") or "security"),
            str(record.get("kind") or "suspected_concealment"),
            float(record["score"]),
            str(record.get("zone_id") or "zone_free"),
            str(record["target_kind"]),
            str(record["body_zone_name"]),
            description,
            json.dumps(record.get("stage_timeline") or []),
            json.dumps(record.get("clip_window") or []),
            json.dumps(record.get("keyframes_pts") or []),
            record.get("clip_path"),
            1 if record.get("requires_human_verification", True) else 0,
            status,
            record.get("reviewer"),
            record.get("reviewed_at"),
            str(record.get("note") or ""),
            str(record.get("created_at") or _now_iso()),
            record.get("terminal_stage"),
            record.get("hold_duration_s"),
            record.get("dwell_duration_s"),
            record.get("thumbnail_path"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def attach_clip(conn: sqlite3.Connection, candidate_id: int, clip_path: str) -> None:
    """Attach a recorded video clip path to an existing candidate."""
    conn.execute(
        "UPDATE concealment_candidates SET clip_path = ? WHERE id = ?",
        (str(clip_path), int(candidate_id)),
    )
    conn.commit()


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> dict | None:
    """Return one candidate dict by ID, or `None` if not found."""
    row = conn.execute(
        "SELECT * FROM concealment_candidates WHERE id = ?",
        (int(candidate_id),),
    ).fetchone()
    return _row_to_candidate(row) if row is not None else None


def list_candidates(
    conn: sqlite3.Connection,
    camera: str | None = None,
    status: str | None = None,
    since: int = 0,
    limit: int = 50,
    before: int | None = None,
) -> list[dict]:
    """List newest-first with exclusive older-page (before) and newer-update (since) cursors."""
    clauses = ["id > ?"]
    params: list[object] = [int(since)]
    if before is not None:
        clauses.append("id < ?")
        params.append(int(before))
    if camera:
        clauses.append("camera = ?")
        params.append(str(camera))
    if status:
        clauses.append("status = ?")
        params.append(str(status))
    where_sql = " AND ".join(clauses)
    params.append(max(1, min(int(limit), 500)))
    rows = conn.execute(
        f"SELECT * FROM concealment_candidates WHERE {where_sql} ORDER BY id DESC LIMIT ?",
        tuple(params),
    ).fetchall()
    return [_row_to_candidate(r) for r in rows]


def set_review_status(
    conn: sqlite3.Connection,
    candidate_id: int,
    status: str,
    note: str = "",
    reviewer: str = "staff",
) -> None:
    """Set staff review verdict (`acknowledged` | `concern_noted` | `dismissed`; never `confirmed`)."""
    if status not in STAFF_DECISION_STATUSES:
        raise ValueError(
            f"Invalid review status '{status}'. Allowed verdicts: {STAFF_DECISION_STATUSES} (never 'confirmed')."
        )
    cur = conn.execute(
        """
        UPDATE concealment_candidates
           SET status = ?, note = ?, reviewer = ?, reviewed_at = ?
         WHERE id = ?
        """,
        (str(status), str(note or ""), str(reviewer or "staff"), _now_iso(), int(candidate_id)),
    )
    conn.commit()
    if cur.rowcount == 0:
        raise KeyError(f"Candidate {candidate_id} not found")


def add_assessment(conn: sqlite3.Connection, row: dict) -> int:
    """Insert an `insufficient_evidence` assessment log row and return its ID."""
    cur = conn.execute(
        """
        INSERT INTO concealment_assessments (camera, track_id, pts, state, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["camera"]),
            int(row.get("track_id", -1)),
            float(row.get("pts", 0.0)),
            str(row.get("state") or "insufficient_evidence"),
            str(row.get("reason") or ""),
            str(row.get("created_at") or _now_iso()),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_assessments(
    conn: sqlite3.Connection,
    camera: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List assessment log rows newest-first."""
    cap = max(1, min(int(limit), 500))
    if camera:
        rows = conn.execute(
            "SELECT * FROM concealment_assessments WHERE camera = ? ORDER BY id DESC LIMIT ?",
            (str(camera), cap),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM concealment_assessments ORDER BY id DESC LIMIT ?",
            (cap,),
        ).fetchall()
    return [dict(r) for r in rows]


def prune(
    conn: sqlite3.Connection,
    *,
    keep_candidates: int = 5000,
    review_days: int = 30,
) -> tuple[int, int]:
    """Prune old assessments and terminal reviewed candidates; return `(assessments_dropped, candidates_dropped)`."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(review_days))).isoformat()
    c_ass = conn.execute(
        """
        DELETE FROM concealment_assessments
         WHERE id NOT IN (
             SELECT id FROM concealment_assessments ORDER BY id DESC LIMIT ?
         )
        """,
        (int(keep_candidates),),
    )
    c_cand = conn.execute(
        """
        DELETE FROM concealment_candidates
         WHERE status IN ('acknowledged', 'concern_noted', 'dismissed')
           AND reviewed_at IS NOT NULL
           AND reviewed_at < ?
        """,
        (cutoff,),
    )
    conn.commit()
    return (int(c_ass.rowcount or 0), int(c_cand.rowcount or 0))


class ReviewStore:
    """Thread-safe wrapper over one SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def add_candidate(self, record: dict) -> int:
        with self._lock:
            return add_candidate(self._conn, record)

    def attach_clip(self, candidate_id: int, clip_path: str) -> None:
        with self._lock:
            attach_clip(self._conn, candidate_id, clip_path)

    def get_candidate(self, candidate_id: int) -> dict | None:
        with self._lock:
            return get_candidate(self._conn, candidate_id)

    def list_candidates(
        self,
        camera: str | None = None,
        status: str | None = None,
        since: int = 0,
        limit: int = 50,
        before: int | None = None,
    ) -> list[dict]:
        with self._lock:
            return list_candidates(self._conn, camera=camera, status=status, since=since, limit=limit, before=before)

    def set_review_status(
        self,
        candidate_id: int,
        status: str,
        note: str = "",
        reviewer: str = "staff",
    ) -> None:
        with self._lock:
            set_review_status(self._conn, candidate_id, status, note=note, reviewer=reviewer)

    def add_assessment(self, row: dict) -> int:
        with self._lock:
            return add_assessment(self._conn, row)

    def list_assessments(self, camera: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            return list_assessments(self._conn, camera=camera, limit=limit)

    def prune(self, *, keep_candidates: int = 5000, review_days: int = 30) -> tuple[int, int]:
        with self._lock:
            return prune(self._conn, keep_candidates=keep_candidates, review_days=review_days)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
