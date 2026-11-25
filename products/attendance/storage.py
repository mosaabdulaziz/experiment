"""SQLite persistence for the attendance service. Schema: schema.sql.
Implements the contract agreed via /mattpocock-skills:grilling on 2026-08-26 —
see ~/.claude/mentor/decisions/attendance-system.md.
"""

import sqlite3
from datetime import date, datetime
from pathlib import Path

from domain import DailyAttendanceRecord, Employee, Sighting

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: str) -> sqlite3.Connection:
    """Open (creating if needed) the attendance SQLite database and ensure
    the schema exists."""
    # timeout=10: how long to wait for a lock before raising "database is
    # locked", instead of the sqlite3 default (5s). Every HTTP request opens
    # its OWN fresh connection here (see server.py's route handlers) while
    # the consumer thread holds a separate long-lived one doing its own
    # writes -- SQLite only allows one writer at a time, so some contention
    # under concurrent polling (every 2s, per browser tab) is expected; this
    # just gives it more room to resolve on its own rather than erroring.
    # WAL mode lets readers (most of these connections, most of the time)
    # proceed without waiting on a writer at all, which is the bigger lever.
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA_PATH.read_text())
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS (above) only ever helps a table that
    doesn't exist yet -- it can't add a column to one that already does, and
    this database already has real rows in it. employee.role (2026-09-02,
    attendance v2) is the first column added after the db existed live, so
    this is the first real migration -- idempotent, safe to run on every
    connect()."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(employee)").fetchall()}
    if "role" not in cols:
        conn.execute("ALTER TABLE employee ADD COLUMN role TEXT")
        conn.commit()
    if "email" not in cols:
        conn.execute("ALTER TABLE employee ADD COLUMN email TEXT")
        conn.commit()
    # Backfill seed_state for a database that already has employees today
    # (see schema.sql's comment on seed_state) -- without this, a db that
    # predates this table would look "never seeded" to the new guard below,
    # so deleting everyone and restarting would still reseed the demo
    # roster ONE more time before the marker got set. Only backfill when
    # employee already has rows: an actually-fresh, empty database must NOT
    # get the marker here, or seed_demo_employees.py would wrongly believe
    # it already ran and never seed the initial 13 at all.
    if (conn.execute("SELECT 1 FROM seed_state WHERE key = 'demo_roster'").fetchone() is None
            and conn.execute("SELECT 1 FROM employee LIMIT 1").fetchone() is not None):
        conn.execute("INSERT INTO seed_state (key, seeded_at) VALUES ('demo_roster', datetime('now'))")
        conn.commit()
    # Backfill id_sequence (see schema.sql's comment) from whatever emp-N/
    # person-N ids are in the table today, so a database that predates this
    # fix keeps counting up from where it already was instead of restarting
    # at 0 and immediately colliding with existing rows.
    if conn.execute("SELECT 1 FROM id_sequence WHERE key = 'employee'").fetchone() is None:
        n = 0
        for r in conn.execute("SELECT id, gallery_rec_id FROM employee").fetchall():
            for val, prefix in ((r["id"], "emp-"), (r["gallery_rec_id"], "person-")):
                val = val or ""
                if val.startswith(prefix) and val[len(prefix):].isdigit():
                    n = max(n, int(val[len(prefix):]))
        conn.execute("INSERT INTO id_sequence (key, n) VALUES ('employee', ?)", (n,))
        conn.commit()


def add_employee(conn: sqlite3.Connection, employee: Employee) -> None:
    conn.execute(
        "INSERT INTO employee (id, name, gallery_rec_id, active) VALUES (?, ?, ?, ?)",
        (employee.id, employee.name, employee.gallery_rec_id, int(employee.active)),
    )
    conn.commit()


# ── Admin-driven enrollment (2026-09-02) ────────────────────────────────────

def next_ids(conn: sqlite3.Connection) -> tuple[str, str]:
    """(employee_id, gallery_rec_id) for a new admin-created employee, as
    ("emp-N", "person-N") sharing one N -- the justfile's 13 seeded rows
    already keep these in lockstep (emp-7 <-> person-7); admin-created ones
    continue the same sequence rather than starting a parallel id scheme.

    2026-09-06: N used to be "one past whichever id space is further along"
    among rows CURRENTLY in the table -- which silently reuses a number the
    moment its row is deleted (delete emp-13, create someone new, they
    become emp-13/person-13 again: same id, same face-recognition gallery
    slot, same foreign keys in sighting/desk_interval/violation as whoever
    used to hold it). N now comes from id_sequence (schema.sql), a
    persistent counter that only ever counts up regardless of what employee
    currently contains, so a deleted number is never handed out again."""
    n = conn.execute("SELECT n FROM id_sequence WHERE key = 'employee'").fetchone()["n"] + 1
    conn.execute("UPDATE id_sequence SET n = ? WHERE key = 'employee'", (n,))
    conn.commit()
    return f"emp-{n}", f"person-{n}"


def create_employee(conn: sqlite3.Connection, employee_id: str, name: str, gallery_rec_id: str,
                    role: str | None = None, department: str | None = None,
                    email: str | None = None, code: str | None = None) -> None:
    """Row only -- no gallery/photo side yet, that needs InsightFace (GPU),
    which this process doesn't have (see register_employee_photo.py). The
    row exists with zero embeddings until at least one photo is
    successfully registered against it."""
    conn.execute(
        "INSERT INTO employee (id, name, gallery_rec_id, active, role, department, email, code) "
        "VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
        (employee_id, name, gallery_rec_id, role, department, email, code),
    )
    conn.commit()


def update_employee(conn: sqlite3.Connection, employee_id: str, **fields) -> bool:
    """Patch name/role/department/active/email/code -- only the keys actually
    passed. Returns False if the employee doesn't exist."""
    allowed = {"name", "role", "department", "active", "email", "code"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return conn.execute("SELECT 1 FROM employee WHERE id = ?", (employee_id,)).fetchone() is not None
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    cur = conn.execute(f"UPDATE employee SET {set_clause} WHERE id = ?",
                       (*updates.values(), employee_id))
    conn.commit()
    return cur.rowcount > 0


def delete_employee(conn: sqlite3.Connection, employee_id: str) -> str | None:
    """Full purge of one employee's DB footprint (2026-09-06, admin
    Employees page "remove employee" button) -- every table with an
    employee_id FK, then the employee row itself, in one transaction.
    Unlike `active` (a soft "stop counting attendance" flag the existing
    تعطيل/تنشيط toggle already covers), this is for correcting a mistaken
    enrollment or de-listing someone who's genuinely gone, so history tied
    only to them (sightings, desk intervals, restricted-zone violations)
    goes with it rather than lingering as orphaned rows.

    Returns the deleted employee's gallery_rec_id (the caller still needs it
    to clean up the face gallery entry and enrolled photos, neither of which
    live in this database) or None if the id didn't exist.
    """
    row = conn.execute("SELECT gallery_rec_id FROM employee WHERE id = ?", (employee_id,)).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM sighting WHERE employee_id = ?", (employee_id,))
    conn.execute("DELETE FROM desk_interval WHERE employee_id = ?", (employee_id,))
    conn.execute("DELETE FROM violation WHERE employee_id = ?", (employee_id,))
    conn.execute("DELETE FROM employee WHERE id = ?", (employee_id,))
    conn.commit()
    return row["gallery_rec_id"]


def get_employee_by_rec_id(conn: sqlite3.Connection, rec_id: str) -> Employee | None:
    """Resolve a gallery match's rec_id to an enrolled Employee. Returns None
    if the gallery recognized someone who isn't enrolled as an employee (e.g.
    a face registered for another purpose) — callers must drop the sighting
    in that case, per the "unknowns filtered out" decision."""
    row = conn.execute(
        "SELECT id, name, gallery_rec_id, active FROM employee WHERE gallery_rec_id = ? AND active = 1",
        (rec_id,),
    ).fetchone()
    if row is None:
        return None
    return Employee(id=row["id"], name=row["name"], gallery_rec_id=row["gallery_rec_id"], active=bool(row["active"]))


def get_employee_by_name(conn: sqlite3.Connection, name: str) -> Employee | None:
    """Fallback for a stabilized (anti-flicker locked) recognition entry: the
    sidecar deliberately strips rec_id whenever the locked name differs from
    the current frame's own raw match (face_recognizer.py's
    _stabilize_names, "stabilized label differs from raw frame match") --
    very common (any frame where the face angle doesn't independently clear
    rec_thresh but the locked identity is still correct), yet `name` itself
    stays correct throughout. Without this fallback, SightingCoalescer.ingest
    silently drops every one of those -- found live 2026-08-31 with a
    genuine "Person-6" entry sitting in the stream with rec_id: null."""
    row = conn.execute(
        "SELECT id, name, gallery_rec_id, active FROM employee WHERE name = ? AND active = 1",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return Employee(id=row["id"], name=row["name"], gallery_rec_id=row["gallery_rec_id"], active=bool(row["active"]))


def upsert_sighting(conn: sqlite3.Connection, sighting: Sighting) -> None:
    """Insert a new coalesced bucket, or extend an existing one's last_seen_at.
    A bucket is identified by (employee_id, camera, first_seen_at) — the
    SightingCoalescer guarantees first_seen_at is stable for a given bucket's
    lifetime, so it doubles as the bucket's natural key here."""
    existing = conn.execute(
        "SELECT id FROM sighting WHERE employee_id = ? AND camera = ? AND first_seen_at = ?",
        (sighting.employee_id, sighting.camera, sighting.first_seen_at.isoformat()),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO sighting (id, employee_id, camera, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
            (sighting.id, sighting.employee_id, sighting.camera,
             sighting.first_seen_at.isoformat(), sighting.last_seen_at.isoformat()),
        )
    else:
        conn.execute(
            "UPDATE sighting SET last_seen_at = ? WHERE id = ?",
            (sighting.last_seen_at.isoformat(), existing["id"]),
        )
    conn.commit()


def move_sighting_first_seen(conn: sqlite3.Connection, sighting_id: str,
                             new_first_seen: datetime) -> None:
    """Pull an existing sighting's first_seen_at backward (retroactive
    back-turned credit, see SightingCoalescer.backdate_first_seen). Keyed on
    the row id, NOT the (employee, camera, first_seen_at) natural key that
    upsert_sighting uses -- first_seen_at is exactly what's changing here,
    so upserting through the natural key would fork a duplicate row instead
    of moving this one."""
    conn.execute(
        "UPDATE sighting SET first_seen_at = ? WHERE id = ?",
        (new_first_seen.isoformat(), sighting_id),
    )
    conn.commit()


def clear_all_sightings(conn: sqlite3.Connection) -> None:
    """Delete every Sighting row. Used to reset the table when a streaming
    camera's video loops back to its start -- compute_daily_attendance's
    MIN/MAX aggregation has no notion of "lap", so without this, every lap
    the dev-loop streamer plays through keeps merging into the same running
    total forever. See ConsumerThread's wrap detection (consumer.py)."""
    conn.execute("DELETE FROM sighting")
    conn.commit()


def clear_sightings_for_cameras(conn: sqlite3.Connection, cameras) -> int:
    """Delete the Sighting rows of just these cameras (every day). Used by the
    lap reset (2026-09-13): a looping TEST recording resets only the test
    cameras' rows, while live cameras' rows -- which must persist from day to
    day -- are never touched. clear_all_sightings stays for `just clear`.
    Returns the number of rows deleted."""
    cams = list(cameras)
    if not cams:
        return 0
    cur = conn.execute(f"DELETE FROM sighting WHERE camera IN ({','.join('?' * len(cams))})", cams)
    conn.commit()
    return cur.rowcount


def sightings_for_day(conn: sqlite3.Connection, employee_id: str, day: date) -> list[Sighting]:
    """Every raw sighting bucket for one employee, one day, across every
    camera, sorted by first_seen_at. Unlike compute_daily_attendance's
    MIN/MAX, this keeps each bucket's own camera and span -- what the
    session-window algorithm (sessions.py) needs to tell an exit from a
    break and know which camera each touch happened on."""
    rows = conn.execute(
        "SELECT id, employee_id, camera, first_seen_at, last_seen_at FROM sighting "
        "WHERE employee_id = ? AND date(first_seen_at) = ? ORDER BY first_seen_at",
        (employee_id, day.isoformat()),
    ).fetchall()
    return [
        Sighting(id=r["id"], employee_id=r["employee_id"], camera=r["camera"],
                 first_seen_at=datetime.fromisoformat(r["first_seen_at"]),
                 last_seen_at=datetime.fromisoformat(r["last_seen_at"]))
        for r in rows
    ]


def compute_daily_attendance(conn: sqlite3.Connection, employee_id: str, day: date) -> DailyAttendanceRecord:
    """Pure query over raw Sighting facts — no live state involved. Absence
    is simply the absence of any matching row."""
    row = conn.execute(
        """
        SELECT MIN(first_seen_at) AS first_arrival, MAX(last_seen_at) AS last_departure
        FROM sighting
        WHERE employee_id = ? AND date(first_seen_at) = ?
        """,
        (employee_id, day.isoformat()),
    ).fetchone()

    if row is None or row["first_arrival"] is None:
        return DailyAttendanceRecord(employee_id=employee_id, day=day, first_arrival=None,
                                      last_departure=None, present=False)

    return DailyAttendanceRecord(
        employee_id=employee_id,
        day=day,
        first_arrival=datetime.fromisoformat(row["first_arrival"]),
        last_departure=datetime.fromisoformat(row["last_departure"]),
        present=True,
    )


# ── Productivity monitoring: desk-state intervals (2026-09-01) ─────────────

def open_desk_interval(conn: sqlite3.Connection, interval_id: str, employee_id: str,
                       state: str, started_at: datetime) -> None:
    conn.execute(
        "INSERT INTO desk_interval (id, employee_id, state, started_at, ended_at, open) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (interval_id, employee_id, state, started_at.isoformat(), started_at.isoformat()),
    )
    conn.commit()


def heartbeat_desk_interval(conn: sqlite3.Connection, interval_id: str, now: datetime) -> None:
    """Advance an open interval's end so live counters stay current."""
    conn.execute("UPDATE desk_interval SET ended_at = ? WHERE id = ? AND open = 1",
                 (now.isoformat(), interval_id))
    conn.commit()


def close_desk_interval(conn: sqlite3.Connection, interval_id: str, ended_at: datetime) -> None:
    conn.execute("UPDATE desk_interval SET ended_at = ?, open = 0 WHERE id = ?",
                 (ended_at.isoformat(), interval_id))
    conn.commit()


def clear_all_desk_intervals(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM desk_interval")
    conn.commit()


def desk_seconds_by_state(conn: sqlite3.Connection, employee_id: str, day: date) -> dict[str, float]:
    """{state: total seconds} for one employee on one day, open intervals
    counted up to their heartbeat."""
    rows = conn.execute(
        "SELECT state, started_at, ended_at FROM desk_interval "
        "WHERE employee_id = ? AND date(started_at) = ?",
        (employee_id, day.isoformat()),
    ).fetchall()
    totals: dict[str, float] = {}
    for r in rows:
        secs = (datetime.fromisoformat(r["ended_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds()
        totals[r["state"]] = totals.get(r["state"], 0.0) + max(0.0, secs)
    return totals


def has_open_desk_interval(conn: sqlite3.Connection, employee_id: str, state: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM desk_interval WHERE employee_id = ? AND state = ? AND open = 1 LIMIT 1",
        (employee_id, state),
    ).fetchone()
    return row is not None


# ── Restricted-zone violations (2026-09-02) ─────────────────────────────────

def open_violation(conn: sqlite3.Connection, violation_id: str, employee_id: str | None,
                   camera: str, zone_id: str, violation_type: str, started_at: datetime) -> None:
    conn.execute(
        "INSERT INTO violation (id, employee_id, camera, zone_id, violation_type, started_at, ended_at, open) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
        (violation_id, employee_id, camera, zone_id, violation_type,
         started_at.isoformat(), started_at.isoformat()),
    )
    conn.commit()


def heartbeat_violation(conn: sqlite3.Connection, violation_id: str, now: datetime) -> None:
    conn.execute("UPDATE violation SET ended_at = ? WHERE id = ? AND open = 1",
                 (now.isoformat(), violation_id))
    conn.commit()


def close_violation(conn: sqlite3.Connection, violation_id: str, ended_at: datetime) -> None:
    conn.execute("UPDATE violation SET ended_at = ?, open = 0 WHERE id = ?",
                 (ended_at.isoformat(), violation_id))
    conn.commit()


def close_stale_violations(conn: sqlite3.Connection) -> int:
    """Close every violation a PREVIOUS process left open; returns how many. The engine knows which rows are open only
    in memory, so when the server stops (or crashes) with an alert open nothing would ever close its row: it stayed
    "running" on the dashboard, and popped up again on every page load, for ever. ended_at already holds the last
    time the engine saw the person (heartbeat_violation), so closing keeps the true end of the stay."""
    cur = conn.execute("UPDATE violation SET open = 0 WHERE open = 1")
    conn.commit()
    return cur.rowcount


def clear_all_violations(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM violation")
    conn.commit()


def clear_violations_for_day(conn: sqlite3.Connection, day: date) -> int:
    """Delete the violation rows that started on this one day (2026-09-13,
    the dashboard's "clear log" button). Sightings, desk intervals and
    employees are never touched. Returns the number of rows deleted."""
    cur = conn.execute("DELETE FROM violation WHERE date(started_at) = ?", (day.isoformat(),))
    conn.commit()
    return cur.rowcount


def recent_violations(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Most recent violations first, open ones included -- feeds both the
    live events panel (open=1 rows) and a simple history list (the rest)."""
    rows = conn.execute(
        "SELECT id, employee_id, camera, zone_id, violation_type, started_at, ended_at, open "
        "FROM violation ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r, open=bool(r["open"])) for r in rows]  # sqlite int -> real JSON boolean
