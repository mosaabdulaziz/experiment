"""SQLite persistence for the barber service. Schema: schema.sql (three cameras).

connect() is forked from products/attendance/storage.py @ 1f5b27b (busy
timeout, WAL, schema + migrations on every connect) -- changes: everything
else is the barber tables.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from domain import VOID_STAFF, Alert, EngineUpdate, Registration, ServiceSession, Visit

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

INTERRUPTED = "interrupted"  # visit outcome: still open when the service stopped


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt is not None else None


def connect(db_path: str) -> sqlite3.Connection:
    """Open (creating if needed) the database and ensure the schema exists.
    Every HTTP request opens its own connection while the consumer thread
    holds a long-lived one: WAL lets readers proceed without waiting on the
    writer, and timeout=10 gives writer contention room to resolve."""
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA_PATH.read_text())
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent upgrades of databases that predate a column."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(visit)").fetchall()}
    if "customer_id" not in cols:
        _add_column(conn, "visit", "customer_id", "TEXT")
    if "ticket" not in cols:
        _add_column(conn, "visit", "ticket", "INTEGER")
    bcols = {r["name"] for r in conn.execute("PRAGMA table_info(customer_body)").fetchall()}
    if bcols and "camera" not in bcols:
        _add_column(conn, "customer_body", "camera", "TEXT")    # 2026-09-29: body re-ID compares per camera
    if "wait_unseen" not in cols:
        _add_column(conn, "visit", "wait_unseen", "INTEGER NOT NULL DEFAULT 0")
        # 2026-09-28: older visits whose person was first joined seated in a chair (binding log)
        conn.execute("""UPDATE visit SET wait_unseen = 1 WHERE person_id IN (
                            SELECT b.person_id FROM binding b WHERE b.kind = 'seat_start'
                            AND b.id = (SELECT MIN(id) FROM binding WHERE person_id = b.person_id))""")
    scols = {r["name"] for r in conn.execute("PRAGMA table_info(service_session)").fetchall()}
    if scols and "occupancy" not in scols:                      # 2026-11-14: ce_low_occupancy
        _add_column(conn, "service_session", "occupancy", "REAL")
    if scols and "low_occupancy" not in scols:
        _add_column(conn, "service_session", "low_occupancy", "INTEGER NOT NULL DEFAULT 0")
    acols = {r["name"] for r in conn.execute("PRAGMA table_info(barber_alert)").fetchall()}
    for col in ("subject_cam", "subject_box", "snapshot_path"):
        if acols and col not in acols:
            _add_column(conn, "barber_alert", col, "TEXT")
    dcols = {r["name"] for r in conn.execute("PRAGMA table_info(door_crossing)").fetchall()}
    for col in ("entry_id", "customer_id"):
        if dcols and col not in dcols:
            _add_column(conn, "door_crossing", col, "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_visit_customer ON visit(customer_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_door_crossing_customer ON door_crossing(customer_id)")
    conn.execute("DROP TABLE IF EXISTS customer_crop")
    conn.commit()


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """ALTER TABLE ... ADD COLUMN, tolerating a concurrent starter that won the race."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


# ── engine output ───────────────────────────────────────────────────────────

def apply_update(conn: sqlite3.Connection, update: EngineUpdate, now: datetime) -> None:
    """Persist one ShopEngine.observe() result atomically. Visits first:
    sessions and events reference them."""
    with conn:
        for v in update.visits.values():
            _upsert_visit(conn, v, now)
        for session_id in update.deleted_sessions:
            conn.execute("DELETE FROM service_session WHERE id = ?", (session_id,))
            # 2026-11-10 (staff haircuts): a session deleted after it was written takes its sittings and alerts along
            conn.execute("DELETE FROM chair_sitting WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM barber_alert WHERE session_id = ?", (session_id,))
        for s in update.sessions.values():
            _upsert_session(conn, s)
        for a in update.alerts:
            _upsert_alert(conn, a)
        conn.executemany(
            "INSERT INTO state_event (visit_id, from_state, to_state, area, camera, at) VALUES (?, ?, ?, ?, ?, ?)",
            [(e.visit_id, e.from_state, e.to_state, e.area, e.camera, iso(e.at)) for e in update.events])
        conn.executemany(
            "INSERT INTO handoff (area, camera_a, track_a, camera_b, track_b, offset_s, reid_sim, at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(h.area, h.camera_a, h.track_a, h.camera_b, h.track_b, h.offset_s, h.reid_sim, iso(h.at))
             for h in update.handoffs])
        conn.executemany(
            "INSERT INTO binding (kind, camera, track_id, person_id, area, partner_camera, partner_track,"
            " reid_sim, offset_s, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(b.kind, b.camera, b.track_id, b.person_id, b.area, b.partner_camera, b.partner_track,
              b.reid_sim, b.offset_s, iso(b.at)) for b in update.bindings])
        conn.executemany(
            "INSERT INTO door_crossing (at, direction, camera, track_id, staff, entry_id) VALUES (?, ?, ?, ?, ?, ?)",
            [(iso(c.at), c.direction, c.camera, c.track_id, int(c.staff), c.entry_id) for c in update.door_crossings])
        conn.executemany(
            "INSERT INTO chair_sitting (session_id, n, chair_id, started_at, ended_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(session_id, n) DO UPDATE SET ended_at = excluded.ended_at",
            [(r.session_id, r.n, r.chair_id, iso(r.started_at), iso(r.ended_at)) for r in update.sittings])
        conn.executemany(
            "INSERT INTO customer_wait (id, started_at, ended_at, wait_s, ended_by, target_s, camera, track_id,"
            " last_track_id, ticket, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET ended_at = excluded.ended_at, wait_s = excluded.wait_s,"
            " ended_by = excluded.ended_by, last_track_id = excluded.last_track_id,"
            " last_seen_at = excluded.last_seen_at",
            [(w.id, iso(w.started_at), iso(w.ended_at), w.wait_s, w.ended_by, w.target_s, w.camera, w.track_id,
              w.last_track_id, w.ticket, iso(w.last_seen_at)) for w in update.waits.values()])
        conn.executemany(
            "INSERT INTO resume_check (chair_id, kind, sat_at, decided_at, gap_s, sim, threshold, resumed, haircut_id,"
            " prev_haircut_id, ref_samples, new_samples) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r.chair_id, r.kind, iso(r.sat_at), iso(r.decided_at), r.gap_s, r.sim, r.threshold, int(r.resumed),
              r.haircut_id, r.prev_haircut_id, r.ref_samples, r.new_samples) for r in update.resume_checks])
        conn.executemany(
            "INSERT INTO pedicure_session (id, chair_id, started_at, ended_at, service_s, open) VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET ended_at = excluded.ended_at, service_s = excluded.service_s,"
            " open = excluded.open",
            [(p.id, p.chair_id, iso(p.started_at), iso(p.ended_at), p.service_s, int(p.open))
             for p in update.pedicures.values()])


def _upsert_visit(conn: sqlite3.Connection, v: Visit, now: datetime) -> None:
    conn.execute(
        """INSERT INTO visit (id, person_id, entered_at, entry_observed, state, waiting_started_at,
               service_started_at, wait_s, wait_target_s, excessive_wait, haircuts, exited_at, exit_reason,
               outcome, ticket, open, updated_at, wait_unseen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               state = excluded.state, waiting_started_at = excluded.waiting_started_at,
               service_started_at = excluded.service_started_at, wait_s = excluded.wait_s,
               wait_target_s = excluded.wait_target_s, excessive_wait = excluded.excessive_wait,
               haircuts = excluded.haircuts, exited_at = excluded.exited_at, exit_reason = excluded.exit_reason,
               outcome = excluded.outcome, ticket = excluded.ticket, open = excluded.open,
               updated_at = excluded.updated_at, wait_unseen = excluded.wait_unseen""",
        (v.id, v.person_id, iso(v.entered_at), int(v.entry_observed), v.state, iso(v.waiting_started_at),
         iso(v.service_started_at), v.wait_s, v.wait_target_s, int(v.excessive_wait), v.haircuts,
         iso(v.exited_at), v.exit_reason, v.outcome, v.ticket, int(v.open), iso(now), int(v.wait_unseen)),
    )


def _upsert_session(conn: sqlite3.Connection, s: ServiceSession) -> None:
    conn.execute(
        """INSERT INTO service_session (id, visit_id, chair_id, started_at, ended_at, service_s, wash_s,
               barber_id, service_target_s, excessive_service, open, occupancy, low_occupancy)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               ended_at = excluded.ended_at, service_s = excluded.service_s, wash_s = excluded.wash_s,
               barber_id = excluded.barber_id, excessive_service = excluded.excessive_service,
               open = excluded.open, occupancy = excluded.occupancy, low_occupancy = excluded.low_occupancy""",
        (s.id, s.visit_id, s.chair_id, iso(s.started_at), iso(s.ended_at), s.service_s, s.wash_s,
         s.barber_id, s.service_target_s, int(s.excessive_service), int(s.open), s.occupancy, int(s.low_occupancy)),
    )


def _upsert_alert(conn: sqlite3.Connection, a: Alert) -> None:
    conn.execute(
        """INSERT INTO barber_alert (id, kind, camera, visit_id, session_id, started_at, ended_at, open,
                                    subject_cam, subject_box)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET ended_at = excluded.ended_at, open = excluded.open,
               subject_cam = COALESCE(barber_alert.subject_cam, excluded.subject_cam),
               subject_box = COALESCE(barber_alert.subject_box, excluded.subject_box)""",
        (a.id, a.kind, a.camera, a.visit_id, a.session_id, iso(a.started_at), iso(a.ended_at), int(a.open),
         a.subject_cam, json.dumps(list(a.subject_box)) if a.subject_box else None),
    )


def close_interrupted(conn: sqlite3.Connection) -> dict[str, int]:
    """At startup: rows still open belong to a previous run whose in-memory
    engine state is gone. Mark them closed and unmeasured (service_s stays
    NULL, so an interrupted haircut never skews the stats). Open waits close at their last sighting and count."""
    with conn:
        visits = conn.execute("UPDATE visit SET open = 0, outcome = ? WHERE open = 1", (INTERRUPTED,)).rowcount
        sessions = conn.execute("UPDATE service_session SET open = 0 WHERE open = 1").rowcount
        pedicures = conn.execute("UPDATE pedicure_session SET open = 0 WHERE open = 1").rowcount
        # 2026-10-05 (user): an alert a restart cut short is stale -- the engine re-detects its wait or haircut and
        # fires a fresh one -- so it is deleted, not left closed without an end (its picture goes with the orphan sweep)
        alerts = conn.execute("DELETE FROM barber_alert WHERE open = 1").rowcount
        waits = conn.execute(
            "UPDATE customer_wait SET ended_at = last_seen_at, ended_by = 'restart',"
            " wait_s = MAX(0.0, ROUND((julianday(last_seen_at) - julianday(started_at)) * 86400.0, 1))"
            " WHERE ended_at IS NULL").rowcount
    return {"visits": visits, "sessions": sessions, "pedicures": pedicures, "alerts": alerts, "waits": waits}


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    r = conn.execute("SELECT value FROM service_state WHERE key = ?", (key,)).fetchone()
    return r["value"] if r is not None else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute("INSERT INTO service_state (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def clear_all(conn: sqlite3.Connection) -> set[str]:
    """Delete every customer row (the test clips looped back to their start).
    Staff and roster stay. The crop files themselves are left on disk --
    the caller is returned the paths that were referenced (before the
    delete) so it can remove them; this function only touches rows."""
    paths = referenced_crops(conn)
    with conn:
        for table in ("barber_alert", "state_event", "customer_face", "customer_body", "customer",
                      "service_session", "handoff", "binding", "door_crossing", "chair_sitting", "customer_wait",
                      "visit_baseline", "resume_check", "haircut_photo", "pedicure_session"):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM visit")
    return paths


def resume_check_stats(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict]:
    """Per chair, the re-ID checks decided in [start, end): for the same customer (resumed) and for a new one, how
    many, and the score's mean / min / max; checks with no score (the time rule) counted apart. 2026-10-18 (user):
    to tune ce_resume_sim_low / _high per chair."""
    rows = conn.execute(
        "SELECT chair_id, resumed, COUNT(sim) AS n, AVG(sim) AS mean, MIN(sim) AS lo, MAX(sim) AS hi,"
        " SUM(sim IS NULL) AS no_score FROM resume_check WHERE decided_at >= ? AND decided_at < ?"
        " GROUP BY chair_id, resumed", (iso(start), iso(end))).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        c = out.setdefault(r["chair_id"], {"chair_id": r["chair_id"], "same": None, "new": None})
        c["same" if r["resumed"] else "new"] = {
            "n": r["n"], "mean": None if r["mean"] is None else round(r["mean"], 3),
            "min": r["lo"], "max": r["hi"], "no_score": r["no_score"]}
    return [out[k] for k in sorted(out)]


def visit_baseline(conn: sqlite3.Connection, day: str) -> dict | None:
    """The shop day's start-of-day baseline (people already inside), or None."""
    r = conn.execute("SELECT * FROM visit_baseline WHERE day = ?", (day,)).fetchone()
    return dict(r) if r is not None else None


def set_visit_baseline(conn: sqlite3.Connection, day: str, taken_at: datetime, haircuts: int, waiting: int,
                       washing: int) -> None:
    """Store the day's baseline; the first call wins (a second the same day is ignored)."""
    with conn:
        conn.execute("INSERT OR IGNORE INTO visit_baseline (day, taken_at, count, haircuts, waiting, washing)"
                     " VALUES (?, ?, ?, ?, ?, ?)",
                     (day, iso(taken_at), haircuts + waiting + washing, haircuts, waiting, washing))


def baselines_in(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Baselines taken in [start, end) (counted into Visits)."""
    return conn.execute("SELECT * FROM visit_baseline WHERE taken_at >= ? AND taken_at < ?",
                        (iso(start), iso(end))).fetchall()


def max_ticket(conn: sqlite3.Connection, start: datetime, end: datetime) -> int:
    """The day's highest ticket, haircut visits and customer waits alike (one daily counter)."""
    r = conn.execute(
        "SELECT MAX(n) FROM (SELECT MAX(ticket) AS n FROM visit WHERE entered_at >= ? AND entered_at < ?"
        " UNION ALL SELECT MAX(ticket) FROM customer_wait WHERE started_at >= ? AND started_at < ?)",
        (iso(start), iso(end), iso(start), iso(end))).fetchone()
    return r[0] or 0


def visit_events(conn: sqlite3.Connection, visit_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT from_state, to_state, area, camera, at FROM state_event WHERE visit_id = ? ORDER BY id",
        (visit_id,)).fetchall()]


def list_handoffs(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM handoff WHERE at >= ? AND at < ? ORDER BY id", (iso(start), iso(end))).fetchall()]


def list_bindings(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM binding WHERE at >= ? AND at < ? ORDER BY id", (iso(start), iso(end))).fetchall()]


# ── staff ───────────────────────────────────────────────────────────────────

def create_staff(conn: sqlite3.Connection, name: str, now: datetime | None = None) -> dict:
    with conn:
        row = conn.execute("SELECT n FROM id_sequence WHERE key = 'staff'").fetchone()
        n = (row["n"] if row else 0) + 1
        conn.execute("INSERT INTO id_sequence (key, n) VALUES ('staff', ?) "
                     "ON CONFLICT(key) DO UPDATE SET n = excluded.n", (n,))
        conn.execute("INSERT INTO staff (id, name, active, created_at) VALUES (?, ?, 1, ?)",
                     (f"staff-{n}", name, iso(now or datetime.now())))
    return get_staff(conn, f"staff-{n}")


def get_staff(conn: sqlite3.Connection, staff_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM staff WHERE id = ?", (staff_id,)).fetchone()
    return dict(r, active=bool(r["active"])) if r else None


def list_staff(conn: sqlite3.Connection, active_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM staff" + (" WHERE active = 1" if active_only else "") + " ORDER BY name"
    return [dict(r, active=bool(r["active"])) for r in conn.execute(sql).fetchall()]


def update_staff(conn: sqlite3.Connection, staff_id: str, **fields) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("name", "active")}
    if allowed:
        sets = ", ".join(f"{k} = ?" for k in allowed)
        values = [int(v) if k == "active" else v for k, v in allowed.items()]
        with conn:
            conn.execute(f"UPDATE staff SET {sets} WHERE id = ?", (*values, staff_id))
    return get_staff(conn, staff_id)


def delete_staff(conn: sqlite3.Connection, staff_id: str) -> bool:
    """History keeps the plain staff id on past haircuts."""
    with conn:
        return conn.execute("DELETE FROM staff WHERE id = ?", (staff_id,)).rowcount > 0


# ── roster ──────────────────────────────────────────────────────────────────

def assign_chair(conn: sqlite3.Connection, chair_id: str, staff_id: str | None, at: datetime) -> None:
    """From `at`, `staff_id` works `chair_id` (None = nobody). A barber works
    one chair at a time: their open assignment elsewhere ends too."""
    with conn:
        conn.execute("UPDATE roster SET valid_to = ? WHERE chair_id = ? AND valid_to IS NULL", (iso(at), chair_id))
        if staff_id is not None:
            conn.execute("UPDATE roster SET valid_to = ? WHERE staff_id = ? AND valid_to IS NULL",
                         (iso(at), staff_id))
            conn.execute("INSERT INTO roster (chair_id, staff_id, valid_from) VALUES (?, ?, ?)",
                         (chair_id, staff_id, iso(at)))


def roster_at(conn: sqlite3.Connection, chair_id: str, at: datetime) -> str | None:
    r = conn.execute(
        "SELECT staff_id FROM roster WHERE chair_id = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
        " ORDER BY valid_from DESC LIMIT 1", (chair_id, iso(at), iso(at))).fetchone()
    return r["staff_id"] if r else None


def current_roster(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["chair_id"]: r["staff_id"] for r in conn.execute(
        "SELECT chair_id, staff_id FROM roster WHERE valid_to IS NULL ORDER BY chair_id").fetchall()}


def roster_detail(conn: sqlite3.Connection) -> dict[str, dict]:
    return {r["chair_id"]: {"staff_id": r["staff_id"], "since": r["valid_from"]} for r in conn.execute(
        "SELECT chair_id, staff_id, valid_from FROM roster WHERE valid_to IS NULL ORDER BY chair_id").fetchall()}


def confirm_roster(conn: sqlite3.Connection, day: date, at: datetime) -> None:
    with conn:
        conn.execute("INSERT OR IGNORE INTO roster_confirmation (day, confirmed_at) VALUES (?, ?)",
                     (day.isoformat(), iso(at)))


def roster_confirmation(conn: sqlite3.Connection, day: date) -> dict:
    r = conn.execute("SELECT confirmed_at FROM roster_confirmation WHERE day = ?", (day.isoformat(),)).fetchone()
    return {"day": day.isoformat(), "confirmed": r is not None, "confirmed_at": r["confirmed_at"] if r else None}


# ── read side ───────────────────────────────────────────────────────────────
# Every visit query takes the same two optional filters -- the Insights page
# applies its Outcome and Barber dropdowns to everything on it:
#   outcome   -- the visit's outcome
#   barber_id -- the visit had a haircut credited to (or, while open, assigned to) this barber

def _visit_filters(outcome: str | None, barber_id: str | None, alias: str = "visit") -> tuple[str, list]:
    sql, args = "", []
    if outcome is not None:
        sql += f" AND {alias}.outcome = ?"
        args.append(outcome)
    if barber_id is not None:
        sql += (f" AND EXISTS (SELECT 1 FROM service_session s WHERE s.visit_id = {alias}.id"
                " AND s.barber_id = ?)")
        args.append(barber_id)
    return sql, args


def visits_entered(conn: sqlite3.Connection, start: datetime, end: datetime,
                   outcome: str | None = None, barber_id: str | None = None) -> list[sqlite3.Row]:
    """Customer visits (staff voids excluded) that entered in [start, end)."""
    extra, args = _visit_filters(outcome, barber_id)
    return conn.execute(
        "SELECT * FROM visit WHERE entered_at >= ? AND entered_at < ? AND COALESCE(outcome, '') != ?" + extra,
        (iso(start), iso(end), VOID_STAFF, *args),
    ).fetchall()


def waits_ended(conn: sqlite3.Connection, start: datetime, end: datetime,
                outcome: str | None = None, barber_id: str | None = None) -> list[sqlite3.Row]:
    """Visits with a measured wait that ENDED in [start, end) -- at the chair,
    or at the exit for a walk-out."""
    extra, args = _visit_filters(outcome, barber_id)
    return conn.execute(
        """SELECT * FROM visit WHERE wait_s IS NOT NULL AND COALESCE(outcome, '') != ?
           AND COALESCE(service_started_at, exited_at) >= ? AND COALESCE(service_started_at, exited_at) < ?"""
        + extra,
        (VOID_STAFF, iso(start), iso(end), *args),
    ).fetchall()


def waits_finished(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Chair-engine customer waits (customer_wait) that ended in [start, end)."""
    return conn.execute("SELECT * FROM customer_wait WHERE wait_s IS NOT NULL AND ended_at >= ? AND ended_at < ?",
                        (iso(start), iso(end))).fetchall()


def waits_started(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Chair-engine customer waits that started in [start, end), open ones included (door-free visits)."""
    return conn.execute("SELECT * FROM customer_wait WHERE started_at >= ? AND started_at < ?",
                        (iso(start), iso(end))).fetchall()


def excessive_wait_ids(conn: sqlite3.Connection, start: datetime, end: datetime) -> set[str]:
    """The customer waits with an excessive-wait alert opened in [start, end) -- once each: a wait lost and found
    closes and reopens its alert (2026-10-18: up to 8 for one wait)."""
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT a.visit_id FROM barber_alert a JOIN customer_wait w ON w.id = a.visit_id"
        " WHERE a.kind = 'excessive_wait' AND a.started_at >= ? AND a.started_at < ?", (iso(start), iso(end)))}


def voided_waits(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Visits voided as staff that had started waiting, voided in [start, end). 2026-09-28 (user):
    their wait (to the void) still counts in the average wait -- flagged to be fixed later (the
    39-min testbed spec, section 7)."""
    return conn.execute(
        "SELECT * FROM visit WHERE outcome = ? AND waiting_started_at IS NOT NULL AND wait_unseen = 0"
        " AND exited_at >= ? AND exited_at < ?",
        (VOID_STAFF, iso(start), iso(end)),
    ).fetchall()


def haircuts_ended(conn: sqlite3.Connection, start: datetime, end: datetime,
                   outcome: str | None = None, barber_id: str | None = None) -> list[sqlite3.Row]:
    """Completed (measured) haircuts that ended in [start, end)."""
    sql = "SELECT * FROM service_session WHERE service_s IS NOT NULL AND ended_at >= ? AND ended_at < ?"
    args: list = [iso(start), iso(end)]
    if barber_id is not None:
        sql += " AND barber_id = ?"
        args.append(barber_id)
    if outcome is not None:
        sql += " AND visit_id IN (SELECT id FROM visit WHERE outcome = ?)"
        args.append(outcome)
    return conn.execute(sql, args).fetchall()



def sessions_started(conn: sqlite3.Connection, start: datetime, end: datetime,
                     barber_id: str | None = None) -> list[sqlite3.Row]:
    """Confirmed haircut sessions that started in [start, end): running and interrupted ones too
    (chair engine: a session row is written at confirmation; dashboard spec 2026-10-05, served customers)."""
    sql = "SELECT * FROM service_session WHERE started_at >= ? AND started_at < ?"
    args: list = [iso(start), iso(end)]
    if barber_id is not None:
        sql += " AND barber_id = ?"
        args.append(barber_id)
    return conn.execute(sql, args).fetchall()


def pedicures_in(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Pedicure sessions that started in [start, end), oldest first (2026-11-10)."""
    return conn.execute("SELECT * FROM pedicure_session WHERE started_at >= ? AND started_at < ? ORDER BY started_at",
                        (iso(start), iso(end))).fetchall()


def any_visit(conn: sqlite3.Connection, start: datetime, end: datetime) -> bool:
    """Whether a visit entered in [start, end) (saved reports, 2026-10-25: a period without one gets no folder)."""
    return conn.execute("SELECT 1 FROM visit WHERE entered_at >= ? AND entered_at < ? LIMIT 1",
                        (iso(start), iso(end))).fetchone() is not None


def door_ins(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Non-staff `in` door crossings in [start, end) (the dashboard's Visits)."""
    return conn.execute("SELECT * FROM door_crossing WHERE direction = 'in' AND staff = 0 AND at >= ? AND at < ?",
                        (iso(start), iso(end))).fetchall()

def _listing_where(start: datetime, end: datetime, outcome: str | None, barber_id: str | None,
                   excessive_only: bool) -> tuple[str, list]:
    sql = " WHERE v.entered_at >= ? AND v.entered_at < ?"
    args: list = [iso(start), iso(end)]
    if outcome is None:
        sql += " AND COALESCE(v.outcome, '') != ?"
        args.append(VOID_STAFF)
    extra, extra_args = _visit_filters(outcome, barber_id, alias="v")
    sql += extra
    args += extra_args
    if excessive_only:
        sql += (" AND (v.excessive_wait = 1 OR EXISTS (SELECT 1 FROM service_session s"
                " WHERE s.visit_id = v.id AND s.excessive_service = 1))")
    return sql, args


def list_visits(conn: sqlite3.Connection, start: datetime, end: datetime, outcome: str | None = None,
                barber_id: str | None = None, excessive_only: bool = False, limit: int = 500,
                offset: int = 0) -> list[dict]:
    """Visits that entered in [start, end), newest first, each with its
    haircuts. Staff voids only when asked for by outcome."""
    where, args = _listing_where(start, end, outcome, barber_id, excessive_only)
    visits = [_visit_row(r) for r in conn.execute(
        "SELECT * FROM visit v" + where + " ORDER BY v.entered_at DESC LIMIT ? OFFSET ?",
        (*args, limit, offset)).fetchall()]
    by_id = {v["id"]: v for v in visits}
    if by_id:
        marks = ",".join("?" * len(by_id))
        for r in conn.execute(f"SELECT * FROM service_session WHERE visit_id IN ({marks}) ORDER BY started_at",
                              list(by_id)).fetchall():
            by_id[r["visit_id"]]["haircut_sessions"].append(_session_row(r))
    return visits


def count_visits(conn: sqlite3.Connection, start: datetime, end: datetime, outcome: str | None = None,
                 barber_id: str | None = None, excessive_only: bool = False) -> int:
    where, args = _listing_where(start, end, outcome, barber_id, excessive_only)
    return conn.execute("SELECT COUNT(*) FROM visit v" + where, args).fetchone()[0]


def list_alerts(conn: sqlite3.Connection, start: datetime | None = None, end: datetime | None = None,
                open_only: bool = False, limit: int = 100, visit_id: str | None = None) -> list[dict]:
    """Alerts, newest first, each with its haircut's chair and barber (NULL
    for a wait alert) so a list can say where it happened, and its visit's
    entered_at so a link can find the visit's shop day, and its ticket (the
    visit's, or the customer wait's for a wait alert)."""
    sql = ("SELECT a.*, s.chair_id AS chair_id, s.barber_id AS barber_id, v.entered_at AS visit_entered_at,"
           " COALESCE(v.ticket, w.ticket) AS ticket"
           " FROM barber_alert a LEFT JOIN service_session s ON s.id = a.session_id"
           " LEFT JOIN visit v ON v.id = a.visit_id LEFT JOIN customer_wait w ON w.id = a.visit_id WHERE 1 = 1")
    args: list = []
    if start is not None:
        sql += " AND a.started_at >= ?"
        args.append(iso(start))
    if end is not None:
        sql += " AND a.started_at < ?"
        args.append(iso(end))
    if open_only:
        sql += " AND a.open = 1"
    if visit_id is not None:
        sql += " AND a.visit_id = ?"
        args.append(visit_id)
    sql += " ORDER BY a.started_at DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(sql, args).fetchall():
        d = dict(r, open=bool(r["open"]), subject_box=json.loads(r["subject_box"]) if r["subject_box"] else None,
                 has_snapshot=r["snapshot_path"] is not None)
        d.pop("snapshot_path", None)
        out.append(d)
    return out


def set_alert_snapshot(conn: sqlite3.Connection, alert_id: str, rel_path: str) -> None:
    with conn:
        conn.execute("UPDATE barber_alert SET snapshot_path = ? WHERE id = ?", (rel_path, alert_id))


def alert_snapshot(conn: sqlite3.Connection, alert_id: str) -> str | None:
    r = conn.execute("SELECT snapshot_path FROM barber_alert WHERE id = ?", (alert_id,)).fetchone()
    return r["snapshot_path"] if r else None


def alert_subject(conn: sqlite3.Connection, alert_id: str) -> dict | None:
    """What an alert's picture shows (its mark in the reports): kind, subject_cam, subject_box (normalised x, y, w, h
    or None), the haircut's chair, the wait's or visit's ticket."""
    r = conn.execute("SELECT a.kind, a.subject_cam, a.subject_box, s.chair_id, COALESCE(v.ticket, w.ticket) AS ticket"
                     " FROM barber_alert a LEFT JOIN service_session s ON s.id = a.session_id"
                     " LEFT JOIN visit v ON v.id = a.visit_id LEFT JOIN customer_wait w ON w.id = a.visit_id"
                     " WHERE a.id = ?", (alert_id,)).fetchone()
    if r is None:
        return None
    return {**dict(r), "subject_box": json.loads(r["subject_box"]) if r["subject_box"] else None}


def purge_snapshots(conn: sqlite3.Connection, now: datetime, days: float) -> list[str]:
    """Alert snapshots older than `days` (photo retention): paths cleared on the rows, files returned."""
    cutoff = iso(now - timedelta(days=days))
    with conn:
        paths = [r["snapshot_path"] for r in conn.execute(
            "SELECT snapshot_path FROM barber_alert WHERE snapshot_path IS NOT NULL AND started_at < ?", (cutoff,))]
        conn.execute("UPDATE barber_alert SET snapshot_path = NULL WHERE snapshot_path IS NOT NULL AND started_at < ?",
                     (cutoff,))
    return paths


# ── haircut photos (2026-10-22, spec 2026-10-22-barber-haircut-photos-design.md) ──────────────────────────────
def set_haircut_photo(conn: sqlite3.Connection, session_id: str, kind: str, rel_path: str, taken_at: datetime) -> None:
    with conn:
        conn.execute("INSERT INTO haircut_photo (session_id, kind, path, taken_at) VALUES (?, ?, ?, ?)"
                     " ON CONFLICT(session_id, kind) DO UPDATE SET path = excluded.path, taken_at = excluded.taken_at",
                     (session_id, kind, rel_path, iso(taken_at)))


def haircut_photos(conn: sqlite3.Connection, session_ids) -> dict[str, dict[str, dict]]:
    """session id -> kind -> {"path", "taken_at"}, for those haircuts that have any."""
    ids, out = list(session_ids), {}
    for i in range(0, len(ids), 500):                  # SQLite's bound-parameter limit
        part = ids[i:i + 500]
        for r in conn.execute(f"SELECT * FROM haircut_photo WHERE session_id IN ({','.join('?' * len(part))})", part):
            out.setdefault(r["session_id"], {})[r["kind"]] = {"path": r["path"], "taken_at": r["taken_at"]}
    return out


def drop_haircut_photos(conn: sqlite3.Connection, session_id: str) -> list[str]:
    """A sitting that never became a haircut: its photo rows deleted, their files returned."""
    with conn:
        paths = [r["path"] for r in conn.execute("SELECT path FROM haircut_photo WHERE session_id = ?", (session_id,))]
        conn.execute("DELETE FROM haircut_photo WHERE session_id = ?", (session_id,))
    return paths


def purge_haircut_photos(conn: sqlite3.Connection, now: datetime, days: float, orphan_s: float = 3600.0) -> list[str]:
    """Photos older than `days` (photo retention), and rows of no haircut older than orphan_s (a restart mid-sitting,
    a request that landed after a lap's clear): rows deleted, files returned."""
    where = ("taken_at < ? OR (taken_at < ? AND session_id NOT IN (SELECT id FROM service_session)"
             " AND session_id NOT IN (SELECT id FROM pedicure_session))")
    args = (iso(now - timedelta(days=days)), iso(now - timedelta(seconds=orphan_s)))
    with conn:
        paths = [r["path"] for r in conn.execute(f"SELECT path FROM haircut_photo WHERE {where}", args)]
        conn.execute(f"DELETE FROM haircut_photo WHERE {where}", args)
    return paths


def _visit_row(r: sqlite3.Row) -> dict:
    return dict(r, entry_observed=bool(r["entry_observed"]), excessive_wait=bool(r["excessive_wait"]),
                open=bool(r["open"]), haircut_sessions=[])


def _session_row(r: sqlite3.Row) -> dict:
    return dict(r, excessive_service=bool(r["excessive_service"]), open=bool(r["open"]),
                low_occupancy=bool(r["low_occupancy"]))


# ── customer register (project C) ───────────────────────────────────────────

def _blob(v) -> bytes | None:
    return None if v is None else np.asarray(v, "<f4").tobytes()


def load_face_gallery(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for r in conn.execute("SELECT customer_id, embedding FROM customer_face"):
        out.setdefault(r["customer_id"], []).append(np.frombuffer(r["embedding"], "<f4"))
    return {cid: np.stack(v) for cid, v in out.items()}


def load_body_gallery(conn: sqlite3.Connection, cid: str) -> dict[str, np.ndarray]:
    """A customer's stored body re-ID embeddings by camera (crop_check)."""
    out: dict[str, list[np.ndarray]] = {}
    for r in conn.execute("SELECT camera, embedding FROM customer_body WHERE customer_id = ? AND embedding IS NOT NULL",
                          (cid,)):
        out.setdefault(r["camera"], []).append(np.frombuffer(r["embedding"], "<f4"))
    return {cam: np.stack(v) for cam, v in out.items()}


def visit_customer(conn: sqlite3.Connection, visit_id: str) -> str | None:
    r = conn.execute("SELECT customer_id FROM visit WHERE id = ?", (visit_id,)).fetchone()
    return r["customer_id"] if r else None


def _merge_best(conn, table: str, cid: str, new: list, cap: int, visit_id: str) -> list[str]:
    """Keep the `cap` highest-quality crops among the stored ones and `new`;
    return the paths of everything not kept."""
    rows = [(r["quality"], r["id"], r["crop_path"], r["taken_at"]) for r in
            conn.execute(f"SELECT id, quality, crop_path, taken_at FROM {table} WHERE customer_id = ?", (cid,))]
    pool = [(q, "old", rid, path, None, taken) for q, rid, path, taken in rows] + \
           [(c.quality, "new", None, c.crop_path, c, iso(datetime.fromtimestamp(c.ts))) for c in new]
    pool.sort(key=lambda x: x[0], reverse=True)
    if table == "customer_body" and pool and cap > 0:
        # 2026-09-29 (user): the customer card shows his earliest body crop -- it's always kept
        first = min(pool, key=lambda x: x[5])
        rest = [x for x in pool if x is not first]
        keep, drop = [first] + rest[:cap - 1], rest[cap - 1:]
    else:
        keep, drop = pool[:cap], pool[cap:]
    for q, kind, rid, path, c, _taken in keep:
        if kind == "new":
            conn.execute(f"INSERT INTO {table} (customer_id, crop_path, embedding, quality, taken_at, visit_id)"
                         " VALUES (?, ?, ?, ?, ?, ?)",
                         (cid, path, _blob(c.embedding), q, iso(datetime.fromtimestamp(c.ts)), visit_id))
            if table == "customer_body":
                conn.execute("UPDATE customer_body SET camera = ? WHERE id = last_insert_rowid()", (c.camera,))
    for q, kind, rid, path, c, _taken in drop:
        if kind == "old":
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (rid,))
    return [path for *_x, path, _c, _taken in drop]


def _prune_faces(conn, cid: str, min_sim: float) -> list[str]:
    """2026-09-29 (user): a customer's stored faces must be one person -- drop those less than min_sim
    like his medoid face (needs 3+: with two that disagree there's no telling which one is him)."""
    rows = conn.execute("SELECT id, crop_path, embedding FROM customer_face WHERE customer_id = ?", (cid,)).fetchall()
    if len(rows) < 3:
        return []
    embs = np.stack([np.frombuffer(r["embedding"], "<f4") for r in rows])
    sims = embs @ embs.T
    medoid = int(np.argmax(sims.sum(axis=1)))
    gone = [r for r, s in zip(rows, sims[medoid]) if s < min_sim]
    for r in gone:
        conn.execute("DELETE FROM customer_face WHERE id = ?", (r["id"],))
    return [r["crop_path"] for r in gone]


def register_visit(conn: sqlite3.Connection, reg: Registration, decision, policy: dict,
                   new_id, bodies: list | None = None) -> tuple[str | None, list[str]]:
    """bodies: the body crops to keep (crop_check); None = all. A visit already linked to a customer
    (register_at 1 registered it when its haircut ended) is topped up: same customer, same visit count."""
    faces = [reg.faces[i] for i in decision.clean] if decision.usable else []
    stray = [f.crop_path for i, f in enumerate(reg.faces) if not decision.usable or i not in decision.clean]
    bodies = list(reg.bodies) if bodies is None else bodies
    stray += [b.crop_path for b in reg.bodies if b not in bodies]
    row = conn.execute("SELECT entered_at, customer_id FROM visit WHERE id = ?", (reg.visit_id,)).fetchone()
    if row is None:
        d = conn.execute("SELECT at FROM door_crossing WHERE entry_id = ?", (reg.visit_id,)).fetchone()
        entered = d["at"] if d else iso(reg.ended_at)
    else:
        entered = row["entered_at"]
    known = row["customer_id"] if row else None
    if known is None and decision.customer_id is None and not faces and not bodies:
        return None, stray              # 2026-09-29 (user): no picture at all -- not a customer
    with conn:
        if known is not None:
            cid = known
            n = conn.execute("UPDATE customer SET last_seen = MAX(last_seen, ?) WHERE id = ?",
                             (iso(reg.ended_at), cid)).rowcount
            if n != 1:
                raise LookupError(f"customer {cid} no longer exists")
        elif decision.customer_id is None:
            cid = new_id()
            conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, last_visit_id, face_count)"
                         " VALUES (?, ?, ?, 1, ?, 0)", (cid, entered, iso(reg.ended_at), reg.visit_id))
        else:
            cid = decision.customer_id
            n = conn.execute("UPDATE customer SET last_seen = MAX(last_seen, ?), visit_count = visit_count + 1,"
                             " last_visit_id = ? WHERE id = ?", (iso(reg.ended_at), reg.visit_id, cid)).rowcount
            if n != 1:      # erased since decide(): never link the visit to him (the with-block rolls back)
                raise LookupError(f"customer {cid} no longer exists")
        dropped = _merge_best(conn, "customer_face", cid, faces, int(policy["customer_max_faces"]), reg.visit_id)
        dropped += _merge_best(conn, "customer_body", cid, bodies, int(policy["customer_max_bodies"]),
                               reg.visit_id)
        if int(policy.get("crop_check", 0)) >= 1:
            dropped += _prune_faces(conn, cid, policy["customer_face_min_sim"])
        conn.execute("UPDATE customer SET face_count = (SELECT COUNT(*) FROM customer_face WHERE customer_id = ?)"
                     " WHERE id = ?", (cid, cid))
        conn.execute("UPDATE visit SET customer_id = ? WHERE id = ?", (cid, reg.visit_id))
        conn.execute("UPDATE door_crossing SET customer_id = ? WHERE entry_id = ?", (cid, reg.visit_id))
    return cid, stray + dropped


def _customer_paths(conn, cid: str) -> list[str]:
    return [r["crop_path"] for t in ("customer_face", "customer_body")
            for r in conn.execute(f"SELECT crop_path FROM {t} WHERE customer_id = ?", (cid,))]


def delete_customer(conn: sqlite3.Connection, customer_id: str) -> list[str] | None:
    """Delete him (his crop rows cascade); return his crop paths, read in the
    same write transaction, so a registration committing in between can't
    leave a crop file nobody deletes. None = no such customer."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM customer WHERE id = ?", (customer_id,)).fetchone() is None:
            return None
        paths = _customer_paths(conn, customer_id)
        conn.execute("UPDATE visit SET customer_id = NULL WHERE customer_id = ?", (customer_id,))
        conn.execute("UPDATE door_crossing SET customer_id = NULL WHERE customer_id = ?", (customer_id,))
        conn.execute("DELETE FROM customer WHERE id = ?", (customer_id,))
    return paths


def purge_inactive(conn: sqlite3.Connection, now: datetime, days: float) -> list[str]:
    cutoff = iso(now - timedelta(days=days))
    paths: list[str] = []
    for r in conn.execute("SELECT id FROM customer WHERE last_seen < ?", (cutoff,)).fetchall():
        paths += delete_customer(conn, r["id"]) or []
    return paths


def referenced_crops(conn: sqlite3.Connection) -> set[str]:
    paths = {r["crop_path"] for t in ("customer_face", "customer_body")
             for r in conn.execute(f"SELECT crop_path FROM {t}")}
    paths |= {r["snapshot_path"] for r in conn.execute(
        "SELECT snapshot_path FROM barber_alert WHERE snapshot_path IS NOT NULL")}
    paths |= {r["path"] for r in conn.execute("SELECT path FROM haircut_photo")}
    return paths


def list_customers(conn: sqlite3.Connection, limit: int, offset: int, sort: str = "last_seen") -> list[dict]:
    order = {"last_seen": "last_seen DESC", "visit_count": "visit_count DESC, last_seen DESC"}.get(sort, "last_seen DESC")
    rows = conn.execute(f"SELECT * FROM customer ORDER BY {order} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    out = []
    for r in rows:
        # 2026-09-29 (user): the card shows his earliest body crop; a face only when he has none
        thumb, kind = conn.execute("SELECT id FROM customer_body WHERE customer_id = ? ORDER BY taken_at, id LIMIT 1",
                                   (r["id"],)).fetchone(), "bodies"
        if thumb is None:
            thumb, kind = conn.execute("SELECT id FROM customer_face WHERE customer_id = ? ORDER BY quality DESC LIMIT 1",
                                       (r["id"],)).fetchone(), "faces"
        # 2026-10-18 (user): the card says at which chair he was served in every visit, newest first (a resumed
        # haircut can move to a second chair; a visit with none was not served)
        served = {v["id"]: {"at": v["entered_at"], "chairs": []} for v in conn.execute(
            "SELECT id, entered_at FROM visit WHERE customer_id = ? ORDER BY entered_at DESC", (r["id"],))}
        for s in conn.execute("SELECT s.visit_id, s.chair_id FROM service_session s JOIN visit v ON v.id = s.visit_id"
                              " WHERE v.customer_id = ? ORDER BY s.started_at", (r["id"],)):
            if s["chair_id"] not in served[s["visit_id"]]["chairs"]:
                served[s["visit_id"]]["chairs"].append(s["chair_id"])
        out.append(dict(r, thumb=f"/api/barber/crops/{kind}/{thumb['id']}" if thumb else None,
                        thumb_kind=None if thumb is None else kind[:-1] if kind == "faces" else "body",
                        served=list(served.values())))
    return out


def count_customers(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM customer").fetchone()[0]


def get_customer(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM customer WHERE id = ?", (customer_id,)).fetchone()
    if r is None:
        return None
    crops = lambda t, kind: [dict(id=x["id"], quality=x["quality"], taken_at=x["taken_at"], visit_id=x["visit_id"],
                                  url=f"/api/barber/crops/{kind}/{x['id']}")
                             for x in conn.execute(f"SELECT id, quality, taken_at, visit_id FROM {t}"
                                                   " WHERE customer_id = ? ORDER BY quality DESC", (customer_id,))]
    visits = [_visit_row(v) for v in conn.execute("SELECT * FROM visit WHERE customer_id = ? ORDER BY entered_at",
                                                  (customer_id,)).fetchall()]
    by_id = {v["id"]: v for v in visits}
    if by_id:
        marks = ",".join("?" * len(by_id))
        for s in conn.execute(f"SELECT * FROM service_session WHERE visit_id IN ({marks}) ORDER BY started_at",
                              list(by_id)).fetchall():
            by_id[s["visit_id"]]["haircut_sessions"].append(_session_row(s))
    return dict(r, faces=crops("customer_face", "faces"), bodies=crops("customer_body", "bodies"), visits=visits,
                entries=[{"id": e["entry_id"], "at": e["at"]} for e in conn.execute(
                    "SELECT entry_id, at FROM door_crossing WHERE customer_id = ? ORDER BY at DESC", (customer_id,))])


def crop_row(conn: sqlite3.Connection, kind: str, row_id: int) -> dict | None:
    table = {"faces": "customer_face", "bodies": "customer_body"}.get(kind)
    if table is None:
        return None
    r = conn.execute(f"SELECT crop_path FROM {table} WHERE id = ?", (row_id,)).fetchone()
    return dict(r) if r else None
