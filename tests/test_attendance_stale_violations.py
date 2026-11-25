"""A violation left open by a stopped server must not stay "ongoing" for ever (products/attendance/violations.py).

The engine tracks its open rows only in memory, so a restart orphaned them: the dashboard kept showing an alert for a
person who was long gone, and the alert popup fired again on every page load.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "products" / "attendance"))

from storage import connect, create_employee, recent_violations  # noqa: E402
from violations import ROLE_NOT_ALLOWED, ZoneEvidence, ZoneViolationEngine  # noqa: E402

T0 = datetime(2026, 11, 2, 19, 59, 0)


def _ev(emp="emp-1"):
    return {f"role:{emp}": ZoneEvidence(key=f"role:{emp}", camera="cam-22", zone_id="whole-stream",
                                        employee_id=emp, violation_type=ROLE_NOT_ALLOWED)}


def _db(tmp_path):
    conn = connect(str(tmp_path / "a.db"))
    create_employee(conn, "emp-1", "Person 1", "person-1")
    return conn


def _alert_that_the_first_run_left_open(conn):
    first = ZoneViolationEngine(conn, lambda: 3.0)
    first.observe(_ev(), T0)
    first.observe(_ev(), T0 + timedelta(seconds=4))        # debounce passed: the row opens
    first.observe(_ev(), T0 + timedelta(seconds=10))       # heartbeat: ended_at = last time seen
    assert [v["open"] for v in recent_violations(conn)] == [True]


def test_a_new_process_closes_what_the_previous_one_left_open(tmp_path):
    conn = _db(tmp_path)
    _alert_that_the_first_run_left_open(conn)
    ZoneViolationEngine(conn, lambda: 3.0)                  # the restarted server
    (row,) = recent_violations(conn)
    assert row["open"] is False
    assert row["ended_at"] == (T0 + timedelta(seconds=10)).isoformat()       # the true end of the stay is kept


def test_a_person_still_there_after_the_restart_gets_a_fresh_row(tmp_path):
    conn = _db(tmp_path)
    _alert_that_the_first_run_left_open(conn)
    second = ZoneViolationEngine(conn, lambda: 3.0)
    later = T0 + timedelta(minutes=5)
    second.observe(_ev(), later)
    second.observe(_ev(), later + timedelta(seconds=4))
    rows = recent_violations(conn)
    assert [r["open"] for r in rows] == [True, False]       # newest first: the new one open, the old one closed
    assert rows[0]["started_at"] != rows[1]["started_at"]


def test_nothing_is_closed_when_nothing_was_left_open(tmp_path):
    conn = _db(tmp_path)
    ZoneViolationEngine(conn, lambda: 3.0)
    assert recent_violations(conn) == []
