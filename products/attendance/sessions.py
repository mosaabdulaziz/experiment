"""Session-window derivation (2026-09-02, /grilling + /mentor design) --
turns one employee's raw Sighting facts for one day into Sessions with
Breaks, and derives every attendance-v2 number/flag from them: total/net
hours, late, early leave, overtime, absent. See
~/.claude/mentor/decisions/attendance-system.md for the design trail and
the /mentor lesson on why this is a session-window algorithm.

Same "facts in, views derived on read" discipline as compute_daily_attendance
in storage.py -- nothing here is stored; call it whenever the answer is
needed, from whatever's in `sighting` right now.

Algorithm, one pass over an employee's day sorted by time, comparing each
consecutive pair of touches:

  gap <  break_min_seconds                              -> noise, ignored
  break_min_seconds <= gap < exit_gap_seconds            -> a Break
  gap >= exit_gap_seconds, either side on a break-room   -> still a Break
      camera (Q3, round 2: "no matter how long")            (exemption)
  gap >= exit_gap_seconds, neither side break-room       -> session closes
      (an exit; the next touch starts a new Session)

A camera absent from the policy's camera_types dict is office by default.
Q2b ("office wins" on a simultaneous doorway overlap): before the pass
above, any break-room touch that overlaps an office touch in time is
dropped -- the office touch already accounts for that moment.
"""
from __future__ import annotations

import calendar
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from domain import Break, DayAttendance, Session
from jsonstore import JsonFileStore
from storage import sightings_for_day

OFFICE, BREAK_ROOM = "office", "break_room"

# Attendance v2's own settings (2026-09-02) -- a SEPARATE store from
# productivity.py's policy.json. That file belongs to the desk/phone
# monitoring feature; this one belongs to attendance. Q5 (round 2, "one
# settings place") was a UI decision -- the Settings PAGE presents both
# together -- not a mandate to merge the two Python modules. Corrected after
# an initial pass wrongly folded these into productivity.DEFAULT_POLICY.
DEFAULT_ATTENDANCE_POLICY = {
    "work_start": "09:00",              # HH:MM, global schedule (Q2: no per-employee yet)
    "work_end": "17:00",
    # 2026-09-14: on TODAY's dashboard the departure column stays empty until
    # this time, then shows each person's last sighting ("آخر ظهور" shows it
    # live all day). A brief sighting mid-day no longer reads as a departure.
    "departure_reveal_time": "19:00",
    "late_grace_minutes": 15.0,        # late = first entry after work_start + this
    "early_leave_grace_minutes": 15.0,  # early leave = last exit before work_end - this
    "break_min_seconds": 300.0,         # gap >= this (but < exit_gap) = a break (Q3)
    "exit_gap_seconds": 1800.0,         # gap >= this on every camera = session closes (Q3)
    "violation_debounce_seconds": 3.0,  # restricted-zone dwell before it's a violation (Q4, round 2)
    # {cam_id: "office" | "break_room"} -- any camera absent from this dict is
    # office by default (Q2, round 2). A break-room sighting is exempt from
    # ever triggering the exit_gap branch of the session-window algorithm --
    # see build_sessions below.
    "camera_types": {},
    # {cam_id: {"allowed_roles": [...]}} -- who may be seen on a camera (2026-11-02). Listed cameras are checked
    # inside the restricted zones drawn on them, or on the whole stream when none are drawn (zones.py
    # effective_restricted_zones); an unlisted camera is only checked inside restricted zones, if it has any.
    "access_control": {},
}


def attendance_policy_store(data_dir: Path) -> JsonFileStore:
    return JsonFileStore(data_dir / "attendance_policy.json", lambda: dict(DEFAULT_ATTENDANCE_POLICY))


def camera_type(policy: dict, cam_id: str) -> str:
    return policy.get("camera_types", {}).get(cam_id, OFFICE)


@dataclass
class _Touch:
    start: datetime
    end: datetime
    type: str  # OFFICE | BREAK_ROOM


def _touches_for_day(conn: sqlite3.Connection, employee_id: str, day: date, policy: dict) -> list[_Touch]:
    raw = [_Touch(s.first_seen_at, s.last_seen_at, camera_type(policy, s.camera))
           for s in sightings_for_day(conn, employee_id, day)]
    raw.sort(key=lambda t: t.start)

    office = [t for t in raw if t.type == OFFICE]

    def overlaps_office(t: _Touch) -> bool:
        return any(t.start < o.end and o.start < t.end for o in office)

    # Q2b: a break-room touch simultaneous with an office touch is dropped --
    # the office touch already covers that moment, and its presence is what
    # should win the exemption/gap classification below.
    kept = [t for t in raw if t.type == OFFICE or not overlaps_office(t)]
    kept.sort(key=lambda t: t.start)
    return kept


def _effective_now(day: date, now: datetime) -> datetime:
    """`now`, clamped to that day's own midnight if the day is already over.

    Without this, a day whose last session never gets an explicit exit --
    always true once its last touch is on a break-room camera, exempt from
    the exit_gap check "no matter how long" -- keeps computing its duration
    against the REAL current instant. For a fixed simulated day (or any past
    day) that's years of wall-clock time once "now" has moved on, producing
    an absurd net-hours figure (found live 2026-09-03: 28380:38:23 -- exactly
    the gap between the spike's 2023-06-08 data and the real current date).
    The break-room exemption's "no matter how long" is about a LIVE day in
    progress ("don't cut off someone's lunch break"); once a day is over,
    nothing more is ever coming for it, so its own end is the right stand-in
    for "now" -- not whatever instant happens to call this function."""
    day_end = datetime.combine(day, time.max)
    return min(now, day_end)


def build_sessions(conn: sqlite3.Connection, employee_id: str, day: date,
                    policy: dict, now: datetime) -> list[Session]:
    now = _effective_now(day, now)
    touches = _touches_for_day(conn, employee_id, day, policy)
    if not touches:
        return []

    break_min = float(policy.get("break_min_seconds", 300.0))
    exit_gap = float(policy.get("exit_gap_seconds", 1800.0))

    sessions: list[Session] = []
    entry = touches[0].start
    breaks: list[Break] = []
    last = touches[0]

    for t in touches[1:]:
        gap = (t.start - last.end).total_seconds()
        if gap < break_min:
            pass  # noise -- same continuous stretch, nothing to record
        elif gap < exit_gap or last.type == BREAK_ROOM or t.type == BREAK_ROOM:
            breaks.append(Break(start=last.end, end=t.start))
        else:
            sessions.append(Session(entry=entry, exit=last.end, breaks=breaks))
            entry, breaks = t.start, []
        last = t

    # Trailing edge: is the day's last touch itself long enough silent since
    # (relative to `now`) to count as an exit? Same rule, `now` standing in
    # for "the next touch that hasn't happened (yet)".
    trailing_gap = (now - last.end).total_seconds()
    if trailing_gap >= exit_gap and last.type != BREAK_ROOM:
        sessions.append(Session(entry=entry, exit=last.end, breaks=breaks))
    else:
        sessions.append(Session(entry=entry, exit=None, breaks=breaks))

    return sessions


def _net_intervals(session: Session, now: datetime) -> list[tuple[datetime, datetime]]:
    """The session's span with its own breaks cut out -- the actual
    net-presence sub-intervals, used for both net-seconds and overtime."""
    end = session.exit or now
    cur = session.entry
    out = []
    for b in session.breaks:
        if b.start > cur:
            out.append((cur, b.start))
        cur = max(cur, b.end)
    if cur < end:
        out.append((cur, end))
    return out


def _overlap_seconds(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    start, end = max(a_start, b_start), min(a_end, b_end)
    return max(0.0, (end - start).total_seconds())


def derive_day(conn: sqlite3.Connection, employee_id: str, day: date,
                policy: dict, now: datetime) -> DayAttendance:
    now = _effective_now(day, now)  # see _effective_now's docstring
    sessions = build_sessions(conn, employee_id, day, policy, now)

    attendance_s = sum(s.span_seconds(now) for s in sessions)
    break_s = sum(s.break_seconds() for s in sessions)
    net_s = attendance_s - break_s

    work_start = datetime.combine(day, time.fromisoformat(policy.get("work_start", "09:00")))
    work_end = datetime.combine(day, time.fromisoformat(policy.get("work_end", "17:00")))
    late_grace = float(policy.get("late_grace_minutes", 15.0)) * 60.0
    early_grace = float(policy.get("early_leave_grace_minutes", 15.0)) * 60.0
    day_end = datetime.combine(day, time.max)

    overtime_s = 0.0
    for s in sessions:
        for start, end in _net_intervals(s, now):
            overtime_s += _overlap_seconds(start, end, work_end, day_end)

    late = bool(sessions) and sessions[0].entry > work_start + timedelta(seconds=late_grace)
    # 2026-09-13: first entry BEFORE work_start. Informational only -- time
    # before the start still doesn't count as overtime (see overtime_s above).
    early_arrival = bool(sessions) and sessions[0].entry < work_start
    last_exit = sessions[-1].exit if sessions else None
    early_leave = last_exit is not None and last_exit < work_end - timedelta(seconds=early_grace)

    # 2026-09-14: first arrival -> last departure (an open last session counts
    # to `now`, already clamped to the day's end above). Time away in between
    # is included -- this is the monthly sheet's "صافي الساعات".
    span_s = 0.0
    if sessions:
        last_end = sessions[-1].exit or now
        span_s = max(0.0, (last_end - sessions[0].entry).total_seconds())

    return DayAttendance(
        employee_id=employee_id, day=day, sessions=sessions,
        attendance_seconds=attendance_s, break_seconds=break_s, net_seconds=net_s,
        overtime_seconds=overtime_s, late=late, early_leave=early_leave,
        absent=not sessions, early_arrival=early_arrival, span_seconds=span_s,
    )


def derive_month(conn: sqlite3.Connection, employee_id: str, year: int, month: int,
                  policy: dict, now: datetime) -> list[DayAttendance]:
    """One DayAttendance per calendar day of the month, day 1 first --
    the per-employee history page (2026-09-02, /grilling round 2 Q1: its own
    page, not a slide-over). A day with no sightings yet (including any day
    still in the future relative to `now`) simply comes back absent=True --
    consistent with "derive on read, never store a verdict", nothing special
    to branch on for "hasn't happened yet"."""
    n_days = calendar.monthrange(year, month)[1]
    return [derive_day(conn, employee_id, date(year, month, d), policy, now)
            for d in range(1, n_days + 1)]
