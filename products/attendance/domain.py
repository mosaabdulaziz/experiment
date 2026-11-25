"""Attendance service domain model — provisional contract, agreed via
/mattpocock-skills:grilling on 2026-08-26. See
~/.claude/mentor/decisions/attendance-system.md for the full decision trail
behind every choice below.

This is a skeleton: types and signatures only, no logic. Nothing here should
be implemented until the CHIRLA spike (.scratch/attendance-system/issues/02-*)
validates the underlying sightings-log approach — this contract exists so the
spike has something concrete to write into, and is expected to be revised once
real data comes back.
"""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Employee:
    """An enrolled employee. Deliberately minimal — department/shift/role
    fields are still open (see supervisor-questions.md); don't add them
    speculatively."""

    id: str  # our own primary key
    name: str
    gallery_rec_id: str  # FK into the forked attendance gallery, set at enrollment
    active: bool = True


@dataclass(frozen=True)
class Sighting:
    """One coalesced observation: identity+camera+minute bucket, not a raw
    per-frame tick. No confidence field — the sidecar's stabilized output is
    trusted as-is, no second threshold applied here."""

    id: str
    employee_id: str  # FK -> Employee.id
    camera: str  # = the recognition stream's source_id
    first_seen_at: datetime  # wall-clock; see wall_clock_for_spike()
    last_seen_at: datetime


@dataclass(frozen=True)
class DailyAttendanceRecord:
    """Derived, never stored as its own mutable entity — always recomputed
    from Sighting rows. There is no PresenceSession: presence is a query
    over raw facts, not a live state machine."""

    employee_id: str
    day: date
    first_arrival: datetime | None  # None if absent that day
    last_departure: datetime | None
    present: bool


# ── Attendance v2 (2026-09-02, /grilling + /mentor design): sessions, breaks,
# net hours — see ~/.claude/mentor/decisions/attendance-system.md. Derived
# from Sighting rows via a session-window algorithm (sessions.py); never
# stored, same "facts in, views derived on read" discipline as
# DailyAttendanceRecord above, just a richer view.

@dataclass(frozen=True)
class Break:
    """One gap inside a session long enough to count as a break (>= the
    break_min_seconds policy knob) but not long enough to end the session
    (< exit_gap_seconds). A break-room camera sighting can never itself
    trigger a break's *closing* edge into an exit -- see sessions.py."""

    start: datetime
    end: datetime

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True)
class Session:
    """One entry/exit pair -- a session window (see sessions.py) over one
    employee's sightings. `exit` is None while the session is still open
    (no sighting yet crossed exit_gap_seconds since the last one)."""

    entry: datetime
    exit: datetime | None
    breaks: list[Break]

    def span_seconds(self, now: datetime) -> float:
        """Session duration; `now` only matters while still open."""
        return ((self.exit or now) - self.entry).total_seconds()

    def break_seconds(self) -> float:
        return sum(b.seconds for b in self.breaks)


@dataclass(frozen=True)
class DayAttendance:
    """One employee, one calendar day: the sessions plus every derived
    number and flag the schedule/gap policy knobs produce from them. The
    single place all of "total hours", "late", "early leave", "overtime",
    "absent" get their answer -- see sessions.derive_day."""

    employee_id: str
    day: date
    sessions: list[Session]
    attendance_seconds: float   # sum of session spans (open session counted to `now`)
    break_seconds: float        # sum of every break in every session
    net_seconds: float          # attendance_seconds - break_seconds (the headline number)
    overtime_seconds: float     # net time attributable to presence after work_end
    late: bool                  # first entry after work_start + late_grace_minutes
    early_leave: bool           # last exit before work_end - early_leave_grace_minutes (False if still open)
    absent: bool                # no sessions at all that day
    early_arrival: bool = False  # first entry before work_start (2026-09-13; informational, not overtime)
    span_seconds: float = 0.0    # first arrival -> last departure, gaps included (2026-09-14)
