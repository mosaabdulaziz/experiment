"""Restricted-zone violations (2026-09-02, /grilling round-2 Q4 + /mentor
design): same debounced open-interval shape as productivity.DeskStateEngine
(see that class's docstring), applied to a different question -- not "are
they at a desk" but "are they somewhere their role doesn't allow, or not
identified at all." See zones.py for the zone-type/role-check primitives
this reads, and schema.sql for the `violation` table.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from storage import (clear_all_violations, close_stale_violations, close_violation, heartbeat_violation,
                     open_violation)

logger = logging.getLogger("attendance.violations")

ROLE_NOT_ALLOWED, UNIDENTIFIED = "role_not_allowed", "unidentified"


@dataclass(frozen=True)
class ZoneEvidence:
    """One cycle's evidence that `key` is currently violating a restricted
    zone. `key` is the debounce identity -- "role:<employee_id>" for a
    recognized-but-unauthorized employee, "track:<camera>:<track_id>" for an
    unidentified body. There's no stable identity for the unidentified case
    by definition, so the debounce clock is per-track: an accepted
    approximation, same spirit as PersonPresenceTracker's own track-based
    fallbacks elsewhere in this service."""

    key: str
    camera: str
    zone_id: str
    employee_id: str | None
    violation_type: str


@dataclass
class _Pending:
    observed: bool
    evidence: ZoneEvidence | None
    since: datetime


class ZoneViolationEngine:
    """Debounced violation open-intervals, persisted as `violation` rows.
    observe() is called once per consumer cycle with this cycle's violating
    keys, unioned across every camera -- exactly DeskStateEngine's shape
    (productivity.py), right down to committing at the evidence's own start
    time rather than when the debounce elapses."""

    def __init__(self, conn: sqlite3.Connection, debounce_seconds: Callable[[], float]):
        self._conn = conn
        self._debounce = debounce_seconds  # () -> float, hot-reloaded from policy
        self._committed: dict[str, bool] = {}
        self._pending: dict[str, _Pending] = {}
        self._open: dict[str, str] = {}
        # Set from the web thread when today's rows are cleared; applied by
        # observe() on the consumer thread, so the dicts above are only ever
        # touched from one thread.
        self._forget = threading.Event()
        # A previous run may have stopped with violations open; this process cannot continue them (their state lived in
        # its memory), so close them now or they stay "ongoing" for ever. A person still there opens a fresh row.
        stale = close_stale_violations(conn)
        if stale:
            logger.info("closed %d violation(s) left open by the previous run", stale)

    def request_forget(self) -> None:
        """Today's violation rows were deleted (2026-09-13, dashboard "clear
        log"): drop the in-memory state on the next cycle, WITHOUT touching
        the table, so an intrusion still in progress is logged as a new row
        instead of heartbeating a row that no longer exists."""
        self._forget.set()

    def observe(self, evidence: dict[str, ZoneEvidence], now: datetime) -> bool:
        """evidence: {key: ZoneEvidence} for every key currently violating
        this cycle. A key absent from the dict (track left, or an employee's
        role became allowed) is "not violating" -- same absence-means-off
        convention as DeskStateEngine.observe's at_desk dict."""
        if self._forget.is_set():
            self._forget.clear()
            self._committed.clear()
            self._pending.clear()
            self._open.clear()
        debounce = self._debounce()
        changed = False
        for key in set(evidence) | set(self._committed) | set(self._pending):
            ev = evidence.get(key)
            observed = ev is not None
            committed = self._committed.get(key, False)
            if observed == committed:
                self._pending.pop(key, None)
            else:
                pend = self._pending.get(key)
                if pend is None or pend.observed != observed:
                    self._pending[key] = _Pending(observed, ev, now)
                elif (now - pend.since).total_seconds() >= debounce:
                    self._commit(key, pend.observed, pend.evidence, pend.since)
                    changed = True
            if self._committed.get(key) and key in self._open:
                heartbeat_violation(self._conn, self._open[key], now)
                changed = True
        return changed

    def _commit(self, key: str, observed: bool, evidence: ZoneEvidence | None, at: datetime) -> None:
        self._pending.pop(key, None)
        if observed and evidence is not None:
            vid = str(uuid.uuid4())
            open_violation(self._conn, vid, evidence.employee_id, evidence.camera,
                           evidence.zone_id, evidence.violation_type, at)
            self._open[key] = vid
        else:
            vid = self._open.pop(key, None)
            if vid:
                close_violation(self._conn, vid, at)
        self._committed[key] = observed

    def clear(self) -> None:
        """Loop-wrap reset: forget everything and wipe the table -- same
        convention as DeskStateEngine.clear() and SeatOccupancy.clear()."""
        self._committed.clear()
        self._pending.clear()
        self._open.clear()
        clear_all_violations(self._conn)
