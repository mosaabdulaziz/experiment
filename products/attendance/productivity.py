"""Productivity monitoring v1 (2026-09-01, designed via /grilling):
observable desk states derived from the same signals attendance already
trusts -- the track->employee binding (with its collision guard) plus the
tracked person boxes in the meta stream -- tested against seat zones.

Decisions this module encodes (see the grilling session in prompt.md):
  Q1b  states: AT_DESK, PHONE (model+UI ready, no signal yet), AWAY
  Q2   any seat zone counts (no per-employee desk assignment in v1)
  Q4   absence (outside the day's presence span) is in neither bucket
  Q7c  working = at-desk minus SUSTAINED phone use (policy knob)
  Q9   3s debounce, no away grace (both policy knobs)
  Q13a away = presence span minus at-desk time, visible or not
Only positive states are stored (desk_interval rows); away is derived.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from jsonstore import JsonFileStore
from storage import (clear_all_desk_intervals, close_desk_interval, compute_daily_attendance,
                     desk_seconds_by_state, heartbeat_desk_interval, open_desk_interval)

DEFAULT_POLICY = {
    "debounce_seconds": 3.0,        # a state must hold this long before it's recorded
    "away_grace_seconds": 0.0,      # away stretches shorter than this stay "working"
    "phone_sustained_seconds": 30.0,  # phone use counts against working only past this
    "working_formula": "at_desk_minus_sustained_phone",
    # Seated-shape gate (2026-09-01, measured on this footage): a seated
    # person's box is squat (h/w ~1.5, p90 2.3), a standing person's tall
    # (p10 2.3, median 3.1). Bodies taller than this ratio are standing at
    # the desk, not sitting at it -- they never take or hold a seat.
    "seated_max_aspect": 2.1,
}

AT_DESK, PHONE = "at_desk", "phone"


def policy_store(data_dir: Path) -> JsonFileStore:
    return JsonFileStore(data_dir / "policy.json", lambda: dict(DEFAULT_POLICY))


# ── state engine ────────────────────────────────────────────────────────────

@dataclass
class _Pending:
    state: str          # observed state waiting to be committed
    since: datetime     # video wall-clock the evidence started


class DeskStateEngine:
    """Debounced AT_DESK/away state per employee, persisted as desk_interval
    rows. observe() is called once per consumer cycle with the union of this
    cycle's evidence across every camera."""

    def __init__(self, conn: sqlite3.Connection, policy: JsonFileStore):
        self._conn = conn
        self._policy = policy
        self._committed: dict[str, str] = {}          # employee -> 'at_desk' | 'away'
        self._pending: dict[str, _Pending] = {}
        self._open: dict[str, str] = {}                # employee -> open interval id

    def observe(self, at_desk: dict[str, bool], now: datetime) -> bool:
        """at_desk: {employee_id: True if in a seat zone on any camera this
        cycle, False if visible but not in one}. Employees absent from the
        dict are treated as not-at-desk (Q13a). Returns True if anything was
        written (a transition or heartbeat)."""
        debounce = float(self._policy.load().get("debounce_seconds", 3.0))
        changed = False
        for emp in set(at_desk) | set(self._committed) | set(self._pending):
            observed = AT_DESK if at_desk.get(emp) else "away"
            committed = self._committed.get(emp, "away")
            if observed == committed:
                self._pending.pop(emp, None)
            else:
                pend = self._pending.get(emp)
                if pend is None or pend.state != observed:
                    self._pending[emp] = _Pending(observed, now)
                elif (now - pend.since).total_seconds() >= debounce:
                    self._commit(emp, observed, pend.since)
                    changed = True
            if self._committed.get(emp) == AT_DESK:
                heartbeat_desk_interval(self._conn, self._open[emp], now)
                changed = True
        return changed

    def _commit(self, emp: str, state: str, at: datetime) -> None:
        self._pending.pop(emp, None)
        if state == AT_DESK:
            iid = str(uuid.uuid4())
            open_desk_interval(self._conn, iid, emp, AT_DESK, at)
            self._open[emp] = iid
        else:
            iid = self._open.pop(emp, None)
            if iid:
                close_desk_interval(self._conn, iid, at)
        self._committed[emp] = state

    def current_state(self, emp: str) -> str:
        return self._committed.get(emp, "away")

    def clear(self) -> None:
        """Loop-wrap reset: forget everything and wipe the table."""
        self._committed.clear()
        self._pending.clear()
        self._open.clear()
        clear_all_desk_intervals(self._conn)


# ── read side ───────────────────────────────────────────────────────────────

def derive(at_desk_s: float, phone_s: float, presence_s: float, policy: dict) -> dict:
    """The one place the working/not-working policy lives (Q7c, Q9, Q13a)."""
    sustained = float(policy.get("phone_sustained_seconds", 30.0))
    # Without per-interval phone durations yet, treat the phone total as a
    # single stretch: it counts once it exceeds the sustained threshold.
    phone_counted = phone_s if phone_s >= sustained else 0.0
    # Working can never exceed the presence span (at-desk time is by
    # definition inside it); absent = 0/0.
    working = min(max(0.0, at_desk_s - phone_counted), presence_s)
    not_working = max(0.0, presence_s - working)
    return {"working_seconds": round(working), "not_working_seconds": round(not_working)}


def monitoring_snapshot(conn: sqlite3.Connection, day: date, policy: dict,
                        engine: DeskStateEngine | None) -> list[dict]:
    employees = conn.execute(
        "SELECT id, name, display_name_ar, department, code FROM employee WHERE active = 1 ORDER BY name"
    ).fetchall()
    out = []
    for emp in employees:
        att = compute_daily_attendance(conn, emp["id"], day)
        totals = desk_seconds_by_state(conn, emp["id"], day)
        at_desk_s = totals.get(AT_DESK, 0.0)
        phone_s = totals.get(PHONE, 0.0)
        presence_s = ((att.last_departure - att.first_arrival).total_seconds()
                      if att.present else 0.0)
        if not att.present:
            state = "absent"
        elif engine is not None and engine.current_state(emp["id"]) == AT_DESK:
            state = AT_DESK
        else:
            state = "away"
        rec = {
            "employee_id": emp["id"],
            "name": emp["name"],
            "display_name_ar": emp["display_name_ar"],
            "department": emp["department"],
            "code": emp["code"],
            "state": state,
            "at_desk_seconds": round(at_desk_s),
            "phone_seconds": round(phone_s),
            "away_seconds": round(max(0.0, presence_s - at_desk_s)),
            "presence_seconds": round(presence_s),
        }
        rec.update(derive(at_desk_s, phone_s, presence_s, policy))
        out.append(rec)
    return out


# ── seat occupancy: identity anchored to the seat, not the track id ────────

@dataclass
class ZoneBody:
    """One tracked person as seen this cycle, for the occupancy rules."""
    track_id: int
    zone_id: str | None
    employee_id: str | None      # untainted track binding, if any
    stationary: bool
    face_confirmed: bool         # a face match landed on this track this cycle
    seated: bool = True          # stationary AND seated-shaped (h/w <= seated_max_aspect)


class SeatOccupancy:
    """Who occupies each seat zone (2026-09-01, user-reported cases: a seated
    person fully occluded by a passer-by lost the seat to them; a seated
    person who turned their back lost credit once their track binding was
    touched).

    Track ids are unreliable for exactly the seated case -- the tracker hands
    a fully-occluded seated person's id to the occluder, and a back-turned
    worker can never re-confirm by face. So identity is anchored to the SEAT:
      * A seat becomes an employee's when their bound track sits (stationary)
        in it.
      * It stays theirs while ANY stationary body keeps occupying it -- the
        body's track id is irrelevant -- surviving gaps (occlusion, detector
        dropout) up to GAP_SECONDS.
      * Someone standing in front of an occupied seat never takes it. Another
        employee takes it only by sitting there with a face confirmation for
        TAKEOVER_SECONDS, or after the seat has been empty longer than GAP.
      * If the occupant's own bound track is seen seated in a DIFFERENT seat,
        the old seat is released (they moved desks).
    """

    GAP_SECONDS = 10.0
    TAKEOVER_SECONDS = 5.0

    def __init__(self):
        # (source_id, zone_id) -> {"employee": str, "last_seen": monotonic}
        self._occupant: dict[tuple[str, str], dict] = {}
        # (source_id, track_id, zone_id) -> monotonic since stationary there
        self._dwell: dict[tuple[str, int, str], float] = {}

    def occupant_of(self, source_id: str, zone_id: str) -> str | None:
        occ = self._occupant.get((source_id, zone_id))
        return occ["employee"] if occ else None

    def observe(self, source_id: str, bodies: list[ZoneBody], now: float) -> dict[str, str]:
        """Apply one cycle of evidence for one camera. Returns
        {employee_id: zone_id} for every employee credited as seated this
        cycle on this camera."""
        # dwell bookkeeping: how long each track has been stationary in a zone
        seen_keys = set()
        for b in bodies:
            if b.zone_id is not None and b.stationary and b.seated:
                key = (source_id, b.track_id, b.zone_id)
                self._dwell.setdefault(key, now)
                seen_keys.add(key)
        for key in [k for k in self._dwell if k[0] == source_id and k not in seen_keys]:
            del self._dwell[key]

        by_zone: dict[str, list[ZoneBody]] = {}
        for b in bodies:
            if b.zone_id is not None and b.stationary and b.seated:
                by_zone.setdefault(b.zone_id, []).append(b)

        credited: dict[str, str] = {}
        zones = set(by_zone) | {z for (s, z) in self._occupant if s == source_id}
        for zone_id in zones:
            seated = by_zone.get(zone_id, [])
            key = (source_id, zone_id)
            occ = self._occupant.get(key)
            alive = occ is not None and now - occ["last_seen"] <= self.GAP_SECONDS
            bound = [b for b in seated if b.employee_id]
            confirmed = [b for b in bound if b.face_confirmed]

            if alive:
                # Takeover only by a face-confirmed DIFFERENT employee who has
                # actually been sitting there a while.
                takers = [b for b in confirmed if b.employee_id != occ["employee"]
                          and now - self._dwell.get((source_id, b.track_id, zone_id), now) >= self.TAKEOVER_SECONDS]
                if takers:
                    occ = {"employee": takers[0].employee_id, "last_seen": now}
                    self._occupant[key] = occ
                elif seated:
                    occ["last_seen"] = now   # continuity: any stationary body keeps the seat
                if seated:
                    credited[occ["employee"]] = zone_id
            else:
                if occ is not None:
                    del self._occupant[key]
                pick = (confirmed or bound)
                if pick:
                    self._occupant[key] = {"employee": pick[0].employee_id, "last_seen": now}
                    credited[pick[0].employee_id] = zone_id
                # anonymous body in an unowned seat: nobody to credit

        # An occupant whose OWN bound track is seated in another zone has
        # moved desks: release the seat they only hold by continuity.
        bound_zone_by_emp = {b.employee_id: b.zone_id for b in bodies
                             if b.employee_id and b.zone_id and b.stationary and b.seated}
        for (src, zone_id), occ in list(self._occupant.items()):
            if src != source_id:
                continue
            real = bound_zone_by_emp.get(occ["employee"])
            if real is not None and real != zone_id:
                del self._occupant[(src, zone_id)]
                if credited.get(occ["employee"]) == zone_id:
                    credited[occ["employee"]] = real
        return credited

    def release_where_elsewhere(self, binding_sources: dict[str, str]) -> list[str]:
        """Global identity: if a seat's occupant currently has a live
        binding on a DIFFERENT camera, they are not in that seat -- release
        it. Returns the employees whose seats were released."""
        released = []
        for (src, zone_id), occ in list(self._occupant.items()):
            where = binding_sources.get(occ["employee"])
            if where is not None and where != src:
                del self._occupant[(src, zone_id)]
                released.append(occ["employee"])
        return released

    # readable alias used by the consumer
    occupant_credits = observe

    def clear(self) -> None:
        self._occupant.clear()
        self._dwell.clear()
