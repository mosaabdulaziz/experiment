"""Customer records: one Visit per person from entry to exit, its state
moved by what the engine reports from the zones, and one ServiceSession per
haircut. Everything that changes lands in the pending EngineUpdate
(take_update). Only confirmed visits are persisted: a visit is confirmed
min_visit_s after a door entry, or at once for a missed entry or a haircut.

States: waiting (entered, no chair yet) -> haircut (seated) <-> washing
(left the seat for the wash; the haircut is suspended, not ended) -> served
(haircut done, still inside) -> finished (door exit or lost)."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from alerts import EXCESSIVE_SERVICE, EXCESSIVE_WAIT, AlertBook
from domain import (COMPLETED, ENTERED, EXIT_DOOR, FINISHED, HAIRCUT, LOST, SERVED, SHOP, VOID_STAFF, WAITING,
                    WALKED_OUT, WASHING, Binding, EngineUpdate, Handoff, ServiceSession, StateEvent, Visit)


def seconds(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds()


class CustomerBook:
    def __init__(self, new_id: Callable[[], str],
                 barber_for: Callable[[str, datetime], str | None] = lambda chair, at: None,
                 ticket_for: Callable[[datetime], int] = lambda at: 0):
        self._new_id, self.barber_for, self.ticket_for = new_id, barber_for, ticket_for
        self.visits: dict[str, Visit] = {}                # person id -> open visit
        self.sessions: dict[str, ServiceSession] = {}     # open haircuts (running or at the wash)
        self.last_closed: dict[str, ServiceSession] = {}  # person id -> his latest closed haircut (chair resume)
        self.alerts = AlertBook(new_id)
        self.update = EngineUpdate()

    def take_update(self) -> EngineUpdate:
        update, self.update = self.update, EngineUpdate()
        return update

    # ── bookkeeping ─────────────────────────────────────────────────────────

    def _changed(self, v: Visit) -> None:
        if v.confirmed:
            self.update.visits[v.id] = v

    def _state(self, v: Visit, to: str, at: datetime, area: str | None = None) -> None:
        if v.state == to:
            return
        self.update.events.append(StateEvent(v.id, None if v.state == ENTERED else v.state, to, area, None, at))
        v.state = to
        v.state_since = at
        self._changed(v)

    def _visit_of(self, s: ServiceSession) -> Visit | None:
        return next((v for v in self.visits.values() if v.id == s.visit_id), None)

    def _alert(self, alert) -> None:
        if alert is not None:
            self.update.alerts.append(alert)

    # ── visits ──────────────────────────────────────────────────────────────

    def open(self, person_id: str, at: datetime, observed: bool, policy: dict, confirmed: bool = False) -> Visit:
        v = Visit(self._new_id(), person_id, at, observed, wait_target_s=policy["wait_target_s"])
        self.visits[person_id] = v
        if confirmed:
            self.confirm(person_id)
        return v

    def confirm(self, person_id: str) -> None:
        v = self.visits[person_id]
        if v.confirmed:
            return
        v.confirmed = True
        v.waiting_started_at = v.entered_at
        v.ticket = self.ticket_for(v.entered_at)
        self._state(v, WAITING, v.entered_at)

    def exit_crossed(self, person_id: str, at: datetime) -> None:
        v = self.visits.get(person_id)
        if v is not None and v.exit_crossed_at is None:
            v.exit_crossed_at = at

    def cancel_exit(self, person_id: str) -> None:
        v = self.visits.get(person_id)
        if v is not None:
            v.exit_crossed_at = None

    def finish(self, person_id: str, at: datetime, reason: str, policy: dict) -> None:
        v = self.visits.get(person_id)
        if v is None:
            return
        if not v.confirmed:            # a peek through the door: no record
            del self.visits[person_id]
            return
        s = self.sessions.get(v.session_id) if v.session_id else None
        if s is not None:
            self._close(s, s.washing_since or at, policy)
        del self.visits[person_id]
        self.last_closed.pop(person_id, None)
        self._alert(self.alerts.close(EXCESSIVE_WAIT, v.id, at))
        v.exited_at = max(at, v.entered_at)
        v.exit_reason = reason
        v.open = False
        if v.haircuts:
            v.outcome = COMPLETED
        else:
            v.outcome = WALKED_OUT if reason == EXIT_DOOR else LOST
            v.wait_s = max(0.0, seconds(v.waiting_started_at, v.exited_at))   # a walk-out waits until the exit
        self._state(v, FINISHED, v.exited_at)
        self._changed(v)

    def void(self, person_id: str, at: datetime) -> bool:
        """The "customer" turned out to be staff. Refused (False) once a
        haircut is on record: that visit was a customer's."""
        v = self.visits.get(person_id)
        if v is None:
            return True
        if v.haircuts or v.session_id:
            return False
        del self.visits[person_id]
        self.last_closed.pop(person_id, None)
        if v.confirmed:
            self._alert(self.alerts.close(EXCESSIVE_WAIT, v.id, at))
            v.outcome, v.open, v.exited_at = VOID_STAFF, False, max(at, v.entered_at)
            self._state(v, FINISHED, v.exited_at)
            self._changed(v)
        return True

    # ── haircuts ────────────────────────────────────────────────────────────

    def start_haircut(self, person_id: str, chair: str, since: datetime, policy: dict) -> ServiceSession:
        v = self.visits[person_id]
        self.confirm(person_id)
        s = self.sessions.get(v.session_id) if v.session_id else None
        if s is not None and s.washing_since is not None and s.chair_id == chair:
            s.wash_s += max(0.0, seconds(s.washing_since, since))
            s.washing_since = None
            self.update.sessions[s.id] = s
            self._state(v, HAIRCUT, since, chair)
            return s
        if s is not None:
            self._close(s, s.washing_since or since, policy)
        s = ServiceSession(self._new_id(), v.id, chair, since, barber_id=self.barber_for(chair, since),
                           service_target_s=policy["service_target_s"])
        self.sessions[s.id] = s
        v.session_id = s.id
        self.update.sessions[s.id] = s
        if v.service_started_at is None:
            v.service_started_at = since
            v.wait_s = max(0.0, seconds(v.waiting_started_at, since))
            self._alert(self.alerts.close(EXCESSIVE_WAIT, v.id, since))
        self._state(v, HAIRCUT, since, chair)
        self._changed(v)
        return s

    def reopen_haircut(self, person_id: str, session_id: str, at: datetime, policy: dict) -> ServiceSession | None:
        """Chair resume (2026-09-28, 39-min testbed spec 6.1): the customer is
        back in the seat of the haircut that just ended -- reopen that very
        session (its end cleared, the gap kept inside it); the visit keeps its
        wait and returns to haircut at `at`. None unless it is his latest
        closed haircut and no other one is open."""
        v = self.visits.get(person_id)
        s = self.last_closed.get(person_id)
        if v is None or s is None or s.id != session_id or v.session_id is not None:
            return None
        del self.last_closed[person_id]
        if s.service_s < policy["min_service_s"]:
            # a sit too short to count: _close deleted it (storage too, once flushed; the upsert below
            # writes it back) and, before any haircut, reset the wait -- restore both
            self.update.deleted_sessions.discard(s.id)
            if v.service_started_at is None:
                v.service_started_at = s.started_at
                v.wait_s = max(0.0, seconds(v.waiting_started_at, s.started_at))
        else:
            v.haircuts -= 1              # counted again when it ends
        s.ended_at = s.service_s = None
        s.open = True
        self.sessions[s.id] = s
        v.session_id = s.id
        self.update.sessions[s.id] = s
        self._alert(self.alerts.close(EXCESSIVE_WAIT, v.id, at))
        self._state(v, HAIRCUT, at, s.chair_id)
        self._changed(v)
        return s

    def end_haircut(self, session_id: str, at: datetime, washing: bool, policy: dict) -> None:
        s = self.sessions.get(session_id)
        if s is None:
            return
        v = self._visit_of(s)
        if washing and v is not None:
            s.washing_since = at
            self.update.sessions[s.id] = s
            self._state(v, WASHING, at)
            return
        self._close(s, at, policy)

    def expire_washing(self, now: datetime, policy: dict) -> None:
        for s in list(self.sessions.values()):
            if s.washing_since is not None and seconds(s.washing_since, now) >= policy["wash_return_s"]:
                self._close(s, s.washing_since, policy)

    def _close(self, s: ServiceSession, at: datetime, policy: dict) -> None:
        self.sessions.pop(s.id, None)
        v = self._visit_of(s)
        s.ended_at = max(at, s.started_at)
        s.open = False
        s.washing_since = None
        s.service_s = seconds(s.started_at, s.ended_at)
        self._alert(self.alerts.close(EXCESSIVE_SERVICE, s.id, s.ended_at))
        if v is not None and v.session_id == s.id:
            v.session_id = None
        if v is not None:
            self.last_closed[v.person_id] = s
        if s.service_s < policy["min_service_s"]:
            self.update.sessions.pop(s.id, None)
            self.update.deleted_sessions.add(s.id)
            if v is not None:
                if v.haircuts == 0:
                    v.service_started_at = v.wait_s = None
                self._state(v, WAITING if v.haircuts == 0 else SERVED, s.ended_at)
            return
        self.update.sessions[s.id] = s
        if v is not None:
            v.haircuts += 1
            self._state(v, SERVED, s.ended_at)
            self._changed(v)

    # ── alerts, handoffs ────────────────────────────────────────────────────

    def check_alerts(self, now: datetime) -> None:
        for v in self.visits.values():
            if v.confirmed and v.state == WAITING and v.service_started_at is None:
                a = self.alerts.check(EXCESSIVE_WAIT, v.id, camera=SHOP, visit_id=v.id, session_id=None,
                                      since=v.waiting_started_at, target_s=v.wait_target_s, now=now)
                if a is not None:
                    v.excessive_wait = True
                    self._alert(a)
                    self._changed(v)
        for s in self.sessions.values():
            if s.excessive_service:
                continue                # one alert per haircut, even reopened (chair resume)
            a = self.alerts.check(EXCESSIVE_SERVICE, s.id, camera=SHOP, visit_id=s.visit_id, session_id=s.id,
                                  since=s.started_at, target_s=s.service_target_s, now=now)
            if a is not None:
                s.excessive_service = True
                self._alert(a)
                self.update.sessions[s.id] = s

    def note_handoff(self, handoff: Handoff) -> None:
        self.update.handoffs.append(handoff)

    def note_binding(self, b: Binding) -> None:
        self.update.bindings.append(b)
