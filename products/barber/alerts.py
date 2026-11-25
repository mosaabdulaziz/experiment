"""Excessive wait / haircut alerts: opened when a running time crosses its
target, closed when that wait or haircut ends. The targets live on the
visit / session (snapshotted from the policy when the clock started), so
changing the policy later never rewrites what already happened."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

from domain import Alert

EXCESSIVE_WAIT, EXCESSIVE_SERVICE = "excessive_wait", "excessive_service"


class AlertBook:
    """Open alerts, keyed by (kind, subject) -- the visit for a wait, the
    session for a haircut -- so each subject has at most one open alert of
    each kind at a time."""

    def __init__(self, new_id: Callable[[], str]):
        self._new_id = new_id
        self._open: dict[tuple[str, str], Alert] = {}

    def check(self, kind: str, subject_id: str, *, camera: str, visit_id: str,
              session_id: str | None, since: datetime, target_s: float | None,
              now: datetime) -> Alert | None:
        """Open and return a new alert if the clock started at `since` has
        passed `target_s` and none is open for this subject yet."""
        key = (kind, subject_id)
        if key in self._open or target_s is None:
            return None
        if (now - since).total_seconds() <= target_s:
            return None
        alert = Alert(self._new_id(), kind, camera, visit_id, session_id,
                      started_at=since + timedelta(seconds=target_s))
        self._open[key] = alert
        return alert

    def close(self, kind: str, subject_id: str, at: datetime) -> Alert | None:
        """Close and return the subject's open alert, if any. `at` is clamped
        to the alert's own start: a backdated end (e.g. the haircut's
        debounced start) can land before the moment the alert fired."""
        alert = self._open.pop((kind, subject_id), None)
        if alert is not None:
            alert.ended_at = max(at, alert.started_at)
            alert.open = False
        return alert

    def open_alerts(self) -> list[Alert]:
        return list(self._open.values())

    def clear(self) -> None:
        self._open.clear()
