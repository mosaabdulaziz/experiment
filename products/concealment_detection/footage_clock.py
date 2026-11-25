"""Owned footage clock for replayed test footage vs live device time.

Provenance: adapted from `products/barber/footage_clock.py`. Owned independently by
`products/concealment_detection` so products share no code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FootageClock:
    start: datetime
    shift: timedelta
    duration_s: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)

    def footage(self, device: datetime) -> datetime:
        # Device datetimes represent epochs. Naive ones follow fromtimestamp's
        # local-time convention; aware ones must be converted, not stripped.
        if self.start.tzinfo is not None:
            device = device.astimezone(self.start.tzinfo)
        elif device.tzinfo is not None:
            device = device.astimezone().replace(tzinfo=None)
        return device - self.shift

    def device(self, footage: datetime) -> datetime:
        if self.start.tzinfo is not None:
            footage = footage.replace(tzinfo=self.start.tzinfo) if footage.tzinfo is None else footage.astimezone(self.start.tzinfo)
        elif footage.tzinfo is not None:
            footage = footage.astimezone().replace(tzinfo=None)
        return footage + self.shift


def footage_clock(footage_start: datetime | None, run: dict | None) -> FootageClock | None:
    """Build a lap-adjusted `FootageClock` when `footage_start` and `run` are present."""
    if not footage_start or not run:
        return None
    lap0_ts = min(float(v[0]) for v in run.values())
    duration_s = max(float(v[1]) for v in run.values())
    lap = max(int(v[2]) for v in run.values())
    current_lap_start = datetime.fromtimestamp(lap0_ts + lap * duration_s, tz=footage_start.tzinfo)
    return FootageClock(
        start=footage_start,
        shift=current_lap_start - footage_start,
        duration_s=duration_s,
    )


def display_timestamp(
    device_dt: datetime,
    pts: float | None = None,
    clock: FootageClock | None = None,
) -> str:
    """Return an ISO-8601 timestamp string using `clock` when active, else `device_dt`."""
    if clock is not None:
        return clock.footage(device_dt).isoformat()
    return device_dt.isoformat()
