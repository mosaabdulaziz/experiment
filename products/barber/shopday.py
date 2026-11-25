"""The shop day: a barber shop's "today" runs from day_start_hour to the same
hour tomorrow, so a haircut after midnight counts toward the evening it
belongs to. Times are naive local wall-clock, like the rest of the service."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta


def shop_day(at: datetime, hour: int) -> date:
    return (at - timedelta(hours=hour)).date()


def day_start(d: date, hour: int) -> datetime:
    return datetime.combine(d, time(hour))


def shop_day_range(at: datetime, hour: int) -> tuple[datetime, datetime]:
    start = day_start(shop_day(at, hour), hour)
    return start, start + timedelta(days=1)


class TicketCounter:
    """The short number a customer is called by (#1, #2, ...), counted per
    shop day. `hour` follows the policy; `seed` resumes today's count after
    a restart."""

    def __init__(self, hour: int = 5, seed: dict[date, int] | None = None):
        self.hour = hour
        self._last: dict[date, int] = dict(seed or {})

    def next(self, at: datetime) -> int:
        d = shop_day(at, self.hour)
        self._last[d] = self._last.get(d, 0) + 1
        return self._last[d]

    def clear(self) -> None:
        """The rows were cleared (a new run of the test clips): numbering starts again at #1."""
        self._last = {}

    def resume(self, day: date, n: int) -> None:
        """A restart: raise `day`'s count to at least `n` (the DB's max ticket
        for that shop day), never lowers it."""
        self._last[day] = max(self._last.get(day, 0), n)
