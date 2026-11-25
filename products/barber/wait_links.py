"""Which customer waits led to a haircut (door-free visits, 2026-10-14; branch 2 spec section 7).

By time only (caped customers can't be matched by re-ID across cameras): each haircut start, in time order, takes
the earliest unlinked wait that ended within window_s before it -- or up to skew_s after it, since the chair camera
dates a start back to the sitting while the wait camera ends the wait at the sofa's edge. Shared by the stats
(visits = served + unserved waits) and the wait book (a return after a served wait is no new visit).
A wait the wait camera's stall ended (ended_by STALL, wt_end_mode 1) is never linked: its customer was most likely
still on the sofa, and a haircut starting meanwhile is someone else's (final review I2, 2026-10-15)."""
from __future__ import annotations

from typing import Iterable

SKEW_S = 10.0
STALL = "stall"     # ended_by of the waits stale() ends in wt_end_mode 1 (mode 0 keeps "lost")


def link_waits(waits: Iterable[tuple[str, float | None]], starts: Iterable[tuple[str, float]], window_s: float,
               skew_s: float = SKEW_S) -> dict[str, str]:
    """wait id -> the session id that served it. waits: (id, end ts or None while open); starts: (session id, ts)."""
    ended = sorted((end, wid) for wid, end in waits if end is not None)
    out: dict[str, str] = {}
    for s_ts, sid in sorted((ts, sid) for sid, ts in starts):
        for end, wid in ended:
            if end > s_ts + skew_s:
                break
            if wid not in out and s_ts - end <= window_s:
                out[wid] = sid
                break
    return out
