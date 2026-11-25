"""The use case's ten statistics over a period, optionally as a time series.

Each figure is anchored to the moment it became true, so a period never
counts half a fact:
  * customer visits           -- by entered_at (staff voids excluded)
  * completed haircuts        -- by the haircut's ended_at
  * average / longest wait    -- customers who reached a chair, by when they
                                 reached it (a no-wait customer counts as 0)
  * excessive-wait cases      -- every measured wait, walk-outs included (a
                                 40-minute wait that ended in a walk-out is
                                 the case that matters most), by when it ended
  * average / longest haircut, excessive-haircut cases -- by ended_at
  * haircuts per barber       -- the credited barber of each haircut, from
                                 the roster (service_session.barber_id)
2026-09-28 (user): the two averages also count what isn't over yet -- with
`live` (the engine's customers) and `now` inside the period, a customer
waiting now counts his wait so far, and a running haircut (min_service_s
in) its time so far -- and a visit voided as staff counts its wait up to
the void (flagged to be fixed later). A customer first seen sitting in a
chair (visit.wait_unseen) has no known wait: every wait figure leaves him out. Counts and excessive-case figures
stay on what ended; the longest wait / haircut count the running ones too, marked running (2026-10-15, user).
Served customers = confirmed haircut sessions by start (chair engine); Visits = the start-of-day baseline
(haircuts + waiting + wash-area people already inside, by when it was taken) + non-staff door ins, never fewer
than the served customers.
visits_source 1 (door-free, 2026-10-14): Visits = served customers + waits that never led to a haircut (wait_links.link_waits).
Chair engine: finished customer waits (customer_wait) join the average wait; longest wait and the
too-long share come from visits only (empty in chair mode) -- except door-free (2026-10-18, user: the reports read
"longest wait —" and "0 excessive waits" beside the wait alerts), where they come from the customer waits: the
longest finished one, and the waits with an excessive-wait alert (each once) out of the waits that started.
"Customers currently waiting" is live state, not history: the API takes
it from the running engines (VisitEngine.live).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from domain import HAIRCUT, LOST, WAITING, WALKED_OUT, WASHING
from storage import (baselines_in, door_ins, excessive_wait_ids, haircuts_ended, iso, list_staff, sessions_started,
                     visits_entered, voided_waits, waits_ended, waits_finished, waits_started)
from wait_links import SKEW_S, STALL, link_waits

BUCKETS = ("hour", "day", "week", "month")
MAX_BUCKETS = 2000


def bucket_start(dt: datetime, bucket: str, day_start_hour: int = 0) -> datetime:
    """Day, week and month buckets run from the shop day's start hour (so a
    visit after midnight lands on the evening it belongs to); hours are hours."""
    if bucket == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    shift = timedelta(hours=day_start_hour)
    day = (dt - shift).replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "day":
        start = day
    elif bucket == "week":
        start = day - timedelta(days=day.weekday())  # Monday
    elif bucket == "month":
        start = day.replace(day=1)
    else:
        raise ValueError(f"bucket must be one of {', '.join(BUCKETS)}")
    return start + shift


def _next_bucket(dt: datetime, bucket: str) -> datetime:
    if bucket == "hour":
        return dt + timedelta(hours=1)
    if bucket == "day":
        return dt + timedelta(days=1)
    if bucket == "week":
        return dt + timedelta(days=7)
    return (dt.replace(day=28) + timedelta(days=4)).replace(day=1)


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _share(count: int, total: int) -> dict:
    return {"count": count, "percent": round(100.0 * count / total, 1) if total else None}


def first_visits(conn: sqlite3.Connection) -> dict[str, str]:
    """Each registered customer's first entered_at."""
    return {r["customer_id"]: r[1] for r in conn.execute(
        "SELECT customer_id, MIN(entered_at) FROM visit WHERE customer_id IS NOT NULL GROUP BY customer_id")}


def returning(conn: sqlite3.Connection, start: datetime, end: datetime,
              earlier: dict[str, str] | None = None) -> dict:
    """Among registered visits that entered in [start, end): the share whose
    customer had an earlier visit, and the mean gap between consecutive visits
    of customers with 2+ visits in the period. `earlier` (first_visits) can
    be passed in when called once per bucket."""
    rows = conn.execute("SELECT customer_id, entered_at FROM visit WHERE customer_id IS NOT NULL"
                        " AND entered_at >= ? AND entered_at < ? ORDER BY customer_id, entered_at",
                        (iso(start), iso(end))).fetchall()
    if earlier is None:
        earlier = first_visits(conn)
    count = sum(1 for r in rows if r["entered_at"] > earlier[r["customer_id"]])
    gaps, prev = [], {}
    for r in rows:
        t = datetime.fromisoformat(r["entered_at"])
        if r["customer_id"] in prev:
            gaps.append((t - prev[r["customer_id"]]).total_seconds() / 86400)
        prev[r["customer_id"]] = t
    return {"returning_visits": _share(count, len(rows)),
            "avg_days_between_visits": round(sum(gaps) / len(gaps), 1) if gaps else None}


def _seconds(a: str, b: str) -> float:
    return max(0.0, (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())


def live_values(live: list[dict], now: datetime, min_service_s: float,
                barber_id: str | None = None) -> tuple[list[float], list[float]]:
    """(waits so far, haircuts so far) of the engine's live customers at `now`."""
    waits, cuts = [], []
    for c in live:
        if c["state"] == WAITING and c.get("waiting_started_at") and barber_id is None:
            waits.append(_seconds(c["waiting_started_at"], now.isoformat()))
        elif c["state"] in (HAIRCUT, WASHING) and c.get("haircut_started_at") \
                and (barber_id is None or c.get("barber_id") == barber_id):
            s = _seconds(c["haircut_started_at"], now.isoformat())
            if s >= min_service_s:
                cuts.append(s)
    return waits, cuts


def live_longest(live: list[dict], now: datetime, min_service_s: float,
                 barber_id: str | None = None) -> tuple[dict | None, dict | None]:
    """The longest running wait and haircut at `now` (time so far), as the longest_* dicts with running True."""
    wait = cut = None
    for c in live:
        if c["state"] == WAITING and c.get("waiting_started_at") and barber_id is None:
            sec = _seconds(c["waiting_started_at"], now.isoformat())
            if wait is None or sec > wait["seconds"]:
                wait = {"seconds": round(sec, 1), "visit_id": c.get("visit_id"), "waiting_started_at": c["waiting_started_at"],
                        "service_started_at": None, "running": True}
        elif c["state"] in (HAIRCUT, WASHING) and c.get("haircut_started_at") \
                and (barber_id is None or c.get("barber_id") == barber_id):
            sec = _seconds(c["haircut_started_at"], now.isoformat())
            if sec >= min_service_s and (cut is None or sec > cut["seconds"]):
                cut = {"seconds": round(sec, 1), "session_id": c.get("session_id"), "visit_id": c.get("visit_id"),
                       "chair_id": c.get("chair_id"), "barber_id": c.get("barber_id"),
                       "started_at": c["haircut_started_at"], "ended_at": None, "running": True}
    return wait, cut


def _take_longest(result: dict, key: str, live: dict | None) -> None:
    """2026-10-15 (user): the longest counts what is still running, as the average does (a report read 'longest
    haircut 09:53, average 15:14' while the long haircuts were still in the chair)."""
    if live is not None and (result[key] is None or live["seconds"] > result[key]["seconds"]):
        result[key] = live


def summarize(visits: list, waits: list, haircuts: list, extra_waits: list[float] = (),
              extra_cuts: list[float] = (), started: list = (), ins: list = (), baselines: list = (),
              min_wait_s: float = 0.0, visitors: int | None = None) -> dict:
    """extra_waits / extra_cuts: values only the averages count (voided and live ones). Waits shorter than
    min_wait_s stay out of the average only (2026-10-12, user: very short waits pulled it down)."""
    served = [w for w in waits if w["service_started_at"] is not None]
    longest_wait = max(served, key=lambda w: w["wait_s"], default=None)
    longest_cut = max(haircuts, key=lambda h: h["service_s"], default=None)
    return {
        "customer_visits": len(visits),
        "walk_outs": sum(1 for v in visits if v["outcome"] == WALKED_OUT),
        "lost": sum(1 for v in visits if v["outcome"] == LOST),
        "completed_haircuts": len(haircuts),
        "served_customers": len(started), "visitors": visitors if visitors is not None else max(len(ins) + sum(b["count"] for b in baselines), len(started)),   # user 2026-10-05: never fewer visits than served customers (door misses)
        "average_wait_s": _avg([s for s in [w["wait_s"] for w in served] + list(extra_waits) if s >= min_wait_s]),
        "longest_wait": None if longest_wait is None else {
            "seconds": round(longest_wait["wait_s"], 1), "visit_id": longest_wait["id"],
            "waiting_started_at": longest_wait["waiting_started_at"],
            "service_started_at": longest_wait["service_started_at"]},
        "excessive_wait": _share(sum(1 for w in waits if w["excessive_wait"]), len(waits)),
        "average_haircut_s": _avg([h["service_s"] for h in haircuts] + list(extra_cuts)),
        "longest_haircut": None if longest_cut is None else {
            "seconds": round(longest_cut["service_s"], 1), "session_id": longest_cut["id"],
            "visit_id": longest_cut["visit_id"], "chair_id": longest_cut["chair_id"],
            "barber_id": longest_cut["barber_id"],
            "started_at": longest_cut["started_at"], "ended_at": longest_cut["ended_at"]},
        "excessive_haircut": _share(sum(1 for h in haircuts if h["excessive_service"]), len(haircuts)),
    }


def per_barber(conn: sqlite3.Connection, haircuts: list, started: list | None = None) -> list[dict]:
    """Haircuts per barber: the count is the served customers (`started`: confirmed sessions by start, running ones
    included -- user 2026-10-05, so the chart matches the Served customers figure); the average is over the
    finished ones (`haircuts`). Without `started` the count is the finished haircuts (as before)."""
    names = {s["id"]: s["name"] for s in list_staff(conn)}
    lengths: dict[str | None, list[float]] = {}
    for h in haircuts:
        lengths.setdefault(h["barber_id"], []).append(h["service_s"])
    counts: dict[str | None, int] = {}
    for h in (started if started is not None else haircuts):
        counts[h["barber_id"]] = counts.get(h["barber_id"], 0) + 1
    rows = [{"barber_id": b, "name": names.get(b) if b else None, "haircuts": n,
             "average_haircut_s": _avg(lengths.get(b, []))} for b, n in counts.items()]
    # most haircuts first; the unattributed bucket (barber_id None) last
    return sorted(rows, key=lambda r: (r["barber_id"] is None, -r["haircuts"], r["barber_id"] or ""))


def compute_stats(conn: sqlite3.Connection, start: datetime, end: datetime,
                  bucket: str | None = None, outcome: str | None = None,
                  barber_id: str | None = None, day_start_hour: int = 0, live: list[dict] | None = None,
                  now: datetime | None = None, min_service_s: float = 0.0, min_wait_s: float = 0.0,
                  visits_source: int = 0, serve_s: float = 150.0, clock_shift: timedelta = timedelta(0)) -> dict:
    """`outcome` / `barber_id` narrow every figure to those visits (and, for
    haircuts, that barber's) -- the Insights page's filters apply to it all.
    `day_start_hour` aligns day/week/month buckets to the shop day. `live`
    and `now`: see the module doc (no outcome filter only). `clock_shift` (2026-10-18): the buckets follow the clock
    that is device time + this -- a run on recordings' footage clock (footage_clock.py) -- and are labelled in it."""
    if end <= start:
        raise ValueError("end must be after start")
    visits = visits_entered(conn, start, end, outcome, barber_id)
    waits = [w for w in waits_ended(conn, start, end, outcome, barber_id) if not w["wait_unseen"]]
    haircuts = haircuts_ended(conn, start, end, outcome, barber_id)
    started = sessions_started(conn, start, end, barber_id) if outcome is None else []
    door_free = visits_source == 1 and outcome is None and barber_id is None
    linked: set[str] = set()
    started_waits: list = []
    if door_free:
        started_waits = waits_started(conn, start, end)
        later = sessions_started(conn, start, end + timedelta(seconds=serve_s + SKEW_S))
        ts = lambda v: datetime.fromisoformat(v).timestamp()
        ended = lambda w: ts(w["ended_at"]) if w["ended_at"] and w["ended_by"] != STALL else None   # stall: unserved
        linked = set(link_waits([(w["id"], ended(w)) for w in started_waits],
                                [(s["id"], ts(s["started_at"])) for s in later], serve_s))

    def visitors_in(waits_rows, started_rows):
        return len(started_rows) + sum(1 for w in waits_rows if w["id"] not in linked) if door_free else None

    ins = door_ins(conn, start, end)
    bases = baselines_in(conn, start, end)
    voided = voided_waits(conn, start, end) if outcome is None and barber_id is None else []
    finished = waits_finished(conn, start, end) if outcome is None and barber_id is None else []
    live_w, live_c = ([], [])
    long_w = long_c = None
    if live and now is not None and outcome is None and start <= now < end:
        live_w, live_c = live_values(live, now, min_service_s, barber_id)
        long_w, long_c = live_longest(live, now, min_service_s, barber_id)

    def void_w(rows):
        return [_seconds(r["waiting_started_at"], r["exited_at"]) for r in rows]

    result = {"from": iso(start), "to": iso(end),
              **summarize(visits, waits, haircuts, void_w(voided) + [r["wait_s"] for r in finished] + live_w, live_c,
                         started=started, ins=ins, baselines=bases, min_wait_s=min_wait_s,
                         visitors=visitors_in(started_waits, started)),
              "haircuts_per_barber": per_barber(conn, haircuts, started if outcome is None else None)}
    _take_longest(result, "longest_wait", long_w)
    _take_longest(result, "longest_haircut", long_c)
    if door_free:                               # the waits are customer waits (module doc, 2026-10-18)
        done = max(finished, key=lambda w: w["wait_s"], default=None)
        if done is not None:
            _take_longest(result, "longest_wait", {"seconds": round(done["wait_s"], 1), "visit_id": done["id"],
                                                   "waiting_started_at": done["started_at"],
                                                   "service_started_at": done["ended_at"]})
        result["excessive_wait"] = _share(len(excessive_wait_ids(conn, start, end)), len(started_waits))
    earlier = first_visits(conn)
    result.update(returning(conn, start, end, earlier))
    if bucket is None:
        return result
    sh = clock_shift
    edges = [bucket_start(start + sh, bucket, day_start_hour) - sh]
    while edges[-1] < end:
        edges.append(_next_bucket(edges[-1] + sh, bucket) - sh)
        if len(edges) > MAX_BUCKETS:
            raise ValueError(f"too many {bucket} buckets for this period (max {MAX_BUCKETS})")

    def within(rows, field, lo, hi):
        return [r for r in rows if lo <= _parse(r[field]) < hi]

    series = []
    for lo, hi in zip(edges, edges[1:]):
        wait_end = [w for w in waits if lo <= _parse(w["service_started_at"] or w["exited_at"]) < hi]
        here = now is not None and lo <= now < hi
        series.append({"bucket": iso(lo + sh), **summarize(within(visits, "entered_at", lo, hi), wait_end,
                                                      within(haircuts, "ended_at", lo, hi),
                                                      void_w(within(voided, "exited_at", lo, hi))
                                                      + [r["wait_s"] for r in within(finished, "ended_at", lo, hi)]
                                                      + (live_w if here else []), live_c if here else [],
                                                      started=within(started, "started_at", lo, hi),
                                                      ins=within(ins, "at", lo, hi),
                                                      baselines=within(bases, "taken_at", lo, hi),
                                                      min_wait_s=min_wait_s,
                                                      visitors=visitors_in(within(started_waits, "started_at", lo, hi),
                                                                           within(started, "started_at", lo, hi))),
                       "returning_visits": returning(conn, lo, hi, earlier)["returning_visits"]})
    result["bucket"] = bucket
    result["series"] = series
    return result
