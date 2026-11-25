"""The report model for one shop period (day / week / month): every figure, chart series, chair list and alert that
every report format shows, in one dict (spec docs/superpowers/specs/2026-10-15-barber-reports-design.md). Built from
the saved rows with the stats the dashboard uses; no rendering here. Times are naive local, like the service."""
from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, datetime, timedelta

from shopday import day_start, shop_day
from footage_clock import FootageClock
from stats import _share, compute_stats
from storage import list_alerts, pedicures_in, sessions_started

KINDS = ("day", "week", "month")
SNAPSHOTS_LONG = 12          # week / month: the longest alerts shown with their snapshots
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def period_bounds(kind: str, first: date, hour: int) -> tuple[datetime, datetime]:
    if kind == "day":
        days = 1
    elif kind == "week":
        days = 7
    elif kind == "month":
        first = first.replace(day=1)
        days = calendar.monthrange(first.year, first.month)[1]
    else:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    start = day_start(first, hour)
    return start, start + timedelta(days=days)


def _tag(d: date) -> str:
    return f"{d.day:02d}{MONTHS[d.month - 1]}"


def file_name(kind: str, start: datetime, end: datetime, place: str, slug: str, ext: str) -> str:
    first, last = start.date(), (end - timedelta(days=1)).date()
    span = _tag(first) if kind == "day" else f"{_tag(first)}_{_tag(last)}"
    return f"{span}_{place}_{slug}_analysis_report.{ext}"


def _ts(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _iso(d: datetime | None) -> str | None:
    return d.isoformat(timespec="seconds") if d else None


def _timed(rows: list[dict]) -> list[float]:
    return [r["duration_s"] for r in rows if r["duration_s"] is not None]


def build_report(conn, kind: str, first: date, now: datetime, policy: dict, site: dict,
                 chair_names: dict[str, str], live: list[dict] | None = None,
                 clock: FootageClock | None = None) -> dict:
    """`live`: the engine's live customers (as /api/barber/stats passes them), so a running period's averages count
    the people waiting and the haircuts in progress like the dashboard (final review I1). `clock` (2026-10-18, user):
    a run on recordings -- `first` is a footage day and every time shown is the footage's (the rows are in device
    time); it covers from the footage's first frame to where it has played. When it was generated stays real."""
    hour = int(policy["day_start_hour"])
    start, end = period_bounds(kind, first, hour)               # in the clock shown: the footage's, else the device's
    sh = clock.shift if clock else timedelta(0)
    seen = min(now + sh, clock.end) if clock else now           # how far the footage has played
    if start > seen:
        raise ValueError("this period has not started yet")
    until = seen if seen < end else None
    cover = max(start, clock.start) if clock else start
    shown = (lambda v: v) if clock is None else (lambda v: _iso(_ts(v) + sh) if v else None)
    d0, d1 = start - sh, end - sh                               # the rows' range (device time)
    st = compute_stats(conn, d0, d1, "hour" if kind == "day" else "day", day_start_hour=hour, live=live, now=now,
                       min_service_s=policy["min_service_s"], min_wait_s=policy["avg_wait_min_s"],
                       visits_source=int(policy["visits_source"]), serve_s=policy["wt_serve_s"], clock_shift=sh)
    staff = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM staff")}
    alerts = list_alerts(conn, d0, d1, limit=100000)
    excessive = {a["session_id"] for a in alerts if a["kind"] == "excessive_service" and a["session_id"]}
    rows = defaultdict(list)
    for s in sorted(sessions_started(conn, d0, d1), key=lambda r: r["started_at"]):
        b, e = _ts(s["started_at"]), _ts(s["ended_at"])
        # running: still open (start to now); interrupted: closed by a restart with no end or time (final review
        # C2: never "running" for days, never in a total); done: its measured time
        state = "running" if s["open"] else "done" if (s["service_s"] is not None or e) else "interrupted"
        dur = (s["service_s"] if s["service_s"] is not None else (e - b).total_seconds() if e
               else (now - b).total_seconds() if state == "running" else None)
        rows[chair_names.get(s["chair_id"], s["chair_id"])].append({
            "id": s["id"], "start": shown(s["started_at"]), "end": shown(s["ended_at"]), "state": state,
            "duration_s": None if dur is None else round(dur, 1),
            "barber": staff.get(s["barber_id"]), "excessive": s["id"] in excessive,
            "low_occupancy": bool(s["low_occupancy"]), "day": shop_day(b + sh, hour).isoformat()})
    chairs = []
    for name in sorted(rows):
        rs = rows[name]
        days = defaultdict(lambda: [0, 0.0])
        for r in rs:
            days[r["day"]][0] += 1
            days[r["day"]][1] += r["duration_s"] or 0.0
        timed = _timed(rs)
        total = round(sum(timed), 1)
        chairs.append({"chair": name, "haircuts": len(rs), "total_s": total,
                       "average_s": round(total / len(timed), 1) if timed else None,
                       "days": [{"day": d, "haircuts": n, "total_s": round(t, 1)} for d, (n, t) in sorted(days.items())],
                       "rows": rs})
    # 2026-11-10 (user): pedicure chairs -- their sessions per chair, never customers (spec pedicure-chairs)
    prows = defaultdict(list)
    for p in pedicures_in(conn, d0, d1):
        b, e = _ts(p["started_at"]), _ts(p["ended_at"])
        state = "running" if p["open"] else "done" if (p["service_s"] is not None or e) else "interrupted"
        dur = (p["service_s"] if p["service_s"] is not None else (e - b).total_seconds() if e
               else (now - b).total_seconds() if state == "running" else None)
        prows[chair_names.get(p["chair_id"], p["chair_id"])].append({
            "id": p["id"], "start": shown(p["started_at"]), "end": shown(p["ended_at"]), "state": state,
            "duration_s": None if dur is None else round(dur, 1)})
    pchairs = [{"chair": n, "rows": rs, "total_s": round(sum(_timed(rs)), 1)} for n, rs in sorted(prows.items())]
    pedicure = {"chairs": pchairs, "count": sum(len(c["rows"]) for c in pchairs),
                "total_s": round(sum(c["total_s"] for c in pchairs), 1)}
    # 2026-10-18 (user): one row per wait or haircut. A wait lost and found closes and reopens its alert (16 Oct: 34
    # alerts, 12 waits): first start to last end (open: running), its first picture. The summary counts these rows.
    groups: dict[tuple, list] = {}
    for a in sorted(alerts, key=lambda a: a["started_at"]):
        groups.setdefault((a["kind"], a["session_id"] or a["visit_id"] or a["id"]), []).append(a)
    arows = []
    for (kd, _), g in groups.items():
        b, ends = _ts(g[0]["started_at"]), [_ts(a["ended_at"]) for a in g]
        e = None if None in ends else max(ends)
        pic = next((a for a in g if a["has_snapshot"]), None)
        one = pic or g[0]
        arows.append({"id": one["id"], "kind": kd, "chair": chair_names.get(one["chair_id"], one["chair_id"]),
                      "barber": staff.get(one["barber_id"]), "start": shown(g[0]["started_at"]), "end": shown(_iso(e)),
                      "duration_s": round(((e or now) - b).total_seconds(), 1), "has_snapshot": pic is not None})
    with_snap = [a for a in arows if a["has_snapshot"]]
    shown = with_snap if kind == "day" else sorted(with_snap, key=lambda a: -a["duration_s"])[:SNAPSHOTS_LONG]
    lw, lh = st.get("longest_wait"), st.get("longest_haircut")
    figures = {k: st.get(k) for k in ("served_customers", "visitors", "walk_outs", "completed_haircuts",
                                       "average_wait_s", "average_haircut_s", "excessive_wait", "excessive_haircut",
                                       "returning_visits", "avg_days_between_visits")}
    # the excessive haircuts are the rows above, of the haircuts served (running ones too, like Served customers); the
    # excessive waits already are on a door-free site (stats: the alerted waits, once each)
    figures["excessive_haircut"] = _share(sum(a["kind"] == "excessive_service" for a in arows), st["served_customers"])
    # 2026-11-14 (user): haircuts whose sitter was on the seat in few of the frames (ce_low_occupancy) -- possibly not
    # customers: kept in every list and total, counted apart beside Served customers, highlighted in the haircut lists
    figures["low_occupancy"] = sorted(({"chair": c["chair"], "start": r["start"]} for c in chairs for r in c["rows"]
                                       if r["low_occupancy"]), key=lambda x: x["start"])
    figures["longest_wait_s"] = lw["seconds"] if lw else None
    figures["longest_haircut_s"] = lh["seconds"] if lh else None
    figures["longest_wait_running"] = bool(lw and lw.get("running"))          # still in progress (time so far)
    figures["longest_haircut_running"] = bool(lh and lh.get("running"))
    return {
        "kind": kind, "first_day": start.date().isoformat(), "last_day": (end - timedelta(days=1)).date().isoformat(),
        "start": _iso(cover), "end": _iso(end), "until": _iso(until), "day_start_hour": hour,
        "generated_at": _iso(now), "footage": clock is not None, "site": dict(site), "figures": figures,
        "series": [{"bucket": p["bucket"], "haircuts": p["served_customers"]} for p in st.get("series", [])],
        "barbers": [{"name": b["name"], "haircuts": b["haircuts"], "average_haircut_s": b["average_haircut_s"]}
                    for b in st["haircuts_per_barber"]],
        "chairs": chairs,
        "pedicure": pedicure,
        "alerts": {"counts": {"excessive_wait": sum(a["kind"] == "excessive_wait" for a in arows),
                              "excessive_service": sum(a["kind"] == "excessive_service" for a in arows)},
                   "rows": arows, "shown": [a["id"] for a in shown]},
    }
