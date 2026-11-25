from datetime import date, datetime, timedelta

import pytest

from domain import COMPLETED, EXIT_DOOR, FINISHED, HAIRCUT, Alert, EngineUpdate, ServiceSession, Visit
from policy import DEFAULT_POLICY
from report_data import build_report, file_name, period_bounds
from storage import apply_update, connect, create_staff

P = {**DEFAULT_POLICY, "day_start_hour": 8}
SITE = {"place": "BekkyBarber", "branch": "Palm Hills", "branch_slug": "palmhills"}


def test_period_bounds_follow_the_shop_day():
    assert period_bounds("day", date(2026, 10, 11), 8) == (datetime(2026, 10, 11, 8), datetime(2026, 10, 12, 8))
    assert period_bounds("week", date(2026, 10, 11), 8)[1] == datetime(2026, 10, 18, 8)
    assert period_bounds("month", date(2026, 2, 17), 8) == (datetime(2026, 2, 1, 8), datetime(2026, 3, 1, 8))
    assert period_bounds("month", date(2026, 10, 5), 0)[1] == datetime(2026, 11, 1, 0)
    with pytest.raises(ValueError):
        period_bounds("year", date(2026, 10, 1), 8)


def test_file_names_as_the_task_gives_them():
    d = period_bounds("day", date(2026, 10, 11), 8)
    w = period_bounds("week", date(2026, 10, 11), 8)
    m = period_bounds("month", date(2026, 10, 11), 8)
    assert file_name("day", *d, "BekkyBarber", "palmhills", "pdf") == "11oct_BekkyBarber_palmhills_analysis_report.pdf"
    assert file_name("week", *w, "BekkyBarber", "palmhills", "docx") == "11oct_17oct_BekkyBarber_palmhills_analysis_report.docx"
    assert file_name("month", *m, "BekkyBarber", "palmhills", "html") == "01oct_31oct_BekkyBarber_palmhills_analysis_report.html"


def _shop(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    lisa = create_staff(conn, "lisa")["id"]
    T = datetime(2026, 10, 11, 12, 0)
    vs, ss = {}, {}
    for i, (chair, start_min, dur_min) in enumerate((("ph2-L7", 0, 30), ("L8", 10, 50), ("ph2-L7", 60, 20))):
        st = T + timedelta(minutes=start_min)
        vs[f"v{i}"] = Visit(f"v{i}", f"p{i}", st, False, state=FINISHED, confirmed=True, service_started_at=st,
                            haircuts=1, exited_at=st + timedelta(minutes=dur_min), exit_reason=EXIT_DOOR,
                            outcome=COMPLETED, open=False)
        ss[f"s{i}"] = ServiceSession(f"s{i}", f"v{i}", chair, st, ended_at=st + timedelta(minutes=dur_min),
                                     service_s=dur_min * 60.0, barber_id=lisa, open=False)
    run = T + timedelta(hours=3)                                   # still running at "now"
    vs["v9"] = Visit("v9", "p9", run, False, state=HAIRCUT, confirmed=True, service_started_at=run)
    ss["s9"] = ServiceSession("s9", "v9", "L8", run, barber_id=None)
    alerts = [Alert("a1", "excessive_service", "shop", "v1", "s1", T + timedelta(minutes=40),
                    ended_at=T + timedelta(minutes=60), open=False)]
    apply_update(conn, EngineUpdate(visits=vs, sessions=ss, alerts=alerts), T + timedelta(hours=3))
    return conn, T


def test_the_model_lists_every_haircut_per_chair_by_name(tmp_path):
    conn, T = _shop(tmp_path)
    now = T + timedelta(hours=3, minutes=10)
    m = build_report(conn, "day", date(2026, 10, 11), now, P, SITE, {"ph2-L7": "L7", "L8": "L8"})
    chairs = {c["chair"]: c for c in m["chairs"]}
    assert set(chairs) == {"L7", "L8"}
    assert chairs["L7"]["haircuts"] == 2 and chairs["L7"]["total_s"] == 3000.0
    run = [r for r in chairs["L8"]["rows"] if r["end"] is None]
    assert len(run) == 1 and run[0]["duration_s"] == 600.0       # start to now
    assert [r["excessive"] for r in chairs["L8"]["rows"]] == [True, False]
    assert m["until"] == now.isoformat(timespec="seconds")       # the day is still running
    assert m["alerts"]["counts"] == {"excessive_wait": 0, "excessive_service": 1}
    assert m["figures"]["served_customers"] >= 3


def test_a_wait_or_haircut_with_several_alerts_is_one_row_and_its_figures_count_the_rows(tmp_path):
    """2026-10-18 (user): 16 Oct read "0 excessive waits" above 34 wait snapshots -- 12 waits, each closing and
    reopening its alert as the customer was lost and found. One row (first start, last end) and one picture each;
    the summary counts those rows."""
    conn, T = _shop(tmp_path)
    w = lambda i, a, b: Alert(f"w{i}", "excessive_wait", "shop", "wait1", None, T + timedelta(minutes=a),
                              ended_at=T + timedelta(minutes=b), open=False)
    apply_update(conn, EngineUpdate(alerts=[w(1, 5, 8), w(2, 10, 12), w(3, 14, 20)]), T + timedelta(hours=3))
    conn.execute("UPDATE barber_alert SET snapshot_path = 'x.jpg' WHERE id IN ('w2', 'w3', 'a1')")
    conn.commit()
    m = build_report(conn, "day", date(2026, 10, 11), T + timedelta(hours=3, minutes=10), P, SITE, {})
    (row,) = [r for r in m["alerts"]["rows"] if r["kind"] == "excessive_wait"]
    assert (row["id"], row["start"][11:16], row["end"][11:16], row["duration_s"], row["has_snapshot"]) == \
        ("w2", "12:05", "12:20", 900.0, True)                    # its first picture; first start to last end
    assert m["alerts"]["counts"] == {"excessive_wait": 1, "excessive_service": 1}
    assert m["alerts"]["shown"] == ["w2", "a1"]
    assert m["figures"]["excessive_haircut"] == {"count": 1, "percent": 25.0}   # 1 of the 4 served (one running)


def test_a_future_period_is_refused(tmp_path):
    conn, T = _shop(tmp_path)
    with pytest.raises(ValueError):
        build_report(conn, "day", date(2026, 10, 20), T, P, SITE, {})


def test_week_snapshots_are_the_12_longest(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    T = datetime(2026, 10, 11, 12, 0)
    alerts = [Alert(f"a{i}", "excessive_wait", "shop", f"w{i}", None, T + timedelta(minutes=i),
                    ended_at=T + timedelta(minutes=i, seconds=10 * i), open=False) for i in range(1, 16)]
    apply_update(conn, EngineUpdate(alerts=alerts), T)
    conn.execute("UPDATE barber_alert SET snapshot_path = 'x.jpg'")
    conn.commit()
    m = build_report(conn, "week", date(2026, 10, 11), T + timedelta(days=8), P, SITE, {})
    assert m["alerts"]["shown"] == [f"a{i}" for i in range(15, 3, -1)]


def test_a_haircut_cut_off_by_a_restart_is_not_running_and_not_timed(tmp_path):
    conn, T = _shop(tmp_path)
    conn.execute("UPDATE service_session SET open = 0 WHERE id = 's9'")      # close_interrupted: no end, no time
    conn.commit()
    m = build_report(conn, "week", date(2026, 10, 11), T + timedelta(days=3), P, SITE, {"ph2-L7": "L7", "L8": "L8"})
    l8 = next(c for c in m["chairs"] if c["chair"] == "L8")
    cut = next(r for r in l8["rows"] if r["end"] is None)
    assert cut["state"] == "interrupted" and cut["duration_s"] is None
    assert l8["total_s"] == 3000.0 and l8["average_s"] == 3000.0              # the 50-min haircut only


def test_figures_are_the_dashboards_with_the_live_customers(tmp_path, monkeypatch):
    import report_data
    conn, T = _shop(tmp_path)
    seen = {}
    real = report_data.compute_stats

    def spy(conn_, start, end, bucket=None, **kw):
        seen.update(end=end, live=kw.get("live"))
        return real(conn_, start, end, bucket, **kw)
    monkeypatch.setattr(report_data, "compute_stats", spy)
    live = [{"state": "waiting"}]
    build_report(conn, "day", date(2026, 10, 11), T + timedelta(hours=3), P, SITE, {}, live=live)
    assert seen["end"] == datetime(2026, 10, 12, 8) and seen["live"] is live


def test_the_model_marks_a_running_longest(tmp_path):
    from domain import HAIRCUT
    conn, T = _shop(tmp_path)
    now = T + timedelta(hours=4)
    live = [{"state": HAIRCUT, "haircut_started_at": (T + timedelta(hours=3)).isoformat(), "session_id": "s9",
             "visit_id": "v9", "chair_id": "L8", "barber_id": None}]
    m = build_report(conn, "day", date(2026, 10, 11), now, P, SITE, {}, live=live)
    assert m["figures"]["longest_haircut_s"] == 3600.0 and m["figures"]["longest_haircut_running"] is True


def test_a_run_on_recordings_reports_the_footage_date_and_times(tmp_path):
    """2026-10-18 (user): a run playing 16 Oct 09:30-19:00 is reported as that day and those times, not the device's."""
    from footage_clock import FootageClock
    conn, T = _shop(tmp_path)                               # haircuts from device 11 Oct 12:00
    foot = datetime(2026, 10, 16, 9, 30)                    # = device T
    clock = FootageClock(foot, foot - T, 34200.0)
    now = T + timedelta(hours=3, minutes=10)                # device: footage 12:40
    m = build_report(conn, "day", date(2026, 10, 16), now, P, SITE, {"ph2-L7": "L7", "L8": "L8"}, clock=clock)
    assert (m["first_day"], m["start"], m["until"], m["footage"]) == \
        ("2026-10-16", "2026-10-16T09:30:00", "2026-10-16T12:40:00", True)     # from the footage's first frame
    l7 = next(c for c in m["chairs"] if c["chair"] == "L7")
    assert [r["start"][11:16] for r in l7["rows"]] == ["09:30", "10:30"] and {r["day"] for r in l7["rows"]} == {"2026-10-16"}
    assert [(a["start"][11:16], a["end"][11:16]) for a in m["alerts"]["rows"]] == [("10:10", "10:30")]
    assert {p["bucket"][11:16]: p["haircuts"] for p in m["series"] if p["haircuts"]} == {"09:00": 2, "10:00": 1, "12:00": 1}
    assert m["generated_at"] == now.isoformat(timespec="seconds")              # when it was made stays real
    with pytest.raises(ValueError):                                             # the footage's next day: not yet
        build_report(conn, "day", date(2026, 10, 17), now, P, SITE, {}, clock=clock)


# ── pedicure chairs (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md) ──────────────────────────────
def test_pedicures_get_their_own_block_and_never_count_as_customers(tmp_path):
    from domain import PedicureRecord
    conn = connect(tmp_path / "b.db")
    t = datetime(2026, 10, 16, 14, 16)
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "ph6-P1", t, t + timedelta(minutes=41), 2460.0,
                                                                     False)}), t)
    m = build_report(conn, "day", date(2026, 10, 16), t + timedelta(hours=2), P, SITE, {"ph6-P1": "P1"})
    assert m["pedicure"]["count"] == 1 and m["pedicure"]["total_s"] == 2460.0
    (c,) = m["pedicure"]["chairs"]
    assert c["chair"] == "P1" and c["rows"][0]["state"] == "done" and c["rows"][0]["duration_s"] == 2460.0
    assert m["figures"]["served_customers"] == 0 and m["chairs"] == []


def test_a_period_without_pedicures_has_an_empty_block(tmp_path):
    m = build_report(connect(tmp_path / "b.db"), "day", date(2026, 10, 16), datetime(2026, 10, 16, 20), P, SITE, {})
    assert m["pedicure"] == {"chairs": [], "count": 0, "total_s": 0.0}


def test_low_occupancy_haircuts_are_listed_beside_the_figures_and_marked_on_their_rows(tmp_path):
    conn, T = _shop(tmp_path)                                      # 2026-11-14 (user): kept, counted apart
    st = T + timedelta(minutes=90)
    v = Visit("v5", "p5", st, False, state=FINISHED, confirmed=True, service_started_at=st, haircuts=1,
              exited_at=st + timedelta(minutes=17), outcome=COMPLETED, open=False)
    s = ServiceSession("s5", "v5", "L8", st, ended_at=st + timedelta(minutes=17), service_s=1020.0, open=False,
                       occupancy=0.2, low_occupancy=True)
    apply_update(conn, EngineUpdate(visits={"v5": v}, sessions={"s5": s}), T + timedelta(hours=3))
    m = build_report(conn, "day", date(2026, 10, 11), T + timedelta(hours=3, minutes=10), P, SITE, {"L8": "L8"})
    assert m["figures"]["low_occupancy"] == [{"chair": "L8", "start": "2026-10-11T13:30:00"}]
    rows = {c["chair"]: c["rows"] for c in m["chairs"]}
    assert [r["low_occupancy"] for r in rows["L8"]] == [False, True, False]
    assert m["figures"]["served_customers"] == 5                   # still counted: the report shows 4 (+1)
