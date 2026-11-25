from datetime import datetime, timedelta

from domain import (COMPLETED, EXIT_DOOR, EXIT_LOST, FINISHED, HAIRCUT, LOST, VOID_STAFF, WAITING, EngineUpdate,
                    ServiceSession, Visit, WaitRecord)
from stats import compute_stats
from storage import apply_update, connect, create_staff

T = datetime(2026, 9, 24, 10, 0, 0)


def t(s):
    return T + timedelta(seconds=s)


def test_stats_count_haircuts_per_barber_and_lost_visits(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    lisa = create_staff(conn, "lisa")["id"]
    v1 = Visit("v1", "p1", t(0), True, state=FINISHED, confirmed=True, waiting_started_at=t(0),
               service_started_at=t(60), wait_s=60, haircuts=1, exited_at=t(700), exit_reason=EXIT_DOOR,
               outcome=COMPLETED, open=False)
    s1 = ServiceSession("s1", "v1", "ch1", t(60), ended_at=t(660), service_s=600, barber_id=lisa, open=False)
    v2 = Visit("v2", "p2", t(10), False, state=FINISHED, confirmed=True, waiting_started_at=t(10), wait_s=50,
               exited_at=t(60), exit_reason=EXIT_LOST, outcome=LOST, open=False)
    apply_update(conn, EngineUpdate(visits={"v1": v1, "v2": v2}, sessions={"s1": s1}), t(700))
    st = compute_stats(conn, t(-60), t(3600))
    assert st["customer_visits"] == 2 and st["completed_haircuts"] == 1 and st["lost"] == 1
    assert st["haircuts_per_barber"] == [{"barber_id": lisa, "name": "lisa", "haircuts": 1,
                                          "average_haircut_s": 600.0}]


def test_returning_visits_and_days_between(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    for vid, day in (("v1", 0), ("v2", 20), ("v3", 30)):
        v = Visit(vid, "p", t(day * 86400), True, state=FINISHED, confirmed=True, waiting_started_at=t(day * 86400),
                  wait_s=0, haircuts=1, exited_at=t(day * 86400 + 600), exit_reason=EXIT_DOOR, outcome=COMPLETED,
                  open=False)
        apply_update(conn, EngineUpdate(visits={vid: v}), t(day * 86400))
    conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, face_count) VALUES ('c1', ?, ?, 2, 1)",
                 (t(0).isoformat(), t(20 * 86400).isoformat()))
    conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, face_count) VALUES ('c2', ?, ?, 1, 1)",
                 (t(30 * 86400).isoformat(), t(30 * 86400).isoformat()))
    conn.execute("UPDATE visit SET customer_id = 'c1' WHERE id IN ('v1', 'v2')")
    conn.execute("UPDATE visit SET customer_id = 'c2' WHERE id = 'v3'")
    conn.commit()
    st = compute_stats(conn, t(-60), t(40 * 86400))
    assert st["returning_visits"] == {"count": 1, "percent": 33.3} and st["avg_days_between_visits"] == 20.0


def test_series_items_carry_returning_visits(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    for vid, offset in (("v1", 0), ("v2", 3600)):
        v = Visit(vid, "p", t(offset), True, state=FINISHED, confirmed=True, waiting_started_at=t(offset),
                  wait_s=0, haircuts=1, exited_at=t(offset + 600), exit_reason=EXIT_DOOR, outcome=COMPLETED,
                  open=False)
        apply_update(conn, EngineUpdate(visits={vid: v}), t(offset))
    conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, face_count) VALUES ('c1', ?, ?, 2, 1)",
                 (t(0).isoformat(), t(3600).isoformat()))
    conn.execute("UPDATE visit SET customer_id = 'c1' WHERE id IN ('v1', 'v2')")
    conn.commit()
    st = compute_stats(conn, t(-3600), t(7200), bucket="day")
    for item in st["series"]:
        assert set(item["returning_visits"]) == {"count", "percent"}
        assert isinstance(item["returning_visits"]["count"], int)
        assert item["returning_visits"]["percent"] is None or isinstance(item["returning_visits"]["percent"], float)
    todays = [item for item in st["series"] if item["bucket"].startswith("2026-09-24")]
    assert len(todays) == 1 and todays[0]["returning_visits"] == {"count": 1, "percent": 50.0}


def _late_visit(conn):
    at = datetime(2026, 9, 25, 1, 0)  # after midnight: still the 24th's shop day
    v = Visit("late", "p", at, True, state=FINISHED, confirmed=True, waiting_started_at=at, wait_s=0, haircuts=1,
              exited_at=at + timedelta(minutes=10), exit_reason=EXIT_DOOR, outcome=COMPLETED, open=False)
    apply_update(conn, EngineUpdate(visits={"late": v}), at)


def test_day_buckets_follow_the_shop_day(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    _late_visit(conn)
    st = compute_stats(conn, datetime(2026, 9, 24, 5), datetime(2026, 9, 26, 5), bucket="day", day_start_hour=5)
    assert [b["bucket"] for b in st["series"]] == ["2026-09-24T05:00:00", "2026-09-25T05:00:00"]
    assert [b["customer_visits"] for b in st["series"]] == [1, 0]


def test_week_and_month_buckets_start_at_the_shop_hour(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    _late_visit(conn)
    wk = compute_stats(conn, datetime(2026, 9, 24, 5), datetime(2026, 9, 26, 5), bucket="week", day_start_hour=5)
    assert [b["bucket"] for b in wk["series"]] == ["2026-09-21T05:00:00"]
    mo = compute_stats(conn, datetime(2026, 9, 1, 5), datetime(2026, 10, 1, 5), bucket="month", day_start_hour=5)
    assert [b["bucket"] for b in mo["series"]] == ["2026-09-01T05:00:00"]
    hr = compute_stats(conn, datetime(2026, 9, 25, 0, 30), datetime(2026, 9, 25, 2), bucket="hour", day_start_hour=5)
    assert [b["bucket"] for b in hr["series"]] == ["2026-09-25T00:00:00", "2026-09-25T01:00:00"]


def _served(conn, vid, wait):
    v = Visit(vid, "p" + vid, t(0), True, state=FINISHED, confirmed=True, waiting_started_at=t(0),
              service_started_at=t(wait), wait_s=wait, haircuts=1, exited_at=t(wait + 900), exit_reason=EXIT_DOOR,
              outcome=COMPLETED, open=False)
    apply_update(conn, EngineUpdate(visits={vid: v}), t(wait + 900))


def test_a_visit_voided_as_staff_counts_its_wait_in_the_average(tmp_path):
    # 2026-09-28 (user): a voided "customer" (a barber) still counts his wait, until that's fixed properly
    conn = connect(str(tmp_path / "b.db"))
    _served(conn, "v1", 100)
    v = Visit("v2", "p2", t(0), False, state=FINISHED, confirmed=True, waiting_started_at=t(0),
              exited_at=t(500), outcome=VOID_STAFF, open=False)
    apply_update(conn, EngineUpdate(visits={"v2": v}), t(500))
    st = compute_stats(conn, t(-60), t(3600))
    assert st["average_wait_s"] == 300.0 and st["customer_visits"] == 1


def test_live_waits_and_running_haircuts_count_in_the_averages(tmp_path):
    # 2026-09-28 (user): don't wait for a wait or a haircut to end to count it
    conn = connect(str(tmp_path / "b.db"))
    _served(conn, "v1", 100)
    s1 = ServiceSession("s1", "v1", "ch1", t(100), ended_at=t(700), service_s=600, open=False)
    apply_update(conn, EngineUpdate(sessions={"s1": s1}), t(700))
    now = t(2000)
    live = [{"state": WAITING, "waiting_started_at": t(1500).isoformat(), "haircut_started_at": None, "barber_id": None},
            {"state": HAIRCUT, "waiting_started_at": t(0).isoformat(), "haircut_started_at": t(1000).isoformat(),
             "barber_id": None},
            {"state": HAIRCUT, "waiting_started_at": t(0).isoformat(), "haircut_started_at": t(1950).isoformat(),
             "barber_id": None}]                                   # 50 s in: not a haircut yet (min_service_s)
    st = compute_stats(conn, t(-60), t(3600), live=live, now=now, min_service_s=180)
    assert st["average_wait_s"] == 300.0                            # (100 + 500) / 2
    assert st["average_haircut_s"] == 800.0                         # (600 + 1000) / 2
    assert st["completed_haircuts"] == 1
    later = compute_stats(conn, t(-60), t(1800), live=live, now=now, min_service_s=180)
    assert later["average_wait_s"] == 100.0                         # now is outside the period: no live values


def test_a_customer_first_seen_in_the_chair_is_left_out_of_the_wait_figures(tmp_path):
    # 2026-09-28 (user): his wait was never seen -- not a 0-second wait
    conn = connect(str(tmp_path / "b.db"))
    _served(conn, "v1", 300)
    v = Visit("v2", "p2", t(0), False, state=FINISHED, confirmed=True, waiting_started_at=t(0),
              service_started_at=t(0), wait_s=0, haircuts=1, exited_at=t(900), exit_reason=EXIT_DOOR,
              outcome=COMPLETED, open=False, wait_unseen=True)
    apply_update(conn, EngineUpdate(visits={"v2": v}), t(900))
    st = compute_stats(conn, t(-60), t(3600))
    assert st["average_wait_s"] == 300.0 and st["excessive_wait"]["count"] == 0
    assert st["excessive_wait"]["percent"] == 0.0 and st["customer_visits"] == 2


def test_older_chair_created_visits_are_marked_from_the_binding_log(tmp_path):
    from domain import Binding
    from storage import _migrate
    conn = connect(str(tmp_path / "b.db"))
    _served(conn, "v1", 300)
    _served(conn, "v2", 0)
    apply_update(conn, EngineUpdate(bindings=[Binding("door", "barber-c", 1, "pv1", None, None, None, None, None, t(0)),
                                              Binding("seat_start", "barber-l", 7, "pv2", "ch1", None, None, None, None, t(0)),
                                              Binding("seat_start", "barber-l", 8, "pv1", "ch1", None, None, None, None,
                                                      t(400))]), t(900))
    conn.execute("ALTER TABLE visit DROP COLUMN wait_unseen")    # a database from before the column
    _migrate(conn)
    marked = {r["id"]: r["wait_unseen"] for r in conn.execute("SELECT id, wait_unseen FROM visit")}
    assert marked == {"v1": 0, "v2": 1}          # v1's first binding was the door


def test_visitors_are_non_staff_in_crossings(tmp_path):
    from domain import DoorCrossing
    conn = connect(str(tmp_path / "b.db"))
    apply_update(conn, EngineUpdate(door_crossings=[
        DoorCrossing(t(10), "in", "barber-c", 1, False), DoorCrossing(t(20), "in", "barber-c", 2, False),
        DoorCrossing(t(30), "in", "barber-c", 3, True), DoorCrossing(t(40), "out", "barber-c", 1, False)]), t(50))
    assert compute_stats(conn, t(-60), t(3600))["visitors"] == 2


def test_chair_waits_finished_and_open_count_in_the_average(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    done = WaitRecord("w1", t(0), t(120), 120.0, "lost", 900.0, "barber-c", 7, 7, 1, t(120))
    still = WaitRecord("w2", t(100), None, None, None, 900.0, "barber-c", 8, 8, 2, t(150))
    apply_update(conn, EngineUpdate(waits={"w1": done, "w2": still}), t(150))
    live = [{"visit_id": "w2", "person_id": "w2", "state": WAITING, "waiting_started_at": t(100).isoformat(),
             "barber_id": None, "chair_id": None, "haircut_started_at": None, "session_id": None}]
    st = compute_stats(conn, t(-60), t(3600), live=live, now=t(160))
    assert st["average_wait_s"] == 90.0                  # (120 + 60) / 2
    # 2026-10-15 (user): the longest counts the still-waiting customer too (was: None until a wait reached a chair)
    assert st["longest_wait"]["seconds"] == 60.0 and st["longest_wait"]["running"] is True
    assert st["excessive_wait"]["count"] == 0
    st = compute_stats(conn, t(-60), t(3600), bucket="hour")
    assert sum(1 for b in st["series"] if b["average_wait_s"] == 120.0) == 1


def test_waits_shorter_than_min_wait_s_stay_out_of_the_average(tmp_path):
    # 2026-10-12 (user): very short waits pulled the average down
    conn = connect(str(tmp_path / "b.db"))
    waits = {"w1": WaitRecord("w1", t(0), t(120), 120.0, "chair", 900.0, "barber-c", 7, 7, 1, t(120)),
             "w2": WaitRecord("w2", t(200), t(212), 12.0, "lost", 900.0, "barber-c", 8, 8, 2, t(212))}
    apply_update(conn, EngineUpdate(waits=waits), t(300))
    live = [{"visit_id": "w3", "person_id": "w3", "state": WAITING, "waiting_started_at": t(290).isoformat(),
             "barber_id": None, "chair_id": None, "haircut_started_at": None, "session_id": None}]
    assert compute_stats(conn, t(-60), t(3600), live=live, now=t(300))["average_wait_s"] == round(142 / 3, 1)
    st = compute_stats(conn, t(-60), t(3600), live=live, now=t(300), min_wait_s=30.0)
    assert st["average_wait_s"] == 120.0                 # the 12 s wait and the 10 s so far don't count
    st = compute_stats(conn, t(-60), t(3600), bucket="hour", min_wait_s=30.0)
    assert [b["average_wait_s"] for b in st["series"] if b["average_wait_s"] is not None] == [120.0]


def _session(sid, start_s, end_s=None, barber=None):
    return ServiceSession(sid, sid, "R1", t(start_s), ended_at=None if end_s is None else t(end_s),
                          service_s=None if end_s is None else end_s - start_s, barber_id=barber,
                          open=end_s is None)


def test_served_customers_count_confirmed_haircuts_by_start_running_included(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    visits = {k: Visit(k, "p" + k, t(a), True, confirmed=True, open=False) for k, a in (("s1", 0), ("s2", 3700))}
    apply_update(conn, EngineUpdate(visits=visits, sessions={"s1": _session("s1", 0, 600),
                                                              "s2": _session("s2", 3700)}), t(3800))
    st = compute_stats(conn, t(-60), t(7200), bucket="hour")
    assert st["served_customers"] == 2 and st["completed_haircuts"] == 1
    assert [b["served_customers"] for b in st["series"] if b["served_customers"]] == [1, 1]
    assert compute_stats(conn, t(3600), t(7200))["served_customers"] == 1


def test_visitors_per_bucket_are_non_staff_door_ins(tmp_path):
    from domain import DoorCrossing
    conn = connect(str(tmp_path / "b.db"))
    apply_update(conn, EngineUpdate(door_crossings=[
        DoorCrossing(t(10), "in", "barber-c", 1, False), DoorCrossing(t(20), "in", "barber-c", 2, True),
        DoorCrossing(t(30), "out", "barber-c", 1, False), DoorCrossing(t(3700), "in", "barber-c", 3, False)]), t(3800))
    st = compute_stats(conn, t(-60), t(7200), bucket="hour")
    assert st["visitors"] == 2 and [b["visitors"] for b in st["series"] if b["visitors"]] == [1, 1]


def test_haircuts_per_barber_count_served_customers_running_included(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    lisa = create_staff(conn, "lisa")["id"]
    visits = {k: Visit(k, "p" + k, t(a), True, confirmed=True, open=False) for k, a in (("s1", 0), ("s2", 900), ("s3", 1000))}
    apply_update(conn, EngineUpdate(visits=visits, sessions={"s1": _session("s1", 0, 600, lisa),
                                                              "s2": _session("s2", 900, None, lisa),
                                                              "s3": _session("s3", 1000)}), t(1100))
    rows = compute_stats(conn, t(-60), t(3600))["haircuts_per_barber"]
    assert rows == [{"barber_id": lisa, "name": "lisa", "haircuts": 2, "average_haircut_s": 600.0},
                    {"barber_id": None, "name": None, "haircuts": 1, "average_haircut_s": None}]


def test_visits_are_never_fewer_than_served_customers(tmp_path):
    from domain import DoorCrossing
    conn = connect(str(tmp_path / "b.db"))
    visits = {k: Visit(k, "p" + k, t(a), True, confirmed=True, open=False) for k, a in (("s1", 0), ("s2", 100), ("s3", 3700))}
    apply_update(conn, EngineUpdate(visits=visits, sessions={k: _session(k, a, a + 600) for k, a in (("s1", 0), ("s2", 100), ("s3", 3700))},
                                    door_crossings=[DoorCrossing(t(10), "in", "barber-c", 1, False),
                                                    DoorCrossing(t(3650), "in", "barber-c", 2, False),
                                                    DoorCrossing(t(3660), "in", "barber-c", 3, False)]), t(5000))
    st = compute_stats(conn, t(-60), t(7200), bucket="hour")
    assert st["served_customers"] == 3 and st["visitors"] == 3          # max(3 door ins, 3 served)
    by = [(b["served_customers"], b["visitors"]) for b in st["series"] if b["served_customers"] or b["visitors"]]
    assert by == [(2, 2), (1, 2)]                                       # first hour: 1 door in, 2 served -> 2


def test_visitors_start_from_the_baseline_then_add_door_ins(tmp_path):
    from domain import DoorCrossing
    from storage import set_visit_baseline
    conn = connect(str(tmp_path / "b.db"))
    set_visit_baseline(conn, "2026-09-24", t(130), haircuts=4, waiting=3, washing=1)
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(t(200 + i), "in", "barber-c", i, False)
                                                    for i in range(3)]), t(400))
    assert compute_stats(conn, t(-60), t(7200))["visitors"] == 11


def test_the_served_floor_still_applies_over_a_baseline(tmp_path):
    from storage import set_visit_baseline
    conn = connect(str(tmp_path / "b.db"))
    set_visit_baseline(conn, "2026-09-24", t(130), 1, 0, 0)
    visits = {k: Visit(k, "p" + k, t(a), True, confirmed=True, open=False) for k, a in (("s1", 0), ("s2", 900), ("s3", 1000))}
    apply_update(conn, EngineUpdate(visits=visits, sessions={k: _session(k, a) for k, a in (("s1", 0), ("s2", 900), ("s3", 1000))}), t(1100))
    assert compute_stats(conn, t(-60), t(7200))["visitors"] == 3


def test_baselines_are_bucketed_by_when_they_were_taken(tmp_path):
    from storage import set_visit_baseline
    conn = connect(str(tmp_path / "b.db"))
    set_visit_baseline(conn, "2026-09-24", t(130), 2, 1, 0)
    set_visit_baseline(conn, "2026-09-25", t(3700), 1, 1, 0)
    st = compute_stats(conn, t(-60), t(7200), bucket="hour")
    assert st["visitors"] == 5 and [b["visitors"] for b in st["series"] if b["visitors"]] == [3, 2]


def wait_rec(wid, start_s, end_s):
    return WaitRecord(wid, t(start_s), t(end_s) if end_s is not None else None,
                      (end_s - start_s) if end_s is not None else None, "left" if end_s is not None else None,
                      900.0, "ph-3", 1, 1, 1, t(end_s if end_s is not None else start_s))


def test_door_free_visits_are_served_plus_unserved_waits(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    s1 = ServiceSession(id="s1", visit_id="s1", chair_id="R1", started_at=t(400), barber_id=None)
    s2 = ServiceSession(id="s2", visit_id="s2", chair_id="R2", started_at=t(900), barber_id=None)   # no wait before it
    waits = {"w1": wait_rec("w1", 0, 300),       # served by s1 (100 s later)
             "w2": wait_rec("w2", 10, 320),      # a companion: no haircut within 150 s left for it
             "w3": wait_rec("w3", 1000, None)}   # still waiting
    visits = {k: Visit(k, "p" + k, t(a), True, confirmed=True, open=False) for k, a in (("s1", 400), ("s2", 900))}
    apply_update(conn, EngineUpdate(visits=visits, sessions={"s1": s1, "s2": s2}, waits=waits), t(1100))
    st = compute_stats(conn, t(-60), t(3600), visits_source=1, serve_s=150.0)
    assert st["served_customers"] == 2 and st["visitors"] == 4      # s1, s2 + w2 + w3
    assert compute_stats(conn, t(-60), t(3600))["visitors"] == 2     # visits_source 0: no door ins -> served


def test_door_free_a_wait_a_stall_ended_is_never_served(tmp_path):
    # final review I2: the wait camera's stall ends every open wait; a haircut starting meanwhile is someone else's
    conn = connect(str(tmp_path / "b.db"))
    s1 = ServiceSession(id="s1", visit_id="s1", chair_id="R1", started_at=t(400), barber_id=None)
    stalled = WaitRecord("w1", t(0), t(300), 300.0, "stall", 900.0, "ph-3", 1, 1, 1, t(300))
    visits = {"s1": Visit("s1", "ps1", t(400), True, confirmed=True, open=False)}
    apply_update(conn, EngineUpdate(visits=visits, sessions={"s1": s1}, waits={"w1": stalled}), t(500))
    assert compute_stats(conn, t(-60), t(3600), visits_source=1, serve_s=150.0)["visitors"] == 2   # s1 + w1


def test_door_free_visits_per_bucket(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    s1 = ServiceSession(id="s1", visit_id="s1", chair_id="R1", started_at=t(3610), barber_id=None)
    visits = {"s1": Visit("s1", "ps1", t(3610), True, confirmed=True, open=False)}
    apply_update(conn, EngineUpdate(visits=visits, sessions={"s1": s1}, waits={"w1": wait_rec("w1", 3400, 3590)}), t(3700))
    st = compute_stats(conn, t(0), t(7200), bucket="hour", visits_source=1, serve_s=150.0)
    assert [b["visitors"] for b in st["series"]] == [0, 1]           # the wait is served by the next hour's haircut


def test_the_longest_counts_what_is_still_running_like_the_average(tmp_path):
    """2026-10-15 (user): a report read 'longest haircut 09:53, average 15:14' -- the average counts running haircuts
    (time so far, 2026-09-28) but the longest did not. Now the longest does too, marked running."""
    from domain import HAIRCUT
    conn = connect(str(tmp_path / "b.db"))
    v1 = Visit("v1", "p1", t(0), True, state=FINISHED, confirmed=True, waiting_started_at=t(0),
               service_started_at=t(60), wait_s=60, haircuts=1, exited_at=t(700), exit_reason=EXIT_DOOR,
               outcome=COMPLETED, open=False)
    s1 = ServiceSession("s1", "v1", "ch1", t(60), ended_at=t(660), service_s=600, open=False)
    apply_update(conn, EngineUpdate(visits={"v1": v1}, sessions={"s1": s1}), t(700))
    live = [{"state": HAIRCUT, "haircut_started_at": t(1000).isoformat(), "session_id": "s9", "visit_id": "v9",
             "chair_id": "ch2", "barber_id": None},
            {"state": WAITING, "waiting_started_at": t(1500).isoformat(), "visit_id": "w9"}]
    st = compute_stats(conn, t(-60), t(7200), live=live, now=t(3400), min_service_s=180)
    assert st["longest_haircut"]["seconds"] == 2400.0 and st["longest_haircut"]["running"] is True
    assert st["longest_haircut"]["chair_id"] == "ch2"
    assert st["average_haircut_s"] <= st["longest_haircut"]["seconds"]
    assert st["longest_wait"]["seconds"] == 1900.0 and st["longest_wait"]["running"] is True
    done = compute_stats(conn, t(-60), t(7200))
    assert done["longest_haircut"]["seconds"] == 600.0 and not done["longest_haircut"].get("running")


def test_door_free_the_longest_and_the_excessive_waits_come_from_the_customer_waits(tmp_path):
    """2026-10-18 (user): the 16-17 Oct reports read "longest wait —" and "0 excessive waits" beside 34 / 49 wait
    alerts: a door-free site's waits are customer waits (the visits have none), and one wait raised up to 8 alerts
    (closed and reopened as the customer was lost and found) -- a wait counts once."""
    from domain import Alert
    conn = connect(str(tmp_path / "b.db"))
    waits = {"w1": wait_rec("w1", 0, 1200), "w2": wait_rec("w2", 100, 400), "w3": wait_rec("w3", 500, None)}
    alerts = [Alert("a1", "excessive_wait", "shop", "w1", None, t(900), ended_at=t(1000), open=False),
              Alert("a2", "excessive_wait", "shop", "w1", None, t(1050), ended_at=t(1200), open=False)]
    apply_update(conn, EngineUpdate(waits=waits, alerts=alerts), t(1300))
    st = compute_stats(conn, t(-60), t(3600), visits_source=1, serve_s=150.0)
    assert st["longest_wait"]["seconds"] == 1200.0 and st["longest_wait"]["visit_id"] == "w1"
    assert st["excessive_wait"] == {"count": 1, "percent": 33.3}             # 1 of the 3 waits that started
    door = compute_stats(conn, t(-60), t(3600))                               # a door site: as before
    assert door["longest_wait"] is None and door["excessive_wait"]["count"] == 0


def test_buckets_follow_the_footage_clock_of_a_run_on_recordings(tmp_path):
    """2026-10-18 (user): a run on recordings reports the footage's hours. Its rows are in device time; the footage is
    4 h 06 min 48 s behind here, so its 19:00 hour runs from device 14:53:12."""
    conn = connect(str(tmp_path / "b.db"))
    shift = timedelta(hours=-4, minutes=-6, seconds=-48) + timedelta(days=-8)       # 18 Oct 23:06:48 -> 10 Oct 19:00
    dev = lambda m: datetime(2026, 10, 18, 23, 6, 48) + timedelta(minutes=m)
    ss = {f"s{i}": ServiceSession(f"s{i}", f"s{i}", "R1", dev(m), barber_id=None) for i, m in enumerate((1, 50, 70))}
    vs = {k: Visit(k, "p" + k, s.started_at, True, confirmed=True, open=False) for k, s in ss.items()}
    apply_update(conn, EngineUpdate(visits=vs, sessions=ss), dev(80))
    st = compute_stats(conn, dev(0), dev(120), bucket="hour", clock_shift=shift)
    assert [(b["bucket"], b["served_customers"]) for b in st["series"]] == [("2026-10-10T19:00:00", 2),
                                                                           ("2026-10-10T20:00:00", 1)]
