from datetime import date, datetime, timedelta

import pytest

from policy import DEFAULT_POLICY
from domain import (COMPLETED, EXIT_DOOR, FINISHED, HAIRCUT, WAITING, Alert, Binding, EngineUpdate, Handoff,
                    ServiceSession, StateEvent, Visit, WaitRecord)
from storage import (any_visit, apply_update, assign_chair, clear_all, close_interrupted, confirm_roster, connect, create_staff,
                     current_roster, delete_customer, get_customer, delete_staff, list_bindings, list_handoffs, list_staff, list_visits, max_ticket,
                     roster_at, roster_confirmation, update_staff, visit_events, waits_finished)

T = datetime(2026, 9, 24, 10, 0, 0)


def t(s):
    return T + timedelta(seconds=s)


@pytest.fixture
def conn(tmp_path):
    return connect(str(tmp_path / "b.db"))


def visit(**kw):
    v = Visit("v1", "p1", t(0), True, state=WAITING, confirmed=True, waiting_started_at=t(0), wait_target_s=900)
    for k, val in kw.items():
        setattr(v, k, val)
    return v


def test_apply_update_round_trip(conn):
    v = visit(state=HAIRCUT, service_started_at=t(40), wait_s=40)
    s = ServiceSession("s1", "v1", "ch1", t(40), barber_id="staff-1", service_target_s=1800)
    u = EngineUpdate(visits={"v1": v}, sessions={"s1": s},
                     events=[StateEvent("v1", None, WAITING, None, None, t(0)),
                             StateEvent("v1", WAITING, HAIRCUT, "ch1", None, t(40))],
                     handoffs=[Handoff("ch1", "barber-c", 1, "barber-l", 7, 0.4, 0.91, t(40))],
                     alerts=[Alert("a1", "excessive_wait", "shop", "v1", None, t(900))])
    apply_update(conn, u, t(41))
    s.ended_at, s.service_s, s.wash_s, s.open = t(200), 160.0, 30.0, False
    v.state, v.haircuts, v.outcome, v.exit_reason, v.exited_at, v.open = FINISHED, 1, COMPLETED, EXIT_DOOR, t(234), False
    apply_update(conn, EngineUpdate(visits={"v1": v}, sessions={"s1": s}), t(235))
    (row,) = list_visits(conn, t(-10), t(10))
    assert (row["person_id"], row["outcome"], row["exit_reason"], row["open"]) == ("p1", COMPLETED, EXIT_DOOR, False)
    (cut,) = row["haircut_sessions"]
    assert (cut["chair_id"], cut["service_s"], cut["wash_s"], cut["barber_id"]) == ("ch1", 160.0, 30.0, "staff-1")
    assert [(e["from_state"], e["to_state"]) for e in visit_events(conn, "v1")] == [(None, WAITING), (WAITING, HAIRCUT)]
    (h,) = list_handoffs(conn, t(0), t(100))
    assert (h["area"], h["track_b"], h["offset_s"]) == ("ch1", 7, 0.4)


def test_ticket_round_trips_and_max_ticket_reflects_it(conn):
    v = visit(ticket=3)
    apply_update(conn, EngineUpdate(visits={"v1": v}), t(1))
    assert list_visits(conn, t(-10), t(10))[0]["ticket"] == 3
    assert max_ticket(conn, t(-10), t(10)) == 3


def test_max_ticket_over_an_empty_range_is_zero(conn):
    assert max_ticket(conn, t(-10), t(10)) == 0


def test_bindings_round_trip_and_clear_all_empties_them(conn):
    b = Binding("handoff", "barber-l", 7, "p1", "ch1", "barber-c", 1, 0.91, 0.4, t(40))
    apply_update(conn, EngineUpdate(bindings=[b]), t(41))
    (row,) = list_bindings(conn, t(0), t(100))
    assert (row["kind"], row["camera"], row["track_id"], row["person_id"], row["area"], row["partner_camera"],
            row["partner_track"], row["reid_sim"], row["offset_s"]) == \
        ("handoff", "barber-l", 7, "p1", "ch1", "barber-c", 1, 0.91, 0.4)
    clear_all(conn)
    assert list_bindings(conn, t(0), t(100)) == []


def test_deleted_session_is_removed(conn):
    apply_update(conn, EngineUpdate(visits={"v1": visit()}, sessions={"s1": ServiceSession("s1", "v1", "ch1", t(40))}), t(41))
    apply_update(conn, EngineUpdate(deleted_sessions={"s1"}), t(70))
    assert list_visits(conn, t(-10), t(10))[0]["haircut_sessions"] == []


def test_close_interrupted_and_clear_all(conn):
    apply_update(conn, EngineUpdate(visits={"v1": visit()}, events=[StateEvent("v1", None, WAITING, None, None, t(0))]), t(1))
    assert close_interrupted(conn)["visits"] == 1
    assert clear_all(conn) == set() and list_visits(conn, t(-10), t(10)) == []


def test_clear_all_returns_the_crop_paths_it_orphaned(conn):
    apply_update(conn, EngineUpdate(visits={"v1": visit()}), t(1))
    conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, last_visit_id, face_count)"
                 " VALUES ('c1', ?, ?, 1, 'v1', 1)", (t(0).isoformat(timespec="seconds"), t(0).isoformat(timespec="seconds")))
    conn.execute("INSERT INTO customer_face (customer_id, crop_path, embedding, quality, taken_at, visit_id)"
                 " VALUES ('c1', 'faces/x.jpg', ?, 0.9, ?, 'v1')", (b"\x00" * 4, t(0).isoformat(timespec="seconds")))
    conn.commit()
    assert clear_all(conn) == {"faces/x.jpg"}
    assert list_visits(conn, t(-10), t(10)) == []
    assert conn.execute("SELECT COUNT(*) FROM customer").fetchone()[0] == 0


def test_staff_crud(conn):
    s = create_staff(conn, "lisa")
    assert s["id"] == "staff-1" and s["active"]
    assert update_staff(conn, s["id"], active=False)["active"] is False
    assert [x["name"] for x in list_staff(conn)] == ["lisa"]
    assert delete_staff(conn, s["id"]) and not delete_staff(conn, s["id"])


def test_roster_answers_who_worked_a_chair_at_a_time(conn):
    lisa, maya = create_staff(conn, "lisa")["id"], create_staff(conn, "maya")["id"]
    assign_chair(conn, "ch1", lisa, t(0))
    assign_chair(conn, "ch1", maya, t(100))
    assign_chair(conn, "ch2", lisa, t(150))               # lisa moves: her ch1 row was already closed
    assert roster_at(conn, "ch1", t(50)) == lisa and roster_at(conn, "ch1", t(120)) == maya
    assert roster_at(conn, "ch1", t(-1)) is None
    assert current_roster(conn) == {"ch1": maya, "ch2": lisa}
    assign_chair(conn, "ch1", None, t(200))
    assert current_roster(conn) == {"ch2": lisa} and roster_at(conn, "ch1", t(250)) is None


def test_roster_confirmation_is_per_shop_day(conn):
    confirm_roster(conn, date(2026, 9, 25), t(0))
    assert roster_confirmation(conn, date(2026, 9, 26))["confirmed"] is False
    assert roster_confirmation(conn, date(2026, 9, 25))["confirmed"] is True


def test_adding_a_column_another_process_just_added_is_harmless(tmp_path):
    """Two processes starting on an old DB both see the column missing; the
    loser's ALTER must not crash the service."""
    from storage import _add_column
    conn = connect(str(tmp_path / "b.db"))
    _add_column(conn, "visit", "ticket", "INTEGER")  # already there
    assert "ticket" in {r["name"] for r in conn.execute("PRAGMA table_info(visit)")}


def test_alerts_carry_their_visits_entered_at(conn):
    """An alert links to its visit's Insights row, found by the visit's day."""
    from storage import list_alerts
    apply_update(conn, EngineUpdate(visits={"v1": visit()}, alerts=[Alert("a1", "excessive_wait", "shop", "v1", None,
                                                                        t(900))]), t(900))
    (a,) = list_alerts(conn)
    assert a["visit_entered_at"] == t(0).isoformat(timespec="seconds")


def test_alerts_carry_their_ticket(conn):
    """A haircut alert shows its visit's ticket, a wait alert its customer wait's."""
    from domain import WaitRecord
    from storage import list_alerts
    w = WaitRecord("w1", t(0), None, None, None, 900, "barber-c", 1, 1, 12, t(10))
    apply_update(conn, EngineUpdate(visits={"v1": visit(ticket=7)}, waits={"w1": w},
                                    alerts=[Alert("a1", "long_haircut", "ch1", "v1", None, t(900)),
                                            Alert("a2", "excessive_wait", "shop", "w1", None, t(950))]), t(960))
    assert {a["id"]: a["ticket"] for a in list_alerts(conn)} == {"a1": 7, "a2": 12}


def test_a_reopened_session_clears_its_end_on_the_existing_row(conn):
    # 2026-09-28 (39-min testbed spec, 6.1): chair resume reopens a closed haircut by id
    s = ServiceSession("s1", "v1", "ch1", t(40), ended_at=t(200), service_s=160.0, open=False)
    apply_update(conn, EngineUpdate(visits={"v1": visit()}, sessions={"s1": s}), t(201))
    s.ended_at, s.service_s, s.open = None, None, True
    apply_update(conn, EngineUpdate(sessions={"s1": s}), t(230))
    row = conn.execute("SELECT ended_at, service_s, open, started_at FROM service_session WHERE id = 's1'").fetchone()
    assert tuple(row) == (None, None, 1, t(40).isoformat(timespec="seconds"))


def test_door_crossings_and_sittings_are_stored(conn):
    from domain import DoorCrossing, SittingRecord
    at = datetime(2026, 10, 4, 12, 0, 5)
    u = EngineUpdate(door_crossings=[DoorCrossing(at, "in", "barber-c", 3, False)],
                     sittings=[SittingRecord("h1", 0, "R3", at, None)])
    apply_update(conn, u, at)
    u2 = EngineUpdate(sittings=[SittingRecord("h1", 0, "R3", at, datetime(2026, 10, 4, 12, 10))])
    apply_update(conn, u2, at)
    assert [tuple(r) for r in conn.execute("SELECT direction, staff FROM door_crossing")] == [("in", 0)]
    assert [tuple(r) for r in conn.execute("SELECT n, ended_at FROM chair_sitting")] == [(0, "2026-10-04T12:10:00")]
    clear_all(conn)
    assert conn.execute("SELECT COUNT(*) FROM door_crossing").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chair_sitting").fetchone()[0] == 0


def wrec(**kw):
    base = dict(id="w1", started_at=t(0), ended_at=None, wait_s=None, ended_by=None, target_s=900.0,
                camera="barber-c", track_id=7, last_track_id=7, ticket=3, last_seen_at=t(50))
    base.update(kw)
    return WaitRecord(**base)


def test_customer_wait_round_trip(conn):
    apply_update(conn, EngineUpdate(waits={"w1": wrec()}), t(1))
    apply_update(conn, EngineUpdate(waits={"w1": wrec(ended_at=t(120), wait_s=120.0, ended_by="chair",
                                                      last_track_id=9, last_seen_at=t(120))}), t(121))
    (r,) = conn.execute("SELECT * FROM customer_wait").fetchall()
    assert (r["ended_by"], r["wait_s"], r["last_track_id"], r["track_id"], r["ticket"]) == ("chair", 120.0, 9, 7, 3)
    assert [w["id"] for w in waits_finished(conn, t(100), t(200))] == ["w1"]
    assert waits_finished(conn, t(200), t(300)) == []


def test_an_open_wait_is_not_finished(conn):
    apply_update(conn, EngineUpdate(waits={"w1": wrec()}), t(1))
    assert waits_finished(conn, t(0), t(1000)) == []


def test_restart_closes_open_waits_at_their_last_sighting(conn):
    apply_update(conn, EngineUpdate(waits={"w1": wrec()}), t(1))
    apply_update(conn, EngineUpdate(waits={"w1": wrec(last_seen_at=t(80))}), t(81))
    assert close_interrupted(conn)["waits"] == 1
    (r,) = conn.execute("SELECT * FROM customer_wait").fetchall()
    assert (r["ended_by"], r["ended_at"], r["wait_s"]) == ("restart", t(80).isoformat(timespec="seconds"), 80.0)


def test_max_ticket_counts_wait_tickets(conn):
    apply_update(conn, EngineUpdate(waits={"w1": wrec(ticket=7)}), t(1))
    assert max_ticket(conn, t(-60), t(3600)) == 7


def test_clear_all_removes_waits(conn):
    apply_update(conn, EngineUpdate(waits={"w1": wrec()}), t(1))
    clear_all(conn)
    assert conn.execute("SELECT COUNT(*) FROM customer_wait").fetchone()[0] == 0


def test_alert_snapshots_are_referenced_cleared_and_purged(conn):
    from storage import purge_snapshots, referenced_crops, set_alert_snapshot
    a = Alert("a1", "excessive_wait", "shop", "w1", None, t(0), subject_cam="barber-c", subject_box=(0.1, 0.2, 0.3, 0.4))
    apply_update(conn, EngineUpdate(alerts=[a]), t(1))
    set_alert_snapshot(conn, "a1", "alerts/a1.jpg")
    assert "alerts/a1.jpg" in referenced_crops(conn)
    assert purge_snapshots(conn, t(0) + timedelta(days=400), 365) == ["alerts/a1.jpg"]
    assert "alerts/a1.jpg" not in referenced_crops(conn)
    set_alert_snapshot(conn, "a1", "alerts/a1.jpg")
    assert "alerts/a1.jpg" in clear_all(conn)


def test_a_door_entry_registration_links_its_crossing(conn, tmp_path):
    from domain import DoorCrossing, FaceCandidate, Registration
    import register
    emb = tuple([1.0] + [0.0] * 511)
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(t(0), "in", "barber-c", 9, False, entry_id="e1")]), t(1))
    faces = tuple(FaceCandidate("barber-c", 9, t(0).timestamp() + i, f"f/{i}.jpg", emb, 0.9) for i in range(3))
    ids = register.apply(conn, EngineUpdate(registrations=[Registration("e1", t(0), faces, ())]), tmp_path,
                         dict(DEFAULT_POLICY), lambda: "cust1")
    assert ids == ["cust1"]
    assert conn.execute("SELECT customer_id FROM door_crossing WHERE entry_id = 'e1'").fetchone()[0] == "cust1"
    c = get_customer(conn, "cust1")
    assert [e["id"] for e in c["entries"]] == ["e1"] and c["visit_count"] == 1


def test_an_entry_with_too_few_clean_faces_gets_no_card(conn, tmp_path):
    # min_visit_faces is 1 since 2026-10-12 (one face makes a card: test_register.py); 2 here keeps the no-card path
    from domain import DoorCrossing, FaceCandidate, Registration
    import register
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "0.jpg").write_bytes(b"x")
    emb = tuple([1.0] + [0.0] * 511)
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(t(0), "in", "barber-c", 9, False, entry_id="e1")]), t(1))
    one = (FaceCandidate("barber-c", 9, t(0).timestamp(), "f/0.jpg", emb, 0.9),)
    ids = register.apply(conn, EngineUpdate(registrations=[Registration("e1", t(0), one, ())]), tmp_path,
                         dict(DEFAULT_POLICY, min_visit_faces=2), lambda: "cust1")
    assert ids == [] and conn.execute("SELECT COUNT(*) FROM customer").fetchone()[0] == 0
    assert not (tmp_path / "f" / "0.jpg").exists()
    assert conn.execute("SELECT customer_id FROM door_crossing WHERE entry_id = 'e1'").fetchone()[0] is None


def test_erasing_a_customer_unlinks_their_entries(conn, tmp_path):
    from domain import DoorCrossing, FaceCandidate, Registration
    import register
    emb = tuple([1.0] + [0.0] * 511)
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(t(0), "in", "barber-c", 9, False, entry_id="e1")]), t(1))
    faces = tuple(FaceCandidate("barber-c", 9, t(0).timestamp() + i, f"f/{i}.jpg", emb, 0.9) for i in range(3))
    register.apply(conn, EngineUpdate(registrations=[Registration("e1", t(0), faces, ())]), tmp_path,
                   dict(DEFAULT_POLICY), lambda: "cust1")
    delete_customer(conn, "cust1")
    assert conn.execute("SELECT customer_id FROM door_crossing WHERE entry_id = 'e1'").fetchone()[0] is None


def test_an_old_door_crossing_table_is_migrated(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE door_crossing (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,"
                " direction TEXT NOT NULL, camera TEXT NOT NULL, track_id INTEGER NOT NULL,"
                " staff INTEGER NOT NULL DEFAULT 0)")
    raw.commit()
    raw.close()
    conn = connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(door_crossing)")}
    assert {"entry_id", "customer_id"} <= cols
    assert "idx_door_crossing_customer" in {r["name"] for r in conn.execute("PRAGMA index_list(door_crossing)")}
    from domain import DoorCrossing
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(t(0), "in", "barber-c", 9, False, entry_id="e1")]), t(1))
    assert conn.execute("SELECT entry_id FROM door_crossing").fetchone()[0] == "e1"


def test_restart_deletes_alerts_it_interrupted(conn):
    apply_update(conn, EngineUpdate(alerts=[Alert("a1", "excessive_service", "shop", "v1", "s1", t(0)),
                                            Alert("a2", "excessive_wait", "shop", "w1", None, t(0), ended_at=t(60), open=False)]), t(61))
    assert close_interrupted(conn)["alerts"] == 1
    assert [r["id"] for r in conn.execute("SELECT id FROM barber_alert")] == ["a2"]


def test_visit_baseline_is_set_once_per_day_and_clear_all_removes_it(conn):
    from storage import baselines_in, set_visit_baseline, visit_baseline
    assert visit_baseline(conn, "2026-09-24") is None
    set_visit_baseline(conn, "2026-09-24", t(120), haircuts=2, waiting=1, washing=1)
    set_visit_baseline(conn, "2026-09-24", t(900), haircuts=9, waiting=9, washing=9)   # ignored
    b = visit_baseline(conn, "2026-09-24")
    assert (b["count"], b["haircuts"], b["waiting"], b["washing"]) == (4, 2, 1, 1)
    assert b["taken_at"] == t(120).isoformat(timespec="seconds")
    set_visit_baseline(conn, "2026-09-25", t(86400 + 120), 1, 0, 0)
    assert [r["day"] for r in baselines_in(conn, t(0), t(3600))] == ["2026-09-24"]
    assert len(baselines_in(conn, t(0), t(2 * 86400))) == 2
    clear_all(conn)
    assert visit_baseline(conn, "2026-09-24") is None and baselines_in(conn, t(0), t(2 * 86400)) == []


def test_every_re_id_check_is_kept_and_summarised_per_chair(conn):
    """2026-10-18 (user): keep every score -- per chair, the same-customer and new-customer scores' mean / min / max."""
    from domain import ResumeCheckRecord
    from storage import resume_check_stats
    t = datetime(2026, 10, 18, 12, 0)

    def rec(chair, sim, resumed, m):
        at = t + timedelta(minutes=m)
        return ResumeCheckRecord(chair, "return", at, at + timedelta(seconds=30), 90.0, sim, 0.62, resumed,
                                 f"h{m}", "h0", 6, 4)
    apply_update(conn, EngineUpdate(resume_checks=[rec("L6", 0.9, True, 1), rec("L6", 0.8, True, 2),
                                                   rec("L6", 0.4, False, 3), rec("L6", None, True, 4),
                                                   rec("R2", 0.95, True, 5)]), t)
    stats = resume_check_stats(conn, t, t + timedelta(days=1))
    assert [s["chair_id"] for s in stats] == ["L6", "R2"]
    assert stats[0]["same"] == {"n": 2, "mean": 0.85, "min": 0.8, "max": 0.9, "no_score": 1}
    assert stats[0]["new"] == {"n": 1, "mean": 0.4, "min": 0.4, "max": 0.4, "no_score": 0}
    assert stats[1]["new"] is None
    assert resume_check_stats(conn, t + timedelta(days=1), t + timedelta(days=2)) == []
    clear_all(conn)
    assert conn.execute("SELECT COUNT(*) FROM resume_check").fetchone()[0] == 0


def test_any_visit_tells_a_period_with_visits_from_one_without(conn):
    t = datetime(2026, 10, 20, 13, 0)
    apply_update(conn, EngineUpdate(visits={"v1": Visit("v1", "p1", t, False, confirmed=True)}), t)
    assert any_visit(conn, datetime(2026, 10, 20, 5), datetime(2026, 10, 21, 5))
    assert not any_visit(conn, datetime(2026, 10, 21, 5), datetime(2026, 10, 22, 5))


def test_a_deleted_session_takes_its_sittings_and_alerts_with_it(tmp_path):
    """2026-11-10 (staff haircuts): a session deleted after it was written leaves no chair_sitting or alert rows."""
    from domain import SittingRecord
    conn = connect(tmp_path / "b.db")
    t = datetime(2026, 10, 16, 13, 7)
    v = Visit(id="h1", person_id="h1", entered_at=t, entry_observed=False, state=HAIRCUT, confirmed=True,
              service_started_at=t, haircuts=1, ticket=3, wait_unseen=True)
    s = ServiceSession(id="h1", visit_id="h1", chair_id="ph7-VIP1", started_at=t)
    apply_update(conn, EngineUpdate(visits={"h1": v}, sessions={"h1": s},
                                    sittings=[SittingRecord("h1", 0, "ph7-VIP1", t, None)]), t)
    conn.execute("INSERT INTO barber_alert (id, kind, visit_id, session_id, camera, started_at, open) VALUES"
                 " ('a1', 'excessive_service', 'h1', 'h1', 'ph-7', ?, 1)", (t.isoformat(),))
    apply_update(conn, EngineUpdate(deleted_sessions={"h1"}), t)
    for table, col in (("service_session", "id"), ("chair_sitting", "session_id"), ("barber_alert", "session_id")):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = 'h1'").fetchone()[0] == 0


# ── pedicure sessions (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md) ─────────────────────────────
from domain import PedicureRecord  # noqa: E402
from storage import pedicures_in  # noqa: E402

PD0 = datetime(2026, 10, 16, 14, 16)


def test_a_pedicure_round_trips_and_its_close_updates_the_row(tmp_path):
    conn = connect(tmp_path / "b.db")
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0)}), PD0)
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0, PD0 + timedelta(minutes=41), 2460.0,
                                                                     False)}), PD0)
    (r,) = pedicures_in(conn, PD0 - timedelta(hours=1), PD0 + timedelta(hours=1))
    assert (r["chair_id"], r["service_s"], r["open"]) == ("P1", 2460.0, 0)


def test_a_pedicure_is_no_visit_and_no_session(tmp_path):
    conn = connect(tmp_path / "b.db")
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0)}), PD0)
    assert conn.execute("SELECT COUNT(*) FROM visit").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM service_session").fetchone()[0] == 0


def test_a_restart_closes_an_open_pedicure(tmp_path):
    conn = connect(tmp_path / "b.db")
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0)}), PD0)
    assert close_interrupted(conn)["pedicures"] == 1
    (r,) = pedicures_in(conn, PD0 - timedelta(hours=1), PD0 + timedelta(hours=1))
    assert r["open"] == 0 and r["service_s"] is None


def test_a_lap_clear_deletes_pedicures(tmp_path):
    conn = connect(tmp_path / "b.db")
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0)}), PD0)
    clear_all(conn)
    assert pedicures_in(conn, PD0 - timedelta(hours=1), PD0 + timedelta(hours=1)) == []


def test_an_update_with_only_a_pedicure_is_not_empty():
    assert EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", PD0)})


def test_an_old_session_table_gains_the_occupancy_columns(tmp_path):
    import sqlite3                                              # 2026-11-14: ce_low_occupancy
    path = str(tmp_path / "old.db")
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE service_session (id TEXT PRIMARY KEY, visit_id TEXT NOT NULL, chair_id TEXT NOT NULL,"
                " started_at TEXT NOT NULL, ended_at TEXT, service_s REAL, wash_s REAL NOT NULL DEFAULT 0,"
                " barber_id TEXT, service_target_s REAL, excessive_service INTEGER NOT NULL DEFAULT 0,"
                " open INTEGER NOT NULL DEFAULT 1)")
    raw.execute("INSERT INTO service_session (id, visit_id, chair_id, started_at) VALUES ('s0', 'v0', 'L6', '2026-10-04T12:00:00')")
    raw.commit()
    raw.close()
    conn = connect(path)
    old = conn.execute("SELECT occupancy, low_occupancy FROM service_session").fetchone()
    assert old["occupancy"] is None and old["low_occupancy"] == 0
    v = Visit("v1", "p1", t(0), False, state="haircut", confirmed=True, service_started_at=t(0))
    s = ServiceSession("s1", "v1", "L6", t(0), occupancy=0.2, low_occupancy=True)
    apply_update(conn, EngineUpdate(visits={"v1": v}, sessions={"s1": s}), t(1))
    row = conn.execute("SELECT occupancy, low_occupancy FROM service_session WHERE id = 's1'").fetchone()
    assert row["occupancy"] == 0.2 and row["low_occupancy"] == 1
