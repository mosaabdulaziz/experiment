import itertools
from datetime import datetime

from alerts import EXCESSIVE_SERVICE, EXCESSIVE_WAIT
from customers import CustomerBook
from domain import COMPLETED, EXIT_DOOR, EXIT_LOST, HAIRCUT, LOST, SERVED, VOID_STAFF, WAITING, WALKED_OUT, WASHING
from shop_helpers import POLICY, at


def book(barber_for=lambda chair, when: None):
    n = itertools.count()
    return CustomerBook(lambda: f"id{next(n)}", barber_for)


def states(update):
    return [(e.from_state, e.to_state) for e in update.events]


def test_unconfirmed_visit_is_not_persisted_until_confirmed():
    b = book()
    b.open("p1", at(3), True, POLICY)
    assert not b.take_update()
    b.confirm("p1")
    u = b.take_update()
    (v,) = u.visits.values()
    assert v.state == WAITING and v.waiting_started_at == at(3)
    assert states(u) == [(None, WAITING)] and u.events[0].at == at(3)


def test_haircut_start_measures_the_wait_and_takes_the_barber_from_the_roster():
    b = book(lambda chair, when: "staff-2" if chair == "ch1" else None)
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    u = b.take_update()
    v = b.visits["p1"]
    assert v.state == HAIRCUT and v.wait_s == 40 and v.service_started_at == at(40)
    assert s.barber_id == "staff-2" and u.sessions[s.id].chair_id == "ch1"
    assert states(u)[-1] == (WAITING, HAIRCUT)


def test_a_short_sit_is_not_a_haircut_and_the_wait_goes_on():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(70), False, POLICY)           # 30 s < min_service_s 60
    u = b.take_update()
    v = b.visits["p1"]
    assert s.id in u.deleted_sessions and s.id not in u.sessions
    assert v.state == WAITING and v.service_started_at is None and v.wait_s is None and v.haircuts == 0


def test_a_full_haircut_ends_served():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(200), False, POLICY)
    u = b.take_update()
    assert u.sessions[s.id].service_s == 160 and not u.sessions[s.id].open
    assert b.visits["p1"].state == SERVED and b.visits["p1"].haircuts == 1


def test_washing_suspends_and_returning_resumes_the_same_haircut():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(150), True, POLICY)
    assert b.visits["p1"].state == WASHING and s.washing_since == at(150)
    again = b.start_haircut("p1", "ch1", at(180), POLICY)
    assert again is s and s.wash_s == 30 and s.washing_since is None and b.visits["p1"].state == HAIRCUT
    b.end_haircut(s.id, at(300), False, POLICY)
    u = b.take_update()
    assert u.sessions[s.id].service_s == 260 and u.sessions[s.id].wash_s == 30
    assert [t for _, t in states(u)] == [WAITING, HAIRCUT, WASHING, HAIRCUT, SERVED]


def test_no_return_from_washing_closes_the_haircut_when_he_left_the_seat():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(150), True, POLICY)
    b.expire_washing(at(449), POLICY)
    assert s.open
    b.expire_washing(at(450), POLICY)                   # wash_return_s 300
    assert not s.open and s.ended_at == at(150) and b.visits["p1"].state == SERVED


def test_finish_by_the_door_without_a_haircut_is_a_walk_out_with_its_wait():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    b.finish("p1", at(90), EXIT_DOOR, POLICY)
    (v,) = b.take_update().visits.values()
    assert (v.outcome, v.exit_reason, v.wait_s, v.open) == (WALKED_OUT, EXIT_DOOR, 90, False)
    assert "p1" not in b.visits


def test_finish_after_a_haircut_is_completed_and_a_lost_one_says_so():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(200), False, POLICY)
    b.finish("p1", at(250), EXIT_LOST, POLICY)
    v = b.take_update().visits
    (v,) = v.values()
    assert (v.outcome, v.exit_reason, v.wait_s) == (COMPLETED, EXIT_LOST, 40)
    b.open("p2", at(0), True, POLICY, confirmed=True)
    b.finish("p2", at(30), EXIT_LOST, POLICY)
    (v2,) = b.take_update().visits.values()
    assert v2.outcome == LOST


def test_finish_while_seated_closes_the_running_haircut():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.finish("p1", at(200), EXIT_DOOR, POLICY)
    u = b.take_update()
    assert u.sessions[s.id].ended_at == at(200) and u.visits[s.visit_id].outcome == COMPLETED


def test_unconfirmed_visit_leaves_no_trace():
    b = book()
    b.open("p1", at(0), True, POLICY)
    b.finish("p1", at(2), EXIT_DOOR, POLICY)
    assert not b.take_update() and not b.visits


def test_void_only_without_a_haircut():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    assert b.void("p1", at(8))
    (v,) = b.take_update().visits.values()
    assert v.outcome == VOID_STAFF and not v.open
    b.open("p2", at(0), True, POLICY, confirmed=True)
    b.start_haircut("p2", "ch1", at(10), POLICY)
    assert not b.void("p2", at(20)) and "p2" in b.visits


def test_confirming_a_visit_gives_it_the_next_ticket_and_stamps_state_since():
    tickets = iter([41, 42])
    book = CustomerBook(lambda: "v", ticket_for=lambda at: next(tickets))
    t = datetime(2026, 9, 25, 10)
    v = book.open("p1", t, True, POLICY, confirmed=True)
    assert v.ticket == 41 and v.state == WAITING and v.state_since == t


def test_an_unconfirmed_visit_has_no_ticket():
    book = CustomerBook(lambda: "v", ticket_for=lambda at: 1)
    assert book.open("p1", datetime(2026, 9, 25, 10), False, POLICY).ticket is None


def test_wait_and_haircut_alerts_open_and_close():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    b.check_alerts(at(100))
    assert not b.take_update().alerts                    # not past wait_target_s 100 yet
    b.check_alerts(at(101))
    (a,) = b.take_update().alerts
    assert a.kind == EXCESSIVE_WAIT and a.started_at == at(100) and b.visits["p1"].excessive_wait
    s = b.start_haircut("p1", "ch1", at(120), POLICY)
    (closed,) = b.take_update().alerts
    assert closed.id == a.id and not closed.open
    b.check_alerts(at(521))
    (h,) = b.take_update().alerts
    assert h.kind == EXCESSIVE_SERVICE and h.session_id == s.id


# 2026-09-28 (39-min testbed spec, 6.1): chair resume reopens the closed haircut

def test_reopening_a_closed_haircut_keeps_its_session_and_wait():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(200), False, POLICY)
    b.take_update()
    again = b.reopen_haircut("p1", s.id, at(230), POLICY)
    u = b.take_update()
    v = b.visits["p1"]
    assert again is s and b.sessions[s.id] is s and v.session_id == s.id
    assert (s.ended_at, s.service_s, s.open) == (None, None, True) and u.sessions[s.id] is s
    assert (v.state, v.haircuts, v.wait_s, v.service_started_at) == (HAIRCUT, 0, 40, at(40))
    assert states(u) == [(SERVED, HAIRCUT)] and u.events[0].at == at(230) and u.events[0].area == "ch1"
    b.end_haircut(s.id, at(300), False, POLICY)
    assert (s.service_s, v.haircuts, v.state) == (260, 1, SERVED)


def test_reopening_a_short_sit_restores_its_session_and_wait():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(s.id, at(70), False, POLICY)           # under min_service_s: deleted, wait reset
    assert b.reopen_haircut("p1", s.id, at(80), POLICY) is s
    u = b.take_update()
    v = b.visits["p1"]
    assert s.id in u.sessions and s.id not in u.deleted_sessions
    assert (v.state, v.wait_s, v.service_started_at) == (HAIRCUT, 40, at(40))


def test_reopening_a_short_sit_after_a_counted_haircut_leaves_the_count_alone():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    first = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.end_haircut(first.id, at(200), False, POLICY)
    s = b.start_haircut("p1", "ch2", at(210), POLICY)
    b.end_haircut(s.id, at(220), False, POLICY)
    b.take_update()                                      # the deletion reached storage
    assert b.reopen_haircut("p1", s.id, at(230), POLICY) is s
    v = b.visits["p1"]
    assert (v.haircuts, v.service_started_at, v.wait_s) == (1, at(40), 40)
    assert s.id in b.take_update().sessions


def test_only_the_persons_last_closed_haircut_can_be_reopened():
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    assert b.reopen_haircut("p1", s.id, at(50), POLICY) is None          # still running
    b.end_haircut(s.id, at(200), False, POLICY)
    assert b.reopen_haircut("p1", "nope", at(210), POLICY) is None
    assert b.reopen_haircut("p2", s.id, at(210), POLICY) is None
    b.finish("p1", at(220), EXIT_DOOR, POLICY)
    assert b.reopen_haircut("p1", s.id, at(230), POLICY) is None


def test_a_reopened_haircut_past_its_target_keeps_one_excessive_service_alert():
    # 2026-09-28 (Task 14 fix round 1, ruling b)
    b = book()
    b.open("p1", at(0), True, POLICY, confirmed=True)
    s = b.start_haircut("p1", "ch1", at(40), POLICY)
    b.check_alerts(at(500))                              # service_target_s 400
    b.end_haircut(s.id, at(500), False, POLICY)
    b.reopen_haircut("p1", s.id, at(520), POLICY)
    b.check_alerts(at(600))
    b.end_haircut(s.id, at(700), False, POLICY)
    ids = {a.id for a in b.take_update().alerts if a.kind == EXCESSIVE_SERVICE and a.session_id == s.id}
    assert len(ids) == 1
