"""The pedicure book (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md): a HaircutBook whose sittings are
pedicures -- never a visit, a session, a ticket, an alert or a customer card; its own minimum; resume by time only."""
from chairs2 import SitEnd, SitStart
from domain import Appearance, EngineUpdate
from haircuts import HaircutBook, pedicure_policy
from policy import DEFAULT_POLICY
from zones import PEDICURE

from test_haircuts import NO_OCC, T0, Ids

PP = pedicure_policy({**DEFAULT_POLICY, "pd_min_s": 300.0, "ce_resume_check": 1.0, "ce_reseat_check_s": 60.0})


def pbook():
    return HaircutBook(Ids(), lambda chair, at: "staff-1", lambda at: 7, kind=PEDICURE)


def sit(b, start, end, chair="P1", tid=4):
    u = EngineUpdate()
    b.on_start(SitStart(chair, start, tid), u, PP)
    t = start
    while t < end:
        b.tick(t, lambda ch: True, u, PP, lambda ch: t)
        t += 10
    b.on_end(SitEnd(chair, end), u, PP)
    return u


def test_the_pedicure_policy_uses_its_own_minimum_and_no_re_id():
    assert PP["ce_min_haircut_s"] == 300.0
    assert PP["ce_resume_check"] == 0.0 and PP["ce_reseat_check_s"] == 0.0


def test_a_pedicure_is_never_a_visit_a_session_a_ticket_or_an_alert():
    b = pbook()
    u = sit(b, T0, T0 + 600)
    assert not u.visits and not u.sessions and not u.sittings and not u.alerts and not u.registrations
    (p,) = u.pedicures.values()
    assert p.chair_id == "P1" and p.open
    h = b.open["P1"]
    assert h.confirmed and h.ticket is None and h.barber_id is None and h.service_target_s is None


def test_a_sit_shorter_than_pd_min_s_is_not_a_pedicure():
    b = pbook()
    u = sit(b, T0, T0 + 120)
    assert not u.pedicures and not b.open["P1"].confirmed


def test_a_pedicure_closes_with_its_sitting_time():
    b = pbook()
    sit(b, T0, T0 + 600)
    u = EngineUpdate()
    b.tick(T0 + 600 + PP["ce_resume_window_s"] + 1, lambda ch: False, u, PP, NO_OCC)
    (p,) = u.pedicures.values()
    assert not p.open and abs(p.service_s - 600) < 1 and p.ended_at is not None
    assert not u.registrations and not u.alerts


def test_back_within_the_window_continues_the_same_pedicure_without_re_id():
    b = pbook()
    sit(b, T0, T0 + 600)
    sit(b, T0 + 700, T0 + 900, tid=9)
    h = b.open["P1"]
    assert [s[0] for s in h.sittings] == [T0, T0 + 700] and not b.pending and not b.decisions


def test_a_pedicure_sitters_crops_are_discarded():
    b = pbook()
    sit(b, T0, T0 + 600)
    u = EngineUpdate()
    b.on_appearance(Appearance("ph-6", 4, T0 + 610, None, 0.1, "bodies/x.jpg", 0.5), "P1", PP, u)
    assert u.discard_crops == ["bodies/x.jpg"] and not b.open["P1"].bodies


def test_pd_min_s_has_a_default_and_a_maximum():
    from policy import MAXIMUMS
    assert DEFAULT_POLICY["pd_min_s"] == 300.0 and MAXIMUMS["pd_min_s"] == 3600.0
