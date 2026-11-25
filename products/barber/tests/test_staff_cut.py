"""ce_staff_cut_check (2026-11-10, user: on 16 Oct two VIP1 haircuts were staff members' own): on a chair where it is on,
a confirmed haircut whose sitter looked staff in the minute before he sat down AND in the two minutes after he got up
is a staff haircut -- its session goes, its visit is voided as staff (no customer, no card, no photos, no alert)."""
from chair_engine import ChairEngine
from domain import VOID_STAFF, Appearance, Frame, TrackObs
from zones import FrameGeometry

from test_chair_engine import P, SITTER, T0, ZONES, sq

WALK = tuple(v for p in sq(0.60, 0.05, 0.70, 0.36) for v in p)          # standing away from the chair
SEATED, AWAY = TrackObs(4, (140, 50, 120, 310), 0.9, SITTER), TrackObs(4, (600, 50, 100, 310), 0.9, WALK)


def frames(t0, t1, track):
    return [Frame("barber-l", t0 + i * 0.5, (track,)) for i in range(int((t1 - t0) / 0.5))]


def apps(t0, t1, p):
    return [Appearance("barber-l", 4, t, None, p) for t in range(int(t0), int(t1), 5)]


def run(policy, before_p=0.97, after_p=0.97):
    e = ChairEngine(ZONES, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")
    ups = [e.observe(frames(T0 - 60, T0, AWAY), apps(T0 - 55, T0, before_p), policy),       # working (or waiting)
           e.observe(frames(T0, T0 + 400, SEATED), apps(T0, T0 + 400, 0.1), policy),       # caped in the chair
           e.observe(frames(T0 + 400, T0 + 560, AWAY), apps(T0 + 405, T0 + 525, after_p), policy)]   # back to work
    return ups


def sessions(ups):
    out = {}
    for u in ups:
        out.update(u.sessions)
        for sid in u.deleted_sessions:
            out.pop(sid, None)
    return out


def test_off_a_staff_members_haircut_counts_as_before():
    assert len(sessions(run({**P}))) == 1


def test_on_a_sitter_staff_before_and_after_is_a_staff_haircut():
    ups = run({**P, "ce_staff_cut_check@R1": 1.0})
    assert sessions(ups) == {}
    voided = [v for u in ups for v in u.visits.values() if v.outcome == VOID_STAFF]
    assert len(voided) == 1 and not voided[0].open
    assert any(u.dropped_photos for u in ups) and not any(u.registrations for u in ups)


def test_a_customer_who_only_looked_staff_before_still_counts():
    assert len(sessions(run({**P, "ce_staff_cut_check@R1": 1.0}, after_p=0.1))) == 1


def test_a_customer_who_only_looked_staff_after_still_counts():
    assert len(sessions(run({**P, "ce_staff_cut_check@R1": 1.0}, before_p=0.1))) == 1


def test_the_check_is_per_chair():
    assert len(sessions(run({**P, "ce_staff_cut_check@R9": 1.0}))) == 1
