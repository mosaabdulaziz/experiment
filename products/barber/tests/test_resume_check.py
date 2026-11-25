"""ce_resume_check (2026-10-07, dev/chairrun/same_person/FINDINGS.md): a chair taken again within
ce_resume_max_s of its haircut's end asks re-ID -- every sample of the new sitter since he sat down against
every sample of the previous customer's last ce_resume_ref_s. Within ce_resume_window_s: his, unless
sim < ce_resume_sim_low; after it: his only if sim >= ce_resume_sim_high (then the finished haircut reopens)."""
import math

from chairs2 import SitEnd, SitStart
from domain import Appearance, EngineUpdate, FINISHED, WASHING
from resume_check import decide, ref_window, similarity

from test_haircuts import NO_OCC, P, T0, book

ON = {**P, "ce_resume_check": 1.0}
A, B = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def vec(sim):
    """A unit vector whose cosine to A is sim."""
    return (sim, math.sqrt(1 - sim * sim), 0.0)


def app(ts, reid, tid=4, staff_p=0.1):
    return Appearance("barber-r", tid, ts, reid, staff_p)


def sit(b, start, end, reid, tid=4, chair="R3", policy=ON):
    """One sitting [start, end] with the sitter's re-ID every 10 s, ticked as it runs."""
    u = EngineUpdate()
    b.on_start(SitStart(chair, start, tid), u, policy)
    t = start
    while t < end:
        b.on_appearance(app(t, reid, tid), chair, policy, u)
        b.tick(t, lambda ch: True, u, policy, lambda ch: t)
        t += 10
    b.on_end(SitEnd(chair, end), u, policy)
    return u


def test_similarity_is_the_cosine_of_the_mean_of_every_sample():
    assert similarity([(0, A), (1, A)], [(0, A)]) == 1.0
    assert abs(similarity([(0, A)], [(0, B)])) < 1e-6
    assert similarity([], [(0, A)]) is None
    # no picking: a single odd sample among many moves the mean only a little (3 picked could be 1 in 3)
    many = [(t, A) for t in range(20)] + [(20, B)]
    assert similarity(many, [(0, A)]) > 0.99


def test_the_reference_window_is_his_last_minute_in_the_chair():
    reid = [(float(t), B) for t in range(0, 540, 10)] + [(float(t), A) for t in range(540, 600, 10)]
    assert {r for _, r in ref_window(reid, 0.0, 600.0, 60.0)} == {A}
    assert ref_window(reid, 0.0, 600.0, 60.0)[0][0] == 540.0
    assert ref_window(reid, 580.0, 600.0, 60.0)[0][0] == 580.0       # not before this sitting's start
    assert [t for t, _ in ref_window([(10.0, A), (20.0, A)], 0.0, 600.0, 60.0)] == [10.0, 20.0]  # < 2: his last 2


def test_decide_two_levels():
    p = ON
    assert decide(100, 0.40, p) is False                  # within 6 min, clearly someone else
    assert decide(100, 0.70, p) is True
    assert decide(100, None, p) is True                   # no evidence: today's time rule
    assert decide(900, 0.80, p) is False                  # 6-30 min: must look like him
    assert decide(900, 0.90, p) is True
    assert decide(900, None, p) is False
    assert decide(2000, 0.99, p) is False                 # empty for longer: a new customer


def test_off_is_the_time_rule():
    b = book()
    sit(b, T0, T0 + 600, A, policy=P)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, P)
    assert b.open["R3"].id == "h1" and len(b.open["R3"].sittings) == 2


def test_within_the_window_a_different_looking_sitter_is_a_new_customer():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = sit(b, T0 + 700, T0 + 760, vec(0.40), tid=9)
    assert b.open["R3"].id == "h2" and b.open["R3"].start == T0 + 700
    assert u.sessions["h1"].open is False and u.visits["h1"].state == FINISHED


def test_within_the_window_a_similar_sitter_resumes():
    b = book()
    sit(b, T0, T0 + 600, A)
    sit(b, T0 + 700, T0 + 760, vec(0.70), tid=9)
    h = b.open["R3"]
    assert h.id == "h1" and [s[0] for s in h.sittings] == [T0, T0 + 700]


def test_the_decision_waits_for_the_new_sitters_samples():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON)
    assert len(b.open["R3"].sittings) == 1 and b.pending["R3"]
    b.on_appearance(app(T0 + 705, vec(0.4), 9), "R3", ON, u)
    b.tick(T0 + 720, lambda ch: True, u, ON, lambda ch: T0 + 720)
    assert b.pending["R3"]                                 # one sample, 20 s: not yet
    b.on_appearance(app(T0 + 725, vec(0.4), 9), "R3", ON, u)
    b.tick(T0 + 731, lambda ch: True, u, ON, lambda ch: T0 + 731)
    assert "R3" not in b.pending and b.open["R3"].id == "h2"


def test_no_samples_by_the_deadline_falls_back_to_the_time_rule():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON)
    b.tick(T0 + 700 + 151, lambda ch: True, u, ON, lambda ch: T0 + 851)
    assert "R3" not in b.pending and b.open["R3"].id == "h1"


def test_after_the_window_a_close_match_reopens_the_finished_haircut():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.tick(T0 + 600 + 361, lambda ch: False, u, ON, NO_OCC)
    assert "R3" not in b.open and u.visits["h1"].state == FINISHED
    u = sit(b, T0 + 1500, T0 + 1560, vec(0.90), tid=9)
    h = b.open["R3"]
    assert h.id == "h1" and len(h.sittings) == 2
    assert u.visits["h1"].open and u.visits["h1"].state == WASHING and u.visits["h1"].exited_at is None


def test_after_the_window_a_middling_match_is_a_new_customer():
    b = book()
    sit(b, T0, T0 + 600, A)
    b.tick(T0 + 600 + 361, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    sit(b, T0 + 1500, T0 + 1560, vec(0.76), tid=9)
    assert b.open["R3"].id != "h1"


def test_after_the_long_window_no_reid_is_asked():
    b = book()
    sit(b, T0, T0 + 600, A)
    b.tick(T0 + 600 + 1801, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 2500, 9), u, ON)
    assert not b.pending and b.open["R3"].id != "h1"


def test_the_buffer_keeps_five_minutes_and_skips_staff():
    b = book()
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0, 4), u, ON)
    b.on_appearance(app(T0 + 10, B), "R3", ON, u)               # early: dropped from the last 5 min
    for t in range(400, 600, 20):
        b.on_appearance(app(T0 + t, A), "R3", ON, u)
    b.on_appearance(app(T0 + 590, B, staff_p=0.95), "R3", ON, u)  # the barber leaning in
    ref = b.open["R3"].reid
    assert all(r == A for _, r in ref) and ref[0][0] >= T0 + 290


def test_a_sitting_ending_while_pending_is_decided_at_once():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON)
    b.on_appearance(app(T0 + 705, vec(0.4), 9), "R3", ON, u)
    b.on_appearance(app(T0 + 708, vec(0.4), 9), "R3", ON, u)
    b.on_end(SitEnd("R3", T0 + 712), u, ON)
    assert not b.pending and b.open["R3"].id == "h2" and b.open["R3"].sittings == [[T0 + 700, T0 + 712]]


def test_clear_drops_pending_and_recent():
    b = book()
    sit(b, T0, T0 + 600, A)
    b.tick(T0 + 961, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    b.on_start(SitStart("R3", T0 + 1000, 9), EngineUpdate(), ON)
    b.clear()
    assert not b.pending and not b.recent and not b.open


def test_a_return_compared_with_his_last_minute_resumes_though_he_looked_different_earlier():
    """Caped for most of the haircut (B), cape off in the last minute (A), back from the wash as A."""
    b = book()
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0, 4), u, ON)
    for t in range(0, 600, 10):
        b.on_appearance(app(T0 + t, B if t < 540 else A), "R3", ON, u)
        b.tick(T0 + t, lambda ch: True, u, ON, lambda ch: T0 + t)
    b.on_end(SitEnd("R3", T0 + 600), u, ON)
    sit(b, T0 + 700, T0 + 760, A, tid=9)
    assert b.open["R3"].id == "h1" and len(b.open["R3"].sittings) == 2


def test_samples_from_the_start_chain_seed_the_check():
    """The start is backdated 200 s: the sitter's samples from then on are the evidence, not nothing."""
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    for t in range(700, 900, 10):                                  # chair still starting: no sitting runs
        b.on_appearance(app(T0 + t, vec(0.4), 9), "R3", ON, u)
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON, now_ts=T0 + 900)
    assert len(b.pending["R3"].samples) == 20
    b.tick(T0 + 900, lambda ch: True, u, ON, lambda ch: T0 + 900)
    assert "R3" not in b.pending and b.open["R3"].id != "h1"       # decided on the seeded samples: someone else


def test_seed_samples_before_the_backdated_start_are_not_his():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_appearance(app(T0 + 650, B, 7), "R3", ON, u)              # someone passing through before he sat
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON, now_ts=T0 + 712)
    assert b.pending["R3"].samples == []


def test_the_deadline_counts_from_when_the_start_was_seen():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON, now_ts=T0 + 900)  # backdated 200 s, no samples yet
    b.tick(T0 + 900, lambda ch: True, u, ON, lambda ch: T0 + 900)
    assert "R3" in b.pending                                        # not decided on nothing
    b.tick(T0 + 900 + 151, lambda ch: True, u, ON, lambda ch: T0 + 1051)
    assert "R3" not in b.pending


def test_a_short_stray_sitter_judged_new_does_not_lock_the_customer_out():
    b = book()
    sit(b, T0, T0 + 600, A)                                         # h1, confirmed, goes to the wash
    sit(b, T0 + 660, T0 + 685, vec(0.3), tid=7)                      # someone else, 25 s: "new", never confirmed
    stray = b.open["R3"]
    assert stray.id != "h1" and not stray.confirmed
    u = sit(b, T0 + 900, T0 + 960, A, tid=9)                        # 300 s after h1 ended: he is back
    h = b.open["R3"]
    assert h.id == "h1" and [s[0] for s in h.sittings] == [T0, T0 + 900]
    assert stray.id not in u.sessions                               # the stray sit never became a haircut


def test_a_new_haircut_confirmed_in_the_chair_ends_the_previous_ones_reopening():
    b = book()
    sit(b, T0, T0 + 600, A)
    sit(b, T0 + 660, T0 + 800, vec(0.3), tid=7)                      # someone else, confirmed (> 60 s)
    assert b.open["R3"].confirmed and "R3" not in b.recent


def test_forget_drops_a_removed_chairs_checks():
    b = book()
    sit(b, T0, T0 + 600, A)
    b.tick(T0 + 961, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    b.on_appearance(app(T0 + 990, A, 9), "R3", ON, EngineUpdate())
    b.on_start(SitStart("R3", T0 + 1000, 9), EngineUpdate(), ON)
    b.forget({"R3"})
    assert not b.pending and not b.recent and not b.seat_samples


def test_a_reopened_haircut_gets_a_fresh_alert_from_the_reopen():
    pol = {**ON, "service_target_s": 900.0}
    b = book()
    u = sit(b, T0, T0 + 1000, A, policy=pol)                        # excessive: alert at 900 s, closed later
    b.tick(T0 + 1000 + 361, lambda ch: False, u, pol, NO_OCC)
    first = [a for a in u.alerts if not a.open]
    assert first and first[-1].ended_at.timestamp() == T0 + 1000
    u = sit(b, T0 + 2000, T0 + 2060, vec(0.9), tid=9, policy=pol)  # reopened 1000 s later
    opened = [a for a in u.alerts if a.open]
    assert b.open["R3"].id == "h1" and opened
    assert opened[0].id != first[-1].id and opened[0].started_at.timestamp() == T0 + 2000


# ── the decision, for the dashboard (2026-10-18, user: say resumed / new customer, with the score) ─────────────

def test_a_resume_is_recorded_with_its_score_and_threshold():
    b = book()
    sit(b, T0, T0 + 600, A)
    sit(b, T0 + 700, T0 + 760, vec(0.70), tid=9)
    d = b.decisions["R3"]
    assert (d.resumed, d.haircut_id, d.sitting_n, d.gap_s) == (True, "h1", 1, 100.0)
    assert abs(d.sim - 0.70) < 1e-6 and d.threshold == ON["ce_resume_sim_low"]
    assert d.at >= T0 + 700 + ON["ce_resume_new_s"] and d.prev_ticket == b.open["R3"].ticket


def test_a_new_customer_is_recorded_too():
    b = book()
    sit(b, T0, T0 + 600, A)
    ticket = b.open["R3"].ticket
    sit(b, T0 + 700, T0 + 760, vec(0.40), tid=9)
    d = b.decisions["R3"]
    assert (d.resumed, d.haircut_id, d.sitting_n, d.prev_ticket) == (False, "h2", 0, ticket)
    assert abs(d.sim - 0.40) < 1e-6


def test_after_the_wash_window_the_high_threshold_applied():
    b = book()
    sit(b, T0, T0 + 600, A)
    b.tick(T0 + 600 + 361, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    sit(b, T0 + 1500, T0 + 1560, vec(0.90), tid=9)
    d = b.decisions["R3"]
    assert d.resumed and d.threshold == ON["ce_resume_sim_high"] and d.gap_s == 900.0


def test_no_re_id_records_the_time_rule_without_a_score():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0 + 700, 9), u, ON)
    b.tick(T0 + 851, lambda ch: True, u, ON, lambda ch: T0 + 851)
    d = b.decisions["R3"]
    assert d.resumed and d.sim is None and d.at == T0 + 851


def test_a_start_with_no_check_drops_the_old_decision():
    b = book()
    sit(b, T0, T0 + 600, A)
    sit(b, T0 + 700, T0 + 760, vec(0.70), tid=9)
    assert "R3" in b.decisions
    b.tick(T0 + 760 + 1801, lambda ch: False, EngineUpdate(), ON, NO_OCC)
    b.on_start(SitStart("R3", T0 + 3000, 11), EngineUpdate(), ON)             # long after: no re-ID asked
    assert "R3" not in b.decisions


def test_clear_and_forget_drop_the_decisions():
    b = book()
    sit(b, T0, T0 + 600, A)
    sit(b, T0 + 700, T0 + 760, vec(0.70), tid=9)
    b.forget({"R3"})
    assert not b.decisions
    sit(b, T0 + 800, T0 + 860, vec(0.70), tid=9)
    b.clear()
    assert not b.decisions


def test_the_thresholds_are_read_per_chair():
    """2026-10-18 (user): ce_resume_sim_low / _high per chair -- 0.55 resumes on R3 what the default 0.65 calls new."""
    loose = {**ON, "ce_resume_sim_low@R3": 0.5}
    b = book()
    sit(b, T0, T0 + 600, A, policy=loose)
    sit(b, T0 + 700, T0 + 760, vec(0.55), tid=9, policy=loose)
    assert b.open["R3"].id == "h1" and b.decisions["R3"].threshold == 0.5
    b = book()
    sit(b, T0, T0 + 600, A, chair="R4", policy=loose)
    sit(b, T0 + 700, T0 + 760, vec(0.55), tid=9, chair="R4", policy=loose)
    assert b.open["R4"].id != "h1" and b.decisions["R4"].threshold == ON["ce_resume_sim_low"]


# ── reseat: the seat taken again after cover only (2026-10-18, Palm Hills L6) ─────────────────────────────────

from chairs2 import Reseat  # noqa: E402


def reseat(b, last_occ, at, until, reid, tid=9, chair="R3", policy=ON):
    """A sitting since T0 with his re-ID (A) until last_occ, cover only, then someone seen from `at` until `until`."""
    u = EngineUpdate()
    b.on_start(SitStart(chair, T0, 4), u, policy)
    t = T0
    while t <= last_occ:
        b.on_appearance(app(t, A, 4), chair, policy, u)
        b.tick(t, lambda ch: True, u, policy, lambda ch: t)
        t += 10
    b.on_reseat(Reseat(chair, at, tid, last_occ), u, policy, at)
    t = at
    while t < until:
        b.on_appearance(app(t, reid, tid), chair, policy, u)
        b.tick(t, lambda ch: True, u, policy, lambda ch: t)
        t += 5
    return u


def test_the_same_customer_after_cover_keeps_his_haircut():
    b = book()
    u = reseat(b, T0 + 600, T0 + 660, T0 + 720, vec(0.9))
    h = b.open["R3"]
    assert h.id == "h1" and len(h.sittings) == 1 and h.last_end is None and not b.pending
    d = b.decisions["R3"]
    assert d.kind == "reseat" and d.resumed and abs(d.sim - 0.9) < 1e-6 and d.gap_s == 60.0
    (r,) = u.resume_checks
    assert (r.chair_id, r.kind, r.resumed, r.haircut_id, r.prev_haircut_id) == ("R3", "reseat", True, "h1", "h1")


def test_someone_else_after_cover_ends_the_haircut_when_its_customer_was_last_seen():
    b = book()
    u = reseat(b, T0 + 600, T0 + 660, T0 + 720, vec(0.3))
    h = b.open["R3"]
    assert h.id == "h2" and h.start == T0 + 660 and h.last_end is None
    assert u.sessions["h1"].open is False and u.sessions["h1"].ended_at.timestamp() == T0 + 600
    d = b.decisions["R3"]
    assert d.kind == "reseat" and not d.resumed and d.haircut_id == "h2"
    assert [(r.resumed, r.haircut_id, r.prev_haircut_id) for r in u.resume_checks] == [(False, "h2", "h1")]


def test_a_reseat_is_decided_on_the_chairs_own_threshold():
    b = book()
    reseat(b, T0 + 600, T0 + 660, T0 + 720, vec(0.3), policy={**ON, "ce_resume_sim_low@R3": 0.2})
    assert b.open["R3"].id == "h1" and b.decisions["R3"].threshold == 0.2


def test_every_return_check_goes_to_storage_too():
    b = book()
    sit(b, T0, T0 + 600, A)
    u = sit(b, T0 + 700, T0 + 760, vec(0.40), tid=9)
    (r,) = u.resume_checks
    assert (r.kind, r.resumed, r.haircut_id, r.prev_haircut_id, r.gap_s) == ("return", False, "h2", "h1", 100.0)
    assert abs(r.sim - 0.40) < 1e-6 and r.threshold == ON["ce_resume_sim_low"] and r.new_samples >= 2


# ce_min_after_new_s (2026-11-10, user; Palm Hills 16 Oct): a customer back in his chair whom re-ID calls new opens a
# second haircut that counts once he sits ce_min_haircut_s -- R1 18:20 sat 6.3 min after a 0.52 "new". A haircut
# opened by a "new customer" verdict needs this longer minimum; every other start keeps ce_min_haircut_s.
AFTER_NEW = {**ON, "ce_min_after_new_s": 420.0}


def test_a_sitter_judged_new_is_not_a_haircut_before_the_longer_minimum():
    b = book()
    sit(b, T0, T0 + 600, A, policy=AFTER_NEW)
    u = sit(b, T0 + 700, T0 + 1000, vec(0.40), tid=9, policy=AFTER_NEW)     # "new customer", sits 300 s
    h = b.open["R3"]
    assert h.id == "h2" and not h.confirmed and "h2" not in u.sessions


def test_a_sitter_judged_new_is_a_haircut_once_he_reaches_the_longer_minimum():
    b = book()
    sit(b, T0, T0 + 600, A, policy=AFTER_NEW)
    sit(b, T0 + 700, T0 + 1150, vec(0.40), tid=9, policy=AFTER_NEW)          # sits 450 s
    assert b.open["R3"].id == "h2" and b.open["R3"].confirmed


def test_a_start_with_no_check_keeps_the_usual_minimum():
    b = book()
    sit(b, T0, T0 + 120, A, policy=AFTER_NEW)                                # the chair's first sitter, 120 s
    assert b.open["R3"].id == "h1" and b.open["R3"].confirmed


def test_a_resumed_customer_keeps_the_usual_minimum():
    b = book()
    sit(b, T0, T0 + 40, A, policy=AFTER_NEW)                                 # 40 s: not yet a haircut
    sit(b, T0 + 100, T0 + 140, vec(0.90), tid=9, policy=AFTER_NEW)           # back, "resume": 80 s in all
    assert b.open["R3"].id == "h1" and b.open["R3"].confirmed
