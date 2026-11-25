from chairs2 import ChairStates, Reseat, SitEnd, SitStart
from policy import DEFAULT_POLICY
from seat import ChairSignal

P = dict(DEFAULT_POLICY)
OCC, COVER, NONE = ChairSignal(True, False, 7), ChairSignal(False, True, None), ChairSignal(False, False, None)


def feed(cs, seq, t0=1000.0, step=0.5):
    """seq: list of (seconds, signal) runs; returns all events."""
    ev, t = [], t0
    for dur, sig in seq:
        for _ in range(int(dur / step)):
            ev += cs.step("R3", t, sig, P)
            t += step
    return ev


def test_one_sitting_starts_backdated_and_ends_at_last_occupied():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(5, NONE), (120, OCC), (90, NONE)])
    assert ev == [SitStart("R3", 1005.0, 7), SitEnd("R3", 1124.5)]
    assert cs.state("R3") == "empty"


def test_short_sit_never_starts():
    cs = ChairStates(["R3"])
    assert feed(cs, [(6, OCC), (30, NONE)]) == []


def test_flicker_within_start_window_still_starts():
    cs = ChairStates(["R3"])
    seq = [(3, OCC), (1, NONE)] * 4 + [(30, OCC)]       # 8 of the first 10 s occupied
    (s,) = [e for e in feed(cs, seq) if isinstance(e, SitStart)]
    assert s.ts == 1000.0


def test_barber_hiding_50s_keeps_70s_ends():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(60, OCC), (50, NONE), (60, OCC), (70, NONE)])
    assert [type(e) for e in ev] == [SitStart, SitEnd]


def test_staff_cover_keeps_a_sitting_but_never_starts_one():
    cs = ChairStates(["R3"])
    assert feed(cs, [(60, COVER)]) == []
    ev = feed(ChairStates(["R3"]), [(60, OCC), (200, COVER), (70, NONE)])
    assert ev == [SitStart("R3", 1000.0, 7), SitEnd("R3", 1059.5)]


def test_maskless_frames_do_not_end_a_sitting():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(60, OCC), (50, NONE), (60, OCC)])
    assert [type(e) for e in ev] == [SitStart]
    assert cs.state("R3") == "sitting"


def test_camera_dropout_pauses_the_clock():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(60, OCC)])
    ev += cs.step("R3", 1000.0 + 600, OCC, P)        # 9 minutes with no frames at all, then seated
    ev += feed(cs, [(30, OCC)], t0=1600.5)
    assert [type(e) for e in ev] == [SitStart]
    assert cs.since("R3") == 1000.0


def test_no_view_after_dropout():
    cs = ChairStates(["R3"])
    feed(cs, [(20, OCC)])
    assert not cs.no_view("R3", 1021.0, P)
    assert cs.no_view("R3", 1100.0, P)


def test_dropout_in_starting_restarts_window():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(3, OCC)])                           # window starts at t=1000.0
    ev += cs.step("R3", 1000.0 + 600, OCC, P)          # 600s gap (> 5s dropout), discard window, start fresh
    ev += feed(cs, [(5, OCC)], t0=1600.5)               # continue from new window start at t0+600
    assert ev == []                                     # no SitStart yet (only 5.5s occupied, need 7s for 70% of 10s)
    assert cs.state("R3") == "starting"


def test_dropout_in_starting_with_none_returns_to_empty():
    cs = ChairStates(["R3"])
    ev = feed(cs, [(3, OCC)])                           # window starts at t=1000.0
    ev += cs.step("R3", 1000.0 + 600, NONE, P)         # 600s gap (> 5s dropout), discard window, state=EMPTY, then handle NONE
    assert ev == []
    assert cs.state("R3") == "empty"


def test_dropout_pause_keeps_sitting_active():
    # (60, OCC) with step=0.5 gives 120 frames ending at t0+59.5 = 1059.5
    cs = ChairStates(["R3"])
    ev = feed(cs, [(60, OCC)])                          # last_occ=1059.5, last_evidence=1059.5
    # Jump to 1600.0: gap=540.5s, pause clock: last_evidence = 1059.5 + 540.5 = 1600.0
    ev += cs.step("R3", 1600.0, NONE, P)
    # NONE for 30s from t0=1600.5: 60 frames, last at 1600.5 + 29.5 = 1630.0
    # Gap 1630.0 - 1600.0 = 30s < 60s, no end
    ev += feed(cs, [(30, NONE)], t0=1600.5)
    assert not any(isinstance(e, SitEnd) for e in ev)
    # NONE for 40s from t0=1630.5: 80 frames, last at 1630.5 + 39.5 = 1670.0
    # Gap 1670.0 - 1600.0 = 70s > 60s, end with SitEnd(1059.5)
    ev += feed(cs, [(40, NONE)], t0=1630.5)
    ends = [e for e in ev if isinstance(e, SitEnd)]
    assert len(ends) == 1
    assert ends[0].ts == 1059.5


def run_seq(pol, seq, t0=1000.0, step=0.5):
    """seq: (seconds, signal) runs, or a number = seconds with no frames at all. Returns [(ts, event)]."""
    cs, out, t = ChairStates(["R3"]), [], t0
    for dur, sig in seq:
        if sig is None:
            t += dur
            continue
        for _ in range(int(dur / step)):
            out += [(t, e) for e in cs.step("R3", t, sig, pol)]
            t += step
    return out


def test_carry_cap_ends_a_sitting_kept_alive_by_cover_alone():
    # 2026-10-04 (Task 14b): 60 s occupied (last occupied frame 1059.5), then cover forever.
    # Cap 180: cover counts while ts - 1059.5 <= 180, so the last accepted cover frame is 1239.5;
    # the end gap (60 s) then runs out at the first frame with ts - 1239.5 > 60 = 1300.0.
    seq = [(60, OCC), (400, COVER)]
    ends = [(t, e) for t, e in run_seq({**P, "ce_carry_max_s": 180.0}, seq) if isinstance(e, SitEnd)]
    assert ends == [(1300.0, SitEnd("R3", 1059.5))]
    assert [e for _, e in run_seq({**P, "ce_carry_max_s": 0.0}, seq) if isinstance(e, SitEnd)] == []


def test_carry_cap_clock_pauses_over_a_camera_dropout():
    # cover 1060-1159.5, no frames until 1210.0 (a 50.5 s gap > ce_dropout_s): the pause must not eat
    # into the cap, so it runs out 50.5 s later than without a dropout: the last accepted cover
    # frame is 1239.5 + 50.5 = 1290.0 and the end fires at the first frame past 60 s, 1350.5.
    seq = [(60, OCC), (100, COVER), (50, None), (300, COVER)]
    ends = [(t, e) for t, e in run_seq({**P, "ce_carry_max_s": 180.0}, seq) if isinstance(e, SitEnd)]
    assert ends == [(1350.5, SitEnd("R3", 1059.5))]


def test_carry_cap_zero_is_no_cap():
    cs, ev, t = ChairStates(["R3"]), [], 1000.0
    for dur, sig in [(60, OCC), (300, COVER), (30, OCC), (100, NONE)]:
        for _ in range(int(dur * 2)):
            ev += cs.step("R3", t, sig, {**P, "ce_carry_max_s": 0.0})
            t += 0.5
    assert [e for e in ev if isinstance(e, SitEnd)] == [SitEnd("R3", 1389.5)]


def test_default_cap_ends_a_cover_only_stretch_three_minutes_after_the_customer():
    # 2026-10-12 (user: away shows too late): cover counts 120 s after the last occupied frame (1059.5), then the
    # 60 s end gap -> away at 1240, not 1300 as with 180.
    ends = [(t, e) for t, e in run_seq(P, [(60, OCC), (400, COVER)]) if isinstance(e, SitEnd)]
    assert ends == [(1240.0, SitEnd("R3", 1059.5))]


P180 = {**P, "ce_carry_max_s": 180.0}     # the refresh-rule cases below were timed against the 180 s cap


def test_lone_occupied_frame_after_a_long_cover_stretch_does_not_refresh_the_sitting():
    # 2026-10-04 (Task 16c, live R4-2): the barber serving the reclined R3 customer stands in R4 (cover) and his
    # track briefly carries a customer's staff_p (id switch) or none yet -> one lone "occupied" frame 100 s
    # after the last real one. It must not move the end or restart the carry cap: same end as pure cover.
    seq = [(60, OCC), (100, COVER), (0.5, OCC), (300, COVER)]
    ends = [(t, e) for t, e in run_seq(P180, seq) if isinstance(e, SitEnd)]
    assert ends == [(1300.0, SitEnd("R3", 1059.5))]


def test_occupied_run_after_a_long_cover_stretch_refreshes_the_sitting():
    # three consecutive occupied frames (1160.0-1161.0) are a seated person again: last occupied 1161.0, the cap
    # runs from there (last accepted cover 1341.0), the end fires at 1401.5.
    seq = [(60, OCC), (100, COVER), (1.5, OCC), (300, COVER)]
    ends = [(t, e) for t, e in run_seq(P180, seq) if isinstance(e, SitEnd)]
    assert ends == [(1401.5, SitEnd("R3", 1161.0))]


def test_lone_occupied_frame_soon_after_the_last_one_still_refreshes():
    seq = [(60, OCC), (20, COVER), (0.5, OCC), (300, COVER)]       # lone frame at 1080.0, 20.5 s after 1059.5
    ends = [(t, e) for t, e in run_seq(P180, seq) if isinstance(e, SitEnd)]
    assert ends == [(1320.5, SitEnd("R3", 1080.0))]


def test_refresh_run_of_one_is_todays_behaviour():
    seq = [(60, OCC), (100, COVER), (0.5, OCC), (300, COVER)]
    ends = [(t, e) for t, e in run_seq({**P180, "ce_refresh_run_n": 1.0}, seq) if isinstance(e, SitEnd)]
    assert ends == [(1400.5, SitEnd("R3", 1160.0))]


def test_a_zero_start_window_never_raises():
    """I-2: the API refuses 0 now, but a stored 0 must not crash the consumer every cycle."""
    cs = ChairStates(["R3"])
    pol = {**P, "ce_start_window_s": 0.0}
    ev = []
    t = 1000.0
    for _ in range(10):
        ev += cs.step("R3", t, OCC, pol)
        t += 0.5
    assert [type(e) for e in ev] == [SitStart]


# 2026-10-05 (haircut accuracy, start chain): occupied frames less than ce_start_chain_s apart while the chair is
# empty or starting form a chain; a confirmed start is backdated to the chain's first frame.
FLICKER_THEN_SIT = [(3, OCC), (20, NONE), (3, OCC), (20, NONE), (30, OCC)]   # runs at 1000, 1023, 1046


def _starts(pol, seq):
    return [(t, e) for t, e in run_seq(pol, seq) if isinstance(e, SitStart)]


def test_start_chain_backdates_a_confirmed_start_to_the_chain_first_frame():
    ((t, s),) = _starts({**P, "ce_start_chain_s": 45.0}, FLICKER_THEN_SIT)
    assert s == SitStart("R3", 1000.0, 7) and t > 1046.0


def test_start_chain_breaks_on_a_gap_of_at_least_the_chain():
    ((_, s),) = _starts({**P, "ce_start_chain_s": 20.0}, FLICKER_THEN_SIT)   # gaps of 20.5 s
    assert s.ts == 1046.0


def test_start_chain_off_is_the_old_start():
    ((_, s),) = _starts({**P, "ce_start_chain_s": 0.0}, FLICKER_THEN_SIT)
    assert s.ts == 1046.0


def test_start_chain_is_live_until_the_gap_runs_out():
    pol = {**P, "ce_start_chain_s": 45.0}
    cs = ChairStates(["R3"])
    t = 1000.0
    for _ in range(6):                                  # 1000.0-1002.5 occupied: too short to start
        cs.step("R3", t, OCC, pol)
        t += 0.5
    for _ in range(40):                                 # empty again from 1003.0
        cs.step("R3", t, NONE, pol)
        t += 0.5
    assert cs.state("R3") == "empty"
    assert cs.chain_live("R3", 1047.5, pol) and not cs.chain_live("R3", 1048.0, pol)
    assert not cs.chain_live("R3", 1010.0, {**P, "ce_start_chain_s": 0.0})


def test_a_confirmed_start_ends_the_chain():
    pol = {**P, "ce_start_chain_s": 45.0}
    cs = ChairStates(["R3"])
    for i in range(40):
        cs.step("R3", 1000.0 + i * 0.5, OCC, pol)
    assert cs.state("R3") == "sitting" and not cs.chain_live("R3", 1020.0, pol)


def test_an_over_age_chain_is_dead_and_never_backdates():
    """Review fix 1: a chain older than ce_start_chain_max_s (first frame to now) neither counts as live nor
    backdates; it stays dead until a gap of ce_start_chain_s breaks it."""
    pol = {**P, "ce_start_chain_s": 45.0, "ce_start_chain_max_s": 240.0}
    cs, ev = ChairStates(["R3"]), []
    for k in range(40):                                 # lone occupied frames 30 s apart, 1000 .. 2170
        t = 1000.0 + 30 * k
        ev += cs.step("R3", t, OCC, pol)
        ev += cs.step("R3", t + 15, NONE, pol)
    assert cs.chain_live("R3", 1000.0 + 30 * 7, pol)            # 210 s old: live
    assert not cs.chain_live("R3", 1000.0 + 30 * 9, pol)        # 270 s old: dead
    t = 2200.0
    for i in range(60):                                 # a real sit, still within 45 s of the last stray
        ev += cs.step("R3", t + i * 0.5, OCC, pol)
    (s,) = [e for e in ev if isinstance(e, SitStart)]
    assert s.ts == 2200.0


def test_start_chain_survives_a_camera_dropout_shorter_than_the_gap():
    """Review fix 3: a dropout ends nothing and starts nothing -- the chain is kept across it (its gap and age are
    plain frame time; a dropout longer than ce_start_chain_s breaks it like any gap)."""
    pol = {**P, "ce_start_chain_s": 45.0}
    seq = [(3, OCC), (20, None), (30, OCC)]             # 1000-1002.5 seen, no frames for 20 s, seated from 1023
    ((_, s),) = _starts(pol, seq)
    assert s.ts == 1000.0
    ((_, s),) = _starts(pol, [(3, OCC), (50, None), (30, OCC)])
    assert s.ts == 1053.0


# 2026-10-05 (far-chair per-chair rules): "<key>@<chair>" start window / frac / end gap for one chair only.
def _feed2(pol, seq, chairs=("R3", "R4"), t0=1000.0, step=0.5):
    cs, ev, t = ChairStates(list(chairs)), {c: [] for c in chairs}, t0
    for dur, sig in seq:
        for _ in range(int(dur / step)):
            for c in chairs:
                ev[c] += cs.step(c, t, sig, pol)
            t += step
    return cs, ev


def test_end_gap_override_applies_only_to_its_chair():
    pol = {**P, "ce_end_gap_s@R3": 200.0}
    cs, ev = _feed2(pol, [(120, OCC), (120, NONE)])
    assert ev["R4"] == [SitStart("R4", 1000.0, 7), SitEnd("R4", 1119.5)]
    assert ev["R3"] == [SitStart("R3", 1000.0, 7)] and cs.state("R3") == "sitting"


def test_start_window_override_applies_only_to_its_chair():
    pol = {**P, "ce_start_chain_s": 0.0, "ce_start_window_s@R3": 30.0, "ce_start_frac@R3": 0.30}
    cs, ev = _feed2(pol, [(2, OCC), (3, NONE)] * 12)        # 40 % occupied, never 70 % of 10 s
    assert [type(e) for e in ev["R3"]] == [SitStart] and ev["R3"][0].ts == 1000.0
    assert ev["R4"] == []


def test_cover_cap_override_applies_only_to_its_chair():
    """2026-10-18 (user): L6's barber hides his customer for minutes; past the 120 s cap the haircut went away and had
    to resume. A longer cap on that chair keeps it sitting under cover; the other chair still ends."""
    pol = {**P, "ce_carry_max_s@R3": 600.0}
    cs, ev = _feed2(pol, [(60, OCC), (300, COVER)])
    assert cs.state("R3") == "sitting" and [type(e) for e in ev["R3"]] == [SitStart]
    assert [type(e) for e in ev["R4"]] == [SitStart, SitEnd] and ev["R4"][1].ts == 1059.5


# ── reseat (2026-10-18, ce_reseat_check_s: Palm Hills L6, the barber cleaning the empty chair held the sitting) ──

def _reseat_feed(seq, **pol):
    cs = ChairStates(["R3"])
    ev, t = [], 1000.0
    p = {**P, **pol}
    for dur, sig in seq:
        for _ in range(int(dur / 0.5)):
            ev += cs.step("R3", t, sig, p)
            t += 0.5
    return ev


def test_the_seat_taken_again_after_cover_only_is_a_reseat():
    ev = _reseat_feed([(60, OCC), (40, COVER), (30, OCC)], ce_reseat_check_s=30.0)
    # 40.5 s without one: only the 3rd occupied frame in a row counts (ce_refresh_run_n) -- a lone stray is no reseat
    assert [e for e in ev if isinstance(e, Reseat)] == [Reseat("R3", 1101.0, 7, 1059.5)]
    assert not [e for e in ev if isinstance(e, SitEnd)]                    # the sitting itself runs on


def test_a_short_cover_stretch_is_no_reseat_and_off_by_default():
    assert not [e for e in _reseat_feed([(60, OCC), (20, COVER), (30, OCC)], ce_reseat_check_s=30.0)
                if isinstance(e, Reseat)]
    assert not [e for e in _reseat_feed([(60, OCC), (40, COVER), (30, OCC)]) if isinstance(e, Reseat)]


def test_the_reseat_check_turns_on_per_chair():
    """2026-10-18 (user): off for every chair; on (with its own stretch) only where a chair's key sets it."""
    p = {**P, "ce_reseat_check_s@R3": 30.0}
    cs = ChairStates(["R3", "R4"])
    ev, t = [], 1000.0
    for dur, sig in [(60, OCC), (40, COVER), (30, OCC)]:
        for _ in range(int(dur / 0.5)):
            ev += cs.step("R3", t, sig, p) + cs.step("R4", t, sig, p)
            t += 0.5
    assert [e.chair for e in ev if isinstance(e, Reseat)] == ["R3"]


# ce_cover_min_frac (2026-11-13, user; Palm Hills 6 Nov, L7 21:08): the seat was empty 100 s between two customers, but
# a staff member brushing the zone gave a cover frame now and then (3-14 % of frames) and each one refreshed the end
# gap, so the sitting ran on into the next customer's. Cover counts only when it fills that share of the last
# ce_cover_window_s of frames.
FLICKER = [(0.5, COVER), (4.5, NONE)] * 20            # one cover frame every 5 s for 100 s: 10 % of frames


def test_cover_flickers_do_not_hold_an_empty_seat_when_a_share_is_asked():
    seq = [(60, OCC)] + FLICKER + [(100, NONE)]
    ends = [(t, e) for t, e in run_seq({**P, "ce_cover_min_frac": 0.3}, seq) if isinstance(e, SitEnd)]
    assert ends == [(1120.0, SitEnd("R3", 1059.5))]   # the end gap (60 s) runs from the last occupied frame


def test_cover_flickers_hold_the_seat_by_default():
    seq = [(60, OCC)] + FLICKER + [(100, NONE)]
    ends = [(t, e) for t, e in run_seq(P, seq) if isinstance(e, SitEnd)]
    assert len(ends) == 1 and ends[0][0] > 1200.0     # held through the flickers, as before


def test_steady_cover_still_holds_the_sitting_when_a_share_is_asked():
    seq = [(60, OCC), (100, COVER), (30, OCC), (100, NONE)]
    ends = [e for _, e in run_seq({**P, "ce_cover_min_frac": 0.3}, seq) if isinstance(e, SitEnd)]
    assert ends == [SitEnd("R3", 1189.5)]             # one sitting, ended after the customer's last frame


def test_cover_share_override_applies_only_to_its_chair():
    cs, pol = ChairStates(["R3", "L7"]), {**P, "ce_cover_min_frac@L7": 0.3}
    ends, t = [], 1000.0
    for dur, sig in [(60, OCC)] + FLICKER:
        for _ in range(int(dur / 0.5)):
            for ch in ("R3", "L7"):
                ends += [e.chair for e in cs.step(ch, t, sig, pol) if isinstance(e, SitEnd)]
            t += 0.5
    assert ends == ["L7"]
