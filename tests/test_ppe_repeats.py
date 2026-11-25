"""modules.ppe: fewer repeat alerts for one worker (test2.mp4: 46 alerts for 12 workers).
- TemporalFilter: an open episode closes only after close_after_present_s of unbroken smoothed PRESENT;
- TrackLinks: which lost track a (re)appearing one continues, for the duplicate hint;
- StreamRules.hold_alert / due_alerts: a second item starting on the same track during the hold joins the alert.
The module-level behaviour is in test_ppe_alerts.py and test_ppe_duplicates.py."""
from perception.modules.ppe.compliance.compliance import evaluate_frame
from perception.modules.ppe.compliance.config import ComplianceConfig, TemporalConfig
from perception.modules.ppe.compliance.temporal import TemporalFilter
from perception.modules.ppe.compliance.types import Detection
from perception.modules.ppe.reid import LinkConfig, TrackLinks
from perception.modules.ppe.state import StreamRules

IMG = (1000, 1000)
PERSON = (100, 100, 200, 400)
HEAD = (125, 85, 175, 160)


def _frame(helmet_cls):
    persons, _ = evaluate_frame([Detection("Person", 0.9, PERSON), Detection(helmet_cls, 0.8, HEAD)],
                                ComplianceConfig(), IMG)
    persons[0].track_id = 1
    return persons


def _fired(close_after_present_s, seq, fps=10.0):
    """Times (s) a helmet episode starts, feeding `seq` at `fps`; window 3, absent/present after 2, cooldown 5 s."""
    cfg = TemporalConfig(window=3, min_absent_frames=2, min_present_frames=2, cooldown_s=5,
                         close_after_present_s=close_after_present_s)
    tf = TemporalFilter(cfg, ["helmet"])
    return [round(i / fps, 2) for i, cls in enumerate(seq) if tf.update(_frame(cls), i / fps)[0].new_violations]


# 1 s bare head, then a "helmet" for `on` frames, then bare-headed for 8 s
def _seq(on):
    return ["NO-Hardhat"] * 10 + ["Hardhat"] * on + ["NO-Hardhat"] * 80


def test_with_0_the_first_present_frame_closes_the_episode_as_evaluated():
    assert _fired(0.0, _seq(4)) == [0.1, 5.1]                 # closed, so a new episode once the cooldown is over


def test_a_present_run_shorter_than_the_setting_keeps_the_episode_open():
    assert _fired(2.0, _seq(15)) == [0.1]                     # PRESENT for 1.4 s: still the same episode


def test_a_present_run_of_the_setting_closes_it():
    assert _fired(2.0, _seq(25)) == [0.1, 5.1]                # PRESENT for 2.4 s: closed


def test_the_present_run_must_be_unbroken():
    seq = ["NO-Hardhat"] * 10 + (["Hardhat"] * 12 + ["NO-Hardhat"] * 3) * 2 + ["NO-Hardhat"] * 80
    assert _fired(2.0, seq) == [0.1]                          # two 1 s runs are not 2 s


# --- track links -------------------------------------------------------------------------------------------------

def box(x, y=200, h=100, w=40):
    return (x, y, x + w, y + h)


def links(**kw):
    return TrackLinks(LinkConfig(**kw), forget_after_s=120.0)


def test_a_new_id_near_a_lost_track_continues_it_and_one_far_away_does_not():
    tl = links()
    tl.observe(0.0, [(1, box(100)), (2, box(600))])
    tl.observe(0.5, [(2, box(600))])                          # 1 lost
    tl.observe(1.0, [(2, box(600)), (40, box(200))])          # 1.0 heights from 1: continues it
    assert tl.chain(40) == tl.chain(1) and tl.link(40) == (1, 1.0, 1.0)
    tl.observe(1.5, [(2, box(600)), (40, box(200)), (41, box(400))])   # 41: nothing lost near it
    assert tl.chain(41) not in (tl.chain(1), tl.chain(2)) and tl.link(41) is None


def test_a_track_still_in_the_frame_is_never_continued():
    tl = links()
    tl.observe(0.0, [(1, box(100))])
    tl.observe(0.1, [(1, box(100)), (40, box(110))])          # two people side by side
    assert tl.chain(40) != tl.chain(1)


def test_gap_and_distance_limits():
    for gap, x, linked in [(4.0, 100, True), (4.1, 100, False), (1.0, 220, True), (1.0, 225, False)]:
        tl = links()
        tl.observe(0.0, [(1, box(100))])
        tl.observe(gap, [(40, box(x))])
        assert (tl.chain(40) == tl.chain(1)) is linked, (gap, x)


def test_two_lost_and_two_new_link_one_to_one_closest_first():
    tl = links()
    tl.observe(0.0, [(1, box(100)), (2, box(160))])
    tl.observe(0.5, [(40, box(140)), (41, box(170))])         # 41-2 (0.1 h) first, then 40-1 (0.4 h)
    assert (tl.link(40)[0], tl.link(41)[0]) == (1, 2)


def test_an_id_back_where_it_was_lost_keeps_its_chain():
    tl = links()
    tl.observe(0.0, [(1, box(100))])
    tl.observe(0.5, [])
    tl.observe(1.0, [(1, box(105)), (40, box(400))])
    assert tl.chain(1) != tl.chain(40) and tl.link(1) is None


def test_a_reused_id_joins_the_lost_track_it_continues():
    tl = links()
    tl.observe(0.0, [(10, box(800)), (17, box(100))])         # 10 is someone else, far away
    tl.observe(0.5, [(17, box(100))])                         # 10 lost
    tl.observe(1.0, [])                                       # 17 lost
    tl.observe(1.2, [(10, box(110))])                         # nvtracker gives 10 to 17's worker
    assert tl.chain(10) == tl.chain(17) and tl.link(10)[0] == 17


# --- the alert hold --------------------------------------------------------------------------------------------------

def _rules():
    return StreamRules(ComplianceConfig(), TemporalConfig(), eval_fps=20.0)


def test_an_item_starting_while_held_joins_the_alert_in_required_order():
    r = _rules()
    assert r.hold_alert(1.0, 0.5, 7, ["vest"], ["vest"]) is True
    assert r.hold_alert(1.3, 0.8, 7, ["helmet"], ["helmet", "vest"]) is False
    assert r.due_alerts(1.9, 1.0) == []
    (a,) = r.due_alerts(2.0, 1.0)
    assert (a["now"], a["pts"], a["track_id"], a["new"], a["missing"]) == (1.0, 0.5, 7, ["helmet", "vest"],
                                                                           ["helmet", "vest"])
    assert r.due_alerts(9.0, 1.0) == []


def test_held_alerts_of_two_tracks_are_separate_and_due_oldest_first():
    r = _rules()
    r.hold_alert(1.2, 1.2, 8, ["helmet"], ["helmet"])
    r.hold_alert(1.0, 1.0, 7, ["helmet"], ["helmet"])
    assert [a["track_id"] for a in r.due_alerts(5.0, 1.0)] == [7, 8]


def test_hold_0_is_due_on_the_same_frame():
    r = _rules()
    r.hold_alert(1.0, 1.0, 7, ["helmet"], ["helmet"])
    assert [a["track_id"] for a in r.due_alerts(1.0, 0.0)] == [7]
