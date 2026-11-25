from datetime import datetime

from chairs2 import SitEnd, SitStart
from alerts import EXCESSIVE_SERVICE
from domain import Appearance, EngineUpdate, FaceCandidate, FINISHED, WASHING
from haircuts import HaircutBook
from policy import DEFAULT_POLICY

P = {**DEFAULT_POLICY, "register_source": 0.0, "fr_after_s": 5.0}   # chair crops unless a test sets 1 (default 1 since 2026-10-06); 5 s face settle
NO_OCC = lambda ch: None      # noqa: E731 -- the chair has no open sitting
T0 = datetime(2026, 10, 4, 12, 0).timestamp()


class Ids:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"h{self.n}"


def book(decider=None):
    kw = {} if decider is None else {"decider": decider}
    return HaircutBook(Ids(), lambda chair, at: "staff-1", lambda at: 7, **kw)


def run(b, events, sitting=lambda ch: False):
    u = EngineUpdate()
    for e in events:
        (b.on_start if isinstance(e, SitStart) else b.on_end)(e, u, P)
    return u


def test_a_haircut_is_persisted_only_once_confirmed():
    b = book()
    u = run(b, [SitStart("R3", T0, 4)])
    assert not u.visits
    u = EngineUpdate()
    b.tick(T0 + 61, lambda ch: True, u, P, lambda ch: T0 + 61)
    (v,) = u.visits.values()
    (s,) = u.sessions.values()
    assert v.confirmed and v.wait_unseen and v.ticket == 7 and s.barber_id == "staff-1" and s.chair_id == "R3"


def test_return_within_window_resumes_with_wash_time():
    b = book()
    run(b, [SitStart("R3", T0, 4), SitEnd("R3", T0 + 600)])
    u = run(b, [SitStart("R3", T0 + 800, 9)])
    assert len(b.open) == 1 and b.open["R3"].id == "h1" and len(b.open["R3"].sittings) == 2
    run(b, [SitEnd("R3", T0 + 1400)])
    u = EngineUpdate()
    b.tick(T0 + 1400 + 361, lambda ch: False, u, P, NO_OCC)
    s = u.sessions["h1"]
    assert not s.open and s.service_s == 1200 and s.wash_s == 200
    assert u.visits["h1"].state == FINISHED and [r.visit_id for r in u.registrations] == ["h1"]
    assert [r.n for r in u.sittings] == [0, 1]


def test_return_after_window_is_a_new_customer():
    b = book()
    run(b, [SitStart("R3", T0, 4), SitEnd("R3", T0 + 600)])
    u = EngineUpdate()
    b.tick(T0 + 600 + 361, lambda ch: False, u, P, NO_OCC)
    run(b, [SitStart("R3", T0 + 1200, 9)])
    assert b.open["R3"].id == "h2"


def test_away_marks_the_visit_washing():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    b.tick(T0 + 100, lambda ch: True, EngineUpdate(), P, lambda ch: T0 + 100)
    u = run(b, [SitEnd("R3", T0 + 600)])
    assert u.visits["h1"].state == WASHING


def test_short_sit_is_dropped_without_a_customer():
    b = book()
    run(b, [SitStart("R3", T0, 4), SitEnd("R3", T0 + 30)])
    u = EngineUpdate()
    b.tick(T0 + 30 + 361, lambda ch: False, u, P, NO_OCC)
    assert not u.visits and not u.sessions and "R3" not in b.open


def test_a_decider_answering_new_splits_the_haircut():
    class Never:
        def resume(self, chair, last, start_ts, policy):
            return False
    b = book(Never())
    run(b, [SitStart("R3", T0, 4), SitEnd("R3", T0 + 600)])
    u = run(b, [SitStart("R3", T0 + 700, 9)])
    assert b.open["R3"].id == "h2"
    assert u.sessions["h1"].open is False


def test_with_the_door_register_every_appearance_crop_is_discarded_and_a_closed_haircut_registers_nothing():
    P = {**globals()["P"], "register_source": 1.0}
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.on_appearance(Appearance("barber-l", 4, T0 + 20, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P, u)
    b.on_appearance(Appearance("barber-l", 5, T0 + 21, (1.0,), 0.95, "c/b.jpg", 0.9), "R3", P, u)
    assert u.discard_crops == ["c/a.jpg", "c/b.jpg"]
    run(b, [SitEnd("R3", T0 + 100)])
    u = EngineUpdate()
    b.tick(T0 + 100 + 361, lambda ch: False, u, P, NO_OCC)
    assert u.sessions and not u.registrations


def test_pictures_come_from_the_sitter_not_staff():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.on_appearance(Appearance("barber-l", 4, T0 + 20, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P, u)
    b.on_appearance(Appearance("barber-l", 5, T0 + 21, (1.0,), 0.95, "c/b.jpg", 0.9), "R3", P, u)
    b.on_appearance(Appearance("barber-l", 4, T0 + 400, (1.0,), 0.1, "c/late.jpg", 0.9), "R3", P, u)
    assert [x.crop_path for x in b.open["R3"].bodies] == ["c/a.jpg"]
    assert set(u.discard_crops) == {"c/b.jpg", "c/late.jpg"}


def test_with_chair_crops_a_closed_confirmed_haircut_registers_its_bodies_and_clear_returns_them():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    for i, q in enumerate((0.5, 0.9, 0.7, 0.8)):
        b.on_appearance(Appearance("barber-l", 4, T0 + 10 + i, (1.0,), 0.1, f"c/{i}.jpg", q), "R3", P, u)
    assert u.discard_crops == ["c/0.jpg"]                 # best customer_max_bodies (3) kept
    assert sorted(x.crop_path for x in b.open["R3"].bodies) == ["c/1.jpg", "c/2.jpg", "c/3.jpg"]
    assert sorted(b.clear()) == ["c/1.jpg", "c/2.jpg", "c/3.jpg"]
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    b.on_appearance(Appearance("barber-l", 4, T0 + 20, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P, u)
    run(b, [SitEnd("R3", T0 + 400)])
    u = EngineUpdate()
    b.tick(T0 + 400 + 361, lambda ch: False, u, P, NO_OCC)
    (reg,) = u.registrations
    assert reg.visit_id == "h1" and reg.faces == () and [x.crop_path for x in reg.bodies] == ["c/a.jpg"]


def test_an_unconfirmed_haircut_discards_its_bodies():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.on_appearance(Appearance("barber-l", 4, T0 + 5, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P, u)
    run(b, [SitEnd("R3", T0 + 20)])
    u = EngineUpdate()
    b.tick(T0 + 20 + 361, lambda ch: False, u, P, NO_OCC)
    assert u.discard_crops == ["c/a.jpg"] and not u.registrations


def test_the_end_gap_does_not_count_as_sitting_time():
    """I-1: a 15 s sit whose chair stays SITTING through the 60 s end gap is never confirmed."""
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    for t in range(1, 75):                      # last occupied frame at +15, end gap still running
        b.tick(T0 + t, lambda ch: True, u, P, lambda ch: T0 + 15)
    assert not u.visits and not u.sessions
    u = run(b, [SitEnd("R3", T0 + 15)])
    b.tick(T0 + 15 + 361, lambda ch: False, u, P, NO_OCC)
    assert not u.visits and not u.sessions and "R3" not in b.open


def test_a_running_haircut_is_confirmed_once_its_occupied_time_reaches_the_minimum():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.tick(T0 + 61, lambda ch: True, u, P, lambda ch: T0 + 59)
    assert not u.visits
    b.tick(T0 + 62, lambda ch: True, u, P, lambda ch: T0 + 61)
    assert u.visits["h1"].state == "haircut"


STILL = {**P, "ce_object_spread": 0.005}


def boxes(b, chair, n, jitter=0.0, conf=0.9):
    """n occupant boxes (left, top, w, h); every other one moved by `jitter` (a head turned, the barber in front),
    each detected with `conf`."""
    for i in range(n):
        d = jitter * (i % 2)
        b.note_box(chair, (0.64 + d, 0.63 + d, 0.11, 0.23), conf)


def test_an_object_that_never_moves_is_never_a_haircut():
    """2026-10-18 (user): a bag on Palm Hills' VIP2 was a 7-min haircut. Its box's top and left edges stayed within
    3 px for 5 min (over 3 tracks); every real sitting that day moved 15 px or more on one of them."""
    b = book()
    run(b, [SitStart("ph7-VIP2", T0, 42)])
    boxes(b, "ph7-VIP2", 600)
    u = EngineUpdate()
    b.tick(T0 + 400, lambda ch: True, u, STILL, lambda ch: T0 + 400)
    assert not u.visits and not u.sessions
    b.on_end(SitEnd("ph7-VIP2", T0 + 450), u, STILL)
    b.tick(T0 + 450 + 361, lambda ch: False, u, STILL, NO_OCC)
    assert not u.visits and not u.sessions and "ph7-VIP2" not in b.open


def test_a_still_customer_is_confirmed_once_he_moves():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    boxes(b, "R3", 300)
    u = EngineUpdate()
    b.tick(T0 + 301, lambda ch: True, u, STILL, lambda ch: T0 + 301)
    assert not u.visits
    boxes(b, "R3", 300, jitter=0.02)
    b.tick(T0 + 400, lambda ch: True, u, STILL, lambda ch: T0 + 400)
    assert u.visits["h1"].state == "haircut" and u.sessions["h1"].started_at == datetime.fromtimestamp(T0)


def test_the_movement_check_applies_only_where_its_chair_key_sets_it():
    """2026-10-18 (user): the VIP chairs only for now, so a still customer elsewhere is never dropped."""
    vip = {**P, "ce_object_spread@ph7-VIP2": 0.005}
    for chair, haircut in (("ph7-VIP2", False), ("R3", True)):
        b = book()
        run(b, [SitStart(chair, T0, 4)])
        boxes(b, chair, 600)
        u = EngineUpdate()
        b.tick(T0 + 61, lambda ch: True, u, vip, lambda ch: T0 + 61)
        assert bool(u.visits) is haircut


def test_object_spread_off_or_no_boxes_confirms_as_before():
    for policy, n in ((P, 600), (STILL, 0)):                   # the switch off; on, but no box seen (an older feed)
        b = book()
        run(b, [SitStart("R3", T0, 4)])
        boxes(b, "R3", n)
        u = EngineUpdate()
        b.tick(T0 + 61, lambda ch: True, u, policy, lambda ch: T0 + 61)
        assert u.visits["h1"].confirmed



HALF_SURE = {**P, "ce_object_conf": 0.72}


def test_a_pile_of_clothes_the_detector_is_half_sure_of_is_never_a_haircut():
    """2026-10-22 (user): clothes on Palm Hills' VIP2 (16 Oct, 22:49) were a 5-min haircut. Re-arranged by the staff
    and with a box flickering between two outlines they "moved" (0.043), but the detector was only half sure: median
    conf 0.66 (the 18 Oct bag 0.48); every real VIP haircut 0.81-0.88."""
    b = book()
    run(b, [SitStart("ph7-VIP2", T0, 38)])
    boxes(b, "ph7-VIP2", 600, jitter=0.04, conf=0.66)
    u = EngineUpdate()
    b.tick(T0 + 400, lambda ch: True, u, HALF_SURE, lambda ch: T0 + 400)
    assert not u.visits and not u.sessions
    b.on_end(SitEnd("ph7-VIP2", T0 + 450), u, HALF_SURE)
    b.tick(T0 + 450 + 361, lambda ch: False, u, HALF_SURE, NO_OCC)
    assert not u.visits and not u.sessions and "ph7-VIP2" not in b.open


def test_a_customer_hidden_at_first_is_confirmed_once_the_median_gets_there():
    """The median of all his boxes so far: a crowded start delays the haircut (backdated), it does not drop it."""
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    boxes(b, "R3", 300, jitter=0.02, conf=0.55)
    u = EngineUpdate()
    b.tick(T0 + 301, lambda ch: True, u, HALF_SURE, lambda ch: T0 + 301)
    assert not u.visits
    boxes(b, "R3", 400, jitter=0.02, conf=0.85)
    b.tick(T0 + 400, lambda ch: True, u, HALF_SURE, lambda ch: T0 + 400)
    assert u.visits["h1"].state == "haircut" and u.sessions["h1"].started_at == datetime.fromtimestamp(T0)


def test_the_confidence_check_applies_only_where_its_chair_key_sets_it():
    """2026-10-22 (user): the two VIP chairs only."""
    vip = {**P, "ce_object_conf@ph7-VIP2": 0.72}
    for chair, haircut in (("ph7-VIP2", False), ("R3", True)):
        b = book()
        run(b, [SitStart(chair, T0, 4)])
        boxes(b, chair, 600, jitter=0.02, conf=0.66)
        u = EngineUpdate()
        b.tick(T0 + 61, lambda ch: True, u, vip, lambda ch: T0 + 61)
        assert bool(u.visits) is haircut


def test_object_conf_off_or_no_conf_seen_confirms_as_before():
    for policy, conf in ((P, 0.3), (HALF_SURE, None)):         # the switch off; on, but no conf given (an older feed)
        b = book()
        run(b, [SitStart("R3", T0, 4)])
        boxes(b, "R3", 600, jitter=0.02, conf=conf)
        u = EngineUpdate()
        b.tick(T0 + 61, lambda ch: True, u, policy, lambda ch: T0 + 61)
        assert u.visits["h1"].confirmed

def test_a_long_haircut_opens_and_closes_an_excessive_service_alert():
    """I-3: the chair engine keeps the old long-haircut alert."""
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.tick(T0 + 100, lambda ch: True, u, P, lambda ch: T0 + 100)
    assert u.sessions["h1"].service_target_s == P["service_target_s"] and not u.alerts
    u = EngineUpdate()
    b.tick(T0 + 1801, lambda ch: True, u, P, lambda ch: T0 + 1801)
    (a,) = u.alerts
    assert a.kind == EXCESSIVE_SERVICE and a.open and a.session_id == "h1" and a.visit_id == "h1"
    assert u.sessions["h1"].excessive_service
    u = EngineUpdate()
    b.tick(T0 + 1810, lambda ch: True, u, P, lambda ch: T0 + 1810)
    assert not u.alerts                         # one alert per haircut
    u = run(b, [SitEnd("R3", T0 + 2000)])
    b.tick(T0 + 2000 + 361, lambda ch: False, u, P, NO_OCC)
    (a2,) = u.alerts
    assert a2.id == a.id and not a2.open and a2.ended_at.timestamp() == T0 + 2000
    assert u.sessions["h1"].excessive_service and not u.sessions["h1"].open



def test_bodies_collected_under_chair_crops_are_discarded_when_the_haircut_closes_under_the_door_register():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.on_appearance(Appearance("barber-l", 4, T0 + 20, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P, u)
    run(b, [SitEnd("R3", T0 + 400)])
    u = EngineUpdate()
    b.tick(T0 + 400 + 361, lambda ch: False, u, {**P, "register_source": 1.0}, NO_OCC)
    assert u.sessions and not u.registrations and u.discard_crops == ["c/a.jpg"]


def test_no_long_haircut_alert_once_the_customer_has_left():
    """2026-10-07 review: a 600 s cut, target 900 s -- away (and the end gap) is not haircut time."""
    pol = {**P, "service_target_s": 900.0}
    b = book()
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0, 4), u, pol)
    for t in range(0, 660, 10):                                     # the end gap: no occupied frame after 600
        b.tick(T0 + t, lambda ch: True, u, pol, lambda ch: T0 + min(t, 600))
    b.on_end(SitEnd("R3", T0 + 600), u, pol)
    for t in range(660, 1000, 10):
        b.tick(T0 + t, lambda ch: False, u, pol, NO_OCC)
    assert not u.alerts and not u.sessions["h1"].excessive_service and not u.sessions["h1"].open


def test_the_long_haircut_alert_fires_when_he_is_still_seen_past_the_target():
    pol = {**P, "service_target_s": 900.0}
    b = book()
    u = EngineUpdate()
    b.on_start(SitStart("R3", T0, 4), u, pol)
    b.tick(T0 + 905, lambda ch: True, u, pol, lambda ch: T0 + 890)  # last seen before the target: not yet
    assert not u.alerts
    b.tick(T0 + 906, lambda ch: True, u, pol, lambda ch: T0 + 906)
    (a,) = u.alerts
    assert a.open and a.started_at.timestamp() == T0 + 900


# ── register_source 2 (Task 13) ──────────────────────────────────────────────
P2 = {**P, "register_source": 2.0}


def fc(path, q=0.8):
    return FaceCandidate("barber-l", 4, T0 + 30, path, (1.0,) + (0.0,) * 511, q)


def test_under_register_source_2_a_closed_confirmed_haircut_registers_its_faces_and_bodies():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.on_appearance(Appearance("barber-l", 4, T0 + 20, (1.0,), 0.1, "c/a.jpg", 0.8), "R3", P2, u)
    b.add_face("R3", fc("f/1.jpg"), P2, u)
    run(b, [SitEnd("R3", T0 + 400)])
    u = EngineUpdate()
    b.tick(T0 + 400 + 361, lambda ch: False, u, P2, NO_OCC)
    (reg,) = u.registrations
    assert [f.crop_path for f in reg.faces] == ["f/1.jpg"] and [x.crop_path for x in reg.bodies] == ["c/a.jpg"]


def test_an_unconfirmed_haircut_discards_its_faces():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.add_face("R3", fc("f/1.jpg"), P2, u)
    run(b, [SitEnd("R3", T0 + 20)])
    u = EngineUpdate()
    b.tick(T0 + 20 + 361, lambda ch: False, u, P2, NO_OCC)
    assert u.discard_crops == ["f/1.jpg"] and not u.registrations


def test_add_face_keeps_the_best_visit_max_faces_and_clear_returns_them():
    b = book()
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    pol = {**P2, "visit_max_faces": 2.0}
    for i, q in enumerate((0.5, 0.9, 0.7)):
        b.add_face("R3", fc(f"f/{i}.jpg", q), pol, u)
    assert u.discard_crops == ["f/0.jpg"]
    assert sorted(b.clear()) == ["f/1.jpg", "f/2.jpg"]


def test_add_face_on_a_chair_with_no_haircut_discards_it():
    b = book()
    u = EngineUpdate()
    b.add_face("R9", fc("f/x.jpg"), P2, u)
    assert u.discard_crops == ["f/x.jpg"]


def _registered_then_reopened():
    b = book()
    pol = {**P2, "ce_resume_check": 1.0}
    run(b, [SitStart("R3", T0, 4)])
    u = EngineUpdate()
    b.add_face("R3", fc("f/1.jpg"), pol, u)
    run(b, [SitEnd("R3", T0 + 400)])
    b.tick(T0 + 400 + 361, lambda ch: False, u, pol, NO_OCC)
    assert len(u.registrations) == 1
    h = b.recent.pop("R3")                  # reopened by the resume path
    b.open["R3"] = h
    h.sittings.append([T0 + 800, None])
    return b, h, pol


def test_a_reopened_registered_haircut_discards_new_faces_and_registers_nothing_again():
    b, h, pol = _registered_then_reopened()
    u = EngineUpdate()
    b.add_face("R3", fc("f/2.jpg"), pol, u)
    assert u.discard_crops == ["f/2.jpg"] and h.faces == []
    run(b, [SitEnd("R3", T0 + 1200)])
    u = EngineUpdate()
    b.tick(T0 + 1200 + 361, lambda ch: False, u, pol, NO_OCC)
    assert not u.registrations and not u.discard_crops and h.faces == []


def test_clear_after_a_registered_reopened_haircut_returns_no_face_paths():
    b, h, pol = _registered_then_reopened()
    assert b.clear() == []


# ce_low_occupancy (2026-11-14, user; Palm Hills 6 Nov L6 19:47): a standing staff member cut by the frame edge was on
# the seat in a fifth of the frames and still held a 17-min haircut. Such a haircut stays, marked on its session.
def frames(b, chair, pattern, policy, at=T0, step=1.0):
    for i, occupied in enumerate(pattern):
        b.note_frame(chair, at + i * step, occupied, policy)


def test_a_haircut_seen_on_the_seat_in_few_frames_is_marked_low_occupancy():
    pol = {**P, "ce_low_occupancy": 0.25}
    b = book()
    run(b, [SitStart("L6", T0, 4)])
    frames(b, "L6", [True] + [False] * 8, pol)                  # on the seat in 1 frame of 9
    frames(b, "L6", [True] + [False] * 30, pol)                 # ... and the end gap after the last one never counts
    u = run(b, [SitEnd("L6", T0 + 600)])
    (s,) = u.sessions.values()
    assert s.low_occupancy and s.occupancy == 2 / 10


def test_a_customer_on_the_seat_most_of_the_time_is_not_marked_and_off_marks_nothing():
    pol = {**P, "ce_low_occupancy": 0.25}
    b = book()
    run(b, [SitStart("L6", T0, 4)])
    frames(b, "L6", [True, False] * 20, pol)
    (s,) = run(b, [SitEnd("L6", T0 + 600)]).sessions.values()
    assert not s.low_occupancy and s.occupancy == 20 / 39
    b = book()
    run(b, [SitStart("L6", T0, 4)])
    frames(b, "L6", [True] + [False] * 9 + [True], P)           # default 0: off
    (s,) = run(b, [SitEnd("L6", T0 + 600)]).sessions.values()
    assert not s.low_occupancy and s.occupancy == 2 / 11


def test_low_occupancy_is_read_per_chair_and_counts_every_sitting_but_not_the_wash():
    pol = {**P, "ce_low_occupancy@L6": 0.5}
    b = book()
    run(b, [SitStart("L6", T0, 4), SitStart("R3", T0, 5)])
    frames(b, "L6", [True, False, False, True], pol)
    frames(b, "R3", [True, False, False, True], pol)
    run(b, [SitEnd("L6", T0 + 400), SitEnd("R3", T0 + 400)])
    frames(b, "L6", [False] * 50, pol)                          # at the wash: no sitting runs, nothing counted
    run(b, [SitStart("L6", T0 + 500, 4)])
    frames(b, "L6", [False, False, True], pol, at=T0 + 500)
    u = run(b, [SitEnd("L6", T0 + 700)])
    assert u.sessions["h1"].occupancy == 3 / 7 and u.sessions["h1"].low_occupancy
    assert not b.open["R3"].low_occupancy and b.open["R3"].occupancy == 0.5


def test_a_short_haircut_needs_the_higher_share_and_a_long_one_only_the_base():
    pol = {**P, "ce_low_occupancy": 0.22, "ce_low_occupancy_short": 0.35, "ce_low_occupancy_short_s": 900.0}
    b = book()                                                  # 6 Nov: L6 19:47, 10 min at 30 % -- the standing staff
    run(b, [SitStart("L6", T0, 4), SitStart("R1", T0, 5)])
    frames(b, "L6", [True, False, False, False, True, False, False, False, False, True], pol, step=60.0)
    frames(b, "R1", ([False] * 3 + [True]) * 6, pol, step=60.0)        # R1 20:08, 21 min at 25 % -- a real customer
    u = run(b, [SitEnd("L6", T0 + 600), SitEnd("R1", T0 + 1300)])
    assert u.sessions["h1"].occupancy == 0.3 and u.sessions["h1"].low_occupancy
    assert u.sessions["h2"].occupancy == 0.25 and not u.sessions["h2"].low_occupancy
    b = book()                                                  # without the short share, the base alone decides
    run(b, [SitStart("L6", T0, 4)])
    frames(b, "L6", [True, False, False, False, True, False, False, False, False, True], {**pol, "ce_low_occupancy_short": 0.0}, step=60.0)
    assert not run(b, [SitEnd("L6", T0 + 600)]).sessions["h1"].low_occupancy
