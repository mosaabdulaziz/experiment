"""Scenario tests: one frame per second from each camera (empty frames when
nobody is in view), appearances half a second after the frame they sample."""
import itertools

from domain import (COMPLETED, EXIT_DOOR, EXIT_LOST, FINISHED, FaceCandidate, HAIRCUT, LOST, SERVED, VOID_STAFF,
                    WAITING, WASHING)
from engine import ShopEngine
from tracklets import unit
from shop_helpers import (C_AISLE, C_CHAIR, C_DOOR, C_SOFA, C_STREET, C_WASH, GEO, L_EDGE, L_SEAT, POLICY, ZONES,
                          at, frame, reid_app, staff_app, ts)


class Shop:
    def __init__(self, policy=POLICY):
        self.policy = policy
        n = itertools.count()
        self.engine = ShopEngine(ZONES, GEO, new_id=lambda: f"id{next(n)}")
        self.visits, self.sessions, self.deleted = {}, {}, set()
        self.events, self.handoffs, self.alerts = [], [], []
        self.registrations, self.discarded = [], []
        self.bindings = []

    def step(self, s, c=None, l=None, apps=(), faces=()):
        frames = [frame("barber-c", s, c or {}), frame("barber-l", s, l or {})]
        u = self.engine.observe(frames, list(apps), self.policy, faces=list(faces))
        self.visits.update(u.visits)
        self.sessions.update(u.sessions)
        for sid in u.deleted_sessions:
            self.sessions.pop(sid, None)
            self.deleted.add(sid)
        self.events += u.events
        self.handoffs += u.handoffs
        self.alerts += u.alerts
        self.registrations += u.registrations
        self.discarded += u.discard_crops
        self.bindings += u.bindings

    def run(self, s0, s1, c=None, l=None, apps_at=None):
        for s in range(s0, s1):
            self.step(s, c, l, apps_at(s) if apps_at else ())

    def states(self):
        return [e.to_state for e in self.events]


def walk_in_and_sit(shop):
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})
    shop.run(10, 40, c={1: C_SOFA})
    shop.step(40, c={1: C_CHAIR}, l={7: L_EDGE}, apps=[staff_app("barber-l", 7, 40.5, 0.1)])
    shop.run(41, 150, c={1: C_CHAIR}, l={7: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.1)] if s < 43 else [])


def test_every_join_writes_one_binding_row():
    shop = Shop()
    walk_in_and_sit(shop)
    kinds = [(b.kind, b.camera, b.track_id) for b in shop.bindings]
    assert kinds[:2] == [("door", "barber-c", 1), ("handoff", "barber-l", 7)]
    h = shop.bindings[1]
    assert h.partner_camera == "barber-c" and h.partner_track == 1 and h.area == "ch1"
    assert h.person_id == shop.bindings[0].person_id and h.offset_s is not None


def test_a_missed_entry_on_the_sofa_writes_a_waiting_binding():
    shop = Shop()
    shop.run(0, 12, c={1: C_SOFA}, apps_at=lambda s: [staff_app("barber-c", 1, s + 0.5, 0.1)])
    assert [(b.kind, b.area) for b in shop.bindings] == [("waiting", "sofa")]


def test_door_to_chair_to_door():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 201, c={1: C_CHAIR}, l={7: L_SEAT})
    shop.run(201, 231, c={1: C_AISLE})
    shop.run(231, 234, c={1: C_DOOR})
    shop.run(234, 237, c={1: C_STREET})
    shop.run(237, 245)
    (v,) = shop.visits.values()
    (s,) = shop.sessions.values()
    assert v.entered_at == at(3) and v.entry_observed and v.wait_s == 38
    assert (s.chair_id, s.started_at, s.ended_at, s.service_s) == ("ch1", at(41), at(200), 159)
    assert (v.outcome, v.exit_reason, v.exited_at) == (COMPLETED, EXIT_DOOR, at(234))
    assert shop.states() == [WAITING, HAIRCUT, SERVED, FINISHED]
    (h,) = shop.handoffs
    assert (h.area, h.camera_a, h.track_a, h.camera_b, h.track_b, h.offset_s) == \
        ("ch1", "barber-c", 1, "barber-l", 7, 0.0)


def test_a_barber_decided_outside_the_seat_starts_nothing_over_it():
    shop = Shop()
    # 2026-09-26 (Task 5a): outside every seat staff_outside_confirm_s (20 s) with a staff vote
    shop.run(0, 25, l={9: L_EDGE}, apps_at=lambda s: [staff_app("barber-l", 9, s + 0.5, 0.75)] if s < 3 else [])
    shop.run(25, 80, l={9: L_SEAT})
    assert shop.sessions == {} and shop.visits == {}


def test_a_body_first_seen_seated_at_barber_like_scores_may_start():
    # the documented cost of the seat rule: 0.75 in a seat with no history outside it is a customer
    shop = Shop()
    shop.run(0, 60, l={9: L_SEAT}, apps_at=lambda s: [staff_app("barber-l", 9, s + 0.5, 0.75)] if s < 3 else [])
    (s,) = shop.sessions.values()
    assert s.chair_id == "ch1"


def test_a_caped_customer_who_stands_up_stays_bound_and_leaves_by_the_door():
    # 2026-09-24 (Task 16 tune): 0.8 is staff by the aisle threshold (0.6 here), but the track
    # latched as a customer after sitting seat_dwell_s
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})
    shop.run(10, 40, c={1: C_SOFA})
    shop.step(40, c={1: C_CHAIR}, l={7: L_EDGE})
    shop.run(41, 150, c={1: C_CHAIR}, l={7: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.8)] if s < 44 else [])
    shop.run(150, 155, c={1: C_CHAIR}, l={7: L_EDGE})
    (v,) = shop.visits.values()
    assert shop.engine.identity.persons[v.person_id].tracks["barber-l"].track_id == 7
    shop.run(155, 185, c={1: C_AISLE})
    shop.run(185, 188, c={1: C_DOOR})
    shop.run(188, 191, c={1: C_STREET})
    shop.run(191, 200)
    (s,) = shop.sessions.values()
    assert s.chair_id == "ch1"
    assert (v.exit_reason, v.exited_at) == (EXIT_DOOR, at(188))


def test_a_barber_walking_in_is_voided_once_recognized():
    shop = Shop()
    shop.run(0, 3, c={5: C_STREET})
    shop.run(3, 20, c={5: C_SOFA},
             apps_at=lambda s: [staff_app("barber-c", 5, s + 0.5, 0.9)] if 6 <= s < 9 else [])
    (v,) = shop.visits.values()
    assert v.outcome == VOID_STAFF and not v.open


def test_a_door_crossing_by_a_customer_still_seen_in_the_chair_is_no_exit():
    # 2026-09-24 (Task 16 tune): the centre view's door line runs next to the first chairs; feet
    # of a customer there flip across it while the side camera sees him seated
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 153, c={1: C_DOOR}, l={7: L_SEAT})
    shop.run(153, 190, c={1: C_STREET}, l={7: L_SEAT})
    (v,) = shop.visits.values()
    assert v.open and v.state == HAIRCUT


def test_a_door_crossing_disproved_by_the_seat_survives_the_side_track_churning():
    # 2026-09-26 (Task 5b, raw1 R1): seen in his own seat after the crossing = he never left;
    # a moment later every track of his is gone (side track 7 -> 8), which must not confirm it
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 153, c={1: C_DOOR}, l={7: L_SEAT})
    shop.run(153, 190, c={1: C_STREET}, l={7: L_SEAT})
    shop.run(190, 194)
    shop.run(194, 260, l={8: L_SEAT})
    (v,) = shop.visits.values()
    (s,) = shop.sessions.values()
    assert v.open and v.state == HAIRCUT and s.ended_at is None
    assert shop.engine.identity.persons[v.person_id].tracks["barber-l"].track_id == 8


def test_a_real_exit_with_the_side_view_lagging_in_the_seat_is_still_a_door_exit():
    # review of 1fb36e0: a moment of seat geometry after the crossing (the side view 2-4 s behind)
    # is no proof he stayed
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 152, c={1: C_DOOR}, l={7: L_SEAT})
    shop.run(152, 154, c={1: C_STREET}, l={7: L_SEAT})
    shop.run(154, 240)
    (v,) = shop.visits.values()
    assert (v.exit_reason, v.exited_at) == (EXIT_DOOR, at(152))


TWO_CHAIRS = {"barber-c": ZONES["barber-c"] + [
    {"id": "c-ch2", "name": "chair 2 (centre)", "type": "chair", "points": [[0.8, 0.3], [1.0, 0.3], [1.0, 0.5],
                                                                           [0.8, 0.5]], "pair": "ch2"}],
         "barber-l": ZONES["barber-l"] + [
    {"id": "l-ch2", "name": "chair 2", "type": "chair", "points": [[0.05, 0.4], [0.35, 0.4], [0.35, 0.9],
                                                                    [0.05, 0.9]], "pair": "ch2", "owner": True},
    {"id": "l-s2", "name": "seat 2", "type": "seat", "points": [[0.1, 0.5], [0.3, 0.5], [0.3, 0.8], [0.1, 0.8]],
     "seat_of": "l-ch2"}]}
SEAT2, C_CHAIR2 = (0.2, 0.65), (0.9, 0.4)


def test_a_body_in_another_seat_while_the_customer_is_still_seen_in_his_chair_is_someone_else():
    # 2026-09-26 (Task 5b, raw3 R4->R3): the side track in the next seat was bound to the R4
    # customer while his centre track stayed in R4 -- that is two bodies, not a move
    shop = Shop()
    shop.engine.set_zones(TWO_CHAIRS)
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.step(150, c={1: C_CHAIR}, l={9: SEAT2}, apps=[staff_app("barber-l", 9, 150.5, 0.1)])
    p = shop.engine.identity.persons[v.person_id]
    shop.engine.identity.bind(shop.engine.books["barber-l"].tracks[9], p, ts(150))   # the wrong bind
    shop.step(151, c={1: C_CHAIR}, l={9: SEAT2}, apps=[staff_app("barber-l", 9, 151.5, 0.1)],
              faces=[fc("barber-l", 9, 151, "wrong151", q=0.95)])
    shop.run(152, 200, c={1: C_CHAIR}, l={9: SEAT2},
             apps_at=lambda s: [staff_app("barber-l", 9, s + 0.5, 0.1)] if s < 153 else [])
    first = next(s for s in shop.sessions.values() if s.visit_id == v.id)
    assert first.chair_id == "ch1" and first.ended_at is None
    (other,) = [s for s in shop.sessions.values() if s.visit_id != v.id]
    assert other.chair_id == "ch2"
    assert "faces/d/wrong151.jpg" in shop.discarded       # its crop left the first customer's pool
    last = shop.bindings[-1]
    assert (last.kind, last.camera, last.track_id) == ("seat_start", "barber-l", 9)


def test_a_customer_who_really_moves_to_the_next_seat_moves_his_haircut():
    # guard for the rule above: once nothing of his is left in the old chair, it is a move
    shop = Shop()
    shop.engine.set_zones(TWO_CHAIRS)
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.run(150, 200, c={1: C_CHAIR2}, l={7: SEAT2})
    first, second = sorted((s for s in shop.sessions.values() if s.visit_id == v.id), key=lambda s: s.started_at)
    assert (first.chair_id, second.chair_id) == ("ch1", "ch2") and second.ended_at is None
    assert len(shop.visits) == 1


def test_a_churned_side_track_in_the_seat_keeps_the_haircut():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 260, l={8: L_SEAT})                    # centre loses him, side track 7 -> 8
    (v,) = shop.visits.values()
    (s,) = shop.sessions.values()
    assert v.open and s.ended_at is None
    assert shop.engine.identity.persons[v.person_id].tracks["barber-l"].track_id == 8


def side_tracks(shop):
    (v,) = shop.visits.values()
    return {c: t.track_id for c, t in shop.engine.identity.persons[v.person_id].tracks.items()}


def test_a_barber_over_the_seat_is_no_claim_for_its_customer():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 175, c={1: C_CHAIR}, l={9: L_EDGE},
             apps_at=lambda s: [staff_app("barber-l", 9, s + 0.5, 0.75)] if s < 153 else [])
    shop.run(175, 220, c={1: C_CHAIR}, l={9: L_SEAT})
    assert side_tracks(shop) == {"barber-c": 1}


# The seat claim (review #2): an unbound body sitting in a running haircut's seat joins its
# customer only if that can be him. Bodies just either side of each rule, and the sequence the
# guard exists for. A declined claim is no evidence: the old haircut ends on its own clock.

def old_and_new(shop):
    """The first customer's visit and haircut; the person who owns side track 8 now."""
    old = min(shop.visits.values(), key=lambda v: v.entered_at)
    (sess,) = [s for s in shop.sessions.values() if s.visit_id == old.id]
    return old, sess, shop.engine.books["barber-l"].tracks[8].person


def lost_seated_customer_and_a_body_like(shop, vec):
    """The centre view loses a seated customer (re-ID known) and the side camera's track churns
    to a body with embedding vec: below reid_match_min, so only the seat claim can join them."""
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.engine.identity.persons[v.person_id].reid = unit((1, 0, 0))
    shop.run(150, 200, l={8: L_SEAT}, apps_at=lambda s: [reid_app("barber-l", 8, s + 0.5, vec)])
    return old_and_new(shop)


def test_a_body_like_the_lost_customer_in_his_seat_is_bound_to_him():
    old, sess, owner = lost_seated_customer_and_a_body_like(Shop(), (0.35, (1 - 0.35 ** 2) ** 0.5, 0))  # >= 0.3
    assert owner == old.person_id and sess.ended_at is None


def test_a_body_unlike_the_lost_customer_in_his_seat_is_no_claim_and_no_evidence():
    old, sess, owner = lost_seated_customer_and_a_body_like(Shop(), (0.25, (1 - 0.25 ** 2) ** 0.5, 0))  # < 0.3
    assert owner != old.person_id
    assert sess.ended_at == at(149)                     # 50 declined frames never kept it alive


def test_a_body_in_the_seat_while_the_customer_sits_on_the_sofa_is_no_claim_and_no_evidence():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 200, c={1: C_SOFA}, l={8: L_SEAT})
    old, sess, owner = old_and_new(shop)
    assert owner != old.person_id
    assert sess.ended_at == at(149)


def test_the_next_customer_sitting_down_is_not_bound_to_the_one_leaving():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 153, c={1: C_CHAIR}, l={7: L_EDGE})              # he stands up...
    shop.run(153, 175, c={1: C_AISLE}, l={8: L_SEAT},               # ...walks off to pay; the next one
             apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < 156 else [])  # sits, the
    shop.run(175, 178, c={1: C_DOOR}, l={8: L_SEAT})                # centre view blocked by the barber
    shop.run(178, 181, c={1: C_STREET}, l={8: L_SEAT})
    shop.run(181, 240, l={8: L_SEAT})
    old, new = sorted(shop.visits.values(), key=lambda v: v.entered_at)
    first, second = sorted(shop.sessions.values(), key=lambda s: s.started_at)
    assert (old.exit_reason, old.exited_at) == (EXIT_DOOR, at(178))
    assert (first.visit_id, first.ended_at) == (old.id, at(152))
    assert (second.visit_id, second.chair_id, second.started_at, second.ended_at) == (new.id, "ch1", at(153), None)
    assert shop.engine.identity.persons[new.person_id].tracks["barber-l"].track_id == 8


def test_a_frozen_side_camera_does_not_hold_the_door_exit():
    # review round 1: barber-l stops sending frames; its last track stays "in ch1" forever
    shop = Shop()
    walk_in_and_sit(shop)

    def centre_only(s0, s1, pt=None):
        for s in range(s0, s1):
            u = shop.engine.observe([frame("barber-c", s, {1: pt} if pt else {})], [], POLICY)
            shop.visits.update(u.visits)
    centre_only(150, 180, C_AISLE)
    centre_only(180, 183, C_DOOR)
    centre_only(183, 186, C_STREET)
    centre_only(186, 230)
    (v,) = shop.visits.values()
    assert (v.exit_reason, v.exited_at) == (EXIT_DOOR, at(183))


def test_the_crossing_tracks_own_zone_does_not_hold_the_door_exit():
    # review round 1: in the centre view the door line runs next to chairs L1/R1; a centre track
    # crossing out can still have its anchor in a chair zone
    zones = {**ZONES, "barber-c": ZONES["barber-c"] + [
        {"id": "c-ch2", "name": "chair by the door", "type": "chair", "points": [[0.4, -0.1], [0.6, -0.1],
                                                                                   [0.6, 0.05], [0.4, 0.05]]}]}
    shop = Shop()
    shop.engine.set_zones(zones)
    walk_in_and_sit(shop)
    shop.run(150, 180, c={1: C_AISLE})
    shop.run(180, 183, c={1: C_DOOR})
    shop.run(183, 230, c={1: C_STREET})                   # lingers outside, anchor in c-ch2
    (v,) = shop.visits.values()
    assert (v.exit_reason, v.exited_at) == (EXIT_DOOR, at(183))


def test_a_wash_trip_keeps_the_same_haircut():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 181, c={1: C_WASH})
    shop.step(181, c={1: C_CHAIR}, l={8: L_EDGE}, apps=[staff_app("barber-l", 8, 181.5, 0.1)])
    shop.run(182, 301, c={1: C_CHAIR}, l={8: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < 184 else [])
    shop.run(301, 330, c={1: C_AISLE})
    (s,) = shop.sessions.values()
    assert (s.started_at, s.ended_at, s.wash_s) == (at(41), at(300), 33)
    assert shop.states() == [WAITING, HAIRCUT, WASHING, HAIRCUT, SERVED]
    assert len(shop.handoffs) == 2


def test_first_seen_on_the_sofa_counts_and_a_vanished_customer_is_lost():
    shop = Shop()
    shop.run(0, 20, c={2: C_SOFA}, apps_at=lambda s: [staff_app("barber-c", 2, s + 0.5, 0.1)] if s < 3 else [])
    shop.run(20, 85)
    (v,) = shop.visits.values()
    assert (v.entered_at, v.entry_observed) == (at(0), False)
    assert (v.outcome, v.exit_reason, v.exited_at, v.wait_s) == (LOST, EXIT_LOST, at(19), 19)


def test_a_new_track_for_the_same_body_rejoins_the_customer():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 20, c={1: C_SOFA}, apps_at=lambda s: [reid_app("barber-c", 1, s + 0.5, (1, 0, 0))])
    shop.run(20, 30, c={2: C_SOFA}, apps_at=lambda s: [reid_app("barber-c", 2, s + 0.5, (0.98, 0.1, 0))])
    shop.run(30, 100, c={2: C_SOFA})
    (v,) = shop.visits.values()
    assert v.open and v.state == WAITING
    assert shop.engine.identity.persons[v.person_id].tracks["barber-c"].track_id == 2


def test_live_view_and_scene():
    shop = Shop()
    walk_in_and_sit(shop)
    live = shop.engine.live()
    (c,) = live["customers"]
    assert c["state"] == HAIRCUT and c["chair_id"] == "ch1" and c["cameras"] == ["barber-c", "barber-l"]
    assert live["chairs"]["ch1"]["person_id"] == c["person_id"] and live["customers_waiting"] == 0
    (body,) = shop.engine.scene(POLICY)["barber-l"]
    assert body["person_id"] == c["person_id"] and body["seat"] == "ch1" and body["state"] == HAIRCUT


def test_live_customers_and_scene_carry_the_daily_ticket():
    shop = Shop()
    walk_in_and_sit(shop)
    live = shop.engine.live()
    (c,) = live["customers"]
    assert c["ticket"] == 1
    assert isinstance(c["state_since"], str) and c["state_since"]
    (body,) = shop.engine.scene(POLICY)["barber-l"]
    assert body["ticket"] == 1


def test_clear_forgets_everyone_but_keeps_the_zones():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.engine.clear()
    assert shop.engine.live()["customers"] == [] and "ch1" in shop.engine.chairs.chairs


def test_a_barber_who_sits_in_the_side_seat_starts_nothing():
    shop = Shop()
    shop.run(0, 3, c={5: C_STREET})
    shop.step(3, c={5: C_DOOR})
    shop.run(4, 10, c={5: C_CHAIR})
    shop.run(10, 35, c={5: C_CHAIR}, l={9: L_EDGE},
             apps_at=lambda s: [staff_app("barber-c", 5, s + 0.5, 0.75), staff_app("barber-l", 9, s + 0.5, 0.75)]
             if s < 13 else [])
    shop.run(35, 60, c={5: C_CHAIR}, l={9: L_SEAT})
    assert shop.sessions == {}
    (v,) = shop.visits.values()
    assert v.outcome == VOID_STAFF and not v.open


def test_a_peek_through_the_door_that_vanishes_never_confirms():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.step(3, c={1: C_DOOR})
    shop.run(4, 80)
    assert shop.visits == {}
    assert shop.engine.live()["customers"] == []


def fc(cam, tid, s, name, q=0.5, emb=(1.0, 0.0)):
    from shop_helpers import ts
    return FaceCandidate(cam, tid, ts(s) + 0.2, f"faces/d/{name}.jpg", emb, q)


def test_a_served_customer_is_registered_with_his_best_faces():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    for s in range(3, 10):
        shop.step(s, c={1: C_DOOR}, faces=[fc("barber-c", 1, s, f"door{s}", q=0.1 * (s - 2))])
    shop.run(10, 40, c={1: C_SOFA})
    shop.step(40, c={1: C_CHAIR}, l={7: L_EDGE})
    shop.run(41, 201, c={1: C_CHAIR}, l={7: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.1)] if s < 44 else [])
    shop.step(201, c={1: C_AISLE}, l={7: L_SEAT}, faces=[fc("barber-l", 7, 201, "seat201", q=0.95)])
    shop.run(202, 231, c={1: C_AISLE})
    shop.run(231, 234, c={1: C_DOOR})
    shop.run(234, 237, c={1: C_STREET})
    shop.run(237, 245)
    (reg,) = shop.registrations
    (v,) = shop.visits.values()
    assert reg.visit_id == v.id and reg.ended_at == v.exited_at
    names = {f.crop_path for f in reg.faces}
    assert "faces/d/seat201.jpg" in names and len(reg.faces) == 8        # 7 door + 1 seat, all under the cap
    assert shop.discarded == []


def test_a_walk_out_is_not_registered_and_its_crops_are_discarded():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    for s in range(3, 30):
        shop.step(s, c={1: C_SOFA}, faces=[fc("barber-c", 1, s, f"f{s}")] if s % 5 == 0 else [])
    shop.run(30, 33, c={1: C_DOOR})
    shop.run(33, 36, c={1: C_STREET})
    shop.run(36, 45)
    assert shop.registrations == []
    assert sorted(shop.discarded) == sorted(f"faces/d/f{s}.jpg" for s in range(5, 30, 5))


def test_a_face_on_a_track_without_a_customer_is_discarded_at_once():
    shop = Shop()
    shop.step(0, l={9: L_SEAT}, faces=[fc("barber-l", 9, 0, "stranger")])
    assert shop.discarded == ["faces/d/stranger.jpg"]


def test_clear_returns_every_pooled_crop():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.step(3, c={1: C_DOOR}, faces=[fc("barber-c", 1, 3, "a")])
    shop.run(4, 8, c={1: C_DOOR})
    assert shop.engine.clear() == ["faces/d/a.jpg"]


def test_two_cameras_feed_one_visit_pool():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT},
              faces=[fc("barber-c", 1, 150, "centre"), fc("barber-l", 7, 150, "side")])
    (pool,) = shop.engine.pools.values()
    assert {f.crop_path for f in pool.faces} == {"faces/d/centre.jpg", "faces/d/side.jpg"}


def test_a_short_sit_then_walk_out_is_not_registered():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})
    shop.step(10, c={1: C_CHAIR}, l={7: L_EDGE}, faces=[fc("barber-c", 1, 10, "x")])
    shop.run(11, 40, c={1: C_CHAIR}, l={7: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.1)] if s < 14 else [])  # 29 s < min_service_s
    shop.run(40, 70, c={1: C_AISLE})
    shop.run(70, 73, c={1: C_DOOR})
    shop.run(73, 76, c={1: C_STREET})
    shop.run(76, 85)
    assert shop.registrations == [] and "faces/d/x.jpg" in shop.discarded


# ── staff never reach the register ─────────────────────────────────────────

def body_app(cam, tid, s, name, staff_p):
    from domain import Appearance
    from shop_helpers import ts
    return Appearance(cam, tid, ts(s) + 0.5, None, staff_p, f"bodies/d/{name}.jpg", 0.9)


def test_a_high_staff_p_body_sample_is_never_pooled():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT},
              apps=[body_app("barber-l", 7, 150, "apron", 0.8), body_app("barber-l", 7, 150, "cape", 0.2)])
    (pool,) = shop.engine.pools.values()
    assert [b.crop_path for b in pool.bodies] == ["bodies/d/cape.jpg"]
    assert shop.discarded == ["bodies/d/apron.jpg"]


def test_a_low_staff_p_body_sample_from_a_staff_voted_track_is_not_pooled():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})                       # bound to an open visit
    shop.step(10, c={1: C_SOFA}, apps=[staff_app("barber-c", 1, 10 + 0.05 * i, 0.9) for i in range(1, 4)]
              + [body_app("barber-c", 1, 10, "lowp", 0.3)])   # the vote is staff, this one sample isn't
    assert "bodies/d/lowp.jpg" in shop.discarded
    assert all(not p.bodies for p in shop.engine.pools.values())


def test_a_body_crop_without_staff_p_is_not_pooled():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT}, apps=[body_app("barber-l", 7, 150, "nop", None)])
    assert shop.discarded == ["bodies/d/nop.jpg"]


def test_bodies_rank_by_how_unlike_staff_they_look_not_by_height():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT},
              apps=[body_app("barber-l", 7, 150, "b", 0.4), body_app("barber-l", 7, 150, "a", 0.1)])
    (pool,) = shop.engine.pools.values()
    assert [b.crop_path for b in pool.bodies] == ["bodies/d/a.jpg", "bodies/d/b.jpg"]
    assert [b.quality for b in pool.bodies] == [0.9, 0.6]


def test_a_face_from_a_staff_voted_track_is_never_pooled():
    shop = Shop()
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})                       # bound to an open visit, no staff samples yet
    shop.step(10, c={1: C_SOFA}, apps=[staff_app("barber-c", 1, 10 + 0.05 * i, 0.9) for i in range(1, 4)],
              faces=[fc("barber-c", 1, 10, "barber")])    # the face (10.2) comes after the vote (10.15)
    assert shop.discarded == ["faces/d/barber.jpg"]
    assert all(not p.faces for p in shop.engine.pools.values())


def test_a_face_from_a_latched_track_that_now_looks_like_staff_is_not_pooled():
    # a tracker id swap: the seated customer's latched track (never staff by vote) now shows the barber
    shop = Shop()
    walk_in_and_sit(shop)
    assert shop.engine.books["barber-l"].tracks[7].customer
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT}, apps=[staff_app("barber-l", 7, 149.5 + 0.1 * i, 0.9) for i in range(3)],
              faces=[fc("barber-l", 7, 150, "swapped")])
    assert shop.discarded == ["faces/d/swapped.jpg"]
    assert all(not p.faces for p in shop.engine.pools.values())


def test_a_barber_bound_to_a_customer_mid_visit_is_purged_when_voted_staff():
    shop = Shop()
    walk_in_and_sit(shop)
    # the left camera loses the customer's track; the barber, not sampled yet, sits still in the seat
    # and is claimed as the customer's new track (seat claim)
    shop.run(150, 154, c={1: C_CHAIR}, l={8: L_SEAT})
    pid = shop.engine.books["barber-c"].tracks[1].person
    assert shop.engine.books["barber-l"].tracks[8].person == pid
    shop.step(154, c={1: C_CHAIR}, l={8: L_SEAT},
              apps=[body_app("barber-l", 8, 154, "barber-body", 0.5)], faces=[fc("barber-l", 8, 154, "barber-face")])
    (pool,) = shop.engine.pools.values()
    assert {c.crop_path for c in pool.faces + pool.bodies} == {"faces/d/barber-face.jpg", "bodies/d/barber-body.jpg"}
    shop.step(155, c={1: C_CHAIR}, l={8: L_SEAT}, apps=[staff_app("barber-l", 8, 155 + 0.1 * i, 0.99) for i in range(1, 6)])
    assert {"faces/d/barber-face.jpg", "bodies/d/barber-body.jpg"} <= set(shop.discarded)
    assert pool.faces == [] and pool.bodies == []
    shop.run(156, 160, c={1: C_CHAIR}, l={8: L_SEAT})
    assert shop.engine.books["barber-l"].tracks[8].person is None
    shop.run(160, 170, c={1: C_CHAIR})     # (a 10 s gap in the centre's frames would end its tracks)
    shop.step(170, c={1: C_CHAIR}, faces=[fc("barber-c", 1, 170, "customer")])
    shop.run(171, 201, c={1: C_CHAIR})
    shop.run(201, 231, c={1: C_AISLE})
    shop.run(231, 234, c={1: C_DOOR})
    shop.run(234, 237, c={1: C_STREET})
    shop.run(237, 245)
    (reg,) = shop.registrations
    assert [f.crop_path for f in reg.faces] == ["faces/d/customer.jpg"] and reg.bodies == ()


def test_a_centre_seat_of_a_side_owned_chair_starts_no_haircut():
    # 2026-09-26 (Task 5a): c-seat-1 drawn in the centre for L5, which barber-r owns
    zones = dict(ZONES, **{"barber-c": ZONES["barber-c"] + [
        {"id": "c-s1", "name": "seat 1 (centre)", "type": "seat", "points": [[0.05, 0.3], [0.15, 0.3], [0.15, 0.45],
                                                                             [0.05, 0.45]], "seat_of": "c-ch1"}]})
    shop = Shop()
    shop.engine = ShopEngine(zones, GEO, new_id=lambda: "id0")
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 10, c={1: C_DOOR})
    shop.run(10, 120, c={1: C_CHAIR}, apps_at=lambda s: [staff_app("barber-c", 1, s + 0.5, 0.1)] if s < 13 else [])
    assert shop.sessions == {}


def test_a_barber_voted_staff_outside_briefly_who_sits_still_may_start():
    # 2026-09-26 (Task 5a): the documented cost of staff_outside_confirm_s (20 s) -- a barber whose
    # staff vote held outside the seats for less than that, then sits still in the seat, starts a haircut
    shop = Shop()
    shop.run(0, 10, l={9: L_EDGE}, apps_at=lambda s: [staff_app("barber-l", 9, s + 0.5, 0.75)] if s < 3 else [])
    shop.run(10, 60, l={9: L_SEAT})
    (s,) = shop.sessions.values()
    assert s.chair_id == "ch1"


# ── 2026-09-26 (final B review C1): sidecar restarts and dead cameras ──────────

def test_a_track_id_reused_after_a_sidecar_restart_is_a_new_body():
    shop = Shop()
    walk_in_and_sit(shop)              # customer: centre 1, left 7 in the seat, haircut from 41
    for _ in range(150, 160):          # the sidecar restarts: no frames at all for 10 s
        shop.engine.observe([], [], POLICY)
    # frames resume with restarted ids: id 7 is now someone at the chair edge, the customer is 8
    shop.run(160, 200, c={3: C_CHAIR}, l={7: L_EDGE, 8: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1), staff_app("barber-c", 3, s + 0.5, 0.1)]
             if s < 165 else [])
    (v,) = shop.visits.values()
    b = shop.engine.books["barber-l"]
    t7, t8, t3 = b.tracks[7], b.tracks[8], shop.engine.books["barber-c"].tracks[3]
    assert t7.person is None and not t7.customer and t7.first_seen == ts(160)
    assert t8.person == t3.person == v.person_id and v.open
    (s,) = shop.sessions.values()
    assert s.open and s.started_at == at(41)


def dead_left(shop, s0, s1, c):
    """barber-l stops for good: only centre frames from s0."""
    for s in range(s0, s1):
        u = shop.engine.observe([frame("barber-c", s, c(s))], [], POLICY)
        shop.visits.update(u.visits)
        shop.sessions.update(u.sessions)


def test_a_dead_cameras_frozen_track_hands_off_nobody_and_shows_no_camera():
    shop = Shop()
    walk_in_and_sit(shop)
    dead_left(shop, 150, 200, lambda s: {1: C_CHAIR} if s < 160 else {})
    dead_left(shop, 200, 230, lambda s: {20: C_CHAIR})     # a new body in the chair, seen by the centre only
    (v,) = [v for v in shop.visits.values() if v.person_id == "id0"]
    assert shop.engine.books["barber-c"].tracks[20].person != v.person_id
    (c,) = [c for c in shop.engine.live()["customers"] if c["person_id"] == v.person_id]
    assert c["cameras"] == []


def test_a_camera_dead_for_camera_dead_s_releases_its_tracks_so_the_customer_is_lost():
    shop = Shop()
    walk_in_and_sit(shop)
    dead_left(shop, 150, 400, lambda s: {1: C_CHAIR} if s < 160 else {})
    (v,) = shop.visits.values()
    assert (v.open, v.exit_reason, v.exited_at) == (False, EXIT_LOST, at(159))
    assert shop.engine.books["barber-l"].tracks == {}


def test_a_seat_claim_ignores_a_frozen_cameras_track_elsewhere():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_AISLE}, l={7: L_SEAT})     # the centre's last frame: he looks up and away
    for s in range(151, 170):                          # the centre freezes; the side churns his track
        u = shop.engine.observe([frame("barber-l", s, {9: L_SEAT} if s >= 153 else {})], [], POLICY)
        shop.sessions.update(u.sessions)
    (v,) = shop.visits.values()
    assert shop.engine.books["barber-l"].tracks[9].person == v.person_id
    (s,) = shop.sessions.values()
    assert s.open


def test_live_reports_each_cameras_frame_age_and_the_scene_drops_a_dead_cameras_boxes():
    # 2026-09-26 (final B review M5): ages on the engine clock (the newest frame of any camera)
    shop = Shop()
    walk_in_and_sit(shop)
    dead_left(shop, 150, 160, lambda s: {1: C_CHAIR})
    assert shop.engine.live()["cameras"] == {"barber-c": {"last_frame_age_s": 0.0},
                                             "barber-l": {"last_frame_age_s": 10.0}}
    scene = shop.engine.scene(POLICY)
    assert scene["barber-l"] == [] and len(scene["barber-c"]) == 1


def test_scene_carries_test_points_and_why_a_seated_body_counts():
    shop = Shop()
    walk_in_and_sit(shop)
    left = {e["track_id"]: e for e in shop.engine.scene(POLICY)["barber-l"]}[7]
    assert len(left["anchor"]) == 2 and len(left["foot"]) == 2 and len(left["seat_line"]) == 2
    assert left["occupancy"] == {"chair": "ch1", "why": "counted"}


def test_scene_reason_for_a_body_in_a_chair_zone_of_a_camera_without_seats():
    shop = Shop()
    walk_in_and_sit(shop)
    centre = {e["track_id"]: e for e in shop.engine.scene(POLICY)["barber-c"]}[1]
    assert centre["occupancy"] == {"chair": "ch1", "why": "in_chair_zone"}


def test_scene_occupancy_not_still_when_body_just_arrived_in_seat():
    # Left track 7 appears at L_SEAT for one frame only: in seat but no 3s history yet
    shop = Shop()
    shop.step(0, l={7: L_SEAT})
    left = {e["track_id"]: e for e in shop.engine.scene(POLICY)["barber-l"]}[7]
    assert left["occupancy"] == {"chair": "ch1", "why": "not_still"}


def test_scene_occupancy_not_owner_when_camera_doesnt_own_the_chair():
    # Build a case where the left track's seat points hit a chair that centre owns
    # Temporarily swap ownership so centre owns ch1, then call _occupancy on a seated left track
    shop = Shop()
    walk_in_and_sit(shop)
    t7 = shop.engine.books["barber-l"].tracks[7]
    # Temporarily set centre as owner of ch1
    old_owner = shop.engine.owner.get("ch1")
    shop.engine.owner["ch1"] = "barber-c"
    occupancy = shop.engine._occupancy(t7, POLICY)
    shop.engine.owner["ch1"] = old_owner  # restore
    assert occupancy == {"chair": "ch1", "why": "not_owner"}


def test_scene_occupancy_cannot_sit_when_staff_outside():
    # A new left track 9 at L_SEAT with staff_outside=True set (seat_ok will return False)
    shop = Shop()
    shop.step(0, l={9: L_SEAT})
    t9 = shop.engine.books["barber-l"].tracks[9]
    # Manually set staff_outside to True to trigger the cannot_sit case
    t9.staff_outside = True
    occupancy = shop.engine._occupancy(t9, POLICY)
    assert occupancy == {"chair": "ch1", "why": "cannot_sit"}


# 2026-09-27 (39-min testbed spec, section 4): owned-area creation and the identity models

def test_owned_area_creates_a_person_on_a_side_camera_chair_under_option_a():
    def run(policy):
        shop = Shop(policy)
        shop.run(0, 10, l={7: L_EDGE}, apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.1)])
        return shop
    assert run(POLICY).visits == {}
    shop = run({**POLICY, "create_in_owned_areas": 1})
    (v,) = shop.visits.values()
    assert v.confirmed and v.entered_at == at(0)
    assert [(b.kind, b.camera, b.area) for b in shop.bindings] == [("owned_area", "barber-l", "ch1")]


def test_option_a_still_writes_a_waiting_binding_on_the_sofa():
    shop = Shop({**POLICY, "create_in_owned_areas": 1})
    shop.run(0, 12, c={1: C_SOFA}, apps_at=lambda s: [staff_app("barber-c", 1, s + 0.5, 0.1)])
    assert [(b.kind, b.area) for b in shop.bindings] == [("waiting", "sofa")]


def test_option_a_creates_nothing_in_a_chair_its_camera_does_not_own():
    shop = Shop({**POLICY, "create_in_owned_areas": 1})
    shop.run(0, 12, c={1: C_CHAIR}, apps_at=lambda s: [staff_app("barber-c", 1, s + 0.5, 0.1)])
    assert shop.visits == {}


def side_camera_goes_blind(policy, until, zones=None):
    """walk_in_and_sit, then the side camera sees nobody while the centre keeps him in the chair."""
    shop = Shop(policy)
    if zones is not None:
        shop.engine.set_zones(zones)
    walk_in_and_sit(shop)
    shop.run(150, until, c={1: C_CHAIR})
    (v,) = shop.visits.values()
    (s,) = shop.sessions.values()
    return v, s


def test_hybrid_model_takes_chair_presence_only_from_the_owner_camera():
    v, s = side_camera_goes_blind(POLICY, 190)
    assert v.open and s.ended_at is None
    v, s = side_camera_goes_blind({**POLICY, "identity_model": 1}, 190)
    assert v.open and s.ended_at is not None and s.ended_at <= at(150 + POLICY["chair_gap_s"])


def test_full_transfer_loses_a_person_only_the_non_owner_camera_still_sees():
    v, _ = side_camera_goes_blind({**POLICY, "identity_model": 1}, 240)
    assert v.open
    v, _ = side_camera_goes_blind({**POLICY, "identity_model": 2}, 240)
    assert not v.open and v.exit_reason == EXIT_LOST and v.exited_at == at(152)


def test_full_transfer_keeps_a_person_the_centre_sees_outside_every_zone():
    shop = Shop({**POLICY, "identity_model": 2})
    walk_in_and_sit(shop)
    shop.run(150, 240, c={1: C_AISLE})
    (v,) = shop.visits.values()
    assert v.open


def unowned_chair():
    return {"barber-c": ZONES["barber-c"],
            "barber-l": [{k: v for k, v in z.items() if k != "owner"} for z in ZONES["barber-l"]]}


def test_hybrid_model_counts_every_camera_for_a_chair_with_no_owner():
    # review focus: a misconfigured pair counts for every camera drawing it, never for none
    v, s = side_camera_goes_blind({**POLICY, "identity_model": 1}, 190, zones=unowned_chair())
    assert v.open and s.ended_at is None
    shop = Shop({**POLICY, "identity_model": 1})
    shop.engine.set_zones(unowned_chair())
    walk_in_and_sit(shop)
    shop.run(150, 190, l={7: L_SEAT})                   # now only the side camera sees him
    (s,) = shop.sessions.values()
    assert s.ended_at is None


def test_full_transfer_dates_the_loss_to_the_owner_camera_when_the_centre_goes_too():
    # fix round 1: the centre track (non-owner in ch1) ending later must not restart the lost timer
    shop = Shop({**POLICY, "identity_model": 2})
    walk_in_and_sit(shop)
    shop.run(150, 180, c={1: C_CHAIR})
    shop.run(180, 300)
    (v,) = shop.visits.values()
    assert not v.open and v.exit_reason == EXIT_LOST and v.exited_at == at(152)


def test_full_transfer_a_non_owner_rebind_does_not_clear_the_lost_time():
    shop = Shop({**POLICY, "identity_model": 2})
    walk_in_and_sit(shop)
    shop.run(150, 170, c={1: C_CHAIR}, apps_at=lambda s: [reid_app("barber-c", 1, s + 0.5, (1, 0))])
    shop.run(170, 171)
    shop.run(171, 300, c={2: C_CHAIR}, apps_at=lambda s: [reid_app("barber-c", 2, s + 0.5, (1, 0.05))])
    assert ("reattach", "barber-c", 2) in [(b.kind, b.camera, b.track_id) for b in shop.bindings]
    (v,) = shop.visits.values()
    assert not v.open and v.exit_reason == EXIT_LOST and v.exited_at == at(152)


def test_set_zones_recomputes_what_existing_tracks_count_for():
    shop = Shop({**POLICY, "identity_model": 1})
    shop.run(0, 3, c={1: C_CHAIR})
    t = shop.engine.books["barber-c"].tracks[1]
    assert t.counted == frozenset()
    shop.engine.set_zones(unowned_chair())
    assert t.owned == t.counted == {"ch1"}


# 2026-09-28 (39-min testbed spec, 6.1): chair resume. A haircut ends on its gap (the customer hidden
# behind the barber); a still body then sits in that seat on the owner camera. chair_resume_s decides
# whether it is the previous customer coming back (same person, same session reopened) or someone else.

RESUME = {**POLICY, "chair_resume_s": 180.0, "lost_timeout_s": 600.0}
FAR = (0.4, (1 - 0.4 ** 2) ** 0.5, 0)          # cosine 0.40 with (1, 0, 0): <= resume_sim_diff 0.50
MIDDLE = (0.6, 0.8, 0)                         # cosine 0.60: between diff and same (and < reid_match_min)


def gap_then_resit(shop, resit=172, c=None, vec=None, until=None):
    """walk_in_and_sit; the customer vanishes (his haircut ends at 149 on the 20 s gap); from `resit`
    a new side track 8 sits still in the seat (centre track 1 at `c` meanwhile, if given)."""
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.engine.identity.persons[v.person_id].reid = unit((1, 0, 0))
    shop.run(150, resit, c={1: c} if c else None)

    def apps(s):
        out = [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < resit + 3 else []
        return out + ([reid_app("barber-l", 8, s + 0.5, vec)] if vec else [])
    shop.run(resit, until or resit + 40, c={1: c} if c else None, l={8: L_SEAT}, apps_at=apps)
    return v


def sitter(shop):
    return shop.engine.books["barber-l"].tracks[8].person


def test_chair_resume_off_a_returning_body_is_a_new_person_and_session():
    shop = Shop({**RESUME, "chair_resume_s": 0.0})
    v = gap_then_resit(shop)
    assert sitter(shop) != v.person_id
    assert len(shop.sessions) == 2 and "chair_resume" not in [b.kind for b in shop.bindings]


def test_chair_resume_keeps_the_same_person_and_reopens_the_same_session():
    shop = Shop(RESUME)
    v = gap_then_resit(shop)
    assert sitter(shop) == v.person_id
    (s,) = shop.sessions.values()                  # the same session id, reopened -- no second one
    assert (s.visit_id, s.started_at, s.ended_at, s.open) == (v.id, at(41), None, True)
    assert (v.wait_s, v.service_started_at, v.state, v.haircuts) == (38, at(41), HAIRCUT, 0)
    assert shop.states() == [WAITING, HAIRCUT, SERVED, HAIRCUT]
    last = shop.bindings[-1]
    assert (last.kind, last.camera, last.track_id, last.person_id, last.area) == \
        ("chair_resume", "barber-l", 8, v.person_id, "ch1")
    assert not any(a.kind == "excessive_wait" for a in shop.alerts)
    shop.run(212, 260, l={8: L_SEAT})
    shop.run(260, 290)
    assert (s.ended_at, s.service_s, v.haircuts) == (at(259), 218, 1)   # the gap is kept inside it


def test_chair_resume_a_middling_embedding_is_still_the_previous_customer():
    shop = Shop(RESUME)
    v = gap_then_resit(shop, vec=MIDDLE)
    assert sitter(shop) == v.person_id
    assert shop.bindings[-1].kind == "chair_resume" and shop.bindings[-1].reid_sim == 0.6


def test_chair_resume_not_when_the_previous_customer_sits_on_the_sofa():
    shop = Shop(RESUME)
    v = gap_then_resit(shop, c=C_SOFA)
    assert sitter(shop) != v.person_id
    assert shop.engine.books["barber-c"].tracks[1].person == v.person_id
    assert len(shop.sessions) == 2 and "chair_resume" not in [b.kind for b in shop.bindings]


def test_chair_resume_not_when_the_sitter_looks_clearly_different():
    shop = Shop(RESUME)
    v = gap_then_resit(shop, vec=FAR)
    assert sitter(shop) != v.person_id
    assert len(shop.sessions) == 2


def test_chair_resume_compares_with_the_owner_cameras_reference_under_a_per_camera_gallery():
    shop = Shop({**RESUME, "reid_gallery": 1})
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    p = shop.engine.identity.persons[v.person_id]
    p.reid, p.reid_by_cam = unit((0, 0, 1)), {"barber-l": (unit((1, 0, 0)), 5)}   # blended: cosine 0
    shop.run(150, 172)
    shop.run(172, 212, l={8: L_SEAT}, apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1),
                                                          reid_app("barber-l", 8, s + 0.5, MIDDLE)])
    assert sitter(shop) == v.person_id                 # the owner camera's mean: 0.60, not a clear stranger
    assert shop.bindings[-1].kind == "chair_resume" and shop.bindings[-1].reid_sim == 0.6


def test_chair_resume_not_when_the_previous_customer_crossed_the_door_outward():
    shop = Shop({**RESUME, "exit_confirm_s": 120.0})
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.run(150, 160, c={1: C_AISLE})
    shop.run(160, 163, c={1: C_DOOR})
    shop.run(163, 172, c={1: C_STREET})              # crossed out after the haircut ended; still in view
    assert v.exit_crossed_at is not None and v.open
    shop.run(172, 212, c={1: C_STREET}, l={8: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < 175 else [])
    assert sitter(shop) != v.person_id
    assert "chair_resume" not in [b.kind for b in shop.bindings]


def test_chair_resume_not_outside_the_window():
    shop = Shop(RESUME)
    v = gap_then_resit(shop, resit=349 + 200)
    assert v.open and sitter(shop) != v.person_id
    assert "chair_resume" not in [b.kind for b in shop.bindings]


def test_chair_resume_on_the_wash_return_still_resumes_via_the_wash_rule():
    shop = Shop(RESUME)
    walk_in_and_sit(shop)
    shop.run(150, 181, c={1: C_WASH})
    shop.step(181, c={1: C_CHAIR}, l={8: L_EDGE}, apps=[staff_app("barber-l", 8, 181.5, 0.1)])
    shop.run(182, 301, c={1: C_CHAIR}, l={8: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < 184 else [])
    shop.run(301, 330, c={1: C_AISLE})
    (s,) = shop.sessions.values()
    assert (s.started_at, s.ended_at, s.wash_s) == (at(41), at(300), 33)
    assert shop.states() == [WAITING, HAIRCUT, WASHING, HAIRCUT, SERVED]
    assert "chair_resume" not in [b.kind for b in shop.bindings]


def back_from_the_wash_unbound(shop):
    """The customer goes to the wash (haircut suspended at 149), vanishes there, and comes back as a
    side track no handoff can bind (no track of his in the chair to hand off from)."""
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    shop.run(150, 175, c={1: C_WASH})
    shop.run(175, 215, l={8: L_SEAT}, apps_at=lambda s: [staff_app("barber-l", 8, s + 0.5, 0.1)] if s < 178 else [])
    return v


def test_chair_resume_off_a_customer_back_from_the_wash_unbound_is_a_new_person():
    shop = Shop({**RESUME, "chair_resume_s": 0.0})
    v = back_from_the_wash_unbound(shop)
    assert sitter(shop) != v.person_id and len(shop.sessions) == 2


def test_chair_resume_a_customer_back_from_the_wash_unbound_resumes_through_the_wash_rule():
    # 2026-09-28 (Task 14 fix round 1, ruling a): a haircut suspended at the wash counts as a recent end
    shop = Shop(RESUME)
    v = back_from_the_wash_unbound(shop)
    assert sitter(shop) == v.person_id
    (s,) = shop.sessions.values()
    assert (s.visit_id, s.started_at, s.ended_at, s.washing_since) == (v.id, at(41), None, None)
    since = shop.bindings[-1].at
    assert s.wash_s == (since - at(149)).total_seconds() > 0
    assert (shop.bindings[-1].kind, shop.bindings[-1].area) == ("chair_resume", "ch1")
    assert shop.states() == [WAITING, HAIRCUT, WASHING, HAIRCUT] and v.haircuts == 0


# 2026-09-28 (39-min testbed spec, 6.2): staff release recovery. A bound track whose staff vote turns
# True is released; staff_release_mode 1 voids the visit only when that track created the person and
# was his only track ever. Otherwise he is lost (and can come back); in his running haircut's chair a
# still, non-staff, unbound body in the seat is bound to him at once (staff_recover).

RECOVER = {**POLICY, "staff_release_mode": 1.0}
CUSTOMER = (1, 0, 0)


def swapped_on_the_sofa(shop):
    """A customer created by the door (centre track 1) sits on the sofa; track 1 ends and centre
    track 2 re-attaches to him by re-ID; at 40 the tracker hands track 2 to a barber (staff by 41)."""
    shop.run(0, 3, c={1: C_STREET})
    shop.run(3, 20, c={1: C_SOFA}, apps_at=lambda s: [reid_app("barber-c", 1, s + 0.5, CUSTOMER)])
    shop.run(20, 40, c={2: C_SOFA}, apps_at=lambda s: [reid_app("barber-c", 2, s + 0.5, (0.98, 0.1, 0))])
    (v,) = shop.visits.values()
    assert [b.kind for b in shop.bindings] == ["door", "reattach"]
    shop.step(40, c={2: C_SOFA}, apps=[staff_app("barber-c", 2, 40 + 0.1 * i, 0.99) for i in range(1, 4)])
    shop.step(41, c={2: C_SOFA})
    return v


def test_staff_release_mode_0_voids_a_customer_whose_reattached_track_turns_staff():
    shop = Shop()
    v = swapped_on_the_sofa(shop)
    assert v.outcome == VOID_STAFF and not v.open


def test_staff_release_mode_1_voids_a_barber_whose_own_track_created_him():
    # (b) he walked in, waited past wait_target_s (100) without a staff sample, then was recognized
    shop = Shop(RECOVER)
    shop.run(0, 3, c={5: C_STREET})
    shop.run(3, 115, c={5: C_SOFA})
    assert [a.kind for a in shop.alerts] == ["excessive_wait"] and shop.alerts[0].open
    shop.run(115, 120, c={5: C_SOFA},
             apps_at=lambda s: [staff_app("barber-c", 5, s + 0.5, 0.9)] if s < 118 else [])
    (v,) = shop.visits.values()
    assert v.outcome == VOID_STAFF and not v.open
    assert not shop.alerts[-1].open and shop.alerts[-1].visit_id == v.id


def test_staff_release_mode_1_keeps_a_customer_whose_second_track_turns_staff_open_and_lost():
    # (c) the id swap: his re-attached track goes to the barber -- not voided, lost from that frame
    shop = Shop(RECOVER)
    v = swapped_on_the_sofa(shop)
    p = shop.engine.identity.persons[v.person_id]
    assert v.open and v.outcome is None and v.state == WAITING
    assert p.tracks == {} and p.lost_since == ts(41)
    # he turns up again on the sofa as a new track: re-attach brings him back
    shop.run(42, 50, c={2: C_AISLE, 3: C_SOFA}, apps_at=lambda s: [reid_app("barber-c", 3, s + 0.5, CUSTOMER)])
    assert shop.engine.books["barber-c"].tracks[3].person == v.person_id and p.lost_since is None
    assert v.open and shop.bindings[-1].kind == "reattach"


def seat_claimed_then_swapped(shop, recover_body, vec=None):
    """walk_in_and_sit; from 150 the left camera loses track 7 and side track 8 sits in the seat and
    is claimed (seat claim) as the customer. At 154 the tracker hands track 8 to the barber, who stands
    up to the chair's edge; if recover_body, the customer sits on as a new track 9 in the seat. Track
    8's staff vote turns True at 158. The centre view loses the customer from 150. With vec, the
    customer's re-ID is (1, 0, 0) and track 9's embedding is vec."""
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    if vec:
        shop.engine.identity.persons[v.person_id].reid = unit(CUSTOMER)
    shop.run(150, 154, l={8: L_SEAT})
    assert shop.engine.books["barber-l"].tracks[8].person == v.person_id
    seat = {9: L_SEAT} if recover_body else {}
    emb = (lambda s: [reid_app("barber-l", 9, s + 0.5, vec)]) if vec and recover_body else (lambda s: [])
    shop.run(154, 157, l={8: L_EDGE, **seat}, apps_at=emb)
    shop.step(157, l={8: L_EDGE, **seat},
              apps=[staff_app("barber-l", 8, 157 + 0.1 * i, 0.99) for i in range(1, 4)] + emb(157))
    shop.step(158, l={8: L_EDGE, **seat}, apps=emb(158))
    return v


def test_staff_release_mode_1_a_lost_customer_rejoins_by_seat_claim_and_the_visit_continues():
    # (c) in the chair: his claimed side track turns out to be the barber; nobody left in the seat
    shop = Shop(RECOVER)
    v = seat_claimed_then_swapped(shop, recover_body=False)
    p = shop.engine.identity.persons[v.person_id]
    assert v.open and p.tracks == {} and p.lost_since == ts(158)
    shop.run(159, 165, l={8: L_EDGE, 10: L_SEAT})       # he sits back down as track 10
    assert shop.engine.books["barber-l"].tracks[10].person == v.person_id and p.lost_since is None
    assert shop.bindings[-1].kind == "seat_claim"
    (s,) = shop.sessions.values()
    assert v.open and s.ended_at is None and v.state == HAIRCUT


def test_staff_release_mode_1_binds_the_body_still_in_his_seat_at_once():
    # (d) the customer sat on in the seat as track 9 (declined as a claim: track 8 was his there)
    shop = Shop(RECOVER)
    v = seat_claimed_then_swapped(shop, recover_body=True)
    assert shop.engine.books["barber-l"].tracks[9].person == v.person_id
    last = shop.bindings[-1]
    assert (last.kind, last.camera, last.track_id, last.person_id, last.area, last.at) == \
        ("staff_recover", "barber-l", 9, v.person_id, "ch1", at(158))
    (s,) = shop.sessions.values()
    assert s.ended_at is None and v.state == HAIRCUT and shop.engine.identity.persons[v.person_id].lost_since is None


def test_staff_release_mode_1_a_customer_nobody_brings_back_ends_lost_with_his_times():
    # (e) lost at 41 (the release); lost_timeout_s is 60
    shop = Shop(RECOVER)
    v = swapped_on_the_sofa(shop)
    shop.run(42, 100, c={2: C_SOFA})
    assert v.open
    shop.run(100, 103, c={2: C_SOFA})
    assert (v.outcome, v.exit_reason, v.entered_at, v.exited_at, v.wait_s) == (LOST, EXIT_LOST, at(3), at(41), 38)


def test_staff_release_mode_1_binds_a_sitter_who_looks_like_him():
    # fix round 1: the re-ID veto (reid_veto 0.3 here) lets a similar sitter through
    shop = Shop(RECOVER)
    v = seat_claimed_then_swapped(shop, recover_body=True, vec=(0.35, (1 - 0.35 ** 2) ** 0.5, 0))
    last = shop.bindings[-1]
    assert (last.kind, last.track_id, last.person_id, last.reid_sim) == ("staff_recover", 9, v.person_id, 0.35)


def test_staff_release_mode_1_does_not_bind_a_sitter_the_reid_veto_rules_out():
    # fix round 1: cosine 0.25 < reid_veto 0.3 against the owner camera's reference -- as the seat claim
    shop = Shop(RECOVER)
    v = seat_claimed_then_swapped(shop, recover_body=True, vec=(0.25, (1 - 0.25 ** 2) ** 0.5, 0))
    p = shop.engine.identity.persons[v.person_id]
    assert shop.engine.books["barber-l"].tracks[9].person is None
    assert "staff_recover" not in [b.kind for b in shop.bindings]
    assert p.tracks == {} and p.lost_since == ts(158) and v.open


# 2026-09-28 (39-min testbed spec, 6.6): seat_claim_mode 2 -- the barber leaning over the chair has
# the customer's left-camera track; the customer, seated, is a new track in the seat

def barber_holds_the_customers_track(policy):
    shop = Shop(policy)
    walk_in_and_sit(shop)
    (v,) = shop.visits.values()
    p = shop.engine.identity.persons[v.person_id]
    p.reid_by_cam["barber-l"] = (unit(CUSTOMER), 5)
    shop.engine.books["barber-l"].tracks[7].reid = unit((0.3, 0.95, 0))
    shop.run(150, 160, c={1: C_CHAIR}, l={7: L_EDGE, 9: L_SEAT},
             apps_at=lambda s: [reid_app("barber-l", 9, s + 0.5, CUSTOMER)])
    return shop, v


def test_seat_claim_mode_0_leaves_the_seated_customer_unjoined():
    shop, v = barber_holds_the_customers_track({**POLICY, "reid_gallery": 1})
    books = shop.engine.books["barber-l"]
    assert books.tracks[7].person == v.person_id and books.tracks[9].person is None


def test_seat_claim_mode_2_swaps_the_customer_onto_the_seated_body():
    shop, v = barber_holds_the_customers_track({**POLICY, "reid_gallery": 1, "seat_claim_mode": 2})
    books = shop.engine.books["barber-l"]
    assert books.tracks[9].person == v.person_id and books.tracks[7].person is None
    assert [b.kind for b in shop.bindings].count("seat_swap") == 1 and v.open


def test_a_customer_first_seen_sitting_in_the_chair_has_an_unseen_wait():
    # 2026-09-28 (user): his wait isn't 0 s, it's unknown -- the stats leave it out
    shop = Shop()
    shop.run(0, 20, l={7: L_SEAT}, apps_at=lambda s: [staff_app("barber-l", 7, s + 0.5, 0.1)] if s < 3 else [])
    (v,) = shop.visits.values()
    assert v.service_started_at is not None and v.wait_unseen
    shop2 = Shop()
    walk_in_and_sit(shop2)
    (w,) = shop2.visits.values()
    assert not w.wait_unseen


def test_waiting_now_counts_the_bodies_the_centre_camera_sees_in_the_waiting_area():
    # 2026-09-28 (user): as simple as that -- no visit state, no identity
    shop = Shop()
    shop.run(0, 3, c={1: C_SOFA, 2: C_SOFA, 3: C_CHAIR, 4: C_SOFA}, l={7: L_SEAT},
             apps_at=lambda s: [staff_app("barber-c", 4, s + 0.1 * i, 0.95) for i in range(1, 4)])
    assert shop.engine.live()["customers_waiting"] == 2            # track 4, voted staff, is a barber


# 2026-09-28 (user): barber_cover -- the barber leaning over the seated customer hides him from every
# camera; a body voted staff in the seat (owner camera) keeps the haircut going while he's seen nowhere

def barber_covers(policy, centre=None):
    shop = Shop(policy)
    walk_in_and_sit(shop)
    shop.run(150, 200, c=centre, l={9: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", 9, s + 0.1 * i, 0.95) for i in range(1, 4)])
    (v,) = shop.visits.values()
    return shop.engine.chairs.chairs["ch1"].person == v.person_id


def test_barber_cover_0_the_haircut_ends_while_the_barber_hides_the_customer():
    assert not barber_covers(POLICY)


def test_barber_cover_1_a_barber_in_the_seat_keeps_the_haircut_while_the_customer_is_unseen():
    assert barber_covers({**POLICY, "barber_cover": 1})


def test_barber_cover_1_does_not_apply_while_the_customer_is_seen_elsewhere():
    assert not barber_covers({**POLICY, "barber_cover": 1}, centre={1: C_SOFA})


# 2026-09-29 (user): washing_state 0 -- no customer is ever shown washing (flagged to fix later)

def test_washing_state_0_a_wash_trip_never_shows_washing():
    shop = Shop({**POLICY, "washing_state": 0})
    walk_in_and_sit(shop)
    shop.run(150, 181, c={1: C_WASH})
    assert WASHING not in shop.states()
    assert all(s.washing_since is None for s in shop.engine.customers.sessions.values())


def test_washing_state_0_ends_a_wash_already_in_progress():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.run(150, 175, c={1: C_WASH})
    assert shop.states()[-1] == WASHING
    shop.policy = {**POLICY, "washing_state": 0}
    shop.run(175, 177, c={1: C_WASH})
    assert shop.states()[-1] != WASHING
    assert not any(s.washing_since for s in shop.engine.customers.sessions.values())


# 2026-09-29 (user): a customer's crops must be his. body_from_seat 1: a body crop joins the visit only when
# its track sits in the seat of the visit's running haircut (body re-ID can't tell caped customers apart,
# so where the crop came from decides); face_min_quality: faces below it never join (noisy embeddings)

def test_body_from_seat_1_keeps_only_bodies_seated_in_his_own_chair():
    shop = Shop({**POLICY, "body_from_seat": 1})
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT},
              apps=[body_app("barber-l", 7, 150, "seated", 0.2), body_app("barber-c", 1, 150, "centre", 0.2)])
    shop.step(151, c={1: C_CHAIR}, l={7: L_EDGE}, apps=[body_app("barber-l", 7, 151, "up", 0.2)])
    (pool,) = shop.engine.pools.values()
    assert [b.crop_path for b in pool.bodies] == ["bodies/d/seated.jpg"]
    assert sorted(shop.discarded) == ["bodies/d/centre.jpg", "bodies/d/up.jpg"]


def test_body_from_seat_0_keeps_todays_behaviour():
    shop = Shop()
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT}, apps=[body_app("barber-c", 1, 150, "centre", 0.2)])
    (pool,) = shop.engine.pools.values()
    assert [b.crop_path for b in pool.bodies] == ["bodies/d/centre.jpg"]


def test_face_min_quality_drops_a_poor_face():
    shop = Shop({**POLICY, "face_min_quality": 0.3})
    walk_in_and_sit(shop)
    shop.step(150, c={1: C_CHAIR}, l={7: L_SEAT}, faces=[fc("barber-c", 1, 150, "good", q=0.6), fc("barber-c", 1, 150, "poor", q=0.1)])
    (pool,) = shop.engine.pools.values()
    assert [f.crop_path for f in pool.faces] == ["faces/d/good.jpg"] and "faces/d/poor.jpg" in shop.discarded
