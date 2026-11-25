import itertools

from identity import IdentityBook, may_claim_seat, reference, seat_claim_displaces
from shop_helpers import (C_AISLE, C_CHAIR, C_SOFA, C_WASH, GEO, L_AISLE, L_EDGE, L_SEAT, POLICY, ZONES, frame,
                          ts)
from tracklets import TrackletBook, unit

CAMS_OF_AREA = {"sofa": {"barber-c"}, "ch1": {"barber-c", "barber-l"}, "wash": {"barber-c"}}


def books():
    return TrackletBook("barber-c", ZONES["barber-c"], GEO), TrackletBook("barber-l", ZONES["barber-l"], GEO)


def new_book():
    n = itertools.count()
    return IdentityBook(lambda: f"p{next(n)}")


def test_handoff_binds_the_side_track_to_the_person_in_the_paired_zone():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 1, {7: L_EDGE}), POLICY)
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], POLICY)
    assert m.person is p and m.offset_s == 1.0 and m.other is c.tracks[1] and m.kind == "handoff"
    idb.bind(l.tracks[7], p, ts(1))
    assert p.tracks == {"barber-c": c.tracks[1], "barber-l": l.tracks[7]} and l.tracks[7].person == p.id


def test_handoff_skips_a_person_who_already_has_a_body_in_that_camera():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    l.observe(frame("barber-l", 0, {6: L_SEAT, 7: L_EDGE}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[6], p, ts(0))
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], POLICY) is None


def test_two_candidates_without_reid_wait_for_evidence():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR, 2: (0.12, 0.42)}), POLICY)
    idb.create(c.tracks[1], ts(0))
    idb.create(c.tracks[2], ts(0))
    l.observe(frame("barber-l", 0, {7: L_EDGE}), POLICY)
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1], c.tracks[2]], POLICY) is None


def test_reid_picks_between_candidates_and_vetoes_a_mismatch():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR, 2: (0.12, 0.42)}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    q = idb.create(c.tracks[2], ts(0))
    p.reid, q.reid = unit((1, 0, 0)), unit((0, 1, 0))
    l.observe(frame("barber-l", 0, {7: L_EDGE, 8: (0.47, 0.47)}), POLICY)
    l.tracks[7].reid = unit((0.9, 0.1, 0))
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1], c.tracks[2]], POLICY)
    assert m.person is p and m.reid_sim > 0.99
    l.tracks[8].reid = unit((0, 0, 1))                  # like neither: both vetoed
    assert idb.find_handoff(l.tracks[8], "ch1", [c.tracks[1], c.tracks[2]], POLICY) is None


def test_handoff_rule_0_still_counts_a_partner_already_inside_for_minutes():
    # 2026-09-28 (39-min testbed spec, 6.3): today's behaviour, unchanged
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 60, {7: L_EDGE}), POLICY)
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], POLICY)
    assert m is not None and m.person is p


def test_handoff_rule_1_rejects_a_partner_that_entered_60s_before():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 60, {7: L_EDGE}), POLICY)
    rule1 = {**POLICY, "handoff_rule": 1}
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule1) is None


def test_handoff_rule_1_accepts_a_partner_within_the_sync_window():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 1, {7: L_EDGE}), POLICY)
    rule1 = {**POLICY, "handoff_rule": 1}
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule1)
    assert m is not None and m.person is p


def test_handoff_rule_2_requires_the_new_track_to_be_still_in_a_chair_area():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 0, {7: L_EDGE}), POLICY)
    rule2 = {**POLICY, "handoff_rule": 2}
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule2, still=False, area_type="chair") is None
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule2, still=True, area_type="chair")
    assert m is not None and m.person is p


def test_handoff_rule_3_requires_the_new_track_to_be_still_in_the_chairs_seat():
    # 2026-09-28: under rule 2 the barber standing still by the chair took the walking customer's handoff
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 0, {7: L_EDGE, 8: L_SEAT}), POLICY)
    rule3 = {**POLICY, "handoff_rule": 3}
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule3, still=True, area_type="chair") is None
    assert idb.find_handoff(l.tracks[8], "ch1", [c.tracks[1]], rule3, still=False, area_type="chair") is None
    m = idb.find_handoff(l.tracks[8], "ch1", [c.tracks[1]], rule3, still=True, area_type="chair")
    assert m is not None and m.person is p


def test_handoff_rule_4_takes_a_track_that_stayed_in_the_seat_even_if_it_jitters():
    # 2026-09-28: the customer being worked on jitters; time in the seat, not stillness, tells him from the barber
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    for s in range(0, 5):
        l.observe(frame("barber-l", s, {7: L_EDGE, 8: (0.6 + (0.05 if s % 2 else 0), 0.65)}), POLICY)
    rule4 = {**POLICY, "handoff_rule": 4, "sync_window_s": 10}
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule4, still=True, area_type="chair") is None
    m = idb.find_handoff(l.tracks[8], "ch1", [c.tracks[1]], rule4, still=False, area_type="chair")
    assert m is not None and m.person is p
    l2 = TrackletBook("barber-l", ZONES["barber-l"], GEO)
    l2.observe(frame("barber-l", 0, {9: L_SEAT}), POLICY)           # in the seat only just now
    assert idb.find_handoff(l2.tracks[9], "ch1", [c.tracks[1]], rule4, still=True, area_type="chair") is None


def test_handoff_rule_2_does_not_require_stillness_outside_a_chair_area():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 0, {7: L_EDGE}), POLICY)
    rule2 = {**POLICY, "handoff_rule": 2}
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], rule2, still=False, area_type="waiting")
    assert m is not None and m.person is p


def test_reattach_a_lost_person_by_reid_where_they_were_last_seen():
    c, _ = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    p.reid = unit((1, 0))
    idb.release(c.tracks[1], ts(5))
    assert p.lost_since == ts(5) and p.gone_from == {"barber-c": ts(5)}
    c.observe(frame("barber-c", 10, {2: C_SOFA}), POLICY)
    c.tracks[2].reid = unit((0.95, 0.05))
    m = idb.find_reattach(c.tracks[2], ts(10), CAMS_OF_AREA, POLICY)
    assert m.person is p and m.kind == "reattach"
    assert idb.find_reattach(c.tracks[2], ts(200), CAMS_OF_AREA, POLICY) is None   # lost too long


def test_reattach_in_one_camera_while_still_seen_in_another():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[7], p, ts(0))
    p.reid = unit((1, 0))
    idb.release(l.tracks[7], ts(3))                     # the side camera's track churned
    assert p.lost_since is None
    l.observe(frame("barber-l", 4, {9: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.1))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY).person is p


# 2026-09-29: reattach_anchor 1 -- while his haircut runs in a chair, a customer is re-attached only by a
# track in that chair (a body in the next chair or in the aisle scored 0.83-0.86 and took him, U5 R1)

def lost_seated_customer():
    c, l = books()
    idb = new_book()
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    p = idb.create(l.tracks[7], ts(0))
    p.reid = unit((1, 0))
    p.reid_by_cam["barber-l"] = (p.reid, 5)
    idb.release(l.tracks[7], ts(3))
    return l, idb, p


def test_reattach_anchor_0_a_body_in_the_aisle_takes_a_lost_seated_customer():
    l, idb, p = lost_seated_customer()
    l.observe(frame("barber-l", 4, {9: L_AISLE}), POLICY)
    l.tracks[9].reid = unit((1, 0.1))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY, chairs={p.id: "ch1"}).person is p


def test_reattach_anchor_1_only_a_track_in_his_chair_takes_a_customer_with_a_running_haircut():
    l, idb, p = lost_seated_customer()
    anchored = {**POLICY, "reattach_anchor": 1}
    l.observe(frame("barber-l", 4, {9: L_AISLE, 10: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.1))
    l.tracks[10].reid = unit((1, 0.1))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, anchored, chairs={p.id: "ch1"}) is None
    assert idb.find_reattach(l.tracks[10], ts(4), CAMS_OF_AREA, anchored, chairs={p.id: "ch1"}).person is p
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, anchored, chairs={}).person is p   # no haircut: as today


def test_reattach_anchor_2_refuses_only_a_track_in_another_chair():
    l, idb, p = lost_seated_customer()
    anchored = {**POLICY, "reattach_anchor": 2}
    l.observe(frame("barber-l", 4, {9: L_AISLE, 10: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.1))
    l.tracks[10].reid = unit((1, 0.1))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, anchored, AREA_TYPE, chairs={p.id: "ch1"}).person is p
    assert idb.find_reattach(l.tracks[10], ts(4), CAMS_OF_AREA, anchored, AREA_TYPE, chairs={p.id: "ch2"}) is None


def test_no_reattach_into_an_area_other_than_where_the_person_is_seen():
    # 2026-09-24 (Task 16 tune): one person was on chair R1 in one camera and chair L3 in another
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[7], p, ts(0))
    p.reid = unit((1, 0))
    idb.release(l.tracks[7], ts(3))                     # still on the sofa in the centre view
    l.observe(frame("barber-l", 4, {9: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.1))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY) is None


AREA_TYPE = {"sofa": "waiting", "ch1": "chair", "ch2": "chair", "wash": "washing"}


def lost_from(area_pt, area_type=AREA_TYPE):
    """A person lost after being seen at area_pt (then stepping out of every zone)."""
    c, _ = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: area_pt}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    p.reid = unit((1, 0))
    c.observe(frame("barber-c", 1, {1: (0.5, 0.2)}), POLICY)   # stepped out of every zone...
    idb.note(c.tracks[1])
    idb.release(c.tracks[1], ts(2))
    return c, idb, p


def test_a_lost_person_keeps_the_last_area_and_reattaches_in_the_next_one():
    # review round 1 (spec decision 6, "zones next to where they were lost"): sofa <-> chair <-> wash
    c, idb, p = lost_from(C_SOFA)
    assert p.last_areas == {"sofa"}                            # the last area is kept
    c.observe(frame("barber-c", 5, {2: C_CHAIR}), POLICY)
    c.tracks[2].reid = unit((1, 0.05))
    assert idb.find_reattach(c.tracks[2], ts(5), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p
    c.observe(frame("barber-c", 6, {3: (0.5, 0.2)}), POLICY)   # somewhere unzoned: can't tell, allowed
    c.tracks[3].reid = unit((1, 0.05))
    assert idb.find_reattach(c.tracks[3], ts(6), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p


def test_a_person_lost_in_one_chair_does_not_reattach_in_the_next_chair():
    c, idb, p = lost_from(C_CHAIR)
    assert p.last_areas == {"ch1"}
    c.observe(frame("barber-c", 5, {2: C_SOFA}), POLICY)
    t = c.tracks[2]
    t.reid, t.areas = unit((1, 0.05)), frozenset({"ch2"})    # the neighbouring chair
    assert idb.find_reattach(t, ts(5), CAMS_OF_AREA, POLICY, AREA_TYPE) is None
    t.areas = frozenset({"wash"})                              # chair -> wash is next door
    assert idb.find_reattach(t, ts(5), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p


def test_a_person_lost_on_the_sofa_does_not_reattach_at_the_wash():
    c, idb, p = lost_from(C_SOFA)
    c.observe(frame("barber-c", 5, {2: C_WASH}), POLICY)
    c.tracks[2].reid = unit((1, 0.05))
    assert idb.find_reattach(c.tracks[2], ts(5), CAMS_OF_AREA, POLICY, AREA_TYPE) is None


def seated_in_side_camera():
    """A person seen right now in chair ch1's seat by the side camera, whose centre track churned."""
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    p = idb.create(l.tracks[7], ts(0))
    idb.bind(c.tracks[1], p, ts(0))
    p.reid = unit((1, 0))
    idb.release(c.tracks[1], ts(3))
    return c, idb, p


def test_a_track_in_no_zone_does_not_reattach_to_someone_seen_in_a_chair_right_now():
    # 2026-09-26 (Task 5a): raw1 replay -- a centre track in no zone re-attached (sim 0.83-0.89) to
    # the L2 customer seated in the side view, walked into R1, and the two chairs ping-ponged
    c, idb, p = seated_in_side_camera()
    c.observe(frame("barber-c", 4, {2: C_AISLE}), POLICY)
    c.tracks[2].reid = unit((1, 0.05))
    assert c.tracks[2].areas == frozenset()
    assert idb.find_reattach(c.tracks[2], ts(4), CAMS_OF_AREA, POLICY, AREA_TYPE) is None


def test_a_track_in_no_zone_reattaches_to_someone_who_left_the_chair():
    # fix round 1: last_areas is sticky -- a customer up from the chair, walking the aisle in the
    # side view (no zone), is still found by the centre camera's track in no zone
    c, idb, p = seated_in_side_camera()
    l = TrackletBook("barber-l", ZONES["barber-l"], GEO)
    l.tracks = {7: p.tracks["barber-l"]}
    l.observe(frame("barber-l", 3, {7: L_AISLE}), POLICY)
    idb.note(l.tracks[7])
    assert p.areas == frozenset() and p.last_areas == {"ch1"}
    c.observe(frame("barber-c", 4, {2: C_AISLE}), POLICY)
    c.tracks[2].reid = unit((1, 0.05))
    assert idb.find_reattach(c.tracks[2], ts(4), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p


def test_a_track_in_the_same_chair_still_reattaches_to_someone_seen_there():
    c, idb, p = seated_in_side_camera()
    c.observe(frame("barber-c", 4, {2: C_CHAIR}), POLICY)
    c.tracks[2].reid = unit((1, 0.05))
    assert idb.find_reattach(c.tracks[2], ts(4), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p


def test_a_track_in_no_zone_still_reattaches_to_someone_seen_on_the_sofa():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_AISLE}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[7], p, ts(0))
    p.reid = unit((1, 0))
    idb.release(l.tracks[7], ts(3))
    l.observe(frame("barber-l", 4, {9: L_AISLE}), POLICY)
    l.tracks[9].reid = unit((1, 0.05))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY, AREA_TYPE).person is p


def test_reattach_needs_reid():
    c, _ = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    p.reid = unit((1, 0))
    idb.release(c.tracks[1], ts(5))
    c.observe(frame("barber-c", 6, {2: C_SOFA}), POLICY)
    assert idb.find_reattach(c.tracks[2], ts(6), CAMS_OF_AREA, POLICY) is None


def test_drop_retires_the_persons_tracks():
    c, _ = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.drop(p.id)
    assert p.id not in idb.persons and c.tracks[1].person is None and c.tracks[1].retired


# The seat claim (engine step 6): an unbound body sitting still in a running haircut's seat
# joins the haircut's customer only if that can be him (review #2).

def claim_setup(centre_pt=None):
    """The customer of ch1's haircut, optionally still tracked by the centre camera at
    centre_pt, and an unbound body sitting in the side camera's seat."""
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: centre_pt or C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    if centre_pt is None:
        idb.release(c.tracks[1], ts(1))                 # lost: no live track anywhere
    l.observe(frame("barber-l", 2, {8: L_SEAT}), POLICY)
    return l.tracks[8], p


def test_a_seat_claim_may_bind_a_customer_with_no_live_track():
    t, p = claim_setup()
    assert may_claim_seat(t, p, "ch1", POLICY)


def test_a_seat_claim_may_bind_a_customer_seen_only_in_that_chair():
    t, p = claim_setup(C_CHAIR)
    assert may_claim_seat(t, p, "ch1", POLICY)


def test_no_seat_claim_for_a_customer_seen_in_another_area():
    t, p = claim_setup(C_SOFA)
    assert not may_claim_seat(t, p, "ch1", POLICY)


def test_no_seat_claim_for_a_customer_seen_out_of_every_area():
    t, p = claim_setup((0.25, 0.7))                     # the aisle: in no zone, not in the chair
    assert not may_claim_seat(t, p, "ch1", POLICY)


def test_no_seat_claim_for_a_customer_with_a_body_in_that_camera_already():
    t, p = claim_setup(C_CHAIR)
    other = TrackletBook("barber-l", ZONES["barber-l"], GEO)
    other.observe(frame("barber-l", 2, {7: L_EDGE}), POLICY)   # in the chair zone, next to the seat
    p.tracks["barber-l"] = other.tracks[7]
    assert not may_claim_seat(t, p, "ch1", POLICY)


def test_seat_claim_reid_veto_just_either_side():
    t, p = claim_setup()
    p.reid = unit((1, 0))
    t.reid = unit((0.35, (1 - 0.35 ** 2) ** 0.5))      # similarity 0.35, veto 0.3: may be him
    assert may_claim_seat(t, p, "ch1", POLICY)
    t.reid = unit((0.25, (1 - 0.25 ** 2) ** 0.5))      # similarity 0.25: not him
    assert not may_claim_seat(t, p, "ch1", POLICY)


def test_a_seat_claim_without_both_embeddings_is_not_vetoed():
    t, p = claim_setup()
    t.reid = unit((0, 1))                               # the customer has none yet
    assert may_claim_seat(t, p, "ch1", POLICY)
    t.reid, p.reid = None, unit((1, 0))
    assert may_claim_seat(t, p, "ch1", POLICY)


# 2026-09-27 (39-min testbed spec, section 4): the re-ID gallery switch

def test_per_camera_gallery_skips_the_veto_when_the_person_has_no_reference_in_that_camera():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    c.tracks[1].reid, c.tracks[1].reid_n = unit((1, 0, 0)), 1
    p = idb.create(c.tracks[1], ts(0))
    l.observe(frame("barber-l", 1, {7: L_EDGE}), POLICY)
    l.tracks[7].reid, l.tracks[7].reid_n = unit((0, 1, 0)), 1          # cross-camera: unrelated vector
    assert idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], POLICY) is None            # blended: vetoed
    m = idb.find_handoff(l.tracks[7], "ch1", [c.tracks[1]], {**POLICY, "reid_gallery": 1})
    assert m is not None and m.person is p and m.reid_sim is None                       # no left reference


def test_per_camera_gallery_reattaches_by_the_tracks_own_camera_reference():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    c.tracks[1].reid, c.tracks[1].reid_n = unit((0, 1)), 20              # centre samples dominate
    l.tracks[7].reid, l.tracks[7].reid_n = unit((1, 0)), 2
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[7], p, ts(0))
    assert set(p.reid_by_cam) == {"barber-c", "barber-l"}
    idb.release(l.tracks[7], ts(3))                     # the side camera's track churned
    l.observe(frame("barber-l", 4, {9: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.05))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY) is None          # blended: too far
    m = idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, {**POLICY, "reid_gallery": 1})
    assert m is not None and m.person is p and m.reid_sim >= POLICY["reid_match_min"]


def test_per_camera_gallery_never_reattaches_without_a_reference_in_that_camera():
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    l.observe(frame("barber-l", 0, {7: L_SEAT}), POLICY)
    c.tracks[1].reid, c.tracks[1].reid_n = unit((1, 0)), 5
    p = idb.create(c.tracks[1], ts(0))
    idb.bind(l.tracks[7], p, ts(0))                     # the side track had no sample
    idb.release(l.tracks[7], ts(3))
    l.observe(frame("barber-l", 4, {9: L_SEAT}), POLICY)
    l.tracks[9].reid = unit((1, 0.05))
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, POLICY).person is p
    assert idb.find_reattach(l.tracks[9], ts(4), CAMS_OF_AREA, {**POLICY, "reid_gallery": 1}) is None


def test_per_camera_gallery_seat_claim_without_a_reference_there_is_not_vetoed():
    t, p = claim_setup()
    p.reid = unit((1, 0))
    t.reid = unit((0, 1))
    assert not may_claim_seat(t, p, "ch1", POLICY)
    assert may_claim_seat(t, p, "ch1", {**POLICY, "reid_gallery": 1})


def test_add_reid_keeps_per_camera_and_owned_only_means():
    c, _ = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_SOFA}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    idb.add_reid(p, unit((1, 0)), "barber-c", owned=False)
    assert p.reid_owned is None and p.reid_by_cam["barber-c"][1] == 1
    assert reference(p, "barber-c", {**POLICY, "reid_gallery": 1}) == p.reid_by_cam["barber-c"][0]
    assert reference(p, "barber-l", {**POLICY, "reid_gallery": 1}) is None
    assert reference(p, "barber-c", {**POLICY, "reid_gallery": 2}) is None
    idb.add_reid(p, unit((0, 1)), "barber-c", owned=True)
    assert p.reid_owned == unit((0, 1)) and p.reid_owned_n == 1
    assert reference(p, "barber-c", {**POLICY, "reid_gallery": 2}) == p.reid_owned
    assert reference(p, "barber-c", POLICY) == p.reid and p.reid_n == 2


def test_a_person_remembers_the_track_that_created_him_and_whether_another_was_bound():
    # 2026-09-28 (39-min testbed spec, 6.2): staff_release_mode 1 voids only a person made of one track
    c, l = books()
    idb = new_book()
    c.observe(frame("barber-c", 0, {1: C_CHAIR}), POLICY)
    p = idb.create(c.tracks[1], ts(0))
    assert p.origin == ("barber-c", 1) and p.only_ever(c.tracks[1])
    idb.release(c.tracks[1], ts(1))
    idb.bind(c.tracks[1], p, ts(1))                     # the same track back: still only that one
    assert p.only_ever(c.tracks[1])
    l.observe(frame("barber-l", 1, {7: L_EDGE}), POLICY)
    idb.bind(l.tracks[7], p, ts(1))
    assert not p.only_ever(c.tracks[1]) and not p.only_ever(l.tracks[7])


# 2026-09-28 (39-min testbed spec, 6.6): seat_claim_mode 2 -- the customer's other track in the same
# camera no longer blocks the claim when it isn't in the seat and the seated body matches him better

def swap_setup(seated_sim, old_sim, old_pt=L_EDGE):
    t, p = claim_setup()
    other = TrackletBook("barber-l", ZONES["barber-l"], GEO)
    other.observe(frame("barber-l", 2, {7: old_pt}), POLICY)
    o = other.tracks[7]
    p.tracks["barber-l"], o.person = o, p.id
    p.reid = unit((1, 0))
    p.reid_by_cam["barber-l"] = (p.reid, 5)
    t.reid = unit((seated_sim, (1 - seated_sim ** 2) ** 0.5))
    o.reid = unit((old_sim, -(1 - old_sim ** 2) ** 0.5))
    return t, p, o


MODE2 = {**POLICY, "seat_claim_mode": 2, "reid_gallery": 1}


def test_mode_2_the_seated_body_that_matches_better_displaces_the_track_next_to_the_seat():
    t, p, o = swap_setup(0.8, 0.5)
    assert not may_claim_seat(t, p, "ch1", MODE2)
    assert seat_claim_displaces(t, p, "ch1", MODE2) is o
    assert seat_claim_displaces(t, p, "ch1", {**MODE2, "seat_claim_mode": 1}) is None


def test_mode_2_no_swap_when_the_old_track_matches_better_or_sits_in_the_seat():
    t, p, o = swap_setup(0.5, 0.8)
    assert seat_claim_displaces(t, p, "ch1", MODE2) is None
    t, p, o = swap_setup(0.8, 0.5, old_pt=L_SEAT)                 # two bodies in the seat: a duplicate box
    assert seat_claim_displaces(t, p, "ch1", MODE2) is None


def test_mode_2_no_swap_without_both_embeddings_or_under_the_veto():
    t, p, o = swap_setup(0.8, 0.5)
    o.reid = None
    assert seat_claim_displaces(t, p, "ch1", MODE2) is None
    t, p, o = swap_setup(0.2, 0.1)                                # under reid_veto (0.3)
    assert seat_claim_displaces(t, p, "ch1", MODE2) is None
