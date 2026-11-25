from domain import TrackObs
from policy import DEFAULT_POLICY
from seat import Verdict, chair_geoms, evaluate, evaluate_detail

P = dict(DEFAULT_POLICY)
CAM = "barber-l"


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def flat(poly):
    return tuple(v for p in poly for v in p)


ZONES = {CAM: [
    {"id": "l-R1", "name": "R1", "type": "chair", "points": sq(0.10, 0.10, 0.30, 0.40), "pair": "R1"},
    {"id": "l-R1-seat", "name": "R1 seat", "type": "seat", "points": sq(0.15, 0.25, 0.25, 0.35), "seat_of": "l-R1"},
    {"id": "l-R2", "name": "R2", "type": "chair", "points": sq(0.30, 0.10, 0.50, 0.40), "pair": "R2"},
    {"id": "l-R2-seat", "name": "R2 seat", "type": "seat", "points": sq(0.35, 0.25, 0.45, 0.35), "seat_of": "l-R2"},
]}
G = chair_geoms(ZONES)[CAM]


def person(tid, poly, conf=0.9):
    return TrackObs(tid, (0, 0, 1, 1), conf, flat(poly))


def run(tracks, staff=None):
    staff = staff or {}
    return evaluate(tracks, G, lambda tid: staff.get(tid), P)


def test_a_seated_body_occupies_its_chair_only():
    out = run([person(1, sq(0.14, 0.05, 0.26, 0.36))])
    assert out["R1"].occupied and out["R1"].track_id == 1
    assert not out["R2"].occupied


def test_reclined_neighbour_head_does_not_occupy():
    # hips on R1's seat, torso and head reaching across R2's seat higher up: lower 35 % only on R1
    body = [[0.15, 0.36], [0.25, 0.36], [0.45, 0.24], [0.45, 0.20], [0.35, 0.20], [0.15, 0.28]]
    out = run([person(1, body)])
    assert out["R1"].occupied and not out["R2"].occupied


def test_staff_never_occupies_but_covers():
    out = run([person(5, sq(0.12, 0.05, 0.28, 0.38))], staff={5: 0.95})
    assert not out["R1"].occupied and out["R1"].cover


def test_staff_score_is_read_per_frame():
    body = [person(5, sq(0.14, 0.05, 0.26, 0.36))]
    assert not run(body, staff={5: 0.9})["R1"].occupied
    assert run(body, staff={5: 0.2})["R1"].occupied
    assert run(body, staff={})["R1"].occupied        # unknown = not staff


def test_low_confidence_and_maskless_tracks_are_ignored():
    assert not run([person(1, sq(0.14, 0.05, 0.26, 0.36), conf=0.1)])["R1"].occupied
    assert not run([TrackObs(1, (0, 0, 1, 1), 0.9, None)])["R1"].occupied


def test_each_person_counts_for_one_chair():
    # one wide body over both seats' lower parts: only the larger overlap wins
    out = run([person(1, sq(0.15, 0.20, 0.40, 0.36))])
    assert out["R1"].occupied != out["R2"].occupied


def test_chair_without_usable_seat_is_skipped():
    logged = []
    zones = {CAM: ZONES[CAM][:2] + [
        {"id": "l-R3", "name": "R3", "type": "chair", "points": sq(0.5, 0.1, 0.7, 0.4), "pair": "R3"},
        {"id": "l-R4", "name": "R4", "type": "chair", "points": sq(0.7, 0.1, 0.9, 0.4), "pair": "R4"},
        {"id": "l-R4-seat", "name": "R4 seat", "type": "seat", "points": [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
         "seat_of": "l-R4"}]}
    geoms = chair_geoms(zones, log=logged.append)
    assert [g.chair for g in geoms[CAM]] == ["R1"]
    assert len(logged) == 2


def why(tracks, staff=None, policy=None):
    staff = staff or {}
    _, v = evaluate_detail(tracks, G, lambda tid: staff.get(tid), policy or P)
    return v


def test_verdict_occupied_with_chair_and_share():
    v = why([person(1, sq(0.14, 0.05, 0.26, 0.36))])[1]
    assert v.why == "occupied" and v.chair == "R1" and 0.5 < v.share <= 1.0


def test_verdict_staff_cover_and_plain_staff():
    assert why([person(5, sq(0.12, 0.05, 0.28, 0.38))], staff={5: 0.95})[5] == Verdict("R1", None, "staff_cover")
    assert why([person(5, sq(0.6, 0.5, 0.7, 0.9))], staff={5: 0.95})[5] == Verdict(None, None, "staff")


def test_verdict_low_conf_and_no_mask():
    assert why([person(1, sq(0.14, 0.05, 0.26, 0.36), conf=0.1)])[1].why == "low_conf"
    assert why([TrackObs(1, (0, 0, 1, 1), 0.9, None)])[1].why == "no_mask"


def test_verdict_below_min_and_no_chair():
    v = why([person(1, sq(0.14, 0.05, 0.26, 0.36))], policy={**P, "ce_seat_min": 1.01})[1]
    assert v.why == "below_min" and v.chair == "R1" and v.share > 0
    assert why([person(1, sq(0.6, 0.5, 0.7, 0.9))])[1] == Verdict(None, None, "no_chair")


def test_verdict_other_chair_for_the_loser_in_either_order():
    big, small = person(1, sq(0.14, 0.05, 0.26, 0.36)), person(2, sq(0.15, 0.20, 0.25, 0.36))
    for order in ([big, small], [small, big]):
        v = why(order)
        assert v[1].why == "occupied" and v[2].why == "other_chair" and v[2].chair == "R1"
        assert evaluate(order, G, lambda t: None, P)["R1"].track_id == 1


# 2026-10-05 (haircut accuracy, inside-outline gate): a body mostly outside the chair outline (staff standing by
# the seat's aisle corner) never occupies it, even though its lower mask touches the seat.
# Upper body left of R1's outline, one leg strip on the seat: whole mask ~23 % inside R1, lower share 0.30.
STANDING_BY = [[0.0, 0.0], [0.12, 0.0], [0.12, 0.25], [0.18, 0.25], [0.18, 0.35], [0.0, 0.35]]
GATE = {**P, "ce_seat_inside_min": 0.30}


def test_inside_gate_rejects_a_body_mostly_outside_the_outline():
    v = why([person(1, STANDING_BY)], policy=GATE)[1]
    assert v.why == "outside_chair" and v.chair == "R1" and v.share >= P["ce_seat_min"]
    assert not evaluate([person(1, STANDING_BY)], G, lambda t: None, GATE)["R1"].occupied


def test_inside_gate_accepts_a_seated_body():
    out = evaluate([person(1, sq(0.14, 0.05, 0.26, 0.36))], G, lambda t: None, GATE)
    assert out["R1"].occupied and out["R1"].track_id == 1


def test_inside_gate_off_is_the_old_rule():
    pol = {**P, "ce_seat_inside_min": 0.0}
    assert evaluate([person(1, STANDING_BY)], G, lambda t: None, pol)["R1"].occupied
    assert why([person(1, STANDING_BY)], policy=pol)[1].why == "occupied"


def test_a_gated_body_does_not_push_the_sitter_out():
    sitter, stander = person(1, sq(0.14, 0.05, 0.26, 0.36)), person(2, STANDING_BY)
    out = evaluate([stander, sitter], G, lambda t: None, GATE)
    assert out["R1"].occupied and out["R1"].track_id == 1


# 2026-10-05 (far-chair per-chair rules): "<key>@<chair>" overrides apply to that chair only; the one-chair-per-person
# assignment (highest share wins) is unchanged -- a chair's own threshold only decides whether the winner counts.
R1_BODY, R2_BODY = sq(0.14, 0.05, 0.26, 0.36), sq(0.34, 0.05, 0.46, 0.36)


def test_seat_min_override_applies_only_to_its_chair():
    strict = {**P, "ce_seat_min": 1.01}
    assert why([person(1, R1_BODY)], policy={**strict, "ce_seat_min@R1": 0.10})[1].why == "occupied"
    assert why([person(1, R1_BODY)], policy={**strict, "ce_seat_min@R2": 0.10})[1].why == "below_min"
    assert why([person(1, R2_BODY)], policy={**strict, "ce_seat_min@R1": 0.10})[1].why == "below_min"


def test_override_never_moves_a_person_to_another_chair():
    wide = person(1, sq(0.15, 0.20, 0.40, 0.36))      # larger share on R1, some on R2
    pol = {**P, "ce_seat_min@R1": 1.01, "ce_seat_min@R2": 0.01}
    out, v = evaluate_detail([wide], G, lambda t: None, pol)
    assert not out["R1"].occupied and not out["R2"].occupied
    assert v[1].why == "below_min" and v[1].chair == "R1"


def test_conf_min_override_applies_only_to_its_chair():
    pol = {**P, "ce_conf_min@R1": 0.10}
    assert why([person(1, R1_BODY, conf=0.15)], policy=pol)[1].why == "occupied"
    assert why([person(1, R2_BODY, conf=0.15)], policy=pol)[1].why == "low_conf"
    assert why([person(5, R1_BODY, conf=0.15)], staff={5: 0.95}, policy=pol)[5].why == "staff_cover"
    out = evaluate([person(5, R2_BODY, conf=0.15)], G, lambda t: 0.95, pol)
    assert not out["R2"].cover                                   # below R2's own floor: no cover there
    assert evaluate([person(5, R2_BODY, conf=0.30)], G, lambda t: 0.95, pol)["R2"].cover


def test_inside_min_override_applies_only_to_its_chair():
    assert why([person(1, STANDING_BY)], policy={**GATE, "ce_seat_inside_min@R1": 0.0})[1].why == "occupied"
    assert why([person(1, STANDING_BY)], policy={**GATE, "ce_seat_inside_min@R2": 0.0})[1].why == "outside_chair"


def test_carry_lower_frac_counts_only_the_bottom_of_a_staff_mask_on_the_seat():
    """2026-10-15 (user, Palm Hills hand-overs): a barber standing at the empty chair has his torso over the seat in
    the image, his legs below it; one hiding a seated customer leans in over the seat with his hips. With
    ce_carry_lower_frac < 1 only the bottom of his mask is measured on the seat (and the outline test is off)."""
    p = {**P, "ce_carry_lower_frac": 0.30}
    standing = person(5, sq(0.13, 0.05, 0.27, 0.60))        # torso over R1's seat, legs and feet below it
    leaning = person(6, sq(0.13, 0.10, 0.27, 0.34))         # hips down on the seat
    staff = {5: 0.95, 6: 0.95}
    assert run([standing], staff)["R1"].cover               # today's rule: whole mask / outline
    out = evaluate([standing], G, staff.get, p)
    assert not out["R1"].cover
    assert evaluate([leaning], G, staff.get, p)["R1"].cover


def test_carry_lower_frac_1_is_todays_rule():
    staff = {5: 0.95}
    body = [person(5, sq(0.12, 0.05, 0.28, 0.38))]
    assert evaluate(body, G, staff.get, {**P, "ce_carry_lower_frac": 1.0})["R1"].cover == run(body, staff)["R1"].cover


# 2026-10-15 (Palm Hills, L7 moved to its better view): a chair drawn in a second camera as a non-owner pair member,
# with a seat there, claims people in that camera so its sitter can't be given to a neighbouring chair, and signals
# nothing (the owner camera decides the chair).
ZP = {CAM: [
    {"id": "l-R1", "name": "R1", "type": "chair", "points": sq(0.10, 0.10, 0.30, 0.40), "pair": "R1"},
    {"id": "l-R1-seat", "name": "R1 seat", "type": "seat", "points": sq(0.15, 0.25, 0.25, 0.35), "seat_of": "l-R1"},
    {"id": "l-R2", "name": "R2", "type": "chair", "points": sq(0.20, 0.10, 0.50, 0.40), "pair": "R2", "owner": True},
    {"id": "l-R2-seat", "name": "R2 seat", "type": "seat", "points": sq(0.22, 0.25, 0.45, 0.35), "seat_of": "l-R2"},
], "other": [
    {"id": "o-R1", "name": "R1", "type": "chair", "points": sq(0.1, 0.1, 0.3, 0.4), "pair": "R1", "owner": True},
    {"id": "o-R1-seat", "name": "R1 seat", "type": "seat", "points": sq(0.15, 0.25, 0.25, 0.35), "seat_of": "o-R1"},
]}


def test_a_non_owner_seat_claims_its_sitter_and_signals_nothing():
    geoms = chair_geoms(ZP)
    assert {(g.chair, g.owner) for g in geoms[CAM]} == {("R1", False), ("R2", True)}
    sitter = [person(1, sq(0.14, 0.05, 0.26, 0.36))]           # on R1's seat, his lower part also on R2's
    out, why = evaluate_detail(sitter, geoms[CAM], lambda tid: None, P)
    assert set(out) == {"R2"} and not out["R2"].occupied
    assert why[1].chair == "R1" and why[1].why == "other_view"
    barber = [person(5, sq(0.12, 0.05, 0.21, 0.38))]               # over R1 only
    out = evaluate(barber, geoms[CAM], {5: 0.95}.get, P)
    assert set(out) == {"R2"} and not out["R2"].cover


def test_a_seatless_non_owner_chair_is_still_skipped():
    zones = {CAM: ZP[CAM][2:] + [ZP[CAM][0]], "other": ZP["other"]}
    assert [g.chair for g in chair_geoms(zones, log=lambda m: None)[CAM]] == ["R2"]


def test_carry_lower_frac_override_applies_only_to_its_chair():
    """2026-10-18: a camera that sees a chair's barber from behind may never see his hips on the seat; that chair can
    keep the whole-mask cover rule (1.0) while the site runs 0.30."""
    p = {**P, "ce_carry_lower_frac": 0.30, "ce_carry_lower_frac@R1": 1.0}
    staff = {5: 0.95, 6: 0.95}
    on_r1, on_r2 = person(5, sq(0.13, 0.05, 0.27, 0.60)), person(6, sq(0.33, 0.05, 0.47, 0.60))   # standing, torso over the seat
    assert evaluate([on_r1], G, staff.get, p)["R1"].cover
    assert not evaluate([on_r2], G, staff.get, p)["R2"].cover


def test_a_chair_geom_carries_its_kind():
    """2026-11-10 (pedicure chairs): the chair zone's kind reaches its geometry; no kind = a barber chair."""
    sq = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    seat = [[0.15, 0.15], [0.25, 0.15], [0.25, 0.25], [0.15, 0.25]]
    zones = {"cam": [{"id": "p1", "name": "P1", "type": "chair", "points": sq, "kind": "pedicure"},
                     {"id": "p1s", "name": "s", "type": "seat", "points": seat, "seat_of": "p1"},
                     {"id": "r1", "name": "R1", "type": "chair", "points": sq},
                     {"id": "r1s", "name": "s", "type": "seat", "points": seat, "seat_of": "r1"}]}
    kinds = {g.chair: g.kind for g in chair_geoms(zones)["cam"]}
    assert kinds == {"p1": "pedicure", "r1": "haircut"}
