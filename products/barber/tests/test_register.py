import numpy as np

from policy import effective
from register import clean_faces, decide, score_customers

POLICY = effective({"match_threshold": 0.5, "match_margin": 0.05, "visit_face_min_sim": 0.3, "min_visit_faces": 2})


def unit(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


def faces(*vs):
    return np.stack([unit(*v) for v in vs])


def test_clean_faces_drops_the_odd_one_out():
    embs = faces((1, 0, 0), (0.95, 0.1, 0), (0.9, 0.05, 0.1), (0, 0, 1))
    assert sorted(clean_faces(embs, 0.3).tolist()) == [0, 1, 2]


def test_score_is_the_median_so_one_lucky_frame_does_not_decide():
    gallery = {"c1": faces((1, 0, 0))}
    lucky = faces((1, 0, 0), (0, 1, 0), (0, 1, 0.1))
    assert score_customers(lucky, gallery)["c1"] < 0.2


def test_decide_links_a_confident_match():
    gallery = {"c1": faces((1, 0, 0), (0.9, 0.1, 0)), "c2": faces((0, 1, 0))}
    d = decide(faces((0.98, 0.05, 0), (0.95, 0.1, 0.05)), gallery, POLICY)
    assert d.customer_id == "c1" and d.usable and d.score > 0.9 and d.runner_up < 0.2


def test_decide_creates_a_new_customer_when_two_are_too_close():
    gallery = {"c1": faces((1, 0.05, 0)), "c2": faces((1, -0.05, 0))}
    d = decide(faces((1, 0, 0), (1, 0.01, 0)), gallery, POLICY)
    assert d.customer_id is None and d.usable


def test_decide_creates_a_new_customer_below_the_threshold():
    d = decide(faces((0, 0, 1), (0, 0.1, 1)), {"c1": faces((1, 0, 0))}, POLICY)
    assert d.customer_id is None and d.usable


def test_decide_with_an_empty_gallery_is_a_new_customer():
    d = decide(faces((1, 0, 0), (1, 0.1, 0)), {}, POLICY)
    assert d.customer_id is None and d.usable and d.score is None


def test_too_few_usable_faces_is_not_usable():
    d = decide(faces((1, 0, 0)), {"c1": faces((1, 0, 0))}, POLICY)
    assert d.customer_id is None and not d.usable
    d0 = decide(np.zeros((0, 3), np.float32), {"c1": faces((1, 0, 0))}, POLICY)
    assert not d0.usable and d0.clean == ()


def test_one_face_makes_a_new_card_but_a_link_needs_two_moments():
    # 2026-10-12 (user): one valid face is enough for a card; linking a returning customer keeps two (review finding 4)
    one = {**POLICY, "min_visit_faces": 1, "min_link_faces": 2}
    gallery = {"c1": faces((1, 0, 0))}
    d = decide(faces((1, 0, 0)), gallery, one, times=[10.0])
    assert d.usable and d.customer_id is None and d.clean == (0,)
    d = decide(faces((1, 0, 0), (0.98, 0.05, 0)), gallery, one, times=[10.0, 10.25])   # one moment
    assert d.usable and d.customer_id is None
    d = decide(faces((1, 0, 0), (0.98, 0.05, 0)), gallery, one, times=[10.0, 11.5])    # two moments
    assert d.customer_id == "c1"
    assert not decide(faces((1, 0, 0)), gallery, {**one, "min_visit_faces": 2}).usable


# 2026-09-29 (user): crop_check -- an ID swap must not put another person's crops into a customer

def test_clean_bodies_drops_the_odd_one_per_camera_and_keeps_a_lone_one():
    from domain import BodyCandidate
    from register import clean_bodies
    b = lambda cam, *v: BodyCandidate(cam, 1, 0.0, "x", tuple(unit(*v).tolist()), 0.5)
    bodies = [b("barber-l", 1, 0, 0), b("barber-l", 0.95, 0.1, 0), b("barber-l", 0, 0, 1),     # l: the third is someone else
              b("barber-r", 0, 1, 0),                                                         # r: alone, kept
              BodyCandidate("barber-l", 1, 0.0, "x", None, 0.5)]                              # no embedding: can't vouch for it
    assert clean_bodies(bodies, 0.6) == (0, 1, 3)
