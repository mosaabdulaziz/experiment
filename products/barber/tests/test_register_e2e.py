"""Engine -> register, end to end with synthetic faces: two visits of one
person give one customer (2 visits); a stranger gives a second; a visit
without a haircut registers nobody and its crop files are deleted."""
import itertools

from domain import EngineUpdate, FaceCandidate
from policy import effective
from register import apply
from shop_helpers import C_AISLE, C_CHAIR, C_DOOR, C_SOFA, C_STREET, L_EDGE, L_SEAT, staff_app, ts
from storage import apply_update, connect, count_customers, list_customers
from test_chair_engine import FACE_ZONES, P2, SITTER, T0, TrackObs, face, frames
from test_engine import Shop
from chair_engine import ChairEngine
from zones import FrameGeometry

REG_POLICY = effective({"match_threshold": 0.5, "match_margin": 0.05, "visit_face_min_sim": 0.3,
                        "min_visit_faces": 2})


def faces_for(cam, tid, s, emb, root, tag):
    out = []
    for k in range(3):
        rel = f"faces/{tag}-{s}-{k}.jpg"
        (root / rel).write_bytes(b"x")
        out.append(FaceCandidate(cam, tid, ts(s) + 1.0 * (k + 1), rel, emb, 0.5 + 0.1 * k))   # 1 s apart: a link needs 2 moments
    return out


def served_visit(shop, t0, tid, emb, root, tag):
    side = tid + 100
    shop.run(t0, t0 + 3, c={tid: C_STREET})
    shop.step(t0 + 3, c={tid: C_DOOR}, faces=faces_for("barber-c", tid, t0 + 3, emb, root, tag))
    shop.run(t0 + 4, t0 + 10, c={tid: C_DOOR})
    shop.step(t0 + 10, c={tid: C_CHAIR}, l={side: L_EDGE})
    shop.run(t0 + 11, t0 + 120, c={tid: C_CHAIR}, l={side: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", side, s + 0.5, 0.1)] if s < t0 + 14 else [])
    shop.run(t0 + 120, t0 + 150, c={tid: C_AISLE})
    shop.run(t0 + 150, t0 + 153, c={tid: C_DOOR})
    shop.run(t0 + 153, t0 + 156, c={tid: C_STREET})
    shop.run(t0 + 156, t0 + 165)


def test_register_end_to_end(tmp_path):
    root = tmp_path / "crops"
    (root / "faces").mkdir(parents=True)
    conn = connect(str(tmp_path / "b.db"))
    shop = Shop()
    served_visit(shop, 0, 1, (1.0, 0.0), root, "a1")
    served_visit(shop, 200, 2, (0.99, 0.14), root, "a2")                 # the same person, a later visit
    served_visit(shop, 400, 3, (0.0, 1.0), root, "b")                     # a stranger
    shop.run(600, 603, c={4: C_STREET})                                   # a walk-out, no haircut
    shop.step(603, c={4: C_SOFA}, faces=faces_for("barber-c", 4, 603, (1.0, 0.0), root, "walk"))
    shop.run(604, 620, c={4: C_SOFA})
    shop.run(620, 623, c={4: C_DOOR})
    shop.run(623, 626, c={4: C_STREET})
    shop.run(626, 640)
    for vid, v in shop.visits.items():                                    # persist the way the consumer does
        apply_update(conn, EngineUpdate(visits={vid: v}), v.entered_at)
    for s in shop.sessions.values():
        apply_update(conn, EngineUpdate(sessions={s.id: s}), s.started_at)
    n = itertools.count()
    ids = apply(conn, EngineUpdate(registrations=list(shop.registrations), discard_crops=list(shop.discarded)),
                root, REG_POLICY, lambda: f"c{next(n)}")
    assert len(ids) == 3 and count_customers(conn) == 2
    assert {r["visit_count"] for r in list_customers(conn, 10, 0, "visit_count")} == {2, 1}
    assert not any(p.name.startswith("walk") for p in (root / "faces").iterdir())


def test_register_at_1_registers_when_the_haircut_completes_and_tops_up_when_he_leaves(tmp_path):
    # 2026-09-29 (user): the customers page lagged the completed haircuts (registered only on leaving)
    root = tmp_path / "crops"
    (root / "faces").mkdir(parents=True)
    shop = Shop({**Shop().policy, "register_at": 1})
    tid, side, t0 = 1, 101, 0
    shop.run(t0, t0 + 3, c={tid: C_STREET})
    shop.step(t0 + 3, c={tid: C_DOOR}, faces=faces_for("barber-c", tid, t0 + 3, (1.0, 0.0), root, "in"))
    shop.run(t0 + 4, t0 + 10, c={tid: C_DOOR})
    shop.step(t0 + 10, c={tid: C_CHAIR}, l={side: L_EDGE})
    shop.run(t0 + 11, t0 + 120, c={tid: C_CHAIR}, l={side: L_SEAT},
             apps_at=lambda s: [staff_app("barber-l", side, s + 0.5, 0.1)] if s < t0 + 14 else [])
    shop.run(t0 + 120, t0 + 148, c={tid: C_AISLE})
    (first,) = shop.registrations                                          # haircut done, still in the shop
    assert {f.crop_path for f in first.faces} == {f"faces/in-3-{k}.jpg" for k in range(3)}
    assert shop.visits[first.visit_id].open
    shop.step(t0 + 148, c={tid: C_AISLE}, faces=faces_for("barber-c", tid, t0 + 148, (0.99, 0.1), root, "out"))
    shop.run(t0 + 149, t0 + 153, c={tid: C_DOOR})
    shop.run(t0 + 153, t0 + 156, c={tid: C_STREET})
    shop.run(t0 + 156, t0 + 165)
    first, second = shop.registrations
    assert second.visit_id == first.visit_id
    assert {f.crop_path for f in second.faces} == {f"faces/out-148-{k}.jpg" for k in range(3)}   # only the new ones
    assert not {f.crop_path for f in first.faces} & set(shop.discarded)                        # never deleted


def test_register_source_2_a_seated_customers_faces_make_one_card(tmp_path):
    root = tmp_path / "crops"
    (root / "faces").mkdir(parents=True)
    conn = connect(str(tmp_path / "b.db"))
    eng = ChairEngine(FACE_ZONES, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")
    box = (140, 50, 120, 310)                                      # in R1's face area
    pol = effective({**P2, "match_threshold": 0.5, "match_margin": 0.05, "visit_face_min_sim": 0.3,
                     "min_visit_faces": 2})
    upd = eng.observe(frames(T0, 300, (TrackObs(4, box, 0.9, SITTER),)), [], pol)
    regs, discarded = list(upd.registrations), list(upd.discard_crops)
    for k in range(2):
        rel = f"faces/seat-{k}.jpg"
        (root / rel).write_bytes(b"x")
        u = eng.observe([], [], pol, faces=(face("barber-l", 4, T0 + 300 + k, rel),))
        regs += u.registrations
        discarded += u.discard_crops
    assert not regs and not discarded
    u = eng.observe(frames(T0 + 302, 800, ()), [], pol)             # he leaves; the haircut closes
    for vid, v in u.visits.items():
        apply_update(conn, EngineUpdate(visits={vid: v}), v.entered_at)
    for s in u.sessions.values():
        apply_update(conn, EngineUpdate(sessions={s.id: s}), s.started_at)
    (reg,) = u.registrations
    assert len(reg.faces) == 2
    n = itertools.count()
    apply(conn, EngineUpdate(registrations=[reg], discard_crops=list(u.discard_crops)), root, pol,
          lambda: f"c{next(n)}")
    (c,) = list_customers(conn, 10, 0, "visit_count")
    assert count_customers(conn) == 1 and c["face_count"] >= 1 and c["visit_count"] == 1
