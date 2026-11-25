import itertools
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from domain import BodyCandidate, EngineUpdate, FaceCandidate, Registration, Visit
from policy import effective
from register import apply, decide, returnable
from storage import (apply_update, connect, count_customers, delete_customer, get_customer, list_customers,
                     load_face_gallery, purge_inactive, referenced_crops)

T = datetime(2026, 9, 25, 10, 0, 0)
POLICY = effective({"match_threshold": 0.5, "match_margin": 0.05, "visit_face_min_sim": 0.3, "min_visit_faces": 2,
                    "customer_max_faces": 2, "customer_max_bodies": 1})


def unit(*v):
    a = np.asarray(v, np.float32)
    return tuple(float(x) for x in a / np.linalg.norm(a))


def files_on_disk(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    root = tmp_path / "crops"
    (root / "faces").mkdir(parents=True)
    (root / "bodies").mkdir(parents=True)
    n = itertools.count()
    return conn, root, (lambda: f"c{next(n)}")


def visit(conn, vid, entered):
    v = Visit(vid, "p-" + vid, entered, True, state="finished", confirmed=True, waiting_started_at=entered,
              haircuts=1, exited_at=entered + timedelta(minutes=30), outcome="completed", open=False)
    apply_update(conn, EngineUpdate(visits={vid: v}), entered)
    return v


def reg(root, v, embs, quality=(0.9, 0.8, 0.7), bodies=1):
    faces = []
    for i, e in enumerate(embs):
        rel = f"faces/{v.id}-{i}.jpg"
        (root / rel).write_bytes(b"x")
        faces.append(FaceCandidate("barber-c", 1, float(i), rel, e, quality[i % len(quality)]))   # 1 s apart
    bs = []
    for i in range(bodies):
        rel = f"bodies/{v.id}-{i}.jpg"
        (root / rel).write_bytes(b"x")
        bs.append(BodyCandidate("barber-l", 7, 0.0, rel, None, 0.5 + 0.1 * i))
    return Registration(v.id, v.exited_at, tuple(faces), tuple(bs))


def test_first_visit_creates_a_customer_and_keeps_the_best_crops(env):
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    u = EngineUpdate(registrations=[reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)],
                                        bodies=2)])
    (cid,) = apply(conn, u, root, POLICY, new_id)
    c = get_customer(conn, cid)
    assert (c["visit_count"], c["face_count"], c["first_seen"], c["last_seen"]) == (1, 2, v.entered_at.isoformat(timespec="seconds"), v.exited_at.isoformat(timespec="seconds"))
    assert [f["quality"] for f in c["faces"]] == [0.9, 0.8]           # the best 2 of 3 kept
    assert not (root / "faces/v1-2.jpg").exists()                     # the third deleted
    assert len(c["bodies"]) == 1 and not (root / "bodies/v1-0.jpg").exists()
    assert c["visits"][0]["id"] == "v1"
    assert files_on_disk(root) == referenced_crops(conn)              # nothing kept is missing, nothing else lingers


def test_a_returning_customer_is_linked(env):
    conn, root, new_id = env
    v1 = visit(conn, "v1", T)
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, v1, [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                   root, POLICY, new_id)
    v2 = visit(conn, "v2", T + timedelta(days=20))
    (cid2,) = apply(conn, EngineUpdate(registrations=[reg(root, v2, [unit(0.97, 0.05, 0), unit(1, 0.02, 0)],
                                                          quality=(0.95, 0.85))]), root, POLICY, new_id)
    assert cid2 == cid and count_customers(conn) == 1
    c = get_customer(conn, cid)
    assert c["visit_count"] == 2 and c["last_visit_id"] == "v2" and [f["quality"] for f in c["faces"]] == [0.95, 0.9]
    assert not (root / "faces/v1-1.jpg").exists()                     # the old 0.8-quality face was evicted
    assert files_on_disk(root) == referenced_crops(conn)              # nothing kept is missing, nothing else lingers


def test_a_stranger_gets_a_second_customer(env):
    conn, root, new_id = env
    apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
          root, POLICY, new_id)
    apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v2", T), [unit(0, 1, 0), unit(0, 0.95, 0.1)])]),
          root, POLICY, new_id)
    assert count_customers(conn) == 2 and len(load_face_gallery(conn)) == 2


def test_no_usable_face_is_a_face_count_zero_customer(env):
    conn, root, new_id = env
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0)])]),
                   root, POLICY, new_id)
    c = get_customer(conn, cid)
    assert c["face_count"] == 0 and c["faces"] == [] and not (root / "faces/v1-0.jpg").exists()
    assert cid not in load_face_gallery(conn)


def test_discarded_crops_are_deleted(env):
    conn, root, new_id = env
    (root / "faces/x.jpg").write_bytes(b"x")
    assert apply(conn, EngineUpdate(discard_crops=["faces/x.jpg"]), root, POLICY, new_id) == []
    assert not (root / "faces/x.jpg").exists()


def test_retention_deletes_inactive_customers_and_keeps_their_visits(env):
    conn, root, new_id = env
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                   root, POLICY, new_id)
    assert purge_inactive(conn, T + timedelta(days=300), 365) == []
    paths = purge_inactive(conn, T + timedelta(days=400), 365)
    assert len(paths) == 3 and get_customer(conn, cid) is None
    assert conn.execute("SELECT customer_id FROM visit WHERE id = 'v1'").fetchone()[0] is None
    assert referenced_crops(conn) == set()


def test_delete_customer_on_request(env):
    conn, root, new_id = env
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                   root, POLICY, new_id)
    assert len(delete_customer(conn, cid)) == 3 and delete_customer(conn, cid) is None
    assert list_customers(conn, 10, 0, "last_seen") == []


def test_a_failed_registration_does_not_lose_the_others(env):
    conn, root, new_id = env
    # A pre-existing customer with id "dup": the first registration below
    # collides with it (a stand-in for a new_id collision, or a foreign-key
    # failure because the matched customer was purged).
    conn.execute("INSERT INTO customer (id, first_seen, last_seen, visit_count, last_visit_id, face_count)"
                 " VALUES ('dup', ?, ?, 1, 'other', 0)", (T.isoformat(timespec="seconds"), T.isoformat(timespec="seconds")))
    conn.commit()
    ids_out = iter(["dup", "good"])
    v1 = visit(conn, "v1", T)
    v2 = visit(conn, "v2", T)
    u = EngineUpdate(registrations=[
        reg(root, v1, [unit(1, 0, 0)]),                              # too few clean faces -> a new customer, id "dup" collides
        reg(root, v2, [unit(0, 1, 0), unit(0, 0.95, 0.1)]),          # a stranger -> a new customer, id "good"
    ])
    ids = apply(conn, u, root, POLICY, lambda: next(ids_out))
    assert ids == ["good"]
    assert count_customers(conn) == 2                                 # "dup" (pre-existing) and "good"
    assert get_customer(conn, "good") is not None


def test_a_visit_is_never_linked_to_a_customer_deleted_after_the_match(env, monkeypatch):
    conn, root, new_id = env
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                   root, POLICY, new_id)
    import register
    real_decide = register.decide

    def decide_then_erase(embs, gallery, policy, times=None):
        d = real_decide(embs, gallery, policy, times)
        delete_customer(conn, cid)                    # the user erases him between decide() and the write
        return d

    monkeypatch.setattr(register, "decide", decide_then_erase)
    v2 = visit(conn, "v2", T + timedelta(days=1))
    assert apply(conn, EngineUpdate(registrations=[reg(root, v2, [unit(1, 0.01, 0), unit(0.96, 0.1, 0)])]),
                 root, POLICY, new_id) == []
    assert conn.execute("SELECT customer_id FROM visit WHERE id = 'v2'").fetchone()[0] is None
    assert conn.execute("SELECT COUNT(*) FROM customer_face WHERE customer_id = ?", (cid,)).fetchone()[0] == 0


def test_delete_customer_reads_its_crops_in_the_deleting_transaction(env):
    conn, root, new_id = env
    (cid,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                   root, POLICY, new_id)
    statements = []
    conn.set_trace_callback(statements.append)
    paths = delete_customer(conn, cid)
    conn.set_trace_callback(None)
    assert len(paths) == 3
    begin = next(i for i, s in enumerate(statements) if s.upper().startswith("BEGIN"))
    reads = [i for i, s in enumerate(statements) if "SELECT crop_path" in s]
    assert reads and all(i > begin for i in reads)


def test_register_visit_refuses_a_customer_that_no_longer_exists(env):
    from register import Decision
    from storage import register_visit
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    with pytest.raises(LookupError):
        register_visit(conn, reg(root, v, [], bodies=0), Decision("gone", 0.9, None, (), False), POLICY, new_id)
    assert conn.execute("SELECT customer_id FROM visit WHERE id = 'v1'").fetchone()[0] is None


# 2026-09-29 (user): register_at 1 registers a visit when its haircut completes and tops it up when he
# leaves; crop_check keeps a customer's crops consistent (an ID swap slipped others' crops in)

CHECK = {**POLICY, "crop_check": 1, "customer_face_min_sim": 0.5, "body_min_sim": 0.6, "customer_max_faces": 5,
         "customer_max_bodies": 5}


def body_reg(root, v, faces_embs, bodies, tag):
    fs = []
    for i, e in enumerate(faces_embs):
        rel = f"faces/{tag}-{i}.jpg"
        (root / rel).write_bytes(b"x")
        fs.append(FaceCandidate("barber-c", 1, float(i), rel, e, 0.9 - 0.01 * i))
    bs = []
    for i, (cam, e) in enumerate(bodies):
        rel = f"bodies/{tag}-{i}.jpg"
        (root / rel).write_bytes(b"x")
        bs.append(BodyCandidate(cam, 7, 0.0, rel, e, 0.9 - 0.01 * i))
    return Registration(v.id, v.exited_at, tuple(fs), tuple(bs))


def test_a_second_registration_of_the_same_visit_tops_up_the_same_customer(env):
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    (cid,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0)],
                                                                [("barber-l", unit(1, 0, 0))], "a")]), root, CHECK, new_id)
    (cid2,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(0.97, 0.05, 0)],
                                                                [("barber-l", unit(0.9, 0.1, 0))], "b")]), root, CHECK, new_id)
    c = get_customer(conn, cid)
    assert cid2 == cid and c["visit_count"] == 1 and count_customers(conn) == 1
    assert len(c["faces"]) == 3 and len(c["bodies"]) == 2


def test_crop_check_keeps_another_persons_crops_out_of_a_customer(env):
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    (cid,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0)],
                                                                [("barber-l", unit(1, 0, 0))], "a")]), root, CHECK, new_id)
    swapped = body_reg(root, v, [unit(0, 1, 0), unit(0.96, 0.05, 0)],             # a stranger's face slipped in
                       [("barber-l", unit(0, 0, 1)), ("barber-r", unit(0, 1, 0))], "b")   # his l body; a lone r body
    apply(conn, EngineUpdate(registrations=[swapped]), root, CHECK, new_id)
    paths = lambda t: sorted(r[0] for r in conn.execute(f"SELECT crop_path FROM {t} WHERE customer_id = ?", (cid,)))
    assert paths("customer_face") == ["faces/a-0.jpg", "faces/a-1.jpg", "faces/b-1.jpg"]
    assert paths("customer_body") == ["bodies/a-0.jpg", "bodies/b-1.jpg"]
    assert not (root / "faces/b-0.jpg").exists() and not (root / "bodies/b-0.jpg").exists()
    assert files_on_disk(root) == referenced_crops(conn)


def test_crop_check_0_keeps_todays_behaviour(env):
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    loose = {**CHECK, "crop_check": 0}
    (cid,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0)],
                                                                [("barber-l", unit(1, 0, 0)), ("barber-l", unit(0, 0, 1))],
                                                                "a")]), root, loose, new_id)
    assert len(get_customer(conn, cid)["bodies"]) == 2


def test_every_registration_prunes_a_customers_faces_unlike_his_medoid(env):
    # 2026-09-29 (user): barber faces and other customers' faces ended up in one record
    conn, root, new_id = env
    strict = {**CHECK, "customer_face_min_sim": 0.6, "visit_face_min_sim": 0.0}
    v = visit(conn, "v1", T)
    (cid,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.2, 0),
                                                                         unit(0.2, 1, 0)], [], "a")]), root, strict, new_id)
    paths = sorted(r[0] for r in conn.execute("SELECT crop_path FROM customer_face WHERE customer_id = ?", (cid,)))
    assert paths == ["faces/a-0.jpg", "faces/a-1.jpg", "faces/a-2.jpg"]           # the odd face is gone
    assert get_customer(conn, cid)["face_count"] == 3
    assert files_on_disk(root) == referenced_crops(conn)


def test_a_customer_without_faces_shows_his_earliest_body_crop(env):
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    late = BodyCandidate("barber-l", 7, 200.0, "bodies/late.jpg", unit(1, 0, 0), 0.99)     # best quality, later
    early = BodyCandidate("barber-l", 7, 100.0, "bodies/early.jpg", unit(0.95, 0.1, 0), 0.5)
    for b in (late, early):
        (root / b.crop_path).write_bytes(b"x")
    (cid,) = apply(conn, EngineUpdate(registrations=[Registration(v.id, v.exited_at, (), (late, early))]),
                   root, CHECK, new_id)
    (row,) = list_customers(conn, 10, 0)
    first = conn.execute("SELECT id FROM customer_body WHERE crop_path = 'bodies/early.jpg'").fetchone()[0]
    assert row["thumb"] == f"/api/barber/crops/bodies/{first}" and row["thumb_kind"] == "body"


def test_a_visit_is_never_linked_to_a_customer_seen_minutes_ago(env):
    # 2026-09-29: 5 visits in 25 minutes were linked to one customer (top-down faces look alike)
    conn, root, new_id = env
    gap = {**POLICY, "min_return_gap_h": 6}
    same = [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)]
    v1 = visit(conn, "v1", T)
    (c1,) = apply(conn, EngineUpdate(registrations=[reg(root, v1, same)]), root, gap, new_id)
    v2 = visit(conn, "v2", T + timedelta(minutes=40))                    # 10 min after v1 left
    (c2,) = apply(conn, EngineUpdate(registrations=[reg(root, v2, same)]), root, gap, new_id)
    assert c2 != c1


def test_the_return_gap_still_links_a_customer_back_days_later(env):
    conn, root, new_id = env
    gap = {**POLICY, "min_return_gap_h": 6}
    same = [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)]
    v1 = visit(conn, "v1", T)
    (c1,) = apply(conn, EngineUpdate(registrations=[reg(root, v1, same)]), root, gap, new_id)
    v3 = visit(conn, "v3", T + timedelta(days=3))
    (c3,) = apply(conn, EngineUpdate(registrations=[reg(root, v3, same)]), root, gap, new_id)
    assert c3 == c1 and get_customer(conn, c1)["visit_count"] == 2


def test_the_card_lists_the_chair_of_each_visit_newest_first(env):
    # 2026-10-18 (user): the customer card says at which chair he was served in every visit
    conn, root, new_id = env
    gap = {**POLICY, "min_return_gap_h": 6}
    same = [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)]
    v1, v3 = visit(conn, "v1", T), visit(conn, "v3", T + timedelta(days=3))
    for vid, chair, at in (("v1", "L6", T), ("v3", "ph5-R1", T + timedelta(days=3, minutes=5)),
                           ("v3", "L7", T + timedelta(days=3, minutes=20))):    # resumed at a second chair
        conn.execute("INSERT INTO service_session (id, visit_id, chair_id, started_at, open) VALUES (?, ?, ?, ?, 0)",
                     (f"{vid}-{chair}", vid, chair, at.isoformat()))
    apply(conn, EngineUpdate(registrations=[reg(root, v1, same)]), root, gap, new_id)
    apply(conn, EngineUpdate(registrations=[reg(root, v3, same)]), root, gap, new_id)
    (row,) = list_customers(conn, 10, 0)
    assert row["served"] == [{"at": (T + timedelta(days=3)).isoformat(), "chairs": ["ph5-R1", "L7"]},
                             {"at": T.isoformat(), "chairs": ["L6"]}]


def test_link_returns_0_every_visit_is_its_own_customer(env):
    # 2026-09-29 (user): the customers page is the day's customers -- no entry has two visits for now
    conn, root, new_id = env
    off = {**POLICY, "link_returns": 0}
    same = [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)]
    v1, v2 = visit(conn, "v1", T), visit(conn, "v2", T + timedelta(days=3))
    (c1,) = apply(conn, EngineUpdate(registrations=[reg(root, v1, same)]), root, off, new_id)
    (c2,) = apply(conn, EngineUpdate(registrations=[reg(root, v2, same)]), root, off, new_id)
    assert c1 != c2 and {r["visit_count"] for r in list_customers(conn, 10, 0)} == {1}


def test_the_earliest_body_crop_survives_the_quality_cap(env):
    # POLICY keeps 1 body: quality alone would keep the later one
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    late = BodyCandidate("barber-l", 7, 200.0, "bodies/late.jpg", unit(1, 0, 0), 0.99)
    early = BodyCandidate("barber-l", 7, 100.0, "bodies/early.jpg", unit(0.95, 0.1, 0), 0.5)
    for b in (late, early):
        (root / b.crop_path).write_bytes(b"x")
    apply(conn, EngineUpdate(registrations=[Registration(v.id, v.exited_at, (), (late, early))]), root, POLICY, new_id)
    assert [r[0] for r in conn.execute("SELECT crop_path FROM customer_body")] == ["bodies/early.jpg"]
    assert files_on_disk(root) == referenced_crops(conn)



def test_the_card_shows_the_earliest_body_crop_even_with_faces(env):
    # 2026-09-29 (user): the body crop is the card's picture; a face only when he has no body crop
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    (cid,) = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [unit(1, 0, 0), unit(0.95, 0.1, 0)],
                                                                [("barber-l", unit(1, 0, 0))], "a")]), root, CHECK, new_id)
    (row,) = list_customers(conn, 10, 0)
    assert row["thumb"].startswith("/api/barber/crops/bodies/") and row["thumb_kind"] == "body"


def test_a_visit_without_any_crop_is_not_a_customer(env):
    # 2026-09-29 (user): no customer entry without a picture
    conn, root, new_id = env
    v = visit(conn, "v1", T)
    ids = apply(conn, EngineUpdate(registrations=[body_reg(root, v, [], [], "a")]), root, CHECK, new_id)
    assert ids == [] and count_customers(conn) == 0
    w = visit(conn, "v2", T)
    ids = apply(conn, EngineUpdate(registrations=[body_reg(root, w, [unit(1, 0, 0)], [], "b")]), root,
                {**CHECK, "min_visit_faces": 2}, new_id)                   # one face: not usable, no body either
    assert ids == [] and count_customers(conn) == 0 and not (root / "faces/b-0.jpg").exists()


def _door_entry(conn, root, eid, at, embs):
    from domain import DoorCrossing
    apply_update(conn, EngineUpdate(door_crossings=[DoorCrossing(at, "in", "barber-c", 1, False, eid)]), at)
    faces = []
    for i, e in enumerate(embs):
        rel = f"faces/{eid}-{i}.jpg"
        (root / rel).write_bytes(b"x")
        faces.append(FaceCandidate("barber-c", 1, float(i), rel, e, 0.9))
    return Registration(eid, at, tuple(faces), ())


@pytest.mark.parametrize("gap_h, later, linked", [(6, timedelta(minutes=30), False), (6, timedelta(hours=7), True),
                                                  (0, timedelta(minutes=30), True)])
def test_the_return_gap_applies_to_door_entries(env, gap_h, later, linked):
    conn, root, new_id = env
    pol = {**POLICY, "min_return_gap_h": gap_h}
    same = [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0.9, 0.05, 0.1)]
    (c1,) = apply(conn, EngineUpdate(registrations=[_door_entry(conn, root, "e1", T, same)]), root, pol, new_id)
    # last_seen = T; the second entry crosses `later` after it
    (c2,) = apply(conn, EngineUpdate(registrations=[_door_entry(conn, root, "e2", T + later, same)]), root, pol, new_id)
    assert (c2 == c1) is linked


def test_faces_from_one_glimpse_never_link_a_customer(env):
    """2026-10-07 review: 'two faces' 0.25 s apart are one view -- a link needs two moments >= 1 s apart."""
    conn, root, new_id = env
    v1, v2 = visit(conn, "v1", T), visit(conn, "v2", T + timedelta(minutes=5))
    (c1,) = apply(conn, EngineUpdate(registrations=[reg(root, v1, [unit(1, 0, 0), unit(0.95, 0.1, 0)])]), root, POLICY, new_id)
    r2 = reg(root, v2, [unit(1, 0.01, 0), unit(0.96, 0.1, 0)])
    glimpse = tuple(replace(f, ts=0.25 * i) for i, f in enumerate(r2.faces))
    (c2,) = apply(conn, EngineUpdate(registrations=[replace(r2, faces=glimpse)]), root, POLICY, new_id)
    assert c2 != c1                                  # a new card, not a merge on one view


def test_the_return_gap_holds_when_the_entry_row_is_missing(env):
    """returnable() used to skip the gap (link anyone) when it found no visit / door row for the entry."""
    conn, root, new_id = env
    pol = {**POLICY, "min_return_gap_h": 6}
    (c1,) = apply(conn, EngineUpdate(registrations=[reg(root, visit(conn, "v1", T), [unit(1, 0, 0), unit(0.95, 0.1, 0)])]),
                  root, pol, new_id)
    gallery = load_face_gallery(conn)
    assert c1 in gallery
    assert c1 not in returnable(conn, "no-such-entry", gallery, pol, T + timedelta(minutes=30))
