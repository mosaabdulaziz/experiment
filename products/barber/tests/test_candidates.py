from candidates import VisitPool
from domain import BodyCandidate, FaceCandidate


def face(q, name):
    return FaceCandidate("barber-c", 1, 0.0, f"faces/d/{name}.jpg", (1.0, 0.0), q)


def test_pool_keeps_the_best_faces_and_returns_what_it_drops():
    pool = VisitPool(max_faces=2, max_bodies=1)
    assert pool.add_face(face(0.5, "a")) == []
    assert pool.add_face(face(0.7, "b")) == []
    assert pool.add_face(face(0.6, "c")) == ["faces/d/a.jpg"]          # c beats a
    assert pool.add_face(face(0.1, "d")) == ["faces/d/d.jpg"]          # d never makes it
    assert [f.crop_path for f in pool.faces] == ["faces/d/b.jpg", "faces/d/c.jpg"]


def test_pool_bodies_and_all_paths():
    pool = VisitPool(max_faces=2, max_bodies=1)
    pool.add_face(face(0.5, "a"))
    assert pool.add_body(BodyCandidate("barber-l", 9, 0.0, "bodies/d/x.jpg", None, 0.4)) == []
    # 2026-09-29 (user): the earliest body is always kept, even over a better later one
    assert pool.add_body(BodyCandidate("barber-l", 9, 1.0, "bodies/d/y.jpg", None, 0.9)) == ["bodies/d/y.jpg"]
    assert sorted(pool.paths()) == ["bodies/d/x.jpg", "faces/d/a.jpg"]


def test_drop_track_removes_only_that_tracks_candidates():
    pool = VisitPool(max_faces=5, max_bodies=5)
    pool.add_face(face(0.5, "a"))                                                  # barber-c / 1
    pool.add_face(FaceCandidate("barber-l", 8, 0.0, "faces/d/b.jpg", (1.0, 0.0), 0.9))
    pool.add_body(BodyCandidate("barber-l", 8, 0.0, "bodies/d/x.jpg", None, 0.4))
    pool.add_body(BodyCandidate("barber-l", 7, 0.0, "bodies/d/y.jpg", None, 0.4))
    assert sorted(pool.drop_track("barber-l", 8)) == ["bodies/d/x.jpg", "faces/d/b.jpg"]
    assert sorted(pool.paths()) == ["bodies/d/y.jpg", "faces/d/a.jpg"]
    assert pool.drop_track("barber-l", 8) == []


def test_drop_track_since_keeps_an_earlier_body_with_the_same_id():
    # 2026-09-26 (final B review C1a): the tracker's ids restart with the sidecar
    pool = VisitPool(max_faces=5, max_bodies=5)
    pool.add_body(BodyCandidate("barber-l", 7, 10.0, "bodies/d/old.jpg", None, 0.4))
    pool.add_body(BodyCandidate("barber-l", 7, 50.0, "bodies/d/new.jpg", None, 0.4))
    assert pool.drop_track("barber-l", 7, since=40.0) == ["bodies/d/new.jpg"]
    assert pool.paths() == ["bodies/d/old.jpg"]


def test_a_visit_pool_never_evicts_its_first_body():
    # 2026-09-29 (user): the customer card shows the earliest body crop
    from candidates import VisitPool
    from domain import BodyCandidate
    pool = VisitPool(5, 2)
    first = BodyCandidate("barber-l", 7, 1.0, "b/first.jpg", None, 0.1)
    dropped = pool.add_body(first)
    for i in range(3):
        dropped += pool.add_body(BodyCandidate("barber-l", 7, 2.0 + i, f"b/{i}.jpg", None, 0.9))
    assert "b/first.jpg" in [b.crop_path for b in pool.bodies] and "b/first.jpg" not in dropped
    assert len(pool.bodies) == 2
