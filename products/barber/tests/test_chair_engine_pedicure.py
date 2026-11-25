"""Pedicure chairs in the chair engine (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md)."""
from chair_engine import ChairEngine
from domain import Appearance, TrackObs
from zones import FrameGeometry

from test_chair_engine import P, SITTER, T0, ZONES, frames, sq

POL = {**P, "pd_min_s": 300.0}
PZ = [{"id": "r-P1", "name": "P1", "type": "chair", "points": sq(0.10, 0.10, 0.30, 0.40), "kind": "pedicure"},
      {"id": "r-P1-seat", "name": "P1 seat", "type": "seat", "points": sq(0.15, 0.25, 0.25, 0.35), "seat_of": "r-P1"}]
SEATED = (TrackObs(4, (140, 50, 120, 310), 0.9, SITTER),)


def eng(zones):
    return ChairEngine(zones, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")


def test_a_pedicure_chair_sitting_is_a_pedicure_never_a_customer():
    e = eng({"barber-r": PZ})
    ups = [e.observe(frames(T0, 400, SEATED, cam="barber-r"), [], POL)]
    ch = e.live()["chairs"]["r-P1"]
    assert ch["kind"] == "pedicure" and ch["state"] == "sitting" and ch["session_id"]
    assert e.live()["customers"] == []
    assert all(not u.visits and not u.sessions and not u.sittings for u in ups)
    assert any(u.pedicures for u in ups)
    assert e.occupancy()["haircuts"] == 0


def test_a_pedicure_ends_into_a_closed_record_with_its_time():
    e = eng({"barber-r": PZ})
    ups = [e.observe(frames(T0, 400, SEATED, cam="barber-r"), [], POL),
           e.observe(frames(T0 + 400, 900, (), cam="barber-r"), [], POL)]
    last = [p for u in ups for p in u.pedicures.values()][-1]
    assert not last.open and abs(last.service_s - 400) < 2


def test_a_barber_chair_beside_a_pedicure_chair_still_makes_haircuts():
    e = eng({**ZONES, "barber-r": PZ})
    seated_l = (TrackObs(5, (140, 50, 120, 310), 0.9, SITTER),)
    u = e.observe(frames(T0, 400, seated_l, cam="barber-l") + frames(T0, 400, SEATED, cam="barber-r"), [], POL)
    assert set(e.kind_of.values()) == {"haircut", "pedicure"}
    assert [s.chair_id for s in u.sessions.values()] == ["R1"] and [p.chair_id for p in u.pedicures.values()] == ["r-P1"]


def test_the_pedicurist_on_the_seat_never_opens_a_pedicure():
    e = eng({"barber-r": PZ})
    u = e.observe(frames(T0, 400, SEATED, cam="barber-r"), [Appearance("barber-r", 4, T0, None, 0.97)], POL)
    assert not u.pedicures and e.live()["chairs"]["r-P1"]["state"] == "empty"


def test_a_pedicure_chair_gets_start_photos():
    e = eng({"barber-r": PZ})
    u = e.observe(frames(T0, 400, SEATED, cam="barber-r"), [], {**POL, "hp_photos": 1.0})
    (pid,) = {p.id for p in u.pedicures.values()}
    assert [(r.haircut_id, r.kind) for r in u.photos][:1] == [(pid, "start")]


def test_a_pedicure_sitters_crops_are_discarded():
    """Inside ce_picture_window_s a barber chair keeps its sitter's crop for his card; a pedicure chair never does."""
    e = eng({"barber-r": PZ})
    e.observe(frames(T0, 20, SEATED, cam="barber-r"), [], POL)
    u = e.observe(frames(T0 + 20, 20, SEATED, cam="barber-r"),
                  [Appearance("barber-r", 4, T0 + 30, None, 0.1, "bodies/a.jpg", 0.5)], POL)
    assert "bodies/a.jpg" in u.discard_crops


def test_removing_a_pedicure_chair_mid_session_closes_it():
    e = eng({"barber-r": PZ})
    e.observe(frames(T0, 400, SEATED, cam="barber-r"), [], POL)
    e.set_zones({"barber-r": []})
    u = e.observe([], [], POL)
    (p,) = u.pedicures.values()
    assert not p.open


def test_switching_a_chair_to_pedicure_mid_haircut_closes_the_haircut():
    e = eng(ZONES)
    e.observe(frames(T0, 400, (TrackObs(5, (140, 50, 120, 310), 0.9, SITTER),), cam="barber-l"), [], POL)
    zones = {**ZONES, "barber-l": [dict(z, kind="pedicure") if z["type"] == "chair" else z for z in ZONES["barber-l"]]}
    e.set_zones(zones)
    u = e.observe([], [], POL)
    assert [s.open for s in u.sessions.values()] == [False] and e.kind_of["R1"] == "pedicure"
