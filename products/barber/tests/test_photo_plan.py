"""When a haircut's photos are asked for (2026-10-22, spec haircut photos section 2)."""
import pytest

from domain import EngineUpdate
from haircuts import Haircut
from photo_plan import PhotoPlanner, crop_rect, outline_boxes
from policy import DEFAULT_POLICY

P = {**DEFAULT_POLICY, "hp_photos": 1.0}
CAM, OUT = {"R1": "cam-l"}, {"R1": (0.10, 0.10, 0.30, 0.40)}
BOX = (0.14, 0.05, 0.26, 0.36)


def step(pl, t, open_, recent=None, box_at=None, policy=P, pending=()):
    u = EngineUpdate()
    pl.step(t, open_, recent or {}, CAM, OUT, {"R1": (t if box_at is None else box_at, BOX)}, policy, u, pending=pending)
    return u


def test_the_close_up_joins_the_outline_and_the_sitter_widens_and_raises_it_within_the_frame():
    assert crop_rect((0.10, 0.10, 0.30, 0.40), BOX, 0.25, 0.5) == pytest.approx((0.05, 0.0, 0.35, 0.40))
    assert crop_rect((0.10, 0.50, 0.30, 0.70), None, 0.25, 0.5) == pytest.approx((0.05, 0.40, 0.35, 0.70))
    assert crop_rect((0.0, 0.5, 1.0, 0.9), None, 0.5, 2.0) == pytest.approx((0.0, 0.0, 1.0, 0.9))


def test_the_outline_box_is_the_owner_cameras_chair_zone():
    zones = {"cam-l": [{"id": "l-R1", "type": "chair", "pair": "R1", "points": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5]]}],
             "cam-c": [{"id": "c-R1", "type": "chair", "pair": "R1", "points": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9]]}]}
    assert outline_boxes(zones, {"R1": "cam-l"}) == {"R1": pytest.approx((0.1, 0.2, 0.3, 0.5))}


def test_one_start_photo_then_an_end_photo_every_interval_while_seated():
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]])
    first = step(pl, 1.0, {"R1": h}).photos
    assert [(r.haircut_id, r.chair, r.camera, r.kind) for r in first] == [("h1", "R1", "cam-l", "start")]
    assert first[0].crop == pytest.approx((0.05, 0.0, 0.35, 0.40))
    assert not step(pl, 100.0, {"R1": h}).photos
    assert [r.kind for r in step(pl, 121.0, {"R1": h}).photos] == ["end"]
    assert not step(pl, 200.0, {"R1": h}).photos
    assert [r.kind for r in step(pl, 241.0, {"R1": h}).photos] == ["end"]


def test_nothing_while_the_customer_is_away_and_a_resume_keeps_its_start_photo():
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]])
    step(pl, 1.0, {"R1": h})
    h.sittings[-1][1] = 50.0                                  # away
    assert not step(pl, 130.0, {"R1": h}).photos
    h.sittings.append([140.0, None])                          # back: the same haircut
    assert [r.kind for r in step(pl, 141.0, {"R1": h}).photos] == ["end"]


def test_a_haircut_reopened_from_recent_gets_no_second_start_photo():
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]], confirmed=True)
    step(pl, 1.0, {"R1": h})
    h.sittings[-1][1] = 50.0
    u = step(pl, 300.0, {}, recent={"R1": h})                 # closed, still reopenable
    assert not u.photos and not u.dropped_photos
    h.sittings.append([400.0, None])
    assert [r.kind for r in step(pl, 401.0, {"R1": h}).photos] == ["end"]


def test_a_sitting_that_closes_unconfirmed_drops_its_photos_and_a_haircut_does_not():
    pl = PhotoPlanner()
    stray, cut = Haircut("h1", "R1", [[0.0, None]]), Haircut("h2", "R1", [[500.0, None]], confirmed=True)
    step(pl, 1.0, {"R1": stray})
    assert step(pl, 200.0, {}).dropped_photos == {"h1"}
    step(pl, 501.0, {"R1": cut})
    assert not step(pl, 900.0, {}).dropped_photos            # closed confirmed, not reopenable: forgotten
    assert pl.seen == {}


def test_no_fresh_sitter_box_waits_and_an_off_switch_asks_for_nothing():
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]])
    assert not step(pl, 10.0, {"R1": h}, box_at=1.0).photos   # the box is 9 s old
    assert [r.kind for r in step(pl, 11.0, {"R1": h}).photos] == ["start"]
    off = {**P, "hp_photos": 0.0}
    assert not step(PhotoPlanner(), 1.0, {"R1": Haircut("h3", "R1", [[0.0, None]])}, policy=off).photos


def test_a_haircut_first_seen_while_off_never_gets_a_late_start_photo():
    """Review focus: switched on mid-haircut, a "start" photo would be a frame from its middle."""
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]])
    step(pl, 1.0, {"R1": h}, policy={**P, "hp_photos": 0.0})
    assert [r.kind for r in step(pl, 2.0, {"R1": h}).photos] == []
    assert [r.kind for r in step(pl, 122.0, {"R1": h}).photos] == ["end"]


def test_nothing_while_re_id_decides_who_took_the_seat():
    """Final review: a reseat check (ce_reseat_check_s) keeps the haircut running while it decides; the new sitter's
    frame must not become the old customer's end photo."""
    pl, h = PhotoPlanner(), Haircut("h1", "R1", [[0.0, None]])
    step(pl, 1.0, {"R1": h})
    assert not step(pl, 200.0, {"R1": h}, pending={"R1"}).photos
    assert [r.kind for r in step(pl, 230.0, {"R1": h}).photos] == ["end"]       # the same customer: on as before
