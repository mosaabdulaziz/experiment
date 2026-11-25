from datetime import datetime

from domain import EngineUpdate, Frame, StateEvent, TrackObs, dt


def test_engine_update_is_falsy_until_something_changes():
    u = EngineUpdate()
    assert not u
    u.events.append(StateEvent("v1", None, "waiting", "sofa", "barber-c", datetime(2026, 9, 24, 10)))
    assert u


def test_dt_is_local_naive():
    ts = datetime(2026, 9, 24, 10, 0, 5).timestamp()
    assert dt(ts) == datetime(2026, 9, 24, 10, 0, 5)


def test_frame_holds_tracks():
    f = Frame("barber-c", 1.0, (TrackObs(3, (1.0, 2.0, 3.0, 4.0)),))
    assert f.tracks[0].box[2] == 3.0


def test_engine_update_counts_registrations_and_discards():
    from domain import Registration
    u = EngineUpdate()
    u.discard_crops.append("faces/2026-09-25/a.jpg")
    assert u
    u2 = EngineUpdate()
    u2.registrations.append(Registration("v1", datetime(2026, 9, 25, 10), (), ()))
    assert u2


from domain import DoorCrossing, SittingRecord


def test_update_counts_door_crossings_and_sittings():
    assert not EngineUpdate()
    assert EngineUpdate(door_crossings=[DoorCrossing(datetime(2026, 10, 4, 12), "in", "barber-c", 3, False)])
    assert EngineUpdate(sittings=[SittingRecord("s1", 0, "R3", datetime(2026, 10, 4, 12), None)])
