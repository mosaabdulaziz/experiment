"""perception/modules/box_stabilizer: replaces flickering boxes with steady whole-object ones."""
import pytest

from perception.engine.frame import FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, load_modules
from perception.modules.box_stabilizer import BoxStabilizerModule, Settings


def _module(**settings):
    return BoxStabilizerModule(Settings(**settings), None)


def _frame(pts, objects):
    f = FrameView(stream_id="cam-1", state=None, pad_index=0, frame_num=int(pts * 10), pts=pts, now=pts,
                  mux_size=(1000, 500))
    f.objects = objects
    return f


def _vehicle(oid, box, conf=0.8, label="vehicle", **kw):
    return ObjectView(oid, label, conf, conf, *box, **kw)


def _shown(frame):
    return [o for o in frame.objects if o.draw]


def test_a_nested_cab_box_is_dropped_when_the_whole_vehicle_box_is_present():
    whole, cab = _vehicle(1, (100, 100, 400, 200)), _vehicle(2, (110, 110, 100, 90), conf=0.5)
    frame = _frame(0.0, [whole, cab])
    _module().transform(frame)
    assert [o.id for o in _shown(frame)] == [1]
    assert not whole.draw and not whole.publish and not cab.draw      # the originals are replaced, not shown twice


def test_every_emitted_box_is_published_so_a_shadow_track_stays_in_the_feed():
    shadow = ObjectView(3, "vehicle", -0.1, 0.55, 100, 100, 200, 100)
    frame = _frame(0.0, [shadow])
    _module().transform(frame)
    (out,) = _shown(frame)
    assert out.publish and out.conf == pytest.approx(0.55)                            # the tracker's confidence is what is shown


def test_a_vanished_vehicle_keeps_a_held_box_for_the_hold_window_only():
    mod = _module(hold_seconds=0.75)
    mod.transform(_frame(0.0, [_vehicle(1, (100, 100, 400, 200))]))
    held = _frame(0.5, [])
    mod.transform(held)
    assert [o.id for o in _shown(held)] == [1]
    gone = _frame(5.0, [])
    mod.transform(gone)
    assert _shown(gone) == []


def test_a_much_smaller_update_moves_the_held_box_without_shrinking_it():
    mod = _module()
    mod.transform(_frame(0.0, [_vehicle(1, (100, 100, 400, 200))]))
    frame = _frame(0.1, [_vehicle(1, (200, 100, 100, 100))])           # a cab-only box 100 px to the right
    mod.transform(frame)
    (box,) = _shown(frame)
    assert (box.width, box.height) == (400, 200)


def test_other_labels_and_other_kinds_are_left_alone():
    person = _vehicle(5, (10, 10, 20, 40), label="person")
    face = _vehicle(6, (12, 12, 8, 8), label="face", kind="face", layer=0)
    frame = _frame(0.0, [person, face, _vehicle(1, (100, 100, 400, 200))])
    _module().transform(frame)
    assert face.draw and face in frame.objects                         # a box another module owns is not touched
    assert {o.label for o in _shown(frame)} == {"person", "face", "vehicle"}


def test_a_failure_leaves_the_raw_boxes_as_they_were(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod.stabilizer, "stabilize", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    raw = _vehicle(1, (100, 100, 400, 200))
    frame = _frame(0.0, [raw])
    mod.transform(frame)
    assert frame.objects == [raw] and raw.draw and raw.publish


def test_settings_are_validated_and_it_loads_by_name():
    for bad in ({"labels": []}, {"containment_ratio": 0}, {"containment_ratio": 1.5}, {"hold_seconds": -1}):
        with pytest.raises(ModuleConfigError):
            Settings(**bad)
    services = EngineServices(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True,
                              request_clip=lambda *a, **k: None, stream_ids=[], timer=lambda i, f: None, telemetry=None)
    assert load_modules({"box_stabilizer": {"labels": ["truck"], "hold_seconds": 1.0}}, services).names() == ["box_stabilizer"]
