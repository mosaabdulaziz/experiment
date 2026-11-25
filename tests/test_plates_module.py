"""perception/modules/plates: one steady plate per vehicle, handed on as objects."""
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, load_modules
from perception.modules.plates import Plates, Settings
from perception.engine.sources import make_stream_state

MUX = (1280, 720)


def _services():
    states = {"cam-1": make_stream_state("cam-1")}
    return EngineServices(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True,
                          request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda i, f: None,
                          telemetry=None, states=states, mux_size=MUX)


def _module(**settings):
    return Plates(Settings(**settings), _services())


def _vehicle(oid, box=(200, 200, 400, 300)):
    return ObjectView(oid, "vehicle", 0.9, 0.9, *box)


def _plate_det(parent, box, conf=0.8):
    return ObjectView(0, "plate_det", conf, conf, *box, parent_id=parent)


def _frame(objects, pts=1.0):
    f = FrameView(stream_id="cam-1", state=None, pad_index=0, frame_num=1, pts=pts, now=pts, mux_size=MUX)
    f.objects = objects
    return f


def _kinds(frame, kind):
    return [o for o in frame.objects if o.kind == kind]


def test_a_plate_on_its_vehicle_is_kept_and_drawn_on_it():
    veh, det = _vehicle(7), _plate_det(7, (350, 440, 100, 40))
    frame = _frame([veh, det])
    _module().transform(frame)
    (fresh,) = _kinds(frame, "plate")
    (shown,) = _kinds(frame, "plate_shown")
    assert (fresh.parent_id, fresh.draw, fresh.publish, fresh.conf) == (7, False, False, 0.8)
    assert (shown.parent_id, shown.draw, shown.publish, shown.id, shown.label) == (7, True, False, -1, "plate")
    assert (det.draw, det.publish, det.kind) == (False, False, "plate_det")      # never a vehicle of its own


def test_only_the_plate_nearest_the_bottom_centre_of_a_vehicle_survives():
    veh = _vehicle(7)
    low, high = _plate_det(7, (350, 440, 100, 40)), _plate_det(7, (250, 230, 100, 40))
    frame = _frame([veh, low, high])
    _module().transform(frame)
    assert [(o.left, o.top) for o in _kinds(frame, "plate")] == [(350, 440)]


def test_a_plate_too_small_or_oddly_shaped_to_read_is_dropped():
    for box in ((350, 440, 10, 6), (350, 440, 30, 40), (350, 440, 300, 20)):     # tiny, taller than wide, far too wide
        frame = _frame([_vehicle(7), _plate_det(7, box)])
        _module().transform(frame)
        assert _kinds(frame, "plate") == [], box


def test_a_plate_below_the_confidence_threshold_is_ignored():
    frame = _frame([_vehicle(7), _plate_det(7, (350, 440, 100, 40), conf=0.2)])
    _module(confidence_threshold=0.3).transform(frame)
    assert _kinds(frame, "plate") == []


def test_a_plate_whose_vehicle_is_gone_goes_to_the_smallest_vehicle_holding_it():
    big, small = _vehicle(1, (100, 100, 800, 500)), _vehicle(2, (300, 300, 300, 200))
    frame = _frame([big, small, _plate_det(99, (420, 440, 100, 40))])           # vehicle 99 was dropped upstream
    _module().transform(frame)
    assert [o.parent_id for o in _kinds(frame, "plate")] == [2]


def test_a_plate_outside_every_vehicle_is_dropped():
    frame = _frame([_vehicle(7), _plate_det(99, (900, 600, 100, 40))])
    _module().transform(frame)
    assert _kinds(frame, "plate") == [] and _kinds(frame, "plate_shown") == []


def test_a_plate_is_held_on_its_vehicle_between_detections_and_then_let_go():
    mod = _module(hold_seconds=1.5)
    first = _frame([_vehicle(7), _plate_det(7, (350, 440, 100, 40))], pts=1.0)
    mod.transform(first)
    between = _frame([_vehicle(7, (220, 210, 400, 300))], pts=2.0)               # the detector did not run; vehicle moved
    mod.transform(between)
    assert _kinds(between, "plate") == []                                      # nothing fresh to hand on...
    (held,) = _kinds(between, "plate_shown")                                   # ...but a plate still drawn, moved with it
    assert held.left > 350 and held.parent_id == 7
    gone = _frame([_vehicle(7)], pts=5.0)
    mod.transform(gone)
    assert _kinds(gone, "plate_shown") == []


def test_the_shown_plates_are_the_plates_meta_field():
    mod = _module()
    frame = _frame([_vehicle(7), _plate_det(7, (350.04, 440.06, 100, 40))])
    mod.transform(frame)
    out = FrameOutput()
    mod.analyze(frame, out)
    ((l, t, w, h, conf, pid),) = out.fields["plates"]
    assert (l, t, pid) == (350.0, 440.1, 7) and conf == 0.8
    out = FrameOutput()
    mod.analyze(_frame([_vehicle(7)]), out)
    assert out.fields == {}


def test_it_asks_for_its_own_detector_with_the_batch_size_in_its_config():
    (spec,) = _module(config="/m/plate.txt", engine="/m/plate.engine").gies()
    assert (spec.name, spec.role, spec.batch_from_mux, spec.engine_path) == ("sgie_plate", "secondary", False,
                                                                           "/m/plate.engine")
    assert _module().gies()[0].engine_path is None                             # no explicit engine: the config's own


def test_settings_refuse_nonsense_and_it_loads_by_name():
    for bad in ({"confidence_threshold": 1.5}, {"min_aspect": 7.0, "max_aspect": 6.0}, {"hold_seconds": -1}):
        with pytest.raises(ModuleConfigError):
            Settings(**bad)
    assert load_modules({"plates": {"hold_seconds": 2.0}}, _services()).names() == ["plates"]
