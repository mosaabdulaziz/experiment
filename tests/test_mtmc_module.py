"""perception/modules/mtmc: the hooks around the cross-camera tracker (the tracker itself: test_sidecar_mtmc.py)."""
import json
import queue
import types

import numpy as np
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, ModuleStartupError, PipelineHandle, load_modules
from perception.modules import mtmc as mtmc_mod
from perception.modules.mtmc import Mtmc, Settings
from perception.modules.mtmc.palette import gid_color_rgb


class FakeManager:
    def __init__(self, gids=None):
        self.gids, self.calls, self.polled, self.closed = gids or {}, [], 0, False
        self.plate_events, self.assigned, self.exited, self.direction = [], [], [], []
        self.vehicle_gallery, self._embed_queue = None, queue.Queue()

    def poll_gate(self, r):
        self.polled += 1

    def process_frame(self, *args, **kw):
        self.calls.append((args, kw))
        return self.gids

    def _drain(self, name):
        events, setattr_ = getattr(self, name), setattr
        setattr_(self, name, [])
        return events

    pop_plate_events = lambda self: self._drain("plate_events")
    pop_assignment_events = lambda self: self._drain("assigned")
    pop_exit_events = lambda self: self._drain("exited")
    pop_direction_events = lambda self: self._drain("direction")

    def health(self):
        return {"open_tracks": 3.0}

    def close(self):
        self.closed = True


def _services(emitted=None):
    emitted = [] if emitted is None else emitted
    return EngineServices(redis=object(), redis_factory=lambda: None,
                          emit=lambda key, fields, maxlen=64: emitted.append((key, fields, maxlen)) or True,
                          request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda i, f: None,
                          telemetry=None, mux_size=(1280, 720))


def _module(manager=None, emitted=None, **settings):
    mod = Mtmc(Settings(**settings), _services(emitted))
    mod.manager = manager
    return mod


def _frame(objects, need_meta=True, bgr="PX"):
    f = FrameView(stream_id="cam-1", state=None, pad_index=0, frame_num=12, pts=3.5, now=3.5, need_meta=need_meta,
                  mux_size=(1280, 720), bgr=bgr)
    f.objects = objects
    return f


def _vehicle(oid, box=(100, 100, 300, 200), conf=0.9, **kw):
    return ObjectView(oid, "vehicle", conf, conf, *box, **kw)


# ── transform / pixels ───────────────────────────────────────────────────────────────────────────────────

def test_it_asks_for_the_boxes_in_the_detections_and_publishes_every_box_it_can_follow():
    shadow = ObjectView(2, "vehicle", -0.1, 0.55, 0, 0, 10, 10)                   # drawn but not detector-confirmed
    shadow.publish = False
    face = ObjectView(3, "face", 0.8, 0.8, 0, 0, 5, 5, kind="face", publish=False, layer=0)
    frame = _frame([_vehicle(1), shadow, face])
    _module().transform(frame)
    assert frame.dets_with_boxes is True
    assert shadow.publish is True and face.publish is False                         # only ordinary objects are changed


def test_it_wants_pixels_only_on_meta_frames_that_have_detections():
    mod = _module(FakeManager())
    assert mod.wants_pixels(_frame([_vehicle(1)])) is True
    assert mod.wants_pixels(_frame([])) is False
    unpublished = _vehicle(1)
    unpublished.publish = False
    assert mod.wants_pixels(_frame([unpublished])) is False
    assert mod.wants_pixels(_frame([_vehicle(1)], need_meta=False)) is False
    assert _module(None).wants_pixels(_frame([_vehicle(1)])) is False


# ── analyze ───────────────────────────────────────────────────────────────────────────────────────────────

def test_the_tracker_gets_the_detections_the_drawn_boxes_and_the_assigned_plates():
    mgr = FakeManager()
    veh = _vehicle(1, conf=0.912345)
    face = ObjectView(9, "face", 0.7, 0.7, 5, 5, 8, 8, kind="face", publish=False, layer=0)
    plate = ObjectView(-1, "plate", 0.8, 0.8, 150, 250, 80, 30, parent_id=1, kind="plate", draw=False, publish=False)
    shown = ObjectView(-1, "plate", 0.8, 0.8, 152, 251, 80, 30, parent_id=1, kind="plate_shown", publish=False)
    _module(mgr).analyze(_frame([veh, face, plate, shown]), FrameOutput())
    ((args, kw),) = mgr.calls
    assert args[:4] == ("cam-1", 12, 3.5, [["vehicle", 0.912, 1]])
    assert kw["frame_bgr"] == "PX"
    assert [b[6] for b in kw["boxes"]] == [1, 9]                                    # the drawn plate box is not a vehicle
    assert kw["plates"] == [(150, 250, 80, 30, 0.8, 1)]
    assert mgr.polled == 1


def test_global_ids_are_published_and_written_under_each_box_but_never_under_a_face():
    mgr = FakeManager(gids={1: 4})
    veh = _vehicle(1)
    face = ObjectView(1, "face", 0.7, 0.7, 5, 5, 8, 8, kind="face", publish=False, layer=0)   # same id as its owner
    out = FrameOutput()
    _module(mgr).analyze(_frame([veh, face]), out)
    assert out.fields["global_ids"] == {1: 4}
    assert out.annotations[veh].extra == [("gid:4", gid_color_rgb(4), 0.75, 2)]
    assert face not in out.annotations
    quiet = FrameOutput()
    _module(FakeManager()).analyze(_frame([_vehicle(1)]), quiet)
    assert quiet.fields == {} and quiet.annotations == {}


def test_a_read_plate_is_written_on_the_drawn_plate_box_with_its_confidence():
    mod = _module(FakeManager())
    mod.plate_digits[("cam-1", 1)] = " 5356"
    shown = ObjectView(-1, "plate", 0.87, 0.87, 152, 251, 80, 30, parent_id=1, kind="plate_shown", publish=False)
    other = ObjectView(-1, "plate", 0.6, 0.6, 400, 251, 80, 30, parent_id=2, kind="plate_shown", publish=False)
    out = FrameOutput()
    mod.analyze(_frame([shown, other]), out)
    assert out.annotations[shown].text == "plate 5356 0.87"
    assert other not in out.annotations                                              # nothing read for vehicle 2 yet


def test_the_trackers_events_are_announced_on_their_streams():
    emitted = []
    mgr = FakeManager()
    mgr.assigned = [{"global_id": 4, "cam_id": "cam-1"}]
    mgr.exited = [{"global_id": 4}]
    mgr.direction = [{"global_id": 4, "cam_id": "cam-1", "direction": "entering", "orientation": "front",
                      "confidence": 0.9}]
    _module(mgr, emitted).analyze(_frame([_vehicle(1)]), FrameOutput())
    assert [(k, m) for k, _f, m in emitted] == [("iris:mtmc:assigned", 2000), ("iris:mtmc:exited", 2000),
                                                ("iris:mtmc:direction", 2000)]
    assert emitted[0][1] == {"global_id": "4", "cam_id": "cam-1"}


def test_a_failing_tracker_costs_the_frame_nothing_else():
    mgr = FakeManager(gids={1: 4})
    mgr.process_frame = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    out = FrameOutput()
    _module(mgr).analyze(_frame([_vehicle(1)]), out)
    assert out.fields == {}


# ── plate announcements ─────────────────────────────────────────────────────────────────────────────────────

def _event(crop=None):
    return {"global_id": 4, "cam_id": "cam-1", "local_id": 1, "conf": 0.8, "box": [1, 2, 3, 4], "crop": crop}


def test_without_ocr_the_best_plate_crop_is_saved_and_announced(tmp_path):
    emitted = []
    mod = _module(FakeManager(), emitted, plates={"dir": str(tmp_path)})
    mod._publish_plate(_event(np.full((20, 60, 3), 200, np.uint8)))
    assert (tmp_path / "4.jpg").exists()
    ((key, fields, maxlen),) = emitted
    assert key == "iris:mtmc:plates" and fields["file"] == str(tmp_path / "4.jpg") and json.loads(fields["box"]) == [1, 2, 3, 4]


def test_with_ocr_the_crop_is_queued_for_reading_and_nothing_is_announced_until_it_is_read(tmp_path):
    emitted, submitted = [], []
    mod = _module(FakeManager(), emitted, plates={"dir": str(tmp_path)})
    mod.plate_book = types.SimpleNamespace(submit=lambda *a: submitted.append(a))
    mod._publish_plate(_event(np.full((20, 60, 3), 200, np.uint8)))
    assert len(submitted) == 1 and submitted[0][:2] == (4, "cam-1") and emitted == []
    mod._publish_plate(_event(None))                                                # no frame copy: logged once, not queued
    assert len(submitted) == 1


def test_a_reading_is_announced_with_its_text_and_its_digits_are_remembered(tmp_path):
    emitted = []
    mod = _module(FakeManager(), emitted, plates={"dir": str(tmp_path)})
    mod._on_plate_read(4, {"cam": "cam-1", "local_id": 1, "det_conf": 0.8, "text": "5356ros", "ocr_conf": 0.91})
    assert mod.plate_digits[("cam-1", 1)] == " 5356"
    ((key, fields, _),) = emitted
    assert key == "iris:mtmc:plates" and fields["text"] == "5356ros" and fields["file"].endswith("4.png")


# ── build, settings ─────────────────────────────────────────────────────────────────────────────────────────

def test_settings_validate_every_section():
    Settings(config={"iou_threshold": 0.2}, vlm_arbiter={"enabled": True, "provider": "openai"},
             vehicle_crops={"enabled": False}, plates={"dir": "d", "ocr_model": "m", "ocr_dict": "k"})
    for bad in ({"config": {"nope": 1}}, {"vlm_arbiter": {"nope": 1}}, {"vehicle_crops": {"nope": 1}},
                {"plates": {"nope": 1}}, {"plates": {"ocr_model": "m"}}, {"vlm_arbiter": 5}):
        with pytest.raises(ModuleConfigError):
            Settings(**bad)


def test_a_tracker_that_cannot_start_refuses_startup_instead_of_silently_tracking_nothing(monkeypatch):
    mod = _module(None)
    monkeypatch.setattr(mtmc_mod, "MTMCManager", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no torch")))
    with pytest.raises(ModuleStartupError, match="could not start: no torch"):
        mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))


def test_gauges_stop_and_gid_persistence(monkeypatch):
    mgr = FakeManager()
    mod = _module(mgr)
    gauges = mod.gauges()
    assert gauges["mtmc_open_tracks"]() == 3.0 and gauges["q_embed_depth"]() == 0
    assert _module(None).gauges() == {}
    mod.stop()
    assert mgr.closed


def test_it_loads_by_name_with_the_real_config_names():
    assert load_modules({"mtmc": {"config": {"iou_threshold": 0.3}}}, _services()).names() == ["mtmc"]


def test_the_tracker_is_built_enabled_even_though_the_config_does_not_say_so(monkeypatch):
    built = {}

    class Recorder:
        def __init__(self, cfg, vlm_arbiter=None):
            built["cfg"], self.association, self.vehicle_gallery = cfg, types.SimpleNamespace(), None

    monkeypatch.setattr(mtmc_mod, "MTMCManager", Recorder)
    mod = Mtmc(Settings(config={"iou_threshold": 0.3}, vehicle_crops={"enabled": False}), _services())
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    assert built["cfg"].enabled is True and built["cfg"].iou_threshold == 0.3
