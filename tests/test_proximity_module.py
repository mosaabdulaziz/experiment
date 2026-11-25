"""modules.proximity: settings, mask extraction, alerts, overlay, hot reload, the saved scale map, the depth hooks."""
import json
import logging
import time
import types

import fakeredis
import numpy as np
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, build_settings, load_modules
from perception.engine.nvds import InstanceMask, infer_done
from perception.modules.proximity import _draw_held
from perception.modules.proximity import (APPLIED_KEY, LINE_FAR, MASK_ALPHA, MODULE, PERSON_COLOR, PREVIEW_KEY,
                                          SCALE_KEY, VEHICLE_COLORS, VERSION_KEY, Settings)
from perception.modules.proximity.depth import DepthNotOnGpu, DepthWorker

FULL_PTS = [[0, 0], [1280, 0], [1280, 720], [0, 720]]
ZONES = {"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS, "proximity": {"enabled": True}}]}}
DEPTH_ZONES = {"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS,
                                    "proximity": {"enabled": True, "depth_check": True}}]}}
SLOW_ZONES = {"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS,
                                   "proximity": {"enabled": True, "stationary_motion": 0.25, "depth_check": True}}]}}
TIMES = [round(i * 0.15, 3) for i in range(31)]             # 0 .. 4.5 s
LONG = [round(i * 0.15, 3) for i in range(51)]              # 0 .. 7.5 s
NO_PTS = (1 << 64) - 1                                      # GST_CLOCK_TIME_NONE


def _services(redis, emitted, clips):
    def emit(key, fields, maxlen=64):
        emitted.append((key, fields))
        return True
    return EngineServices(redis=redis, redis_factory=lambda: redis, emit=emit,
                          request_clip=lambda sid, now, age=None: clips.append(sid), stream_ids=["cam-1"],
                          timer=lambda *a: None, telemetry=None, mux_size=(1280, 720),
                          states={"cam-1": types.SimpleNamespace(module_state={})})


def _module(tmp_path, zones=ZONES, redis=None, **settings):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zones))
    redis = redis if redis is not None else fakeredis.FakeRedis(decode_responses=True)
    emitted, clips = [], []
    mod = MODULE(build_settings(Settings, {"config_path": str(path), **settings}, "proximity"),
                 _services(redis, emitted, clips))
    mod.on_built(None)
    return mod, emitted, clips, path, redis


def _frame(t, objects, analyze=True, now=None, buf_pts=None, inferred=None):
    """A frame at pts `t`; `inferred` (True/False) gives it frame metadata saying whether the detector ran on it."""
    meta = None if inferred is None else types.SimpleNamespace(bInferDone=int(inferred))
    return FrameView("cam-1", None, 0, int(t * 100), t, t if now is None else now,
                     buf_pts=int(t * 1e9) if buf_pts is None else buf_pts, analyze=analyze, mux_size=(1280, 720),
                     capture_size=(1280, 720), frame_meta=meta, _objects=objects)


def _person(oid=1, x=450.0, foot_y=560.0, h=100.0, conf=0.9, mask=None):
    o = ObjectView(oid, "person", conf, conf, x - 15, foot_y - h, 30, h)
    if mask is not None:
        o.attrs["mask"] = mask
    return o


def _truck(oid=10, x=0.0, mask=None):
    o = ObjectView(oid, "truck", 0.8, 0.8, x, 350, 200, 150)
    if mask is not None:
        o.attrs["mask"] = mask
    return o


def _driving(t, **kw):
    return [_person(mask=kw.get("person_mask")), _truck(x=100 + 150 * t, mask=kw.get("truck_mask"))]


PERSON_MASK = InstanceMask(np.ones(32, np.float32), 4, 8, 0.5)


def _driving_masked(t):
    return _driving(t, person_mask=PERSON_MASK)


def _creeping(t):
    """A truck creeping past at 30 px/s (0.3 ph/s): close to the person from ~1.3 s to ~6.5 s."""
    return [_person(mask=PERSON_MASK), _truck(x=300 + 30 * t)]


def _drive(mod, make, times=TIMES):
    out = None
    for t in times:
        frame, out = _frame(t, make(t)), FrameOutput()
        mod.on_frame(frame, out)
        mod.analyze(frame, out)
    return out


def _alerts(emitted):
    return [f for k, f in emitted if k == "iris:alert:cam-1"]


def _raised(emitted):
    """The alerts proper: not the quiet records of pairs the depth check held back (Ticket 6)."""
    return [f for f in _alerts(emitted) if not f.get("suppressed")]


def test_defaults_are_the_design_table():
    s = Settings()
    assert (s.max_gap, s.stationary_window_s, s.stationary_motion, s.persist_s, s.grace_s, s.cooldown_s,
            s.hidden_feet) == (0.75, 2.0, 0.75, 0.25, 0.5, 10.0, "unconfirmed")
    assert s.vehicle_classes == ["car", "truck", "bus", "motorcycle"] and s.person_classes == ["person"]
    assert (s.depth_model, s.depth_max_gap) == ("", 0.25)


def test_the_rule_uses_each_frames_gap_and_the_drawn_line_end_is_steadied_over_7(tmp_path):
    # Ticket 3: smoothing the rule's gap gained nothing on held-out footage; a steady line end is for the eye
    s = Settings()
    assert (s.gap_smoothing_frames, s.line_steady_frames) == (1, 7)
    mod, *_ = _module(tmp_path, gap_smoothing_frames=5, line_steady_frames=3)
    cam = mod.services.state("proximity", "cam-1")["camera"]
    assert (cam.gap_smoothing_frames, cam.line_steady_frames) == (5, 3)


def test_the_gap_runs_to_the_lowest_quarter_of_each_vehicles_mask_by_default(tmp_path):
    assert Settings().vehicle_band == 0.25
    mod, *_ = _module(tmp_path, vehicle_band=0.0)
    assert mod.services.state("proximity", "cam-1")["camera"].vehicle_band == 0.0


def test_a_vehicle_moves_only_if_its_box_moves_too_by_default(tmp_path):
    assert Settings().box_motion == 0.5
    mod, *_ = _module(tmp_path, box_motion=0.0)
    assert mod.services.state("proximity", "cam-1")["camera"].motion.box_motion == 0.0


def test_a_moving_vehicle_must_look_parked_for_three_quarters_of_a_second_before_it_reads_parked_by_default(tmp_path):
    assert Settings().parked_hold_s == 0.75
    mod, *_ = _module(tmp_path, parked_hold_s=0.0)
    cam = mod.services.state("proximity", "cam-1")["camera"]
    assert cam.motion.parked_hold_s == 0.0
    cam._rewind()                                       # the clock went back: fresh memory, same setting
    assert cam.motion.parked_hold_s == 0.0


@pytest.mark.parametrize("bad", [{"max_gap": 0}, {"grace_s": -1}, {"vehicle_classes": []},
                                 {"hidden_feet": "maybe"}, {"scale_min_samples": 3}, {"preview_line_range": 0},
                                 {"gap_smoothing_frames": 0}, {"line_steady_frames": 0}, {"vehicle_band": 1.0},
                                 {"vehicle_band": -0.1}, {"box_motion": -0.1}, {"parked_hold_s": -0.5},
                                 {"preview_box_smoothing_s": -0.1}, {"preview_box_hold_s": -1}])
def test_bad_settings_refuse_startup(bad):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, bad, "proximity")


@pytest.mark.parametrize("gone", ["min_speed", "depth_max_age_s"])
def test_unknown_settings_refuse_startup(gone):
    with pytest.raises(ModuleConfigError, match="unknown setting"):
        build_settings(Settings, {gone: 1.0}, "proximity")


def test_the_engine_loads_it_by_name(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(ZONES))
    modules = load_modules({"proximity": {"config_path": str(path)}},
                           _services(fakeredis.FakeRedis(decode_responses=True), [], []))
    assert modules.names() == ["proximity"]


def test_a_person_near_a_moving_truck_raises_one_safety_alert_with_its_evidence(tmp_path):
    mod, emitted, clips, *_ = _module(tmp_path)
    _drive(mod, _driving)
    (a,) = _alerts(emitted)
    assert (a["kind"], a["category"], a["severity"], a["zone"]) == ("proximity", "safety", "warning", "yard")
    assert (a["person_ids"], a["vehicle_ids"], a["vehicle_class"], a["gap_unit"]) == ("1", "10", "truck", "ph")
    assert float(a["gap"]) <= 0.75 and float(a["duration_s"]) >= 0.25 and float(a["pts"]) > 0
    assert (a["feet_visible"], a["foot_source"], a["confirmation"], a["escalation"]) == ("1", "visible", "image", "0")
    assert "truck" in a["message"] and "Yard" in a["message"]
    assert clips == ["cam-1"]


@pytest.mark.parametrize("masks", ["none", "full", "broken"])
def test_the_rule_runs_with_masks_without_and_with_broken_ones(tmp_path, masks):
    kinds = {"none": (None, None),
             "full": (InstanceMask(np.ones(32, np.float32), 4, 8, 0.5), InstanceMask(np.ones(200, np.float32), 20, 10, 0.5)),
             "broken": (InstanceMask(np.ones(5, np.float32), 4, 8, 0.5), InstanceMask(np.ones(7, np.float32), 20, 10, 0.5))}
    person_mask, truck_mask = kinds[masks]
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, lambda t: _driving(t, person_mask=person_mask, truck_mask=truck_mask))
    assert len(_alerts(emitted)) == 1


def test_a_parked_truck_raises_nothing_and_is_tagged_parked(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    out = _drive(mod, lambda t: [_person(), _truck(x=300.0)])
    assert _alerts(emitted) == []
    (tag,) = [a for o, a in out.annotations.items() if o.id == 10]
    assert tag.extra and tag.extra[0][0] == "parked"


def test_no_proximity_zone_means_no_work(tmp_path):
    mod, emitted, *_ = _module(tmp_path, zones={"cam-1": {"zones": [{"id": "z", "points": FULL_PTS}]}})
    assert mod.extract(_frame(1.0, [])) is None
    _drive(mod, _driving)
    assert _alerts(emitted) == [] and "overlay" not in mod.services.state("proximity", "cam-1")


def test_frames_without_inference_are_skipped(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    for t in TIMES:
        mod.on_frame(_frame(t, _driving(t), analyze=False), FrameOutput())
    assert _alerts(emitted) == [] and mod.extract(_frame(1.0, [], analyze=False)) is None


def test_the_rule_runs_on_the_frames_the_detector_ran_on(tmp_path):
    # the engine's `analyze` counter can disagree with nvinfer's interval; the detector's own flag decides
    mod, emitted, *_ = _module(tmp_path)
    for t in TIMES:
        mod.on_frame(_frame(t, _driving(t), analyze=True, inferred=False), FrameOutput())
    assert _alerts(emitted) == [] and mod.extract(_frame(1.0, [], inferred=False)) is None
    again, emitted, *_ = _module(tmp_path)
    for t in TIMES:
        again.on_frame(_frame(t, _driving(t), analyze=False, inferred=True), FrameOutput())
    assert len(_alerts(emitted)) == 1 and again.extract(_frame(1.0, [], analyze=False, inferred=True)) is not None
    assert infer_done(types.SimpleNamespace(bInferDone=1)) is True


def test_an_untimestamped_frame_changes_nothing(tmp_path):
    mod, emitted, *_ = _module(tmp_path, zones=SLOW_ZONES)
    mod._depth = _FakeDepth((0.0, 0.05))
    for t in LONG:
        frame = _frame(1.8e10, _creeping(t), buf_pts=NO_PTS) if t == 3.0 else _frame(t, _creeping(t))
        mod.on_frame(frame, FrameOutput())
    assert len(_alerts(emitted)) == 1


def test_extract_reads_masks_only_for_the_classes_it_needs(tmp_path, monkeypatch):
    mod, *_ = _module(tmp_path)
    monkeypatch.setattr("perception.modules.proximity.instance_mask", lambda raw: ("mask of", raw))
    person, truck = _person(), _truck()
    dog = ObjectView(3, "dog", 0.9, 0.9, 0, 0, 10, 10)
    shadow = ObjectView(4, "person", -0.1, 0.5, 0, 0, 10, 10)
    kept = _person(oid=5, mask="already read")
    for o in (person, truck, dog, shadow, kept):
        o.raw = f"raw{o.id}"
    read = mod.extract(_frame(1.0, []))
    for o in (person, truck, dog, shadow, kept):
        read(o)
    assert person.attrs["mask"] == ("mask of", "raw1") and truck.attrs["mask"] == ("mask of", "raw10")
    assert "mask" not in dog.attrs and "mask" not in shadow.attrs and kept.attrs["mask"] == "already read"


def test_the_overlay_holds_the_distance_line_between_inferences(tmp_path):
    mod, *_ = _module(tmp_path)
    _drive(mod, _driving, TIMES[:15])                       # up to 2.1 s: close
    out = FrameOutput()
    mod.analyze(_frame(2.3, []), out)                      # a frame without inference, 0.2 s later
    (draw,) = _overlay(out)
    img = np.zeros((720, 1280, 3), np.uint8)
    draw(img, 1.0, 1.0)
    assert img.any()
    late = FrameOutput()
    mod.analyze(_frame(3.0, []), late)
    assert _overlay(late) == []


def test_a_zone_edit_applies_live_and_is_acknowledged(tmp_path):
    mod, emitted, _clips, path, redis = _module(tmp_path)
    assert redis.get(APPLIED_KEY) == "0"
    path.write_text(json.dumps({"cam-1": {"zones": [{"id": "yard", "points": FULL_PTS,
                                                     "proximity": {"enabled": False}}]}}))
    redis.set(VERSION_KEY, 3)
    mod.control()[VERSION_KEY]("3")
    assert redis.get(APPLIED_KEY) == "3"
    _drive(mod, _driving)
    assert _alerts(emitted) == []


def test_a_zone_edit_mid_episode_applies_the_new_gap(tmp_path):
    slow = {"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS,
                                 "proximity": {"enabled": True, "persist_s": 1.0}}]}}
    mod, emitted, _clips, path, redis = _module(tmp_path, zones=slow)
    _drive(mod, _driving, TIMES[:9])                        # close since ~0.75 s, not yet for 1 s
    path.write_text(json.dumps({"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS,
                                                     "proximity": {"enabled": True, "max_gap": 0.3}}]}}))
    redis.set(VERSION_KEY, 1)
    mod.control()[VERSION_KEY]("1")
    _drive(mod, _driving, TIMES[9:])                        # the gap (~0.6 ph) is over the new 0.3
    assert _alerts(emitted) == []


def _teach_scale(mod, start=0.0):
    for i in range(60):
        v = 200.0 + 8 * i
        h = 0.5 * v - 50
        mod.on_frame(_frame(start + i * 0.1, [ObjectView(100 + i, "person", 0.9, 0.9, 100, v - h, 20, h)]),
                     FrameOutput())


def test_the_scale_map_is_learned_saved_and_restored(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path)
    _teach_scale(mod)
    assert json.loads(redis.get(SCALE_KEY.format("cam-1")))["a"] == pytest.approx(0.5, rel=0.05)
    again, *_ = _module(tmp_path, redis=redis)
    assert again.services.state("proximity", "cam-1")["camera"].scale.ready


def test_the_scale_map_is_saved_at_most_once_a_minute_and_on_stop(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path, scale_refit_every=1)
    _teach_scale(mod)                                       # refits on every sample after the first fit
    first = redis.get(SCALE_KEY.format("cam-1"))
    _teach_scale(mod, start=6.0)
    assert redis.get(SCALE_KEY.format("cam-1")) == first    # < 60 s later: not saved again
    mod.stop()
    assert redis.get(SCALE_KEY.format("cam-1")) != first


def test_a_broken_saved_scale_map_is_relearned(tmp_path):
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.set(SCALE_KEY.format("cam-1"), "{not json")
    mod, *_ = _module(tmp_path, redis=redis)
    assert not mod.services.state("proximity", "cam-1")["camera"].scale.ready


class _FakeDepth:
    """A depth worker whose latest measurement for every pair is `latest` ((measured t, gap) or None) and whose
    latest map is `map` ((t, bgr) or None)."""
    def __init__(self, latest, map=None):
        self.measured, self.submitted, self.map = latest, [], map

    def latest_map(self, sid):
        return self.map

    def idle(self):
        return getattr(self, "is_idle", True)

    def busy(self, sid):
        return False

    def submit(self, sid, t, bgr, mux_to_frame, requests):
        self.submitted.append((sid, t, mux_to_frame, requests))
        return True

    def latest(self, sid, key):
        return self.measured

    def start(self):
        pass

    def stop(self):
        pass


def _drive_with_pixels(mod, times=TIMES, make=_driving_masked, clock=0.0, settle=None):
    bgr = np.zeros((720, 1280, 3), np.uint8)
    for t in times:
        frame = _frame(t, make(t), now=clock + t)
        mod.on_frame(frame, FrameOutput())
        if mod.wants_pixels(frame):
            mod.on_pixels(frame, bgr)
            if settle:
                settle()


def test_a_depth_zone_asks_for_frames_at_most_every_depth_every_s(tmp_path):
    mod, emitted, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    fake = mod._depth = _FakeDepth(None)
    _drive_with_pixels(mod)
    sent = [t for _sid, t, _k, _r in fake.submitted]
    assert sent and all(b - a >= 0.5 - 1e-9 for a, b in zip(sent, sent[1:]))
    assert fake.submitted[0][2] == (1.0, 1.0) and fake.submitted[0][3][0].key == ("yard", 1, 10)
    # it never answered: the alert comes without it, after the wait
    assert [(a["confirmation"], float(a["duration_s"]) >= 0.75) for a in _alerts(emitted)] == [("image", True)]


@pytest.mark.parametrize("gap, expected", [(0.05, ["depth"]), (0.5, [])])
def test_a_depth_zone_alerts_on_a_same_depth_measurement_only(tmp_path, gap, expected):
    mod, emitted, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    mod._depth = _FakeDepth((0.0, gap))
    _drive_with_pixels(mod)
    assert [a["confirmation"] for a in _raised(emitted)] == expected


def test_a_depth_zone_without_a_model_alerts_on_the_image_and_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="iris.sidecar"):
        mod, emitted, *_ = _module(tmp_path, zones=DEPTH_ZONES)
    assert "depth_model is not set" in caplog.text
    _drive(mod, _driving)
    assert [a["confirmation"] for a in _alerts(emitted)] == ["image"]


class _ApartModel:
    """Depth with the person (the box every test person has) at 5 and everything else at 20: never the same depth."""
    def __call__(self, bgr):
        depth = np.full((720, 1280), 20.0, np.float32)
        depth[460:560, 435:465] = 5.0
        return depth, 1.0


def test_a_pair_the_depth_check_puts_apart_never_alerts_with_the_real_worker(tmp_path):
    mod, emitted, *_ = _module(tmp_path, zones=SLOW_ZONES, depth_model="unused.onnx")
    worker = mod._depth = DepthWorker(_ApartModel)
    worker.start()

    def settle():
        deadline = time.monotonic() + 5
        while worker.busy("cam-1") and time.monotonic() < deadline:
            time.sleep(0.005)
    try:
        _drive_with_pixels(mod, LONG, _creeping, settle=settle)
    finally:
        worker.stop()
    assert _raised(emitted) == []


def test_the_depth_cadence_survives_the_clock_going_back(tmp_path):
    mod, _emitted, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    fake = mod._depth = _FakeDepth(None)
    _drive_with_pixels(mod)
    first_loop = len(fake.submitted)
    _drive_with_pixels(mod, clock=10.0)                     # the file looped: pts from 0 again, wall clock on
    assert any(t < 2.0 for _sid, t, _k, _r in fake.submitted[first_loop:])


def test_a_depth_model_on_the_cpu_switches_the_check_off(tmp_path, monkeypatch, caplog):
    def on_cpu(path):
        raise DepthNotOnGpu("after a warm-up inference the session runs on ['CPUExecutionProvider']")
    monkeypatch.setattr("perception.modules.proximity._load_depth", on_cpu)
    mod, emitted, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="yolo26m-depth.onnx")
    with caplog.at_level(logging.ERROR, logger="iris.sidecar"):
        mod.start()
    assert mod._depth is None and "CPUExecutionProvider" in caplog.text
    _drive(mod, _driving)
    assert [a["confirmation"] for a in _alerts(emitted)] == ["image"]


def test_a_depth_model_that_cannot_load_refuses_startup(tmp_path, monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)
    monkeypatch.setattr("perception.modules.proximity._load_depth", missing)
    mod, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="nope.onnx")
    with pytest.raises(FileNotFoundError):
        mod.start()


def test_person_masks_are_read_only_while_a_vehicle_in_a_zone_may_move(tmp_path, monkeypatch):
    mod, *_ = _module(tmp_path)
    reads = []
    monkeypatch.setattr("perception.modules.proximity.instance_mask", lambda raw: reads.append(raw) or None)

    def run(t, truck_x):
        """One frame as the engine runs it: the extractor on each object when they are read, then the rule."""
        objects = [_person(), _truck(x=truck_x)]
        for o in objects:
            o.raw = o.label
        frame = _frame(t, objects)
        read = mod.extract(frame)
        for o in objects:
            read(o)
        mod.on_frame(frame, FrameOutput())
    for t in TIMES[:12]:                                   # a parked truck: known parked after ~0.75 s
        run(t, 300.0)
    reads.clear()
    for t in TIMES[12:16]:                                 # still within stationary_window_s of the last doubt
        run(t, 300.0)
    assert reads.count("person") == 4
    for t in TIMES[16:24]:
        run(t, 300.0)
    reads.clear()
    for t in TIMES[24:28]:                                 # parked for longer: no person masks
        run(t, 300.0)
    assert reads.count("truck") == 4 and reads.count("person") == 0
    reads.clear()
    for t in [round(TIMES[28] + 0.15 * i, 3) for i in range(14)]:     # it drives off: person masks are back
        run(t, 300.0 + 150 * (t - TIMES[28]))
    assert reads.count("person") > 0


def test_a_seed_file_gives_a_camera_its_scale_map_until_one_is_saved(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"cam-1": {"a": 0.3, "b": 10.0, "n": 900}}))
    mod, *_ = _module(tmp_path, scale_seed=str(seed))
    scale = mod.services.state("proximity", "cam-1")["camera"].scale
    assert scale.ready and scale.height(500) == pytest.approx(160.0)
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.set(SCALE_KEY.format("cam-1"), json.dumps({"a": 0.5, "b": -50.0, "n": 300}))
    saved, *_ = _module(tmp_path, redis=redis, scale_seed=str(seed))           # a saved map wins
    assert saved.services.state("proximity", "cam-1")["camera"].scale.height(500) == pytest.approx(200.0)


def test_a_broken_seed_file_is_skipped_with_a_warning(tmp_path, caplog):
    seed = tmp_path / "seed.json"
    seed.write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="iris.sidecar"):
        mod, *_ = _module(tmp_path, scale_seed=str(seed))
    assert not mod.services.state("proximity", "cam-1")["camera"].scale.ready and "seed" in caplog.text


def _overlays(emitted):
    return [f for k, f in emitted if k == "iris:overlay:cam-1"]


def test_each_frame_with_a_pair_in_a_zone_puts_its_lines_boxes_and_masks_on_the_overlay_stream(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, _driving_with_masks)
    entries = _overlays(emitted)
    assert len(entries) == len(TIMES)                       # every analysed frame, close or not
    assert all((e["module"], e["w"], e["h"]) == ("proximity", "1280", "720") for e in entries)
    frames = [json.loads(e["shapes"]) for e in entries]
    styles = {s["style"] for shapes in frames for s in shapes if s["type"] == "line"}
    assert styles == {"far", "close", "latched"}
    last = frames[-1]
    boxes = {(s["style"], s["id"]): s for s in last if s["type"] == "box"}
    assert {k: b["label"] for k, b in boxes.items()} == {("person", 1): "person 1", ("vehicle", 10): "truck 10"}
    assert boxes[("vehicle", 10)]["state"] == "moving" and "state" not in boxes[("person", 1)]
    assert all(len(b["mask"]) >= 3 and all(len(pt) == 2 for pt in b["mask"]) for b in boxes.values())
    line = next(s for s in last if s["type"] == "line")
    assert (line["person"], line["vehicle"]) == (1, 10) and len(line["p1"]) == len(line["p2"]) == 2


def test_a_standing_truck_is_measured_for_the_clips_too(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, lambda t: [_person(), _truck(x=900.0)])
    shapes = json.loads(_overlays(emitted)[-1]["shapes"])
    assert {s["style"] for s in shapes if s["type"] == "line"} == {"far"}
    assert next(s for s in shapes if s.get("style") == "vehicle")["state"] == "parked"


def test_frames_without_a_person_and_a_vehicle_in_a_zone_write_nothing_to_the_overlay_stream(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, lambda t: [_truck(x=900.0)])
    assert _overlays(emitted) == []


def test_an_incident_opens_with_its_alert_and_resolves_once_the_episode_is_over(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, _driving)
    _drive(mod, lambda t: [_person()], [round(4.65 + i * 0.15, 3) for i in range(80)])    # 4.65 .. 16.5 s, no truck
    opened, resolved = _alerts(emitted)
    assert opened["incident_state"] == "opened" and len(opened["incident"]) == 12
    assert 0 <= float(opened["min_gap_pts"]) <= float(opened["pts"])
    assert (resolved["incident"], resolved["incident_state"], resolved["kind"]) == (
        opened["incident"], "resolved", "proximity")
    assert float(resolved["duration_s"]) >= 1.0 and float(resolved["min_gap"]) <= float(opened["gap"])
    assert (resolved["person_ids"], resolved["vehicle_ids"], resolved["zone"]) == ("1", "10", "yard")


def test_an_episode_cut_short_by_the_clock_going_back_is_resolved(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, _driving)
    _drive(mod, lambda t: [_person()], [0.0])                       # the file looped
    assert [a["incident_state"] for a in _alerts(emitted)] == ["opened", "resolved"]


def test_an_escalation_carries_the_same_incident(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    episode = types.SimpleNamespace(incident="abc123def456", person_ids=[1], vehicle_ids=[10], min_gap_t=3.25)
    a = types.SimpleNamespace(
        confirmation="image", gap=0.1, vehicle=types.SimpleNamespace(label="truck", id=10),
        person=types.SimpleNamespace(id=1), zone=types.SimpleNamespace(name="Yard", zone_id="yard"),
        escalated=True, severity="critical", episode=episode, held_s=2.0, depth_gap=None,
        foot=types.SimpleNamespace(hidden=False, source="visible"))
    mod._raise(_frame(3.5, []), a)
    (entry,) = _alerts(emitted)
    assert (entry["incident"], entry["incident_state"], entry["min_gap_pts"]) == ("abc123def456", "escalated", "3.250")


def test_an_incident_still_resolves_after_its_cameras_last_proximity_zone_is_turned_off(tmp_path):
    mod, emitted, _clips, path, redis = _module(tmp_path)
    _drive(mod, _driving)                                           # an incident is open
    path.write_text(json.dumps({"cam-1": {"zones": [{"id": "yard", "points": FULL_PTS,
                                                     "proximity": {"enabled": False}}]}}))
    redis.set(VERSION_KEY, 3)
    mod.control()[VERSION_KEY]("3")
    _drive(mod, lambda t: [_person()], [round(4.65 + i * 0.15, 3) for i in range(80)])
    assert [a["incident_state"] for a in _alerts(emitted)] == ["opened", "resolved"]


TRUCK_MASK = InstanceMask(np.ones(64, np.float32), 8, 8, 0.5)


def _driving_with_masks(t):
    return _driving(t, person_mask=PERSON_MASK, truck_mask=TRUCK_MASK)


def _overlay(out):
    """The preview's own drawings (lines, masks, depth map), not the boxes held after their objects vanished."""
    return [d for d in out.draws if getattr(d, "func", None) is not _draw_held]


def _preview(mod, frame_t):
    out = FrameOutput()
    mod.analyze(_frame(frame_t, []), out)
    return _overlay(out)


def test_the_live_preview_draws_the_distance_lines_and_no_masks_by_default(tmp_path):
    mod, *_ = _module(tmp_path)
    _drive(mod, _driving_with_masks, TIMES[:15])
    (lines,) = _preview(mod, 2.2)
    img = np.zeros((720, 1280, 3), np.uint8)
    lines(img, 1.0, 1.0)
    assert img.any()


def _switch(mod, redis, **layers):
    redis.set(PREVIEW_KEY, json.dumps(layers))
    mod.control()[PREVIEW_KEY](redis.get(PREVIEW_KEY))


def _tint(color, under=200.0):
    return np.array(color, float) * MASK_ALPHA + under * (1 - MASK_ALPHA)


def test_masks_and_lines_switch_on_and_off_live(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path)
    _switch(mod, redis, masks=True, lines=False)
    _drive(mod, _driving_with_masks, TIMES[:16])
    (masks,) = _preview(mod, 2.3)                                    # masks only: the lines are off
    img = np.full((720, 1280, 3), 200, np.uint8)
    masks(img, 1.0, 1.0)
    # see-through: inside the person's mask (box x 435-465, y 460-560) and the moving truck's (x 418-618, y 350-500)
    assert np.allclose(img[510, 450], _tint(PERSON_COLOR), atol=2)
    assert np.allclose(img[380, 520], _tint(VEHICLE_COLORS["moving"]), atol=2)


def _away(t):
    """A truck driving away from the person at 60 px/s: moving, and 1-2 person heights off once known to move."""
    return [_person(), _truck(x=480 + 60 * t)]


def test_the_line_range_sets_how_far_apart_a_pair_still_gets_a_line(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path)
    _drive(mod, _away, TIMES[:16])
    (lines,) = _preview(mod, 2.3)                                    # the default range: 2 person heights
    img = np.zeros((720, 1280, 3), np.uint8)
    lines(img, 1.0, 1.0)
    assert (img == LINE_FAR).all(axis=2).any()
    _switch(mod, redis, line_range=1.0)
    assert _preview(mod, 2.3) == []                                  # 1.7 ph apart: out of range now


def test_boxes_take_the_colour_of_what_they_are_and_switch_off(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path)
    out = _drive(mod, lambda t: [_person(), _truck(x=100 + 150 * t), _truck(oid=11, x=900.0)], TIMES[:16])
    colours = {o.id: a.color for o, a in out.annotations.items() if a.color is not None}
    assert colours == {1: PERSON_COLOR, 10: VEHICLE_COLORS["moving"], 11: VEHICLE_COLORS["parked"]}
    _switch(mod, redis, boxes=False)
    out = FrameOutput()
    frame = _frame(2.3, [_person(), _truck()])
    mod.analyze(frame, out)
    assert {o.id for o, a in out.annotations.items() if a.hidden} == {1, 10}


def _show(mod, t, objects):
    frame, out = _frame(t, objects), FrameOutput()
    mod.on_frame(frame, out)
    mod.analyze(frame, out)
    return out


def _painted(out):
    img = np.zeros((720, 1280, 3), np.uint8)
    for draw in out.draws:
        draw(img, 1.0, 1.0)
    return img


def test_the_preview_draws_each_box_steadier_than_its_detection(tmp_path):
    mod, *_ = _module(tmp_path)
    _show(mod, 2.0, [_person(x=450.0)])
    out = _show(mod, 2.04, [_person(x=470.0)])          # the detection jumps 20 px in one frame
    (note,) = [a for o, a in out.annotations.items() if o.id == 1]
    assert 435.0 < note.box[0] < 450.0                  # the drawn box moved, by well under 20 px
    assert note.color == PERSON_COLOR


def test_a_box_whose_object_vanished_is_drawn_for_half_a_second(tmp_path):
    mod, *_ = _module(tmp_path)
    _show(mod, 2.0, [_person(x=450.0)])                 # box (435, 460) - (465, 560)
    assert (_painted(_show(mod, 2.3, [])))[460, 440:460].tolist() == [list(PERSON_COLOR)] * 20
    assert not _painted(_show(mod, 2.6, [])).any()


def test_box_steadying_is_on_by_default_and_can_be_switched_off(tmp_path):
    assert (Settings().preview_box_smoothing_s, Settings().preview_box_hold_s) == (0.08, 0.5)
    mod, *_ = _module(tmp_path, preview_box_smoothing_s=0.0, preview_box_hold_s=0.0)
    _show(mod, 2.0, [_person(x=450.0)])
    out = _show(mod, 2.04, [_person(x=470.0)])
    (note,) = [a for o, a in out.annotations.items() if o.id == 1]
    assert note.box is None or note.box[0] == 455.0     # the detection's own box
    assert not _painted(_show(mod, 2.1, [])).any()


def test_a_bad_line_range_keeps_the_default(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path)
    for bad in (0, -1, "far", None):
        _switch(mod, redis, line_range=bad)
        assert mod._preview["line_range"] == 2.0


def test_a_preview_setting_saved_before_start_applies_at_start(tmp_path):
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.set(PREVIEW_KEY, json.dumps({"masks": False, "lines": False}))
    mod, *_ = _module(tmp_path, redis=redis)
    _drive(mod, _driving_with_masks, TIMES[:15])
    assert _preview(mod, 2.2) == []



def _drawn(style):
    """One horizontal line (100,300)-(500,300) drawn by the live preview on a light grey frame."""
    from perception.modules.proximity import _draw
    from perception.modules.proximity.camera import Line
    img = np.full((720, 1280, 3), 200, np.uint8)
    _draw([Line((100.0, 300.0), (500.0, 300.0), 1.5 if style == "far" else 0.5, style == "latched",
                close=style != "far")], img, 1.0, 1.0)
    return img[:, 200]


def _rows(column, color):
    return [r for r in range(len(column)) if tuple(column[r]) == tuple(color)]


def test_a_far_line_is_pale_green_apart_from_the_people_and_thick_over_a_dark_outline():
    col = _drawn("far")
    rows = _rows(col, LINE_FAR)
    assert len(rows) >= 2
    assert LINE_FAR[1] > max(LINE_FAR[0], LINE_FAR[2])                        # green the brightest (BGR)
    assert np.abs(np.array(LINE_FAR, int) - PERSON_COLOR).max() >= 60         # not the people's green
    assert col[min(rows) - 1].max() < 100 and col[max(rows) + 1].max() < 100  # dark edges on a light frame


@pytest.mark.parametrize("style, color", [("close", "LINE_CLOSE"), ("latched", "LINE_LATCHED")])
def test_close_lines_are_drawn_thicker_than_far_ones(style, color):
    from perception.modules import proximity as px
    assert len(_rows(_drawn(style), getattr(px, color))) > len(_rows(_drawn("far"), LINE_FAR))


def test_the_overlay_line_says_when_its_foot_is_hidden_and_what_the_depth_check_said():
    from perception.modules.proximity import _shapes
    from perception.modules.proximity.camera import Line
    ln = Line((450.0, 560.0), (450.0, 500.0), 0.6, False, 1, (435, 460, 30, 60), 10, (0, 350, 200, 150), "truck",
              foot_hidden=True, seen_foot=(450.0, 520.0), depth=("same", 0.0512))
    line = _shapes([ln], {}, {})[0]
    assert (line["hidden"], line["seen"], line["depth"]) == (True, [450.0, 520.0], {"state": "same", "gap": 0.051})
    plain = Line((450.0, 560.0), (450.0, 500.0), 0.6, False, 1, (435, 460, 30, 100), 10, (0, 350, 200, 150), "truck")
    assert not {"hidden", "seen", "depth"} & set(_shapes([plain], {}, {})[0])


def _drawn_line(**kw):
    """A line (300,400) -> (500,400) drawn by the live preview on a grey frame: the image."""
    from perception.modules.proximity import _draw
    from perception.modules.proximity.camera import Line
    img = np.full((720, 1280, 3), 128, np.uint8)
    close = kw.pop("close", True)
    _draw([Line((300.0, 400.0), (500.0, 400.0), 0.5 if close else 1.5, False, close=close, **kw)], img, 1.0, 1.0)
    return img


def _is(px, color, tol=40):
    return bool(np.abs(px.astype(int) - np.array(color)).max() <= tol)


def test_a_hidden_foot_is_a_hollow_ring_joined_to_the_person_by_a_dashed_leg():
    from perception.modules.proximity import LINE_CLOSE, RING_R
    img = _drawn_line(foot_hidden=True, seen_foot=(300.0, 340.0))
    assert _is(img[400, 300 - RING_R], LINE_CLOSE) and not _is(img[400, 300], LINE_CLOSE)       # ring, hollow
    leg = [_is(img[y, 300], LINE_CLOSE) for y in range(342, 400 - RING_R - 1)]
    assert 0.3 < sum(leg) / len(leg) < 0.9                                                        # dashed


def test_a_visible_foot_is_a_dot():
    from perception.modules.proximity import LINE_CLOSE
    assert _is(_drawn_line()[400, 297], LINE_CLOSE)                  # left of the foot, where the line does not go


@pytest.mark.parametrize("depth, color", [(("same", 0.05), "DEPTH_SAME"), (("apart", 0.4), "DEPTH_APART"),
                                          (("pending", None), "DEPTH_WAIT")])
def test_the_depth_verdict_follows_the_gap_in_its_colour(depth, color):
    from perception.modules import proximity as px
    region = _drawn_line(depth=depth)[380:410, 440:640]               # right of the gap label
    assert any(_is(p, getattr(px, color), 30) for p in region.reshape(-1, 3))


def test_a_line_the_depth_check_says_is_apart_is_dashed():
    from perception.modules.proximity import LINE_CLOSE
    row = _drawn_line(depth=("apart", 0.4))[400, 312:390]
    on = [_is(p, LINE_CLOSE) for p in row]
    assert 0.3 < sum(on) / len(on) < 0.9


def _far_apart(t):
    """A person and a truck nowhere near each other: no close pair, no depth request."""
    return [_person(x=1200.0, mask=PERSON_MASK), _truck(x=0.0, mask=TRUCK_MASK)]


def test_the_depth_map_switch_forces_depth_frames_on_depth_zone_cameras(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    fake = mod._depth = _FakeDepth(None)
    _drive_with_pixels(mod, make=_far_apart)
    assert fake.submitted == []                                          # switch off: only close pairs ask
    _switch(mod, redis, depth=True)
    _drive_with_pixels(mod, times=[t + 10 for t in TIMES], make=_far_apart, clock=10.0)
    assert fake.submitted and all(requests == [] for _sid, _t, _k, requests in fake.submitted)


def test_the_depth_map_switch_blends_the_latest_map_under_the_preview(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    mod._depth = _FakeDepth(None, map=(2.0, np.full((360, 640, 3), (0, 0, 255), np.uint8)))   # red
    _switch(mod, redis, depth=True)
    out = FrameOutput()
    mod.analyze(_frame(2.5, []), out)
    img = np.full((720, 1280, 3), 100, np.uint8)
    for draw in out.draws:
        draw(img, 1.0, 1.0)
    assert np.allclose(img[10, 10], (50, 50, 177.5), atol=1)              # half the map, half the frame
    out = FrameOutput()
    mod.analyze(_frame(5.0, []), out)                                      # the map is 3 s old: not shown
    assert out.draws == []


def test_the_depth_map_switch_without_a_depth_model_draws_nothing(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path, zones=DEPTH_ZONES)
    _switch(mod, redis, depth=True)
    out = FrameOutput()
    mod.analyze(_frame(1.0, []), out)
    assert out.draws == [] and not mod.wants_pixels(_frame(1.0, []))


def test_a_depth_confirmed_alert_carries_its_depth_gap_its_zones_cut_off_and_image(tmp_path):
    import base64
    zones = {"cam-1": {"zones": [{"id": "yard", "name": "Yard", "points": FULL_PTS,
                                  "proximity": {"enabled": True, "depth_check": True, "depth_max_gap": 0.3}}]}}
    mod, emitted, *_ = _module(tmp_path, zones=zones, depth_model="unused.onnx")
    mod._depth = _FakeDepth((0.0, 0.05), map=(0.0, np.full((360, 640, 3), 90, np.uint8)))
    _drive_with_pixels(mod)
    (a,) = _alerts(emitted)
    assert a["confirmation"] == "depth" and (a["depth_gap"], a["depth_max_gap"]) == ("0.050", "0.30")
    assert base64.b64decode(a["depth_image"])[:2] == b"\xff\xd8"


def test_a_pair_the_depth_check_holds_back_goes_out_as_one_quiet_record(tmp_path):
    # Ticket 6: no alert, but one record the app keeps out of the live view (no incident to open or resolve)
    import base64
    mod, emitted, clips, *_ = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    mod._depth = _FakeDepth((0.0, 0.5), map=(0.0, np.full((360, 640, 3), 90, np.uint8)))
    _drive_with_pixels(mod)
    (q,) = _alerts(emitted)
    assert (q["suppressed"], q["severity"], q["kind"]) == ("depth", "info", "proximity")
    assert q["would_be"] in ("warning", "critical")
    assert (q["depth_gap"], q["depth_max_gap"]) == ("0.500", "0.25")
    assert "held back by the depth check" in q["message"].lower()
    assert not {"incident", "incident_state"} & set(q)
    assert base64.b64decode(q["depth_image"])[:2] == b"\xff\xd8"
    assert clips == ["cam-1"]


def test_an_image_confirmed_alert_carries_no_depth_fields(tmp_path):
    mod, emitted, *_ = _module(tmp_path)
    _drive(mod, _driving)
    (a,) = _alerts(emitted)
    assert not {"depth_gap", "depth_max_gap", "depth_image"} & set(a)


def test_the_depth_map_blends_into_a_four_channel_preview_too():
    # the pipeline draws on a BGRA surface: a channel slice is not a view OpenCV writes into
    from perception.modules.proximity import _draw_depth_map
    img = np.full((720, 1280, 4), 100, np.uint8)
    _draw_depth_map(np.full((360, 640, 3), (0, 0, 255), np.uint8), img, 1.0, 1.0)
    assert np.allclose(img[10, 10, :3], (50, 50, 177.5), atol=1) and img[10, 10, 3] == 100



def _first_pair_request(tmp_path, switch):
    """When a close pair's first depth request is sent (frame time), with the Depth map switch on or off."""
    mod, _emitted, _clips, _path, redis = _module(tmp_path / str(switch), zones=DEPTH_ZONES, depth_model="unused.onnx")
    fake = mod._depth = _FakeDepth(None)
    if switch:
        _switch(mod, redis, depth=True)
    _drive_with_pixels(mod)
    return next(t for _sid, t, _k, requests in fake.submitted if requests)


def test_the_depth_map_switch_never_delays_a_close_pairs_depth_check(tmp_path):
    # map refreshes keep their own clock and never displace a pair's request (final review, Ticket 4)
    (tmp_path / "True").mkdir()
    (tmp_path / "False").mkdir()
    assert _first_pair_request(tmp_path, True) == _first_pair_request(tmp_path, False)


def test_map_refreshes_wait_for_an_idle_depth_worker(tmp_path):
    mod, _emitted, _clips, _path, redis = _module(tmp_path, zones=DEPTH_ZONES, depth_model="unused.onnx")
    fake = mod._depth = _FakeDepth(None)
    fake.is_idle = False                                                  # other cameras' frames are queued
    _switch(mod, redis, depth=True)
    _drive_with_pixels(mod, make=_far_apart)
    assert fake.submitted == []
