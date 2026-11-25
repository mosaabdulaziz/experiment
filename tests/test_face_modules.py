"""face_detect, face_recognition and customer_faces, driven with stand-ins (no GStreamer, no insightface)."""
import queue
import threading
import types

import pytest

from perception.engine.gie_gate import SampleGate

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, ModuleStartupError, PipelineHandle, load_modules
from perception.modules import customer_faces as cf_mod
from perception.modules import face_detect as fd_mod
from perception.modules import face_recognition as fr_mod


def _services(**kw):
    base = dict(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True, request_clip=lambda *a, **k: None,
                stream_ids=["cam-1"], timer=lambda i, f: None, telemetry=None, mux_size=(1000, 500),
                index_to_id={0: "cam-1"},
                engine=types.SimpleNamespace(pipeline_preview=False, infer_interval=1, sample_interval=0.1))
    base.update(kw)
    return EngineServices(**base)


def _frame(objects, need_meta=True, bgr="px", stream="cam-1"):
    f = FrameView(stream_id=stream, state=None, pad_index=0, frame_num=1, pts=2.0, now=2.0, buf_pts=2_000_000_000,
                  need_meta=need_meta, mux_size=(1000, 500), capture_size=(2000, 1000), bgr=bgr)
    f.objects = objects
    return f


def _person(oid, box=(100, 100, 200, 400), **kw):
    return ObjectView(oid, "person", 0.9, 0.9, *box, **kw)


def _face(box=(150, 110, 40, 50), conf=0.8, parent=None):
    return ObjectView(0, "face", conf, conf, *box, parent_id=parent, draw=False, publish=False)


def _detect(**settings):
    return fd_mod.FaceDetect(fd_mod.Settings(**settings), _services())


# ── face_detect ────────────────────────────────────────────────────────────────────────────────────────────

def test_a_face_is_drawn_first_on_its_owners_id_and_published_in_meta():
    person, face = _person(7), _face()
    frame = _frame([person, face])
    mod = _detect()
    mod.transform(frame)
    boxes = [o for o in frame.objects if o.kind == "face"]
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.id, b.parent_id, b.layer, b.draw, b.publish) == (7, 7, 0, True, False)
    assert b.conf == 0.8 and b.bbox == (150, 110, 40, 50)
    out = FrameOutput()
    mod.analyze(frame, out)
    assert out.fields["faces"] == [(150, 110, 40, 50, 0.8, 7)]
    assert face.draw is False and face.publish is False         # the raw detection is not drawn a second time


def test_a_face_below_the_confidence_threshold_is_left_to_the_engine():
    frame = _frame([_person(7), _face(conf=0.1)])
    _detect(confidence_threshold=0.18).transform(frame)
    assert not [o for o in frame.objects if o.kind == "face"]
    assert FrameOutput().fields == {}


def test_a_face_smaller_than_min_pixels_in_capture_pixels_is_dropped_but_a_bigger_one_stays():
    small, big = _face(box=(150, 110, 10, 10)), _face(box=(160, 120, 30, 40))
    frame = _frame([_person(7), small, big])
    _detect(min_pixels=24).transform(frame)
    kept = [o for o in frame.objects if o.kind == "face"]
    assert [o.bbox for o in kept] == [(160, 120, 30, 40)]               # 30 mux px = 60 capture px


def test_a_face_in_no_person_box_is_drawn_on_minus_one():
    frame = _frame([_person(7), _face(box=(800, 400, 40, 50))])
    _detect().transform(frame)
    (b,) = [o for o in frame.objects if o.kind == "face"]
    assert (b.id, b.parent_id) == (-1, None)


def test_the_tracked_rule_keeps_a_face_on_the_body_it_was_on_a_moment_ago():
    standing, seated = _person(1, (500, 305, 260, 560)), _person(2, (540, 310, 160, 260))
    mod = _detect(owner_mode="tracked")
    first = _frame([standing, seated, _face(box=(560, 290, 40, 50))])
    mod.transform(first)
    assert [o.id for o in first.objects if o.kind == "face"] == [1]


def test_landmarks_are_mapped_to_capture_pixels_only_when_asked_for(monkeypatch):
    raw = ([100.0, 100.0, 1.0] * 5, 100.0, 200.0)
    monkeypatch.setattr(fd_mod, "instance_mask", lambda obj: "mask")
    monkeypatch.setattr(fd_mod, "face_landmarks_raw", lambda mask: raw)
    for want in (True, False):
        frame = _frame([_person(7, (0, 0, 100, 200), raw=object()), _face(box=(10, 10, 40, 50))])
        mod = _detect(landmarks=want)
        read = mod.extract(frame)                   # the engine's object loader runs this while pyds is valid
        assert (read is not None) is want
        for o in frame.objects:
            if read:
                read(o)
        mod.transform(frame)
        (b,) = [o for o in frame.objects if o.kind == "face"]
        assert (b.attrs.get("landmarks") is not None) is want
        if want:
            assert len(b.attrs["landmarks"]) == 5 and all(len(p) == 2 for p in b.attrs["landmarks"])


def test_landmarks_raw_reads_fifteen_floats_and_the_network_size():
    from perception.engine.nvds import InstanceMask
    import numpy as np
    assert fd_mod.face_landmarks_raw(None) is None
    assert fd_mod.face_landmarks_raw(InstanceMask(np.zeros(14, np.float32), 100, 200, 0.5)) is None
    pts, w, h = fd_mod.face_landmarks_raw(InstanceMask(np.arange(20, dtype=np.float32), 100, 200, 0.5))
    assert len(pts) == 15 and (w, h) == (100.0, 200.0)


def test_landmarks_to_capture_undoes_the_padding_and_scale_of_the_network_input():
    # a 100x200 box in a 100x200 network input: scale 1, no padding; capture scale 2
    raw = ([10.0, 20.0, 1.0] * 5, 100.0, 200.0)
    pts = fd_mod.landmarks_to_capture(raw, (50.0, 60.0, 100.0, 200.0), 2.0, 2.0)
    assert pts[0] == [(50.0 + 10.0) * 2.0, (60.0 + 20.0) * 2.0]
    assert fd_mod.landmarks_to_capture(None, (0, 0, 1, 1), 1, 1) is None
    assert fd_mod.landmarks_to_capture(raw, (0, 0, 0, 10), 1, 1) is None


def test_the_gie_it_asks_for_is_a_secondary_with_its_own_batch_and_a_derived_engine_path(tmp_path):
    (spec,) = _detect(config="/models/face/config_sgie_face.txt").gies()
    assert (spec.name, spec.role, spec.batch_from_mux) == ("sgie_face", "secondary", False)
    assert spec.engine_path == "/models/face/face_det_lite.onnx_b16_gpu0_fp32.engine"
    cfg = tmp_path / "config_sgie_face.txt"
    cfg.write_text("[property]\nonnx-file=face_det_lite.onnx\nbatch-size=8\ngpu-id=0\nnetwork-mode=0\n")
    (derived,) = _detect(config=str(cfg)).gies()
    assert derived.engine_path == str(tmp_path / "face_det_lite.onnx_b8_gpu0_fp32.engine")
    (explicit,) = _detect(engine="/e/x.engine").gies()
    assert explicit.engine_path == "/e/x.engine"


def test_settings_refuse_what_cannot_work():
    for bad in ({"owner_mode": "other"}, {"owner_fallback": "other"}, {"regions_key": "k"},
                {"idle_when_unused": {"capture": True}}, {"idle_when_unused": {"gallery_key": "g", "bogus": 1}}):
        with pytest.raises(ModuleConfigError):
            fd_mod.Settings(**bad)
    fd_mod.Settings(regions_key="k", sample_only=True)
    fd_mod.Settings(idle_when_unused={"gallery_key": "g", "capture": True, "poll_seconds": 5})


def test_sample_defers_to_the_engines_clock_unless_the_gate_decided():
    mod = _detect()
    assert mod.sample(_frame([])) is None
    mod.gate = SampleGate(0.1)
    mod.gate.decide(0, 2_000_000_000, 2.0)
    assert mod.sample(_frame([])) is True                        # the decision the gate made for this very frame
    assert mod.sample(_frame([])) is None                        # taken once


def test_a_regions_key_without_a_working_gate_refuses_startup():
    mod = _detect(sample_only=True, regions_key="barber:face_regions")
    el = types.SimpleNamespace(get_property=lambda n: -1)           # no operate-on-gie-id: the gate cannot be steered
    with pytest.raises(ModuleStartupError, match="sampling gate could not be installed"):
        mod.on_built(PipelineHandle(pipeline=None, get=lambda n: el))


def test_the_gate_is_refused_with_a_warning_when_the_preview_is_on(caplog):
    mod = fd_mod.FaceDetect(fd_mod.Settings(sample_only=True), _services(
        engine=types.SimpleNamespace(pipeline_preview=True, infer_interval=1, sample_interval=0.1)))
    with caplog.at_level("WARNING", logger="iris.sidecar"):
        mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    assert mod.gate is None and any("sample_only ignored" in r.message for r in caplog.records)


def test_the_idle_monitor_switches_the_detector_with_the_gallery():
    calls = []
    els = {"classes": "0", "set": lambda v: calls.append(v)}
    el = types.SimpleNamespace(get_property=lambda n: els["classes"], set_property=lambda n, v: els["set"](v))
    redis = types.SimpleNamespace(hlen=lambda key: 0)
    mod = fd_mod.FaceDetect(fd_mod.Settings(idle_when_unused={"gallery_key": "g", "capture": False}),
                            _services(redis=redis))
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: el))
    assert calls == ["999"]                                       # empty gallery, no capture: idle at boot
    redis.hlen = lambda key: 3
    mod._monitor.poll()
    assert calls == ["999", "0"]                                  # a face was registered: the configured classes
    assert mod.gauges()["sgie_mode"]() == 1


# ── face_recognition ───────────────────────────────────────────────────────────────────────────────────────

class _Recognizer:
    def __init__(self, names=None):
        self.names, self.alive, self.enqueued = names or {}, [], []
        self._queue = queue.Queue()

    def snapshot_names(self, stream):
        return dict(self.names)

    def note_alive_tracks(self, stream, boxes):
        self.alive.append((stream, boxes))

    def enqueue(self, frame, faces, stream, pts):
        self.enqueued.append((frame, faces, stream, pts))


def _recognition(rec):
    mod = fr_mod.FaceRecognition(fr_mod.Settings(), _services())
    mod.recognizer = rec
    return mod


def _face_box(parent, box=(150, 110, 40, 50), landmarks=None):
    return ObjectView(parent, "face", 0.8, 0.8, *box, parent_id=parent, kind="face", publish=False, layer=0,
                      attrs={"landmarks": landmarks})


def test_it_asks_for_pixels_only_when_there_are_faces_to_recognize():
    mod = _recognition(_Recognizer())
    assert mod.wants_pixels(_frame([_person(7, raw=object()), _face_box(7)]))
    assert not mod.wants_pixels(_frame([_person(7, raw=object())]))
    assert not mod.wants_pixels(_frame([_face_box(7)], need_meta=False))
    assert not _recognition(None).wants_pixels(_frame([_face_box(7)]))


def test_the_clean_frame_and_the_faces_go_to_the_recognizer_after_publish():
    rec = _Recognizer()
    out = FrameOutput()
    lm = [[1, 2]] * 5
    _recognition(rec).analyze(_frame([_person(7, raw=object()), _face_box(7, landmarks=lm)]), out)
    (hook,) = out.publish_hooks
    hook("CLEAN")
    ((frame, faces, stream, pts),) = rec.enqueued
    assert (frame, stream, pts) == ("CLEAN", "cam-1", 2.0)
    assert faces == [{"left": 300, "top": 220, "width": 80, "height": 100, "parent_id": 7, "confidence": 0.8,
                      "landmarks": lm}]


def test_it_does_not_enqueue_without_pixels_or_on_a_frame_that_feeds_no_meta():
    rec = _Recognizer()
    for frame in (_frame([_face_box(7)], bgr=None), _frame([_face_box(7)], need_meta=False)):
        out = FrameOutput()
        _recognition(rec).analyze(frame, out)
        assert out.publish_hooks == []


def test_the_recognizer_is_told_which_persons_are_still_tracked_in_capture_coordinates():
    rec = _Recognizer()
    _recognition(rec).analyze(_frame([_person(7, (100, 100, 200, 400), raw=object()), _person(8)]), FrameOutput())
    assert rec.alive == [("cam-1", [(200.0, 200.0, 400.0, 800.0, 7)])]      # the object with no pyds object is skipped


def test_a_name_is_written_once_on_the_face_box_and_hidden_on_the_other_boxes_of_that_name():
    rec = _Recognizer(names={7: "Ada"})
    out = FrameOutput()
    person, face = _person(7, raw=object()), _face_box(7)
    other_with_same_name = _person(7)
    _recognition(rec).analyze(_frame([person, face, other_with_same_name]), out)
    assert out.annotations[face].text == "Ada"                              # faces are drawn first
    assert out.annotations[person].text == ""                               # the name is already on screen
    assert out.annotations[other_with_same_name].text == ""


def test_with_no_face_this_frame_the_name_stays_on_the_persons_own_box():
    rec = _Recognizer(names={7: "Ada"})
    out = FrameOutput()
    person = _person(7, raw=object())
    _recognition(rec).analyze(_frame([person]), out)
    assert out.annotations[person].text == "Ada"


def test_unnamed_boxes_and_the_minus_one_id_are_not_annotated():
    rec = _Recognizer(names={-1: "Nobody", 5: "Eve"})
    out = FrameOutput()
    nobody, someone = _face_box(-1), _person(6)
    _recognition(rec).analyze(_frame([nobody, someone]), out)
    assert out.annotations == {}


def test_a_recognizer_that_could_not_start_leaves_the_module_inert():
    out = FrameOutput()
    _recognition(None).analyze(_frame([_face_box(7)]), out)
    assert out.annotations == {} and out.publish_hooks == [] and _recognition(None).gauges() == {}


def test_recognition_settings_validate_their_nested_sections():
    s = fr_mod.Settings(capture={"min_pixels": 30}, enhancement={})
    assert s.capture.min_pixels == 30 and s.capture.novelty_threshold == 0.25 and s.enhancement.backend == "preprocess"
    assert fr_mod.Settings().capture is None and fr_mod.Settings().enhancement is None
    for bad in ({"capture": {"nope": 1}}, {"enhancement": {"nope": 1}}, {"capture": 5}):
        with pytest.raises(ModuleConfigError):
            fr_mod.Settings(**bad)


def test_recognition_needs_face_detect_listed_before_it():
    with pytest.raises(ModuleConfigError, match="requires"):
        load_modules({"face_recognition": {}}, _services())
    with pytest.raises(ModuleConfigError, match="must be listed before"):
        load_modules({"face_recognition": {}, "face_detect": {}}, _services())
    assert load_modules({"face_detect": {}, "face_recognition": {}}, _services()).names() == [
        "face_detect", "face_recognition"]


# ── customer_faces ─────────────────────────────────────────────────────────────────────────────────────────

class _Collector:
    def __init__(self):
        self.offers, self.done = [], threading.Event()

    def offer(self, stream, frame, face, now):
        self.offers.append((stream, frame, face))
        self.done.set()


def _customers(**settings):
    mod = cf_mod.CustomerFaces(cf_mod.Settings(**settings), _services())
    mod.collector = _Collector()
    return mod


def test_faces_reach_the_collector_on_the_modules_own_thread():
    mod = _customers()
    out = FrameOutput()
    mod.analyze(_frame([_face_box(7, landmarks=[[1, 2]] * 5)]), out)
    mod.start()
    try:
        out.publish_hooks[0]("CLEAN")
        assert mod.collector.done.wait(2.0)
    finally:
        mod.stop()
    ((stream, frame, face),) = mod.collector.offers
    assert (stream, frame, face["parent_id"], face["width"]) == ("cam-1", "CLEAN", 7, 80)


def test_a_full_queue_drops_the_frame_instead_of_blocking_the_publisher():
    mod = _customers(queue_size=1)
    out = FrameOutput()
    mod.analyze(_frame([_face_box(7)]), out)
    hook = out.publish_hooks[0]
    hook("a")
    hook("b")                                                        # nobody is draining: this one is dropped
    assert mod._queue.qsize() == 1


def test_it_wants_pixels_only_for_meta_frames_with_faces_and_a_collector():
    mod = _customers()
    assert mod.wants_pixels(_frame([_face_box(7)]))
    assert not mod.wants_pixels(_frame([_person(7)]))
    assert not mod.wants_pixels(_frame([_face_box(7)], need_meta=False))
    mod.collector = None
    assert not mod.wants_pixels(_frame([_face_box(7)]))


def test_customer_face_settings_are_validated_and_it_needs_face_detect():
    for bad in ({"queue_size": 0}, {"max_yaw": 0}):
        with pytest.raises(ModuleConfigError):
            cf_mod.Settings(**bad)
    with pytest.raises(ModuleConfigError, match="requires"):
        load_modules({"customer_faces": {}}, _services())


def test_a_face_box_carries_no_default_caption_but_recognition_can_name_it():
    from perception.engine.frame import FrameOutput
    frame = _frame([_person(7, (0, 0, 100, 200), raw=object()), _face(box=(10, 10, 40, 50))])
    mod = _detect()
    mod.transform(frame)
    out = FrameOutput()
    mod.analyze(frame, out)
    (face,) = [o for o in frame.objects if o.kind == "face"]
    assert out.annotations[face].text == ""
    out.annotate(face, text="Ann")                        # a later module's name wins over the silence
    assert out.annotations[face].text == "Ann"
