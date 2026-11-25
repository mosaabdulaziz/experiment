"""The pose module, driven with stand-ins (no GStreamer, no TensorRT) — the face_detect harness pattern.

ACs: AC-F2.1, AC-F3.1 (unit half), AC-F4.1, AC-F5.1, AC-F6.1, and the AC-F1.1 gies() half.
Contract: FALL_DETECTION_DESIGN.md §3.1-3.4. The sgie's items arrive with parent_id unset (§3.3);
ownership is the containment match. The published field is capture px, xyxy, 17 COCO keypoints.
"""
import types
from pathlib import Path

import numpy as np
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, ModuleStartupError, load_modules
from perception.engine.nvds import InstanceMask
from perception.modules import pose as pose_mod

REPO = Path(__file__).resolve().parents[1]


def _services(**kw):
    base = dict(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True, request_clip=lambda *a, **k: None,
                stream_ids=["cam-1"], timer=lambda i, f: None, telemetry=None, mux_size=(1000, 500),
                index_to_id={0: "cam-1"}, config={"overlay_boxes": True},     # as the safety stack's fall role
                engine=types.SimpleNamespace(pipeline_preview=True, infer_interval=1, sample_interval=0.1))
    base.update(kw)
    return EngineServices(**base)


def _frame(objects, stream="cam-1"):
    f = FrameView(stream_id=stream, state=None, pad_index=0, frame_num=1, pts=2.0, now=2.0, buf_pts=2_000_000_000,
                  need_meta=True, mux_size=(1000, 500), capture_size=(2000, 1000), bgr="px")
    f.objects = objects
    return f


def _person(oid, box=(100, 100, 200, 400), **kw):
    return ObjectView(oid, "person", 0.9, 0.9, *box, **kw)


def _mask(kps, net_w, net_h):
    """InstanceMask the way the parser writes it: 17*3 floats then the network size on the mask struct."""
    flat = np.array([v for kp in kps for v in kp], dtype=np.float32)
    return InstanceMask(flat, net_w, net_h, 0.9)


def _pose_item(kps, net_w, net_h, box=(120, 150, 150, 300)):
    """A secondary-GIE item: no parent pointer, its keypoints in the instance mask, network-input px."""
    return ObjectView(0, "pose", 0.9, 0.9, *box, parent_id=None, kind="pose", draw=False, publish=False,
                      raw=object(), attrs={}) , _mask(kps, net_w, net_h)


def _detect(**settings):
    return pose_mod.Pose(pose_mod.Settings(**settings), _services())


def _config(tmp_path, text):
    """A config file beside the model it names, the way the shipped ones are.

    The module refuses to start when the config names a model that is not there, so a fixture
    that starts the module has to have one. The two letterbox keys are added to the [property]
    section too: the module's letterbox inverse assumes them, and a config that declares none is
    refused (which is what the gate's own test says)."""
    (tmp_path / "yolo26x-pose.onnx").write_bytes(b"")
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text(_with_letterbox(text))
    return cfg


def _with_letterbox(text):
    """The config text with the two letterbox keys in its [property] section, unless it has them."""
    if "maintain-aspect-ratio" in text:
        return text
    lines = []
    for line in text.splitlines(keepends=True):
        lines.append(line)
        if line.strip().lower() == "[property]":
            lines.append("maintain-aspect-ratio=1\n")
            lines.append("symmetric-padding=1\n")
    return "".join(lines)


def _run(mod, frame, masks, monkeypatch):
    """extract → transform → analyze with the engine's instance_mask stand-in per raw object."""
    monkeypatch.setattr(pose_mod, "instance_mask", lambda raw: masks.get(id(raw)))
    read = mod.extract(frame)
    if read is not None:
        for o in frame.objects:
            read(o)
    mod.transform(frame)
    out = FrameOutput()
    mod.analyze(frame, out)
    return out


KPS = [[10.0, 20.0, 0.9]] * 17


# ── the GIE it asks for (AC-F1.1 config half) ──────────────────────────────────────────────────────────────

def test_it_asks_for_one_secondary_gie_whose_engine_is_derived_from_the_config(tmp_path):
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n")
    mod = pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services())
    (spec,) = mod.gies()
    assert (spec.name, spec.role, spec.batch_from_mux) == ("sgie_pose", "secondary", False)
    # the engine name follows from the config's own onnx-file, batch-size, gpu-id and mode
    assert spec.engine_path == str(tmp_path / "yolo26x-pose.onnx_b16_gpu0_fp32.engine")
    (explicit,) = pose_mod.Pose(pose_mod.Settings(config=str(cfg), engine="/e/x.engine"), _services()).gies()
    assert explicit.engine_path == "/e/x.engine"


def test_the_secondary_gies_engine_is_built_at_the_batch_in_its_own_config(tmp_path):
    """A secondary GIE's batch is the one in its config file -- the element's batch-size property
    is not set for it -- so the engine's name has to carry THAT batch, not the pipeline's."""
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=8\ngpu-id=0\n")
    (spec,) = pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).gies()
    assert spec.engine_path == str(tmp_path / "yolo26x-pose.onnx_b8_gpu0_fp32.engine")


# ── the GIE's re-inference interval (the feature's first GPU lever) ───────────────────────────────────────

def test_the_config_file_holds_the_re_inference_interval_and_a_mismatch_refuses_startup(tmp_path):
    cfg = _config(tmp_path, "[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                             "secondary-reinfer-interval=2\n"
                             "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    assert pose_mod.config_reinfer_interval(str(cfg)) == 2
    # the setting and the file agree: the module starts
    pose_mod.Pose(pose_mod.Settings(config=str(cfg), interval=2), _services()).on_built(None)
    # they do not: the module refuses to start rather than run at an interval nobody asked for
    with pytest.raises(ModuleStartupError):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg), interval=4), _services()).on_built(None)


def test_the_shipped_config_declares_the_interval_the_module_asks_for():
    cfg = REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "config_sgie_pose.txt"
    assert pose_mod.config_reinfer_interval(str(cfg)) == pose_mod.Settings().interval


def test_the_gie_asks_for_no_element_interval(tmp_path):
    # nvinfer's element property gates the PRIMARY GIE; a secondary one's interval is the config
    # file's `secondary-reinfer-interval`. Asking for an element interval would make the engine
    # set a property that does nothing for this GIE -- and would need perception/engine/** to
    # change, which the requirements forbid.
    cfg = _config(tmp_path, "[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                             "secondary-reinfer-interval=2\n"
                             "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    spec = pose_mod.Pose(pose_mod.Settings(config=str(cfg), interval=2), _services()).gies()[0]
    assert spec.role == "secondary"
    assert spec.interval == 0


def test_a_config_file_that_cannot_be_read_is_read_as_every_frame():
    assert pose_mod.config_reinfer_interval("no-such-file.txt") == 0


# ── the model the module runs (the config file names it, the module declares it) ───────────────────────────

def test_the_config_file_names_the_model_and_the_module_refuses_a_different_one(tmp_path):
    cfg = _config(tmp_path, "[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                             "secondary-reinfer-interval=2\n"
                             "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    assert pose_mod.config_model(str(cfg)) == "yolo26x-pose"
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)
    with pytest.raises(ModuleStartupError):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg), model="yolo11n-pose"), _services()).on_built(None)


def test_a_config_that_names_no_model_refuses_startup_naming_the_cause(tmp_path):
    """An unreadable config and a config with no onnx-file both leave nvinfer with no model."""
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text(_with_letterbox("[property]\nbatch-size=16\nsecondary-reinfer-interval=2\n"
                                   "[class-attrs-all]\npre-cluster-threshold=0.25\n"))
    with pytest.raises(ModuleStartupError, match="onnx-file"):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)


def test_a_config_that_names_a_model_that_is_not_there_refuses_startup(tmp_path):
    """The config file names the ONNX nvinfer loads. If the file is not beside the config, the
    engine cannot be built -- so the module refuses to start rather than fail inside nvinfer."""
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text(_with_letterbox("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                                   "secondary-reinfer-interval=2\n"
                                   "[class-attrs-all]\npre-cluster-threshold=0.25\n"))
    with pytest.raises(ModuleStartupError, match="yolo26x-pose.onnx"):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)
    # the model is there: the module starts
    (tmp_path / "yolo26x-pose.onnx").write_bytes(b"")
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)


def test_the_model_is_resolved_beside_the_config_file():
    """nvinfer reads the config file's paths relative to the config file's own directory -- the
    labels file and the engine it writes are there too -- so the module resolves the ONNX the
    same way the engine's own name is derived."""
    cfg = REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "config_sgie_pose.txt"
    assert pose_mod.config_onnx_path(str(cfg)) == cfg.parent / "yolo26x-pose.onnx"
    assert pose_mod.config_onnx_path("no-such-file.txt") is None


def test_the_engine_path_follows_the_model_so_another_model_can_run(tmp_path):
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                   "secondary-reinfer-interval=2\n"
                   "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    (spec,) = pose_mod.Pose(pose_mod.Settings(config=str(cfg), model="yolo26x-pose"), _services()).gies()
    assert spec.engine_path == str(tmp_path / "yolo26x-pose.onnx_b16_gpu0_fp32.engine")


def _shipped_config(tmp_path):
    """The shipped config byte for byte, beside an empty stand-in for the ONNX it names: the ONNX is a DVC pointer in
    the repo (CI does not pull it), and these tests are about what the config says, not about the model's bytes."""
    shipped = REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "config_sgie_pose.txt"
    cfg = tmp_path / shipped.name
    cfg.write_text(shipped.read_text())
    (tmp_path / pose_mod.config_onnx_path(str(shipped)).name).write_bytes(b"")
    return cfg


def test_the_shipped_config_names_the_model_the_module_declares(tmp_path):
    cfg = _shipped_config(tmp_path)
    assert pose_mod.config_model(str(cfg)) == pose_mod.Settings().model
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)


# ── the two floors: the detection's and the joints' ───────────────────────────────────────────────────────

def test_the_config_file_holds_the_detection_floor_and_a_mismatch_refuses_startup(tmp_path):
    """The detection floor is the parser's threshold (the config file's pre-cluster-threshold) and
    the JOINT floor is the module's. They are different numbers, and a config file edited without
    the module would silently change what the module publishes."""
    cfg = _config(tmp_path, "[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                             "secondary-reinfer-interval=2\n"
                             "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    assert pose_mod.config_detection_threshold(str(cfg)) == 0.25
    # the setting and the file agree: the module starts
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)
    # they do not: the module refuses to start rather than run at a floor nobody asked for
    with pytest.raises(ModuleStartupError, match="pre-cluster-threshold"):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg), detection_threshold=0.5), _services()).on_built(None)


def test_a_config_file_that_declares_no_detection_floor_refuses_startup(tmp_path):
    """A config with no pre-cluster-threshold leaves the parser with no floor at all (every
    detection becomes a detection), which is not the floor the module declared."""
    cfg = tmp_path / "config_sgie_pose.txt"
    cfg.write_text(_with_letterbox("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                                   "secondary-reinfer-interval=2\n"))
    assert pose_mod.config_detection_threshold(str(cfg)) is None
    with pytest.raises(ModuleStartupError, match="pre-cluster-threshold"):
        pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)


def test_the_shipped_config_declares_the_detection_floor_the_module_asks_for():
    cfg = REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "config_sgie_pose.txt"
    assert pose_mod.config_detection_threshold(str(cfg)) == pose_mod.Settings().detection_threshold


def test_the_joint_floor_and_the_detection_floor_are_two_different_numbers():
    """The two floors used to be called the same thing, which hid what each one did."""
    s = pose_mod.Settings()
    assert s.keypoint_confidence_threshold != s.detection_threshold
    for bad in ({"keypoint_confidence_threshold": 5.0}, {"keypoint_confidence_threshold": 0.0},
                {"detection_threshold": 5.0}, {"detection_threshold": 0.0}):
        with pytest.raises(ModuleConfigError):
            pose_mod.Settings(**bad)


# ── the letterbox the module inverts (the config file's own scaling) ─────────────────────────────────────

def test_the_config_file_declares_the_letterbox_the_module_inverts(tmp_path):
    """nvinfer scales the crop into the network's input, and the module undoes THAT scaling to put
    the joints back in the frame. Another scaling would put every joint in the wrong place -- in
    every frame, silently -- so the two keys are declared numbers, and the module refuses to start
    when they are not the ones it inverts."""
    cfg = _config(tmp_path, "[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                             "secondary-reinfer-interval=2\n"
                             "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    assert pose_mod.config_flag(str(cfg), pose_mod.ASPECT_KEY) == 1
    assert pose_mod.config_flag(str(cfg), pose_mod.PADDING_KEY) == 1
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)

    # a config that declares none: the module cannot know how nvinfer scaled the crop
    bare = tmp_path / "bare.txt"
    bare.write_text("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                    "secondary-reinfer-interval=2\n"
                    "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    assert pose_mod.config_flag(str(bare), pose_mod.ASPECT_KEY) is None
    with pytest.raises(ModuleStartupError, match="maintain-aspect-ratio"):
        pose_mod.Pose(pose_mod.Settings(config=str(bare)), _services()).on_built(None)

    # a config that stretches the crop to fit: every joint would land in the wrong place
    stretched = tmp_path / "stretched.txt"
    stretched.write_text("[property]\nonnx-file=yolo26x-pose.onnx\nbatch-size=16\n"
                         "secondary-reinfer-interval=2\nmaintain-aspect-ratio=0\n"
                         "symmetric-padding=1\n"
                         "[class-attrs-all]\npre-cluster-threshold=0.25\n")
    with pytest.raises(ModuleStartupError, match="maintain-aspect-ratio"):
        pose_mod.Pose(pose_mod.Settings(config=str(stretched)), _services()).on_built(None)


def test_the_shipped_config_declares_the_letterbox_the_module_inverts(tmp_path):
    cfg = _shipped_config(tmp_path)
    assert pose_mod.config_flag(str(cfg), pose_mod.ASPECT_KEY) == 1
    assert pose_mod.config_flag(str(cfg), pose_mod.PADDING_KEY) == 1
    pose_mod.Pose(pose_mod.Settings(config=str(cfg)), _services()).on_built(None)


# ── the parser the config names is the parser the repository ships ─────────────────────────────────────────

def test_the_config_names_the_pose_parser_the_repository_ships():
    model_dir = REPO / "perception" / "models" / "DeepStream-Yolo-Pose"
    cfg = (model_dir / "config_sgie_pose.txt").read_text()
    parser = (model_dir / "nvdsinfer_custom_impl_Yolo_pose" / "nvdsparsepose_Yolo.cpp").read_text()
    makefile = (model_dir / "nvdsinfer_custom_impl_Yolo_pose" / "Makefile").read_text()
    assert "parse-bbox-instance-mask-func-name=NvDsInferParseYoloPose" in cfg
    assert "NvDsInferParseYoloPose" in parser
    assert "libnvdsinfer_custom_impl_Yolo_pose.so" in makefile
    assert "libnvdsinfer_custom_impl_Yolo_pose.so" in cfg


def test_the_parser_packs_the_floats_the_module_reads():
    """The module reads 17 joints of x, y and conf; the parser has to write exactly that many."""
    parser = (REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "nvdsinfer_custom_impl_Yolo_pose"
              / "nvdsparsepose_Yolo.cpp").read_text()
    assert "KEYPOINT_COUNT = 17" in parser
    assert "KEYPOINT_DIMS = 3" in parser
    assert pose_mod.KEYPOINT_COUNT == 17


# ── the pose items are the network's own, not the operator's ───────────────────────────────────────────

def test_the_pose_items_are_neither_drawn_nor_published(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    item, mask = _pose_item(KPS, 200.0, 400.0, box=(100, 100, 200, 400))
    item.draw = item.publish = True              # as the engine leaves them before the module runs
    frame = _frame([person, item])
    _run(_detect(), frame, {id(item.raw): mask}, monkeypatch)
    assert item.draw is False and item.publish is False


# ── a mask that cannot carry 17×3 floats omits the person (AC-F3.1's omission half) ───────────────────────

def test_a_malformed_or_missing_instance_mask_omits_the_person_and_never_crashes(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    item, _ = _pose_item(KPS, 200.0, 400.0, box=(100, 100, 200, 400))
    short = InstanceMask(np.zeros(50, np.float32), 100, 200, 0.9)      # fewer than 17*3 floats
    for masks in ({id(item.raw): short}, {}, {id(item.raw): None}):
        out = _run(_detect(), _frame([person, item]), masks, monkeypatch)
        # a person is tracked but no pose can be read: the frame says nothing about them, so it
        # must NOT publish `items: []` (the consumer would drop the track)
        assert "pose" not in out.fields


# ── the coordinate contract (AC-F4.1) ──────────────────────────────────────────────────────────────────────

def test_the_letterbox_inverse_undoes_padding_and_scale_then_the_capture_scale():
    # box 100x100 mux in a 200x100 network input: scale 1, pillarbox pad_x 50; capture scale 1
    raw = ([[60.0, 20.0, 0.9]] * 17, 200.0, 100.0)
    pts = pose_mod.keypoints_to_capture(raw, (0.0, 0.0, 100.0, 100.0), 1.0, 1.0)
    assert pts[0] == [pytest.approx(10.0, abs=2.0), pytest.approx(20.0, abs=2.0)]
    # no padding: box 100x200 in a 100x200 network input; capture scale 2 (the face_detect math)
    raw = ([[10.0, 20.0, 0.9]] * 17, 100.0, 200.0)
    pts = pose_mod.keypoints_to_capture(raw, (50.0, 60.0, 100.0, 200.0), 2.0, 2.0)
    assert pts[0] == [pytest.approx(120.0, abs=2.0), pytest.approx(160.0, abs=2.0)]
    assert pose_mod.keypoints_to_capture(None, (0, 0, 1, 1), 1, 1) is None


def test_published_keypoints_land_where_the_geometry_says_within_two_px_and_inside_the_box(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    # the sgie ran on the person's crop (200x400 mux -> a 150x300 network input: scale 0.75, no
    # padding), so the keypoints are the network's INPUT pixels and the crop is the OWNER's rect.
    # The item's own box is the network's detection box, not the crop.
    item, mask = _pose_item(KPS, 150.0, 300.0, box=(110, 120, 190, 380))
    frame = _frame([person, item])
    out = _run(_detect(), frame, {id(item.raw): mask}, monkeypatch)
    (entry,) = out.fields["pose"]["items"]
    x, y, conf = entry["kps"][0]
    assert x == pytest.approx((100.0 + 10.0 / 0.75) * 2.0, abs=2.0)
    assert y == pytest.approx((100.0 + 20.0 / 0.75) * 2.0, abs=2.0)
    x1, y1, x2, y2 = entry["box"]
    mx, my = (x2 - x1) * 0.10, (y2 - y1) * 0.10
    assert all(x1 - mx <= kx <= x2 + mx and y1 - my <= ky <= y2 + my for kx, ky, _ in entry["kps"])


# ── the published field (AC-F2.1, AC-F3.1) ─────────────────────────────────────────────────────────────────

def test_two_tracked_people_publish_one_item_each_on_the_pose_field_only(monkeypatch):
    p1, p2 = _person(7, (100, 100, 200, 400), raw=object()), _person(9, (600, 100, 150, 350), raw=object())
    i1, m1 = _pose_item(KPS, 200.0, 400.0, box=(100, 100, 200, 400))
    i2, m2 = _pose_item([[5.0, 5.0, 0.8]] * 17, 150.0, 350.0, box=(600, 100, 150, 350))
    frame = _frame([p1, p2, i1, i2])
    out = _run(_detect(), frame, {id(i1.raw): m1, id(i2.raw): m2}, monkeypatch)
    pose = out.fields["pose"]
    assert list(out.fields) == ["pose"]                       # no other stream, no other field
    assert pose["shape"] == [2000, 1000]                      # capture w, h
    assert sorted(it["id"] for it in pose["items"]) == [7, 9]  # tracker ids, joinable with dets
    for it in pose["items"]:
        x1, y1, x2, y2 = it["box"]
        assert x1 < x2 and y1 < y2
        assert 0 <= x1 and x2 <= 2000 and 0 <= y1 and y2 <= 1000
        assert len(it["kps"]) == 17 and all(len(k) == 3 for k in it["kps"])


def test_a_person_without_confident_keypoints_is_omitted_not_zero_published(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    faint = [[10.0, 20.0, 0.05]] * 17
    item, mask = _pose_item(faint, 200.0, 400.0, box=(100, 100, 200, 400))
    out = _run(_detect(keypoint_confidence_threshold=0.30), _frame([person, item]), {id(item.raw): mask}, monkeypatch)
    assert "pose" not in out.fields           # a tracked person, no publishable pose: nothing said


# ── ownership without parent_id (AC-F5.1) ──────────────────────────────────────────────────────────────────

def test_items_with_no_parent_pointer_are_matched_to_the_person_that_contains_them(monkeypatch):
    p1, p2 = _person(7, (100, 100, 200, 400), raw=object()), _person(9, (600, 100, 150, 350), raw=object())
    i1, m1 = _pose_item(KPS, 200.0, 400.0, box=(110, 120, 180, 360))          # inside p1
    i2, m2 = _pose_item(KPS, 150.0, 350.0, box=(610, 110, 130, 330))          # inside p2
    frame = _frame([p1, p2, i1, i2])
    out = _run(_detect(), frame, {id(i1.raw): m1, id(i2.raw): m2}, monkeypatch)
    items = out.fields["pose"]["items"]
    assert sorted(it["id"] for it in items) == [7, 9]                          # nothing dropped
    by_id = {it["id"]: it for it in items}
    assert by_id[7]["box"] == [200.0, 200.0, 600.0, 1000.0]                    # p1 in capture xyxy
    assert by_id[9]["box"] == [1200.0, 200.0, 1500.0, 900.0]                   # p2 in capture xyxy


def test_when_no_rect_explains_the_item_the_owners_rect_is_the_crop(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    # the keypoints sit in the network's top-left corner: no person's letterbox inverse puts them
    # inside the item's own box, so the crop is found by containment -- the OWNER's rect, which is
    # the crop the network actually ran on (the item's own box would place the joints wrong)
    item, mask = _pose_item([[5.0, 5.0, 0.9]] * 17, 200.0, 400.0, box=(110, 120, 180, 360))
    frame = _frame([person, item])
    out = _run(_detect(), frame, {id(item.raw): mask}, monkeypatch)
    (entry,) = out.fields["pose"]["items"]
    x, y, _ = entry["kps"][0]
    assert x == pytest.approx(210.0, abs=2.0)          # the owner's rect, in capture px
    assert y == pytest.approx(210.0, abs=2.0)


# ── one entry per person, not per detection (the monitor's kinematics) ───────────────────────────────────

def test_two_detections_of_one_person_publish_one_entry(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    # two detections of the SAME person: the first one's joints land inside its own box (the crop
    # rect is identified), the second one's do not (the crop is found by containment). One entry
    # per person: the monitor would otherwise see the same track twice in one frame.
    good, good_mask = _pose_item(KPS, 200.0, 400.0, box=(100, 100, 200, 400))
    weak, weak_mask = _pose_item([[5.0, 5.0, 0.9]] * 17, 200.0, 400.0, box=(110, 120, 180, 360))
    frame = _frame([person, good, weak])
    out = _run(_detect(), frame, {id(good.raw): good_mask, id(weak.raw): weak_mask}, monkeypatch)
    items = out.fields["pose"]["items"]
    assert len(items) == 1 and items[0]["id"] == 7
    x, y, _ = items[0]["kps"][0]
    assert x == pytest.approx((100.0 + 10.0) * 2.0, abs=2.0)     # the detection that explains the box


# ── a person seen through too few joints is not observed (AC-F3.1's omission half) ───────────────────────

def test_a_person_seen_through_too_few_joints_is_omitted(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    for confident in (1, 3, 5):
        kps = [[10.0, 20.0, 0.9]] * confident + [[10.0, 20.0, 0.05]] * (17 - confident)
        item, mask = _pose_item(kps, 200.0, 400.0, box=(100, 100, 200, 400))
        out = _run(_detect(), frame=_frame([person, item]), masks={id(item.raw): mask}, monkeypatch=monkeypatch)
        assert "pose" not in out.fields, f"{confident} confident joints published an observation"


def test_a_person_seen_through_enough_joints_is_published(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    kps = [[10.0, 20.0, 0.9]] * 6 + [[10.0, 20.0, 0.05]] * 11
    item, mask = _pose_item(kps, 200.0, 400.0, box=(100, 100, 200, 400))
    out = _run(_detect(), frame=_frame([person, item]), masks={id(item.raw): mask}, monkeypatch=monkeypatch)
    (entry,) = out.fields["pose"]["items"]
    assert entry["id"] == 7


# ── the frames the network skipped say nothing about the people (the monitor's tracks survive) ───────────

def test_a_frame_the_network_skipped_publishes_no_pose_field(monkeypatch):
    person = _person(7, (100, 100, 200, 400), raw=object())
    out = _run(_detect(), _frame([person]), {}, monkeypatch)
    assert "pose" not in out.fields          # the frame says nothing, so it says nobody is gone


def test_an_empty_scene_publishes_an_empty_pose_field(monkeypatch):
    out = _run(_detect(), _frame([]), {}, monkeypatch)
    assert out.fields["pose"] == {"shape": [2000, 1000], "items": []}


# ── settings (AC-F6.1) ─────────────────────────────────────────────────────────────────────────────────────

def test_settings_validate_and_an_undeclared_yaml_key_refuses_startup():
    s = pose_mod.Settings()
    assert s.interval == 2 and s.keypoint_confidence_threshold == 0.30 and s.engine == ""
    assert s.min_keypoints == 6
    assert s.model == "yolo26x-pose"
    for bad in ({"interval": 0}, {"keypoint_confidence_threshold": 5.0}, {"keypoint_confidence_threshold": 0.0},
                {"min_keypoints": 0}, {"min_keypoints": 18}):
        with pytest.raises(ModuleConfigError):
            pose_mod.Settings(**bad)
    with pytest.raises(ModuleConfigError):
        load_modules({"pose": {"bogus_key": 1}}, _services())
    assert load_modules({"pose": {}}, _services()).names() == ["pose"]


def test_the_people_go_to_the_overlay_stream_so_a_fall_clip_shows_the_falling_one(monkeypatch):
    """The fall service names the person's track in its alert; the clip draws that person's box from here."""
    import json
    emitted = []
    mod = pose_mod.Pose(pose_mod.Settings(), _services(emit=lambda key, fields, maxlen=64: emitted.append((key, fields)) or True))
    item, mask = _pose_item(KPS, 640, 640)
    _run(mod, _frame([_person(4), item]), {id(item.raw): mask}, monkeypatch)
    (entry,) = [json.loads(f["shapes"]) for k, f in emitted if k == "iris:overlay:cam-1"]
    assert entry == [{"type": "box", "rect": [100.0, 100.0, 200.0, 400.0], "style": "alert", "id": 4}]
