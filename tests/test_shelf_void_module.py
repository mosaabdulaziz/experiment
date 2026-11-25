"""modules.shelf_void: settings, the detector it asks the engine for (full frame, after the tracker), the alerts it
raises from the heat map, what it draws and publishes, and the opt-in preset."""
import configparser
import pathlib
import types

import numpy as np
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, build_settings
from perception.modules.shelf_void import BOX_VOID, MODULE, RAW_KIND, Settings
from perception.modules.shelf_void.adapter import SHELF_GIE

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "perception/models/DeepStream-Yolo-Shelf/config_infer_shelf.txt"
CONFIG_640 = CONFIG.parent / "config_infer_shelf_640.txt"
SHELF_640 = "/workspace/iris/perception/models/DeepStream-Yolo-Shelf/config_infer_shelf_640.txt"
UNTRACKED = 0xFFFFFFFFFFFFFFFF
MUX = (1080, 720)
FPS = 30.0                                   # the default eval_fps: feeding at this rate updates the heat map every frame


def _model_dir(tmp_path, with_onnx=True):
    """A copy of the model folder's config and labels; the ONNX (not in git) as an empty stand-in, or absent."""
    for name in ("config_infer_shelf.txt", "labels_shelf.txt"):
        (tmp_path / name).write_text((CONFIG.parent / name).read_text())
    if with_onnx:
        (tmp_path / "detect_product_empty_space.onnx").write_bytes(b"")
    return tmp_path / "config_infer_shelf.txt"


def _module(batch=1, **overrides):
    emitted, clips = [], []
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: emitted.append((key, fields)),
        request_clip=lambda sid, now, age=None: clips.append((sid, round(now, 3))), stream_ids=["cam-1"],
        timer=lambda *a: None, telemetry=None, batch_size=batch,
        states={"cam-1": types.SimpleNamespace(module_state={})})
    mod = MODULE(build_settings(Settings, overrides, "shelf_void"), services)
    mod.on_built(None)
    return mod, emitted, clips


def _void(left=100, top=100, width=80, height=40, conf=0.05, label="empty", gie=SHELF_GIE):
    return ObjectView(UNTRACKED, label, conf, conf, left, top, width, height, gie_name=gie)


def _feed(mod, frames, start=0.0):
    """frames: a list of object lists, one per frame at FPS from `start`; pts = now, like a file from 0."""
    for i, objs in enumerate(frames):
        t = start + i / FPS
        mod.on_frame(FrameView("cam-1", None, 0, i, t, t, mux_size=MUX, _objects=objs), FrameOutput())
    return start + len(frames) / FPS


def _alerts(emitted):
    return [f for key, f in emitted if key == "iris:alert:cam-1"]


# --- settings and the detector ---

def test_defaults_are_the_standalone_tuning():
    s = Settings()
    assert s.config == "/workspace/iris/perception/models/DeepStream-Yolo-Shelf/config_infer_shelf.txt"
    assert (s.interval, s.label, s.eval_fps) == (0, "empty", 30.0)
    assert (s.min_area, s.decay, s.show_thresh, s.close_kernel, s.roi) == (400, 0.85, 0.5, [40, 9], {})
    assert (s.confirm_s, s.clear_s, s.match_iou) == (2.0, 3.0, 0.3)
    assert (s.stability, s.tracker) == ("heatmap", {})


@pytest.mark.parametrize("bad", [{"interval": -1}, {"eval_fps": 0}, {"decay": 1.0}, {"show_thresh": 0},
                                 {"close_kernel": [40]}, {"close_kernel": [40, 0]}, {"min_area": -1},
                                 {"match_iou": 0}, {"clear_s": -1}, {"roi": [[0, 0], [1, 0], [1, 1]]},
                                 {"roi": {"cam-1": [[0, 0], [1, 1]]}}, {"classes": ["empty"]},
                                 {"stability": "kalman"}, {"tracker": {"confirm": 2}}, {"tracker": {"conf": 1.5}},
                                 {"tracker": {"min_iou": 0}}, {"tracker": {"fps": -1}}, {"tracker": []}])
def test_invalid_settings_refuse_startup(bad):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, bad, "shelf_void")


def test_the_detector_runs_on_the_full_frame_after_the_tracker_with_the_mux_batch(tmp_path):
    config = _model_dir(tmp_path)
    mod, _, _ = _module(batch=4, config=str(config))
    (spec,) = mod.gies()
    # "secondary" places it after nvtracker, so its low-confidence boxes are never deleted by the tracker
    assert (spec.name, spec.role, spec.interval, spec.batch_from_mux) == ("pgie_shelf", "secondary", 0, True)
    assert "_b4_gpu0_" in spec.engine_path and spec.engine_path.startswith(str(tmp_path))


def test_a_missing_model_refuses_startup_and_says_where_to_get_it(tmp_path):
    config = _model_dir(tmp_path, with_onnx=False)
    mod, _, _ = _module(config=str(config))
    with pytest.raises(ModuleConfigError) as err:
        mod.gies()
    msg = str(err.value)
    assert str(tmp_path / "detect_product_empty_space.onnx") in msg and "is missing" in msg
    assert "just data-pull models-shelf" in msg and "perception/models/DeepStream-Yolo-Shelf/README.md" in msg


def test_a_crop_config_refuses_startup(tmp_path):
    config = _model_dir(tmp_path)
    config.write_text(config.read_text().replace("process-mode=1", "process-mode=2"))
    mod, _, _ = _module(config=str(config))
    with pytest.raises(ModuleConfigError, match="full frame"):
        mod.gies()


def test_its_config_keeps_only_the_empty_class_at_the_standalone_thresholds_and_1280_input():
    cp = configparser.ConfigParser()
    cp.read(CONFIG)
    p, c = cp["property"], cp["class-attrs-all"]
    assert (p["gie-unique-id"], p["process-mode"], p["interval"]) == ("5", "1", "0")
    assert (p["num-detected-classes"], p["filter-out-class-ids"]) == ("2", "1")
    assert (c["pre-cluster-threshold"], c["nms-iou-threshold"]) == ("0.02", "0.3")
    labels = (CONFIG.parent / p["labelfile-path"]).read_text().split()
    assert labels == ["empty", "product"]
    others = [f for f in REPO.glob("perception/models/*/config_infer*.txt") if f not in (CONFIG, CONFIG_640)]   # the two shelf configs never run together
    used = set()
    for f in others:
        o = configparser.ConfigParser(strict=False)
        o.read(f)
        if o.has_option("property", "gie-unique-id"):
            used.add(o.get("property", "gie-unique-id"))
    assert "5" not in used


def test_the_model_folder_readme_carries_provenance_licence_and_how_to_export_at_1280():
    text = (CONFIG.parent / "README.md").read_text()
    for needed in ("no licence", "AGPL-3.0", "pickle", "6da67fa200359842b34e1e17548cbf5de27eac28",
                   "d06fd0ee0a27c38b2c124046b20df5313af7b247286fd7f70e9b208f9dbd0e02",
                   "export_raw_anchor_onnx.py", "--imgsz 1280"):
        assert needed.lower() in text.lower(), needed


# --- alerts ---

def test_a_persistent_void_raises_one_shelf_empty_alert_with_a_clip():
    mod, emitted, clips = _module()
    _feed(mod, [[_void()] for _ in range(90)])                 # 3 s of the same empty place
    (a,) = _alerts(emitted)
    assert (a["kind"], a["severity"], a["category"]) == ("shelf_empty", "warning", "operations")
    assert a["message"].startswith("Empty shelf space on cam-1 at (")
    assert a["bbox"].count(",") == 3 and a["alert_id"] and float(a["score"]) > 0.5
    assert clips == [("cam-1", float(a["pts"]))]


def test_flicker_never_alerts():
    mod, emitted, _ = _module()
    _feed(mod, [[_void()] if i % 4 == 0 else [] for i in range(120)])
    assert _alerts(emitted) == []


def test_a_restocked_place_gives_shelf_restocked_with_the_same_alert_id():
    mod, emitted, _ = _module()
    t = _feed(mod, [[_void()] for _ in range(90)])             # empty 3 s
    _feed(mod, [[] for _ in range(150)], start=t)              # restocked 5 s
    raised, cleared = _alerts(emitted)
    assert (cleared["kind"], cleared["severity"], cleared["category"]) == ("shelf_restocked", "info", "operations")
    assert cleared["alert_id"] == raised["alert_id"] and cleared["message"].startswith("Shelf restocked on cam-1")
    assert float(cleared["empty_for_s"]) > 2


def test_other_gies_and_labels_never_warm_the_heat_map():
    mod, emitted, _ = _module()
    _feed(mod, [[_void(gie="pgie"), _void(label="product")] for _ in range(90)])
    assert _alerts(emitted) == []


def test_frames_faster_than_eval_fps_are_skipped():
    mod, emitted, _ = _module(eval_fps=10)
    _feed(mod, [[_void()] for _ in range(45)])                 # 1.5 s at 30 fps = 15 updates at 10/s
    st = mod.services.state("shelf_void", "cam-1")
    expected = 1 - 0.85 ** 15
    assert st["detector"].heat.max() == pytest.approx(expected, rel=1e-4)


def test_roi_is_per_camera_in_mux_pixels():
    mod, emitted, _ = _module(roi={"cam-1": [[0, 0], [90, 0], [90, 720], [0, 720]]})
    _feed(mod, [[_void()] for _ in range(90)])                 # the void is at x 100-180, outside the ROI
    assert _alerts(emitted) == []


# --- what it draws and publishes ---

def test_raw_boxes_are_never_drawn_nor_published_and_open_voids_are_drawn_and_published():
    mod, _, _ = _module()
    _feed(mod, [[_void()] for _ in range(90)])
    raw = _void()
    view = FrameView("cam-1", None, 0, 90, 3.0, 3.0, mux_size=MUX, need_meta=True, _objects=[raw])
    mod.transform(view)
    assert (raw.kind, raw.draw, raw.publish) == (RAW_KIND, False, False)
    out = FrameOutput()
    mod.analyze(view, out)
    (field,) = out.fields["shelf_void"]
    assert len(field["bbox"]) == 4 and field["alert_id"]
    img = np.zeros((720, 1080, 3), np.uint8)
    (draw,) = out.draws
    draw(img, 1.0, 1.0)
    assert (img == np.array(BOX_VOID, np.uint8)).all(axis=2).any()


def test_nothing_is_drawn_without_an_open_alert():
    mod, _, _ = _module()
    out = FrameOutput()
    mod.analyze(FrameView("cam-1", None, 0, 0, 0.0, 0.0, mux_size=MUX, need_meta=True, _objects=[]), out)
    assert out.draws == [] and out.fields == {}


def test_the_640_config_is_the_1280_one_with_the_640_onnx_and_the_demos_nms():
    def read(path):
        cp = configparser.ConfigParser()
        cp.read(path)
        return {sec: dict(cp[sec]) for sec in cp.sections()}
    a, b = read(CONFIG), read(CONFIG_640)
    assert b["property"].pop("onnx-file") == "detect_product_empty_space_640.onnx"
    assert b["class-attrs-all"].pop("nms-iou-threshold") == "0.7"
    a["property"].pop("onnx-file"), a["class-attrs-all"].pop("nms-iou-threshold")
    assert a == b
    assert (CONFIG.parent / "detect_product_empty_space_640.onnx.dvc").is_file()


# --- the opt-in preset: the shared pipeline plus shelf_void, and the shared file itself unchanged ---

def test_the_shelf_preset_switches_on_shelf_void_and_nothing_else_over_the_shared_pipeline():
    from perception.engine.modules import load_modules
    from perception.engine.sources import load_sidecar_config
    shared = load_sidecar_config(str(REPO / "perception/pipeline/sidecar.yaml"))
    preset = load_sidecar_config(str(REPO / "configs/presets/shelf/sidecar.yaml"))
    assert shared["modules"] == {}
    assert preset["modules"] == {"shelf_void": {"config": SHELF_640, "stability": "tracker",
                                                "tracker": {"conf": 0.08}, "min_area": 800, "match_iou": 0.2}}
    rest = lambda cfg: {k: v for k, v in cfg.items() if k not in ("modules", "extends")}   # noqa: E731
    assert rest(preset) == rest(shared)
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None, request_clip=lambda *a, **k: None,
        stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None, batch_size=1)
    assert load_modules(preset["modules"], services).names() == ["shelf_void"]


def test_the_live_demo_preset_names_its_camera_in_go2rtc():
    from perception.engine.sources import load_sidecar_config
    live = load_sidecar_config(str(REPO / "configs/presets/shelf/live-demo.yaml"))
    assert live["modules"] == {"shelf_void": {"config": SHELF_640, "stability": "tracker",
                                              "tracker": {"conf": 0.08}, "min_area": 800, "match_iou": 0.2}}
    (src,) = live["sources"]
    assert src["id"] == "cam-shelf-demo" and src["uri"] == "rtsp://go2rtc:8554/raw-cam-shelf-demo"
    assert "raw-cam-shelf-demo:" in (REPO / "configs/presets/shelf/go2rtc-live.yaml").read_text()


def test_the_test_camera_preset_has_the_demos_two_regions_and_its_go2rtc_stream():
    from perception.engine.sources import load_sidecar_config
    live = load_sidecar_config(str(REPO / "configs/presets/shelf/live-test.yaml"))
    (src,) = live["sources"]
    assert src["id"] == "cam-shelf-test" and src["uri"] == "rtsp://go2rtc:8554/raw-cam-shelf-test"
    roi = live["modules"]["shelf_void"]["roi"]["cam-shelf-test"]
    assert roi["size"] == [1920, 1080]
    assert roi["regions"] == [[[1369, 328], [1684, 352], [1581, 931], [1294, 933], [1357, 327]],     # shelf-demo
                              [[225, 346], [516, 358], [594, 835], [544, 928], [276, 880], [150, 451], [222, 343]]]
    s = build_settings(Settings, live["modules"]["shelf_void"], "shelf_void")
    assert [n for n, _ in s.regions("cam-shelf-test", MUX)] == ["R1", "R2"]
    assert "raw-cam-shelf-test:" in (REPO / "configs/presets/shelf/go2rtc-live.yaml").read_text()
