"""modules.ppe: settings and the detector it asks the engine for (on person crops, placed after the tracker)."""
import configparser
import dataclasses
import pathlib
import types

import pytest

from perception.engine.modules import EngineServices, ModuleConfigError, build_settings
from perception.modules.ppe import MODULE, Settings
from perception.modules.ppe.compliance.config import ComplianceConfig, TemporalConfig

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt"


def _module(batch=4, **overrides):
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None, request_clip=lambda *a, **k: None,
        stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None, batch_size=batch,
        states={"cam-1": types.SimpleNamespace(module_state={})})
    return MODULE(build_settings(Settings, {"config": str(CONFIG), **overrides}, "ppe"), services)


def _detector_config():
    cp = configparser.ConfigParser()
    cp.read(CONFIG)
    return cp["property"]


def test_defaults():
    s = Settings()
    assert (s.interval, s.config) == (0, "/workspace/iris/perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt")


@pytest.mark.parametrize("bad", [{"interval": -1}, {"classes": ["Hardhat"]}])
def test_invalid_settings_refuse_startup(bad):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, bad, "ppe")


def _model_dir(tmp_path, with_onnx=True):
    """A copy of the model folder's config and labels; the ONNX (a DVC pointer in the repo) as an empty stand-in, or absent."""
    for name in ("config_infer_ppe.txt", "labels_ppe.txt"):
        (tmp_path / name).write_text((CONFIG.parent / name).read_text())
    if with_onnx:
        (tmp_path / "voxdroid_yolov8s_css.onnx").write_bytes(b"")
    return tmp_path / "config_infer_ppe.txt"


def test_the_detector_sits_after_the_tracker_and_keeps_its_crop_batch_size(tmp_path):
    config = _model_dir(tmp_path)
    (spec,) = _module(batch=4, config=str(config)).gies()
    # "secondary" places it after nvtracker, so its boxes are never tracked (and never deleted by the tracker)
    assert (spec.name, spec.role, spec.interval, spec.batch_from_mux) == ("pgie_ppe", "secondary", 0, False)
    assert spec.config_path == str(config)
    # process-mode=2: a variable number of crops per batch, so the config's batch size (16), not the mux's (4)
    assert "_b16_gpu0_" in spec.engine_path and spec.engine_path.startswith(str(tmp_path))


def test_a_missing_model_refuses_startup_and_says_where_to_get_it(tmp_path):
    config = _model_dir(tmp_path, with_onnx=False)
    with pytest.raises(ModuleConfigError) as err:
        _module(config=str(config)).gies()
    msg = str(err.value)
    assert str(tmp_path / "voxdroid_yolov8s_css.onnx") in msg and "is missing" in msg
    assert "just data-pull models-ppe" in msg and "perception/models/DeepStream-Yolo-PPE/README.md" in msg


def test_a_missing_detector_config_refuses_startup(tmp_path):
    with pytest.raises(ModuleConfigError, match="detector config .* is missing"):
        _module(config=str(tmp_path / "nope.txt")).gies()


def test_the_model_folder_readme_says_how_to_rebuild_the_model():
    readme = (CONFIG.parent / "README.md").read_text()
    for needle in ("Getting the model", "470cc1d2f39774ade966488719d20635da56431123a8b189ec87fec041f0bc47",
                   "cdd4f0afe9081144dc972eb396594f995386017807aef2ce9dce4a0b41c44656",
                   "export_raw_anchor_onnx.py"):
        assert needle in readme


def test_its_config_runs_on_the_pgie_persons_crops_with_its_own_id_and_the_four_ppe_classes():
    p = _detector_config()
    # each tracked person of the pgie (gie 1, COCO class 0), crops scaled bilinear (the default, nearest, gave the
    # model crops it could not read), at the evaluated thresholds
    assert (p["process-mode"], p["operate-on-gie-id"], p["operate-on-class-ids"]) == ("2", "1", "0")
    assert (p["scaling-filter"], p["batch-size"]) == ("1", "16")
    assert p["gie-unique-id"] == "4"
    labels = (CONFIG.parent / p["labelfile-path"]).read_text().split("\n")
    dropped = {labels[int(i)] for i in p["filter-out-class-ids"].split(";")}
    assert dropped == {"Mask", "NO-Mask", "Person", "Safety Cone", "machinery", "vehicle"}
    assert set(labels[:10]) - dropped == {"Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"}


# --- the rule settings: ports of ppe-compliance's four config tests, against Settings instead of load_config() ---

def test_defaults_are_the_evaluated_rules_except_the_documented_iris_changes():
    # ppe-compliance test_default_yaml_matches_code_defaults: there the yaml had to equal the code defaults; here the
    # module's defaults must equal the vendored ones except for what IRIS changes on purpose.
    s = Settings()
    got, ref = s.compliance_config(), ComplianceConfig()
    assert got.person_classes == ["person"] and ref.person_classes == ["Person"]     # the pgie's label
    assert (got.edge_margin_px, ref.edge_margin_px) == (2, 3)                          # mux pixels
    assert [it.min_person_height_px for it in got.items] == [32, 32]                   # mux pixels (48 evaluated)
    restore = dataclasses.replace(got, person_classes=ref.person_classes, edge_margin_px=ref.edge_margin_px,
                                  items=[dataclasses.replace(it, min_person_height_px=48) for it in got.items])
    assert restore == ref
    # an episode closes after 2 s of PRESENT, not on the first PRESENT frame (test2.mp4)
    assert (s.temporal_config().close_after_present_s, TemporalConfig().close_after_present_s) == (2.0, 0.0)
    assert dataclasses.replace(s.temporal_config(), close_after_present_s=0.0) == TemporalConfig()
    assert s.eval_fps == 20.0


def test_the_new_settings_are_read_and_bad_values_refuse_startup():
    s = build_settings(Settings, {"temporal": {"close_after_present_s": 0}, "links": {"max_gap_s": 2},
                                  "alert_hold_s": 0.5}, "ppe")
    assert (s.temporal_config().close_after_present_s, s.link_config().max_gap_s, s.link_config().max_dist,
            s.alert_hold_s) == (0, 2, 1.2, 0.5)
    for bad in ({"alert_hold_s": -1}, {"links": {"max_dist": -1}}, {"links": {"speed": 1}},
                {"temporal": {"close_after_present_s": -1}}):
        with pytest.raises(ModuleConfigError):
            build_settings(Settings, bad, "ppe")


def test_partial_override():
    # ppe-compliance test_config_partial_override
    s = build_settings(Settings, {"temporal": {"window": 3}, "required_items": ["vest"],
                                  "items": {"vest": {"absent_conf": 0.5}}}, "ppe")
    t, c = s.temporal_config(), s.compliance_config()
    assert t.window == 3 and t.min_absent_frames == 6
    assert c.required_items == ["vest"]
    assert c.item("helmet").present_conf == 0.40
    assert (c.item("vest").absent_conf, c.item("vest").present_conf) == (0.5, 0.40)


def test_an_item_can_override_the_minimum_person_height():
    c = build_settings(Settings, {"items": {"helmet": {"min_person_height_px": 40}}}, "ppe").compliance_config()
    assert (c.item("helmet").min_person_height_px, c.item("vest").min_person_height_px) == (40, 32)


@pytest.mark.parametrize("raw", [{"min_persn_conf": 0.5}, {"temporal": {"windw": 3}},
                                 {"items": {"vest": {"absnt_conf": 0.5}}}, {"items": {"vest": 0.5}}])
def test_unknown_keys_refuse_startup(raw):
    # ppe-compliance test_config_rejects_unknown_keys, at every level: setting, temporal field, item field
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, raw, "ppe")


@pytest.mark.parametrize("raw", [{"required_items": ["gloves"]}, {"required_items": []}, {"items": {"gloves": {}}}])
def test_unknown_or_no_required_item_refuses_startup(raw):
    # ppe-compliance test_config_rejects_unknown_required_item (a KeyError there; a startup refusal here)
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, raw, "ppe")


@pytest.mark.parametrize("raw", [{"eval_fps": 0}, {"eval_fps": -5}, {"min_person_height_px": -1},
                                 {"edge_margin_px": -1}])
def test_invalid_rule_settings_refuse_startup(raw):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, raw, "ppe")


# --- the opt-in preset: the shared pipeline plus ppe, and the shared file itself unchanged ---

def test_the_ppe_preset_switches_on_ppe_and_nothing_else_over_the_shared_pipeline():
    from perception.engine.modules import load_modules
    from perception.engine.sources import load_sidecar_config
    repo = pathlib.Path(__file__).resolve().parents[1]
    shared = load_sidecar_config(str(repo / "perception/pipeline/sidecar.yaml"))
    preset = load_sidecar_config(str(repo / "configs/presets/ppe/sidecar.yaml"))
    assert shared["modules"] == {}
    assert preset["modules"] == {"ppe": {}}
    rest = lambda cfg: {k: v for k, v in cfg.items() if k not in ("modules", "extends")}   # noqa: E731
    assert rest(preset) == rest(shared)
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None, request_clip=lambda *a, **k: None,
        stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None, batch_size=1)
    assert load_modules(preset["modules"], services).names() == ["ppe"]


def test_a_full_frame_config_runs_on_every_frame_of_the_mux_batch(tmp_path):
    """process-mode=1 -> the full frame, still after the tracker; the batch size and engine are the mux's."""
    config = _model_dir(tmp_path)
    text = config.read_text().replace("process-mode=2\noperate-on-gie-id=1\noperate-on-class-ids=0", "process-mode=1")
    config.write_text(text)
    assert "process-mode=1" in text and "operate-on" not in text
    (spec,) = _module(batch=4, config=str(config)).gies()
    assert (spec.role, spec.batch_from_mux) == ("secondary", True)
    assert "_b4_gpu0_" in spec.engine_path
