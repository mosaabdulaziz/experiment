"""The preflight that tells a fresh clone what to `just data-pull` (tools/data/check_models.py, DATA.md)."""
import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("check_models_under_test", REPO / "tools" / "data" / "check_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_file_uri_video_source_is_found():
    mod = load()
    text = '  - {id: cam-01, uri: "file:///workspace/iris/products/el-mohandes/videos/1-D41-enter.mp4"}\n'
    assert [os.path.relpath(p, mod.ROOT) for p in mod.models_in(text)] == ["products/el-mohandes/videos/1-D41-enter.mp4"]


def test_an_rtsp_source_is_not_a_file():
    assert list(load().models_in('  - {id: cam-1, uri: "rtsp://go2rtc:8554/raw-cam-1"}\n')) == []


def test_the_message_names_the_group_that_owns_a_missing_clip():
    mod = load()
    clip = os.path.join(mod.ROOT, "products", "el-mohandes", "videos", "1-D41-enter.mp4")
    assert mod.groups_for([clip]) == ["el-mohandes-samples"]


def test_a_file_the_catalog_does_not_know_gets_no_group():
    mod = load()
    assert mod.groups_for([os.path.join(mod.ROOT, "nowhere", "x.mp4")]) == []


def test_a_preset_that_only_overrides_sources_still_sees_the_detector_its_base_names():
    mod = load()
    text = mod.chain_text(os.path.join(mod.ROOT, "configs", "presets", "ppe", "live-test.yaml"))
    named = {os.path.relpath(p, mod.ROOT) for p in mod.models_in(text)}
    assert "perception/models/DeepStream-Yolo/yolo11x.onnx" in named


def test_an_extends_cycle_does_not_loop(tmp_path):
    mod = load()
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n")
    assert "x: 1" in mod.chain_text(str(tmp_path / "a.yaml")) and "y: 2" in mod.chain_text(str(tmp_path / "a.yaml"))
