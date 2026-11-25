"""The fall product's sidecar overlay: its own cameras, the pose module, and nothing else's files.

ACs: AC-F1.1 (config half), AC-F29.1 (existence half), AC-F32.1.
Design: FALL_DETECTION_DESIGN.md §3, §9, §13.1; plan decision 6.
"""
from pathlib import Path

import yaml

from perception.engine.sources import load_sidecar_config, normalize_roles

REPO = Path(__file__).resolve().parents[1]
FALL = REPO / "products" / "fall_detection" / "sidecar.yaml"
PRESET = REPO / "configs" / "presets" / "fall_3cam" / "sidecar.yaml"
GOLDEN = REPO / "tests" / "configs" / "golden-fall_detection.yaml"


def _fall():
    return load_sidecar_config(str(FALL))


def test_the_overlay_declares_the_pose_module_with_its_sgie_interval():
    cfg = _fall()
    assert "pose" in cfg["modules"]
    assert cfg["modules"]["pose"]["interval"] == 2


def test_the_overlay_tags_its_sources_with_its_own_role():
    cfg = _fall()
    assert cfg["sidecar_role"] == "fall_detection"
    assert all(normalize_roles(s.get("roles")) == ["fall_detection"] for s in cfg["sources"])


def test_the_pilot_camera_set_is_exactly_cam_1():
    assert [s["id"] for s in _fall()["sources"]] == ["cam-1"]


def test_the_three_camera_preset_replaces_only_the_sources():
    cfg = load_sidecar_config(str(PRESET))
    assert [s["id"] for s in cfg["sources"]] == ["cam-1", "cam-2", "cam-3"]
    assert "pose" in cfg["modules"]
    raw = yaml.safe_load(PRESET.read_text())
    assert raw.get("extends", "").endswith("products/fall_detection/sidecar.yaml")


def test_the_golden_config_exists_and_extends_the_fall_overlay():
    assert GOLDEN.is_file()
    raw = yaml.safe_load(GOLDEN.read_text())
    assert raw.get("extends", "").endswith("products/fall_detection/sidecar.yaml")
    assert raw.get("file_loop") is False
    src, = raw["sources"]
    assert str(src.get("uri", src.get("url", ""))).startswith("file://")
