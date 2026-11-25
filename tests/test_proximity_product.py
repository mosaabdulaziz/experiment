"""products/proximity: the profile runs the seg detector, the proximity module and the tuned tracker."""
import json
import re
from pathlib import Path

import yaml

from perception.engine.modules import build_settings
from perception.engine.sources import load_sidecar_config
from perception.modules.proximity import Settings
from perception.modules.proximity.scale import PersonScale

REPO = Path(__file__).resolve().parents[1]


def _cfg():
    return load_sidecar_config(str(REPO / "products" / "proximity" / "sidecar.yaml"))


def _local(path: str) -> Path:
    return REPO / path.replace("/workspace/iris/", "")


def test_the_profile_runs_only_proximity_on_the_seg_detector():
    cfg = _cfg()
    assert list(cfg["modules"]) == ["proximity"]
    text = _local(cfg["pgie_config"]).read_text()
    for line in ("onnx-file=yolo26l-seg-736x1280.onnx", "output-instance-mask=1", "network-type=3",
                 "parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg"):
        assert line in text


def test_every_class_the_module_reads_survives_relevant_classes():
    cfg = _cfg()
    s = build_settings(Settings, cfg["modules"]["proximity"], "proximity")
    assert set(s.person_classes) | set(s.vehicle_classes) <= set(cfg["relevant_classes"])


def _body(text):
    return [line.split("#")[0].rstrip() for line in text.splitlines() if line.split("#")[0].strip()]


def test_the_tracker_is_the_attendance_nvdcf_with_class_matching_off_and_the_accuracy_visual_tracker():
    cfg = _cfg()
    prox = _body((REPO / cfg["tracker_config"]).read_text())
    base = _body((REPO / "perception/pipeline/config_tracker_nvdcf.yml").read_text())
    assert prox[0] == "%YAML:1.0" and len(prox) == len(base)
    assert [(a, b) for a, b in zip(base, prox) if a != b] == [
        ("  checkClassMatch: 1", "  checkClassMatch: 0"),
        ("  visualTrackerType: 1", "  visualTrackerType: 2"),
        ("  useHog: 0", "  useHog: 1"),
        ("  featureImgSizeLevel: 2", "  featureImgSizeLevel: 3"),
    ]


def test_the_models_are_where_just_proximity_models_writes_them():
    cfg = _cfg()
    depth = cfg["modules"]["proximity"]["depth_model"]
    assert depth == "/workspace/iris/perception/models/depth/yolo26m-depth-736x1280.onnx"
    recipe = (REPO / "justfile").read_text()
    assert "proximity-models:" in recipe and "perception/models/depth/yolo26m-depth-736x1280.onnx" in recipe
    assert "perception/models/DeepStream-Yolo-Seg/yolo26l-seg-736x1280.onnx" in recipe


def test_the_compose_service_runs_the_profile():
    svc = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["deepstream-sidecar-proximity"]
    assert svc["profiles"] == ["proximity"]
    assert "products/proximity/sidecar.yaml" in " ".join(svc["command"])
    assert any(e.startswith("LD_PRELOAD=") for e in svc["environment"])     # the seg engine build needs it


def test_the_image_installs_the_module():
    text = (REPO / "perception/pipeline/Dockerfile").read_text()
    assert "proximity" in re.search(r'ARG BASE_MODULES="([^"]*)"', text).group(1).split()


def test_the_scale_seed_gives_every_camera_a_usable_map():
    cfg = _cfg()
    seed = json.loads((REPO / cfg["modules"]["proximity"]["scale_seed"]).read_text())
    for source in cfg["sources"]:
        scale = PersonScale()
        scale.restore(seed[source["id"]])                  # raises on a fit that is not one
        assert scale.ready and 20 < scale.height(400) < 300
