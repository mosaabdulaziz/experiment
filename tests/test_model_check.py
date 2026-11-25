"""perception/engine/model_check.py: what the sidecar is about to load (engine GIE/tracker + Module.model_files), checked
against a model manifest at startup. Uses the real product yamls; no model file needs to be present."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("numpy")
pytest.importorskip("cv2")

from perception.engine import model_manifest as mm  # noqa: E402
from perception.engine.model_check import REPO_ROOT, engine_model_refs, verify_models  # noqa: E402
from perception.engine.sources import apply_role_overrides, load_sidecar_config  # noqa: E402


def _files(cfg, every_variant=False):
    files, problems = mm.expand(engine_model_refs(cfg, every_variant), REPO_ROOT)
    assert problems == []
    return files


def test_barber_bekky_loads_and_its_staff_switch():
    cfg = load_sidecar_config(str(REPO_ROOT / "products/barber/sidecar.yaml"))
    now = _files(cfg)
    assert now["perception/models/staff/staff_dinov2_s.onnx"] == "modules.appearance.staff_classifier"
    assert "perception/models/staff/vision_model.onnx" not in now                 # staff_model: classifier
    assert "models/tracker/resnet50_market1501_aicity156.onnx" not in now          # tracker.yml reidType: 0
    assert now["perception/models/DeepStream-Yolo-Seg/labels.txt"] == "pgie_config -> labelfile-path"
    pack = "perception/models/insightface/models/buffalo_l"         # the repo's DVC copy (perception/insightface_pack)
    if (REPO_ROOT / pack / "w600k_r50.onnx").is_file():             # pulled: every .onnx of the pack, by name
        assert now[f"{pack}/w600k_r50.onnx"] == now[f"{pack}/det_10g.onnx"] == "modules.face_recognition.model"
        assert len([k for k in now if k.startswith(pack + "/")]) == 5
    else:                                                           # not pulled (CI): the pack as a glob
        assert now[f"{pack}/*.onnx"] == "modules.face_recognition.model"
    assert now["perception/models/DeepStream-Yolo-Face/yolov8n-face.onnx"] == "modules.face_detect.config -> onnx-file"
    assert not [k for k in now if ".insightface" in k]              # never the home-directory pack
    every = _files(cfg, every_variant=True)
    assert every["perception/models/staff/vision_model.onnx"] == "modules.appearance.clip_model"
    assert every["perception/models/staff/prompts.json"] == "modules.appearance.prompts"
    assert every["models/tracker/resnet50_market1501_aicity156.onnx"] == "tracker_config -> ReID.onnxFile"
    assert set(now) < set(every)


def test_palm_hills_loads_its_own_staff_classifier():
    for name in ("sidecar.palm-hills.yaml", "sidecar.palm-hills-live.yaml", "sidecar.palm-hills-16oct.yaml"):
        files = _files(load_sidecar_config(str(REPO_ROOT / "products/barber" / name)))
        assert "perception/models/staff/staff_dinov2_s_palm.onnx" in files
        assert "perception/models/staff/staff_dinov2_s.onnx" not in files


def test_a_role_without_the_pgie_is_not_asked_for_it():
    cfg = load_sidecar_config(str(REPO_ROOT / "products/attendance/sidecar.yaml"))
    assert "perception/models/DeepStream-Yolo/yolo11x.onnx" in _files(cfg)
    fire_only = apply_role_overrides(cfg, "fire")
    assert "perception/models/DeepStream-Yolo/yolo11x.onnx" not in _files(fire_only)
    assert "perception/models/DeepStream-Yolo-Fire/fire_smoke_box.onnx" in _files(fire_only)


def test_attendance_loads_face_det_lite_and_auraface():
    cfg = load_sidecar_config(str(REPO_ROOT / "products/attendance/sidecar.yaml"))
    now = _files(cfg)
    assert now["perception/models/DeepStream-Yolo-Face/face_det_lite.onnx"] == "modules.face_detect.config -> onnx-file"
    pack = "perception/models/insightface/models/auraface"
    if (REPO_ROOT / pack / "glintr100.onnx").is_file():
        assert now[f"{pack}/glintr100.onnx"] == "modules.face_recognition.model"
        assert len([k for k in now if k.startswith(pack + "/")]) == 1
    else:
        assert now[f"{pack}/*.onnx"] == "modules.face_recognition.model"


def test_multicam_reid_weights_and_ocr_follow_their_settings():
    cfg = load_sidecar_config(str(REPO_ROOT / "products/multicam/sidecar.yaml"))
    files = _files(cfg)
    assert files["mtmc/models/reid/resnet50-11ad3fa6.pth"] == "modules.mtmc.config.reid_enabled"
    assert "products/el-mohandes/models/v5_arabic_mobile-fine-tuned/plate_dict.txt" in files
    cfg["modules"]["mtmc"]["config"]["reid_enabled"] = False
    assert "mtmc/models/reid/resnet50-11ad3fa6.pth" not in _files(cfg)


def test_startup_refuses_a_model_missing_from_the_manifest(tmp_path):
    cfg = load_sidecar_config(str(REPO_ROOT / "products/fall_detection/sidecar.yaml"))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema": 1, "product": "fall_detection", "models": []}))
    cfg["model_manifest"] = str(manifest)
    with pytest.raises(mm.ManifestError) as e:
        verify_models(cfg, what="test", env={})
    msg = str(e.value)
    assert "pgie_config -> onnx-file loads perception/models/DeepStream-Yolo/yolo11x.onnx, which this manifest" in msg
    assert "modules.pose.config -> onnx-file loads perception/models/DeepStream-Yolo-Pose/yolo26x-pose.onnx" in msg


def test_startup_without_a_manifest_warns_or_refuses():
    cfg = load_sidecar_config(str(REPO_ROOT / "products/fall_detection/sidecar.yaml"))
    cfg.pop("model_manifest", None)
    assert verify_models(cfg, what="test", env={}).manifest is None
    with pytest.raises(mm.ManifestError, match="MODEL_MANIFEST_REQUIRED is set"):
        verify_models(cfg, what="test", env={"MODEL_MANIFEST_REQUIRED": "1"})
