"""Autonomy, sidecar overlay, and out-of-scope guards (no cross-product imports, no face recognition)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from perception.engine.sources import load_sidecar_config, normalize_roles

PRODUCT = Path(__file__).resolve().parents[1]
OTHER_PRODUCTS = ("barber", "attendance", "multicam", "fall_detection")


def _product_py_files():
    return sorted(p for p in PRODUCT.rglob("*.py") if "node_modules" not in p.parts)


def test_no_python_file_imports_another_product():
    files = _product_py_files()
    assert files, "the product has no python files"
    offenders = []
    for path in files:
        if path.name == "test_autonomy_and_sidecar.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root == "products" or root in OTHER_PRODUCTS:
                    offenders.append(f"{path.relative_to(PRODUCT)}: {name}")
    assert offenders == []


def test_pure_library_imports_no_redis_fastapi_or_gstreamer():
    lib_files = sorted((PRODUCT / "concealment").rglob("*.py"))
    assert lib_files
    forbidden = {"redis", "fastapi", "pyds", "gi", "cv2", "sqlite3"}
    offenders = []
    for path in lib_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in forbidden:
                    offenders.append(f"{path.name}: {n}")
    assert offenders == []


def test_sidecar_overlay_declares_high_rate_pose_and_excludes_face_or_reid_modules():
    sidecar_path = PRODUCT / "sidecar.yaml"
    assert sidecar_path.is_file()
    cfg = load_sidecar_config(str(sidecar_path))
    assert cfg["sidecar_role"] == "concealment_detection"
    assert all(normalize_roles(s.get("roles")) == ["concealment_detection"] for s in cfg["sources"])
    assert cfg["sample_fps"] == 25
    assert cfg["recorder"] == "none"
    assert cfg["record_enabled"] is False
    assert "svr_archive_dir" not in cfg
    assert "svr_archive_dir_app" not in cfg
    assert "pose" in cfg["modules"]
    assert cfg["modules"]["pose"]["interval"] == 1
    # Out of scope per user story: facial recognition or shopper identification modules must never be enabled.
    for forbidden_mod in ("face_detect", "face_recognition", "customer_faces", "appearance"):
        assert forbidden_mod not in cfg["modules"]


def test_pose_sgie_config_matches_declared_interval_and_onnx():
    raw_sidecar = yaml.safe_load((PRODUCT / "sidecar.yaml").read_text(encoding="utf-8"))
    pose_cfg = PRODUCT / "config_sgie_pose.txt"
    assert pose_cfg.is_file()
    text = pose_cfg.read_text(encoding="utf-8")
    assert re.search(r"^secondary-reinfer-interval=1\s*$", text, re.M)
    assert int(raw_sidecar["modules"]["pose"]["interval"]) == 1
    onnx_match = re.search(r"^onnx-file=(.+)$", text, re.M)
    assert onnx_match is not None
    onnx_path = (pose_cfg.parent / onnx_match.group(1).strip()).resolve()
    assert onnx_path.name == "yolo26x-pose.onnx"


def test_compose_justfile_env_and_dockerfile_wiring():
    repo = PRODUCT.parents[1]
    compose = (repo / "docker-compose.yml").read_text(encoding="utf-8")
    assert "deepstream-sidecar-concealment:" in compose
    assert 'profiles: ["concealment"]' in compose
    assert "SIDECAR_CONCEALMENT_TELEMETRY_PORT:-9107" in compose

    just = (repo / "justfile").read_text(encoding="utf-8")
    for recipe in ("concealment-check:", "concealment-up:", "concealment-down:",
                   "concealment-log:", "concealment-sidecar-log:", "concealment-test"):
        assert recipe in just
    assert "--profile concealment down" in just

    env_ex = (repo / ".env.example").read_text(encoding="utf-8")
    for var in ("CONCEALMENT_BACKEND_PORT=8005", "SIDECAR_CONCEALMENT_TELEMETRY_PORT=9107",
                "CONCEALMENT_SIDECAR_CONFIG=", "CONCEALMENT_CONFIG="):
        assert var in env_ex

    dockerfile = (repo / "Dockerfile").read_text(encoding="utf-8")
    assert "frontend/src/products/concealment_detection/package.json" in dockerfile
