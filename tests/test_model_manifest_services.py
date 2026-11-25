"""The product services' own model checks: barber's kids car model and fall detection's checkpoints, against the model
manifest their yaml names. Their helpers (products/<p>/model_files.py) are loaded by path, as the generator and the
completeness test load them; no torch, no fastapi."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def _load(product: str):
    spec = importlib.util.spec_from_file_location(f"{product}_model_files", ROOT / "products" / product / "model_files.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


barber = _load("barber")
fall = _load("fall_detection")


def _manifest(path: Path, *entries):
    path.write_text(json.dumps({"schema": 1, "product": "p", "models": [
        {"path": p, "registered_model": n, "version": "1", "alias": "production-p", "run_id": "r" * 32,
         "sha256": s, "size_bytes": 1} for p, n, s in entries]}))
    return str(path)


def test_barber_service_loads_the_kids_car_model_its_yaml_names():
    assert barber.service_model_files({}) == [("perception/models/kids_car/kids_car.onnx", "barber service kids_car_model")]
    assert barber.service_model_files({"kids_car_model": "x/k.onnx"})[0][0] == "x/k.onnx"


@pytest.fixture
def repo(tmp_path):
    """A repo with the verifier where the services look for it, and a kids car model."""
    r = tmp_path / "repo"
    (r / "perception" / "engine").mkdir(parents=True)
    shutil.copy(ROOT / "perception" / "engine" / "model_manifest.py", r / "perception" / "engine")
    (r / "perception" / "models" / "kids_car").mkdir(parents=True)
    (r / "perception" / "models" / "kids_car" / "kids_car.onnx").write_bytes(b"kids-v1")
    sys.modules.pop("iris_model_manifest", None)
    yield r
    sys.modules.pop("iris_model_manifest", None)


def test_barber_service_check(repo, tmp_path):
    good = hashlib.sha256(b"kids-v1").hexdigest()
    m = _manifest(tmp_path / "m.json", ("perception/models/kids_car/kids_car.onnx", "kids-car-detector", good))
    v = barber.verify_service_models({"model_manifest": m}, repo, env={})
    assert v.checked == ["perception/models/kids_car/kids_car.onnx"]
    (repo / "perception" / "models" / "kids_car" / "kids_car.onnx").write_bytes(b"kids-v2")
    mm = barber.load_verifier(repo)
    with pytest.raises(mm.ManifestError, match="is not kids-car-detector v1"):
        barber.verify_service_models({"model_manifest": m}, repo, env={})
    m2 = _manifest(tmp_path / "m2.json")
    with pytest.raises(mm.ManifestError, match="barber service kids_car_model loads perception/models/kids_car/"
                                               "kids_car.onnx, which this manifest does not list"):
        barber.verify_service_models({"model_manifest": m2}, repo, env={})
    assert barber.verify_service_models({}, repo, env={}).manifest is None          # no manifest: warns
    with pytest.raises(mm.ManifestError, match="MODEL_MANIFEST_REQUIRED"):
        barber.verify_service_models({}, repo, env={"MODEL_MANIFEST_REQUIRED": "1"})


def test_barber_site_yamls_name_their_sites_manifest():
    import yaml
    for name, manifest in (("barber.yaml", "models.manifest.bekky.json"),
                           ("barber.palm-hills.yaml", "models.manifest.palm-hills.json")):
        raw = yaml.safe_load((ROOT / "products" / "barber" / name).read_text())
        assert raw["model_manifest"] == f"products/barber/{manifest}"


def test_fall_service_checkpoints_come_from_the_vendor_map_without_torch():
    assert fall.stream_file_map() == {"j": "mmaction2_joint.pth", "jm": "mmaction2_jm.pth", "b": "mmaction2_bone.pth",
                                      "bm": "mmaction2_bm.pth"}
    assert fall.default_streams() == ("j", "jm")
    files = fall.service_model_files({"weights_dir": "weights"})
    assert [Path(p).relative_to(ROOT).as_posix() for p, _ in files] == [
        "products/fall_detection/weights/mmaction2_joint.pth", "products/fall_detection/weights/mmaction2_jm.pth"]
    assert "torch" not in (ROOT / "products" / "fall_detection" / "model_files.py").read_text().split('"""')[2]


def test_fall_checkpoints_missing_from_disk_refuse_with_dvc_pull(tmp_path):
    sys.modules.pop("iris_model_manifest", None)
    paths = [Path(p) for p, _ in fall.service_model_files({"weights_dir": "weights"})]
    m = _manifest(tmp_path / "m.json", *[(p.relative_to(ROOT).as_posix(), "fall-classifier", "0" * 64) for p in paths])
    mm = fall.load_verifier()
    if all(p.is_file() for p in paths):
        pytest.skip("the checkpoints are pulled here")
    with pytest.raises(mm.ManifestError, match=r"Run `dvc pull products/fall_detection/weights/mmaction2_joint\.pth\.dvc`"):
        fall.verify_checkpoints(paths, m, env={})
