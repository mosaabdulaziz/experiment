"""mtmc.tracking.manifest generate: import -> register/promote (registry baseline) -> generate -> the startup verifier
accepts it. Site alias first, product alias as fallback; refusals for a local file that is not the registered bytes, a
model without a production alias, and a loaded file nobody registered. Temp repo, temp sqlite store."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("numpy")
pytest.importorskip("cv2")

from mtmc.tracking import import_models as im  # noqa: E402
from mtmc.tracking import manifest as gen  # noqa: E402
from mtmc.tracking import registry as reg  # noqa: E402
from perception.engine import model_manifest as mm  # noqa: E402
from perception.engine.model_check import engine_model_refs  # noqa: E402
from perception.engine.sources import load_sidecar_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWIN = "models/reid/swin_tiny_market1501_aicity156_featuredim256.onnx"
MANIFEST = "products/barber/models.manifest.bekky.json"


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    for k in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_ID", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def store(mlflow_store) -> str:
    return mlflow_store                 # tests/conftest.py: this test's own copy of a migrated store


@pytest.fixture
def repo(tmp_path):
    """A barber-like deployment: a PGIE (nvinfer config), the appearance module (re-ID, staff classifier, CLIP
    fallback) and the service's kids car model."""
    r = tmp_path / "repo"
    files = {"det/seg.onnx": b"seg-v1", "det/labels.txt": b"person\n", "staff/s.onnx": b"staff-v1",
             "staff/clip.onnx": b"clip-v1", "staff/prompts.json": b"{}", SWIN: b"swin-v1", "kids/kids.onnx": b"kids-v1"}
    for rel, data in files.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_bytes(data)
    (r / "det" / "config.txt").write_text("[property]\nonnx-file=seg.onnx\nlabelfile-path=labels.txt\n")
    (r / "products" / "barber").mkdir(parents=True)
    shutil.copy(ROOT / "products" / "barber" / "model_files.py", r / "products" / "barber")
    (r / "products" / "barber" / "sidecar.yaml").write_text(yaml.safe_dump({
        "model_manifest": MANIFEST, "pgie_config": "/workspace/iris/det/config.txt",
        "modules": {"appearance": {"staff_model": "classifier", "staff_classifier": "/workspace/iris/staff/s.onnx",
                                   "clip_model": "/workspace/iris/staff/clip.onnx",
                                   "prompts": "/workspace/iris/staff/prompts.json"}}}))
    (r / "products" / "barber" / "barber.yaml").write_text(yaml.safe_dump(
        {"model_manifest": MANIFEST, "kids_car_model": "kids/kids.onnx"}))
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "fixture")
    return r


@pytest.fixture
def setup(tmp_path, repo, mlflow_env, store, capsys):
    """Import the catalog, register every entry and set the production aliases (registry baseline)."""
    catalog = tmp_path / "catalog.yaml"
    entries = [("person-segmenter/seg", "person-segmenter", ["det/seg.onnx", "det/labels.txt"], ["production-barber"]),
               ("staff-classifier/s", "staff-classifier", ["staff/s.onnx"], ["production-barber-bekky"]),
               ("staff-clip/c", "staff-clip", ["staff/clip.onnx", "staff/prompts.json"], ["production-barber"]),
               ("person-reid/r", "person-reid", [SWIN], ["production-barber"]),
               ("kids-car-detector/k", "kids-car-detector", ["kids/kids.onnx"], ["production-barber"])]
    catalog.write_text(yaml.safe_dump({"entries": [
        {"id": i, "capability": c, "source": "imported", "files": f, "provenance": "unknown"} for i, c, f, _ in entries]}))
    conv = yaml.safe_load(reg.CONVENTIONS.read_text())
    conv["baseline"] = {i: a for i, _, _, a in entries}
    conv["deployments"] = [{"product": "barber", "site": "bekky", "sidecar": "products/barber/sidecar.yaml",
                            "service": "products/barber/barber.yaml", "manifest": MANIFEST}]
    conventions = tmp_path / "conventions.yaml"
    conventions.write_text(yaml.safe_dump(conv))
    assert im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store]) == 0
    assert reg.main(["--tracking-uri", store, "--conventions", str(conventions), "baseline", "--catalog",
                     str(catalog), "--root", str(repo)]) == 0
    capsys.readouterr()
    return catalog, conventions


def _generate(repo, store, setup, *extra):
    catalog, conventions = setup
    return gen.main(["generate", "--product", "barber", "--site", "bekky", "--root", str(repo), "--tracking-uri",
                     store, "--conventions", str(conventions), "--catalog", str(catalog), *extra])


def test_generate_then_the_startup_verifier_accepts_it(repo, store, setup, capsys):
    assert _generate(repo, store, setup) == 0
    out = capsys.readouterr().out
    assert f"wrote {repo / MANIFEST} (7 model files)" in out
    m = json.loads((repo / MANIFEST).read_text())
    assert (m["product"], m["site"], m["schema"]) == ("barber", "bekky", 1) and m["tracking_uri"] == store
    assert m["generated_from_commit"] == _git(repo, "rev-parse", "HEAD")
    by_path = {e["path"]: e for e in m["models"]}
    assert set(by_path) == {"det/labels.txt", "det/seg.onnx", "kids/kids.onnx", SWIN, "staff/clip.onnx",
                            "staff/prompts.json", "staff/s.onnx"}
    staff = by_path["staff/s.onnx"]
    assert staff["alias"] == "production-barber-bekky" and staff["registered_model"] == "staff-classifier"
    assert by_path["det/seg.onnx"]["alias"] == "production-barber"                  # no site alias: the product's
    assert by_path["staff/clip.onnx"]["loaded_by"] == "modules.appearance.clip_model"    # the switch's fallback
    assert by_path["kids/kids.onnx"]["loaded_by"] == "barber service kids_car_model"
    assert by_path["det/labels.txt"]["sha256"] == mm.sha256_file(repo / "det" / "labels.txt")
    c = reg.client_for(store)
    v = c.get_model_version_by_alias("staff-classifier", "production-barber-bekky")
    assert staff["version"] == str(v.version) and staff["run_id"] == v.run_id

    # the startup check of the running configuration accepts the generated manifest
    cfg = load_sidecar_config(str(repo / "products" / "barber" / "sidecar.yaml"))
    verified = mm.verify(cfg["model_manifest"], engine_model_refs(cfg), repo, env={"MODEL_MANIFEST_REQUIRED": "1"})
    assert sorted(verified.checked) == ["det/labels.txt", "det/seg.onnx", SWIN, "staff/s.onnx"]


def test_site_alias_wins_over_the_product_alias(repo, store, setup, capsys):
    c = reg.client_for(store)
    conv = reg.load_conventions(setup[1])
    v = c.get_model_version_by_alias("person-segmenter", "production-barber")
    reg.promote(c, conv, "person-segmenter", str(v.version), "production-barber-bekky")
    assert _generate(repo, store, setup) == 0
    entries = {e["path"]: e for e in json.loads((repo / MANIFEST).read_text())["models"]}
    assert entries["det/seg.onnx"]["alias"] == "production-barber-bekky"
    assert entries["staff/clip.onnx"]["alias"] == "production-barber"


def test_refuses_a_local_file_that_is_not_the_registered_version(repo, store, setup, capsys):
    (repo / "staff" / "s.onnx").write_bytes(b"staff-retrained")
    assert _generate(repo, store, setup) == 1
    err = capsys.readouterr().err
    assert "staff/s.onnx (modules.appearance.staff_classifier) is not staff-classifier v1 (production-barber-bekky): " \
           "local sha256" in err
    assert "run `dvc checkout staff/s.onnx.dvc` (or `dvc pull staff/s.onnx.dvc`) for the registered version" in err
    assert not (repo / MANIFEST).exists()                                           # nothing half-written


def test_refuses_a_missing_local_file_and_a_model_without_a_production_alias(repo, store, setup, capsys):
    (repo / "kids" / "kids.onnx").unlink()
    c = reg.client_for(store)
    c.delete_registered_model_alias("person-reid", "production-barber")
    assert _generate(repo, store, setup) == 1
    err = capsys.readouterr().err
    assert "kids/kids.onnx (barber service kids_car_model) is not here; the manifest is only written from the " \
           "registered bytes: run `dvc pull kids/kids.onnx.dvc` (it has no DVC pointer yet" in err
    assert f"{SWIN} (modules.appearance.reid_spec=tao_swin_t): person-reid has no version with alias " \
           "production-barber-bekky or production-barber; promote one" in err


def test_refuses_a_loaded_file_nobody_registered(repo, store, setup, capsys):
    sidecar = repo / "products" / "barber" / "sidecar.yaml"
    cfg = yaml.safe_load(sidecar.read_text())
    cfg["modules"]["appearance"]["staff_classifier"] = "/workspace/iris/staff/new.onnx"
    sidecar.write_text(yaml.safe_dump(cfg))
    (repo / "staff" / "new.onnx").write_bytes(b"unregistered")
    assert _generate(repo, store, setup) == 1
    assert re.search(r"staff/new\.onnx \(modules\.appearance\.staff_classifier\) is not a registered model",
                     capsys.readouterr().err)


def test_files_lists_the_deployment_without_mlflow(repo, setup, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)                # no MLflow needed for `files`
    catalog, conventions = setup
    assert gen.main(["files", "--product", "barber", "--site", "bekky", "--root", str(repo),
                     "--conventions", str(conventions)]) == 0
    out = capsys.readouterr().out
    assert "staff/clip.onnx  <- modules.appearance.clip_model" in out and "kids/kids.onnx  <- barber service" in out
