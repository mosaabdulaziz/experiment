"""The whole MLflow chain end to end (sub-task 10 of the MLflow story), with dummy files and a fake trainer.

    fake finetune run -> export child run (source_run_id) -> evaluation on a named set -> register (refused without
    lineage, then accepted) -> promote to candidate and production -> manifest -> verifier passes -> a new version is
    deployed -> the file changes -> verifier refuses -> rollback (promote the previous version, put its bytes back,
    regenerate) -> verifier passes; and: tracking server unreachable during training -> offline, training completes.

Off by default (not in the default CI job): it runs only with IRIS_MLFLOW_E2E=1.
- default store: a temp sqlite store (no server);
- IRIS_MLFLOW_E2E_URI=http://10.2.0.50:15050 runs it against that server, as the user in MLFLOW_TRACKING_USERNAME /
  MLFLOW_TRACKING_PASSWORD (the shared `iris` client user). It cleans up after itself: the dummy registered model is
  deleted with its versions, and its runs are deleted (soft; their ids are written to IRIS_MLFLOW_E2E_RECORD if set,
  for `mlflow gc --run-ids` by an operator: the API cannot purge runs).

    IRIS_MLFLOW_E2E=1 python -m pytest tests/test_mlflow_e2e.py                       # temp sqlite store
    IRIS_MLFLOW_E2E=1 IRIS_MLFLOW_E2E_URI=http://10.2.0.50:15050 python -m pytest tests/test_mlflow_e2e.py

Runs go to the experiment `instrumentation-check`. The registered model is `e2e-dummy-detector-<random>`, declared in
a test-only conventions file (the registry and the manifest generator take the conventions as an argument, as their
CLIs take --conventions), with a test-only product `e2e-dummy`: no version of a real model is ever created.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e,
              pytest.mark.skipif(os.environ.get("IRIS_MLFLOW_E2E") != "1",
                                 reason="end-to-end MLflow chain: set IRIS_MLFLOW_E2E=1 to run it")]

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mlflow_instrument_training as T                 # noqa: E402  (the fake Ultralytics and export)

from mtmc import tracking                                   # noqa: E402
from mtmc.tracking import eval as tev                       # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = "instrumentation-check"
PRODUCT = "e2e-dummy"
DATA_MD5 = "ad5ed68445f31c338d2c5fdf8c8f4283.dir"            # vehicle_detection_1cls's real pointer md5
SET_MD5 = "e2e0e2e0e2e0e2e0e2e0e2e0e2e0e2e0"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The tracking store: IRIS_MLFLOW_E2E_URI (a server) or a temp sqlite store."""
    for k in tracking._ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    uri = os.environ.get("IRIS_MLFLOW_E2E_URI") or f"sqlite:///{tmp_path / 'store' / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def chain(store, tmp_path, monkeypatch):
    """The working dir (dataset pointer, base weights), the fake Ultralytics and export, a unique dummy capability, and
    the test-only conventions. Cleans up the dummy model and its runs afterwards."""
    from mlflow.tracking import MlflowClient
    from mtmc.tracking import registry

    name = f"e2e-dummy-detector-{uuid.uuid4().hex[:8]}"
    work = tmp_path / "work"
    ds = work / "products/el-mohandes/datasets"
    (ds / "vehicle_detection_1cls").mkdir(parents=True)
    (ds / "vehicle_detection_1cls.dvc").write_text(
        f"outs:\n- md5: {DATA_MD5}\n  size: 1\n  nfiles: 1\n  hash: md5\n  path: vehicle_detection_1cls\n")
    (ds / "vehicle_detection_1cls/data.yaml").write_text("nc: 1\nnames: [vehicle]\n")
    (work / "yolo11s.pt").write_bytes(b"e2e base weights")
    golden = work / "data" / "golden-e2e"
    golden.mkdir(parents=True)
    (golden / "labels.txt").write_text("fixture\n")
    (work / "data" / "golden-e2e.dvc").write_text(
        f"outs:\n- md5: {SET_MD5}.dir\n  size: 8\n  nfiles: 1\n  hash: md5\n  path: golden-e2e\n")
    monkeypatch.chdir(work)

    T._install_fake_ultralytics(monkeypatch, tmp_path, "simulated")
    v = importlib.import_module("mtmc.finetune_vehicle_detector")
    monkeypatch.setattr(v, "CAPABILITY", name)
    monkeypatch.setattr(v, "EXPERIMENT", EXPERIMENT)
    launcher = tmp_path / "export_launcher.py"
    launcher.write_text(f"import sys\nsys.path.insert(0, {str(REPO)!r})\n" + T.FAKE_EXPORT
                        + "import mtmc.export_vehicle_onnx as e\n"
                        + f"e.CAPABILITY, e.EXPERIMENT = {name!r}, {EXPERIMENT!r}\n"
                        + "e.export_one, e.check_upstream, e.load_upstream = export_one, check_upstream, load_upstream\n"
                        + "e.main(sys.argv[1:])\n")
    monkeypatch.setattr(v, "EXPORT_CMD", [sys.executable, str(launcher)])

    conventions = tmp_path / "e2e_conventions.yaml"
    real = (REPO / "mtmc/tracking/registry_conventions.yaml").read_text()
    import yaml
    metrics = yaml.safe_load(real)["evaluation_metrics"]
    conventions.write_text(yaml.safe_dump({
        "products": {PRODUCT: {"sites": []}},
        "capabilities": {name: {"loaded_by": [PRODUCT]}},
        "evaluation_metrics": metrics}))
    conv = registry.load_conventions(conventions)
    client = MlflowClient(tracking_uri=store)
    ctx = argparse.Namespace(name=name, work=work, golden=golden, v=v, conv=conv, client=client, store=store,
                             registry=registry, root=tmp_path / "deploy", created_runs=[])
    yield ctx

    # -- cleanup: the dummy registered model (with its versions) and every run of the dummy capability
    try:
        client.delete_registered_model(name)
    except Exception:                                               # never registered (an early failure)
        pass
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], f"tags.capability = '{name}'") if exp else []
    ids = sorted({r.info.run_id for r in runs} | set(ctx.created_runs))
    for rid in ids:
        try:
            client.delete_run(rid)
        except Exception:
            pass
    if record := os.environ.get("IRIS_MLFLOW_E2E_RECORD"):
        with open(record, "a", encoding="utf-8") as f:
            f.write("".join(f"{rid}\n" for rid in ids))


def _train_and_export(c, run_name: str, imgsz: int) -> tuple[str, str, Path]:
    """A fake finetune with --export; returns (training run, export run, export dir). imgsz changes the ONNX bytes."""
    out = c.work / f"export-{run_name}"
    c.v.main(["--data", "products/el-mohandes/datasets/vehicle_detection_1cls/data.yaml", "--model", "yolo11s.pt",
              "--epochs", "2", "--name", run_name, "--device", "cpu", "--imgsz", str(imgsz),
              "--export", "--export-dir", str(out)])
    meta = tracking.read_model_sidecar(out / "best.onnx")
    export_run = c.client.get_run(meta["run_id"])
    return export_run.data.tags["mlflow.parentRunId"], export_run.info.run_id, out


def _deploy(c, export_dir: Path) -> None:
    """Put an export's files (with their sidecars) where the dummy deployment's nvinfer config loads them."""
    models = c.root / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in ("best.onnx", "labels.txt"):
        for suffix in ("", tracking.SIDECAR_SUFFIX):
            shutil.copyfile(export_dir / (f + suffix), models / (f + suffix))


def _deployment(c):
    """A minimal deployment under c.root: sidecar yaml -> nvinfer config -> models/best.onnx + labels.txt."""
    from mtmc.tracking.manifest import Deployment
    c.root.mkdir(parents=True, exist_ok=True)
    (c.root / "config_infer_dummy.txt").write_text(
        "[property]\nonnx-file=models/best.onnx\nlabelfile-path=models/labels.txt\nnum-detected-classes=1\n")
    (c.root / "sidecar.yaml").write_text("pgie_config: config_infer_dummy.txt\nmodel_manifest: models.manifest.json\n"
                                         "modules: {}\n")
    return Deployment(PRODUCT, None, "sidecar.yaml", None, "models.manifest.json")


def _generate(c, dep) -> dict:
    from mtmc.tracking import manifest
    m = manifest.generate(c.client, dep, c.root, c.conv, tracking_uri=c.store)
    manifest.write(m, c.root / dep.manifest)
    return m


def _verify(c):
    from perception.engine.model_check import verify_models
    from perception.engine.sources import load_sidecar_config
    return verify_models(load_sidecar_config(str(c.root / "sidecar.yaml")), what="the e2e deployment", root=c.root,
                         env={})


def test_the_whole_chain(chain):
    c = chain
    from perception.engine.model_manifest import ManifestError
    RegistryError = c.registry.RegistryError

    # 1. fake finetune -> export child run, linked by source_run_id
    train1, export1, out1 = _train_and_export(c, "e2e-v1", 1280)
    t = c.client.get_run(export1).data.tags
    assert t["source_run_id"] == train1 and t["mlflow.parentRunId"] == train1 and t["capability"] == c.name
    assert c.client.get_run(train1).info.status == "FINISHED"

    # 2. evaluation on a named, versioned set, into the export run
    args = argparse.Namespace(eval_set="golden-e2e", run_id=None, force=False, no_track=False)
    target = tev.prepare_evaluation(args, out1 / "best.onnx", c.golden, "tests.test_mlflow_e2e")
    assert target.log({"recall": 0.9}) == export1
    r = c.client.get_run(export1)
    assert r.data.metrics["eval/golden-e2e/recall"] == pytest.approx(0.9)
    assert r.data.tags["eval.golden-e2e.dvc_md5"] == f"{SET_MD5}.dir"

    # 3. register: a run without lineage is refused, the export run is accepted
    exp_id = c.client.get_experiment_by_name(EXPERIMENT).experiment_id
    bare = c.client.create_run(exp_id, run_name="e2e-no-lineage", tags={"capability": c.name})
    c.created_runs.append(bare.info.run_id)
    c.client.log_artifact(bare.info.run_id, str(out1 / "best.onnx"), "onnx")
    c.client.set_tag(bare.info.run_id, "model_file.best.onnx.sha256", tracking.sha256_file(out1 / "best.onnx"))
    c.client.set_terminated(bare.info.run_id, "FINISHED")
    with pytest.raises(RegistryError, match="lacks lineage: git.commit, env.lockfile, data"):
        c.registry.register(c.client, c.conv, c.name, bare.info.run_id, "best.onnx")
    reg1 = c.registry.register(c.client, c.conv, c.name, export1, "best.onnx", out1 / "best.onnx")
    assert reg1.created and str(reg1.version) == "1"
    assert tracking.read_model_sidecar(out1 / "best.onnx")["version"] == "1"

    # 4. promote: candidate, then production
    assert c.registry.promote(c.client, c.conv, c.name, "1", "candidate") == ("candidate", None)
    assert c.registry.promote(c.client, c.conv, c.name, "1", f"production-{PRODUCT}") == (f"production-{PRODUCT}", None)

    # 5. manifest for the deployment, verifier passes
    dep = _deployment(c)
    _deploy(c, out1)
    m = _generate(c, dep)
    assert {(e["path"], e["version"]) for e in m["models"]} == {("models/best.onnx", "1"), ("models/labels.txt", "1")}
    assert sorted(_verify(c).checked) == ["models/best.onnx", "models/labels.txt"]

    # 6. a new version is trained, registered, promoted and deployed: the manifest follows
    _, export2, out2 = _train_and_export(c, "e2e-v2", 960)
    reg2 = c.registry.register(c.client, c.conv, c.name, export2, "best.onnx", out2 / "best.onnx")
    assert str(reg2.version) == "2"
    assert c.registry.promote(c.client, c.conv, c.name, "2", f"production-{PRODUCT}")[1] == "1"
    _deploy(c, out2)
    assert {e["version"] for e in _generate(c, dep)["models"]} == {"2"}
    _verify(c)

    # 7. the deployed file changes: the verifier refuses
    (c.root / "models" / "best.onnx").write_bytes(b"someone swapped the model")
    with pytest.raises(ManifestError, match=r"models/best.onnx is not .* v2"):
        _verify(c)

    # 8. rollback: the previous version back on the alias, its bytes back, the manifest regenerated
    assert c.registry.promote(c.client, c.conv, c.name, "1", f"production-{PRODUCT}")[1] == "2"
    _deploy(c, out1)                                              # = `dvc checkout` of the pointer from that commit
    m = _generate(c, dep)
    assert {e["version"] for e in m["models"]} == {"1"}
    assert sorted(_verify(c).checked) == ["models/best.onnx", "models/labels.txt"]
    assert json.loads((c.root / "models.manifest.json").read_text())["models"][0]["run_id"] == export1


def test_server_unreachable_during_training_goes_offline_and_completes(chain, monkeypatch):
    c = chain
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{T._closed_port()}")
    c.v.main(["--data", "products/el-mohandes/datasets/vehicle_detection_1cls/data.yaml", "--model", "yolo11s.pt",
              "--epochs", "2", "--name", "e2e-offline", "--device", "cpu"])
    best = c.work / "products/el-mohandes/runs/vehicle_detector/e2e-offline/weights/best.pt"
    meta = tracking.read_model_sidecar(best)
    assert meta["tracking_mode"] == "offline"
    from mlflow.tracking import MlflowClient
    off = MlflowClient(tracking_uri=meta["tracking_uri"])
    run = off.get_run(meta["run_id"])
    assert run.info.status == "FINISHED" and run.data.tags["tracking.mode"] == "offline"
    assert "eval/val/mAP50-95" in run.data.metrics
