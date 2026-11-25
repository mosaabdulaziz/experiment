"""mtmc.tracking.registry: conventions, alias rules, register (plain files, lineage, metrics, idempotency, sidecar),
promote/rollback, baseline from the import catalog.

Runs go to a temp sqlite store (no server). Needs pyyaml for the conventions and mlflow for everything that touches a
store; each test skips without what it needs.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from mtmc.tracking import import_models as im  # noqa: E402
from mtmc.tracking import registry as reg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
D2_CAPABILITIES = {
    "vehicle-detector", "plate-detector", "plate-ocr", "vehicle-reid", "staff-classifier", "staff-clip",
    "person-reid", "tracker-reid", "person-segmenter", "kids-car-detector", "object-detector", "fire-detector",
    "face-detector", "face-recognizer", "pose-estimator", "fall-classifier", "depth-estimator",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def conv():
    return reg.load_conventions()


# -- conventions --------------------------------------------------------------------------------------------------
def test_committed_conventions_match_d2_the_catalog_and_products(conv):
    assert set(conv.capabilities) == D2_CAPABILITIES
    entries = im.load_catalog(im.CATALOG)
    assert {e.capability for e in entries} == D2_CAPABILITIES
    assert set(conv.baseline) == {e.id for e in entries}               # every catalog entry has a baseline line
    # products = the products/ folders that have a sidecar yaml (el-mohandes is a client data folder, not a product)
    folders = {p.parent.name for p in (REPO / "products").glob("*/sidecar.yaml")}
    assert set(conv.products) == folders == {"attendance", "barber", "concealment_detection", "fall_detection", "multicam",
                                                  "proximity"}
    sites = re.findall(r"^\s+([a-z0-9-]+)\)$", (REPO / "products" / "barber" / "site.sh").read_text(), re.M)
    assert set(conv.products["barber"]) <= set(sites) and conv.products["barber"] == ("bekky", "palm-hills")
    for e in entries:                                                  # baseline aliases fit who loads the model
        for a in conv.baseline[e.id]:
            assert reg.alias_product(a, conv) in conv.capabilities[e.capability], (e.id, a)
    production = {a for aliases in conv.baseline.values() for a in aliases}
    assert production == {"production-multicam", "production-barber", "production-barber-bekky",
                          "production-barber-palm-hills", "production-attendance", "production-fall_detection",
                          "production-concealment_detection",
                          "production-proximity"}


@pytest.mark.parametrize("text, canonical", [
    ("candidate", "candidate"),
    ("production-multicam", "production-multicam"),
    ("production-fall_detection", "production-fall_detection"),
    ("Production-Fall-Detection", "production-fall_detection"),       # normalised to the products/ spelling
    ("production-barber", "production-barber"),
    ("production-barber-bekky", "production-barber-bekky"),
    ("production-barber-palm_hills", "production-barber-palm-hills"),
    ("PRODUCTION-BARBER-PALM-HILLS", "production-barber-palm-hills"),
])
def test_valid_aliases(conv, text, canonical):
    assert reg.parse_alias(text, conv) == canonical


@pytest.mark.parametrize("text, why", [
    ("production-ppe", "unknown product"),                            # Q6: not a product on main
    ("production-multicam-bekky", "unknown site"),                    # multicam has no sites
    ("production-barber-nowhere", "unknown site"),
    ("production-barber-palm-hills-live", "unknown site"),            # extends palm-hills: uses its alias
    ("production-", "unknown product"),
    ("prod-barber", "must be"),
    ("latest", "must be"),
    ("v1", "must be"),
    ("production-barber.bekky", "may only contain"),                # MLflow's alias characters
    ("production barber", "may only contain"),
    ("staging", "must be"),
])
def test_invalid_aliases(conv, text, why):
    with pytest.raises(reg.RegistryError, match=why):
        reg.parse_alias(text, conv)


def test_conventions_reject_unknown_loaded_by_and_bad_baseline(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"products": {"barber": {"sites": ["bekky"]}},
                                 "capabilities": {"x": {"loaded_by": ["ppe"]}}}))
    with pytest.raises(reg.RegistryError, match="unknown products"):
        reg.load_conventions(p)
    p.write_text(yaml.safe_dump({"products": {"barber": {}}, "capabilities": {"x": {"loaded_by": ["barber"]}},
                                 "baseline": {"x/1": ["production-multicam"]}}))
    with pytest.raises(reg.RegistryError, match="unknown product"):
        reg.load_conventions(p)


# -- store fixtures -----------------------------------------------------------------------------------------------
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
def client(mlflow_env, store):
    return reg.client_for(store)


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    (r / "models").mkdir(parents=True)
    (r / "datasets").mkdir()
    (r / "models" / "det.onnx").write_bytes(b"det-onnx-v1")
    (r / "models" / "reid.pth").write_bytes(b"reid-pth")
    (r / "uv.lock").write_text("version = 1\n")
    (r / "datasets" / "ds.dvc").write_text("outs:\n- md5: 0123456789abcdef0123456789abcdef.dir\n  path: ds\n")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "fixture")
    return r


def _make_run(store, repo, model: Path, *, source: str | None = None, metrics=None, data=True, lockfile=True,
              tags=None, name="run") -> str:
    """A run logged through the tracking helper, as a training (or import) script would. `lockfile`: True = the
    fixture repo's uv.lock, False = none, a path = that path (one that does not exist is tagged `missing`)."""
    from mtmc.tracking import log_model_file, tracked_run
    t = {**({"source": source} if source else {}), **(tags or {})}
    lockfile = "uv.lock" if lockfile is True else (lockfile or None)
    with tracked_run("registry-test", name, data={"ds": "datasets/ds.dvc"} if data else None,
                     lockfile=lockfile, tags=t, tracking_uri=store, root=repo) as run:
        if metrics:
            run.log_metrics(metrics)
        log_model_file(model, artifact_path="weights", run=run)
    return run.run_id


EVAL = {"final/mAP50-95": 0.81, "metrics/precisionB": 0.9, "train/box_loss": 0.5}


# -- Step 0: why register does not use mlflow.register_model ------------------------------------------------------
def test_mlflow_register_model_refuses_a_plain_file(mlflow_env, store, repo):
    """mlflow 3.18's register_model wants an MLmodel or a LoggedModel; a plain .onnx is NOT_FOUND. If this starts
    passing after an mlflow upgrade, register() could switch to it."""
    run_id = _make_run(store, repo, repo / "models" / "det.onnx", metrics=EVAL)
    mlflow_env.set_tracking_uri(store)
    with pytest.raises(mlflow_env.exceptions.MlflowException, match="Unable to find a logged_model"):
        mlflow_env.register_model(f"runs:/{run_id}/weights/det.onnx", "plain")


# -- register -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("file", ["det.onnx", "reid.pth"])
def test_register_plain_file_copies_tags_and_updates_sidecar(client, conv, store, repo, mlflow_env, file):
    model = repo / "models" / file
    run_id = _make_run(store, repo, model, metrics=EVAL)
    r = reg.register(client, conv, "vehicle-detector", run_id, model_path=model)
    assert r.created and r.version == "1" and r.sidecar == model.with_name(file + ".mlflow.json")

    v = client.get_model_version("vehicle-detector", "1")
    assert v.status == "READY" and v.run_id == run_id and v.source == f"runs:/{run_id}/weights/{file}"
    assert Path(mlflow_env.artifacts.download_artifacts(v.source, dst_path=str(repo.parent / "dl"))).read_bytes() \
        == model.read_bytes()
    t = v.tags
    assert t["run_id"] == run_id and t["source"] == "trained" and t["sha256"] == _sha(model.read_bytes())
    assert t["git.commit"] == _git(repo, "rev-parse", "HEAD")
    assert t["data.ds.dvc_md5"] == "0123456789abcdef0123456789abcdef.dir" and "data" not in t
    assert t["env.lockfile"] == "uv.lock" and t["env.lockfile_sha256"] == _sha(b"version = 1\n")
    assert t[f"model_file.{file}.sha256"] == t["sha256"] and t["model.file"] == file
    assert t["metrics"] == "final/mAP50-95=0.81; metrics/precisionB=0.9"         # train/box_loss is not eval
    side = json.loads(model.with_name(file + ".mlflow.json").read_text())
    assert side["registered_model"] == "vehicle-detector" and side["version"] == "1" and side["run_id"] == run_id


def test_register_also_names_the_versions_companion_files(client, conv, store, repo):
    """A retrained export logs best.onnx and labels.txt: registering best.onnx names labels.txt's sidecar too (the
    manifest generator needs it), but only a sidecar of the same run with the same bytes."""
    from mtmc.tracking import log_model_file, tracked_run
    models = repo / "export"
    models.mkdir()
    (models / "best.onnx").write_bytes(b"onnx")
    (models / "labels.txt").write_text("vehicle\n")
    (models / "other.txt").write_text("not this run's\n")
    with tracked_run("registry-test", "export", data={"ds": "datasets/ds.dvc"}, lockfile="uv.lock",
                     tracking_uri=store, root=repo) as run:
        run.log_metrics(EVAL)
        log_model_file(models / "best.onnx", "onnx", run=run)
        log_model_file(models / "labels.txt", "onnx", run=run)
    with tracked_run("registry-test", "elsewhere", tracking_uri=store, root=repo) as other:
        log_model_file(models / "other.txt", "onnx", run=other)
    r = reg.register(client, conv, "vehicle-detector", run.run_id, "best.onnx", models / "best.onnx")
    assert r.companions == [models / "labels.txt.mlflow.json"]
    side = json.loads((models / "labels.txt.mlflow.json").read_text())
    assert (side["registered_model"], side["version"]) == ("vehicle-detector", r.version)
    assert json.loads((models / "other.txt.mlflow.json").read_text())["registered_model"] is None
    # a companion whose bytes changed since it was logged is left alone
    (models / "labels.txt").write_text("changed\n")
    (models / "best.onnx").write_bytes(b"onnx")
    side["registered_model"] = side["version"] = None
    (models / "labels.txt.mlflow.json").write_text(json.dumps(side))
    assert reg.register(client, conv, "vehicle-detector", run.run_id, "best.onnx", models / "best.onnx").companions == []


def test_register_is_idempotent_on_sha256(client, conv, store, repo):
    model = repo / "models" / "det.onnx"
    first = _make_run(store, repo, model, metrics=EVAL)
    a = reg.register(client, conv, "vehicle-detector", first)
    b = reg.register(client, conv, "vehicle-detector", first)
    assert a.created and not b.created and a.version == b.version == "1"
    second = _make_run(store, repo, model, metrics=EVAL, name="same bytes again")     # same file, another run
    c = reg.register(client, conv, "vehicle-detector", second)
    assert not c.created and c.version == "1" and c.run_id == first and "registered earlier" in c.note
    assert len(client.search_model_versions("name = 'vehicle-detector'")) == 1


def test_refuses_missing_lineage(client, conv, store, repo):
    model = repo / "models" / "det.onnx"
    no_data = _make_run(store, repo, model, metrics=EVAL, data=False)
    with pytest.raises(reg.RegistryError, match=r"lacks lineage: data\.\*"):
        reg.register(client, conv, "vehicle-detector", no_data)
    no_commit = _make_run(store, repo, model, metrics=EVAL)
    client.delete_tag(no_commit, "git.commit")
    with pytest.raises(reg.RegistryError, match="lacks lineage: git.commit"):
        reg.register(client, conv, "vehicle-detector", no_commit)
    no_lock = _make_run(store, repo, model, metrics=EVAL)
    client.delete_tag(no_lock, "env.lockfile")
    client.delete_tag(no_lock, "env.lockfile_sha256")
    with pytest.raises(reg.RegistryError, match="lacks lineage: env.lockfile"):
        reg.register(client, conv, "vehicle-detector", no_lock)
    no_sha = _make_run(store, repo, model, metrics=EVAL)
    client.delete_tag(no_sha, "model_file.det.onnx.sha256")
    with pytest.raises(reg.RegistryError, match="no model file sha256"):
        reg.register(client, conv, "vehicle-detector", no_sha)
    assert client.search_registered_models() == []                       # a refusal registers nothing


def test_refuses_trained_run_without_eval_metrics(client, conv, store, repo):
    run_id = _make_run(store, repo, repo / "models" / "det.onnx", metrics={"train/box_loss": 0.5, "lr/pg0": 0.01})
    with pytest.raises(reg.RegistryError, match="no evaluation metrics"):
        reg.register(client, conv, "vehicle-detector", run_id)
    assert client.search_registered_models() == []


# env.lockfile=none: the script named no lockfile; =missing: it named one this checkout does not have
LOCKLESS = {"none": False, "missing": "tools/mtmc-research/uv.lock"}


@pytest.mark.parametrize("state", LOCKLESS)
def test_refuses_trained_run_without_a_locked_environment(client, conv, store, repo, state):
    run_id = _make_run(store, repo, repo / "models" / "det.onnx", metrics=EVAL, lockfile=LOCKLESS[state])
    assert client.get_run(run_id).data.tags["env.lockfile"] == state
    with pytest.raises(reg.RegistryError, match=rf"lacks lineage: env.lockfile \(it is '{state}': a trained model "
                                                r"must come from a locked environment\)"):
        reg.register(client, conv, "vehicle-detector", run_id)
    assert client.search_registered_models() == []                       # a refusal registers nothing


@pytest.mark.parametrize("state", LOCKLESS)
@pytest.mark.parametrize("source", ["imported", "kaggle"])
def test_imported_and_kaggle_runs_register_without_a_lockfile(client, conv, store, repo, source, state):
    run_id = _make_run(store, repo, repo / "models" / "reid.pth", source=source, lockfile=LOCKLESS[state])
    r = reg.register(client, conv, "vehicle-reid", run_id)
    assert r.created and client.get_model_version("vehicle-reid", r.version).tags["env.lockfile"] == state


@pytest.mark.parametrize("source", ["imported", "kaggle"])
def test_imported_run_without_metrics_or_data_gets_none_tags(client, conv, store, repo, source):
    run_id = _make_run(store, repo, repo / "models" / "reid.pth", source=source, data=False,
                       tags={"origin": "torchvision", "provenance.status": "partial"})
    r = reg.register(client, conv, "vehicle-reid", run_id)
    t = client.get_model_version("vehicle-reid", r.version).tags
    assert t["metrics"] == "none" and t["data"] == "none" and t["source"] == source
    assert t["origin"] == "torchvision" and t["provenance.status"] == "partial"


def test_refuses_wrong_name_capability_and_unfinished_run(client, conv, store, repo):
    run_id = _make_run(store, repo, repo / "models" / "det.onnx", metrics=EVAL, tags={"capability": "plate-detector"})
    with pytest.raises(reg.RegistryError, match="not a registered model name"):
        reg.register(client, conv, "plate-detecter", run_id)
    with pytest.raises(reg.RegistryError, match="is a plate-detector run"):
        reg.register(client, conv, "vehicle-detector", run_id)
    client.set_terminated(run_id, "FAILED")
    with pytest.raises(reg.RegistryError, match="FAILED, not FINISHED"):
        reg.register(client, conv, "plate-detector", run_id)


def test_refuses_a_local_file_that_is_not_the_runs(client, conv, store, repo):
    model = repo / "models" / "det.onnx"
    run_id = _make_run(store, repo, model, metrics=EVAL)
    model.write_bytes(b"retrained since")
    with pytest.raises(reg.RegistryError, match="is not the file run"):
        reg.register(client, conv, "vehicle-detector", run_id, model_path=model)
    assert client.search_registered_models() == []


def test_cli_register_from_sidecar(mlflow_env, store, repo, capsys):
    model = repo / "models" / "det.onnx"
    run_id = _make_run(store, repo, model, metrics=EVAL, tags={"capability": "vehicle-detector"})
    assert reg.main(["--tracking-uri", store, "register", "--model", str(model)]) == 0
    assert reg.main(["--tracking-uri", store, "register", "--model", str(model)]) == 0
    out = capsys.readouterr().out
    assert f"registered: vehicle-detector v1 (run {run_id}" in out and "exists: vehicle-detector v1" in out
    assert json.loads((repo / "models" / "det.onnx.mlflow.json").read_text())["version"] == "1"


# -- promote ------------------------------------------------------------------------------------------------------
def test_promote_rollback_and_previous_version(client, conv, store, repo, capsys):
    model = repo / "models" / "det.onnx"
    v1 = reg.register(client, conv, "staff-classifier", _make_run(store, repo, model, metrics=EVAL)).version
    model.write_bytes(b"det-onnx-v2")
    v2 = reg.register(client, conv, "staff-classifier", _make_run(store, repo, model, metrics=EVAL)).version
    assert (v1, v2) == ("1", "2")

    assert reg.promote(client, conv, "staff-classifier", v1, "production-barber-bekky") == \
        ("production-barber-bekky", None)
    assert reg.promote(client, conv, "staff-classifier", v2, "Production-Barber-Bekky") == \
        ("production-barber-bekky", "1")
    assert reg.main(["--tracking-uri", store, "promote", "staff-classifier", "1",       # rollback
                     "--alias", "production-barber-bekky"]) == 0
    assert "production-barber-bekky: staff-classifier v1 (previously v2)" in capsys.readouterr().out
    assert str(client.get_model_version_by_alias("staff-classifier", "production-barber-bekky").version) == "1"
    # a site alias and the product alias are independent
    reg.promote(client, conv, "staff-classifier", v2, "production-barber-palm-hills")
    assert str(client.get_model_version_by_alias("staff-classifier", "production-barber-palm-hills").version) == "2"
    reg.promote(client, conv, "staff-classifier", v2, "candidate")


def test_promote_refusals(client, conv, store, repo, capsys):
    v = reg.register(client, conv, "staff-classifier",
                     _make_run(store, repo, repo / "models" / "det.onnx", metrics=EVAL)).version
    with pytest.raises(reg.RegistryError, match="multicam does not load staff-classifier"):
        reg.promote(client, conv, "staff-classifier", v, "production-multicam")
    with pytest.raises(reg.RegistryError, match="unknown site"):
        reg.promote(client, conv, "staff-classifier", v, "production-barber-zamalek")
    with pytest.raises(reg.RegistryError, match="has no version 9"):
        reg.promote(client, conv, "staff-classifier", "9", "candidate")
    assert reg.main(["--tracking-uri", store, "promote", "staff-classifier", v, "--alias", "production-ppe"]) == 1
    assert "refused: alias 'production-ppe': unknown product" in capsys.readouterr().err


def test_no_store_and_unreachable_server_are_refused(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(reg.RegistryError, match="no tracking store"):
        reg.client_for(None)
    with pytest.raises(reg.RegistryError, match="not available"):
        reg.client_for("http://127.0.0.1:9")                        # discard port: nothing listens


# -- baseline -----------------------------------------------------------------------------------------------------
@pytest.fixture
def baseline_setup(tmp_path, repo):
    (repo / "models" / "staff").mkdir()
    (repo / "models" / "staff" / "s.onnx").write_bytes(b"staff-onnx")
    (repo / "models" / "staff" / "s.json").write_text(json.dumps({"threshold": 0.8, "sha256": _sha(b"staff-onnx")}))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "staff")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"entries": [
        {"id": "object-detector/det", "capability": "object-detector", "source": "imported",
         "files": ["models/det.onnx"], "provenance": {"origin": "Ultralytics COCO"}},
        {"id": "staff-classifier/s", "capability": "staff-classifier", "source": "imported",
         "files": ["models/staff/s.onnx", "models/staff/s.json"],
         "records": [{"kind": "staff_json", "file": "models/staff/s.json"}], "provenance": "unknown"},
        {"id": "vehicle-reid/r", "capability": "vehicle-reid", "source": "imported", "files": ["models/reid.pth"]},
    ]}))
    raw = yaml.safe_load(reg.CONVENTIONS.read_text())
    raw["baseline"] = {"object-detector/det": ["production-attendance", "production-fall_detection"],
                       "staff-classifier/s": ["production-barber-bekky"], "vehicle-reid/r": []}
    conventions = tmp_path / "conventions.yaml"
    conventions.write_text(yaml.safe_dump(raw))
    return catalog, conventions


def _baseline(store, repo, catalog, conventions, *extra):
    return reg.main(["--tracking-uri", store, "--conventions", str(conventions), "baseline", "--catalog",
                     str(catalog), "--root", str(repo), *extra])


def test_baseline_registers_imports_and_sets_aliases(mlflow_env, store, repo, baseline_setup, capsys):
    catalog, conventions = baseline_setup
    assert im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store,
                    "--only", "object-detector/det", "staff-classifier/s"]) == 0
    capsys.readouterr()

    assert _baseline(store, repo, catalog, conventions, "--dry-run") == 1        # vehicle-reid/r not imported
    out = capsys.readouterr().out
    assert re.search(r"would-register\s+object-detector/det", out) and re.search(r"missing\s+vehicle-reid/r", out)
    c = reg.client_for(store)
    assert c.search_registered_models() == []                                     # dry run wrote nothing

    assert _baseline(store, repo, catalog, conventions) == 1
    out = capsys.readouterr().out
    assert re.search(r"registered\s+object-detector/det\s+v1\s+production-attendance,production-fall_detection", out)
    assert re.search(r"registered\s+staff-classifier/s\s+v1\s+production-barber-bekky", out)
    for alias in ("production-attendance", "production-fall_detection"):
        assert str(c.get_model_version_by_alias("object-detector", alias).version) == "1"
    staff = c.get_model_version_by_alias("staff-classifier", "production-barber-bekky")
    assert staff.tags["metrics"] == "none" and staff.tags["data"] == "none" and staff.tags["source"] == "imported"
    assert staff.tags["model.file"] == "s.onnx" and staff.tags["import.catalog_entry"] == "staff-classifier/s"
    assert json.loads((repo / "models" / "det.onnx.mlflow.json").read_text())["version"] == "1"
    assert json.loads((repo / "models" / "staff" / "s.onnx.mlflow.json").read_text())["registered_model"] == \
        "staff-classifier"

    assert im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store]) == 0
    capsys.readouterr()
    assert _baseline(store, repo, catalog, conventions) == 0                      # idempotent, now complete
    out = capsys.readouterr().out
    assert re.search(r"exists\s+object-detector/det\s+v1", out) and re.search(r"registered\s+vehicle-reid/r\s+v1\s+-", out)
    assert len(c.search_model_versions("name = 'object-detector'")) == 1


def test_baseline_registers_what_it_can_then_summarises_the_missing(mlflow_env, store, repo, tmp_path, capsys):
    """Entries that cannot be registered (file never pulled, no DVC pointer at all, not imported yet) don't stop the
    others; they are listed at the end with the reason, and the exit code is 1."""
    (repo / "models" / "pose.onnx.dvc").write_text("outs:\n- md5: 11111111111111111111111111111111\n  path: pose.onnx\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pose pointer")
    outside = tmp_path / "home" / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"     # never created
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"entries": [
        {"id": "pose-estimator/p", "capability": "pose-estimator", "source": "imported", "files": ["models/pose.onnx"]},
        {"id": "face-recognizer/buffalo_l", "capability": "face-recognizer", "source": "imported",
         "files": [str(outside)]},
        {"id": "object-detector/det", "capability": "object-detector", "source": "imported",
         "files": ["models/det.onnx"]},
        {"id": "vehicle-reid/r", "capability": "vehicle-reid", "source": "imported", "files": ["models/reid.pth"]},
    ]}))
    raw = yaml.safe_load(reg.CONVENTIONS.read_text())
    raw["baseline"] = {"pose-estimator/p": ["production-fall_detection"],
                       "face-recognizer/buffalo_l": ["production-attendance"],
                       "object-detector/det": ["production-attendance"], "vehicle-reid/r": ["production-multicam"]}
    conventions = tmp_path / "conventions.yaml"
    conventions.write_text(yaml.safe_dump(raw))
    # the import skips the two absent entries; vehicle-reid/r is left out of it on purpose
    im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store,
             "--only", "pose-estimator/p", "face-recognizer/buffalo_l", "object-detector/det"])
    capsys.readouterr()

    assert _baseline(store, repo, catalog, conventions) == 1
    out = capsys.readouterr().out
    table, summary = out.split("Not registered (3), everything else was:")
    assert re.search(r"registered\s+object-detector/det\s+v1\s+production-attendance", table)
    assert "1 registered, 3 missing" in table
    lines = dict(re.findall(r"^  missing\s+(\S+): (.*)$", summary, re.M))
    assert lines["pose-estimator/p"] == "not present locally: models/pose.onnx (`dvc pull models/pose.onnx.dvc`)"
    assert lines["face-recognizer/buffalo_l"] == f"not present locally: {outside} (no DVC pointer)"
    assert lines["vehicle-reid/r"].startswith("files present but not imported yet")
    c = reg.client_for(store)
    assert str(c.get_model_version_by_alias("object-detector", "production-attendance").version) == "1"
    assert {m.name for m in c.search_registered_models()} == {"object-detector"}


def test_baseline_entry_error_does_not_stop_the_others(mlflow_env, store, repo, baseline_setup, monkeypatch, capsys):
    catalog, conventions = baseline_setup
    im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store])
    capsys.readouterr()
    real = reg.register

    def flaky(client, conv, name, *a, **k):
        if name == "staff-classifier":
            raise OSError("artifact store hiccup")
        return real(client, conv, name, *a, **k)
    monkeypatch.setattr(reg, "register", flaky)
    assert _baseline(store, repo, catalog, conventions) == 1
    out = capsys.readouterr().out
    assert re.search(r"registered\s+object-detector/det", out) and re.search(r"registered\s+vehicle-reid/r", out)
    assert "failed   staff-classifier/s: OSError: artifact store hiccup" in out


def test_a_companion_shared_by_two_versions_keeps_the_first_registered_one(mlflow_env, store, repo, tmp_path, caplog):
    """The proximity import of 2026-11-15: the barber r384 seg and the proximity 1280 seg both load the same
    labels.txt. One sidecar per file: once it names a registered version, a second import or registration of the
    same bytes keeps it (and warns) instead of repointing it."""
    seg = repo / "models" / "seg"
    seg.mkdir()
    (seg / "r384.onnx").write_bytes(b"seg-r384")
    (seg / "1280.onnx").write_bytes(b"seg-1280")
    (seg / "labels.txt").write_text("person\ncar\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seg")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"entries": [
        {"id": f"person-segmenter/{v}", "capability": "person-segmenter", "source": "imported",
         "files": [f"models/seg/{v}.onnx", "models/seg/labels.txt"], "provenance": "unknown"} for v in ("r384", "1280")]}))
    raw = yaml.safe_load(reg.CONVENTIONS.read_text())
    raw["baseline"] = {"person-segmenter/r384": ["production-barber"], "person-segmenter/1280": ["production-proximity"]}
    conventions = tmp_path / "conventions.yaml"
    conventions.write_text(yaml.safe_dump(raw))
    labels = seg / "labels.txt.mlflow.json"

    def imp(entry):
        assert im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store, "--only", entry]) == 0

    imp("person-segmenter/r384")
    _baseline(store, repo, catalog, conventions)
    first = json.loads(labels.read_text())
    assert (first["registered_model"], first["version"]) == ("person-segmenter", "1")

    caplog.clear()
    imp("person-segmenter/1280")                                       # logs labels.txt again, from a new run
    assert json.loads(labels.read_text()) == first
    assert "shared by several versions" in caplog.text
    second_run = json.loads((seg / "1280.onnx.mlflow.json").read_text())["run_id"]
    assert second_run != first["run_id"]
    c = reg.client_for(store)
    assert c.get_run(second_run).data.tags["model_file.labels.txt.sha256"] == first["sha256"]   # still its artifact

    _baseline(store, repo, catalog, conventions)
    assert json.loads(labels.read_text()) == first                     # the registry's companion update keeps it too
    assert json.loads((seg / "1280.onnx.mlflow.json").read_text())["version"] == "2"
    assert str(c.get_model_version_by_alias("person-segmenter", "production-proximity").version) == "2"
    caplog.clear()
    assert reg.update_sidecar(seg / "labels.txt", "person-segmenter", "2", first["sha256"]) is None
    assert json.loads(labels.read_text()) == first and "shared by several versions" in caplog.text
