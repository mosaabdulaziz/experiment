"""mtmc.tracking: lineage tags, sidecars, offline fallback, no-op without mlflow, Ultralytics run lifecycle.

Needs no MLflow server: runs go to a sqlite store in a temp dir. Tests that need the mlflow package skip without it
(the root `test` group does not install it); the lineage and no-op tests run anyway. The Ultralytics tests run the
real callback source (ultralytics/utils/callbacks/mlflow.py of the installed version) against a fake trainer, with a
stub for `ultralytics.utils` so neither torch nor a training run is needed; they skip when ultralytics is absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

from mtmc import tracking
from mtmc.tracking import _lineage

REPO = Path(__file__).resolve().parents[1]

DIR_MD5 = "ad5ed68445f31c338d2c5fdf8c8f4283.dir"
FILE_MD5 = "a86ed5a372e8307445c9863a0713ad33"


# -- fixtures -----------------------------------------------------------------------------------------------------
def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo with a directory pointer, a file pointer, a lockfile and a tracked file; clean."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "datasets").mkdir()
    (r / "datasets" / "vehicle_detection_1cls.dvc").write_text(
        f"outs:\n- md5: {DIR_MD5}\n  size: 3553166105\n  nfiles: 4819\n  hash: md5\n  path: vehicle_detection_1cls\n")
    (r / "args.yaml.dvc").write_text(f"outs:\n- md5: {FILE_MD5}\n  size: 1865\n  hash: md5\n  path: args.yaml\n")
    (r / "uv.lock").write_text("version = 1\n")
    (r / "train.py").write_text("print('train')\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    # the dataset itself is pulled data: untracked, so it never makes the tree dirty
    (r / "datasets" / "vehicle_detection_1cls").mkdir()
    (r / "datasets" / "vehicle_detection_1cls" / "data.yaml").write_text("nc: 1\n")
    return r


@pytest.fixture
def mlflow_mod(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    for k in tracking._ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def store(mlflow_store) -> str:
    return mlflow_store                 # tests/conftest.py: this test's own copy of a migrated store


def _client(uri: str):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=uri)


def _runs(uri: str, experiment: str):
    c = _client(uri)
    exp = c.get_experiment_by_name(experiment)
    return c.search_runs([exp.experiment_id]) if exp else []


def _artifacts(uri: str, run_id: str, path: str = "") -> list[str]:
    return [a.path for a in _client(uri).list_artifacts(run_id, path)]


# -- lineage (no mlflow needed) -----------------------------------------------------------------------------------
def test_parse_dvc_pointer_dir_and_file(repo):
    assert _lineage.parse_dvc_pointer(repo / "datasets" / "vehicle_detection_1cls.dvc") == {
        "md5": DIR_MD5, "path": "vehicle_detection_1cls"}
    assert _lineage.parse_dvc_pointer(repo / "args.yaml.dvc") == {"md5": FILE_MD5, "path": "args.yaml"}


def test_parse_dvc_pointer_real_repo_pointers():
    """Every pointer in this repo parses (the format the parser assumes is the one DVC writes here)."""
    out = subprocess.run(["git", "ls-files", "-z", "*.dvc"], cwd=REPO, capture_output=True, text=True).stdout
    pointers = [REPO / p for p in out.split("\0") if p]
    if not pointers:
        pytest.skip("no .dvc pointers in this checkout")
    for p in pointers:
        out = _lineage.parse_dvc_pointer(p)
        assert len(out["md5"].removesuffix(".dir")) == 32, p


def test_data_tags_pointer_inside_tracked_dir_and_unversioned(repo, caplog):
    tags = _lineage.data_tags(["datasets/vehicle_detection_1cls/data.yaml", "train.py"], repo)
    assert tags["data.vehicle_detection_1cls.dvc_md5"] == DIR_MD5
    assert tags["data.vehicle_detection_1cls.dvc_path"] == "datasets/vehicle_detection_1cls.dvc"
    assert tags["data.train.dvc_md5"] == "unversioned"
    assert tags["data.train.sha256"] == hashlib.sha256(b"print('train')\n").hexdigest()


def test_git_dirty_detection(repo):
    assert _lineage.git_info(repo).tags["git.dirty"] == "false"     # untracked pulled data does not count
    (repo / "train.py").write_text("print('changed')\n")
    info = _lineage.git_info(repo)
    assert info.tags["git.dirty"] == "true"
    assert info.dirty_files == ["train.py"]


# -- runs ---------------------------------------------------------------------------------------------------------
def test_tracked_run_sets_lineage_tags(mlflow_mod, repo, store):
    with tracking.tracked_run("exp", "r1", params={"epochs": 150, "aug": {"mosaic": 1.0}},
                              data={"vehicle_detection_1cls": "datasets/vehicle_detection_1cls.dvc",
                                    "args": "args.yaml.dvc"},
                              lockfile="uv.lock", tags={"purpose": "test"}, tracking_uri=store, root=repo) as run:
        assert run.active and run.mode == "local"
        run_id = run.run_id
    r = _client(store).get_run(run_id)
    t = r.data.tags
    assert t["git.commit"] == _git(repo, "rev-parse", "HEAD")
    assert t["git.branch"] == "main" and t["git.dirty"] == "false"
    assert t["data.vehicle_detection_1cls.dvc_md5"] == DIR_MD5
    assert t["data.vehicle_detection_1cls.dvc_path"] == "datasets/vehicle_detection_1cls.dvc"
    assert t["data.args.dvc_md5"] == FILE_MD5
    assert t["env.lockfile"] == "uv.lock"
    assert t["env.lockfile_sha256"] == hashlib.sha256(b"version = 1\n").hexdigest()
    assert t["env.python"] and t["host"] == socket.gethostname()
    assert t["env.mlflow"] == mlflow_mod.__version__
    assert t["tracking.mode"] == "local" and t["purpose"] == "test"
    assert r.data.params == {"epochs": "150", "aug": '{"mosaic": 1.0}'}
    assert r.info.status == "FINISHED"
    assert "lineage" not in _artifacts(store, run_id)               # clean tree + lockfile: nothing extra


def test_missing_lockfile_is_tagged_and_does_not_fail(mlflow_mod, repo, store):
    with tracking.tracked_run("exp", lockfile="tools/mtmc-research/uv.lock", tracking_uri=store, root=repo) as run:
        run_id = run.run_id
    t = _client(store).get_run(run_id).data.tags
    assert t["env.lockfile"] == "missing"
    assert t["env.lockfile_expected"] == "tools/mtmc-research/uv.lock"
    assert "env.lockfile_sha256" not in t
    assert "lineage/installed_packages.txt" in _artifacts(store, run_id, "lineage")


def test_dirty_tree_tag_and_artifact(mlflow_mod, repo, store):
    (repo / "train.py").write_text("print('changed')\n")
    with tracking.tracked_run("exp", lockfile="uv.lock", tracking_uri=store, root=repo) as run:
        run_id = run.run_id
    assert _client(store).get_run(run_id).data.tags["git.dirty"] == "true"
    local = _client(store).download_artifacts(run_id, "lineage/git_dirty_files.txt")
    assert Path(local).read_text() == "train.py\n"


def test_failed_block_marks_run_failed_and_reraises(mlflow_mod, repo, store):
    with pytest.raises(RuntimeError, match="boom"):
        with tracking.tracked_run("exp", tracking_uri=store, root=repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")
    assert _client(store).get_run(run_id).info.status == "FAILED"


def test_environment_restored_after_run(mlflow_mod, repo, store, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "outer")
    with tracking.tracked_run("exp", tracking_uri=store, root=repo) as run:
        tracking.configure_ultralytics(run)
        assert os.environ["MLFLOW_RUN_ID"] == run.run_id and os.environ["MLFLOW_EXPERIMENT_NAME"] == "exp"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "outer"
    for k in ("MLFLOW_RUN_ID", "MLFLOW_RUN", "MLFLOW_KEEP_RUN_ACTIVE", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID"):
        assert k not in os.environ, k


# -- sidecar ------------------------------------------------------------------------------------------------------
def test_sidecar_written_read_back_and_verified(mlflow_mod, repo, store):
    model = repo / "best.onnx"
    model.write_bytes(b"\x08\x01onnx-bytes")
    with tracking.tracked_run("exp", tracking_uri=store, root=repo) as run:
        meta = tracking.log_model_file(model, artifact_path="model")
        run_id = run.run_id
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    sidecar = repo / "best.onnx.mlflow.json"
    on_disk = json.loads(sidecar.read_text())
    assert on_disk == meta
    assert on_disk["sha256"] == digest and on_disk["run_id"] == run_id
    assert on_disk["experiment"] == "exp" and on_disk["tracking_uri"] == store
    assert on_disk["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert on_disk["registered_model"] is None and on_disk["version"] is None
    assert tracking.read_model_sidecar(model) == on_disk
    assert _artifacts(store, run_id, "model") == ["model/best.onnx"]
    assert _client(store).get_run(run_id).data.tags["model_file.best.onnx.sha256"] == digest

    model.write_bytes(b"other bytes")                               # the file is no longer the logged one
    assert tracking.read_model_sidecar(model) is None
    assert tracking.read_model_sidecar(model, verify=False)["run_id"] == run_id
    assert tracking.read_model_sidecar(repo / "train.py") is None   # no sidecar at all


# -- offline fallback ---------------------------------------------------------------------------------------------
def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_refusing_connections_goes_offline_fast(mlflow_mod, repo, tmp_path):
    t0 = time.monotonic()
    with tracking.tracked_run("exp", tracking_uri=f"http://127.0.0.1:{_closed_port()}", root=repo) as run:
        assert run.mode == "offline"
        run.log_metrics({"loss": 0.5}, step=1)
        run_id, uri = run.run_id, run.tracking_uri
    assert time.monotonic() - t0 < 15
    assert uri == f"sqlite:///{(tmp_path / 'offline' / 'mlflow.db').resolve()}"
    r = _client(uri).get_run(run_id)
    assert r.data.tags["tracking.mode"] == "offline" and "unreachable" in r.data.tags["tracking.offline_reason"]
    assert r.data.metrics == {"loss": 0.5}


def test_server_that_never_answers_times_out_into_offline(mlflow_mod, repo):
    """A server that accepts the connection and then hangs: the probe's timeout, not MLflow's minutes of retries."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    conns = []
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conns.append(srv.accept()[0])                       # hold it open, never reply
            except OSError:
                pass

    th = threading.Thread(target=accept, daemon=True)
    th.start()
    try:
        t0 = time.monotonic()
        with tracking.tracked_run("exp", tracking_uri=f"http://127.0.0.1:{srv.getsockname()[1]}", root=repo,
                                  probe_timeout=1.0) as run:
            assert run.mode == "offline"
        assert time.monotonic() - t0 < 15
    finally:
        stop.set()
        th.join()
        for c in conns:
            c.close()
        srv.close()


def test_no_uri_configured_goes_offline(mlflow_mod, repo):
    with tracking.tracked_run("exp", root=repo) as run:
        assert run.mode == "offline" and run.active


# -- mlflow not installed -----------------------------------------------------------------------------------------
def test_without_mlflow_is_a_noop(repo, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "mlflow", None)                # `import mlflow` raises ImportError
    monkeypatch.setitem(sys.modules, "mlflow.tracking", None)
    model = repo / "best.onnx"
    model.write_bytes(b"bytes")
    with caplog.at_level(logging.WARNING, logger="mtmc.tracking"):
        with tracking.tracked_run("exp", "r", params={"a": 1}, data=["args.yaml.dvc"], lockfile="uv.lock",
                                  root=repo) as run:
            assert not run.active and run.mode == "disabled" and run.run_id is None
            run.log_params({"b": 2})
            run.log_metrics({"m": 1.0})
            run.set_tags({"t": "x"})
            run.log_artifact(model)
            tracking.configure_ultralytics(run)
            meta = tracking.log_model_file(model)
    assert "mlflow is not installed" in caplog.text
    assert meta["run_id"] is None and meta["sha256"] == hashlib.sha256(b"bytes").hexdigest()
    assert not (repo / "best.onnx.mlflow.json").exists()            # no sidecar pointing at no run
    assert "MLFLOW_RUN_ID" not in os.environ


# -- nesting / child processes ------------------------------------------------------------------------------------
def test_inherited_run_id_makes_a_child_run_not_a_resume(mlflow_mod, repo, store, monkeypatch):
    """`finetune --export` runs the export script as a subprocess that inherits MLFLOW_RUN_ID from
    configure_ultralytics: the export's tracked_run must open its own run, as a child of the training run."""
    with tracking.tracked_run("train", tracking_uri=store, root=repo) as parent:
        parent_id = parent.run_id
    monkeypatch.setenv("MLFLOW_RUN_ID", parent_id)
    with tracking.tracked_run("export", tracking_uri=store, root=repo) as child:
        assert child.run_id != parent_id
        child_id = child.run_id
    assert _client(store).get_run(child_id).data.tags["mlflow.parentRunId"] == parent_id
    assert os.environ["MLFLOW_RUN_ID"] == parent_id                  # restored for the caller


def test_nested_in_process_run(mlflow_mod, repo, store):
    with tracking.tracked_run("train", tracking_uri=store, root=repo) as outer:
        with tracking.tracked_run("train", "export", tracking_uri=store, root=repo) as inner:
            assert inner.run_id != outer.run_id
            inner_id = inner.run_id
        assert mlflow_mod.active_run().info.run_id == outer.run_id
    assert _client(store).get_run(inner_id).data.tags["mlflow.parentRunId"] == outer.run_id


# -- Ultralytics' built-in callback -------------------------------------------------------------------------------
def _ultralytics_callback_path() -> Path:
    spec = importlib.util.find_spec("ultralytics")                  # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("ultralytics is not installed")
    path = Path(spec.submodule_search_locations[0]) / "utils" / "callbacks" / "mlflow.py"
    if not path.is_file():
        pytest.skip(f"no MLflow callback at {path}")
    return path


STUB = textwrap.dedent('''
    import importlib.util, logging, os, sys, types
    from pathlib import Path

    def load_callback(path, runs_dir):
        """The real callback module, with a stub ultralytics.utils (no torch import)."""
        utils = types.ModuleType("ultralytics.utils")
        utils.LOGGER = logging.getLogger("ultralytics")
        utils.RUNS_DIR = Path(runs_dir)
        utils.SETTINGS = {"mlflow": True}
        utils.TESTS_RUNNING = False
        utils.colorstr = lambda *a: a[-1]
        utils.env_bool = lambda name, default=False: (os.environ[name].strip().lower() in
                                                      {"1", "true", "yes", "on", "y", "t"}) if name in os.environ else default
        pkg = types.ModuleType("ultralytics")
        pkg.utils = utils
        sys.modules.setdefault("ultralytics", pkg)
        sys.modules["ultralytics.utils"] = utils
        spec = importlib.util.spec_from_file_location("ul_mlflow_callback", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    class Args(dict):
        __getattr__ = dict.get

    def fake_trainer(save_dir, project, name):
        save_dir = Path(save_dir)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"pt")
        (save_dir / "results.csv").write_text("epoch,loss\\n1,0.5\\n")
        t = types.SimpleNamespace()
        t.args = Args(project=project, name=name, epochs=2, batch=48)
        t.lr = {"lr/pg0": 0.01}
        t.tloss = 0.5
        t.label_loss_items = lambda loss, prefix="train": {f"{prefix}/box_loss": loss}
        t.metrics = {"metrics/mAP50-95(B)": 0.9}
        t.epoch = 0
        t.best = save_dir / "weights" / "best.pt"
        t.save_dir = save_dir
        return t

    def train(cb, trainer, epochs=2):
        cb.on_pretrain_routine_end(trainer)
        for e in range(epochs):
            trainer.epoch = e
            cb.on_train_epoch_end(trainer)
            cb.on_fit_epoch_end(trainer)
        cb.on_train_end(trainer)
''')


@pytest.fixture
def ul(tmp_path, monkeypatch):
    path = _ultralytics_callback_path()
    ns: dict = {}
    exec(compile(STUB, "ul_stub", "exec"), ns)
    monkeypatch.setitem(sys.modules, "ultralytics", sys.modules.get("ultralytics") or types.ModuleType("ultralytics"))
    monkeypatch.setitem(sys.modules, "ultralytics.utils", types.ModuleType("ultralytics.utils"))
    cb = ns["load_callback"](path, tmp_path / "ul_runs")
    return types.SimpleNamespace(cb=cb, ns=ns, path=path)


def test_mlflow_ultralytics_callback_logs_into_the_helper_run_and_leaves_it_open(mlflow_mod, ul, repo, store,
                                                                                tmp_path):
    trainer = ul.ns["fake_trainer"](tmp_path / "save", project=str(tmp_path / "abs" / "project"), name="ul_name")
    with tracking.tracked_run("vehicle-detector", "helper_name", tracking_uri=store, root=repo) as run:
        tracking.configure_ultralytics(run)
        ul.ns["train"](ul.cb, trainer)
        assert trainer._mlflow_active and not trainer._mlflow_started_run   # it reused our run
        active = mlflow_mod.active_run()
        assert active is not None and active.info.run_id == run.run_id    # not ended by on_train_end
        run.log_metrics({"final/mAP50-95": 0.91})                         # the script's own final val
        run_id = run.run_id
    runs = _runs(store, "vehicle-detector")
    assert [r.info.run_id for r in runs] == [run_id]                      # exactly one run
    r = runs[0]
    assert r.info.status == "FINISHED"
    assert r.data.params["batch"] == "48"                                  # trainer.args, by the callback
    assert r.data.metrics["metrics/mAP50-95B"] == 0.9                      # per-epoch, by the callback
    assert r.data.metrics["final/mAP50-95"] == 0.91
    names = {e.name for e in _client(store).search_experiments()}
    assert names <= {"Default", "vehicle-detector"}                       # no experiment named after args.project
    assert "weights" in _artifacts(store, run_id)


def test_mlflow_ultralytics_callback_in_a_child_process_resumes_the_run(mlflow_mod, ul, repo, store, tmp_path):
    """Multi-GPU: Ultralytics re-runs training in a subprocess with no active run. The callback there resumes the
    helper's run (MLFLOW_RUN_ID) instead of opening a second one, and does not end it itself (MLFLOW_KEEP_RUN_ACTIVE).

    MLflow's own atexit hook in that child (mlflow/tracking/fluent.py, `_safe_end_run`) still marks the run
    FINISHED when the child exits. The parent keeps logging into it, and the helper sets the final status when its
    block exits: here FAILED, to show the parent's status is the one that sticks.
    """
    child = tmp_path / "child.py"
    child.write_text(STUB + textwrap.dedent(f'''
        cb = load_callback({str(ul.path)!r}, {str(tmp_path / "ul_runs")!r})
        t = fake_trainer({str(tmp_path / "save")!r}, project="p", name="n")
        train(cb, t)
        assert t._mlflow_started_run, "the child had no active run, so the callback started (resumed) one"
        import mlflow
        assert mlflow.active_run() is not None, "the callback ended the run"
        print("CHILD_RUN", mlflow.active_run().info.run_id)
    '''))
    with pytest.raises(RuntimeError, match="after DDP"):
        with tracking.tracked_run("vehicle-detector", "helper_name", tracking_uri=store, root=repo) as run:
            tracking.configure_ultralytics(run)
            out = subprocess.run([sys.executable, str(child)], capture_output=True, text=True, timeout=300,
                                 env={**os.environ, "PYTHONPATH": ""})
            assert out.returncode == 0, out.stderr[-3000:]
            assert f"CHILD_RUN {run.run_id}" in out.stdout
            assert _client(store).get_run(run.run_id).info.status == "FINISHED"   # the child's atexit hook
            run.log_metrics({"final/mAP50-95": 0.91})                            # the parent still logs into it
            run_id = run.run_id
            raise RuntimeError("after DDP")
    runs = _runs(store, "vehicle-detector")
    assert [r.info.run_id for r in runs] == [run_id]                              # one run, child included
    assert runs[0].info.status == "FAILED"                                        # the helper's status wins
    assert runs[0].data.metrics["metrics/mAP50-95B"] == 0.9                       # logged by the child
    assert runs[0].data.metrics["final/mAP50-95"] == 0.91
