"""The finetune and export scripts' MLflow runs (sub-tasks 4 and 5 of the MLflow story; spike D4, D8, D9).

No torch, no Ultralytics training, no server: runs go to a temp sqlite store. The scripts get a fake `ultralytics`
package whose `YOLO.train` builds a trainer the way Ultralytics' BaseTrainer does (the model's callbacks, then the
integration callbacks) and fires the same events. The MLflow integration callback in it is the **real** source file of
the installed ultralytics (ultralytics/utils/callbacks/mlflow.py, loaded with a stub `ultralytics.utils`, as in
tests/test_mlflow_tracking.py) when ultralytics is installed, and otherwise a simulation of that file's behaviour in
8.4.150: reuse the active run or start one, log trainer.args as params, per-epoch metrics, the weights dir and save_dir
files as artifacts, end the run only if it started it and MLFLOW_KEEP_RUN_ACTIVE is unset.

The export runs as the real `python -m`-style subprocess from `finetune_vehicle_detector --export` (a small launcher
that stubs only the torch export itself), so the child run is made from the inherited environment, as in production.
"""
from __future__ import annotations

import collections
import importlib
import importlib.util
import json
import logging
import os
import re
import socket
import sys
import textwrap
import types
from pathlib import Path

import pytest

from mtmc import tracking

REPO = Path(__file__).resolve().parents[1]
DIR_MD5 = "ad5ed68445f31c338d2c5fdf8c8f4283.dir"
PLATE_MD5 = "7371b10e31c0964d765bb931d76498e2.dir"
CALLBACK_MODULE = "ultralytics.utils.callbacks.mlflow"

mlflow = pytest.importorskip("mlflow")


# -- the fake Ultralytics -----------------------------------------------------------------------------------------
def _real_callback_path() -> Path | None:
    spec = importlib.util.find_spec("ultralytics") if "ultralytics" not in sys.modules else None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(spec.submodule_search_locations[0]) / "utils" / "callbacks" / "mlflow.py"
    return path if path.is_file() else None


SIMULATED_CALLBACK = textwrap.dedent('''
    """Simulation of ultralytics/utils/callbacks/mlflow.py (8.4.150) for machines without ultralytics."""
    import os
    import mlflow
    from ultralytics.utils import SETTINGS, env_bool

    def sanitize_dict(x):
        return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}

    def on_pretrain_routine_end(trainer):
        if SETTINGS["mlflow"] is not True:
            return
        trainer._mlflow_active = trainer._mlflow_started_run = False
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "runs/mlflow")
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project)
        run = mlflow.active_run()
        if run is None:
            run = mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN") or trainer.args.name)
            trainer._mlflow_started_run = True
        mlflow.log_params(dict(trainer.args))
        trainer._mlflow_active = True

    def on_train_epoch_end(trainer):
        if getattr(trainer, "_mlflow_active", False):
            mlflow.log_metrics({**sanitize_dict(trainer.lr),
                                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train"))},
                               step=trainer.epoch)

    def on_fit_epoch_end(trainer):
        if getattr(trainer, "_mlflow_active", False):
            mlflow.log_metrics(sanitize_dict(trainer.metrics), step=trainer.epoch)

    def on_train_end(trainer, status="FINISHED"):
        if status == "FINISHED" and getattr(trainer, "_mlflow_active", False):
            mlflow.log_artifact(str(trainer.best.parent))
            for f in trainer.save_dir.glob("*"):
                if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                    mlflow.log_artifact(str(f))
        if getattr(trainer, "_mlflow_started_run", False):
            trainer._mlflow_started_run = False
            if not env_bool("MLFLOW_KEEP_RUN_ACTIVE"):
                mlflow.end_run(status=status)

    callbacks = {"on_pretrain_routine_end": on_pretrain_routine_end, "on_train_epoch_end": on_train_epoch_end,
                 "on_fit_epoch_end": on_fit_epoch_end, "on_train_end": on_train_end}
''')


class Args(dict):
    __getattr__ = dict.get


class FakeYOLO:
    """`ultralytics.YOLO` for the scripts: train() writes a run dir and fires the trainer's callbacks; val() returns
    fixed metrics. Records what happened for the assertions."""
    log: list[dict] = []

    def __init__(self, weights, task=None):
        self.weights = str(weights)
        self.callbacks = collections.defaultdict(list)

    def add_callback(self, event, fn):
        self.callbacks[event].append(fn)

    def train(self, **kw):
        cb = sys.modules[CALLBACK_MODULE]
        save_dir = Path(kw["project"]) / kw["name"]
        assert save_dir.is_absolute(), "the scripts must pass an absolute project (Ultralytics nests relative ones)"
        trainer = types.SimpleNamespace(
            args=Args(model=self.weights, task="detect", mode="train", **kw),
            callbacks=collections.defaultdict(list, {e: list(f) for e, f in self.callbacks.items()}),
            lr={"lr/pg0": 0.01}, tloss=0.5, epoch=0, save_dir=save_dir, best=save_dir / "weights" / "best.pt",
            label_loss_items=lambda loss, prefix="train": {f"{prefix}/box_loss": loss},
        )
        for event, fn in getattr(cb, "callbacks", {}).items():            # BaseTrainer.__init__: integrations
            trainer.callbacks[event].append(fn)

        def fire(event):
            for fn in list(trainer.callbacks.get(event, [])):
                fn(trainer)

        fire("on_pretrain_routine_start")
        fire("on_pretrain_routine_end")
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        for e in range(kw["epochs"]):
            trainer.epoch = e
            if kw.get("save_period", 0) > 0 and e % kw["save_period"] == 0:     # trainer.save_model, 8.4.150 :803
                (save_dir / "weights" / f"epoch{e}.pt").write_bytes(b"checkpoint")
            trainer.metrics = {"metrics/mAP50-95(B)": 0.5 + e / 100, "val/box_loss": 1.0 - e / 10}
            fire("on_train_epoch_end")
            fire("on_fit_epoch_end")
        (save_dir / "weights" / "best.pt").write_bytes(b"best-weights")
        (save_dir / "weights" / "last.pt").write_bytes(b"last-weights")
        (save_dir / "args.yaml").write_text("epochs: 2\n")
        (save_dir / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n0,0.5\n1,0.51\n")
        (save_dir / "results.png").write_bytes(b"png")
        fire("on_train_end")
        fire("teardown")
        active = mlflow.active_run()
        FakeYOLO.log.append({"event": "train", "mlflow_active": getattr(trainer, "_mlflow_active", None),
                             "run_after_train": active.info.run_id if active else None})
        return None

    def val(self, **kw):
        active = mlflow.active_run()
        FakeYOLO.log.append({"event": "val", "weights": self.weights,
                             "run_during_val": active.info.run_id if active else None})
        box = types.SimpleNamespace(map50=0.81, map=0.61, mp=0.71, mr=0.66, maps=[0.61], ap_class_index=[0])
        return types.SimpleNamespace(box=box, names={0: "vehicle"})


def _install_fake_ultralytics(monkeypatch, tmp_path, want: str) -> str:
    """Put the fake package in sys.modules; returns which MLflow callback it uses ("real <path>" or "simulated")."""
    real = _real_callback_path() if want == "real" else None
    if want == "real" and real is None:
        pytest.skip("ultralytics is not installed: the simulated callback variant covers this")
    pkg = types.ModuleType("ultralytics")
    pkg.__path__ = []
    pkg.YOLO = FakeYOLO
    utils = types.ModuleType("ultralytics.utils")
    utils.__path__ = []
    utils.LOGGER = logging.getLogger("ultralytics")
    utils.RUNS_DIR = tmp_path / "ul_runs"
    utils.SETTINGS = {"mlflow": True}
    utils.TESTS_RUNNING = False
    utils.colorstr = lambda *a: a[-1]
    utils.env_bool = lambda name, default=False: (os.environ[name].strip().lower() in
                                                  {"1", "true", "yes", "on", "y", "t"}) if name in os.environ else default
    callbacks_pkg = types.ModuleType("ultralytics.utils.callbacks")
    callbacks_pkg.__path__ = []
    pkg.utils, utils.callbacks = utils, callbacks_pkg
    for name, mod in (("ultralytics", pkg), ("ultralytics.utils", utils), ("ultralytics.utils.callbacks", callbacks_pkg)):
        monkeypatch.setitem(sys.modules, name, mod)
    if real is not None:
        spec = importlib.util.spec_from_file_location(CALLBACK_MODULE, real)
        cb = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, CALLBACK_MODULE, cb)
        spec.loader.exec_module(cb)
        source = f"real {real}"
    else:
        cb = types.ModuleType(CALLBACK_MODULE)
        monkeypatch.setitem(sys.modules, CALLBACK_MODULE, cb)
        exec(compile(SIMULATED_CALLBACK, "simulated_ultralytics_mlflow_callback", "exec"), cb.__dict__)
        source = "simulated"
    for script in ("mtmc.finetune_vehicle_detector", "mtmc.finetune_plate_detector", "mtmc.export_vehicle_onnx"):
        monkeypatch.delitem(sys.modules, script, raising=False)            # re-imported against the fake
    FakeYOLO.log = []
    return source


# -- fixtures -----------------------------------------------------------------------------------------------------
@pytest.fixture
def work(tmp_path, monkeypatch):
    """Working directory the scripts run from: the two datasets (pointer + pulled data.yaml) and base weights."""
    w = tmp_path / "work"
    ds = w / "products" / "el-mohandes" / "datasets"
    for name, md5 in (("vehicle_detection_1cls", DIR_MD5), ("plate_detection", PLATE_MD5)):
        (ds / name).mkdir(parents=True)
        (ds / f"{name}.dvc").write_text(f"outs:\n- md5: {md5}\n  size: 1\n  nfiles: 1\n  hash: md5\n  path: {name}\n")
        (ds / name / "data.yaml").write_text("nc: 1\nnames: [vehicle]\n")
    (w / "yolo11s.pt").write_bytes(b"base-weights")
    (w / "plate_base.pt").write_bytes(b"plate-base-weights")
    monkeypatch.chdir(w)
    return w


@pytest.fixture
def env(tmp_path, monkeypatch, mlflow_store):
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    for k in tracking._ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    store = mlflow_store                # tests/conftest.py: this test's own copy of a migrated store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store)
    yield types.SimpleNamespace(store=store, db=tmp_path / "store" / "mlflow.db", offline=tmp_path / "offline")
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture(params=["real", "simulated"])
def ul(request, env, work, tmp_path, monkeypatch):
    return _install_fake_ultralytics(monkeypatch, tmp_path, request.param)


@pytest.fixture
def uploads(monkeypatch):
    """Every local path handed to mlflow.log_artifact (the callback's and the scripts' uploads)."""
    seen: list[str] = []
    orig = mlflow.log_artifact

    def spy(local_path, artifact_path=None, *a, **k):
        seen.append(str(local_path))
        return orig(local_path, artifact_path, *a, **k)

    monkeypatch.setattr(mlflow, "log_artifact", spy)
    return seen


def _client(uri):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=uri)


def _runs(uri, experiment):
    c = _client(uri)
    exp = c.get_experiment_by_name(experiment)
    return c.search_runs([exp.experiment_id]) if exp else []


def _artifacts(uri, run_id, path=None):
    out = []
    for a in _client(uri).list_artifacts(run_id, path):
        out += _artifacts(uri, run_id, a.path) if a.is_dir else [a.path]
    return out


def _vehicle(*argv):
    return importlib.import_module("mtmc.finetune_vehicle_detector").main(
        ["--data", "products/el-mohandes/datasets/vehicle_detection_1cls/data.yaml", "--model", "yolo11s.pt",
         "--epochs", "2", "--name", "t", "--device", "cpu", *argv])


# -- finetune -----------------------------------------------------------------------------------------------------
def test_vehicle_finetune_one_run_with_lineage_params_metrics_and_sidecar(ul, env, work, uploads, capsys):
    _vehicle()
    runs = _runs(env.store, "vehicle-detector")
    assert len(runs) == 1, "one run per finetune"
    r = runs[0]
    rid = r.info.run_id
    assert r.info.status == "FINISHED"
    # not ended early: Ultralytics' on_train_end left it open, and the final val ran inside it
    train, val = FakeYOLO.log[0], FakeYOLO.log[1]
    assert train["mlflow_active"] is True and train["run_after_train"] == rid
    assert val["event"] == "val" and val["run_during_val"] == rid and val["weights"].endswith("weights/best.pt")

    p, t, m = r.data.params, r.data.tags, r.data.metrics
    assert p["cli.epochs"] == "2" and p["cli.batch"] == "48" and p["cli.no_track"] == "False" and p["cli.cache"] == "True"
    assert p["train.optimizer"] == "SGD" and p["train.mosaic"] == "1.0" and p["train.save_period"] == "10"
    assert p["epochs"] == "2" and p["cache"] == "True"            # Ultralytics' own trainer.args, no clash
    assert t["capability"] == "vehicle-detector" and t["source"] == "trained"
    assert len(t["git.commit"]) == 40
    assert t["data.vehicle_detection_1cls.dvc_md5"] == DIR_MD5
    assert t["data.vehicle_detection_1cls.dvc_path"].endswith("datasets/vehicle_detection_1cls.dvc")
    assert t["env.lockfile"] == "tools/mtmc-research/uv.lock"     # the research tools' lock (registry refuses without)
    assert t["env.lockfile_sha256"] == __import__("hashlib").sha256(
        (REPO / "tools/mtmc-research/uv.lock").read_bytes()).hexdigest()
    assert t["base_weights.sha256"] == __import__("hashlib").sha256(b"base-weights").hexdigest()
    assert t["eval.val.set"] == "vehicle-detector/vehicle_detection_1cls:val"
    assert m["eval/val/mAP50"] == pytest.approx(0.81) and m["eval/val/mAP50-95"] == pytest.approx(0.61)
    assert m["eval/val/precision"] == pytest.approx(0.71) and m["eval/val/recall"] == pytest.approx(0.66)
    assert m["eval/val/mAP50-95/vehicle"] == pytest.approx(0.61)
    history = _client(env.store).get_metric_history(rid, "metrics/mAP50-95B")
    assert [h.step for h in history] == [0, 1]                    # per epoch, by the callback

    arts = _artifacts(env.store, rid)
    assert {"weights/best.pt", "weights/last.pt", "args.yaml", "results.csv", "results.png"} <= set(arts)
    assert not any(a.startswith("best.pt") for a in arts)
    # best.pt uploaded exactly once (end of training); log_model_file only tags it and writes the sidecar
    assert len([u for u in uploads if u.endswith("/best.pt")]) == 1, uploads
    assert not [u for u in uploads if u.endswith("/weights")], "the whole weights/ folder was uploaded"

    best = work / "products/el-mohandes/runs/vehicle_detector/t/weights/best.pt"
    meta = tracking.read_model_sidecar(best)
    assert meta and meta["run_id"] == rid and meta["artifact_path"] == "weights"
    assert t["model_file.best.pt.sha256"] == meta["sha256"]

    out = capsys.readouterr().out
    assert f"MLflow run {rid}" in out
    assert f"python -m mtmc.tracking.registry --tracking-uri {env.store} register --model {best.relative_to(work)}" \
        in out
    print("ultralytics callback used:", ul)


def test_periodic_checkpoints_never_reach_the_run(ul, env, work):
    """save_period stays in the recipe (epoch<N>.pt on disk) but only best.pt and last.pt are uploaded."""
    _vehicle("--epochs", "12")                                         # save_period=10: epoch0.pt and epoch10.pt
    weights = work / "products/el-mohandes/runs/vehicle_detector/t/weights"
    assert {"epoch0.pt", "epoch10.pt"} <= {p.name for p in weights.iterdir()}, "the fake must write checkpoints"
    (r,) = _runs(env.store, "vehicle-detector")
    arts = _artifacts(env.store, r.info.run_id)
    assert not [a for a in arts if re.search(r"(^|/)epoch\d+\.pt$", a)], arts
    assert [a for a in arts if a.startswith("weights/")] == ["weights/best.pt", "weights/last.pt"]
    assert {"args.yaml", "results.csv", "results.png"} <= set(arts)


def test_plate_finetune_one_run(ul, env, work, capsys):
    importlib.import_module("mtmc.finetune_plate_detector").main(
        ["--data", "products/el-mohandes/datasets/plate_detection/data.yaml", "--model", "plate_base.pt",
         "--epochs", "3", "--name", "p", "--device", "cpu"])
    runs = _runs(env.store, "plate-detector")
    assert len(runs) == 1
    r = runs[0]
    assert r.info.status == "FINISHED"
    assert r.data.tags["data.plate_detection.dvc_md5"] == PLATE_MD5 and r.data.tags["capability"] == "plate-detector"
    assert r.data.params["cli.patience"] == "20" and r.data.params["train.fliplr"] == "0.5"
    assert "eval/val/mAP50-95" in r.data.metrics
    assert len(_client(env.store).get_metric_history(r.info.run_id, "val/box_loss")) == 3
    best = work / "products/el-mohandes/runs/finetune/p/weights/best.pt"   # where the script looks, not runs/detect/
    assert tracking.read_model_sidecar(best)["run_id"] == r.info.run_id
    assert _runs(env.store, "vehicle-detector") == []


@pytest.mark.mlflow_store_absent
def test_no_track_creates_nothing(ul, env, work, tmp_path, capsys):
    _vehicle("--no-track")
    assert FakeYOLO.log[0]["mlflow_active"] is None, "Ultralytics' MLflow callback must not run with --no-track"
    assert not env.db.exists() and not env.offline.exists() and not (work / "runs").exists()
    assert not (tmp_path / "ul_runs").exists() and not (work / "mlruns").exists()
    assert not list(work.rglob("*.mlflow.json"))
    assert "tracking: off" in capsys.readouterr().out


# -- export -------------------------------------------------------------------------------------------------------
# Only the torch export is replaced (no torch here); everything around it (tracking, sidecars) is the real script.
FAKE_EXPORT = textwrap.dedent('''
    import shutil, types
    from pathlib import Path

    def export_one(mod, weights, out_dir, size, opset, dynamic, simplify):
        local = (out_dir / weights.name).resolve()
        if local != weights.resolve():
            shutil.copyfile(weights, local)
        onnx = local.with_suffix(".onnx")
        onnx.write_bytes(b"onnx:" + weights.read_bytes() + f":{size}:{opset}:{dynamic}:{simplify}".encode())
        Path("labels.txt").write_text("vehicle\\n")       # the upstream exporter writes it into cwd (= out_dir)
        return onnx

    def check_upstream(path, simplify):
        pass

    def load_upstream(path):
        return types.SimpleNamespace(torch=types.SimpleNamespace(onnx=types.SimpleNamespace(export=lambda *a, **k: None)))
''')


def _fake_export_ns() -> dict:
    ns: dict = {}
    exec(compile(FAKE_EXPORT, "fake_export", "exec"), ns)
    return ns


@pytest.fixture
def launcher(tmp_path):
    """What `finetune_vehicle_detector --export` runs as its subprocess: the export script with the fake export."""
    p = tmp_path / "export_launcher.py"
    p.write_text(f"import sys\nsys.path.insert(0, {str(REPO)!r})\n" + FAKE_EXPORT + textwrap.dedent('''
        import mtmc.export_vehicle_onnx as e
        e.export_one, e.check_upstream, e.load_upstream = export_one, check_upstream, load_upstream
        e.main(sys.argv[1:])
    '''))
    return p


def _patch_export_in_process(monkeypatch):
    e = importlib.import_module("mtmc.export_vehicle_onnx")
    ns = _fake_export_ns()
    for name in ("export_one", "check_upstream", "load_upstream"):
        monkeypatch.setattr(e, name, ns[name])
    return e


def test_vehicle_finetune_export_is_a_child_run_with_source_run_id(ul, env, work, launcher, monkeypatch, capsys):
    v = importlib.import_module("mtmc.finetune_vehicle_detector")
    monkeypatch.setattr(v, "EXPORT_CMD", [sys.executable, str(launcher)])
    _vehicle("--export", "--export-dir", "export")
    runs = {r.info.run_name: r for r in _runs(env.store, "vehicle-detector")}
    assert set(runs) == {"t", "export-best-1280"}
    train, child = runs["t"], runs["export-best-1280"]
    tid, cid = train.info.run_id, child.info.run_id
    assert train.info.status == "FINISHED" and child.info.status == "FINISHED"
    assert child.data.tags["mlflow.parentRunId"] == tid                     # nested under the training run
    assert child.data.tags["source_run_id"] == tid                          # from best.pt's sidecar
    assert child.data.tags["source_run.found"] == "true"
    assert child.data.tags["source_weights_sha256"] == __import__("hashlib").sha256(b"best-weights").hexdigest()
    assert child.data.tags["exporter.path"] == "perception/models/DeepStream-Yolo/utils/export_yolo11.py"
    assert child.data.tags["exporter.matches_pin"] == "True"
    assert child.data.tags["data.vehicle_detection_1cls.dvc_md5"] == DIR_MD5  # inherited lineage
    assert child.data.metrics["source_run/eval/val/mAP50-95"] == pytest.approx(0.61)
    p = child.data.params
    assert (p["size"], p["opset"], p["dynamic"], p["simplify"], p["batch"]) == ("1280", "17", "True", "True", "1")
    assert {"onnx/best.onnx", "onnx/labels.txt"} <= set(_artifacts(env.store, cid))
    assert not any(a.endswith(".pt") for a in _artifacts(env.store, cid))  # the .pt is the training run's
    exp = work / "export"
    assert tracking.read_model_sidecar(exp / "best.onnx")["run_id"] == cid
    assert tracking.read_model_sidecar(exp / "labels.txt")["run_id"] == cid
    assert tracking.read_model_sidecar(exp / "best.pt")["run_id"] == tid    # the copied weights keep their run
    out = capsys.readouterr().out
    assert "a child of this run" in out and "register --model export/best.onnx" in out


def _source_run(env, work, data=True):
    """A finished training-like run that logged weights.pt with log_model_file (sidecar written)."""
    w = work / "weights.pt"
    w.write_bytes(b"trained-weights")
    with tracking.tracked_run("vehicle-detector", "src",
                              data={"vehicle_detection_1cls": work / "products/el-mohandes/datasets/"
                                                                     "vehicle_detection_1cls/data.yaml"} if data else None,
                              tracking_uri=env.store) as run:
        run.log_metrics({"eval/val/mAP50": 0.9})
        tracking.log_model_file(w, "weights", run=run)
    return w, run.run_id


def test_standalone_export_with_sidecar_is_a_top_level_run(env, work, monkeypatch):
    w, src = _source_run(env, work)
    e = _patch_export_in_process(monkeypatch)
    e.main(["--weights", str(w), "--out-dir", "out", "--size", "640", "--static-batch", "--no-simplify"])
    child = [r for r in _runs(env.store, "vehicle-detector") if r.info.run_id != src][0]
    assert "mlflow.parentRunId" not in child.data.tags
    assert child.data.tags["source_run_id"] == src and child.data.tags["source"] == "trained"
    assert (child.data.params["size"], child.data.params["dynamic"], child.data.params["simplify"]) == \
        ("640", "False", "False")
    assert child.data.metrics["source_run/eval/val/mAP50"] == pytest.approx(0.9)
    assert tracking.read_model_sidecar(work / "out" / "weights.onnx")["run_id"] == child.info.run_id


def test_standalone_export_without_sidecar_tags_unknown_and_warns(env, work, monkeypatch, capsys):
    w = work / "yolo11s.pt"                                                   # third-party weights: no sidecar
    e = _patch_export_in_process(monkeypatch)
    e.main(["--weights", str(w), "--out-dir", "out"])
    (run,) = _runs(env.store, "vehicle-detector")
    assert run.data.tags["source_run_id"] == "unknown" and "source" not in run.data.tags
    assert "source_run.found" not in run.data.tags
    assert "source_run_id=unknown" in capsys.readouterr().err
    assert tracking.read_model_sidecar(work / "out" / "yolo11s.onnx")["run_id"] == run.info.run_id
    assert not (work / "out" / "yolo11s.pt.mlflow.json").exists()


def test_export_no_track_creates_nothing(env, work, monkeypatch):
    w, _ = _source_run(env, work)
    before = len(_runs(env.store, "vehicle-detector"))
    e = _patch_export_in_process(monkeypatch)
    e.main(["--weights", str(w), "--out-dir", "out", "--no-track"])
    assert len(_runs(env.store, "vehicle-detector")) == before
    assert (work / "out" / "weights.onnx").exists() and not list((work / "out").glob("*.mlflow.json"))


# -- offline ------------------------------------------------------------------------------------------------------
def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_server_unreachable_goes_offline_and_the_script_finishes(ul, env, work, launcher, monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{_closed_port()}")
    v = importlib.import_module("mtmc.finetune_vehicle_detector")
    monkeypatch.setattr(v, "EXPORT_CMD", [sys.executable, str(launcher)])
    _vehicle("--export", "--export-dir", "export")
    offline = "sqlite:///" + str((env.offline / "mlflow.db").resolve())
    runs = {r.info.run_name: r for r in _runs(offline, "vehicle-detector")}
    assert set(runs) == {"t", "export-best-1280"}
    assert runs["t"].data.tags["tracking.mode"] == "offline" and runs["t"].info.status == "FINISHED"
    assert runs["export-best-1280"].data.tags["tracking.mode"] == "offline"   # the child inherited the offline store
    assert runs["export-best-1280"].data.tags["mlflow.parentRunId"] == runs["t"].info.run_id
    assert json.loads((work / "export" / "best.onnx.mlflow.json").read_text())["tracking_mode"] == "offline"
    out = capsys.readouterr().out
    assert "logged OFFLINE" in out and "python -m mtmc.tracking.registry" not in out   # registry refuses offline
