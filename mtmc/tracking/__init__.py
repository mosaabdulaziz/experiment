"""MLflow tracking for the training, export and evaluation scripts (sub-task 2 of the MLflow story).

MLflow is the record, not the server: nothing here is imported by the sidecar or the products, and tracking never
fails or blocks a run. Public API:

    from mtmc.tracking import tracked_run, configure_ultralytics, log_model_file, read_model_sidecar

    with tracked_run("vehicle-detector", run_name="vehicle_yolo11s_1cls_repro",
                     params={"epochs": 150, "batch": 48},
                     data={"vehicle_detection_1cls": "products/el-mohandes/datasets/vehicle_detection_1cls.dvc"},
                     lockfile="tools/mtmc-research/uv.lock") as run:
        configure_ultralytics(run)            # Ultralytics' callback logs into this run and leaves it open
        model.train(...)                      # per-epoch metrics, args, weights: the Ultralytics callback
        run.log_metrics({"final/mAP50-95": r.box.map})   # the script's own final val
        log_model_file(best_onnx)             # artifact + sha256 tag + <file>.mlflow.json beside it
    # the run ends here (FINISHED, or FAILED/KILLED if the block raised), never earlier

    meta = read_model_sidecar("products/el-mohandes/models/vehicle_1280_1cls/best.onnx")   # eval scripts (D9)

Every run is tagged with its lineage (`_lineage.py`): git.commit/branch/dirty (+ the dirty files as an artifact),
data.<name>.dvc_md5/dvc_path per dataset (read from the .dvc pointer, no dvc package), env.lockfile(_sha256),
env.python and the versions of torch/ultralytics/onnx/onnxruntime/mlflow, host, tracking.mode.

Stores (`_backend.py`): an http(s) MLFLOW_TRACKING_URI is probed first (GET /health, 3 s); unreachable, or no URI set
-> the offline sqlite store <repo>/mlruns-offline/ (tracking.mode=offline). mlflow not installed -> a no-op run
(`run.active` is False, every method does nothing) and a warning.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import _backend, _lineage
from ._lineage import sha256_file

__all__ = ["TrackedRun", "tracked_run", "configure_ultralytics", "log_model_file", "read_model_sidecar",
           "SIDECAR_SUFFIX"]

logger = logging.getLogger("mtmc.tracking")

SIDECAR_SUFFIX = ".mlflow.json"
SIDECAR_SCHEMA = 1

_active: list["TrackedRun"] = []          # innermost last; log_model_file's default run

# Process environment a run changes: ours (for Ultralytics and child processes) and MLflow's own, which
# mlflow.set_tracking_uri / set_experiment write into os.environ. All restored when the run ends.
_ENV_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_ID", "MLFLOW_RUN_ID", "MLFLOW_RUN",
             "MLFLOW_KEEP_RUN_ACTIVE")


class TrackedRun:
    """A handle on the run `tracked_run` opened. Every method is safe: a tracking failure is a warning, never an
    exception in the training script. With `active` False (mlflow missing, store unusable) they do nothing."""

    def __init__(self, experiment: str, run_name: str | None, mode: str, tracking_uri: str | None,
                 mlflow_mod: Any = None, run: Any = None, root: Path | None = None, git_commit: str | None = None):
        self.experiment = experiment
        self.run_name = run_name
        self.mode = mode                                  # server | offline | local | disabled
        self.tracking_uri = tracking_uri
        self.root = root
        self.git_commit = git_commit
        self._mlflow = mlflow_mod
        self._run = run
        self._env_restore: dict[str, str | None] = {}

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run is not None else None

    @property
    def experiment_id(self) -> str | None:
        return self._run.info.experiment_id if self._run is not None else None

    @property
    def url(self) -> str | None:
        """The run's page on the tracking server, or None when the run is not on a server (offline, local, none)."""
        if self._run is None or self.mode != "server" or not self.tracking_uri:
            return None
        return f"{self.tracking_uri.rstrip('/')}/#/experiments/{self.experiment_id}/runs/{self.run_id}"

    # -- safe wrappers ------------------------------------------------------------------------------------------
    def _call(self, what: str, *args, **kwargs):
        """mlflow.<what>(*args, **kwargs), or nothing without a run; a failure is logged, never raised."""
        if self._run is None:
            return None
        try:
            return getattr(self._mlflow, what)(*args, **kwargs)
        except Exception as exc:                          # tracking must never stop training
            logger.warning("tracking: %s failed for run %s (%s); training continues", what, self.run_id, exc)
            return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if params:
            self._call("log_params", {k: _param(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if metrics:
            self._call("log_metrics", {k: float(v) for k, v in metrics.items()}, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if tags:
            self._call("set_tags", {k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._call("log_artifact", str(path), artifact_path)

    def log_text(self, text: str, artifact_file: str) -> None:
        self._call("log_text", text, artifact_file)

    # -- process environment the run owns (restored when the run ends) -----------------------------------------
    def _setenv(self, key: str, value: str | None) -> None:
        if key not in self._env_restore:
            self._env_restore[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def _restore_env(self) -> None:
        _restore(self._env_restore)
        self._env_restore.clear()


def _param(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)
    return s if len(s) <= 6000 else s[:5997] + "..."   # MLflow's param value limit


def _restore(env: Mapping[str, str | None]) -> None:
    for key, old in env.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _import_mlflow():
    try:
        import mlflow                                     # noqa: PLC0415 (optional dependency, imported on use)
        from mlflow.tracking import MlflowClient          # noqa: PLC0415
    except ImportError as exc:
        logger.warning("tracking: mlflow is not installed (%s); this run is NOT tracked, training continues", exc)
        return None, None
    return mlflow, MlflowClient


def _open_experiment(mlflow, client_cls, backend: _backend.Backend, experiment: str) -> str:
    mlflow.set_tracking_uri(backend.tracking_uri)
    client = client_cls(tracking_uri=backend.tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(experiment,
                                                                   artifact_location=backend.artifact_root)
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


@contextlib.contextmanager
def tracked_run(experiment: str, run_name: str | None = None, params: Mapping[str, Any] | None = None,
                data: Mapping[str, str | Path] | list[str | Path] | None = None, lockfile: str | Path | None = None,
                tags: Mapping[str, Any] | None = None, *, tracking_uri: str | None = None,
                root: str | Path | None = None, probe_timeout: float = _backend.PROBE_TIMEOUT_S) -> Iterator[TrackedRun]:
    """Start a run with its lineage tags, yield a `TrackedRun`, end it when the block exits.

    The run ends FINISHED, or FAILED (KILLED on Ctrl-C) when the block raises; the exception propagates. Tracking
    errors never do. Inside another tracked run (same process) the new run is nested under it. In a child process
    started from a run configured for Ultralytics (MLFLOW_RUN_ID inherited, e.g. `--export` running the export
    script), the new run is a child of that run instead of silently resuming it.
    """
    root = Path(root) if root else _lineage.repo_root()
    git = _lineage.git_info(root)
    lineage = {**git.tags, **_lineage.data_tags(data, root), **_lineage.lockfile_tags(lockfile, root),
               **_lineage.env_tags()}

    mlflow, client_cls = _import_mlflow()
    if mlflow is None:
        handle = TrackedRun(experiment, run_name, "disabled", None, root=root, git_commit=git.tags.get("git.commit"))
        yield handle
        return

    env_before = {k: os.environ.get(k) for k in _ENV_KEYS}
    backend = _backend.resolve(root, tracking_uri, probe_timeout)
    try:
        _open_experiment(mlflow, client_cls, backend, experiment)
    except Exception as exc:
        if backend.mode == "offline":
            logger.warning("tracking: the offline store %s failed too (%s); this run is NOT tracked",
                           backend.tracking_uri, exc)
            _restore(env_before)
            yield TrackedRun(experiment, run_name, "disabled", None, root=root, git_commit=git.tags.get("git.commit"))
            return
        logger.warning("tracking: %s failed (%s); falling back to the offline store", backend.tracking_uri, exc)
        backend = _backend.sqlite_backend(_backend.offline_dir(root), "offline", str(exc))
        try:
            _open_experiment(mlflow, client_cls, backend, experiment)
        except Exception as exc2:
            logger.warning("tracking: offline store failed too (%s); this run is NOT tracked", exc2)
            _restore(env_before)
            yield TrackedRun(experiment, run_name, "disabled", None, root=root, git_commit=git.tags.get("git.commit"))
            return

    handle = TrackedRun(experiment, run_name, backend.mode, backend.tracking_uri, mlflow, None, root,
                        git.tags.get("git.commit"))
    handle._env_restore = dict(env_before)               # restore to the state before _open_experiment touched it
    # Ultralytics' callback and child processes read these (configure_ultralytics); restored at the end.
    handle._setenv("MLFLOW_TRACKING_URI", backend.tracking_uri)
    handle._setenv("MLFLOW_EXPERIMENT_NAME", experiment)
    inherited_parent = env_before["MLFLOW_RUN_ID"]
    handle._setenv("MLFLOW_RUN_ID", None)                 # never let start_run() resume another run by accident

    run_tags = {**lineage, "tracking.mode": backend.mode, **({"tracking.offline_reason": backend.reason}
                                                            if backend.mode == "offline" else {}),
                **{k: str(v) for k, v in (tags or {}).items()}}
    nested = mlflow.active_run() is not None
    if inherited_parent and not nested:
        run_tags["mlflow.parentRunId"] = inherited_parent
    try:
        handle._run = mlflow.start_run(run_name=run_name, nested=nested, tags=run_tags)
    except Exception as exc:
        logger.warning("tracking: could not start the run (%s); this run is NOT tracked", exc)
        handle._restore_env()
        handle.mode = "disabled"
        yield handle
        return

    if backend.mode == "offline":
        logger.warning("tracking: run %s is logged OFFLINE to %s", handle.run_id, backend.tracking_uri)
    handle.log_params(params or {})
    if git.dirty_files:
        handle.log_text("\n".join(git.dirty_files) + "\n", "lineage/git_dirty_files.txt")
    if lineage.get("env.lockfile") in ("missing", "none"):
        handle.log_text(_lineage.installed_packages(), "lineage/installed_packages.txt")

    _active.append(handle)
    status = "FINISHED"
    try:
        yield handle
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except BaseException:
        status = "FAILED"
        raise
    finally:
        _active.remove(handle)
        try:
            if mlflow.active_run() is not None and mlflow.active_run().info.run_id == handle.run_id:
                mlflow.end_run(status=status)
            else:                                         # someone ended or replaced it; close ours by id
                client_cls(tracking_uri=backend.tracking_uri).set_terminated(handle.run_id, status)
        except Exception as exc:
            logger.warning("tracking: could not end run %s (%s)", handle.run_id, exc)
        handle._restore_env()


def configure_ultralytics(run: TrackedRun) -> None:
    """Make Ultralytics' built-in MLflow callback log into `run` and leave its lifecycle to `tracked_run`.

    Verified against ultralytics 8.4.150 (the version in the research lock), ultralytics/utils/callbacks/mlflow.py:
    - on_pretrain_routine_end (:42-96) is a no-op unless `SETTINGS["mlflow"] is True` (:61). It calls
      `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI or RUNS_DIR/"mlflow")` (:66, :76) and
      `mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME or trainer.args.project)` (:70, :77), then **reuses the active
      run** if there is one and only starts its own otherwise (`mlflow.start_run(run_name=MLFLOW_RUN or args.name)`,
      :79-82, remembering that in `trainer._mlflow_started_run`). It logs `trainer.args` as params (:87).
    - on_train_end (:126-149) logs the weights dir and the plots/csv/yaml of save_dir as artifacts (:132-135), and
      **ends the run only if it started it** (:141), unless MLFLOW_KEEP_RUN_ACTIVE is truthy (:142, parsed by
      `env_bool`). Ultralytics calls it after its own final_eval (engine/trainer.py:675-678).
    - It also calls `mlflow.autolog()` (:78), which turns on MLflow's autologging for other installed libraries.
    So: the env vars below point the callback at this run's store and experiment, and the active run makes it log
    into this run instead of opening one. MLFLOW_RUN_ID and MLFLOW_KEEP_RUN_ACTIVE cover multi-GPU (DDP) training,
    where Ultralytics re-runs training in a subprocess (engine/trainer.py:227-246) that has no active run: there
    the callback's start_run() resumes this run by MLFLOW_RUN_ID and leaves it open. Everything is restored when the
    run ends.

    Known DDP limitation: MLflow's own atexit hook (mlflow 3.18, mlflow/tracking/fluent.py:763-768,
    `_safe_end_run`) marks the run FINISHED when that subprocess exits. The parent keeps logging into the same run
    and `tracked_run` sets the final status (FINISHED/FAILED/KILLED) at the end, so only the status shown *during*
    the parent's final eval is early. Single-GPU training (the finetune scripts' default `--device 0`) has no
    subprocess.
    """
    if not run.active:
        return
    run._setenv("MLFLOW_TRACKING_URI", run.tracking_uri)
    run._setenv("MLFLOW_EXPERIMENT_NAME", run.experiment)
    run._setenv("MLFLOW_RUN", run.run_name or run.run_id)
    run._setenv("MLFLOW_RUN_ID", run.run_id)
    run._setenv("MLFLOW_KEEP_RUN_ACTIVE", "true")
    try:
        from ultralytics.utils import SETTINGS            # noqa: PLC0415 (the finetune scripts import it anyway)
    except Exception:                                     # not installed here: nothing else to check
        return
    if SETTINGS.get("mlflow") is not True:
        logger.warning("tracking: Ultralytics' MLflow integration is off in its settings, so per-epoch metrics and "
                       "weights will NOT be logged. Turn it on with `yolo settings mlflow=True`.")


def log_model_file(path: str | Path, artifact_path: str | None = None, run: TrackedRun | None = None, *,
                   upload: bool = True) -> dict:
    """Log a produced model file to the run, tag its sha256, and write `<file>.mlflow.json` beside it (D9).

    `upload=False` when the file is already in the run's artifacts under `artifact_path` (Ultralytics' callback logs
    `weights/best.pt` itself): only the tag and the sidecar are written, so a large file is not uploaded twice.
    Returns the sidecar dict. With no active run (mlflow missing, nothing tracked) no sidecar is written: a sidecar
    without a run would send an eval script to a run that does not exist.
    """
    path = Path(path)
    run = run or (_active[-1] if _active else None)
    digest = sha256_file(path)
    meta = {
        "schema": SIDECAR_SCHEMA,
        "file": path.name,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_id": run.run_id if run else None,
        "experiment": run.experiment if run else None,
        "experiment_id": run.experiment_id if run else None,
        "tracking_uri": run.tracking_uri if run else None,
        "tracking_mode": run.mode if run else "disabled",
        "artifact_path": artifact_path,
        "git_commit": run.git_commit if run else None,
        "registered_model": None,                         # filled by the registry task
        "version": None,
    }
    if run is None or not run.active:
        logger.warning("tracking: %s is not logged (no active run); no %s written", path, SIDECAR_SUFFIX)
        return meta
    if upload:
        run.log_artifact(path, artifact_path)
    run.set_tags({f"model_file.{path.name}.sha256": digest})
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    kept = _registered_sidecar(sidecar, digest)
    if kept is not None:
        # One sidecar per file: the same bytes logged by another run (a labels.txt two exports share) keep the
        # version they already name. This run still has the file as an artifact and its sha256 tag.
        logger.warning("tracking: %s already names %s v%s (run %s); kept, not repointed to run %s: the file is "
                       "byte-identical and shared by several versions (MLFLOW.md, known gaps)", sidecar,
                       kept["registered_model"], kept.get("version"), kept.get("run_id"), run.run_id)
        return kept
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(sidecar)
    return meta


def _registered_sidecar(sidecar: Path, sha256: str) -> dict | None:
    """The sidecar's contents when it already names a registered version of these same bytes, else None."""
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if meta.get("registered_model") and meta.get("sha256") == sha256 else None


def read_model_sidecar(path: str | Path, verify: bool = True) -> dict | None:
    """The `<file>.mlflow.json` written for `path`, or None when there is none.

    With `verify` (the default) the file's sha256 must match the sidecar's: a mismatch means the file on disk is not
    the logged one, so None is returned (with a warning) and the eval script falls through to the next way of
    finding its run (D9: --run-id, sidecar, registry search by sha256, standalone run).
    """
    path = Path(path)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("tracking: unreadable %s (%s); ignoring it", sidecar, exc)
        return None
    if verify:
        if not path.is_file():
            logger.warning("tracking: %s has a sidecar but the file itself is missing", path)
            return None
        digest = sha256_file(path)
        if digest != meta.get("sha256"):
            logger.warning("tracking: %s does not match its sidecar (sha256 %s..., sidecar %s...); not using run %s",
                           path, digest[:12], str(meta.get("sha256"))[:12], meta.get("run_id"))
            return None
    return meta
