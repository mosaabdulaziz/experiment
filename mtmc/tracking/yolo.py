"""The Ultralytics finetune scripts' tracking (sub-task 4 of the MLflow story; spike D4, D8, D9).

Shared by `mtmc.finetune_vehicle_detector` and `mtmc.finetune_plate_detector`:

    with training_run(not args.no_track, "vehicle-detector", args.name, params=..., data=..., model=model) as run:
        model.train(**hyperparameters)          # Ultralytics' callback: trainer.args as params, per-epoch metrics,
                                                # weights/ + args.yaml + results.csv + plots as artifacts
        log_val_metrics(run, best_model.val(...), "vehicle-detector/vehicle_detection_1cls")   # eval/val/...
        log_model_file(best, "weights", upload=False)    # sha256 tag + best.pt.mlflow.json (already uploaded)
    print_run_report(run, best)

What goes where:
- params: `cli.<arg>` for every CLI argument and `train.<name>` for the hyperparameters the script fixes in code.
  Prefixed because Ultralytics' callback logs `trainer.args` unprefixed (`epochs`, `batch`, `lr0`, ...), and MLflow
  refuses a param logged twice with a different value (e.g. `cache` "True" from the CLI vs True from Ultralytics).
- metrics: per epoch from the callback (`metrics/mAP50-95B`, `val/box_loss`, `train/box_loss`, `lr/pg0`, step =
  epoch); the script's final validation of best.pt as `eval/<split>/<metric>` (+ per class), the names the registry
  counts as evaluation metrics.
- artifacts: weights/best.pt and weights/last.pt, and the .png/.jpg/.csv/.yaml/.pt files of save_dir (args.yaml,
  results.csv, plots), uploaded once at the end of training. Not the save_period checkpoints (weights/epoch<N>.pt):
  Ultralytics' callback would upload the whole weights/ folder, so its on_train_end is replaced for this training
  (`upload_final_weights_only`). The checkpoints stay on disk; only the upload skips them.

With tracking off (`--no-track`), or a run that could not start, Ultralytics' own MLflow callback is removed from the
trainer as well: left alone, it would open its own run in MLFLOW_TRACKING_URI or ./runs/mlflow whenever mlflow is
installed and `yolo settings mlflow=True`.
"""
from __future__ import annotations

import contextlib
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import TrackedRun, configure_ultralytics, read_model_sidecar, tracked_run
from ._lineage import sha256_file

logger = logging.getLogger("mtmc.tracking")

LOCKFILE = "tools/mtmc-research/uv.lock"            # D7: the finetune and export scripts' environment
_ULTRALYTICS_MLFLOW = "ultralytics.utils.callbacks.mlflow"
_METRIC_NAME = re.compile(r"[^\w\-./ ]")


def cli_params(args: Any) -> dict[str, Any]:
    return {f"cli.{k}": v for k, v in sorted(vars(args).items())}


def fixed_params(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    return {f"train.{k}": v for k, v in sorted(hyperparameters.items())}


def remove_ultralytics_mlflow(model: Any) -> None:
    """Keep Ultralytics' MLflow callback out of the training `model` is about to run.

    Its trainer copies the model's callbacks and then adds the integration callbacks (MLflow among them) in its
    constructor; `on_pretrain_routine_start` runs after that and before the callback's `on_pretrain_routine_end`
    opens a run, so removing them there works whatever `yolo settings mlflow` says, without changing those settings.
    Multi-GPU (DDP) training rebuilds the trainer in a subprocess, which this does not reach: use one device with
    tracking off.
    """
    def strip(trainer: Any) -> None:
        for fns in getattr(trainer, "callbacks", {}).values():
            fns[:] = [f for f in fns if getattr(f, "__module__", "") != _ULTRALYTICS_MLFLOW]

    model.add_callback("on_pretrain_routine_start", strip)


WEIGHTS_UPLOADED = ("best.pt", "last.pt")
_SAVE_DIR_SUFFIXES = {".png", ".jpg", ".csv", ".pt", ".yaml"}     # the callback's own list (8.4.150 and 8.4.192)


def upload_final_weights_only(model: Any, run: TrackedRun) -> None:
    """Replace Ultralytics' MLflow `on_train_end` for this training: upload weights/best.pt and weights/last.pt and
    the save_dir files it uploads, but not the periodic weights/epoch<N>.pt checkpoints (`save_period`), which its
    `mlflow.log_artifact(str(trainer.best.parent))` would send with the whole weights/ folder.

    Swapped at on_pretrain_routine_start, like `remove_ultralytics_mlflow`. The rest of the callback (params,
    per-epoch metrics) is untouched. Its run-ending branch is not needed: `configure_ultralytics` makes it reuse
    `run` (never starting its own) and sets MLFLOW_KEEP_RUN_ACTIVE, and `tracked_run` ends the run. Its `teardown`
    hook (8.4.192) calls the module's own on_train_end with status FAILED, which uploads nothing; it is left alone.
    Multi-GPU (DDP) training rebuilds the trainer in a subprocess this does not reach, so there the original callback
    (and the checkpoints) apply.
    """
    def on_train_end(trainer: Any) -> None:
        if not getattr(trainer, "_mlflow_active", False):          # the callback could not start: nothing to log
            return
        weights = Path(trainer.best).parent
        for name in WEIGHTS_UPLOADED:
            if (weights / name).is_file():
                run.log_artifact(weights / name, "weights")
        for f in sorted(Path(trainer.save_dir).glob("*")):
            if f.is_file() and f.suffix in _SAVE_DIR_SUFFIXES:
                run.log_artifact(f)

    def swap(trainer: Any) -> None:
        fns = getattr(trainer, "callbacks", {}).get("on_train_end", [])
        fns[:] = [on_train_end if (getattr(f, "__module__", "") == _ULTRALYTICS_MLFLOW
                                   and getattr(f, "__name__", "") == "on_train_end") else f for f in fns]

    model.add_callback("on_pretrain_routine_start", swap)


def base_weights_tags(path: str | Path) -> dict[str, str]:
    """The weights training starts from (D4). Ultralytics downloads a bare `yolo11s.pt` into the working directory
    when the model is constructed, so this is read after `YOLO(...)`."""
    p = Path(path)
    tags = {"base_weights.path": str(path)}
    if p.is_file():
        tags["base_weights.sha256"] = sha256_file(p)
        if (meta := read_model_sidecar(p)) and meta.get("run_id"):
            tags["base_weights.run_id"] = meta["run_id"]          # fine-tuning one of our own tracked models
    else:
        tags["base_weights.sha256"] = "unknown"
    return tags


@contextlib.contextmanager
def training_run(enabled: bool, experiment: str, run_name: str, *, params: Mapping[str, Any],
                 data: Mapping[str, str | Path], model: Any, tags: Mapping[str, Any] | None = None,
                 lockfile: str | None = LOCKFILE, capability: str | None = None) -> Iterator[TrackedRun | None]:
    """The training run (`tracked_run` + `configure_ultralytics`), or None with tracking off. The run ends when the
    block exits, after the script's final validation and export, never at Ultralytics' on_train_end.

    `capability` (the registered model name the run's model is a version of, tag `capability`) defaults to the
    experiment name, as in the scripts; the e2e check logs a dummy capability into another experiment."""
    if not enabled:
        remove_ultralytics_mlflow(model)
        print("tracking: off (--no-track): no MLflow run, and Ultralytics' MLflow callback is disabled")
        yield None
        return
    with tracked_run(experiment, run_name, params=params, data=data, lockfile=lockfile,
                     tags={"capability": capability or experiment, "source": "trained", **(tags or {})}) as run:
        if run.active:
            configure_ultralytics(run)
            upload_final_weights_only(model, run)
        else:
            remove_ultralytics_mlflow(model)                        # no run of ours: no stray run of its own
        yield run


def box_metrics(results: Any) -> dict[str, float]:
    """`mAP50`, `mAP50-95`, `precision`, `recall` and `mAP50-95/<class>` from an Ultralytics DetMetrics."""
    box = results.box
    metrics = {"mAP50": box.map50, "mAP50-95": box.map, "precision": box.mp, "recall": box.mr}
    try:
        maps = list(box.maps)                                       # per class id (nc long), not per ap_class_index
        names = results.names if isinstance(results.names, dict) else dict(enumerate(results.names or []))
        for c in getattr(box, "ap_class_index", []):
            c = int(c)
            metrics[f"mAP50-95/{_METRIC_NAME.sub('_', str(names.get(c, c)))}"] = maps[c]
    except Exception as exc:                                        # per-class is a bonus; the totals are what count
        logger.warning("tracking: per-class mAP not logged (%s)", exc)
    return {k: float(v) for k, v in metrics.items()}


def val_metrics(results: Any, split: str = "val", prefix: str = "eval") -> dict[str, float]:
    """`<prefix>/<split>/<metric>` for the box_metrics of a DetMetrics."""
    return {f"{prefix}/{split}/{k}": v for k, v in box_metrics(results).items()}


def log_val_metrics(run: TrackedRun | None, results: Any, eval_set: str, split: str = "val") -> dict[str, float]:
    """Log the script's final validation under `eval/<split>/...` and name the evaluation set (D5)."""
    metrics = val_metrics(results, split)
    if run is not None:
        run.log_metrics(metrics)
        run.set_tags({f"eval.{split}.set": f"{eval_set}:{split}"})
    return metrics


def print_run_report(run: TrackedRun | None, model_file: Path | None, what: str = "the model") -> None:
    """Where the run is, and the exact command that registers `model_file` (registration is never automatic)."""
    print("\n" + "=" * 70)
    if run is None:
        print("tracking: off; nothing recorded")
        return
    if not run.active:
        print("tracking: this run was NOT recorded (see the tracking warnings above)")
        return
    print(f"MLflow run {run.run_id} ({run.mode}), experiment {run.experiment}")
    print(f"  {run.url or run.tracking_uri}")
    if model_file is None or not Path(model_file).is_file():
        print("  no model file was produced, so there is nothing to register")
        return
    if run.mode == "offline":
        print(f"  logged OFFLINE: sync it to the server before registering {what} (the registry refuses the "
              "offline store)")
        return
    print(f"To register {what} (not done automatically):")
    print(f"  python -m mtmc.tracking.registry --tracking-uri {run.tracking_uri} register "
          f"--model {shlex.quote(str(model_file))}")
