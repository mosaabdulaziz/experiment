"""Log an evaluation into the run of the model it evaluated, on a named, versioned evaluation set (sub-task 6 of the
MLflow story; spike D5, D9, Q9).

    ap = argparse.ArgumentParser(...)
    add_eval_args(ap, default_set="el-mohandes-13clips")
    args = ap.parse_args()
    target = prepare_evaluation(args, model=args.vehicle_model, data=clip_paths, script="mtmc.audit_vehicle_detector")
    ...                                              # the evaluation itself (refusals happen before it, never after)
    target.log({"recall": 0.91}, artifacts=[report_json])

Two phases, so a refusal (no run for the model, a different data version on the same set) costs nothing:
`prepare_evaluation` resolves the run and the set's version and checks them against the run; `EvalTarget.log` writes.

The model's run (D9): `--run-id`, else the model file's `<file>.mlflow.json` sidecar, whose sha256 must match the file.
Neither: refused, with the command that gives the model a run, unless `--no-track`.

The evaluation set: `--eval-set <name>` names the data the script reads (one path or several). Its version is the md5
of the data's DVC pointer; for a set of several pointers (the audit's 13 clips), the md5 of the sorted
"<pointer> <md5>" lines, and the members are logged as `eval/<set>/dvc_pointers.txt`. Data that no pointer versions
makes the set `unversioned` (tagged, and warned about), with the sha256 of the content when it is a single file (the
AICity22 GT, D5), so a change can at least be detected.

What is written into the model's run:
- metrics `eval/<set>/<metric>`;
- tags `eval.<set>.dvc_md5`, `.path`, `.git_commit`, `.evaluated_at`, `.script` (+ `.versioned`, `.dvc_pointers`,
  `.content_sha256`, `.previous_dvc_md5` after --force, and the script's own `eval.<set>.*` tags);
- the report files under `eval/<set>/`. Only aggregate reports: never images or contact sheets (client footage,
  spike §8), and never anything holding plate text (the OCR eval's per-crop CSV stays local).

Safety: a run that already has `eval.<set>.dvc_md5` with a DIFFERENT value is refused (its metrics would mix two data
versions under one name); `--force` overwrites and keeps the old md5 in `eval.<set>.previous_dvc_md5`. The same version
again is fine (the metrics get a new latest value).

The run's status is never touched: everything goes through MlflowClient(run_id=...), which logs into a FINISHED run
without reopening it. (Resuming with mlflow.start_run(run_id) would show the run RUNNING during the write, and
end_run() would overwrite its status and end time, which then need restoring.)

Server unreachable: as for training (D10), the evaluation still completes and is recorded OFFLINE, as its own run in
the offline store's experiment `eval-pending`, tagged `eval.target_run_id` with everything above. There is no sync
tool: an offline evaluation stays local (only training is required not to fail, AC6); re-run it with the server up.
mlflow not installed: a warning, nothing recorded. `--no-track`: nothing at all.

Who may write (the team server's basic auth, mlflow 3.18 mlflow/server/auth): the user that creates an experiment is
granted MANAGE on it (set_can_manage_experiment_permission), writing to a run needs UPDATE on its experiment
(_run_requirement), and every other user gets the server's default_permission, READ. An evaluation writes into the
model's run, so:
- run training (finetune scripts, export), imports (`python -m mtmc.tracking.import_models`) and evaluations as the
  shared `iris` client user (MLFLOW_TRACKING_USERNAME/PASSWORD from ~/mlflow-server/secrets/client-iris.env), so the
  experiments those runs live in are iris's and the evaluations can write to them;
- the `admin` user is for administration only (users, permissions, cleanup), never for runs or evaluations;
- a run in an experiment another user created is readable but not writable for iris: the read in
  prepare_evaluation passes, and the write fails at log() as a warning (the evaluation and its report are kept). Fix
  it once per experiment: the admin grants iris EDIT (POST /api/2.0/mlflow/experiments/permissions/create with
  experiment_id, username=iris, permission=EDIT).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import SIDECAR_SUFFIX, _backend, _lineage, read_model_sidecar
from ._lineage import sha256_file

logger = logging.getLogger("mtmc.tracking")

OFFLINE_EXPERIMENT = "eval-pending"
UNVERSIONED = "unversioned"
_SET_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")        # one segment of metric names, tag keys and artifact paths


class EvalRefused(SystemExit):
    """The evaluation would log to no run, the wrong run, or mix data versions. Exits the script with the message."""

    def __init__(self, message: str):
        super().__init__(f"evaluation refused: {message}")


# -- CLI ---------------------------------------------------------------------------------------------------------
def add_eval_args(ap: argparse.ArgumentParser, default_set: str | None, *, no_track: bool = True,
                  default_help: str | None = None) -> None:
    """--eval-set, --run-id, --force (and --no-track, unless the script already has it)."""
    g = ap.add_argument_group("MLflow (the evaluated model's run)")
    g.add_argument("--eval-set", default=default_set,
                   help="name of the evaluation set, used in eval/<set>/<metric> "
                        f"(default: {default_help or default_set})")
    g.add_argument("--run-id", default=None,
                   help="the evaluated model's run (default: read from <model>.mlflow.json beside the model file)")
    g.add_argument("--force", action="store_true",
                   help="log even though the run already has this set evaluated on another data version "
                        "(the old version is kept in eval.<set>.previous_dvc_md5)")
    if no_track:
        g.add_argument("--no-track", action="store_true", help="record nothing in MLflow")


# -- the evaluation set -------------------------------------------------------------------------------------------
@dataclass
class EvalSet:
    name: str
    paths: list[str]                                   # as given, relative to the repo where possible
    dvc_md5: str                                       # the version, or "unversioned"
    pointers: list[tuple[str, str]] = field(default_factory=list)   # (pointer, md5) per versioned member
    unversioned: list[str] = field(default_factory=list)
    content_sha256: str | None = None                  # an unversioned single file's bytes

    @property
    def versioned(self) -> bool:
        return self.dvc_md5 != UNVERSIONED


def check_set_name(name: str) -> str:
    if not _SET_NAME.match(name) or name in (".", ".."):
        raise EvalRefused(f"--eval-set {name!r}: use letters, digits, '.', '_' and '-' only "
                          "(it becomes a metric-name segment, a tag key and an artifact folder)")
    return name


def resolve_eval_set(name: str, data: str | Path | Sequence[str | Path], root: Path | None = None) -> EvalSet:
    """Version the data an evaluation reads by its DVC pointer(s) (D5)."""
    root = Path(root) if root else _lineage.repo_root()
    check_set_name(name)
    items = [data] if isinstance(data, (str, Path)) else list(data)
    if not items:
        raise EvalRefused(f"evaluation set {name!r} names no data")
    paths, pointers, missing = [], [], []
    for raw in items:
        p = Path(raw)
        p = p if p.is_absolute() else Path.cwd() / p
        paths.append(_lineage._rel(p, root))
        pointer = _lineage.find_dvc_pointer(p)
        if pointer is None:
            missing.append(_lineage._rel(p, root))
            continue
        try:
            pointers.append((_lineage._rel(pointer, root), _lineage.parse_dvc_pointer(pointer)["md5"]))
        except (OSError, ValueError) as exc:
            logger.warning("tracking: cannot read %s (%s); counting %s as unversioned", pointer, exc, p)
            missing.append(_lineage._rel(p, root))
    pointers = sorted(set(pointers))
    if missing:
        logger.warning("tracking: evaluation set %r is UNVERSIONED: no .dvc pointer versions %s. Its metrics cannot be "
                       "tied to a data version (dvc add it to fix that)", name, ", ".join(missing[:5])
                       + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""))
        single = Path(items[0]) if len(items) == 1 else None
        content = sha256_file(single) if single is not None and single.is_file() else None
        return EvalSet(name, paths, UNVERSIONED, pointers, missing, content)
    md5 = pointers[0][1] if len(pointers) == 1 else hashlib.md5(
        "".join(f"{ptr} {m}\n" for ptr, m in pointers).encode()).hexdigest()
    return EvalSet(name, paths, md5, pointers)


# -- the model's run ----------------------------------------------------------------------------------------------
def give_it_a_run_hint(model: Path) -> str:
    return ("Give it a run first: a model trained here gets one from its training script (sidecar written by "
            "log_model_file); an imported or Kaggle model from `python -m mtmc.tracking.import_models` (catalog: "
            "mtmc/tracking/imported_models.yaml), which writes the sidecar; a model whose run you know: pass "
            f"--run-id. Or pass --no-track to evaluate without recording. (model: {model})")


def resolve_model_run(model: str | Path, run_id: str | None = None) -> tuple[str, dict | None]:
    """(run id, sidecar) of the evaluated model (D9): --run-id, else its sidecar whose sha256 matches the file."""
    model = Path(model)
    if run_id:
        return run_id, read_model_sidecar(model, verify=False) if model.is_file() else None
    sidecar = model.with_name(model.name + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        raise EvalRefused(f"{model} has no {model.name}{SIDECAR_SUFFIX}, so its run is unknown. "
                          + give_it_a_run_hint(model))
    meta = read_model_sidecar(model)                     # verifies the sha256
    if meta is None:
        raise EvalRefused(f"{model} does not match its {sidecar.name} (different sha256, or the file or sidecar is "
                          "unreadable): the file on disk is not the logged model, so its metrics would land in the "
                          "wrong run. `dvc checkout` the logged version, or pass --run-id for this file's run.")
    if not meta.get("run_id"):
        raise EvalRefused(f"{sidecar} names no run. " + give_it_a_run_hint(model))
    return meta["run_id"], meta


# -- logging ------------------------------------------------------------------------------------------------------
@dataclass
class EvalTarget:
    """Where prepare_evaluation decided to log: `mode` server/local (into the model's run), offline (an eval-pending
    run in the offline store), or disabled (nothing)."""
    mode: str
    eval_set: EvalSet | None = None
    run_id: str | None = None
    tracking_uri: str | None = None
    script: str = ""
    model: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    previous_md5: str | None = None
    root: Path | None = None
    logged_run_id: str | None = None                   # the run actually written (the pending run when offline)

    @property
    def active(self) -> bool:
        return self.mode != "disabled"

    def metric_names(self, metrics: Mapping[str, float]) -> dict[str, float]:
        return {f"eval/{self.eval_set.name}/{k}": float(v) for k, v in metrics.items()}

    def _tags(self) -> dict[str, str]:
        s, k = self.eval_set, f"eval.{self.eval_set.name}"
        tags = {f"{k}.dvc_md5": s.dvc_md5, f"{k}.versioned": str(s.versioned).lower(),
                f"{k}.path": ",".join(s.paths)[:5000],
                f"{k}.git_commit": _lineage.git_info(self.root).tags.get("git.commit", "unknown"),
                f"{k}.evaluated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                f"{k}.script": self.script, f"{k}.model_path": self.model}
        if s.pointers:
            tags[f"{k}.dvc_pointers"] = str(len(s.pointers))
        if s.content_sha256:
            tags[f"{k}.content_sha256"] = s.content_sha256
        if self.previous_md5:
            tags[f"{k}.previous_dvc_md5"] = self.previous_md5
        return {**tags, **{f"{k}.{t}": str(v) for t, v in self.tags.items()}}

    def log(self, metrics: Mapping[str, float], artifacts: Iterable[str | Path] = (),
            tags: Mapping[str, Any] | None = None) -> str | None:
        """Write the evaluation; returns the run written to. Never raises: a failure is a warning (the evaluation
        itself is done and its report is on disk)."""
        if not self.active:
            return None
        self.tags.update({k: str(v) for k, v in (tags or {}).items()})
        names = self.metric_names(metrics)
        try:
            from mlflow.entities import Metric, RunTag          # noqa: PLC0415
            from mlflow.tracking import MlflowClient             # noqa: PLC0415
            client = MlflowClient(tracking_uri=self.tracking_uri)
            run_id = self.run_id if self.mode != "offline" else self._pending_run(client)
            now = int(_dt.datetime.now().timestamp() * 1000)
            client.log_batch(run_id, metrics=[Metric(k, v, now, 0) for k, v in names.items()],
                             tags=[RunTag(k, v) for k, v in self._tags().items()])
            pointers = "".join(f"{ptr} {m}\n" for ptr, m in self.eval_set.pointers)
            if len(self.eval_set.pointers) > 1 or self.eval_set.unversioned:
                client.log_text(run_id, pointers + "".join(f"{p} unversioned\n" for p in self.eval_set.unversioned),
                                f"eval/{self.eval_set.name}/dvc_pointers.txt")
            for a in artifacts:
                if Path(a).is_file():
                    client.log_artifact(run_id, str(a), f"eval/{self.eval_set.name}")
            if self.mode == "offline":
                client.set_terminated(run_id, "FINISHED")
        except Exception as exc:                                  # the evaluation is done; never lose it to tracking
            logger.warning("tracking: could not record the evaluation in run %s (%s); the report is on disk",
                           self.run_id, exc)
            return None
        self.logged_run_id = run_id
        where = (f"run {run_id}" if self.mode != "offline"
                 else f"OFFLINE run {run_id} ({OFFLINE_EXPERIMENT}, for model run {self.run_id}: sync it)")
        print(f"[tracking] {len(names)} metrics eval/{self.eval_set.name}/* -> {where} "
              f"(set version {self.eval_set.dvc_md5})")
        return run_id

    def _pending_run(self, client) -> str:
        exp = client.get_experiment_by_name(OFFLINE_EXPERIMENT)
        exp_id = exp.experiment_id if exp else client.create_experiment(
            OFFLINE_EXPERIMENT, artifact_location=(_backend.offline_dir(self.root) / "artifacts").resolve().as_uri())
        run = client.create_run(exp_id, run_name=f"eval-{self.eval_set.name}",
                                tags={"eval.target_run_id": self.run_id or "unknown", "eval.set": self.eval_set.name,
                                      "tracking.mode": "offline", "mlflow.source.name": self.script})
        return run.info.run_id


def prepare_evaluation(args: argparse.Namespace, model: str | Path, data: str | Path | Sequence[str | Path],
                       script: str, *, root: Path | None = None, tracking_uri: str | None = None,
                       tags: Mapping[str, Any] | None = None) -> EvalTarget:
    """Decide where the evaluation of `model` on `data` (named args.eval_set) is logged, and refuse early.

    Raises EvalRefused (exits the script with its message): no run for the model, a sidecar that does not match the
    file, a run the store does not have, or the set already evaluated on another data version without --force.
    """
    root = Path(root) if root else _lineage.repo_root()
    if getattr(args, "no_track", False):
        print("[tracking] off (--no-track): the evaluation is not recorded")
        return EvalTarget("disabled")
    eval_set = resolve_eval_set(args.eval_set, data, root)
    run_id, meta = resolve_model_run(model, getattr(args, "run_id", None))
    try:
        import mlflow  # noqa: F401, PLC0415
        from mlflow.tracking import MlflowClient  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("tracking: mlflow is not installed (%s); the evaluation is NOT recorded", exc)
        return EvalTarget("disabled")
    backend = _backend.resolve(root, tracking_uri)
    target = EvalTarget(backend.mode, eval_set, run_id, backend.tracking_uri, script, str(model),
                        {k: str(v) for k, v in (tags or {}).items()}, root=root)
    if backend.mode == "offline":
        logger.warning("tracking: the evaluation will be recorded OFFLINE (%s) for model run %s; the data-version "
                       "check against that run happens when it is synced", backend.reason, run_id)
        return target
    if meta and meta.get("tracking_mode") == "offline":
        raise EvalRefused(f"the run of {model} ({run_id}) was logged offline; sync it to {backend.tracking_uri} first")
    try:
        run = MlflowClient(tracking_uri=backend.tracking_uri).get_run(run_id)
    except Exception as exc:
        raise EvalRefused(f"run {run_id} of {model} is not in {backend.tracking_uri} ({type(exc).__name__}). "
                          + give_it_a_run_hint(Path(model))) from exc
    if run.info.lifecycle_stage == "deleted":
        raise EvalRefused(f"run {run_id} of {model} is deleted")
    previous = run.data.tags.get(f"eval.{eval_set.name}.dvc_md5")
    if previous is not None and previous != eval_set.dvc_md5:
        if not getattr(args, "force", False):
            raise EvalRefused(
                f"run {run_id} already has evaluation set {eval_set.name!r} on data version {previous}, and this "
                f"data is {eval_set.dvc_md5}: the metrics eval/{eval_set.name}/* would mix two versions. Use another "
                "--eval-set name for this data, `dvc checkout` the evaluated version, or pass --force to overwrite "
                "(the old version is kept in a tag).")
        print(f"[tracking] --force: replacing set {eval_set.name!r} version {previous} with {eval_set.dvc_md5}",
              file=sys.stderr)
        target.previous_md5 = previous
    return target


def model_tags(path: str | Path, role: str) -> dict[str, str]:
    """`model.<role>.*` tags describing a model file an evaluation used (Q9): path, sha256, and from its sidecar the
    run id and registered name/version."""
    p = Path(path)
    tags = {f"model.{role}.path": str(path)}
    if not p.is_file():
        tags[f"model.{role}.sha256"] = "missing"
        return tags
    tags[f"model.{role}.sha256"] = sha256_file(p)
    meta = read_model_sidecar(p)
    if meta:
        for key in ("run_id", "registered_model", "version"):
            if meta.get(key):
                tags[f"model.{role}.{key}"] = str(meta[key])
    return tags


def manifest_tags(manifest: str | Path, used: Mapping[str, str | Path]) -> dict[str, str]:
    """`model.<role>.registered_model/version/run_id` for the used model files listed in a deployment manifest (the
    JSON `python -m mtmc.tracking.manifest generate` writes), matched by sha256, else by path."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    entries = data.get("models", [])
    tags = {"manifest.path": str(manifest), "manifest.sha256": sha256_file(Path(manifest))}
    for role, path in used.items():
        p = Path(path)
        digest = sha256_file(p) if p.is_file() else None
        hit = next((e for e in entries if digest and e.get("sha256") == digest), None) \
            or next((e for e in entries if Path(e.get("path", "")).name and str(path).endswith(e["path"])), None)
        if hit is None:
            tags[f"model.{role}.in_manifest"] = "false"
            continue
        tags[f"model.{role}.in_manifest"] = "true"
        for key in ("registered_model", "version", "run_id"):
            if hit.get(key):
                tags[f"model.{role}.{key}"] = str(hit[key])
        if digest and hit.get("sha256") != digest:
            tags[f"model.{role}.manifest_sha256_mismatch"] = "true"
    return tags


# -- the whole pipeline (Q9) --------------------------------------------------------------------------------------
PIPELINE_EXPERIMENT = "mtmc-pipeline-eval"


def log_pipeline_eval(*, eval_set: str, gt_path: str | Path, metrics: Mapping[str, float],
                      params: Mapping[str, Any], models: Mapping[str, str | Path], manifest: str | Path | None = None,
                      artifacts: Iterable[str | Path] = (), run_name: str | None = None, script: str = "",
                      extra_tags: Mapping[str, Any] | None = None, root: Path | None = None,
                      tracking_uri: str | None = None):
    """The pipeline evaluation's own run in `mtmc-pipeline-eval` (Q9): it scores the association of several models,
    so it belongs to none of their runs. Tagged with every model it used (`model.<role>.*`: sha256 and, from the
    sidecar or the deployment manifest, registered name/version and run id) and with its evaluation set
    (AICity22 is not in DVC: `unversioned`, plus the sha256 of the GT file, D5). Returns the TrackedRun."""
    from . import tracked_run                                   # noqa: PLC0415
    from .yolo import LOCKFILE                                  # noqa: PLC0415
    root = Path(root) if root else _lineage.repo_root()
    s = resolve_eval_set(eval_set, gt_path, root)
    k = f"eval.{s.name}"
    tags = {"eval.set": s.name, f"{k}.dvc_md5": s.dvc_md5, f"{k}.versioned": str(s.versioned).lower(),
            f"{k}.path": ",".join(s.paths), f"{k}.script": script,
            f"{k}.evaluated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    if s.content_sha256:
        tags[f"{k}.content_sha256"] = s.content_sha256
    for role, path in models.items():
        tags.update(model_tags(path, role))
    if manifest:
        tags.update(manifest_tags(manifest, models))
    tags.update({key: str(v) for key, v in (extra_tags or {}).items()})
    with tracked_run(PIPELINE_EXPERIMENT, run_name, params=params, lockfile=LOCKFILE, tags=tags, root=root,
                     tracking_uri=tracking_uri) as run:
        tags[f"{k}.git_commit"] = run.git_commit or "unknown"
        run.set_tags({f"{k}.git_commit": tags[f"{k}.git_commit"]})
        run.log_metrics({f"eval/{s.name}/{m}": v for m, v in metrics.items()})
        for a in artifacts:
            if Path(a).is_file():
                run.log_artifact(a, f"eval/{s.name}")
    if run.active:
        print(f"[tracking] pipeline evaluation -> run {run.run_id} ({run.mode}) in {PIPELINE_EXPERIMENT}"
              + (f"\n  {run.url}" if run.url else ""))
    return run
