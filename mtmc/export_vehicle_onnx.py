#!/usr/bin/env python3
"""Export COCO YOLO weights to a DeepStream-compatible ONNX at a chosen input size.

Thin wrapper around the DeepStream-Yolo exporter
(``perception/models/DeepStream-Yolo/utils/export_yolo11.py``, vendored unmodified from upstream:
perception/models/DeepStream-Yolo/VENDORED.md) so the exact settings that produced
the checked-in artifacts are reproducible.  It reuses that script's model prep —
including its ``DeepStreamOutput`` head — instead of reimplementing the export.

Why a wrapper at all:

  * The upstream script calls ``torch.onnx.export`` without ``dynamo=``.  On
    torch >= 2.9 that selects the new torch.export-based exporter, which needs
    ``onnxscript``.  We pin the legacy TorchScript exporter (``dynamo=False``)
    that the DeepStream-Yolo code was written against, so no extra dependency is
    needed.
  * The upstream script writes ``labels.txt`` into the CURRENT WORKING
    DIRECTORY, and the ONNX beside the weights file.  Running it from an output
    directory keeps generated label files out of unrelated dirs (the repo root
    ``labels.txt`` belongs to the face SGIE and must not be clobbered).
  * Input size and dynamic batch are set explicitly: DeepStream's YOLO parser and
    the existing plate model both expect a dynamic batch dim with a fixed
    spatial size (e.g. ``[0, 3, 1280, 1280]`` -> ``[0, 33600, 6]``), and the
    export head emits ``[x, y, w, h, score, label_index]`` per anchor with NMS
    applied later by the PGIE config's ``nms-iou-threshold``.  No NMS is baked
    into the graph.

Tracking (MLflow, spike D8): each exported weights file is its own run in the
``vehicle-detector`` experiment. Called from ``finetune_vehicle_detector --export``
it is a child run of the training run (the subprocess inherits ``MLFLOW_RUN_ID``);
standalone it is a top-level run. Its params are the export settings (size, opset,
dynamic, simplify, batch, dynamo), its tags the exporter's path, actual and pinned
sha256 and upstream commit, and the lineage: ``source_run_id`` is the training run,
read from the input weights' ``<weights>.mlflow.json`` sidecar (``--source-run-id``
overrides it; no matching sidecar -> a warning and ``source_run_id=unknown``), plus
``source_weights_sha256``. When the training run is in the same store, its
``data.*`` tags and its ``eval/*`` metrics (as ``source_run/eval/*``: measured on the
.pt, not on this ONNX) are copied, so the ONNX can be registered from this run. The
ONNX and labels.txt are logged with their own sidecars. ``--no-track`` records nothing.

Usage:
    .venv/bin/python -m mtmc.export_vehicle_onnx
    .venv/bin/python -m mtmc.export_vehicle_onnx --weights yolo11s.pt yolo11m.pt --size 1280
    .venv/bin/python -m mtmc.export_vehicle_onnx --weights yolo11l.pt --out-dir perception/models/DeepStream-Yolo/vehicle_1280
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

from mtmc.tracking import SIDECAR_SUFFIX, TrackedRun, log_model_file, read_model_sidecar, tracked_run
from mtmc.tracking.yolo import LOCKFILE

# The DeepStream-Yolo exporter this wraps, vendored unmodified from upstream at the
# commit below (MIT; perception/models/DeepStream-Yolo/VENDORED.md and LICENSE.md).
# The sha256 pin is checked before every export.
REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "perception/models/DeepStream-Yolo/utils/export_yolo11.py"
UPSTREAM_REPO = "https://github.com/marcoslucianops/DeepStream-Yolo"
UPSTREAM_COMMIT = "2894babce8e75c49115dbe0c7b516289ed853565"
UPSTREAM_SHA256 = "46d9dcb897eeb013b5d47ab1917f7694c7b8a0d706dbeb1f875a344611bb551a"
UPSTREAM_URL = ("https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/"
                f"{UPSTREAM_COMMIT}/utils/export_yolo11.py")
DEFAULT_OUT_DIR = Path("perception/models/DeepStream-Yolo/vehicle_1280")
DEFAULT_WEIGHTS = ("yolo11s.pt", "yolo11m.pt", "yolo11l.pt")
CAPABILITY = "vehicle-detector"                   # spike D2: the registered model name (tag `capability`)
EXPERIMENT = CAPABILITY                           # the export run sits next to (under) its training run


def check_upstream(path: Path, simplify: bool) -> None:
    """Refuse before any work when the exporter or a package it imports is missing.

    Without this, a clean clone fails inside the import of a file it does not have, and a missing
    ``onnxslim`` fails only after torch.onnx.export has already written an unsimplified ONNX.
    """
    if not path.is_file():
        raise SystemExit(
            f"the DeepStream-Yolo exporter is missing: {path}\n"
            "It is tracked in git (perception/models/DeepStream-Yolo/VENDORED.md), so this checkout is incomplete "
            f"or predates it: restore it with `git checkout -- {path.relative_to(REPO)}` (from the repo root), "
            f"or fetch the pinned upstream copy ({UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]}, MIT):\n"
            f"  curl -fsSL -o {path} {UPSTREAM_URL}\n"
            f"  echo '{UPSTREAM_SHA256}  {path}' | sha256sum -c")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        print(f"WARNING: {path} is not the pinned exporter (sha256 {digest[:12]}..., expected "
              f"{UPSTREAM_SHA256[:12]}... = {UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]}); the ONNX may differ "
              "from the deployed one. The vendored file must stay unmodified (VENDORED.md).",
              file=sys.stderr, flush=True)
    needed = ["torch", "onnx", "ultralytics"] + (["onnxslim"] if simplify else [])
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(
            f"the exporter needs {', '.join(missing)}, which this environment does not have "
            f"(it imports torch, onnx and ultralytics{', and onnxslim for --simplify' if simplify else ''}). "
            "Install them in the environment you run this from, or pass --no-simplify if only onnxslim is missing.")


def copy_weights_sidecar(weights: Path, out_dir: Path, meta: dict | None) -> None:
    """export_one copies the weights into out_dir (best.pt next to best.onnx); give that copy its sidecar too, so it
    still names its training run (D9) once it is dvc-added there."""
    local = out_dir / weights.name
    if meta is None or local.resolve() == weights.resolve() or not local.is_file():
        return
    src = weights.with_name(weights.name + SIDECAR_SUFFIX)
    dst = local.with_name(local.name + SIDECAR_SUFFIX)
    shutil.copyfile(src, dst)
    if read_model_sidecar(local) is None:             # the copy is not the same bytes: a stale sidecar is worse
        dst.unlink()


def load_upstream(path: Path):
    """Import the exporter from its file path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("export_yolo11", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_yolo11"] = mod
    spec.loader.exec_module(mod)
    return mod


def export_one(mod, weights: Path, out_dir: Path, size: int, opset: int,
               dynamic: bool, simplify: bool) -> Path:
    """Export a single .pt to ONNX inside out_dir; returns the ONNX path."""
    # Absolute paths: the caller chdir's into out_dir (the upstream script writes
    # labels.txt to cwd), and relative paths would resolve against the wrong dir.
    local = (out_dir / weights.name).resolve()
    if local != weights.resolve():
        shutil.copyfile(weights, local)

    args = types.SimpleNamespace(
        weights=str(local), size=[size], opset=opset,
        simplify=simplify, dynamic=dynamic, batch=1,
    )
    print(f"### exporting {weights.name} @{size} (dynamic={dynamic}, simplify={simplify})",
          flush=True)
    mod.main(args)

    onnx_path = local.with_suffix(".onnx")
    if not onnx_path.exists():
        raise SystemExit(f"export did not produce {onnx_path}")
    return onnx_path


def source_run(weights: Path, override: str | None) -> tuple[str, dict | None]:
    """The training run that produced `weights` (D8): --source-run-id, else the weights' sidecar, else unknown."""
    if override:
        return override, None
    meta = read_model_sidecar(weights)            # None when absent, unreadable, or not this file's sha256
    if meta and meta.get("run_id"):
        return meta["run_id"], meta
    why = ("its sidecar does not match the file or names no run"
           if weights.with_name(weights.name + SIDECAR_SUFFIX).is_file() else f"no {weights.name}{SIDECAR_SUFFIX}")
    print(f"WARNING: cannot tell which training run produced {weights} ({why}); tagging source_run_id=unknown. "
          "Pass --source-run-id if you know it.", file=sys.stderr, flush=True)
    return "unknown", None


def inherit_from_source(run: TrackedRun, source_run_id: str) -> None:
    """Copy the training run's dataset lineage (data.*) and evaluation metrics (eval/* -> source_run/eval/*) when it is
    in this run's store, so the exported ONNX carries the lineage the registry asks for. Never fails the export."""
    if not run.active or source_run_id == "unknown":
        return
    try:
        from mlflow.tracking import MlflowClient  # noqa: PLC0415 (present: the run is active)
        src = MlflowClient(tracking_uri=run.tracking_uri).get_run(source_run_id)
    except Exception as exc:
        print(f"WARNING: training run {source_run_id} is not in {run.tracking_uri} ({type(exc).__name__}); its "
              "dataset lineage and metrics are not copied to the export run", file=sys.stderr, flush=True)
        run.set_tags({"source_run.found": "false"})
        return
    run.set_tags({"source_run.found": "true",
                  **{k: v for k, v in src.data.tags.items() if k.startswith("data.")}})
    run.log_metrics({f"source_run/{k}": v for k, v in src.data.metrics.items() if k.startswith("eval/")})


def export_run(enabled: bool, weights: Path, args, source_run_id: str):
    """The export's run (a child of the training run when MLFLOW_RUN_ID is inherited), or nothing with --no-track."""
    if not enabled:
        return contextlib.nullcontext(None)
    exporter_sha = hashlib.sha256(UPSTREAM.read_bytes()).hexdigest()
    return tracked_run(
        EXPERIMENT, f"export-{weights.stem}-{args.size}", lockfile=LOCKFILE,
        params={"weights": weights.name, "size": args.size, "opset": args.opset, "dynamic": not args.static_batch,
                "simplify": not args.no_simplify, "batch": 1, "dynamo": False, "out_dir": args.out_dir},
        tags={"capability": CAPABILITY, "export.format": "onnx-deepstream",
              **({"source": "trained"} if source_run_id != "unknown" else {}),
              "source_run_id": source_run_id,
              "source_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "source_weights.path": str(weights),
              "exporter.path": str(UPSTREAM.relative_to(REPO)), "exporter.sha256": exporter_sha,
              "exporter.pinned_sha256": UPSTREAM_SHA256, "exporter.matches_pin": str(exporter_sha == UPSTREAM_SHA256),
              "exporter.upstream": f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT}"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=list(DEFAULT_WEIGHTS),
                    help="weights files, relative to the repo root")
    ap.add_argument("--size", type=int, default=1280,
                    help="square input size; 1280 recovers small-vehicle recall lost at 640")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--static-batch", action="store_true",
                    help="export a fixed batch of 1 instead of a dynamic batch axis "
                         "(DeepStream builds per-batch engines off the dynamic axis)")
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--source-run-id", default=None,
                    help="the training run of --weights (default: read from <weights>.mlflow.json)")
    ap.add_argument("--no-track", action="store_true", help="record nothing in MLflow")
    args = ap.parse_args(argv)
    check_upstream(UPSTREAM, simplify=not args.no_simplify)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_upstream(UPSTREAM)

    # Pin the legacy TorchScript exporter (see module docstring).
    _orig_export = mod.torch.onnx.export

    def _export(*a, **kw):
        kw.setdefault("dynamo", False)
        return _orig_export(*a, **kw)

    mod.torch.onnx.export = _export

    # The upstream script writes labels.txt into cwd, so run from the output dir.
    # Resolve weights to absolute paths BEFORE chdir: relative ones would
    # otherwise be resolved against out_dir and fail to copy (see export_one).
    weights = [Path(w).resolve() for w in args.weights]

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        for w in weights:
            source_run_id, source_meta = source_run(w, args.source_run_id)
            with export_run(not args.no_track, w, args, source_run_id) as run:
                if run is not None:
                    inherit_from_source(run, source_run_id)
                onnx_path = export_one(
                    mod, w, out_dir, args.size, args.opset,
                    dynamic=not args.static_batch, simplify=not args.no_simplify,
                )
                print(f"    -> {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
                if run is not None:
                    log_model_file(onnx_path, "onnx", run=run)
                    labels = out_dir / "labels.txt"
                    if labels.is_file():
                        log_model_file(labels, "onnx", run=run)
                    copy_weights_sidecar(w, out_dir, source_meta)
                    if run.active:
                        print(f"    MLflow export run {run.run_id} ({run.mode}), source_run_id={source_run_id}"
                              + (f"\n    {run.url}" if run.url else ""))
    finally:
        os.chdir(cwd)

    print(f"\nlabels + ONNX written to {out_dir}")
    print("Next: point a PGIE config at the ONNX and let DeepStream build its engine.")


if __name__ == "__main__":
    main()
