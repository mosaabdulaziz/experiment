#!/usr/bin/env python3
"""Fine-tune the YOLO11-s vehicle detector: the deployed 1-class ``vehicle`` recipe.

With no arguments this reproduces the vehicle detector the multicam product runs
(``perception/models/DeepStream-Yolo/config_infer_primary_vehicle_1cls.txt`` ->
``products/el-mohandes/models/vehicle_1280_1cls/best.onnx``). That model is the
``best.pt`` of run ``vehicle_yolo11s_1cls_v3`` (DATA.md, "Retired versions"), and
every default below is taken from that run's recorded
``products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/args.yaml``
(a DVC pointer: ``dvc pull`` it to compare).

The dataset is ``products/el-mohandes/datasets/vehicle_detection_1cls`` (DVC group
``el-mohandes-datasets``, ``just data-pull el-mohandes-datasets``): the reviewed
CCTV frames with every kept class collapsed to ``vehicle`` and ``Jarar`` dropped
(``mtmc.prepare_vehicle_dataset --one-class vehicle --exclude Jarar``).

Why imgsz=1280: the native frames are 3200x1800 and distant vehicles are small;
1280 is what the deployed config uses and recovers small-vehicle recall that 640
lost.

Why a smaller than default lr0: a few thousand images, so heavy regularization
beats fast convergence.

Batch 48 is what the recorded run used on the PRO6000 (96 GB). On a smaller GPU
pass a smaller ``--batch``; the run is then no longer the recorded recipe.

The default run name is ``vehicle_yolo11s_1cls_repro``, not the recorded
``vehicle_yolo11s_1cls_v3``: that run directory holds the DVC-tracked weights
of the deployed model, and Ultralytics (``exist_ok=True``) would overwrite them.

Flow (training machine, e.g. PRO6000):

    .venv/bin/python -m mtmc.finetune_vehicle_detector             # train with the deployed recipe
    .venv/bin/python -m mtmc.finetune_vehicle_detector --export    # ... then export best.pt -> ONNX

``--export`` runs ``mtmc.export_vehicle_onnx`` into ``--export-dir`` (default
``products/el-mohandes/models/vehicle_1280_1cls``): it writes ``best.pt``,
``best.onnx`` and a 1-class ``labels.txt`` there, the exact files the PGIE
config loads, so it **replaces the deployed model in place**. The previous
bytes stay in git + DVC history (DATA.md, "Versions: one path, history in git").
To ship it: ``dvc add`` the two weights, commit the pointers, ``just data-push``.
The sidecar rebuilds a cached engine when the ONNX's size differs from the one it
was built from (``<engine>.src``, perception/engine/engine_path.py); delete the
cached ``best.onnx_b*_gpu*_*.engine`` to be sure.

The older 7-class recipe (``vehicle_yolo11s``, retired, dataset no longer in the
repo) is still reachable with explicit arguments:

    .venv/bin/python -m mtmc.finetune_vehicle_detector --data <7-class data.yaml> \
        --batch 16 --name vehicle_yolo11s --export-dir perception/models/DeepStream-Yolo/vehicle_1280

Tracking (MLflow, mtmc/tracking/yolo.py): each training is one run in the
``vehicle-detector`` experiment of ``MLFLOW_TRACKING_URI`` (offline store when it is
unset or unreachable; training never waits on it). It records every CLI argument
(``cli.*``) and fixed hyperparameter (``train.*``), Ultralytics' own params and
per-epoch metrics, the final validation of best.pt (``eval/val/*``), the git
commit, the dataset's DVC pointer md5, the lockfile hash, and best.pt (with a
``best.pt.mlflow.json`` sidecar beside it), args.yaml, results.csv and the plots.
``--export`` logs the export as a child run. The script prints the run URL and the
command that registers the result; it never registers it. ``--no-track`` records
nothing. ``--test-only`` evaluates ``--model`` on the val split and logs into
that model's run (its ``best.pt.mlflow.json`` sidecar, or ``--run-id``) as
``eval/<set>/*``, the set named by ``--eval-set`` (default ``<dataset>-val``) and
versioned by the dataset's DVC pointer (mtmc/tracking/eval.py).

Usage:
    .venv/bin/python -m mtmc.finetune_vehicle_detector
    .venv/bin/python -m mtmc.finetune_vehicle_detector --model yolo11m.pt --batch 32
    .venv/bin/python -m mtmc.finetune_vehicle_detector --epochs 10 --name smoke   # smoke test
    .venv/bin/python -m mtmc.finetune_vehicle_detector --epochs 10 --name smoke --no-track
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from mtmc.tracking import log_model_file, read_model_sidecar
from mtmc.tracking.eval import add_eval_args, prepare_evaluation
from mtmc.tracking.yolo import (base_weights_tags, box_metrics, cli_params, fixed_params, log_val_metrics,
                                print_run_report, training_run)

# Every default below comes from the deployed run's recorded args
# (products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/args.yaml).
BASE_MODEL = Path("yolo11s.pt")                       # args.yaml `model`
# The run read mtmc/datasets/vehicle_detection_1cls (args.yaml `data`); the dataset is versioned at the
# products/el-mohandes path now (DATA.md), and the old location is still honoured.
DATASET_YAML = next((p for p in (Path("products/el-mohandes/datasets/vehicle_detection_1cls/data.yaml"),
                                 Path("mtmc/datasets/vehicle_detection_1cls/data.yaml")) if p.exists()),
                    Path("products/el-mohandes/datasets/vehicle_detection_1cls/data.yaml"))
OUTPUT_DIR = Path("products/el-mohandes/runs/vehicle_detector")
RUN_NAME = "vehicle_yolo11s_1cls_repro"               # not `_v3`: that run dir holds the deployed weights
EXPORT_DIR = Path("products/el-mohandes/models/vehicle_1280_1cls")   # what config_infer_primary_vehicle_1cls.txt loads
CAPABILITY = "vehicle-detector"                       # the registered model name (spike D2): tag `capability`
EXPERIMENT = CAPABILITY                               # the MLflow experiment its runs go to
EXPORT_CMD = [sys.executable, "-m", "mtmc.export_vehicle_onnx"]


def fixed_hyperparameters(args) -> dict:
    """Everything model.train gets that is not a CLI argument (logged as train.* params)."""
    augment = not args.no_augment
    return dict(
        # absolute path: otherwise ultralytics nests the project under runs/detect/
        # and the best.pt / --export lookup below would miss it
        project=str(OUTPUT_DIR.resolve()),
        exist_ok=True,
        # Augmentation. Mosaic/mixup/copy-paste help a lot on a small set; the
        # imbalanced classes (car/bus/tanker have <60 boxes) rely on it.
        mosaic=1.0 if augment else 0.0,
        mixup=0.1 if augment else 0.0,
        copy_paste=0.1 if augment else 0.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,  # vehicles on the ground: no vertical flip
        fliplr=0.5,
        scale=0.5,
        optimizer="SGD",
        cos_lr=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,
    )


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=str(BASE_MODEL),
                    help=f"base weights to fine-tune from (default: {BASE_MODEL})")
    ap.add_argument("--data", type=str, default=str(DATASET_YAML),
                    help=f"dataset yaml (default: {DATASET_YAML})")
    ap.add_argument("--epochs", type=int, default=150,
                    help="epochs (default 150; small dataset, early stopping kicks in)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="input size (default 1280, matches the deployed vehicle config)")
    ap.add_argument("--batch", type=int, default=48,
                    help="batch size (default 48, the recorded run on a 96 GB PRO6000; lower it on smaller GPUs)")
    ap.add_argument("--lr0", type=float, default=0.005,
                    help="initial LR (default 0.005, low for a small dataset)")
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stopping patience (default 30)")
    ap.add_argument("--device", type=str, default="0",
                    help="cuda device id or 'cpu' (default 0)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", type=str, default="True",
                    help="dataset cache: True/False/'ram' (default True; speeds up small sets)")
    ap.add_argument("--name", type=str, default=RUN_NAME,
                    help=f"run name under {OUTPUT_DIR} (default {RUN_NAME}; the recorded run was "
                         "vehicle_yolo11s_1cls_v3, whose directory holds the deployed weights)")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable mosaic/mixup/copy-paste")
    ap.add_argument("--export", action="store_true",
                    help="after training, export best.pt to a DeepStream ONNX in --export-dir via "
                         "mtmc.export_vehicle_onnx (writes best.pt, best.onnx and labels.txt there)")
    ap.add_argument("--export-dir", type=str, default=str(EXPORT_DIR),
                    help=f"where --export writes (default {EXPORT_DIR}, the deployed model: replaced in place)")
    ap.add_argument("--test-only", action="store_true",
                    help="skip training, just val the --model weights on the val split")
    ap.add_argument("--no-track", action="store_true",
                    help="record nothing in MLflow (no run, Ultralytics' MLflow callback off, no sidecar)")
    add_eval_args(ap, None, no_track=False, default_help="<dataset>-val; with --test-only")
    args = ap.parse_args(argv)
    # cache accepts True/False/'ram'/'disk'; normalize the CLI string.
    cache_val = args.cache
    if cache_val.lower() in ("true", "1", "yes"):
        cache_val = True
    elif cache_val.lower() in ("false", "0", "no"):
        cache_val = False

    if not Path(args.data).exists():
        print(f"ERROR: dataset not found at {args.data}")
        print("Run first: .venv/bin/python -m mtmc.prepare_vehicle_dataset")
        sys.exit(1)

    if args.test_only:
        weights = args.model
        if not Path(weights).exists():
            print(f"ERROR: weights not found at {weights}")
            sys.exit(1)
        args.eval_set = args.eval_set or f"{Path(args.data).resolve().parent.name}-val"
        target = prepare_evaluation(args, weights, args.data, "mtmc.finetune_vehicle_detector --test-only",
                                    tags={"split": "val", "imgsz": args.imgsz})
        model = YOLO(weights)
        r = model.val(data=args.data, imgsz=args.imgsz, device=args.device, batch=args.batch)
        print_summary(r)
        target.log(box_metrics(r))
        return

    print(f"base model : {args.model}")
    print(f"dataset    : {args.data}")
    print(f"training   : {args.epochs} epochs, imgsz={args.imgsz}, batch={args.batch}")
    print(f"output     : {OUTPUT_DIR / args.name}")
    print()

    model = YOLO(args.model)
    hyperparameters = fixed_hyperparameters(args)
    best = Path(str(OUTPUT_DIR / args.name / "weights" / "best.pt"))
    data_yaml = Path(args.data).resolve()
    export_failed = None
    with training_run(not args.no_track, EXPERIMENT, args.name, model=model, capability=CAPABILITY,
                      params={**cli_params(args), **fixed_params(hyperparameters)},
                      data={data_yaml.parent.name: data_yaml},         # the helper finds <dataset>.dvc
                      tags=base_weights_tags(args.model)) as run:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            cache=cache_val,
            name=args.name,
            **hyperparameters,
        )

        print("\n" + "=" * 70)
        print(f"training done. best: {best}  last: {OUTPUT_DIR / args.name / 'weights' / 'last.pt'}")
        print("=" * 70)

        if not best.exists():
            print("no best.pt produced; val skipped")
        else:
            best_model = YOLO(str(best))
            print("\nFinal validation (best.pt) on the held-out val sessions:")
            r = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device,
                               batch=args.batch)
            print_summary(r)
            log_val_metrics(run, r, f"{CAPABILITY}/{data_yaml.parent.name}")
            if run is not None:
                # Ultralytics' callback already uploaded weights/best.pt: tag + sidecar only
                log_model_file(best, "weights", run=run, upload=False)

        if args.export and best.exists():
            try:      # inside the run: the export subprocess inherits MLFLOW_RUN_ID and becomes its child run
                export_to_deepstream(best, args.imgsz, Path(args.export_dir), track=not args.no_track)
            except subprocess.CalledProcessError as exc:
                export_failed = exc     # the training itself succeeded: its run stays FINISHED, the export is FAILED
                if run is not None:
                    run.set_tags({"export.status": "failed"})

    onnx = Path(args.export_dir) / "best.onnx"
    if args.export and export_failed is None and (meta := read_model_sidecar(onnx, verify=False)):
        print_run_report(run, onnx, f"the exported ONNX (export run {meta.get('run_id')}, a child of this run)")
    else:
        print_run_report(run, best if best.exists() else None, "best.pt")
    if export_failed is not None:
        raise SystemExit(f"export failed (exit {export_failed.returncode}); training run kept")


def print_summary(r):
    print(f"\n  mAP@50:    {r.box.map50:.4f}")
    print(f"  mAP@50-95: {r.box.map:.4f}")
    print(f"  Precision: {r.box.mp:.4f}")
    print(f"  Recall:    {r.box.mr:.4f}")
    try:
        ap_classes = [int(c) for c in r.box.ap_class_index]
        maps = list(r.box.maps)          # indexed by class id (nc long), NOT aligned with ap_class_index
        names = r.names if isinstance(r.names, dict) else {}
        print("  per-class mAP@50-95:")
        for c in ap_classes:
            print(f"    {names.get(c, c):<14} {maps[c]:.4f}")
    except Exception:
        pass


def export_to_deepstream(best: Path, imgsz: int, export_dir: Path, track: bool = True):
    """Re-export best.pt to the DeepStream ONNX the pipeline loads.

    Run inside the training run: the subprocess inherits MLFLOW_RUN_ID (set by configure_ultralytics), so its run is
    a child of the training run (mtmc.tracking.tracked_run), and reads best.pt's sidecar for source_run_id."""
    cmd = [*EXPORT_CMD, "--weights", str(best), "--size", str(imgsz), "--out-dir", str(export_dir)]
    if not track:
        cmd.append("--no-track")
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nbest.pt, best.onnx and labels.txt written to {export_dir}.")
    print("To ship it (DATA.md, everyday workflow):")
    print(f"  dvc add {export_dir / 'best.pt'} {export_dir / 'best.onnx'}")
    print(f"  git add {export_dir / 'best.pt.dvc'} {export_dir / 'best.onnx.dvc'} && git commit; just data-push")
    print("The PGIE config (config_infer_primary_vehicle_1cls.txt) already points at this ONNX. The sidecar "
          "rebuilds a cached engine only when the ONNX size changed: delete "
          f"{export_dir}/best.onnx_b*_gpu*_*.engine to force it.")


if __name__ == "__main__":
    main()