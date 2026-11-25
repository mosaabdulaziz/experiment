#!/usr/bin/env python3
"""Fine-tune YOLOv11-s for Egyptian plate detection on CCTV footage.

Uses the YOLOv11-s checkpoint (best from benchmark) as the starting weights
and fine-tunes on the combined CCTV + EALPR dataset prepared by
mtmc/prepare_dataset.py.

Usage:
    # Default fine-tuning (100 epochs, imgsz=1280)
    python -m mtmc.finetune_plate_detector

    # Quick test run (10 epochs)
    python -m mtmc.finetune_plate_detector --epochs 10

    # Custom settings
    python -m mtmc.finetune_plate_detector --epochs 200 --imgsz 1280 --batch 8

    # Resume from checkpoint
    python -m mtmc.finetune_plate_detector --resume runs/detect/plate_yolov11s/weights/last.pt

    # Without MLflow tracking
    python -m mtmc.finetune_plate_detector --epochs 10 --no-track

Tracking (MLflow, mtmc/tracking/yolo.py): each training is one run in the ``plate-detector`` experiment of
``MLFLOW_TRACKING_URI`` (offline store when it is unset or unreachable; training never waits on it). It records every
CLI argument (``cli.*``) and fixed hyperparameter (``train.*``), Ultralytics' own params and per-epoch metrics, the
final validation of best.pt (``eval/val/*``), the git commit, the dataset's DVC pointer md5, the lockfile hash, and
best.pt (with a ``best.pt.mlflow.json`` sidecar beside it), args.yaml, results.csv and the plots. The script prints the
run URL and the command that registers best.pt; it never registers it. ``--no-track`` records nothing.
``--test-only`` evaluates the --resume weights (default: this run's best.pt) on the test split and logs into that
model's run (its ``best.pt.mlflow.json`` sidecar, or ``--run-id``) as ``eval/<set>/*``, the set named by
``--eval-set`` (default ``<dataset>-test``) and versioned by the dataset's DVC pointer (mtmc/tracking/eval.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from mtmc.tracking import log_model_file
from mtmc.tracking.eval import add_eval_args, prepare_evaluation
from mtmc.tracking.yolo import (base_weights_tags, box_metrics, cli_params, fixed_params, log_val_metrics,
                                print_run_report, training_run)


# Default paths
BASE_MODEL = Path("mtmc/models/yolov11-license-plate-detection/license-plate-finetune-v1s.pt")
# datasets are versioned under datasets/ (DATA.md); the old mtmc/datasets/ is still honoured
DATASET_YAML = next((p for p in (Path("products/el-mohandes/datasets/plate_detection/data.yaml"), Path("mtmc/datasets/plate_detection/data.yaml")) if p.exists()),
                    Path("products/el-mohandes/datasets/plate_detection/data.yaml"))
OUTPUT_DIR = Path("products/el-mohandes/runs/finetune")
CAPABILITY = "plate-detector"                       # the registered model name (spike D2): tag `capability`
EXPERIMENT = CAPABILITY                               # the MLflow experiment its runs go to


def fixed_hyperparameters(args) -> dict:
    """Everything model.train gets that is not a CLI argument (logged as train.* params)."""
    augment = not args.no_augment
    return dict(
        # absolute: Ultralytics 8.4 puts a relative project under <runs_dir>/detect/ (cfg.get_save_dir), so the
        # best.pt lookup after training (final val, sidecar) would miss it
        project=str(OUTPUT_DIR.resolve()),
        exist_ok=True,
        # Augmentation (aggressive for small dataset)
        mosaic=1.0 if augment else 0.0,
        mixup=0.1 if augment else 0.0,
        copy_paste=0.1 if augment else 0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,  # No vertical flip for plates
        fliplr=0.5,
        scale=0.5,  # Random scale ±50%
        # Small object handling
        # close_mosaic=10,  # Disable mosaic for last 10 epochs
        # Optimizer
        optimizer="SGD",
        cos_lr=True,
        # Misc
        verbose=True,
        plots=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=str(BASE_MODEL),
                        help=f"Base model weights (default: {BASE_MODEL})")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help=f"Dataset YAML (default: {DATASET_YAML})")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280 — important for small CCTV plates)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4, reduce if OOM on 8GB VRAM)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate factor (default: 0.01)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0, cpu, etc. (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--name", type=str, default="plate_yolov11s",
                        help="Experiment name (default: plate_yolov11s)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable Mosaic/MixUp augmentation")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, just run test evaluation on --resume weights")
    parser.add_argument("--no-track", action="store_true",
                        help="Record nothing in MLflow (no run, Ultralytics' MLflow callback off, no sidecar)")
    add_eval_args(parser, None, no_track=False, default_help="<dataset>-test; with --test-only")
    args = parser.parse_args(argv)
    if not Path(args.model).exists():
        raise SystemExit(f"base weights not found: {args.model}\n"
                         "They are third-party and not versioned (mtmc/models/THIRD_PARTY.md): get "
                         "license-plate-finetune-v1s.pt from Hugging Face morsetechlab/yolov11-license-plate-detection, "
                         "or pass --model.")

    # Verify dataset exists
    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found at {args.data}")
        print("Run first: python -m mtmc.prepare_dataset")
        return

    # Test-only mode: evaluate on test set and exit
    if args.test_only:
        weights = args.resume or str(OUTPUT_DIR / args.name / "weights" / "best.pt")
        if not Path(weights).exists():
            print(f"ERROR: Weights not found at {weights}")
            print("Train first or pass --resume <path_to_weights>")
            return
        print(f"Test-only mode: evaluating {weights}")
        args.eval_set = args.eval_set or f"{Path(args.data).resolve().parent.name}-test"
        target = prepare_evaluation(args, weights, args.data, "mtmc.finetune_plate_detector --test-only",
                                    tags={"split": "test", "imgsz": args.imgsz})
        model = YOLO(weights)
        results = model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
        print(f"\nTest set results:")
        print(f"  mAP@50:    {results.box.map50:.4f}")
        print(f"  mAP@50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall:    {results.box.mr:.4f}")
        target.log(box_metrics(results))
        return

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Base model: {args.model}")
        model = YOLO(args.model)

    print(f"Dataset: {args.data}")
    print(f"Training: {args.epochs} epochs, imgsz={args.imgsz}, batch={args.batch}")
    print(f"Output: {OUTPUT_DIR / args.name}")
    print()

    hyperparameters = fixed_hyperparameters(args)
    best_weights = OUTPUT_DIR / args.name / "weights" / "best.pt"
    data_yaml = Path(args.data).resolve()
    with training_run(not args.no_track, EXPERIMENT, args.name, model=model, capability=CAPABILITY,
                      params={**cli_params(args), **fixed_params(hyperparameters)},
                      data={data_yaml.parent.name: data_yaml},         # the helper finds <dataset>.dvc
                      tags=base_weights_tags(args.resume or args.model)) as run:
        # Train
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            name=args.name,
            **hyperparameters,
        )

        print()
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Best weights: {best_weights}")
        print(f"Last weights: {OUTPUT_DIR / args.name / 'weights' / 'last.pt'}")
        print(f"Results:      {OUTPUT_DIR / args.name / 'results.csv'}")
        print()

        # Run validation on best weights
        print("Running final validation on best weights ...")
        if best_weights.exists():
            best_model = YOLO(str(best_weights))
            val_results = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
            print(f"\nFinal validation metrics:")
            print(f"  mAP@50:    {val_results.box.map50:.4f}")
            print(f"  mAP@50-95: {val_results.box.map:.4f}")
            print(f"  Precision: {val_results.box.mp:.4f}")
            print(f"  Recall:    {val_results.box.mr:.4f}")
            log_val_metrics(run, val_results, f"{CAPABILITY}/{data_yaml.parent.name}")
            if run is not None:
                # Ultralytics' callback already uploaded weights/best.pt: tag + sidecar only
                log_model_file(best_weights, "weights", run=run, upload=False)

    print_run_report(run, best_weights if best_weights.exists() else None, "best.pt")


if __name__ == "__main__":
    main()
