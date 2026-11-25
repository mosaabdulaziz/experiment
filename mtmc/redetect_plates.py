"""Re-detect plates in an existing plate_crops session using the fine-tuned model.

Overwrites the labels/, crops/, and plates.csv with higher-quality detections
from the fine-tuned YOLO model.

Usage:
    # Single session folder (name or full path)
    python -m mtmc.redetect_plates D16_20260726081054

    # Full path
    python -m mtmc.redetect_plates mtmc/output/plate_crops/D16_20260726081054

    # Custom model / confidence
    python -m mtmc.redetect_plates D16_20260726081054 --model products/el-mohandes/models/finetuned/best.pt --conf 0.3

    # Dry run: show what would change without writing
    python -m mtmc.redetect_plates D16_20260726081054 --dry-run

    # Batch: process all sessions under plate_crops/
    python -m mtmc.redetect_plates --all
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = Path("products/el-mohandes/models/finetuned/best.pt")
PLATE_CROPS_ROOT = Path("mtmc/output/plate_crops")


def _resolve_folder(folder_arg: str) -> Path:
    """Resolve a folder argument to an absolute plate_crops session path."""
    p = Path(folder_arg)
    if p.is_dir():
        return p.resolve()
    # Try as a name under plate_crops root
    candidate = PLATE_CROPS_ROOT / folder_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Session folder not found: {folder_arg}\n"
        f"  Tried: {p}\n  Tried: {candidate}"
    )


def redetect_session(
    session_dir: Path,
    model: YOLO,
    conf: float = 0.25,
    imgsz: int = 1280,
    dry_run: bool = False,
    device: str | None = None,
    backup: bool = True,
) -> dict:
    """Run the fine-tuned model on all frames in a session and overwrite labels/crops/csv.

    Returns a summary dict with counts.
    """
    frames_dir = session_dir / "frames"
    labels_dir = session_dir / "labels"
    crops_dir = session_dir / "crops"
    csv_path = session_dir / "plates.csv"

    if not frames_dir.is_dir():
        print(f"  [SKIP] No frames/ directory in {session_dir.name}")
        return {"skipped": True}

    frame_files = sorted(frames_dir.glob("frame*.jpg"))
    if not frame_files:
        print(f"  [SKIP] No frame images in {frames_dir}")
        return {"skipped": True}

    # Count old detections
    old_label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.is_dir() else 0
    old_crop_count = len(list(crops_dir.glob("*.jpg"))) if crops_dir.is_dir() else 0

    if dry_run:
        print(f"  [DRY RUN] Would process {len(frame_files)} frames "
              f"(existing: {old_label_count} labels, {old_crop_count} crops)")
        return {"dry_run": True, "frames": len(frame_files)}

    # Backup existing labels
    if backup and labels_dir.is_dir() and old_label_count > 0:
        backup_dir = session_dir / "labels_old"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(labels_dir, backup_dir)

    # Ensure output dirs exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    total_plates = 0
    total_frames = 0
    csv_rows = []
    plate_counter = 0

    for frame_path in frame_files:
        # Parse frame index from filename: frame00000.jpg → 0
        stem = frame_path.stem  # frame00000
        try:
            frame_idx = int(stem.replace("frame", ""))
        except ValueError:
            continue

        frame_num = frame_idx + 1  # 1-indexed for label filenames

        # Read frame
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [WARN] Cannot read {frame_path.name}, skipping")
            continue

        h, w = img.shape[:2]

        # Run detection
        results = model(
            img, imgsz=imgsz, conf=conf,
            verbose=False, device=device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            # Write empty label file
            label_file = labels_dir / f"{stem}.txt"
            label_file.write_text("")
            continue

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        n_plates = len(confs)
        total_plates += n_plates
        total_frames += 1

        # Write YOLO label file (class 0 = plate, normalized xywh)
        label_lines = []
        for box, c in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(label_lines) + "\n")

        # Crop and save each plate
        for i, (box, c) in enumerate(zip(boxes_xyxy, confs)):
            x1, y1, x2, y2 = box.astype(int)
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{stem}_plate{plate_counter:04d}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            csv_rows.append({
                "frame": frame_idx,
                "crop_file": crop_name,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": round(float(c), 3),
            })
            plate_counter += 1

    # Write plates.csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "crop_file", "x1", "y1", "x2", "y2", "conf"])
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "frames": len(frame_files),
        "frames_with_plates": total_frames,
        "total_plates": total_plates,
        "old_labels": old_label_count,
        "old_crops": old_crop_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", nargs="?",
                        help="Session folder name or path (e.g. D16_20260726081054)")
    parser.add_argument("--all", action="store_true",
                        help="Process all session folders under plate_crops/")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"Path to fine-tuned model (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO input size (default: 1280)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't back up old labels/ to labels_old/")
    args = parser.parse_args()

    if not args.folder and not args.all:
        parser.error("Provide a session folder name or use --all")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"  Model loaded. conf={args.conf}, imgsz={args.imgsz}")

    # Collect sessions
    if args.all:
        sessions = sorted([
            d for d in PLATE_CROPS_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        print(f"\nProcessing {len(sessions)} sessions under {PLATE_CROPS_ROOT}")
    else:
        sessions = [_resolve_folder(args.folder)]
        print(f"\nProcessing: {sessions[0].name}")

    # Process each session
    t0 = time.time()
    for i, session_dir in enumerate(sessions):
        print(f"\n[{i+1}/{len(sessions)}] {session_dir.name}")
        stats = redetect_session(
            session_dir, model,
            conf=args.conf, imgsz=args.imgsz,
            dry_run=args.dry_run, device=args.device,
            backup=not args.no_backup,
        )
        if stats.get("skipped"):
            continue
        if not stats.get("dry_run"):
            print(f"  → {stats['total_plates']} plates across {stats['frames_with_plates']}/{stats['frames']} frames "
                  f"(was {stats['old_labels']} labels, {stats['old_crops']} crops)")

    elapsed = time.time() - t0
    print(f"\nDone. {len(sessions)} sessions processed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
