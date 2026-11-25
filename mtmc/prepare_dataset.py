#!/usr/bin/env python3
"""Prepare YOLO-format dataset for plate detection fine-tuning.

Creates a standard YOLO dataset from CCTV + EALPR data:
  mtmc/datasets/plate_detection/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml

Split strategy (3-way):
  - CCTV folders are split into train / val / test by camera
  - Latest folder per camera → TEST (held out completely for final eval)
  - Second-latest folder per camera → VAL (used during training for early stopping)
  - Everything else → TRAIN
  - EALPR: all goes to training (generic domain data)
  - Uses symlinks to save disk space

The test set is ONLY for final evaluation after training is complete.
Ultralytics will auto-evaluate on val during training, and you manually
run test evaluation at the end.

Usage:
    python -m mtmc.prepare_dataset
    python -m mtmc.prepare_dataset --test-folders ch23_20260726210235 D15_20260726185811 D16_20260726015532
    python -m mtmc.prepare_dataset --val-folders ch23_20260726192320 D15_20260726173533 D16_20260726001425
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REVIEWED_DIR = Path("mtmc/output/plate_crops/a_REVIEWED")
DATASET_DIR = Path("mtmc/datasets/plate_detection")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_cctv_folders() -> list[Path]:
    """Find all CCTV video folders (exclude EALPR)."""
    folders = []
    for d in sorted(REVIEWED_DIR.iterdir()):
        if not d.is_dir():
            continue
        if "EALPR" in d.name or "ealpr" in d.name.lower():
            continue
        frames_dir = d / "frames"
        if frames_dir.exists() and any(frames_dir.iterdir()):
            folders.append(d)
    return folders


def find_ealpr_data() -> tuple[Path, Path]:
    """Find EALPR images and labels directories."""
    ealpr_root = REVIEWED_DIR / "EALPR Vechicles dataset"
    images_dir = ealpr_root / "Vehicles"
    labels_dir = ealpr_root / "Vehicles Labeling"
    return images_dir, labels_dir


def prepare_dataset(
    val_folders: list[str] | None = None,
    test_folders: list[str] | None = None,
    copy: bool = False,
):
    """Create the YOLO dataset directory structure with train/val/test.

    Args:
        copy: If True, copy files instead of symlinking (for portability).
    """
    # Clean and create dataset dir
    if DATASET_DIR.exists():
        print(f"Cleaning existing dataset at {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Find data
    cctv_folders = find_cctv_folders()
    ealpr_images, ealpr_labels = find_ealpr_data()

    print(f"Found {len(cctv_folders)} CCTV folders")
    print(f"EALPR: {ealpr_images}")

    # Auto-assign splits per camera if not specified
    cam_groups: dict[str, list[Path]] = {}
    for f in cctv_folders:
        cam = f.name.split("_")[0]
        cam_groups.setdefault(cam, []).append(f)

    test_set = set(test_folders) if test_folders else set()
    val_set = set(val_folders) if val_folders else set()

    if not test_folders or not val_folders:
        for cam, folders in cam_groups.items():
            # Sorted by name = sorted by timestamp
            sorted_names = [f.name for f in folders]
            if not test_folders and len(sorted_names) >= 1:
                # Latest folder → test
                test_set.add(sorted_names[-1])
            if not val_folders and len(sorted_names) >= 2:
                # Second-latest → val
                val_set.add(sorted_names[-2])

    # Remove overlaps (test takes priority)
    val_set -= test_set

    print(f"Test folders:   {sorted(test_set)}")
    print(f"Val folders:    {sorted(val_set)}")

    # Link CCTV data
    counts = {"train": 0, "val": 0, "test": 0}
    for folder in cctv_folders:
        if folder.name in test_set:
            split = "test"
        elif folder.name in val_set:
            split = "val"
        else:
            split = "train"

        frames_dir = folder / "frames"
        labels_dir = folder / "labels"

        for img_path in sorted(frames_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            unique_name = f"{folder.name}_{img_path.name}"
            lbl_path = labels_dir / (img_path.stem + ".txt")

            # Link or copy image
            dst_img = DATASET_DIR / "images" / split / unique_name
            if copy:
                shutil.copy2(img_path.resolve(), dst_img)
            else:
                rel_target = os.path.relpath(img_path.resolve(), dst_img.parent)
                os.symlink(rel_target, dst_img)

            # Link or copy label
            if lbl_path.exists():
                dst_lbl = DATASET_DIR / "labels" / split / f"{folder.name}_{img_path.stem}.txt"
                if copy:
                    shutil.copy2(lbl_path.resolve(), dst_lbl)
                else:
                    rel_target_lbl = os.path.relpath(lbl_path.resolve(), dst_lbl.parent)
                    os.symlink(rel_target_lbl, dst_lbl)

            counts[split] += 1

    print(f"CCTV: {counts['train']} train, {counts['val']} val, {counts['test']} test")

    # Link EALPR data (all to training)
    ealpr_count = 0
    if ealpr_images.exists():
        for img_path in sorted(ealpr_images.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            unique_name = f"EALPR_{img_path.name}"
            lbl_path = ealpr_labels / (img_path.stem + ".txt")

            dst_img = DATASET_DIR / "images" / "train" / unique_name
            if copy:
                shutil.copy2(img_path.resolve(), dst_img)
            else:
                rel_target = os.path.relpath(img_path.resolve(), dst_img.parent)
                os.symlink(rel_target, dst_img)

            if lbl_path.exists():
                dst_lbl = DATASET_DIR / "labels" / "train" / unique_name.replace(img_path.suffix, ".txt")
                if copy:
                    shutil.copy2(lbl_path.resolve(), dst_lbl)
                else:
                    rel_target_lbl = os.path.relpath(lbl_path.resolve(), dst_lbl.parent)
                    os.symlink(rel_target_lbl, dst_lbl)

            ealpr_count += 1

    print(f"EALPR: {ealpr_count} train")

    # Generate data.yaml
    # Use path relative to project root (where user runs the command from)
    rel_path = os.path.relpath(DATASET_DIR.resolve(), Path.cwd())
    data_yaml = f"""# Plate detection dataset
# Generated by prepare_dataset.py

path: {rel_path}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names
names:
  0: License_Plate
"""
    (DATASET_DIR / "data.yaml").write_text(data_yaml)

    train_total = counts["train"] + ealpr_count
    print(f"\nDataset written to {DATASET_DIR}")
    print(f"  Train: {train_total} images")
    print(f"  Val:   {counts['val']} images  (used during training)")
    print(f"  Test:  {counts['test']} images  (held out for final eval)")
    print(f"  Total: {train_total + counts['val'] + counts['test']} images")
    print(f"\n  data.yaml: {DATASET_DIR / 'data.yaml'}")
    print(f"\n⚠️  Remember: run test evaluation ONLY after training is complete:")
    print(f"   python -m mtmc.finetune_plate_detector --test-only --weights <best.pt>")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-folders", nargs="*", default=None,
                        help="Folder names for validation (default: second-latest per camera)")
    parser.add_argument("--test-folders", nargs="*", default=None,
                        help="Folder names for test (default: latest per camera)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (for portability to other machines)")
    args = parser.parse_args()
    prepare_dataset(args.val_folders, args.test_folders, copy=args.copy)


if __name__ == "__main__":
    main()
