"""Filter plate crops to keep only real plates based on shape/size rules.

Usage:
    .venv/bin/python -m mtmc.filter_plates
"""

import csv
import os
import shutil
from pathlib import Path


def main():
    crops_dir = Path("mtmc/output/plate_crops")
    filtered_dir = Path("mtmc/output/plates_filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    with open(crops_dir / "plates.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Input detections: {len(rows)}")

    kept = []
    for r in rows:
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        w = x2 - x1
        h = y2 - y1
        conf = float(r["conf"])

        # Plate should be wider than tall (aspect ratio > 0.8)
        if h == 0:
            continue
        aspect = w / h
        if aspect < 0.8:
            continue

        # Reasonable plate size in 3200x1800 video
        if w < 30 or w > 500:
            continue
        if h < 15 or h > 250:
            continue

        # Reasonable confidence
        if conf < 0.4:
            continue

        kept.append(r)

    print(f"After filtering: {len(kept)}")

    # Copy kept crops to filtered folder
    for r in kept:
        src = crops_dir / r["crop_file"]
        if src.exists():
            shutil.copy2(str(src), str(filtered_dir / r["crop_file"]))

    # Write filtered CSV
    with open(filtered_dir / "plates_filtered.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "crop_file", "x1", "y1", "x2", "y2", "conf"])
        writer.writeheader()
        writer.writerows(kept)

    print(f"Saved {len(kept)} filtered plate crops to: {filtered_dir}")

    # Stats
    ws = [int(r["x2"]) - int(r["x1"]) for r in kept]
    hs = [int(r["y2"]) - int(r["y1"]) for r in kept]
    print(f"Size — width: {min(ws)}-{max(ws)}px, height: {min(hs)}-{max(hs)}px")


if __name__ == "__main__":
    main()
