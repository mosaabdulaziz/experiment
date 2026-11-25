#!/usr/bin/env python3
"""Dump decoded frames from the DeepStream pipeline to disk for visual inspection.

Reads the latest JPEG previews from Redis (iris:jpeg:<cam>) and saves them
as images, so you can visually compare what DeepStream sees vs the original
video files.

Usage (inside the sidecar container):
    python3 perception/pipeline/dump_frames.py --out /workspace/iris/data/frames_dump

Or from the host (reads from Redis):
    .venv/bin/python perception/pipeline/dump_frames.py --redis redis://localhost:6379/0 --out mtmc/output/frames_dump
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import redis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--redis", default="redis://localhost:6379/0")
    p.add_argument("--cam", nargs="*", default=None, help="Cameras (default: cam-1, cam-2, cam-3)")
    p.add_argument("--out", default="mtmc/output/frames_dump")
    p.add_argument("--count", type=int, default=5, help="Number of frames to save per camera")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    args = p.parse_args()

    r = redis.Redis.from_url(args.redis, decode_responses=False)
    cameras = args.cam or ["cam-1", "cam-2", "cam-3"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam in cameras:
        cam_dir = out_dir / cam
        cam_dir.mkdir(exist_ok=True)

    print(f"Saving {args.count} frames per camera to {args.out}")
    print(f"Cameras: {cameras}")

    for i in range(args.count):
        for cam in cameras:
            # Get latest JPEG
            jpeg_key = f"iris:jpeg:{cam}"
            jpeg_data = r.get(jpeg_key)
            if jpeg_data is None:
                print(f"  [{cam}] No JPEG in Redis yet")
                continue

            # Decode JPEG
            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"  [{cam}] Failed to decode JPEG")
                continue

            h, w = frame.shape[:2]
            path = out_dir / cam / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  [{cam}] Saved {w}x{h} -> {path}")

        if i < args.count - 1:
            time.sleep(args.interval)

    # Also check what detections each camera has
    print("\n--- Detection summary (last 20 entries) ---")
    for cam in cameras:
        key = f"iris:meta:{cam}"
        entries = r.xrange(key, count=20)
        total = len(entries)
        with_dets = 0
        for msg_id, fields in entries:
            dets_raw = fields.get(b"dets", b"[]")
            if isinstance(dets_raw, bytes):
                dets_raw = dets_raw.decode("utf-8", errors="replace")
            dets = json.loads(dets_raw)
            if dets:
                with_dets += 1
        print(f"  {cam}: {with_dets}/{total} frames with detections")


if __name__ == "__main__":
    main()
