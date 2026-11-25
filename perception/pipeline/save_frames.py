#!/usr/bin/env python3
"""Save DeepStream decoded JPEG frames from Redis to disk for visual inspection."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
OUT_DIR = Path("/workspace/iris/data/frames_dump")
CAMERAS = ["cam-1", "cam-2", "cam-3"]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)

    for cam in CAMERAS:
        cam_dir = OUT_DIR / cam
        cam_dir.mkdir(exist_ok=True)

        # Save the latest JPEG
        jpeg_data = r.get(f"iris:jpeg:{cam}")
        if jpeg_data is None:
            print(f"[{cam}] No JPEG")
            continue
        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{cam}] Bad JPEG")
            continue
        h, w = frame.shape[:2]
        cv2.imwrite(str(cam_dir / "ds_frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Get detection count
        entries = r.xrange(f"iris:meta:{cam}", count=64)
        with_dets = 0
        det_detail = []
        for msg_id, fields in entries:
            dets_raw = fields.get(b"dets", b"[]")
            if isinstance(dets_raw, bytes):
                dets_raw = dets_raw.decode("utf-8", errors="replace")
            dets = json.loads(dets_raw)
            if dets:
                with_dets += 1
                if len(det_detail) < 3:
                    det_detail.extend(dets[:2])

        print(f"[{cam}] {w}x{h} JPEG saved, {with_dets}/{len(entries)} frames with detections")
        for d in det_detail:
            print(f"  {d}")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
