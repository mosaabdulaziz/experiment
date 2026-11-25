#!/usr/bin/env python3
"""Diagnostic: compare DeepStream probe frames vs offline model on same frames.

Pulls the latest JPEG preview from each camera's Redis key, saves them,
then runs the plate detector on those same images to check if the model
CAN detect plates when given the exact pixels DeepStream decoded.

Usage:
    .venv/bin/python perception/pipeline/diagnose_frames.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import redis

# ---- Config ----
REDIS_URL = "redis://localhost:6379/0"
MODEL_PATH = "products/el-mohandes/models/finetuned/best.pt"
OUT_DIR = Path("mtmc/output/diagnose_frames")
CAMERAS = ["cam-1", "cam-2", "cam-3"]
CAMERA_NAMES = {"cam-1": "D41", "cam-2": "D52", "cam-3": "CH23"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)

    # Load the plate model
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    for cam in CAMERAS:
        cam_dir = OUT_DIR / cam
        cam_dir.mkdir(exist_ok=True)

        # Pull JPEG preview
        jpeg_key = f"iris:jpeg:{cam}"
        jpeg_data = r.get(jpeg_key)
        if jpeg_data is None:
            print(f"[{cam}] No JPEG in Redis!")
            continue

        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{cam}] Failed to decode JPEG")
            continue

        h, w = frame.shape[:2]

        # Save the raw frame
        raw_path = cam_dir / "raw_frame.jpg"
        cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Run model offline on the exact same frame
        results = model(frame, imgsz=1280, conf=0.10, verbose=False, device="cuda:0")
        r0 = results[0]
        n_dets = len(r0.boxes) if r0.boxes is not None else 0

        # Draw detections
        annotated = frame.copy()
        if r0.boxes is not None and len(r0.boxes) > 0:
            for box in r0.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ann_path = cam_dir / "offline_detection.jpg"
        cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Get DeepStream detection count
        meta_key = f"iris:meta:{cam}"
        entries = r.xrange(meta_key, count=5)
        ds_dets = 0
        for msg_id, fields in entries:
            dets_raw = fields.get(b"dets", b"[]")
            if isinstance(dets_raw, bytes):
                dets_raw = dets_raw.decode("utf-8", errors="replace")
            dets = json.loads(dets_raw)
            if dets:
                ds_dets += 1

        name = CAMERA_NAMES.get(cam, cam)
        print(f"[{cam} / {name}] {w}x{h} | "
              f"DeepStream: {ds_dets}/5 frames with dets | "
              f"Offline model: {n_dets} detections -> {ann_path}")

    # Also: run offline model on the raw source videos for comparison
    print("\n--- Running offline on source videos for comparison ---")
    VIDEO_DIR = Path("products/el-mohandes/videos/experiment")
    for cam in CAMERAS:
        name = CAMERA_NAMES.get(cam, cam)
        # Find the video file
        video_files = list(VIDEO_DIR.glob(f"*{name}*.mp4"))
        if not video_files:
            print(f"[{cam}] No video found in {VIDEO_DIR}")
            continue
        video_path = video_files[0]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{cam}] Cannot open {video_path}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read first frame with a plate (scan up to 200 frames)
        found = False
        for i in range(min(200, total)):
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=1280, conf=0.10, verbose=False, device="cuda:0")
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Save this frame
                annotated = frame.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cam_dir = OUT_DIR / cam
                cam_dir.mkdir(exist_ok=True)
                ann_path = cam_dir / "offline_raw_frame.jpg"
                cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[{cam}] Raw video: {w}x{h}, frame {i}, "
                      f"{len(results[0].boxes)} detections -> {ann_path}")
                found = True
                break
        if not found:
            print(f"[{cam}] Raw video: {w}x{h}, NO plates found in first 200 frames")
        cap.release()

    print(f"\nAll outputs in: {OUT_DIR}")
    print("Compare: raw_frame.jpg (what DeepStream decoded) vs "
          "offline_raw_frame.jpg (what the model sees from the source video)")


if __name__ == "__main__":
    main()
