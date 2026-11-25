#!/usr/bin/env python3
"""Compare TRT engine (DeepStream) vs ultralytics on the same frames.

Captures frames from the RTSP streams, runs ultralytics on them, and
compares with the raw nvinfer output logged by the diagnostic probe.

This tells us whether the TRT engine and ultralytics produce the same
detections on identical input pixels.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import redis


RTSP_URIS = {
    "cam-1": "rtsp://localhost:8554/raw-cam-1",
    "cam-2": "rtsp://localhost:8554/raw-cam-2",
    "cam-3": "rtsp://localhost:8554/raw-cam-3",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="products/el-mohandes/models/finetuned/best.pt")
    p.add_argument("--raw-log", default="/workspace/iris/data/raw_detections.jsonl")
    p.add_argument("--out", default="/workspace/iris/data/trt_vs_ultra")
    p.add_argument("--n-frames", type=int, default=50, help="Frames to capture per camera")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--redis", default="redis://redis:6379/0")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    # Read raw TRT detections from log
    trt_by_cam = {0: [], 1: [], 2: []}
    with open(args.raw_log) as f:
        for line in f:
            entry = json.loads(line)
            pad = entry["pad"]
            if pad in trt_by_cam:
                trt_by_cam[pad].append(entry)

    r = redis.Redis.from_url(args.redis, decode_responses=False)

    results = {}
    cam_names = {0: "cam-1", 1: "cam-2", 2: "cam-3"}

    for pad, cam in cam_names.items():
        uri = RTSP_URIS[cam]
        print(f"\n=== {cam} ({uri}) ===")

        # Capture frames from RTSP
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            print(f"  Cannot open {uri}")
            continue

        ultra_dets = []  # per-frame: list of (conf, bbox)
        trt_dets = []    # per-frame: list of (conf, bbox)
        frame_idx = 0
        matched = 0

        # Get the latest TRT detections for this pad
        trt_entries = trt_by_cam.get(pad, [])
        # Build a lookup by frame_num
        trt_by_frame = {e["frame"]: e for e in trt_entries}

        while frame_idx < args.n_frames:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Run ultralytics
            results_ul = model(frame, imgsz=1280, conf=args.conf, verbose=False, device="cuda:0")
            r0 = results_ul[0]
            u_dets = []
            if r0.boxes is not None:
                for box in r0.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    u_dets.append({"conf": round(conf, 4), "bbox": [round(float(x1), 1), round(float(y1), 1),
                                                                    round(float(x2 - x1), 1), round(float(y2 - y1), 1)]})

            ultra_dets.append(u_dets)

            # Save frame
            if frame_idx < 5:
                annotated = frame.copy()
                for d in u_dets:
                    x, y, w, h = d["bbox"]
                    cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(annotated, f"U:{d['conf']:.2f}", (int(x), int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Find matching TRT entry (approximate by frame count)
                trt_frame_idx = frame_idx  # rough match
                if trt_frame_idx < len(trt_entries):
                    te = trt_entries[trt_frame_idx]
                    for obj in te["objects"]:
                        bx, by, bw, bh = obj["bbox"]
                        cv2.rectangle(annotated, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 0, 255), 2)
                        cv2.putText(annotated, f"T:{obj['conf']:.2f}", (int(bx), int(by) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                fpath = out_dir / f"{cam}_frame{frame_idx:04d}.jpg"
                cv2.imwrite(str(fpath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_idx += 1

        cap.release()

        # Compare: for frames where TRT has detections, does ultra also have them?
        n_compared = min(len(ultra_dets), len(trt_entries))
        both_det = 0
        trt_only = 0
        ultra_only = 0
        neither = 0
        for i in range(n_compared):
            u_has = len(ultra_dets[i]) > 0
            t_has = trt_entries[i]["n_objects"] > 0
            if u_has and t_has:
                both_det += 1
            elif t_has and not u_has:
                trt_only += 1
            elif u_has and not t_has:
                ultra_only += 1
            else:
                neither += 1

        ultra_rate = sum(1 for d in ultra_dets if d) / max(len(ultra_dets), 1)
        trt_rate = sum(1 for e in trt_entries[:n_compared] if e["n_objects"] > 0) / max(n_compared, 1)

        print(f"  Frames compared: {n_compared}")
        print(f"  Ultra detection rate: {ultra_rate:.1%} ({sum(1 for d in ultra_dets if d)}/{len(ultra_dets)})")
        print(f"  TRT detection rate:   {trt_rate:.1%}")
        print(f"  Both detect:          {both_det} ({both_det/max(n_compared,1)*100:.0f}%)")
        print(f"  TRT only:             {trt_only} ({trt_only/max(n_compared,1)*100:.0f}%)")
        print(f"  Ultra only:           {ultra_only} ({ultra_only/max(n_compared,1)*100:.0f}%)")
        print(f"  Neither:              {neither} ({neither/max(n_compared,1)*100:.0f}%)")

        results[cam] = {
            "n_compared": n_compared,
            "ultra_rate": ultra_rate,
            "trt_rate": trt_rate,
            "both_det": both_det,
            "trt_only": trt_only,
            "ultra_only": ultra_only,
            "neither": neither,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRT vs Ultralytics Summary")
    print(f"{'='*60}")
    for cam, r in results.items():
        print(f"  {cam}: Ultra={r['ultra_rate']:.0%} TRT={r['trt_rate']:.0%} "
              f"both={r['both_det']} trt_only={r['trt_only']} ultra_only={r['ultra_only']}")

    # Save
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'comparison.json'}")
    print(f"Annotated frames in {out_dir}/")


if __name__ == "__main__":
    main()
