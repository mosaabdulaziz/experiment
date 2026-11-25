#!/usr/bin/env python3
"""Sequential multi-camera vehicle tracking pipeline.

Processes vehicles through a camera chain (1→2→3) with known topology
and minimum travel-time constraints. Uses YOLO for detection, ResNet50
for appearance embeddings, and stateful ID propagation across cameras.

Usage:
    .venv/bin/python -m mtmc.run_sequential
    .venv/bin/python -m mtmc.run_sequential --min-travel 15 --sim 0.55
    .venv/bin/python -m mtmc.run_sequential --model medium
"""
from __future__ import annotations

import argparse
import time

from .feature_extractor import AppearanceExtractor
from .sequential_tracker import CameraConfig, SequentialTracker
from .yolo_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam1", default="products/el-mohandes/videos/1-D41-enter.mp4",
                   help="Camera 1 (entrance) video")
    p.add_argument("--cam2", default="products/el-mohandes/videos/2-D52-enter.mp4",
                   help="Camera 2 video")
    p.add_argument("--cam3", default="products/el-mohandes/videos/3-CH23-enter.mp4",
                   help="Camera 3 video")
    p.add_argument("--min-travel", type=float, default=20.0,
                   help="Minimum travel time between cameras (seconds)")
    p.add_argument("--sim", type=float, default=0.60,
                   help="Appearance similarity threshold (cosine)")
    p.add_argument("--model", default="small",
                   choices=["nano", "small", "medium", "large", "xlarge"],
                   help="YOLO model size")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="YOLO input size")
    p.add_argument("--skip", type=int, default=1,
                   help="Process every N-th frame (for speed)")
    p.add_argument("--out", default="mtmc/output/sequential",
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("SEQUENTIAL MULTI-CAMERA VEHICLE TRACKING")
    print("=" * 70)
    print(f"  Camera 1: {args.cam1}")
    print(f"  Camera 2: {args.cam2}")
    print(f"  Camera 3: {args.cam3}")
    print(f"  Min travel time: {args.min_travel}s")
    print(f"  Sim threshold: {args.sim:.2f}")
    print(f"  YOLO model: {args.model}")
    print(f"  Detection conf: {args.conf}")
    print(f"  YOLO input size: {args.imgsz}")
    print(f"  Frame skip: {args.skip}")
    print()

    # Build components
    model_name = f"yolo11{args.model[0]}.pt"  # nano→n, small→s, etc.
    detector = VehicleDetector(
        model_path=model_name,
        conf_threshold=args.conf,
        iou_threshold=0.45,
        min_tracklet_frames=5,
        max_lost_frames=8,
        imgsz=args.imgsz,
    )
    extractor = AppearanceExtractor()

    # Camera chain
    camera_chain = [
        CameraConfig(cam_id="D41", video_path=args.cam1),
        CameraConfig(cam_id="D52", video_path=args.cam2),
        CameraConfig(cam_id="CH23", video_path=args.cam3),
    ]

    # Run pipeline
    tracker = SequentialTracker(
        camera_chain=camera_chain,
        detector=detector,
        extractor=extractor,
        min_travel_time_s=args.min_travel,
        sim_threshold=args.sim,
        frame_skip=args.skip,
        output_dir=args.out,
    )

    global_tracks = tracker.run()
    summary = tracker.write_summary()
    log_path = tracker.write_debug_log()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Debug log: {log_path}")


if __name__ == "__main__":
    main()
