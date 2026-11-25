#!/usr/bin/env python3
"""Capture live detections from Redis and compare with offline annotated videos.

Reads detections from Redis streams for the full video loop duration (~150s),
computes detection rates, confidence distributions, and box statistics,
then compares with the offline tracked videos.

Usage (inside sidecar container):
    python3 perception/pipeline/capture_and_compare.py

Or from host (if Redis is accessible):
    .venv/bin/python perception/pipeline/capture_and_compare.py --redis redis://localhost:6379/0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import redis


CAMERAS = ["cam-1", "cam-2", "cam-3"]
CAMERA_NAMES = {"cam-1": "D41", "cam-2": "D52", "cam-3": "CH23"}
VIDEO_DURATIONS = {"cam-1": 146.15, "cam-2": 145.54, "cam-3": 145.47}


def collect_detection_stats(r, cam, max_entries=1000):
    """Collect detection statistics from Redis for a camera."""
    key = f"iris:meta:{cam}"
    entries = r.xrange(key, count=max_entries)
    
    total = len(entries)
    with_dets = 0
    all_confs = []
    all_boxes = []  # (x, y, w, h) in mux coords
    all_frame_nums = []
    
    for msg_id, fields in entries:
        dets_raw = fields.get(b"dets", b"[]")
        if isinstance(dets_raw, bytes):
            dets_raw = dets_raw.decode("utf-8", errors="replace")
        dets = json.loads(dets_raw)
        
        fn_raw = fields.get(b"frame_num", b"0")
        if isinstance(fn_raw, bytes):
            fn_raw = fn_raw.decode()
        try:
            frame_num = int(fn_raw)
        except ValueError:
            frame_num = 0
        all_frame_nums.append(frame_num)
        
        if dets:
            with_dets += 1
            for det in dets:
                if len(det) >= 7:
                    label, conf, obj_id = det[0], det[1], int(det[2])
                    left, top, bw, bh = det[3], det[4], det[5], det[6]
                    all_confs.append(conf)
                    all_boxes.append((left, top, bw, bh))
    
    # Compute frame range
    if all_frame_nums:
        frame_min = min(all_frame_nums)
        frame_max = max(all_frame_nums)
        frame_span = frame_max - frame_min + 1
    else:
        frame_span = 0
    
    stats = {
        "total_frames": total,
        "frames_with_dets": with_dets,
        "detection_rate": with_dets / max(total, 1),
        "num_detections": len(all_confs),
        "avg_confidence": np.mean(all_confs) if all_confs else 0,
        "min_confidence": np.min(all_confs) if all_confs else 0,
        "max_confidence": np.max(all_confs) if all_confs else 0,
        "median_confidence": np.median(all_confs) if all_confs else 0,
        "frame_span": frame_span,
        "det_per_frame": len(all_confs) / max(total, 1),
    }
    
    # Box statistics
    if all_boxes:
        widths = [b[2] for b in all_boxes]
        heights = [b[3] for b in all_boxes]
        areas = [b[2] * b[3] for b in all_boxes]
        stats["avg_box_width"] = np.mean(widths)
        stats["avg_box_height"] = np.mean(heights)
        stats["avg_box_area"] = np.mean(areas)
        stats["min_box_area"] = np.min(areas)
        stats["max_box_area"] = np.max(areas)
    
    # Confidence distribution
    if all_confs:
        stats["conf_lt_0.3"] = sum(1 for c in all_confs if c < 0.3)
        stats["conf_0.3_to_0.5"] = sum(1 for c in all_confs if 0.3 <= c < 0.5)
        stats["conf_0.5_to_0.7"] = sum(1 for c in all_confs if 0.5 <= c < 0.7)
        stats["conf_0.7_to_0.9"] = sum(1 for c in all_confs if 0.7 <= c < 0.9)
        stats["conf_gte_0.9"] = sum(1 for c in all_confs if c >= 0.9)
    
    return stats


def compute_offline_stats(video_path, model_path, conf_threshold=0.10):
    """Run offline model on video and compute detection statistics."""
    from ultralytics import YOLO
    import cv2
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_confs = []
    all_boxes = []
    frames_with_dets = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, imgsz=1280, conf=conf_threshold, verbose=False, device="cuda:0")
        r0 = results[0]
        
        if r0.boxes is not None and len(r0.boxes) > 0:
            frames_with_dets += 1
            for box in r0.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_confs.append(conf)
                all_boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    frame {frame_idx}/{total_frames}")
    
    cap.release()
    
    stats = {
        "total_frames": frame_idx,
        "frames_with_dets": frames_with_dets,
        "detection_rate": frames_with_dets / max(frame_idx, 1),
        "num_detections": len(all_confs),
        "avg_confidence": np.mean(all_confs) if all_confs else 0,
        "min_confidence": np.min(all_confs) if all_confs else 0,
        "max_confidence": np.max(all_confs) if all_confs else 0,
        "median_confidence": np.median(all_confs) if all_confs else 0,
        "det_per_frame": len(all_confs) / max(frame_idx, 1),
    }
    
    if all_boxes:
        widths = [b[2] for b in all_boxes]
        heights = [b[3] for b in all_boxes]
        areas = [b[2] * b[3] for b in all_boxes]
        stats["avg_box_width"] = np.mean(widths)
        stats["avg_box_height"] = np.mean(heights)
        stats["avg_box_area"] = np.mean(areas)
    
    if all_confs:
        stats["conf_lt_0.3"] = sum(1 for c in all_confs if c < 0.3)
        stats["conf_0.3_to_0.5"] = sum(1 for c in all_confs if 0.3 <= c < 0.5)
        stats["conf_0.5_to_0.7"] = sum(1 for c in all_confs if 0.5 <= c < 0.7)
        stats["conf_0.7_to_0.9"] = sum(1 for c in all_confs if 0.7 <= c < 0.9)
        stats["conf_gte_0.9"] = sum(1 for c in all_confs if c >= 0.9)
    
    return stats


def print_comparison(cam, live_stats, offline_stats):
    """Print a side-by-side comparison."""
    name = CAMERA_NAMES.get(cam, cam)
    print(f"\n{'='*70}")
    print(f"  {cam} ({name})")
    print(f"{'='*70}")
    
    headers = ["Metric", "DeepStream (Live)", "Offline (Same Model)"]
    rows = [
        ("Total frames analyzed", 
         str(live_stats["total_frames"]),
         str(offline_stats["total_frames"])),
        ("Frames with detections",
         f"{live_stats['frames_with_dets']}/{live_stats['total_frames']}",
         f"{offline_stats['frames_with_dets']}/{offline_stats['total_frames']}"),
        ("Detection rate",
         f"{live_stats['detection_rate']:.1%}",
         f"{offline_stats['detection_rate']:.1%}"),
        ("Total detections",
         str(live_stats["num_detections"]),
         str(offline_stats["num_detections"])),
        ("Detections/frame",
         f"{live_stats['det_per_frame']:.2f}",
         f"{offline_stats['det_per_frame']:.2f}"),
        ("Avg confidence",
         f"{live_stats['avg_confidence']:.3f}",
         f"{offline_stats['avg_confidence']:.3f}"),
        ("Median confidence",
         f"{live_stats['median_confidence']:.3f}",
         f"{offline_stats['median_confidence']:.3f}"),
        ("Conf range",
         f"{live_stats['min_confidence']:.3f} - {live_stats['max_confidence']:.3f}",
         f"{offline_stats['min_confidence']:.3f} - {offline_stats['max_confidence']:.3f}"),
    ]
    
    if "avg_box_width" in live_stats and "avg_box_width" in offline_stats:
        rows.extend([
            ("Avg box size (px)",
             f"{live_stats['avg_box_width']:.1f}x{live_stats['avg_box_height']:.1f}",
             f"{offline_stats['avg_box_width']:.1f}x{offline_stats['avg_box_height']:.1f}"),
        ])
    
    if "conf_lt_0.3" in live_stats:
        rows.extend([
            ("Conf < 0.3", str(live_stats["conf_lt_0.3"]), str(offline_stats["conf_lt_0.3"])),
            ("Conf 0.3-0.5", str(live_stats["conf_0.3_to_0.5"]), str(offline_stats["conf_0.3_to_0.5"])),
            ("Conf 0.5-0.7", str(live_stats["conf_0.5_to_0.7"]), str(offline_stats["conf_0.5_to_0.7"])),
            ("Conf 0.7-0.9", str(live_stats["conf_0.7_to_0.9"]), str(offline_stats["conf_0.7_to_0.9"])),
            ("Conf >= 0.9", str(live_stats["conf_gte_0.9"]), str(offline_stats["conf_gte_0.9"])),
        ])
    
    # Print table
    col_widths = [max(len(row[0]), len(row[1]), len(row[2])) for row in rows]
    col_widths[0] = max(col_widths[0], 22)
    col_widths[1] = max(col_widths[1], 20)
    col_widths[2] = max(col_widths[2], 20)
    
    header = f"  {'Metric':<{col_widths[0]}} | {'DeepStream':^{col_widths[1]}} | {'Offline':^{col_widths[2]}}"
    print(header)
    print(f"  {'-'*col_widths[0]}-+-{'-'*col_widths[1]}-+-{'-'*col_widths[2]}")
    
    for metric, live, offline in rows:
        # Color coding
        live_val = live_stats.get(metric.lower().replace(" ", "_").replace("(", "").replace(")", ""), None)
        offline_val = offline_stats.get(metric.lower().replace(" ", "_").replace("(", "").replace(")", ""), None)
        
        print(f"  {metric:<{col_widths[0]}} | {live:^{col_widths[1]}} | {offline:^{col_widths[2]}}")
    
    # Compute gap
    rate_gap = live_stats["detection_rate"] - offline_stats["detection_rate"]
    conf_gap = live_stats["avg_confidence"] - offline_stats["avg_confidence"]
    
    print(f"\n  Gap: detection rate {rate_gap:+.1%}, confidence {conf_gap:+.3f}")
    
    if rate_gap > -0.1:
        print(f"  ✅ Detection rate is within 10% of offline")
    elif rate_gap > -0.2:
        print(f"  ⚠️ Detection rate is 10-20% below offline")
    else:
        print(f"  ❌ Detection rate is >20% below offline")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--redis", default="redis://redis:6379/0",
                   help="Redis URL (default: redis://redis:6379/0 for in-container)")
    p.add_argument("--model", default="products/el-mohandes/models/finetuned/best.pt")
    p.add_argument("--video-dir", default="products/el-mohandes/videos/experiment")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--out", default="mtmc/output/comparison")
    args = p.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    r = redis.Redis.from_url(args.redis, decode_responses=False)
    video_dir = Path(args.video_dir)
    
    print("=" * 70)
    print("  DETECTION QUALITY COMPARISON: DeepStream vs Offline")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Video dir: {video_dir}")
    print()
    
    all_results = {}
    
    for cam in CAMERAS:
        name = CAMERA_NAMES.get(cam, cam)
        
        # 1. Collect live detection stats from Redis
        print(f"\n[{cam}] Collecting live detection stats from Redis...")
        live_stats = collect_detection_stats(r, cam)
        
        # 2. Run offline model on source video
        video_files = list(video_dir.glob(f"*{name}*.mp4"))
        if not video_files:
            print(f"  No video found for {cam} in {video_dir}")
            continue
        video_path = video_files[0]
        
        print(f"[{cam}] Running offline model on {video_path.name}...")
        offline_stats = compute_offline_stats(video_path, args.model, args.conf)
        
        if offline_stats is None:
            print(f"  Failed to process video")
            continue
        
        # 3. Print comparison
        print_comparison(cam, live_stats, offline_stats)
        
        all_results[cam] = {
            "live": live_stats,
            "offline": offline_stats,
        }
    
    # Save results
    results_path = out_dir / "comparison_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for cam in CAMERAS:
        if cam in all_results:
            live = all_results[cam]["live"]
            offline = all_results[cam]["offline"]
            rate_gap = live["detection_rate"] - offline["detection_rate"]
            print(f"  {CAMERA_NAMES[cam]:>6}: Live={live['detection_rate']:.1%} "
                  f"Offline={offline['detection_rate']:.1%} "
                  f"Gap={rate_gap:+.1%} "
                  f"Conf: {live['avg_confidence']:.3f} vs {offline['avg_confidence']:.3f}")


if __name__ == "__main__":
    main()
