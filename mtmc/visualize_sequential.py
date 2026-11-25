#!/usr/bin/env python3
"""Render a 3-panel side-by-side cross-camera tracking video.

Reads the sequential tracking results and produces an MP4 with all 3 cameras
displayed simultaneously, bounding boxes colored by global vehicle ID,
and cross-camera matches highlighted.

Two modes:
  --mode run   (default)  Run the full pipeline, then visualize
  --mode viz              Load summary.json and re-render (skip pipeline)

Usage:
    .venv/bin/python -m mtmc.visualize_sequential --mode run --skip 3
    .venv/bin/python -m mtmc.visualize_sequential --mode viz --skip 3
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PANEL_W, PANEL_H = 640, 360
HEADER_H = 80
OUT_FPS = 10
PAD = 2


def _id_color(gid: int) -> tuple[int, int, int]:
    """Stable BGR color per global ID (golden-ratio hue walk)."""
    hue = ((gid * 0.61803398875) % 1.0) * 179.0
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def _draw_header(canvas: np.ndarray, title: str, t_wall: float,
                 cam_ids: list[str], complete_ids: set[int],
                 partial_ids: set[int]) -> None:
    """Draw the header bar with title, time, and legend."""
    canvas[:] = 40
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"t = {t_wall:6.2f}s", (PANEL_W * len(cam_ids) - 160, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Camera labels
    for ci, cam in enumerate(cam_ids):
        cx = ci * (PANEL_W + PAD) + PANEL_W // 2 - 40
        cv2.putText(canvas, cam, (cx, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)

    # Legend for cross-camera IDs
    all_ids = sorted(complete_ids | partial_ids)
    if all_ids:
        x, y = 10, 66
        cv2.putText(canvas, "matched:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        x += 70
        for gid in all_ids:
            col = _id_color(gid)
            txt = f"id{gid} "
            cv2.rectangle(canvas, (x, y - 10), (x + 10, y + 2), col, -1)
            cv2.putText(canvas, txt, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] + 18
            if x > PANEL_W * len(cam_ids) - 100:
                x, y = 10, 66
    else:
        cv2.putText(canvas, "no cross-camera matches", (10, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)


def _build_boxes_by_cam(global_tracks, cam_ids, fpses, frame_counts):
    """Build per-camera frame→boxes mapping from global tracks.

    Uses mean box for all frames the track is visible.
    """
    boxes_by_cam: dict[str, dict[int, list[tuple[int, np.ndarray, bool]]]] = {
        cid: {} for cid in cam_ids
    }
    complete_ids: set[int] = set()
    partial_ids: set[int] = set()

    for gt in global_tracks:
        if gt.num_cameras >= 2:
            if gt.is_complete:
                complete_ids.add(gt.global_id)
            else:
                partial_ids.add(gt.global_id)

        for ci, cam_id in enumerate(cam_ids):
            if cam_id not in gt.mean_boxes:
                continue
            mean_box = gt.mean_boxes[cam_id]
            start_s = gt.start_times[cam_id]
            end_s = gt.end_times[cam_id]
            start_frame = int(round(start_s * fpses[ci]))
            end_frame = int(round(end_s * fpses[ci]))
            is_cross = gt.num_cameras >= 2
            for f in range(max(0, start_frame), min(end_frame, frame_counts[ci])):
                boxes_by_cam[cam_id].setdefault(f, []).append(
                    (gt.global_id, mean_box, is_cross)
                )

    return boxes_by_cam, complete_ids, partial_ids


def render_sequential_video(
    video_paths: list[str],
    cam_ids: list[str],
    global_tracks: list[GlobalTrack],
    out_mp4: Path,
    title: str = "SEQUENTIAL MULTI-CAMERA TRACKING",
    output_fps: float = OUT_FPS,
) -> None:
    """Render 3-panel video. Reads frames sequentially (no seeking)."""
    from .sequential_tracker import GlobalTrack as _GT  # noqa: ensure type

    n_cams = len(video_paths)
    total_w = PANEL_W * n_cams + PAD * (n_cams - 1)
    total_h = PANEL_H + HEADER_H

    # Open videos and get metadata
    caps = []
    fpses = []
    frame_counts = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {vp}")
        caps.append(cap)
        fpses.append(cap.get(cv2.CAP_PROP_FPS))
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Output frame count based on shortest video
    durations = [fc / f for fc, f in zip(frame_counts, fpses)]
    total_duration = min(durations)
    total_out_frames = int(total_duration * output_fps)
    print(f"  [viz] {total_out_frames} output frames at {output_fps} fps, "
          f"duration {total_duration:.1f}s")

    # Build boxes
    boxes_by_cam, complete_ids, partial_ids = _build_boxes_by_cam(
        global_tracks, cam_ids, fpses, frame_counts
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, output_fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_mp4}")

    # ---- Sequential read strategy ----
    # For each output frame, advance each camera to the right input frame
    # by reading sequentially (cap.read()). We keep a running counter per camera.
    cap_pos = [0] * n_cams  # current frame index in each video
    cached = [None] * n_cams

    # Pre-read first frames
    for ci in range(n_cams):
        ok, frame = caps[ci].read()
        if ok:
            cached[ci] = frame
            cap_pos[ci] = 0

    t0 = time.time()

    for out_i in range(total_out_frames):
        t_wall = out_i / output_fps
        canvas = np.zeros((total_h, total_w, 3), np.uint8)

        # Header
        _draw_header(canvas, title, t_wall, cam_ids, complete_ids, partial_ids)

        # Advance each camera to the correct frame
        for ci in range(n_cams):
            target_in = int(round(t_wall * fpses[ci]))
            target_in = max(0, min(target_in, frame_counts[ci] - 1))

            # Read forward sequentially until we reach the target frame
            while cap_pos[ci] < target_in and cap_pos[ci] < frame_counts[ci] - 1:
                ok, frame = caps[ci].read()
                if not ok:
                    break
                cached[ci] = frame
                cap_pos[ci] += 1

            # Draw panel
            panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            if cached[ci] is not None:
                panel = cv2.resize(cached[ci], (PANEL_W, PANEL_H))

            # Draw boxes for this frame
            for gid, box, is_cross in boxes_by_cam[cam_ids[ci]].get(target_in, []):
                x1, y1, x2, y2 = box.astype(int)
                orig_w = int(caps[ci].get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(caps[ci].get(cv2.CAP_PROP_FRAME_HEIGHT))
                sx, sy = PANEL_W / orig_w, PANEL_H / orig_h
                px1, py1 = int(x1 * sx), int(y1 * sy)
                px2, py2 = int(x2 * sx), int(y2 * sy)
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(PANEL_W, px2), min(PANEL_H, py2)

                if is_cross:
                    color = _id_color(gid)
                    cv2.rectangle(panel, (px1, py1), (px2, py2), color, 3)
                    label = f"id{gid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    lx = max(px1, 0)
                    ly = max(py1 - 4, th + 6)
                    cv2.rectangle(panel, (lx, ly - th - 6), (lx + tw + 8, ly + 2), color, -1)
                    cv2.putText(panel, label, (lx + 4, ly - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(panel, (px1, py1), (px2, py2), (120, 120, 120), 2)

            x_off = ci * (PANEL_W + PAD)
            canvas[HEADER_H:HEADER_H + PANEL_H, x_off:x_off + PANEL_W] = panel

        writer.write(canvas)
        if out_i % 200 == 0 and out_i > 0:
            elapsed = time.time() - t0
            speed = out_i / elapsed
            eta = (total_out_frames - out_i) / speed
            print(f"  [viz] {out_i}/{total_out_frames} frames "
                  f"({100 * out_i / total_out_frames:.0f}%, {speed:.1f} fps, ETA {eta:.0f}s)")

    for cap in caps:
        cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  [viz] Done: {out_mp4} ({total_out_frames} frames, {elapsed:.1f}s)")


def _load_summary(summary_path: Path):
    """Load tracks from summary.json."""
    from .sequential_tracker import GlobalTrack
    data = json.loads(summary_path.read_text())
    tracks = []
    for t in data["tracks"]:
        gt = GlobalTrack(
            global_id=t["global_id"],
            camera_ids=t["cameras"],
            local_ids={k: v for k, v in t["local_ids"].items()},
            start_times={k: v for k, v in t["start_times"].items()},
            end_times={k: v for k, v in t["end_times"].items()},
            mean_boxes={k: np.array(v) for k, v in t.get("mean_boxes", {}).items()},
            is_complete=t.get("is_complete", False),
        )
        tracks.append(gt)
    return tracks, data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["run", "viz"], default="run",
                   help="'run' = full pipeline + viz; 'viz' = load cached summary")
    p.add_argument("--cam1", default="products/el-mohandes/videos/1-D41-enter.mp4")
    p.add_argument("--cam2", default="products/el-mohandes/videos/2-D52-enter.mp4")
    p.add_argument("--cam3", default="products/el-mohandes/videos/3-CH23-enter.mp4")
    p.add_argument("--min-travel", type=float, default=20.0)
    p.add_argument("--sim", type=float, default=0.60)
    p.add_argument("--model", default="small",
                    choices=["nano", "small", "medium", "large", "xlarge"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--out", default=None, help="Output mp4 path")
    args = p.parse_args()

    cam_ids = ["D41", "D52", "CH23"]
    video_paths = [args.cam1, args.cam2, args.cam3]
    out_dir = Path("mtmc/output/sequential")
    out_path = Path(args.out or out_dir / "visualize.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEQUENTIAL TRACKING VISUALIZATION")
    print(f"  Mode: {args.mode}")
    print("=" * 70)

    if args.mode == "run":
        from .feature_extractor import AppearanceExtractor
        from .sequential_tracker import CameraConfig, SequentialTracker
        from .yolo_detector import VehicleDetector

        model_name = f"yolo11{args.model[0]}.pt"
        detector = VehicleDetector(
            model_path=model_name,
            conf_threshold=args.conf,
            iou_threshold=0.45,
            min_tracklet_frames=5,
            max_lost_frames=8,
            imgsz=args.imgsz,
        )
        extractor = AppearanceExtractor()

        camera_chain = [
            CameraConfig(cam_id=cid, video_path=vp)
            for cid, vp in zip(cam_ids, video_paths)
        ]

        print("Running sequential tracking pipeline...")
        tracker = SequentialTracker(
            camera_chain=camera_chain,
            detector=detector,
            extractor=extractor,
            min_travel_time_s=args.min_travel,
            sim_threshold=args.sim,
            frame_skip=args.skip,
            output_dir=str(out_dir),
        )
        global_tracks = tracker.run()
        tracker.write_summary()
        tracker.write_debug_log()

    else:  # viz mode
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            print(f"Error: {summary_path} not found. Run with --mode run first.")
            return
        print(f"Loading cached results from {summary_path}")
        global_tracks, data = _load_summary(summary_path)
        # Reconstruct mean_boxes from video metadata (empty for viz-only)
        print(f"  Loaded {len(global_tracks)} tracks")

    vis_tracks = [gt for gt in global_tracks if gt.num_cameras >= 2]
    n_complete = sum(1 for gt in global_tracks if gt.is_complete)
    n_partial = sum(1 for gt in global_tracks if gt.num_cameras >= 2 and not gt.is_complete)
    print(f"  {n_complete} complete (1→2→3), {n_partial} partial (2 cams)")

    title = (f"TRACKING — {n_complete} complete, {n_partial} partial, "
             f"sim>={args.sim:.2f}, travel>={args.min_travel:.0f}s")

    render_sequential_video(
        video_paths=video_paths,
        cam_ids=cam_ids,
        global_tracks=global_tracks,
        out_mp4=out_path,
        title=title,
    )
    print(f"\nAnnotated video: {out_path}")


if __name__ == "__main__":
    main()
