#!/usr/bin/env python3
"""Capture live detections from Redis and annotate raw RTSP video.

This bypasses DeepStream's JPEG preview entirely:
  1. Reads raw RTSP streams via OpenCV (no DeepStream overhead)
  2. Reads detections from Redis (iris:meta:<cam>)
  3. Draws bounding boxes + track IDs on the raw frames
  4. Writes annotated video for offline comparison

Usage:
    .venv/bin/python -m mtmc.capture_live_detections --duration 60
    .venv/bin/python -m mtmc.capture_live_detections --duration 120 --cam cam-1
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import redis


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RTSP_URIS = {
    "cam-1": "rtsp://localhost:8554/raw-cam-1",
    "cam-2": "rtsp://localhost:8554/raw-cam-2",
    "cam-3": "rtsp://localhost:8554/raw-cam-3",
}

CAMERA_NAMES = {"cam-1": "D41", "cam-2": "D52", "cam-3": "CH23"}

# Detection format from Redis: [label, conf, obj_id, left, top, width, height, ...]
VEHICLE_CLASSES = {"vehicle", "license_plate", "plate", "car", "truck", "bus", "motorcycle", "bicycle"}


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
def id_color(tid: int) -> tuple[int, int, int]:
    """Deterministic BGR color from track ID."""
    hue = ((tid * 0.61803398875) % 1.0) * 179.0
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


# ---------------------------------------------------------------------------
# RTSP reader (background thread)
# ---------------------------------------------------------------------------
class RTSPReader:
    def __init__(self, uri: str, name: str):
        self.uri = uri
        self.name = name
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def _loop(self):
        while self._running:
            self._cap = cv2.VideoCapture(self.uri)
            if not self._cap.isOpened():
                time.sleep(1)
                continue
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
            if self._cap:
                self._cap.release()

    def get_frame(self):
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None


# ---------------------------------------------------------------------------
# Redis detection reader
# ---------------------------------------------------------------------------
class RedisDetectionReader:
    """Reads detections from Redis streams for each camera."""

    def __init__(self, redis_url: str, cameras: list[str]):
        self.r = redis.Redis.from_url(redis_url, decode_responses=False)
        self.cameras = cameras
        self._last_id: dict[str, str] = {c: "0" for c in cameras}
        self._latest_dets: dict[str, list] = {c: [] for c in cameras}
        self._latest_pts: dict[str, float] = {c: 0.0 for c in cameras}
        self._latest_frame: dict[str, int] = {c: 0 for c in cameras}
        self._frame_count: dict[str, int] = {c: 0 for c in cameras}
        self._det_count: dict[str, int] = {c: 0 for c in cameras}

    def poll(self):
        """Poll Redis for new detection messages."""
        for cam in self.cameras:
            key = f"iris:meta:{cam}"
            try:
                entries = self.r.xread({key.encode(): self._last_id[cam].encode()},
                                       count=50, block=10)
            except Exception:
                continue
            if not entries:
                continue
            for _name, messages in entries:
                for msg_id, fields in messages:
                    self._last_id[cam] = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    self._frame_count[cam] += 1

                    # Parse dets (handle bytes)
                    dets_raw = fields.get(b"dets", fields.get("dets", b"[]"))
                    if isinstance(dets_raw, bytes):
                        dets_raw = dets_raw.decode("utf-8", errors="replace")
                    try:
                        dets = json.loads(dets_raw)
                    except json.JSONDecodeError:
                        dets = []

                    self._latest_dets[cam] = dets
                    if dets:
                        self._det_count[cam] += 1

                    # Parse pts
                    pts_raw = fields.get(b"pts", fields.get("pts", b"0"))
                    if isinstance(pts_raw, bytes):
                        pts_raw = pts_raw.decode()
                    try:
                        self._latest_pts[cam] = float(pts_raw)
                    except ValueError:
                        pass

                    # Parse frame_num
                    fn_raw = fields.get(b"frame_num", fields.get("frame_num", b"0"))
                    if isinstance(fn_raw, bytes):
                        fn_raw = fn_raw.decode()
                    try:
                        self._latest_frame[cam] = int(fn_raw)
                    except ValueError:
                        pass

    def get_dets(self, cam: str) -> list:
        """Return latest detections for a camera."""
        return self._latest_dets.get(cam, [])

    def get_stats(self, cam: str) -> dict:
        """Return stats for a camera."""
        total = self._frame_count.get(cam, 0)
        with_dets = self._det_count.get(cam, 0)
        return {
            "frames_polled": total,
            "frames_with_dets": with_dets,
            "detection_rate": with_dets / max(total, 1),
        }


# ---------------------------------------------------------------------------
# Draw detections on frame
# ---------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, dets: list, mux_w: int, mux_h: int,
                    cam_name: str, frame_num: int, stats: dict) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    h, w = frame.shape[:2]
    scale_x = w / mux_w
    scale_y = h / mux_h

    annotated = frame.copy()

    n_dets = 0
    for det in dets:
        if len(det) < 7:
            continue
        label, conf, obj_id = det[0], det[1], int(det[2])
        left, top, bw, bh = det[3], det[4], det[5], det[6]

        # Convert to pixel coords
        x1 = int(left * scale_x)
        y1 = int(top * scale_y)
        x2 = int((left + bw) * scale_x)
        y2 = int((top + bh) * scale_y)

        # Clamp to frame
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = id_color(obj_id)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label_text = f"ID:{obj_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        n_dets += 1

    # HUD
    det_rate = stats.get("detection_rate", 0)
    hud = f"{cam_name} | Frame {frame_num} | Dets: {n_dets} | Rate: {det_rate:.1%}"
    cv2.putText(annotated, hud, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redis", default="redis://localhost:6379/0",
                   help="Redis URL (default: redis://localhost:6379/0)")
    p.add_argument("--cam", nargs="*", default=None,
                   help="Cameras to capture (default: all 3)")
    p.add_argument("--duration", type=float, default=60,
                   help="Recording duration in seconds")
    p.add_argument("--fps", type=int, default=10,
                   help="Output video FPS")
    p.add_argument("--out-dir", default="mtmc/output/live_detections",
                   help="Output directory")
    p.add_argument("--mux-width", type=int, default=1280,
                   help="Mux width (matches sidecar config)")
    p.add_argument("--mux-height", type=int, default=720,
                   help="Mux height (matches sidecar config)")
    args = p.parse_args()

    cameras = args.cam or ["cam-1", "cam-2", "cam-3"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing live detections for {args.duration}s...")
    print(f"  Cameras: {cameras}")
    print(f"  Redis: {args.redis}")
    print(f"  Output: {out_dir}")
    print(f"  Mux: {args.mux_width}x{args.mux_height}")
    print()

    # Start RTSP readers
    readers = {}
    for cam in cameras:
        uri = RTSP_URIS.get(cam)
        if not uri:
            print(f"  [WARN] No RTSP URI for {cam}, skipping")
            continue
        r = RTSPReader(uri, cam)
        r.start()
        readers[cam] = r
        time.sleep(0.3)
        print(f"  [OK] {cam} -> {uri}")

    # Start Redis reader
    redis_reader = RedisDetectionReader(args.redis, cameras)

    # Create video writers
    writers = {}
    frame_counts = {}
    for cam in cameras:
        if cam not in readers:
            continue
        # Get frame size from RTSP
        frame = readers[cam].get_frame()
        if frame is None:
            # Wait a bit for first frame
            for _ in range(30):
                time.sleep(0.5)
                frame = readers[cam].get_frame()
                if frame is not None:
                    break
        if frame is None:
            print(f"  [WARN] Cannot get frame from {cam}, skipping")
            continue

        h, w = frame.shape[:2]
        out_path = out_dir / f"{cam}_live_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
        writers[cam] = writer
        frame_counts[cam] = 0
        print(f"  [OK] {cam} output: {out_path} ({w}x{h} @ {args.fps}fps)")

    print()
    print(f"Recording for {args.duration}s... (Ctrl+C to stop early)")
    print()

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            # Poll Redis
            redis_reader.poll()

            # Draw and write for each camera
            for cam in cameras:
                if cam not in readers or cam not in writers:
                    continue

                frame = readers[cam].get_frame()
                if frame is None:
                    continue

                dets = redis_reader.get_dets(cam)
                stats = redis_reader.get_stats(cam)

                annotated = draw_detections(
                    frame, dets, args.mux_width, args.mux_height,
                    CAMERA_NAMES.get(cam, cam),
                    redis_reader._latest_frame.get(cam, 0),
                    stats,
                )

                writers[cam].write(annotated)
                frame_counts[cam] += 1

            # Progress every 5s
            elapsed = time.time() - t0
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                parts = []
                for cam in cameras:
                    if cam in frame_counts:
                        stats = redis_reader.get_stats(cam)
                        parts.append(f"{cam}: {frame_counts[cam]} frames, "
                                     f"{stats['detection_rate']:.0%} dets")
                print(f"  [{elapsed:.0f}s] {' | '.join(parts)}")

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\nStopped early.")

    # Cleanup
    for r in readers.values():
        r.stop()
    for w in writers.values():
        w.release()

    # Summary
    print()
    print("=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    for cam in cameras:
        if cam in frame_counts:
            stats = redis_reader.get_stats(cam)
            out_path = out_dir / f"{cam}_live_detections.mp4"
            print(f"  {cam}: {frame_counts[cam]} frames -> {out_path}")
            print(f"    Detection rate: {stats['detection_rate']:.1%} "
                  f"({stats['frames_with_dets']}/{stats['frames_polled']} frames)")

    # Save stats
    all_stats = {}
    for cam in cameras:
        all_stats[cam] = redis_reader.get_stats(cam)
    stats_path = out_dir / "capture_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\n  Stats saved to: {stats_path}")

    print("\nTo compare with offline:")
    print(f"  .venv/bin/python -m mtmc.compare_diagnostics \\")
    print(f"    --offline mtmc/output/tracked_videos_v2/1-D41-enter_tracked.mp4 \\")
    print(f"    --online {out_dir}/cam-1_live_detections.mp4")


if __name__ == "__main__":
    main()
