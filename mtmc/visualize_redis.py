#!/usr/bin/env python3
"""Render cross-camera tracking visualization from Redis/DeepStream data.

Two modes:
  --mode live     Subscribe to Redis in real-time, render live from RTSP
  --mode replay   Load a captured JSON + video files, render offline

In both modes, vehicles are drawn with per-global-ID colors. Matched vehicles
get the same color across all camera panels.

Usage:
    # Live mode (sidecar must be running)
    .venv/bin/python -m mtmc.visualize_redis --mode live --record 145

    # Offline mode (after recording)
    .venv/bin/python -m mtmc.visualize_redis --mode replay
"""
from __future__ import annotations

import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import redis

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MUX_W = 1280
MUX_H = 720
PANEL_W = 640
PANEL_H = 360
HEADER_H = 80
PAD = 2
OUT_FPS = 10

VEHICLE_CLASSES = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle"}

CAMERA_CHAIN = ["cam-1", "cam-2", "cam-3"]
CAMERA_NAMES = {"cam-1": "D41", "cam-2": "D52", "cam-3": "CH23"}
RTSP_URIS = {
    "cam-1": "rtsp://localhost:8554/raw-cam-1",
    "cam-2": "rtsp://localhost:8554/raw-cam-2",
    "cam-3": "rtsp://localhost:8554/raw-cam-3",
}


# ---------------------------------------------------------------------------
# Color palette — same function used everywhere
# ---------------------------------------------------------------------------
def _id_color(gid: int) -> tuple[int, int, int]:
    """Stable BGR color per global ID (golden-ratio hue walk)."""
    hue = ((gid * 0.61803398875) % 1.0) * 179.0
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


# ---------------------------------------------------------------------------
# RTSP reader (background thread)
# ---------------------------------------------------------------------------
class RTSPReader:
    def __init__(self, uri, name):
        self.uri = uri
        self.name = name
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

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
# Redis frame reader — reads detections, manages local tracks + cross-cam match
# ---------------------------------------------------------------------------
class RedisFrameReader:
    """Reads detections from Redis, tracks vehicles per camera, matches across cameras."""

    def __init__(self, redis_url, cameras, precomputed_gid_map=None):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.cameras = cameras
        self._last_id = {c: "0" for c in cameras}
        # Per-cam: {object_id: {label, first_seen, last_seen, disappeared}}
        self._tracks = {c: {} for c in cameras}
        self._finished = {c: [] for c in cameras}
        # Latest detections per camera
        self._latest_dets = {c: [] for c in cameras}
        # Global ID mapping: {(cam_id, track_id): global_id}
        self._gid_map = dict(precomputed_gid_map) if precomputed_gid_map else {}
        self._next_gid = max(self._gid_map.values(), default=-1) + 1
        self._min_travel_s = 20.0
        self._consumed = {c: set() for c in cameras}
        self._t0 = time.time()  # pipeline start for wall-clock time

    def poll(self):
        """Poll Redis for new messages across all cameras."""
        for cam in self.cameras:
            key = "iris:meta:" + cam
            try:
                entries = self.r.xread({key: self._last_id[cam]}, count=100)
            except Exception:
                continue
            if not entries:
                continue
            for _name, messages in entries:
                for msg_id, fields in messages:
                    self._last_id[cam] = msg_id
                    dets_raw = fields.get("dets", "[]")
                    pts = float(fields.get("pts", 0))
                    dets = json.loads(dets_raw)
                    # Use wall-clock time instead of PTS for travel-time constraints
                    now = time.time() - self._t0

                    active_ids = set()
                    for det in dets:
                        if len(det) < 3:
                            continue
                        label, conf, obj_id = det[0], det[1], int(det[2])
                        if label not in VEHICLE_CLASSES:
                            continue
                        active_ids.add(obj_id)

                        if obj_id in self._tracks[cam]:
                            self._tracks[cam][obj_id]["last_seen"] = now
                        else:
                            self._tracks[cam][obj_id] = {
                                "label": label,
                                "first_seen": now,
                                "last_seen": now,
                                "disappeared": None,
                            }

                    # Mark disappeared
                    for obj_id, trk in list(self._tracks[cam].items()):
                        if obj_id not in active_ids:
                            if trk["disappeared"] is None:
                                trk["disappeared"] = now
                            if now - trk["last_seen"] > 5.0:
                                self._finished[cam].append(
                                    (obj_id, self._tracks[cam].pop(obj_id))
                                )

                    self._latest_dets[cam] = dets

    def try_match(self):
        """Try cross-camera matching. Returns new match decisions."""
        new_matches = []
        for ci in range(1, len(self.cameras)):
            src, tgt = self.cameras[ci - 1], self.cameras[ci]
            for src_id, src_trk in list(self._finished[src]):
                if src_id in self._consumed[src]:
                    continue
                if (src, src_id) in self._gid_map:
                    continue
                src_end = src_trk["disappeared"] or src_trk["last_seen"]

                # Check both active and finished target tracks
                tgt_candidates = dict(self._tracks[tgt].items())
                for tgt_id, tgt_trk_finished in self._finished[tgt]:
                    if tgt_id not in tgt_candidates:
                        tgt_candidates[tgt_id] = tgt_trk_finished

                for tgt_id, tgt_trk in tgt_candidates.items():
                    if tgt_id in self._consumed[tgt]:
                        continue
                    time_diff = tgt_trk["first_seen"] - src_end
                    if time_diff < self._min_travel_s:
                        continue

                    # Assign global ID
                    src_gid = self._gid_map.get((src, src_id))
                    if src_gid is None:
                        src_gid = self._next_gid
                        self._next_gid += 1
                    self._gid_map[(src, src_id)] = src_gid
                    self._gid_map[(tgt, tgt_id)] = src_gid
                    self._consumed[src].add(src_id)

                    new_matches.append({
                        "global_id": src_gid,
                        "source_cam": src,
                        "source_id": src_id,
                        "target_cam": tgt,
                        "target_id": tgt_id,
                        "time_diff": time_diff,
                    })
        return new_matches

    def get_dets_for_drawing(self, cam):
        """Return (object_id, label, conf, left, top, width, height) for current detections."""
        result = []
        for det in self._latest_dets.get(cam, []):
            if len(det) >= 3 and det[0] in VEHICLE_CLASSES:
                item = (int(det[2]), det[0], det[1])
                if len(det) >= 7:
                    item = (int(det[2]), det[0], det[1], det[3], det[4], det[5], det[6])
                result.append(item)
        return result

    def get_global_id(self, cam, track_id):
        # Check precomputed map first, then live match map
        return self._gid_map.get((cam, track_id))


# ---------------------------------------------------------------------------
# Header drawing
# ---------------------------------------------------------------------------
def _draw_header(canvas, title, t_wall, cam_ids, n_matched):
    canvas[:] = 40
    n_cams = len(cam_ids)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "t = {:.2f}s".format(t_wall),
                (PANEL_W * n_cams - 160, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "matched: {}".format(n_matched),
                (PANEL_W * n_cams - 160, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for ci, cam in enumerate(cam_ids):
        cx = ci * (PANEL_W + PAD) + PANEL_W // 2 - 40
        name = CAMERA_NAMES.get(cam, cam)
        cv2.putText(canvas, name, (cx, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Live render
# ---------------------------------------------------------------------------
def render_live(redis_reader, cameras, rtsp_uris, out_mp4, duration_s):
    """Render live from RTSP + Redis overlays."""
    readers = {}
    for cam in cameras:
        r = RTSPReader(rtsp_uris.get(cam, ""), cam)
        r.start()
        readers[cam] = r
        time.sleep(0.5)

    n_cams = len(cameras)
    total_w = PANEL_W * n_cams + PAD * (n_cams - 1)
    total_h = PANEL_H + HEADER_H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, OUT_FPS, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError("Cannot open writer: " + str(out_mp4))

    t0 = time.time()
    frame_count = 0
    all_matches = []

    print("Recording for {}s...".format(duration_s))

    while time.time() - t0 < duration_s:
        redis_reader.poll()
        new_matches = redis_reader.try_match()
        all_matches.extend(new_matches)
        for m in new_matches:
            print("  Match: gid={} {}:{} -> {}:{} (travel={:.1f}s)".format(
                m["global_id"], m["source_cam"], m["source_id"],
                m["target_cam"], m["target_id"], m["time_diff"]))

        t_wall = time.time() - t0
        canvas = np.zeros((total_h, total_w, 3), np.uint8)

        gids = set(redis_reader._gid_map.values())
        _draw_header(canvas, "MULTI-CAMERA TRACKING", t_wall, cameras, len(gids))

        for ci, cam in enumerate(cameras):
            frame = readers[cam].get_frame()
            if frame is not None:
                panel = cv2.resize(frame, (PANEL_W, PANEL_H))
            else:
                panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)

            # Draw bounding boxes from Redis detections
            scale_x = PANEL_W / MUX_W
            scale_y = PANEL_H / MUX_H
            dets = redis_reader.get_dets_for_drawing(cam)
            for det_info in dets:
                obj_id = det_info[0]
                label = det_info[1]
                conf = det_info[2]
                gid = redis_reader.get_global_id(cam, obj_id)

                if len(det_info) >= 7:
                    left, top, w, h = det_info[3], det_info[4], det_info[5], det_info[6]
                    x1 = int(left * scale_x)
                    y1 = int(top * scale_y)
                    x2 = int((left + w) * scale_x)
                    y2 = int((top + h) * scale_y)

                    if gid is not None:
                        color = _id_color(gid)
                        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(panel, "G{}".format(gid),
                                    (x1, max(y1 - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(panel, (x1, y1), (x2, y2), (128, 128, 128), 1)

            x_off = ci * (PANEL_W + PAD)
            canvas[HEADER_H:HEADER_H + PANEL_H, x_off:x_off + PANEL_W] = panel

        writer.write(canvas)
        frame_count += 1

        if frame_count % 100 == 0:
            elapsed = time.time() - t0
            print("  [{:.0f}s] frames={}, matches={}, gids={}".format(
                elapsed, frame_count, len(all_matches), len(gids)))

        time.sleep(1.0 / OUT_FPS)

    for r in readers.values():
        r.stop()
    writer.release()

    print("\nDone: {} ({} frames)".format(out_mp4, frame_count))
    match_path = out_mp4.parent / "live_matches.json"
    match_path.write_text(json.dumps(all_matches, indent=2))
    print("Matches: {}".format(match_path))
    return all_matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["live", "replay"], default="live")
    p.add_argument("--redis", default="redis://172.19.0.5:6379/0")
    p.add_argument("--cameras", nargs="+", default=CAMERA_CHAIN)
    p.add_argument("--record", type=float, default=145,
                   help="Duration to record in seconds")
    p.add_argument("--min-travel", type=float, default=20.0)
    p.add_argument("--summary", default=None,
                   help="Path to summary.json from mtmc_redis.py for pre-computed global IDs")
    p.add_argument("--out", default="mtmc/output/redis_mtmc")
    args = p.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed global IDs if summary provided
    precomputed_gid = None
    if args.summary:
        summary_path = Path(args.summary)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            precomputed_gid = {}
            for track in summary.get("tracks", []):
                gid = track["global_id"]
                for cam, tid in track["track_ids"].items():
                    precomputed_gid[(cam, int(tid))] = gid
            print("Loaded {} global ID mappings from {}".format(
                len(precomputed_gid), summary_path))

    if args.mode == "live":
        reader = RedisFrameReader(args.redis, args.cameras, precomputed_gid)
        reader._min_travel_s = args.min_travel
        out_mp4 = output_dir / "redis_tracking_live.mp4"
        render_live(
            redis_reader=reader,
            cameras=args.cameras,
            rtsp_uris=RTSP_URIS,
            out_mp4=out_mp4,
            duration_s=args.record,
        )
    else:
        print("Replay mode: load captured data and render")


if __name__ == "__main__":
    main()
