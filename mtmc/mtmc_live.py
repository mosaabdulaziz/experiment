#!/usr/bin/env python3
"""Live multi-camera tracker + visualization combined.

Runs tracker and video rendering in the same process so they share
the same track IDs. Subscribes to Redis detections from DeepStream,
crops vehicles from RTSP, extracts ResNet50 appearance embeddings,
matches across cameras (D41→D52→CH23), and renders a 3-panel annotated
video with colored bounding boxes for matched vehicles.

Usage:
    .venv/bin/python -m mtmc.mtmc_live --record 145
"""
from __future__ import annotations

import argparse
import json
import random
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import redis
from ultralytics import YOLO

from .feature_extractor import AppearanceExtractor
from .yolo_detector import PlateVehicleDetector
from .detection_postprocess import containment_suppression

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_REDIS_URL = "redis://172.19.0.5:6379/0"
CAMERA_CHAIN = ["cam-1", "cam-2", "cam-3"]
CAMERA_NAMES = {"cam-1": "D41", "cam-2": "D52", "cam-3": "CH23"}
RTSP_URIS = {
    "cam-1": "rtsp://localhost:8554/raw-cam-1",
    "cam-2": "rtsp://localhost:8554/raw-cam-2",
    "cam-3": "rtsp://localhost:8554/raw-cam-3",
}
MUX_W, MUX_H = 1280, 720
PANEL_W, PANEL_H = 640, 360
HEADER_H = 80
PAD = 2
OUT_FPS = 10
VEHICLE_CLASSES = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TrackedVehicle:
    cam_id: str
    track_id: int
    label: str
    first_seen_s: float
    last_seen_s: float
    boxes: list  # list of [left, top, width, height] in mux coords
    embeddings: list = field(default_factory=list)
    disappeared_s: float | None = None

    @property
    def mean_embedding(self):
        if not self.embeddings:
            return None
        mean = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None


@dataclass
class GlobalMatch:
    global_id: int
    cameras: list = field(default_factory=list)
    track_ids: dict = field(default_factory=dict)
    start_times: dict = field(default_factory=dict)
    end_times: dict = field(default_factory=dict)

    @property
    def num_cameras(self):
        return len(self.cameras)


# ---------------------------------------------------------------------------
# Color palette
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
    def __init__(self, uri, cam_id):
        self.uri = uri
        self.cam_id = cam_id
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
            cap = cv2.VideoCapture(self.uri)
            if not cap.isOpened():
                time.sleep(1)
                continue
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
            cap.release()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def _extract_crop(frame, left, top, width, height,
                  mux_w=MUX_W, mux_h=MUX_H, pad=0.15):
    if frame is None:
        return None
    fh, fw = frame.shape[:2]
    sx, sy = fw / mux_w, fh / mux_h
    px, py = width * pad, height * pad
    x1 = max(0, int((left - px) * sx))
    y1 = max(0, int((top - py) * sy))
    x2 = min(fw, int((left + width + px) * sx))
    y2 = min(fh, int((top + height + py) * sy))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ---------------------------------------------------------------------------
# Redis collector
# ---------------------------------------------------------------------------
class RedisCollector:
    def __init__(self, redis_url, cameras):
        self.r = redis.Redis.from_url(redis_url, decode_responses=False)
        self.cameras = cameras
        self._last_id = {c: "0" for c in cameras}
        self._active = {c: {} for c in cameras}
        self._finished = {c: [] for c in cameras}

    def _decode(self, val):
        return val.decode() if isinstance(val, bytes) else val

    def poll_once(self):
        result = {}
        for cam in self.cameras:
            key = f"iris:meta:{cam}"
            try:
                entries = self.r.xread({key: self._last_id[cam]}, count=50)
            except redis.RedisError:
                continue
            if not entries:
                continue
            for _name, messages in entries:
                for msg_id, fields in messages:
                    self._last_id[cam] = msg_id
                    dets_raw = self._decode(fields.get(b"dets", b"[]"))
                    pts = float(self._decode(fields.get(b"pts", b"0")))
                    dets = json.loads(dets_raw)
                    result.setdefault(cam, []).append({
                        "pts": pts, "dets": dets,
                    })
        return result

    def update_tracks(self, cam_id, dets, pts, wall_time=None):
        now = wall_time if wall_time is not None else time.time()
        active = self._active[cam_id]
        seen_ids = set()

        for det in dets:
            if len(det) < 7:
                continue
            label, conf, obj_id = det[0], det[1], int(det[2])
            left, top, width, height = det[3], det[4], det[5], det[6]

            if label not in VEHICLE_CLASSES:
                continue
            seen_ids.add(obj_id)

            if obj_id in active:
                tv = active[obj_id]
                tv.last_seen_s = now
                tv.boxes.append([left, top, width, height])
            else:
                tv = TrackedVehicle(
                    cam_id=cam_id, track_id=obj_id, label=label,
                    first_seen_s=now, last_seen_s=now,
                    boxes=[[left, top, width, height]],
                )
                active[obj_id] = tv

        disappeared = []
        for obj_id, tv in list(active.items()):
            if obj_id not in seen_ids:
                if tv.disappeared_s is None:
                    tv.disappeared_s = now
                if now - tv.last_seen_s > 5.0:
                    disappeared.append(obj_id)
        for obj_id in disappeared:
            self._finished[cam_id].append(active.pop(obj_id))

    def get_active(self, cam_id):
        return self._active[cam_id]

    def get_finished(self, cam_id):
        return self._finished[cam_id]


# ---------------------------------------------------------------------------
# Plate-based detector collector (replaces Redis/DeepStream)
# ---------------------------------------------------------------------------
class PlateDetectorCollector:
    """Run plate detection directly on RTSP frames using ByteTrack.

    Uses ultralytics built-in ByteTrack (Kalman filter motion model,
    two-stage IoU matching) instead of a hand-rolled greedy matcher.
    This eliminates flickering from missed detections.
    """

    def __init__(self, cameras: list[str], model_path: str | None = None,
                 conf: float = 0.25, imgsz: int = 1280, device: str | None = None,
                 containment: float = 0.85):
        self.cameras = cameras
        self.model = YOLO(model_path or "products/el-mohandes/models/finetuned/best.pt")
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.containment = containment
        # Per-camera state: ByteTrack IDs → TrackedVehicle
        self._active: dict[str, dict[int, TrackedVehicle]] = {c: {} for c in cameras}
        self._finished: dict[str, list[TrackedVehicle]] = {c: [] for c in cameras}

    def detect_and_track(self, cam_id: str, frame: np.ndarray, wall_time: float,
                         mux_w: int = MUX_W, mux_h: int = MUX_H):
        """Run plate detection + ByteTrack on frame, update tracks for this camera."""
        if frame is None:
            return

        fh, fw = frame.shape[:2]
        sx, sy = mux_w / fw, mux_h / fh

        # ByteTrack: persist=True keeps Kalman filter state between calls
        results = self.model.track(
            frame, imgsz=self.imgsz, conf=self.conf,
            iou=0.45, tracker="bytetrack.yaml",
            persist=True, verbose=False, device=self.device,
        )[0]

        active = self._active[cam_id]
        seen_ids: set[int] = set()

        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            # Keep only the outer box when a cab/part box sits fully inside the
            # whole-vehicle box (same class, IoU under NMS bar).
            keep = containment_suppression(boxes_xyxy, self.containment, confs)
            ids, boxes_xyxy = ids[keep], boxes_xyxy[keep]

            for bt_id, box_xyxy in zip(ids, boxes_xyxy):
                x1, y1, x2, y2 = box_xyxy
                # Convert to mux xywh
                mux_box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]

                if bt_id in active:
                    # Update existing track
                    active[bt_id].last_seen_s = wall_time
                    active[bt_id].boxes.append(mux_box)
                    active[bt_id].disappeared_s = None  # reappeared
                else:
                    # New track from ByteTrack
                    tv = TrackedVehicle(
                        cam_id=cam_id, track_id=int(bt_id), label="vehicle",
                        first_seen_s=wall_time, last_seen_s=wall_time,
                        boxes=[mux_box],
                    )
                    active[bt_id] = tv
                seen_ids.add(bt_id)

        # Age tracks not seen this frame
        disappeared = []
        for tid, tv in list(active.items()):
            if tid not in seen_ids:
                if tv.disappeared_s is None:
                    tv.disappeared_s = wall_time
                # ByteTrack handles Kalman prediction for ~30 frames (~3s).
                # We add a generous timeout for truly lost tracks.
                if wall_time - tv.last_seen_s > 8.0:
                    disappeared.append(tid)
        for tid in disappeared:
            self._finished[cam_id].append(active.pop(tid))

    def get_active(self, cam_id):
        return self._active[cam_id]

    def get_finished(self, cam_id):
        return self._finished[cam_id]


# ---------------------------------------------------------------------------
# Header drawing
# ---------------------------------------------------------------------------
def _draw_header(canvas, title, t_wall, cam_ids, n_matched, n_total):
    canvas[:] = 40
    n_cams = len(cam_ids)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "t = {:.2f}s".format(t_wall),
                (PANEL_W * n_cams - 160, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "matched: {}  total: {}".format(n_matched, n_total),
                (PANEL_W * n_cams - 200, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for ci, cam in enumerate(cam_ids):
        cx = ci * (PANEL_W + PAD) + PANEL_W // 2 - 40
        name = CAMERA_NAMES.get(cam, cam)
        cv2.putText(canvas, name, (cx, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Combined tracker + renderer
# ---------------------------------------------------------------------------
class MergedCollector:
    """Runs both plate detector and generic (Redis/DeepStream) detections.

    Plate detector: primary vehicle tracker (each plate = one vehicle).
    Generic detector: provides non-vehicle classes (pedestrians, cyclists, etc.)
    and acts as fallback for vehicles without visible plates.

    Merge strategy:
      - Plate-detected vehicles are tracked via IoU (primary)
      - Generic vehicle detections that overlap with plate tracks are suppressed
      - Generic non-vehicle detections (pedestrian, cyclist, etc.) are kept
    """

    def __init__(self, redis_url, cameras, plate_model=None, plate_conf=0.25,
                 plate_imgsz=1280, plate_device=None):
        self.redis_collector = RedisCollector(redis_url, cameras)
        self.plate_detector = PlateVehicleDetector(
            model_path=plate_model or "products/el-mohandes/models/finetuned/best.pt",
            conf_threshold=plate_conf, imgsz=plate_imgsz, device=plate_device,
        )
        self.cameras = cameras
        # ByteTrack handles tracking state internally (Kalman filter)
        # Per-camera track ID → TrackedVehicle mapping
        self._plate_active: dict[str, dict[int, TrackedVehicle]] = {c: {} for c in cameras}
        self._plate_finished: dict[str, list[TrackedVehicle]] = {c: [] for c in cameras}

    def poll_once(self):
        """Poll Redis for generic detections."""
        return self.redis_collector.poll_once()

    def update_tracks(self, cam_id, dets, pts, wall_time=None):
        """Update generic tracks (non-vehicle classes only — vehicles come from plate detector)."""
        now = wall_time if wall_time is not None else time.time()
        # Forward to Redis collector but filter OUT vehicle classes
        # (plate detector handles vehicles)
        filtered_dets = []
        for det in dets:
            if len(det) < 7:
                continue
            label = det[0]
            if label not in VEHICLE_CLASSES:
                filtered_dets.append(det)
        self.redis_collector.update_tracks(cam_id, filtered_dets, pts, wall_time)

    def detect_plates(self, cam_id: str, frame: np.ndarray, wall_time: float):
        """Run plate detection + ByteTrack on frame, track vehicles for this camera."""
        if frame is None:
            return

        fh, fw = frame.shape[:2]
        sx, sy = MUX_W / fw, MUX_H / fh

        # ByteTrack: persist=True keeps Kalman filter state between calls
        results = self.plate_detector.model.track(
            frame, imgsz=self.plate_detector.imgsz,
            conf=self.plate_detector.conf_threshold,
            iou=0.45, tracker="bytetrack.yaml",
            persist=True, verbose=False,
            device=self.plate_detector.device,
        )[0]

        active = self._plate_active[cam_id]
        seen_ids: set[int] = set()

        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()

            for bt_id, box_xyxy in zip(ids, boxes_xyxy):
                x1, y1, x2, y2 = box_xyxy
                mux_box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]

                if bt_id in active:
                    active[bt_id].last_seen_s = wall_time
                    active[bt_id].boxes.append(mux_box)
                    active[bt_id].disappeared_s = None
                else:
                    tv = TrackedVehicle(
                        cam_id=cam_id, track_id=int(bt_id), label="vehicle",
                        first_seen_s=wall_time, last_seen_s=wall_time,
                        boxes=[mux_box],
                    )
                    active[bt_id] = tv
                seen_ids.add(bt_id)

        disappeared = []
        for tid, tv in list(active.items()):
            if tid not in seen_ids:
                if tv.disappeared_s is None:
                    tv.disappeared_s = wall_time
                if wall_time - tv.last_seen_s > 8.0:
                    disappeared.append(tid)
        for tid in disappeared:
            self._plate_finished[cam_id].append(active.pop(tid))

    def get_active(self, cam_id):
        """Merge plate-tracked vehicles + generic non-vehicle detections."""
        merged = dict(self._plate_active[cam_id])  # plate vehicles (primary)
        # Add generic non-vehicle detections with prefixed IDs to avoid collision
        for obj_id, tv in self.redis_collector.get_active(cam_id).items():
            merged[f"gen_{obj_id}"] = tv
        return merged

    def get_finished(self, cam_id):
        merged = list(self._plate_finished[cam_id])
        for tv in self.redis_collector.get_finished(cam_id):
            merged.append(tv)
        return merged


# ---------------------------------------------------------------------------
def run_live(redis_url, cameras, min_travel_s, sim_threshold,
             duration_s, out_mp4, poll_interval=1.0,
             plate_detect=False, plate_model=None, plate_conf=0.25,
             plate_imgsz=1280, plate_device=None):
    extractor = AppearanceExtractor()

    if plate_detect:
        # BOTH detectors run: plate detector (primary vehicles) + DeepStream (other classes)
        print("  Mode: PLATE + GENERIC (plate detector for vehicles, DeepStream for other classes)")
        collector = MergedCollector(
            redis_url, cameras,
            plate_model=plate_model, plate_conf=plate_conf,
            plate_imgsz=plate_imgsz, plate_device=plate_device,
        )
    else:
        collector = RedisCollector(redis_url, cameras)

    # Build a global_id lookup: {(cam_id, track_id): global_id}
    gid_map = {}
    next_gid = 0
    consumed = {c: set() for c in cameras}

    # Start RTSP readers
    readers = {}
    for cam in cameras:
        uri = RTSP_URIS.get(cam, "")
        if uri:
            r = RTSPReader(uri, cam)
            r.start()
            readers[cam] = r
            time.sleep(0.3)

    # Video writer
    n_cams = len(cameras)
    total_w = PANEL_W * n_cams + PAD * (n_cams - 1)
    total_h = PANEL_H + HEADER_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, OUT_FPS, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_mp4}")

    print(f"\n{'='*70}")
    print("LIVE MULTI-CAMERA TRACKER + VISUALIZER")
    print(f"  Chain: {' -> '.join(cameras)}")
    print(f"  Min travel: {min_travel_s}s  Sim threshold: {sim_threshold}")
    print(f"  Output: {out_mp4}")
    print(f"{'='*70}\n")

    t0 = time.time()
    frame_count = 0
    embed_count = 0
    last_match = 0.0
    all_decisions = []

    while time.time() - t0 < duration_s:
        wall_time = time.time() - t0

        # Detection: always poll Redis (generic detections)
        results = collector.poll_once()
        for cam_id, messages in results.items():
            for msg in messages:
                collector.update_tracks(cam_id, msg["dets"], msg["pts"],
                                        wall_time=wall_time)

        # Also run plate detector on RTSP frames (when enabled)
        if plate_detect:
            for cam_id in cameras:
                reader = readers.get(cam_id)
                if reader is None:
                    continue
                frame = reader.get_frame()
                if frame is not None:
                    collector.detect_plates(cam_id, frame, wall_time)

        # Extract embeddings for active + recently-finished vehicles
        for cam_id in cameras:
            reader = readers.get(cam_id)
            if reader is None:
                continue
            frame = reader.get_frame()
            if frame is None:
                continue
            targets = list(collector.get_active(cam_id).values())
            for tv in collector.get_finished(cam_id):
                if not tv.embeddings and tv.boxes:
                    targets.append(tv)
            for tv in targets:
                if len(tv.embeddings) >= 5:
                    continue
                if not tv.boxes:
                    continue
                box = random.choice(tv.boxes) if len(tv.embeddings) > 0 else tv.boxes[-1]
                left, top, w, h = box
                crop = _extract_crop(frame, left, top, w, h)
                if crop is None or crop.shape[0] < 8 or crop.shape[1] < 8:
                    continue
                emb = extractor.embed_crop(crop)
                if emb is not None:
                    tv.embeddings.append(emb)
                    embed_count += 1

        # Try cross-camera matching every 5s
        now = time.time()
        if now - last_match >= 5.0:
            last_match = now
            for ci in range(1, len(cameras)):
                src, tgt = cameras[ci - 1], cameras[ci]
                src_finished = collector.get_finished(src)
                tgt_active = list(collector.get_active(tgt).values())
                tgt_finished = collector.get_finished(tgt)
                tgt_all = tgt_active + tgt_finished
                if not src_finished or not tgt_all:
                    continue

                for tt in tgt_all:
                    t_emb = tt.mean_embedding
                    if t_emb is None:
                        continue
                    best_sim, best_st = -1.0, None
                    for st in src_finished:
                        if st.track_id in consumed.get(src, set()):
                            continue
                        s_emb = st.mean_embedding
                        if s_emb is None:
                            continue
                        src_end = st.disappeared_s or st.last_seen_s
                        time_diff = tt.first_seen_s - src_end
                        if time_diff < min_travel_s:
                            continue
                        sim = float(np.dot(s_emb, t_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_st = st

                    if best_st is not None and best_sim >= sim_threshold:
                        src_end = best_st.disappeared_s or best_st.last_seen_s
                        travel = tt.first_seen_s - src_end
                        # Get or create global ID
                        gk = (src, best_st.track_id)
                        if gk not in gid_map:
                            gid_map[gk] = next_gid
                            next_gid += 1
                        gid = gid_map[gk]
                        gid_map[(tgt, tt.track_id)] = gid
                        consumed[src].add(best_st.track_id)

                        decision = {
                            "source_cam": src,
                            "source_id": best_st.track_id,
                            "target_cam": tgt,
                            "target_id": tt.track_id,
                            "sim": best_sim,
                            "time_diff": travel,
                            "global_id": gid,
                        }
                        all_decisions.append(decision)
                        print(f"  MATCH: gid={gid} "
                              f"{src}:{best_st.track_id} -> {tgt}:{tt.track_id} "
                              f"(sim={best_sim:.3f}, travel={travel:.1f}s)")

        # Render frame
        canvas = np.zeros((total_h, total_w, 3), np.uint8)
        n_matched = len(set(gid_map.values()))
        _draw_header(canvas, "MULTI-CAMERA TRACKING", wall_time,
                     cameras, n_matched, embed_count)

        for ci, cam in enumerate(cameras):
            frame = readers[cam].get_frame() if cam in readers else None
            if frame is not None:
                panel = cv2.resize(frame, (PANEL_W, PANEL_H))
            else:
                panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)

            # Draw bounding boxes
            scale_x = PANEL_W / MUX_W
            scale_y = PANEL_H / MUX_H
            active = collector.get_active(cam)
            for obj_id, tv in active.items():
                if not tv.boxes:
                    continue
                left, top, w, h = tv.boxes[-1]
                x1 = int(left * scale_x)
                y1 = int(top * scale_y)
                x2 = int((left + w) * scale_x)
                y2 = int((top + h) * scale_y)
                gk = (cam, obj_id)
                gid = gid_map.get(gk)
                if gid is not None:
                    color = _id_color(gid)
                    cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(panel, f"G{gid}", (x1, max(y1 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
                                cv2.LINE_AA)
                else:
                    cv2.rectangle(panel, (x1, y1), (x2, y2), (128, 128, 128), 1)

            x_off = ci * (PANEL_W + PAD)
            canvas[HEADER_H:HEADER_H + PANEL_H, x_off:x_off + PANEL_W] = panel

        writer.write(canvas)
        frame_count += 1

        elapsed = time.time() - t0
        if int(elapsed) % 30 == 0 and int(elapsed) > 0 and int(elapsed) != int(elapsed - poll_interval):
            counts = {c: len(collector.get_active(c)) for c in cameras}
            fins = {c: len(collector.get_finished(c)) for c in cameras}
            cam_counts = {}
            for (c, _), g in gid_map.items():
                cam_counts.setdefault(g, set()).add(c)
            n_2plus = sum(1 for cs in cam_counts.values() if len(cs) >= 2)
            n_3 = sum(1 for cs in cam_counts.values() if len(cs) >= 3)
            print(f"  [{elapsed:.0f}s] active={counts} finished={fins} "
                  f"embeds={embed_count} gids={len(cam_counts)} "
                  f"2+cam={n_2plus} 3cam={n_3}")

        time.sleep(1.0 / OUT_FPS)

    for r in readers.values():
        r.stop()
    writer.release()

    # Write outputs
    output_dir = out_mp4.parent
    summary = {
        "total": len(cam_counts),
        "complete": n_3 if 'n_3' in dir() else 0,
        "partial": n_2plus if 'n_2plus' in dir() else 0,
        "min_travel_s": min_travel_s,
        "sim_threshold": sim_threshold,
        "tracks": [],
    }
    # Rebuild tracks from gid_map
    track_info = {}
    for (cam, tid), gid in gid_map.items():
        track_info.setdefault(gid, {"cameras": [], "track_ids": {}})
        if cam not in track_info[gid]["track_ids"]:
            track_info[gid]["cameras"].append(cam)
            track_info[gid]["track_ids"][cam] = tid
    for gid, info in track_info.items():
        summary["tracks"].append({
            "global_id": gid,
            "cameras": info["cameras"],
            "track_ids": info["track_ids"],
        })

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "decisions.json").write_text(json.dumps(all_decisions, indent=2))

    print(f"\nDone: {out_mp4} ({frame_count} frames)")
    print(f"Summary: {json.dumps({k: v for k, v in summary.items() if k != 'tracks'}, indent=2)}")
    return all_decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redis", default=DEFAULT_REDIS_URL)
    p.add_argument("--cameras", nargs="+", default=CAMERA_CHAIN)
    p.add_argument("--min-travel", type=float, default=20.0)
    p.add_argument("--sim", type=float, default=0.60)
    p.add_argument("--record", type=float, default=145)
    p.add_argument("--out", default="mtmc/output/redis_mtmc")
    p.add_argument("--plate-detect", action="store_true",
                   help="Use plate detector directly on RTSP (bypasses DeepStream/Redis)")
    p.add_argument("--plate-model", default=None,                   help="Path to plate detector model (default: products/el-mohandes/models/finetuned/best.pt)")
    p.add_argument("--plate-conf", type=float, default=0.25,
                   help="Plate detection confidence threshold")
    p.add_argument("--plate-imgsz", type=int, default=1280,
                   help="Plate detector input size")
    p.add_argument("--plate-device", default=None,
                   help="Plate detector device (cuda:0, cpu, etc.)")
    args = p.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / "mtmc_live.mp4"

    run_live(
        redis_url=args.redis,
        cameras=args.cameras,
        min_travel_s=args.min_travel,
        sim_threshold=args.sim,
        duration_s=args.record,
        out_mp4=out_mp4,
        plate_detect=args.plate_detect,
        plate_model=args.plate_model,
        plate_conf=args.plate_conf,
        plate_imgsz=args.plate_imgsz,
        plate_device=args.plate_device,
    )


if __name__ == "__main__":
    main()
