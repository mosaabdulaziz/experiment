#!/usr/bin/env python3
"""Multi-camera vehicle tracker using DeepStream Redis detections + RTSP frames.

Subscribes to iris:meta:<cam-id> Redis streams (published by the sidecar),
crops vehicles from the RTSP streams, extracts ResNet50 appearance embeddings,
and matches vehicles across cameras using:
  - Known topology: D41 → D52 → CH23
  - Minimum travel-time: >=20s between cameras
  - Visual similarity: cosine similarity on ResNet50 embeddings

Usage:
    .venv/bin/python -m mtmc.mtmc_redis --record 145
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

from .feature_extractor import AppearanceExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_REDIS_URL = "redis://172.19.0.5:6379/0"
CAMERA_CHAIN = ["cam-1", "cam-2", "cam-3"]
RTSP_URIS = {
    "cam-1": "rtsp://localhost:8554/raw-cam-1",
    "cam-2": "rtsp://localhost:8554/raw-cam-2",
    "cam-3": "rtsp://localhost:8554/raw-cam-3",
}
MUX_W, MUX_H = 1280, 720
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


def _extract_crop(frame, left, top, width, height, mux_w=MUX_W, mux_h=MUX_H, pad=0.15):
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
        """Update tracks using wall-clock time for travel-time constraints.
        
        Args:
            wall_time: Wall-clock seconds since pipeline start. If None, uses time.time().
        """
        now = wall_time if wall_time is not None else time.time()
        active = self._active[cam_id]
        seen_ids = set()

        for det in dets:
            # New format: [label, conf, obj_id, left, top, width, height]
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
                    cam_id=cam_id,
                    track_id=obj_id,
                    label=label,
                    first_seen_s=now,
                    last_seen_s=now,
                    boxes=[[left, top, width, height]],
                )
                active[obj_id] = tv

        # Mark disappeared
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
# Matcher
# ---------------------------------------------------------------------------
class CrossCameraMatcher:
    def __init__(self, extractor, cameras, min_travel_s=20.0, sim_threshold=0.60):
        self.extractor = extractor
        self.cameras = cameras
        self.min_travel_s = min_travel_s
        self.sim_threshold = sim_threshold
        self.global_tracks = []
        self._next_gid = 0
        self._consumed = {c: set() for c in cameras}

    def _new_gid(self):
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def try_match(self, src_cam, tgt_cam, src_tracks, tgt_tracks):
        decisions = []
        matched_src = set()
        matched_tgt = set()

        n_src_emb = sum(1 for st in src_tracks if st.mean_embedding is not None)
        n_tgt_emb = sum(1 for tt in tgt_tracks if tt.mean_embedding is not None)
        print(f"  try_match {src_cam}->{tgt_cam}: {len(src_tracks)} src ({n_src_emb} with emb), "
              f"{len(tgt_tracks)} tgt ({n_tgt_emb} with emb)")

        for tt in tgt_tracks:
            t_emb = tt.mean_embedding
            if t_emb is None:
                continue

            best_sim, best_st = -1.0, None
            for st in src_tracks:
                if st.track_id in self._consumed.get(src_cam, set()):
                    continue
                s_emb = st.mean_embedding
                if s_emb is None:
                    continue

                src_end = st.disappeared_s or st.last_seen_s
                time_diff = tt.first_seen_s - src_end

                if time_diff < self.min_travel_s:
                    print(f"    SKIP {src_cam}:{st.track_id}->{tgt_cam}:{tt.track_id}: "
                          f"time_diff={time_diff:.1f}s < {self.min_travel_s}s")
                    continue

                sim = float(np.dot(s_emb, t_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_st = st

            if best_st is not None and best_sim >= self.sim_threshold:
                print(f"    MATCH {src_cam}:{best_st.track_id}->{tgt_cam}:{tt.track_id} "
                      f"(sim={best_sim:.3f}, travel={tt.first_seen_s - (best_st.disappeared_s or best_st.last_seen_s):.1f}s)")
                # Find or create global track
                existing = None
                for gt in self.global_tracks:
                    if src_cam in gt.track_ids and gt.track_ids[src_cam] == best_st.track_id:
                        existing = gt
                        break
                if existing is None:
                    existing = GlobalMatch(
                        global_id=self._new_gid(),
                        cameras=[src_cam],
                        track_ids={src_cam: best_st.track_id},
                        start_times={src_cam: best_st.first_seen_s},
                        end_times={src_cam: best_st.last_seen_s},
                    )
                    self.global_tracks.append(existing)

                existing.cameras.append(tgt_cam)
                existing.track_ids[tgt_cam] = tt.track_id
                existing.start_times[tgt_cam] = tt.first_seen_s
                existing.end_times[tgt_cam] = tt.last_seen_s

                matched_src.add(best_st.track_id)
                matched_tgt.add(tt.track_id)
                src_end = best_st.disappeared_s or best_st.last_seen_s

                decisions.append({
                    "source_cam": src_cam,
                    "source_id": best_st.track_id,
                    "target_cam": tgt_cam,
                    "target_id": tt.track_id,
                    "sim": best_sim,
                    "time_diff": tt.first_seen_s - src_end,
                    "matched": True,
                    "global_id": existing.global_id,
                })
                print(f"  MATCH: gid={existing.global_id} "
                      f"{src_cam}:{best_st.track_id} -> {tgt_cam}:{tt.track_id} "
                      f"(sim={best_sim:.3f}, travel={tt.first_seen_s - src_end:.1f}s)")

        self._consumed[src_cam].update(matched_src)
        return decisions

    def run(self, collector, readers, duration_s, poll_interval=1.0):
        print(f"\n{'='*70}")
        print("MULTI-CAMERA TRACKER (Redis + DeepStream)")
        print(f"  Chain: {' -> '.join(self.cameras)}")
        print(f"  Min travel: {self.min_travel_s}s")
        print(f"  Sim threshold: {self.sim_threshold}")
        print(f"{'='*70}\n")

        t0 = time.time()
        all_decisions = []
        last_match = 0.0
        embed_count = 0

        while time.time() - t0 < duration_s:
            # Poll Redis
            wall_time = time.time() - t0  # seconds since pipeline start
            results = collector.poll_once()
            for cam_id, messages in results.items():
                for msg in messages:
                    collector.update_tracks(cam_id, msg["dets"], msg["pts"], wall_time=wall_time)

            # Extract embeddings for active + recently-finished vehicles
            for cam_id in self.cameras:
                reader = readers.get(cam_id)
                if reader is None:
                    continue
                frame = reader.get_frame()
                if frame is None:
                    continue
                # Active tracks
                active = collector.get_active(cam_id)
                targets = list(active.values())
                # Recently-finished tracks with no embeddings yet
                for tv in collector.get_finished(cam_id):
                    if not tv.embeddings and tv.boxes:
                        targets.append(tv)
                for tv in targets:
                    if len(tv.embeddings) >= 5:
                        continue
                    if not tv.boxes:
                        continue
                    # Use a random box (not just latest) for variety
                    box = random.choice(tv.boxes) if len(tv.embeddings) > 0 else tv.boxes[-1]
                    left, top, w, h = box
                    crop = _extract_crop(frame, left, top, w, h)
                    if crop is None or crop.shape[0] < 8 or crop.shape[1] < 8:
                        continue
                    emb = self.extractor.embed_crop(crop)
                    if emb is not None:
                        tv.embeddings.append(emb)
                        embed_count += 1

            # Try cross-camera matching every 5s
            now = time.time()
            if now - last_match >= 5.0:
                last_match = now
                for ci in range(1, len(self.cameras)):
                    src, tgt = self.cameras[ci - 1], self.cameras[ci]
                    src_finished = collector.get_finished(src)
                    tgt_active = list(collector.get_active(tgt).values())
                    tgt_finished = collector.get_finished(tgt)
                    tgt_all = tgt_active + tgt_finished

                    if src_finished and tgt_all:
                        dec = self.try_match(src, tgt, src_finished, tgt_all)
                        all_decisions.extend(dec)

            # Status
            elapsed = time.time() - t0
            if int(elapsed) % 30 == 0 and int(elapsed) > 0 and int(elapsed) != int(elapsed - poll_interval):
                counts = {c: len(collector.get_active(c)) for c in self.cameras}
                fins = {c: len(collector.get_finished(c)) for c in self.cameras}
                n_matched = sum(1 for gt in self.global_tracks if gt.num_cameras >= 2)
                print(f"  [{elapsed:.0f}s] active={counts} finished={fins} "
                      f"embeds={embed_count} global={len(self.global_tracks)} "
                      f"matched={n_matched}")

            time.sleep(poll_interval)

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
    args = p.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    extractor = AppearanceExtractor()
    collector = RedisCollector(args.redis, args.cameras)
    matcher = CrossCameraMatcher(extractor, args.cameras, args.min_travel, args.sim)

    # Start RTSP readers
    readers = {}
    for cam in args.cameras:
        uri = RTSP_URIS.get(cam, "")
        if uri:
            r = RTSPReader(uri, cam)
            r.start()
            readers[cam] = r
            time.sleep(0.3)

    # Run
    decisions = matcher.run(collector, readers, args.record)

    # Stop readers
    for r in readers.values():
        r.stop()

    # Write summary
    summary = {
        "total": len(matcher.global_tracks),
        "complete": sum(1 for gt in matcher.global_tracks if gt.num_cameras >= 3),
        "partial": sum(1 for gt in matcher.global_tracks if gt.num_cameras == 2),
        "min_travel_s": matcher.min_travel_s,
        "sim_threshold": matcher.sim_threshold,
        "tracks": [{
            "global_id": gt.global_id,
            "cameras": gt.cameras,
            "track_ids": gt.track_ids,
            "start_times": {k: round(v, 1) for k, v in gt.start_times.items()},
            "end_times": {k: round(v, 1) for k, v in gt.end_times.items()},
        } for gt in matcher.global_tracks],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "decisions.json").write_text(json.dumps(decisions, indent=2))
    print(f"\nSummary: {json.dumps({k: v for k, v in summary.items() if k != 'tracks'}, indent=2)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
