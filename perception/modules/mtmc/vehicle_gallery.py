"""Best-shot vehicle crop gallery: one representative image per (camera, global vehicle id).

Every analysis frame, MTMCManager already extracts a tight crop of each tracked
vehicle to feed the Re-ID embedder (see mtmc/_tracklets.py::CameraTracklets.
_extract_crop). This module scores those same crops for how "representative"
they are of the vehicle -- confident detection, reasonably large in frame, not
clipped at the frame edge, roughly vehicle-shaped -- and keeps only the single
best-scoring crop per (cam_id, global_id), persisting it to MinIO (or local
disk) and indexing it in Redis so the API/frontend can show one photo per
vehicle per camera (see app/vehicle_gallery.py for the read side).

Persistence is fully async: scoring happens on the probe/analysis thread (cheap,
no I/O), but JPEG encode + the MinIO/disk write + Redis index update run on a
background worker thread so a slow upload never stalls the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from io import BytesIO

import cv2
import numpy as np

logger = logging.getLogger("iris.vehicle_gallery")

_REDIS_KEY = "iris:vehicle:crops"


def crop_score(conf: float, box: tuple[float, float, float, float],
                frame_w: int, frame_h: int, target_aspect: float = 1.4) -> float:
    """Score how "representative" a vehicle crop is likely to be, in [0, ~1].

    Rewards detector confidence (dominant term), a box that's a reasonable
    fraction of the frame (not a tiny distant speck), a roughly vehicle-like
    aspect ratio (penalizes the cab-only slivers box_stabilizer.py works
    around), and not touching the frame edge (an edge-clipped vehicle is a
    worse "photo" of it even if the detector was confident).
    """
    left, top, w, h = box
    if w <= 0 or h <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0
    area_frac = (w * h) / float(frame_w * frame_h)
    size_score = min(area_frac / 0.05, 1.0)  # saturates once vehicle fills >=5% of the frame
    aspect = w / h
    aspect_score = max(0.0, 1.0 - abs(aspect - target_aspect) / target_aspect)
    margin = 4.0
    touches_edge = (left <= margin or top <= margin
                    or left + w >= frame_w - margin or top + h >= frame_h - margin)
    edge_score = 0.6 if touches_edge else 1.0
    return float(conf) * (0.5 + 0.3 * size_score + 0.2 * aspect_score) * edge_score


class VehicleGallery:
    """Tracks the best crop seen per (cam_id, global_id) and persists it.

    Cheap to call every frame: ``maybe_submit`` only does real work (JPEG
    encode + upload) when a crop beats the current best by
    ``min_relative_improvement`` AND at least ``min_update_interval_s`` has
    passed since the last write for that vehicle -- most calls just compare a
    float and return.
    """

    def __init__(self, enabled: bool = True, storage_backend: str = "minio",
                 minio_endpoint: str = "minio:9000", minio_bucket: str = "iris-vehicle-crops",
                 minio_secure: bool = False, local_dir: str = "/workspace/iris/data/vehicle_crops",
                 redis_client=None, min_relative_improvement: float = 0.0,
                 min_update_interval_s: float = 0.0, jpeg_quality: int = 85,
                 stay_gap_s: float = 90.0, clock=None):
        self.enabled = bool(enabled)
        self._storage_backend = storage_backend
        self._bucket = minio_bucket
        self._local_dir = local_dir
        self._redis = redis_client
        self._min_rel_improve = float(min_relative_improvement)
        self._min_update_interval = float(min_update_interval_s)
        self._jpeg_quality = int(jpeg_quality)
        # A vehicle out of sight of a camera for longer than this and then seen again is a new STAY there
        # (the tracker splits visits the same way): each stay keeps its own best crop, so a vehicle that
        # enters and later comes back to the same camera has two different pictures.
        self._stay_gap_s = float(stay_gap_s)
        self._clock = clock or time.time

        self._minio = None
        if self.enabled and storage_backend == "minio":
            try:
                from minio import Minio
                self._minio = Minio(
                    minio_endpoint,
                    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
                    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
                    secure=bool(minio_secure),
                )
                if not self._minio.bucket_exists(self._bucket):
                    self._minio.make_bucket(self._bucket)
            except Exception as exc:
                logger.warning("[vehicle_gallery] MinIO init failed (falling back to local disk): %s", exc)
                self._minio = None
                self._storage_backend = "local"
        if self.enabled and self._storage_backend == "local":
            os.makedirs(self._local_dir, exist_ok=True)

        # (cam_id, global_id) -> best score persisted so far (in the current stay)
        self._best: dict[tuple[str, int], float] = {}
        self._last_update: dict[tuple[str, int], float] = {}
        # (cam_id, global_id) -> [current stay number, wall time of the last offer]
        self._stays: dict[tuple[str, int], list] = {}
        self._load_index()

        self._queue: "queue.Queue" = queue.Queue()
        self._worker: threading.Thread | None = None
        if self.enabled:
            self._worker = threading.Thread(
                target=self._loop, name="vehicle-gallery", daemon=True)
            self._worker.start()

    def _load_index(self) -> None:
        """After a restart, continue the stay numbering and best scores from the Redis index, so a
        stay's crop is never overwritten by a worse one and a returning vehicle gets the next number."""
        if self._redis is None or not self.enabled:
            return
        try:
            raw = self._redis.hgetall(_REDIS_KEY)
        except Exception as exc:
            logger.warning("[vehicle_gallery] could not read the crop index: %s", exc)
            return
        newest: dict[tuple[str, int], dict] = {}
        for v in raw.values():
            try:
                e = json.loads(v.decode() if isinstance(v, bytes) else v)
                key = (str(e["cam_id"]), int(e["global_id"]))
                if key not in newest or int(e.get("stay", 0)) > int(newest[key].get("stay", 0)):
                    newest[key] = e
            except Exception:
                continue
        for key, e in newest.items():
            self._stays[key] = [int(e.get("stay", 0)), float(e.get("updated_at", 0.0))]
            self._best[key] = float(e.get("score", 0.0))

    def close(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            try:
                self._worker.join(timeout=5.0)
            except Exception:
                pass
            self._worker = None

    def maybe_submit(self, cam_id: str, global_id: int, crop_bgr: np.ndarray | None,
                      score: float, conf: float, box: tuple, pts: float) -> None:
        """Queue *crop_bgr* for persistence if it's a meaningfully better shot."""
        if not self.enabled or crop_bgr is None or global_id is None or global_id < 0:
            return
        if crop_bgr.size == 0:
            return
        key = (cam_id, int(global_id))
        now = self._clock()
        st = self._stays.get(key)
        if st is None:
            st = self._stays[key] = [0, now]
        elif now - st[1] > self._stay_gap_s:
            st[0] += 1                      # back after a long absence: a new stay with its own best shot
            self._best.pop(key, None)
        st[1] = now
        stay = st[0]
        best = self._best.get(key)
        if best is not None:
            if score <= best * (1.0 + self._min_rel_improve):
                return
            if now - self._last_update.get(key, 0.0) < self._min_update_interval:
                return
        self._best[key] = score
        self._last_update[key] = now
        # Copy: the caller's crop is a view into the pipeline's frame buffer.
        self._queue.put((cam_id, int(global_id), crop_bgr.copy(), score, conf, box, pts, now, stay))

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._persist(*item)
            except Exception as exc:
                logger.warning("[vehicle_gallery] persist failed: %s", exc)

    def _persist(self, cam_id, global_id, crop_bgr, score, conf, box, pts, ts, stay=0) -> None:
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        data = buf.tobytes()
        # stay 0 keeps the original names; a later stay gets its own file and index field
        suffix = f"-{stay}" if stay else ""
        object_key = f"{cam_id}/{global_id}{suffix}.jpg"

        if self._storage_backend == "minio" and self._minio is not None:
            self._minio.put_object(
                self._bucket, object_key, BytesIO(data), length=len(data),
                content_type="image/jpeg")
            location = {"backend": "minio", "bucket": self._bucket, "object_key": object_key}
        else:
            cam_dir = os.path.join(self._local_dir, cam_id)
            os.makedirs(cam_dir, exist_ok=True)
            path = os.path.join(cam_dir, f"{global_id}{suffix}.jpg")
            with open(path, "wb") as f:
                f.write(data)
            location = {"backend": "local", "path": path}

        entry = {
            "cam_id": cam_id, "global_id": global_id, "score": round(float(score), 4),
            "conf": round(float(conf), 3), "box": [round(float(v), 1) for v in box],
            "pts": round(float(pts), 3), "updated_at": ts, "stay": int(stay), **location,
        }
        if self._redis is not None:
            try:
                field = f"{cam_id}:{global_id}" + (f":{stay}" if stay else "")
                self._redis.hset(_REDIS_KEY, field, json.dumps(entry))
            except Exception as exc:
                logger.warning("[vehicle_gallery] redis index update failed: %s", exc)
        logger.debug("[vehicle_gallery] saved best crop %s gid=%s score=%.3f",
                     cam_id, global_id, score)
