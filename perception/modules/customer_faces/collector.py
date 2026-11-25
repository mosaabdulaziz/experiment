"""Customer faces for the barber register (project C). For each face the
detector reports, it applies the quality gates, embeds the aligned face with
ArcFace, saves the padded face crop to disk, and publishes it on
iris:customer_face:<cam>. It runs on a worker thread of the customer_faces
module, never the streaming thread. Staff are not filtered here: barber-service
drops the faces of tracks that don't belong to a customer's visit.

Gates: short side >= min_pixels (capture px), yaw from the 5 landmarks <=
max_yaw, Laplacian sharpness of the aligned 112x112 face >= min_sharpness.
Crop files are world-writable because the sidecar runs as root and
barber-service (a host user) deletes the ones it doesn't keep."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime

import cv2
import numpy as np

from perception.lib.imaging import encode_embedding, l2n, save_jpeg  # noqa: F401 (save_jpeg re-exported)

logger = logging.getLogger("sidecar.customer_faces")
CUSTOMER_FACE_STREAM_PREFIX = "iris:customer_face:"


def face_yaw(landmarks) -> float:
    """|nose x - eye-midpoint x| / eye distance, from the 5 landmarks
    (left eye, right eye, nose, mouth left, mouth right). 0 = frontal."""
    p = np.asarray(landmarks, np.float32)
    eye_mid = (p[0] + p[1]) / 2.0
    dist = float(np.linalg.norm(p[1] - p[0]))
    if dist <= 1e-6:
        return float("inf")
    return abs(float(p[2][0] - eye_mid[0])) / dist


def sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(short_side: float, sharp: float, yaw: float, min_sharpness: float, max_yaw: float) -> float:
    size = min(1.0, short_side / 112.0)
    sharp_n = min(1.0, sharp / (2.0 * min_sharpness))
    frontal = max(0.0, 1.0 - yaw / max_yaw)
    return round(size * sharp_n * frontal, 4)


class CustomerFaceCollector:
    def __init__(self, redis_client, crops_dir: str, align, embed, min_pixels: int = 35,
                 min_sharpness: float = 60.0, max_yaw: float = 0.35, interval_s: float = 1.0,
                 maxlen: int = 1024):
        self.r, self.crops_dir, self.align, self.embed = redis_client, crops_dir, align, embed
        self.min_pixels, self.min_sharpness, self.max_yaw = min_pixels, min_sharpness, max_yaw
        self.interval_s, self.maxlen = interval_s, maxlen
        self.rejected = {"size": 0, "landmarks": 0, "yaw": 0, "sharpness": 0, "embed": 0, "write": 0}
        self._last: dict[tuple[str, int], float] = {}

    def offer(self, source_id: str, frame: np.ndarray, face: dict, now: float) -> dict | None:
        # 2026-10-05 (face register): what the gates drop, once a minute -- a door camera that publishes
        # nothing is otherwise silent
        self.offered = getattr(self, "offered", 0) + 1
        if now - getattr(self, "_logged_at", 0.0) >= 60.0:
            self._logged_at = now
            logger.info("customer faces: %d offered, %d published, rejected %s", self.offered,
                        getattr(self, "published", 0), self.rejected)
        tid = face.get("parent_id")
        if tid is None:
            return None
        left, top = float(face["left"]), float(face["top"])
        w, h = float(face["width"]), float(face["height"])
        if min(w, h) < self.min_pixels:
            self.rejected["size"] += 1
            return None
        key = (source_id, int(tid))
        if now - self._last.get(key, -1e9) < self.interval_s:
            return None
        lmk = face.get("landmarks")
        if not lmk or len(lmk) != 5:
            self.rejected["landmarks"] += 1
            return None
        yaw = face_yaw(lmk)
        if yaw > self.max_yaw:
            self.rejected["yaw"] += 1
            return None
        aligned = self.align(frame, np.asarray(lmk, np.float32))
        sharp = sharpness(aligned)
        if sharp < self.min_sharpness:
            self.rejected["sharpness"] += 1
            return None
        emb = self.embed(aligned)
        if emb is None:
            self.rejected["embed"] += 1
            return None
        fh, fw = frame.shape[:2]
        px, py = 0.3 * w, 0.3 * h
        x1, y1 = max(0, int(left - px)), max(0, int(top - py))
        x2, y2 = min(fw, int(left + w + px)), min(fh, int(top + h + py))
        try:
            rel = save_jpeg(self.crops_dir, "faces", frame[y1:y2, x1:x2], now)
        except OSError as exc:
            self.rejected["write"] += 1
            if self.rejected["write"] % 100 == 1:
                logger.warning("customer face crop not written (%d so far): %s", self.rejected["write"], exc)
            return None
        self._last[key] = now
        if len(self._last) > 4096:
            self._last = {k: t for k, t in self._last.items() if now - t < 60.0}
        item = {"track_id": int(tid), "crop_path": rel,
                "embedding": encode_embedding(l2n(np.asarray(emb, np.float32)[None])[0]),
                "quality": face_quality(min(w, h), sharp, yaw, self.min_sharpness, self.max_yaw),
                "face_box": [round(left, 1), round(top, 1), round(w, 1), round(h, 1)]}
        self.r.xadd(f"{CUSTOMER_FACE_STREAM_PREFIX}{source_id}", {"item": json.dumps(item)},
                    maxlen=self.maxlen, approximate=True)
        self.published = getattr(self, "published", 0) + 1
        return item
