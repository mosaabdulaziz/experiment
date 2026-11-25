"""Crops, saved pictures and embeddings, shared by the modules that cut people and faces out of a frame."""
from __future__ import annotations

import base64
import os
import uuid
from datetime import datetime

import cv2
import numpy as np

MIN_CROP_PX = 8


def crop(frame_bgr: np.ndarray, box) -> np.ndarray | None:
    """The box's pixels, clipped to the frame; None when too small to embed."""
    h, w = frame_bgr.shape[:2]
    left, top, bw, bh = box
    x1, y1 = max(0, int(left)), max(0, int(top))
    x2, y2 = min(w, int(left + bw)), min(h, int(top + bh))
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return frame_bgr[y1:y2, x1:x2]


def save_jpeg(root: str, kind: str, img: np.ndarray, now: float) -> str:
    """Write img as root/kind/<day>/<uuid>.jpg; return the path relative to root. The folders and the file are
    world-writable: the sidecar runs as root and a service running as a host user deletes the pictures it does not
    keep."""
    day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    rel = f"{kind}/{day}/{uuid.uuid4().hex}.jpg"
    path = os.path.join(root, rel)
    for d in (os.path.join(root, kind), os.path.dirname(path)):
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o777)
    if not cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"could not write {path}")
    os.chmod(path, 0o666)
    return rel


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def encode_embedding(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, "<f2").tobytes()).decode("ascii")


def decode_embedding(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), "<f2").astype(np.float32)
