#!/usr/bin/env python3
"""Register a one-person face image into the IRIS face gallery in Redis.

Usage:
    python scripts/register_face.py <name> <image_path> [--redis-url <url>]

The image should contain a clear, front-facing view of the person's face.
The embedding (512-d ArcFace vector) is computed by InsightFace and stored
in Redis under the iris:face:gallery hash, keyed by a UUID.

This script is designed to run on the host (or inside the iris container
if insightface is installed).  The gallery is read by the sidecar's
FaceRecognizer which runs in the GPU-enabled deepstream-sidecar container.

Requires:
    pip install redis opencv-python insightface onnxruntime
"""
import argparse
import json
import os
import sys
import time
import uuid

import cv2
import numpy as np
import redis

# Path to the local insightface clone (falls back to pip-installed if present)
_INSIGHTFACE_PKG = os.path.join(
    os.path.dirname(__file__), "..", "..", "helper-repos", "insightface", "python-package"
)
_INSIGHTFACE_PKG = os.path.abspath(_INSIGHTFACE_PKG)
if os.path.isdir(_INSIGHTFACE_PKG):
    sys.path.insert(0, _INSIGHTFACE_PKG)

from insightface.app import FaceAnalysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # perception.insightface_pack
from perception import insightface_pack  # noqa: E402

REDIS_DEFAULT = "redis://localhost:6379/0"
FACE_GALLERY_KEY = "iris:face:gallery"


def _draw_debug(image: np.ndarray, embedding: np.ndarray, name: str) -> np.ndarray:
    """Overlay the name and embedding-norm on the image for visual confirmation."""
    h, w = image.shape[:2]
    norm = np.linalg.norm(embedding)
    label = f"{name}  |embed|={norm:.2f}"
    cv2.putText(image, label, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def register(name: str, image_path: str, redis_url: str = REDIS_DEFAULT,
             det_thresh: float = 0.5, det_size: tuple = (640, 640),
             save_debug: str | None = None):
    """Compute embedding for *image_path* and store it in the Redis gallery."""

    # ── connect to Redis ──────────────────────────────────────────────────
    r = redis.Redis.from_url(redis_url)
    try:
        r.ping()
    except redis.ConnectionError as exc:
        print(f"ERROR: cannot connect to Redis at {redis_url}: {exc}")
        sys.exit(1)

    # ── load image ────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: cannot read image: {image_path}")
        sys.exit(1)
    print(f"Loaded {image_path}  ({img.shape[1]}x{img.shape[0]})")

    # ── InsightFace ───────────────────────────────────────────────────────
    print("Initializing InsightFace (auraface, GPU)...")
    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    faces = app.get(img, max_num=1)
    if not faces:
        print("ERROR: no face detected in the image")
        # Try CPU fallback
        print("Retrying with CPU (ctx_id=-1)...")
        app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=det_size)
        faces = app.get(img, max_num=1)
        if not faces:
            print("ERROR: still no face detected — try a brighter/more frontal image")
            sys.exit(1)

    embedding = faces[0].embedding  # (512,)
    print(f"Face detected — embedding dim={embedding.shape[0]}")

    # ── store in Redis ────────────────────────────────────────────────────
    face_id = str(uuid.uuid4())[:8]
    entry = {
        "name": name,
        "embedding": embedding.tolist(),
        "photo_path": os.path.abspath(image_path),
        "created_at": time.time(),
    }
    r.hset(FACE_GALLERY_KEY, face_id, json.dumps(entry))
    print(f"Registered '{name}' → face_id={face_id} in Redis gallery")

    # ── save debug image with overlay ─────────────────────────────────────
    if save_debug:
        debug_img = _draw_debug(img.copy(), embedding, name)
        cv2.imwrite(save_debug, debug_img)
        print(f"Debug image saved to {save_debug}")

    # ── summary ───────────────────────────────────────────────────────────
    existing = r.hlen(FACE_GALLERY_KEY)
    print(f"\nGallery now has {existing} face(s)")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 8 values:  {embedding[:8].round(4).tolist()}")

    # Show all registered names
    all_entries = r.hgetall(FACE_GALLERY_KEY)
    print("\nRegistered persons:")
    for k, v in all_entries.items():
        k = k.decode() if isinstance(k, bytes) else k
        v = json.loads(v.decode() if isinstance(v, bytes) else v)
        print(f"  [{k}] {v['name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Register a face image in the IRIS face gallery.")
    parser.add_argument("name", help="Person's name (e.g. 'Mahmoud Tayee')")
    parser.add_argument("image_path", help="Path to a face photo (JPEG/PNG)")
    parser.add_argument("--redis-url", default=REDIS_DEFAULT,
                        help=f"Redis URL (default: {REDIS_DEFAULT})")
    parser.add_argument("--det-thresh", type=float, default=0.5,
                        help="Face detection threshold (default: 0.5)")
    parser.add_argument("--debug", help="Save a debug image with overlay to this path")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"ERROR: file not found: {args.image_path}")
        sys.exit(1)

    register(args.name, args.image_path, redis_url=args.redis_url,
             det_thresh=args.det_thresh, save_debug=args.debug)


if __name__ == "__main__":
    main()
