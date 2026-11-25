#!/usr/bin/env python3
"""Register a personnel face photo into the Redis face gallery.

Usage:
    uv run --locked --isolated --only-group tools python perception/pipeline/register_face.py \\
        --name "Mahmoud-Tayee" \\
        --photo perception/pipeline/personnel/Mahmoud-Tayee/front.jpeg

Runs InsightFace on CPU or GPU to extract an embedding, then stores it in
Redis under the key iris:face:gallery. The sidecar's FaceRecognizer reads
this gallery on startup and on each subsequent recognition cycle.

Requires: insightface, onnxruntime, opencv-python-headless, redis, numpy: the root lock's
`tools` group (DEPENDENCIES.md).
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

# Try PyPI insightface first, fall back to local helper repo for development
try:
    from insightface.app import FaceAnalysis  # noqa: E402
except ImportError:
    _HELPER_REPO = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "helper-repos", "insightface", "python-package")
    )
    if os.path.isdir(_HELPER_REPO):
        sys.path.insert(0, _HELPER_REPO)
    from insightface.app import FaceAnalysis  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # perception.insightface_pack
from perception import insightface_pack  # noqa: E402

REDIS_URL = "redis://localhost:6379/0"
GALLERY_KEY = "iris:face:gallery"
PHOTO_DIR_HOST = "data/face_photos"        # relative to project root, shared host dir
PHOTO_DIR_CONTAINER = "/app/data/face_photos"  # how the API container sees it


def main():
    parser = argparse.ArgumentParser(description="Register a face into the IRIS gallery")
    parser.add_argument("--name", required=True, help="Person's display name (e.g. Mahmoud-Tayee)")
    parser.add_argument("--photo", required=True, help="Path to the face photo")
    parser.add_argument("--redis-url", default=REDIS_URL, help="Redis URL")
    parser.add_argument("--face-id", help="Optional gallery key (auto-generated if omitted)")
    parser.add_argument("--gallery-key", default=GALLERY_KEY,
                         help="Redis hash key to register into (default: %(default)s). "
                              "Override to target a forked gallery, e.g. attendance's own.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (ctx_id=0) instead of CPU")
    args = parser.parse_args()

    # ── Read photo ──────────────────────────────────────────────────────────
    img = cv2.imread(args.photo)
    if img is None:
        print(f"ERROR: could not read image: {args.photo}")
        sys.exit(1)
    print(f"Photo loaded: {args.photo} ({img.shape[1]}x{img.shape[0]})")

    # ── Initialize InsightFace ──────────────────────────────────────────────
    ctx_id = 0 if args.gpu else -1
    print(f"Initializing InsightFace (ctx_id={ctx_id}) ...")
    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=(640, 640))
    print("InsightFace ready.")

    # ── Detect + embed ──────────────────────────────────────────────────────
    faces = app.get(img, max_num=1)
    if not faces:
        print("ERROR: no face detected in the photo.")
        sys.exit(1)

    embedding = faces[0].embedding  # (512,) float32
    print(f"Face detected. Embedding shape: {embedding.shape}")

    # ── Generate face_id ────────────────────────────────────────────────────
    face_id = args.face_id or str(uuid.uuid4())[:8]

    # ── Save photo to the shared directory (accessible by the API container) ─
    os.makedirs(PHOTO_DIR_HOST, exist_ok=True)
    dst_path = os.path.join(PHOTO_DIR_HOST, f"{face_id}.jpg")
    cv2.imwrite(dst_path, img)
    container_photo_path = os.path.join(PHOTO_DIR_CONTAINER, f"{face_id}.jpg")
    print(f"Photo saved: {dst_path} (container path: {container_photo_path})")

    # ── Store in Redis ──────────────────────────────────────────────────────
    r = redis.Redis.from_url(args.redis_url)

    entry = {
        "name": args.name,
        "embedding": embedding.tolist(),
        "photo_path": container_photo_path,  # API-accessible path
        "created_at": time.time(),
    }
    r.hset(args.gallery_key, face_id, json.dumps(entry))
    print(f"Registered face: {face_id} → name={args.name} (gallery: {args.gallery_key})")

    # Verify
    stored = json.loads(r.hget(args.gallery_key, face_id))
    print(f"Verified: name={stored['name']}, emb_len={len(stored['embedding'])}, photo={stored['photo_path']}")


if __name__ == "__main__":
    main()
