#!/usr/bin/env python3
"""benchmarks/bench_gallery.py — register 2-3 gallery faces from the test video
frames already stored in Redis (iris:meta:cam-1), so the recognizer does real
embedding + matching work during benchmark runs.

Run inside the sidecar container before `run_bench.py before|after`:
  docker exec surveillance-camera-sys-deepstream-sidecar-1 \
      python3 /workspace/iris/benchmarks/bench_gallery.py
"""

import sys
import json

sys.path.insert(0, "/workspace/iris")

import cv2
import numpy as np
import redis

from perception.modules.face_recognition.recognizer import FaceRecognizer, FACE_GALLERY_KEY  # noqa: E402

META_KEY = "iris:meta:cam-1"
N_ENTRIES = 3


def clean_bench_entries():
    """Remove the bench-* gallery entries this script creates (run with --clean)."""
    r = redis.Redis.from_url("redis://redis:6379/0")
    removed = 0
    for k in r.hkeys(FACE_GALLERY_KEY):
        k = k.decode() if isinstance(k, bytes) else k
        if k.startswith("bench-"):
            r.hdel(FACE_GALLERY_KEY, k)
            removed += 1
    print(f"[gallery] removed {removed} bench-* entries")


def main():
    if "--clean" in sys.argv:
        clean_bench_entries()
        return 0
    r = redis.Redis.from_url("redis://redis:6379/0")
    recognizer = FaceRecognizer(redis_client=r, model_name="buffalo_l",
                                det_size=(640, 640), det_thresh=0.5,
                                rec_thresh=0.3, ctx_id=0)

    # Pull stored frames that carry a JPEG, newest first.
    entries = r.xrevrange(META_KEY, "+", "-", count=200)
    added = 0
    for _eid, fields in entries:
        if added >= N_ENTRIES:
            break
        f = {k.decode(): v for k, v in fields.items()}
        jpeg = f.get("jpeg")
        if not jpeg:
            continue
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        faces = recognizer.app.get(img, max_num=5)
        if not faces:
            continue
        # Pick the largest face in the frame.
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        px = int((x2 - x1) * 0.3)
        py = int((y2 - y1) * 0.3)
        h, w = img.shape[:2]
        crop = img[max(0, y1 - py):min(y2 + py, h), max(0, x1 - px):min(x2 + px, w)]
        emb = recognizer._embed_crop(crop)
        if emb is None:
            continue
        name = f"bench_person_{added + 1}"
        eid = _eid.decode() if isinstance(_eid, bytes) else _eid
        face_id = f"bench-{added + 1}-{eid}"
        recognizer.gallery.add(face_id, name, emb)
        print(f"[gallery] added {name} (face {added + 1})")
        added += 1
    print(f"[gallery] total added: {added} (gallery now {len(recognizer.gallery.list_all())} entries)")
    print("[gallery] re-run with --clean to remove these bench-* entries")
    return 0 if added >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
