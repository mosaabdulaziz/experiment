#!/usr/bin/env python3
"""Append extra per-angle embeddings to already-registered gallery entries.

Multi-embedding enrollment (2026-09-01): each person's primary photo is
registered by perception/pipeline/register_face.py (which RESETS the entry to that one
embedding); this script then appends one embedding per photo found under
<photos-dir>/person-N/*.jpg. The recognizer flattens every embedding into its
match matrix, so a live face at any angle only needs to be close to ONE of
them -- coverage, not consensus (the temporal vote lock is where consensus
lives). Ordering matters: run AFTER register_face.py, so a re-run of the whole
registration step stays idempotent (reset to 1, append N -> always 1+N).

Runs inside the deepstream-sidecar image (needs insightface + the sidecar's
FaceGallery), invoked from the justfile's _register_people the same way
register_face.py is. Attendance-service-side code: imports the sidecar as a
library, modifies nothing in it.

Usage (in-container paths):
    python3 products/attendance/register_extra_embeddings.py \
        --photos-dir /photos/multi \
        --redis-url redis://redis:6379/0 \
        --gallery-key attendance:face:gallery [--gpu]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import redis

# The pipeline package, found from the repo root rather than by counting
# "..": this product sits two levels down (products/attendance/), and a
# hard-coded depth breaks whenever a folder moves (2026-09-22).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))                           # perception.* packages
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception" / "pipeline"))  # telemetry, which they import bare
from perception.modules.face_recognition.recognizer import FaceGallery  # noqa: E402

from insightface.app import FaceAnalysis  # noqa: E402

from perception import insightface_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos-dir", required=True,
                    help="dir holding person-N/ subdirs of extra face photos")
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--gallery-key", default="attendance:face:gallery")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if args.gpu else -1, det_thresh=0.4, det_size=(640, 640))

    r = redis.Redis.from_url(args.redis_url)
    gallery = FaceGallery(r, gallery_key=args.gallery_key)

    total = 0
    for pid in sorted(os.listdir(args.photos_dir)):
        pdir = os.path.join(args.photos_dir, pid)
        if not os.path.isdir(pdir):
            continue
        if r.hget(args.gallery_key, pid) is None:
            print(f"SKIP {pid}: not registered yet (run register_face.py first)")
            continue
        appended = 0
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                print(f"  {pid}/{fn}: unreadable, skipped")
                continue
            faces = app.get(img)
            if not faces:
                print(f"  {pid}/{fn}: no face detected, skipped")
                continue
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
            if gallery.append_embedding(pid, faces[0].normed_embedding):
                appended += 1
        total += appended
        print(f"{pid}: appended {appended} extra embedding(s)")
    print(f"done: {total} extra embeddings appended")


if __name__ == "__main__":
    main()
