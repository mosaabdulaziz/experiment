#!/usr/bin/env python3
"""Re-register every active employee into the redis face gallery
(2026-09-02, DB-driven enrollment). Runs on every `just up` -- `down` wipes
redis (see justfile) but never the sqlite db or the saved photos, so the
gallery needs rebuilding from what's already on disk every time.

Unlike register_employee_photo.py (the admin-upload path), this does NOT
run the distinctness check: everyone here already passed it once, at
upload time (or was seeded pre-vetted, see seed_demo_employees.py + the
committed face_photos). Re-validating known-good state on every restart
would only add risk (a stricter threshold tightened later could suddenly
"reject" people who were always fine) for no benefit -- this step is
replay, not validation.

Primary photo: <photos-dir>/<gallery_rec_id>.jpg -- skipped with a warning
if missing (an employee row can exist with zero photos yet, e.g. right
after admin_create_employee, before any upload). Extra angles:
<photos-dir>/multi/<gallery_rec_id>/*.jpg.

Usage (in-container paths, mirrors register_extra_embeddings.py):
    python3 products/attendance/reregister_all.py \\
        --db /db/attendance.db --photos-dir /photos \\
        --redis-url redis://redis:6379/0 --gallery-key attendance:face:gallery --gpu
"""
import argparse
import os
import sqlite3
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
    ap.add_argument("--db", required=True)
    ap.add_argument("--photos-dir", required=True)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--gallery-key", default="attendance:face:gallery")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    # det_thresh=0.4, not register_face.py's 0.5: this script's job also
    # includes the multi/ extra angles, which need the LOOSER threshold
    # register_extra_embeddings.py always used (harder angles, partial
    # occlusion) -- confirmed live 2026-09-02: running this at 0.5 first
    # silently dropped 1-2 of persons 9/10/11/12's extra embeddings
    # (5->4, 5->4, 5->3, 5->4) that had registered fine before at 0.4. One
    # shared app instance, since 0.4 is strictly more permissive -- it
    # accepts everything 0.5 did for the primaries too, not a trade-off.
    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if args.gpu else -1, det_thresh=0.4, det_size=(640, 640))

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    employees = conn.execute("SELECT id, name, gallery_rec_id FROM employee WHERE active = 1").fetchall()

    r = redis.Redis.from_url(args.redis_url)
    gallery = FaceGallery(r, gallery_key=args.gallery_key)

    def embed(path):
        img = cv2.imread(path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0].normed_embedding

    registered = 0
    for emp in employees:
        primary = os.path.join(args.photos_dir, f"{emp['gallery_rec_id']}.jpg")
        if not os.path.isfile(primary):
            print(f"SKIP {emp['id']} ({emp['name']}): no primary photo on disk yet")
            continue
        emb = embed(primary)
        if emb is None:
            print(f"SKIP {emp['id']} ({emp['name']}): primary photo unreadable/no face")
            continue
        gallery.add(emp["gallery_rec_id"], emp["name"], emb, photo_path=primary)
        extras = 0
        multi_dir = os.path.join(args.photos_dir, "multi", emp["gallery_rec_id"])
        if os.path.isdir(multi_dir):
            for fn in sorted(os.listdir(multi_dir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                extra_emb = embed(os.path.join(multi_dir, fn))
                if extra_emb is not None and gallery.append_embedding(emp["gallery_rec_id"], extra_emb):
                    extras += 1
        registered += 1
        print(f"{emp['id']} ({emp['name']}): registered, {1 + extras} embedding(s)")
    print(f"done: {registered}/{len(employees)} employees registered")


if __name__ == "__main__":
    main()
