#!/usr/bin/env python3
"""Validate + register one admin-uploaded employee photo (2026-09-02,
DB-driven enrollment -- /grilling round 1 Q7/Q8 + /mentor design). Runs
inside the deepstream-sidecar image (needs InsightFace/GPU) --
products/attendance's own process has neither, same reason
register_extra_embeddings.py already shells out here. Invoked as a
subprocess from server.py's photo-upload endpoint; the only interface
between the two processes is this script's stdout (one JSON line) and exit
code.

Validates two things a raw register_face.py call does not:
  1. exactly one face in the photo (0 -> reject; >1 -> reject, ambiguous
     which face is the employee's)
  2. the embedding is pairwise-distinct from every OTHER employee already in
     the gallery (Q7: "make sure this image is distinct") -- same
     cosine-similarity math AND threshold face_recognizer.py itself uses to
     decide a live match (see its rival-mask logic), so "distinct enough to
     enroll" and "distinct enough to tell apart live" are the same bar by
     construction, never two numbers that could quietly drift apart.

A second (or third...) photo for an employee who already has a gallery
entry is appended as an extra embedding rather than rejected as a
duplicate -- that's the whole point of multi-angle enrollment; only a
collision with a DIFFERENT employee counts as "not distinct".

Usage (in-container paths, mirrors register_extra_embeddings.py's
invocation convention):
    python3 products/attendance/register_employee_photo.py \\
        --photo /photos/upload.jpg --face-id person-14 --name "New Employee" \\
        --redis-url redis://redis:6379/0 --gallery-key attendance:face:gallery \\
        --threshold 0.2 [--gpu]

Prints exactly one JSON line to stdout, e.g.
    {"ok": true, "face_id": "person-14", "embeddings": 1, "mode": "primary"}
    {"ok": false, "reason": "not_distinct", "conflicts_with_name": "...", "similarity": 0.34}
Exit code 0 iff "ok" is true.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import redis

# The pipeline package, found from the repo root rather than by counting
# "..": this product sits two levels down (products/attendance/), and a
# hard-coded depth breaks whenever a folder moves (2026-09-22).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))                           # perception.* packages
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception" / "pipeline"))  # telemetry, which they import bare
from perception.modules.face_recognition.recognizer import FaceGallery  # noqa: E402

from insightface.app import FaceAnalysis  # noqa: E402

from perception import insightface_pack  # noqa: E402


def _emit(payload: dict) -> None:
    print(json.dumps(payload))
    sys.exit(0 if payload.get("ok") else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--face-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--gallery-key", default="attendance:face:gallery")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="cosine similarity at/above which two photos count as the same "
                         "person -- default matches sidecar.yaml's face_recognition_threshold")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.photo)
    if img is None:
        _emit({"ok": False, "reason": "unreadable_photo"})

    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if args.gpu else -1, det_thresh=0.5, det_size=(640, 640))

    faces = app.get(img)
    if len(faces) == 0:
        _emit({"ok": False, "reason": "no_face"})
    if len(faces) > 1:
        _emit({"ok": False, "reason": "multiple_faces", "count": len(faces)})

    embedding = np.asarray(faces[0].normed_embedding, dtype=np.float32)

    r = redis.Redis.from_url(args.redis_url)
    gallery = FaceGallery(r, gallery_key=args.gallery_key)
    gallery.load_all()
    face_ids, names, gallery_embs = gallery.get_embeddings()

    # Distinctness (Q7): only a DIFFERENT employee's embedding counts as a
    # collision -- a second angle of this SAME employee's own face_id must
    # never be rejected against itself. Same rival-mask exclusion
    # face_recognizer.py's live matching already applies.
    other_mask = np.array([fid != args.face_id for fid in face_ids]) if face_ids else np.array([], dtype=bool)
    if other_mask.any():
        other_embs = gallery_embs[other_mask]
        other_norm = other_embs / (np.linalg.norm(other_embs, axis=1, keepdims=True) + 1e-6)
        similarities = other_norm @ embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score >= args.threshold:
            other_names = np.array(names)[other_mask]
            other_ids = np.array(face_ids)[other_mask]
            _emit({"ok": False, "reason": "not_distinct",
                  "conflicts_with_face_id": str(other_ids[best_idx]),
                  "conflicts_with_name": str(other_names[best_idx]),
                  "similarity": round(best_score, 4)})

    already_enrolled = r.hget(args.gallery_key, args.face_id) is not None
    if already_enrolled:
        ok = gallery.append_embedding(args.face_id, embedding)
        mode = "extra"
    else:
        ok = gallery.add(args.face_id, args.name, embedding, photo_path=args.photo)
        mode = "primary"
    if not ok:
        _emit({"ok": False, "reason": "gallery_write_failed"})

    stored = json.loads(r.hget(args.gallery_key, args.face_id))
    _emit({"ok": True, "face_id": args.face_id, "embeddings": len(stored["embeddings"]), "mode": mode})


if __name__ == "__main__":
    main()
