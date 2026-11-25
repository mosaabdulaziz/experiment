#!/usr/bin/env python3
"""Enrol the people in a video as employees, with no photos given (2026-11-02).

The usual enrolment starts from a photo per person. Here the roster is *whoever appears in the video*: faces are
detected in every frame, followed from frame to frame (IoU tracklets), the tracklets that are the same face are
merged by their ArcFace embeddings, and each resulting person becomes an employee "Person N" with their best crops
as the enrolment photos. Nothing else is touched: `just up` then re-registers the employees into the face gallery
exactly as for photo-enrolled ones (reregister_all.py reads <photos-dir>/<gallery_rec_id>.jpg and multi/<id>/*.jpg).

Runs inside the deepstream-sidecar image (InsightFace + GPU), like the other register_* scripts. Outputs:
  * employee rows in the attendance DB (emp-N / person-N, name "<prefix> N"), biggest face first;
  * <photos-dir>/person-N.jpg (best crop) and <photos-dir>/multi/person-N/*.jpg (a few more angles/moments);
  * <report-dir>/enrolment.json and contact_sheet.jpg: who was found, how many frames, how sure the merge was.
    LOOK AT THE CONTACT SHEET: two rows for one person means --link-sim is too strict, one row holding two people
    means it is too loose.

Merging rules, so a wrong merge is hard: two tracklets that are visible in the same frame are never the same
person (cannot-link); the rest merge greedily by mean-embedding cosine similarity, highest first, above --link-sim,
and a merge only counts when it is also close on average to everything already in the group (average linkage).

Usage (see `just attendance-enroll-video`):
    python3 products/attendance/enroll_from_video.py --video /video/clip.mp4 --db /db/attendance.db \\
        --photos-dir /photos --report-dir /report [--reset] [--gpu]
"""
import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # perception.insightface_pack
from storage import connect, create_employee, next_ids  # noqa: E402

from insightface.app import FaceAnalysis  # noqa: E402

from perception import insightface_pack  # noqa: E402


@dataclass
class Sample:
    frame: int
    bbox: tuple          # x1, y1, x2, y2 in the frame
    det: float
    kps: np.ndarray
    emb: np.ndarray      # L2-normalised ArcFace embedding
    quality: float
    crop: np.ndarray = field(repr=False, default=None)


@dataclass
class Tracklet:
    samples: list = field(default_factory=list)
    last_frame: int = -1

    @property
    def frames(self) -> set:
        return {s.frame for s in self.samples}


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yaw_ratio(kps) -> float:
    """0 = looking at the camera, 1+ = profile: the nose's offset from the eyes' midpoint over the eye distance."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_d = np.linalg.norm(re - le) + 1e-6
    return float(abs(nose[0] - (le[0] + re[0]) / 2) / eye_d)


def sharpness(crop) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)       # same scale for every face
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def crop_with_margin(frame, bbox, margin):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return frame[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))].copy()


def detect_and_track(app, video, every, min_face, min_det, max_gap, track_iou):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracklets: list[Tracklet] = []
    active: list[Tracklet] = []
    idx = -1
    n_read = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % every:
            continue
        n_read += 1
        faces = app.get(frame)
        dets = []
        for f in faces:
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            if min(x2 - x1, y2 - y1) < min_face or float(f.det_score) < min_det:
                continue
            crop = crop_with_margin(frame, (x1, y1, x2, y2), 0.0)
            if crop.size == 0:
                continue
            sharp = sharpness(crop)
            size = min(x2 - x1, y2 - y1)
            # bigger, more frontal, sharper, surer = better enrolment photo
            q = size * float(f.det_score) * max(0.2, 1.0 - yaw_ratio(f.kps)) * np.log1p(sharp)
            dets.append(Sample(idx, (x1, y1, x2, y2), float(f.det_score), np.asarray(f.kps, dtype=np.float32),
                               np.asarray(f.normed_embedding, dtype=np.float32), float(q),
                               crop_with_margin(frame, (x1, y1, x2, y2), 0.6)))
        # greedy IoU match to tracklets seen within the last `max_gap` sampled frames
        active = [t for t in active if idx - t.last_frame <= max_gap * every]
        pairs = sorted(((iou(d.bbox, t.samples[-1].bbox), di, ti)
                        for di, d in enumerate(dets) for ti, t in enumerate(active)), reverse=True)
        used_d, used_t = set(), set()
        for score, di, ti in pairs:
            if score < track_iou or di in used_d or ti in used_t:
                continue
            active[ti].samples.append(dets[di])
            active[ti].last_frame = idx
            used_d.add(di)
            used_t.add(ti)
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Tracklet([d], idx)
                tracklets.append(t)
                active.append(t)
    cap.release()
    return tracklets, fps, idx + 1, n_read


def tracklet_embedding(t: Tracklet, top: int = 5) -> np.ndarray:
    best = sorted(t.samples, key=lambda s: s.quality, reverse=True)[:top]
    e = np.mean([s.emb for s in best], axis=0)
    return e / (np.linalg.norm(e) + 1e-9)


def merge_tracklets(tracklets, link_sim):
    """Greedy average-linkage with cannot-link on co-visible tracklets. Returns groups (lists of tracklet indices)
    and the pairwise similarity matrix for the report."""
    n = len(tracklets)
    embs = np.stack([tracklet_embedding(t) for t in tracklets]) if n else np.zeros((0, 512))
    sim = embs @ embs.T if n else np.zeros((0, 0))
    frames = [t.frames for t in tracklets]
    groups = {i: [i] for i in range(n)}
    pairs = sorted(((sim[i, j], i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
    for s, i, j in pairs:
        if s < link_sim:
            break
        gi = next(g for g in groups if i in groups[g])
        gj = next(g for g in groups if j in groups[g])
        if gi == gj:
            continue
        a, b = groups[gi], groups[gj]
        if any(frames[x] & frames[y] for x in a for y in b):      # seen together in one frame -> two people
            continue
        if float(np.mean([sim[x, y] for x in a for y in b])) < link_sim:
            continue
        groups[gi] = a + b
        del groups[gj]
    return list(groups.values()), sim


def pick_photos(samples, n_extra, min_gap_frames):
    """Best sample first, then the best of the rest that is not a near-duplicate moment of one already taken."""
    ranked = sorted(samples, key=lambda s: s.quality, reverse=True)
    chosen = []
    for s in ranked:
        if all(abs(s.frame - c.frame) >= min_gap_frames for c in chosen):
            chosen.append(s)
        if len(chosen) == 1 + n_extra:
            break
    return chosen


def to_photo(crop, app, min_side=256):
    """Upscale a small crop so the registration step's own face detector finds it, and prove it does."""
    h, w = crop.shape[:2]
    scale = max(1.0, min_side / min(h, w))
    img = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
    return img, app.get(img)


def contact_sheet(people, path, cell=128):
    rows = []
    for p in people:
        tiles = []
        for s in p["_samples"][:8]:
            c = cv2.resize(s.crop, (cell, cell))
            tiles.append(c)
        while len(tiles) < 8:
            tiles.append(np.zeros((cell, cell, 3), np.uint8))
        row = np.hstack(tiles)
        cv2.putText(row, p["name"], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rows.append(row)
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--photos-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--prefix", default="Person", help='employee names are "<prefix> N"')
    ap.add_argument("--every", type=int, default=1, help="look at every Nth frame")
    ap.add_argument("--det-size", type=int, nargs=2, default=[1024, 640], metavar=("W", "H"),
                    help="detector input; larger finds smaller (farther) faces")
    ap.add_argument("--det-thresh", type=float, default=0.4)
    ap.add_argument("--min-face", type=int, default=20, help="shorter side in px; smaller faces are ignored")
    ap.add_argument("--min-track-frames", type=int, default=4, help="a face seen in fewer frames is noise")
    ap.add_argument("--max-gap", type=int, default=3, help="sampled frames a tracklet may go undetected")
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--link-sim", type=float, default=0.45, help="cosine similarity above which two tracklets are "
                                                               "one person (same-video, same-day: strict)")
    ap.add_argument("--extra-photos", type=int, default=4)
    ap.add_argument("--reset", action="store_true",
                    help="start clean: empty the employee/sighting/desk/violation tables, drop the demo-roster "
                         "re-seed, and clear photos-dir (the recipe backs both up first)")
    ap.add_argument("--dry-run", action="store_true", help="detect and report only; write no DB rows or photos")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    app = FaceAnalysis(name="auraface", root=insightface_pack.cli_root("auraface"),
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if args.gpu else -1, det_thresh=args.det_thresh, det_size=tuple(args.det_size))

    tracklets, fps, n_frames, n_read = detect_and_track(
        app, args.video, args.every, args.min_face, args.det_thresh, args.max_gap, args.track_iou)
    raw = len(tracklets)
    tracklets = [t for t in tracklets if len(t.samples) >= args.min_track_frames]
    print(f"video: {n_frames} frames at {fps:.1f} fps ({n_read} looked at); {raw} face tracklets, "
          f"{len(tracklets)} lasting >= {args.min_track_frames} frames")
    if not tracklets:
        sys.exit("no usable faces found -- try a larger --det-size or a lower --min-face / --det-thresh")

    groups, sim = merge_tracklets(tracklets, args.link_sim)
    people = []
    for g in groups:
        samples = [s for i in g for s in tracklets[i].samples]
        people.append({"_tracklets": g, "_samples": sorted(samples, key=lambda s: s.quality, reverse=True),
                       "max_face_px": max(min(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) for s in samples)})
    people.sort(key=lambda p: p["max_face_px"], reverse=True)          # nearest person is "Person 1"

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    photos_dir = Path(args.photos_dir)
    min_gap = max(1, int(fps * 0.3 / args.every))

    if not args.dry_run:
        conn = connect(args.db)
        if args.reset:
            for table in ("sighting", "desk_interval", "violation", "employee"):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM id_sequence WHERE key = 'employee'")
            conn.execute("INSERT INTO id_sequence (key, n) VALUES ('employee', 0)")
            # without this marker `just up` would seed the 13 demo employees back in
            conn.execute("INSERT OR REPLACE INTO seed_state (key, seeded_at) VALUES ('demo_roster', datetime('now'))")
            conn.commit()
            if photos_dir.is_dir():
                for child in photos_dir.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
        photos_dir.mkdir(parents=True, exist_ok=True)

    for n, p in enumerate(people, start=1):
        name = f"{args.prefix} {n}"
        chosen = pick_photos(p["_samples"], args.extra_photos, min_gap)
        p["name"] = name
        p["frames_seen"] = len({s.frame for s in p["_samples"]})
        p["tracklets"] = len(p["_tracklets"])
        p["photos"] = []
        for k, s in enumerate(chosen):
            img, found = to_photo(s.crop, app)
            # the registration step embeds the LARGEST face in the file; make sure that is this one
            ok = len(found) >= 1
            sim_to_sample = float(max((f.normed_embedding @ s.emb) for f in found)) if ok else None
            p["photos"].append({"frame": s.frame, "face_px": round(min(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])),
                                "det": round(s.det, 2), "redetected": ok,
                                "sim_to_source": None if sim_to_sample is None else round(sim_to_sample, 3)})
            p["photos"][-1]["_img"] = img
        usable = [x for x in p["photos"] if x["redetected"]]
        p["enrolled"] = bool(usable)
        if not usable:
            print(f"SKIP {name}: no crop survived the photo round trip (faces too small/blurred)")
            continue
        if args.dry_run:
            continue
        emp_id, rec_id = next_ids(conn)
        create_employee(conn, emp_id, name, rec_id)
        p["employee_id"], p["gallery_rec_id"] = emp_id, rec_id
        cv2.imwrite(str(photos_dir / f"{rec_id}.jpg"), usable[0]["_img"])
        for k, x in enumerate(usable[1:], start=1):
            d = photos_dir / "multi" / rec_id
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{k:02d}.jpg"), x["_img"])
        print(f"{emp_id} {name}: {p['frames_seen']} frames, largest face {p['max_face_px']:.0f}px, "
              f"{len(usable)} photo(s)")

    contact_sheet(people, report_dir / "contact_sheet.jpg")
    pair_sims = sorted((float(sim[i, j]), i, j) for i in range(len(tracklets)) for j in range(i + 1, len(tracklets)))
    out = {
        "video": str(args.video), "frames": n_frames, "fps": fps, "link_sim": args.link_sim,
        "tracklets": len(tracklets), "persons": len(people),
        "people": [{k: v for k, v in p.items() if not k.startswith("_")} | {
            "photos": [{kk: vv for kk, vv in x.items() if kk != "_img"} for x in p["photos"]]} for p in people],
        # the closest pair that was NOT merged and the loosest pair that WAS: how far apart the decision was
        "closest_unmerged_similarity": max((s for s, i, j in pair_sims
                                            if not any(i in g and j in g for g in groups)), default=None),
    }
    (report_dir / "enrolment.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"report: {report_dir}/enrolment.json and contact_sheet.jpg")
    if args.dry_run:
        print("dry run: nothing written to the DB or the photos folder")


if __name__ == "__main__":
    main()
