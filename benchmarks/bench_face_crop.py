#!/usr/bin/env python3
"""benchmarks/bench_face_crop.py — Change 4 micro-benchmark (recognition path).

Times the two face-embedding paths on identical real face crops extracted from
the test video, inside the GPU sidecar container:

  BEFORE (32aaf69):  app.get(crop, max_num=1)
                     -> InsightFace re-runs SCRFD's 640x640 detection pass on
                        every SGIE crop, then returns faces_detected[0].embedding
  AFTER  (HEAD):     FaceRecognizer._embed_crop(crop)
                     -> feeds the already-located crop straight to ArcFace's
                        get_feat (recognition-only, no bundled re-detection)

Also checks the two paths produce agreeing embeddings (cosine similarity) so the
speedup is not bought with different embeddings.

Run:  docker exec surveillance-camera-sys-deepstream-sidecar-1 \
          python3 /workspace/iris/benchmarks/bench_face_crop.py
"""

import sys
import time
import statistics

sys.path.insert(0, "/workspace/iris")

import cv2
import numpy as np
import redis

from perception.modules.face_recognition.recognizer import FaceRecognizer  # noqa: E402

VIDEO = "/workspace/iris/data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi"
N_ITER = 30        # timing iterations per crop
WARMUP = 5         # warmup iterations before timing
MAX_CROPS = 8
SCAN_FPS = 2.0     # candidate-frame scan rate while harvesting crops
SCAN_UNTIL = 90.0  # scan up to this video-second for face crops


def harvest_crops(recognizer, max_crops=MAX_CROPS):
    """Find up to max_crops face crops in the test video (one app.get per scan
    frame on the FULL frame — this harvest pass is the same in both paths)."""
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops = []
    seen = set()
    step = int(fps / SCAN_FPS)
    for idx in range(0, min(total, int(SCAN_UNTIL * fps)), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        faces = recognizer.app.get(frame, max_num=10)
        for f in faces:
            if len(crops) >= max_crops:
                break
            bbox = f.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue  # too small for either path to be meaningful
            # Mirror the live crop geometry: 30% padding around the face box.
            px = int((x2 - x1) * 0.3)
            py = int((y2 - y1) * 0.3)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1 - py):min(y2 + py, h), max(0, x1 - px):min(x2 + px, w)]
            key = (crop.shape[0], crop.shape[1], x1)
            if key in seen or crop.size == 0:
                continue
            seen.add(key)
            crops.append(crop)
        if len(crops) >= max_crops:
            break
    cap.release()
    print(f"[harvest] found {len(crops)} face crops from {VIDEO}")
    return crops


def bench(path, crops, label, n_iter=N_ITER, warmup=WARMUP):
    """Time `path(crop)` over every crop; return list of per-call ms."""
    for crop in crops:  # warm the GPU
        for _ in range(warmup):
            try:
                path(crop)
            except Exception:
                pass
    samples = []
    for crop in crops:
        for _ in range(n_iter):
            t0 = time.perf_counter()
            try:
                path(crop)
            except Exception as exc:
                print(f"  [skip] {label}: {exc}")
                continue
            samples.append((time.perf_counter() - t0) * 1000.0)
    if not samples:
        return []
    return samples


def before_path(recognizer):
    """Exact BEFORE recognition snippet (32aaf69 face_recognizer.py)."""
    def run(crop):
        faces_detected = recognizer.app.get(crop, max_num=1)
        if not faces_detected:
            raise RuntimeError("no face")
        embedding = faces_detected[0].embedding
        return embedding / (np.linalg.norm(embedding) + 1e-6)
    return run


def after_path(recognizer):
    """AFTER recognition path (HEAD face_recognizer.py)."""
    def run(crop):
        emb = recognizer._embed_crop(crop)
        if emb is None:
            raise RuntimeError("no embedding")
        return emb
    return run


def report(label, samples):
    if not samples:
        print(f"{label}: NO SAMPLES")
        return None
    med = statistics.median(samples)
    mean = statistics.mean(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
    print(f"{label}: n={len(samples)} mean={mean:.2f}ms median={med:.2f}ms p95={p95:.2f}ms "
          f"max={max(samples):.2f}ms")
    return mean


def main():
    r = redis.Redis.from_url("redis://redis:6379/0")
    print("[init] loading InsightFace (buffalo_l) on GPU ...")
    recognizer = FaceRecognizer(redis_client=r, model_name="buffalo_l",
                                det_size=(640, 640), det_thresh=0.5,
                                rec_thresh=0.3, ctx_id=0)
    crops = harvest_crops(recognizer)
    if len(crops) < 2:
        print("NOT ENOUGH CROPS — cannot compare paths; is there a face in the video?")
        return 1

    b_path = before_path(recognizer)
    a_path = after_path(recognizer)

    # Embedding agreement on every crop (before vs after on the same crop).
    sims = []
    for crop in crops:
        eb = eb_n = None
        try:
            eb = b_path(crop)
        except Exception:
            pass
        try:
            ea = a_path(crop)
        except Exception:
            ea = None
        if eb is not None and ea is not None:
            sims.append(float(np.dot(eb, ea)))
    if sims:
        print(f"[agreement] before-vs-after embedding cosine sim: "
              f"mean={statistics.mean(sims):.4f} min={min(sims):.4f}")

    print("[bench] timing paths (same crops, same GPU) ...")
    b_samples = bench(b_path, crops, "before/app.get")
    a_samples = bench(a_path, crops, "after/_embed_crop")
    b_mean = report("BEFORE app.get(crop)   ", b_samples)
    a_mean = report("AFTER  _embed_crop(crop)", a_samples)
    if b_mean and a_mean:
        print(f"\nSPEEDUP: {b_mean / a_mean:.1f}x per crop  "
              f"({b_mean:.2f}ms -> {a_mean:.2f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
