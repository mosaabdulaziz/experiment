#!/usr/bin/env python3
"""Phase 1 benchmark harness for commercially licensed face detectors & AuraFace.

Evaluates ranked commercial detector candidates (Qualcomm face_det_lite,
MediaPipe Face Detection, and modular Box + MediaPipe Landmark adapters)
against the current baselines (yolov8n-face SGIE and buffalo_l det_10g) across:

  Suite 1: Enrolment photo recall, single-face clarity, and landmark alignment
  Suite 2: Live SGIE person-crop recall by capture-pixel size bucket (<24, 24-35, 35-64, >64)
  Suite 3: Landmark plausibility, Barber yaw/sharpness pass rate, and coupled
           FAR/FRR + open-set margin matching with w600k_r50 vs AuraFace (glintr100.onnx)
  Suite 4: ONNX Runtime batch-1 / batch-16 GPU and CPU latency (+ optional trtexec)

Example (inside GPU sidecar or workstation venv):
    python3 benchmarks/bench_face_detectors.py \\
        --photos-dir data/face_photos \\
        --clips data/uploads/attendance/enter-3.mp4 data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi \\
        --fetch-dir scratch/face_bench_models \\
        --out-dir benchmarks/results/face_det_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.face_bench_adapters import (  # noqa: E402
    ArcFaceOnnxEmbedder,
    BaseFaceDetector,
    BoxPlusMediaPipeLandmarkDetector,
    DetectedFace,
    MediaPipeFaceDetector,
    QualcommFaceDetLiteDetector,
    ScrfdDet10gDetector,
    YoloV8FaceDetector,
    box_iou_xyxy,
    face_yaw_from_5pt,
    fetch_commercial_face_models,
    l2_normalize,
    landmarks_plausible,
    letterbox_params,
    nms_xyxy,
    norm_crop_112,
    unmap_coords,
)

logger = logging.getLogger("iris.bench_face_detectors")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_BUCKETS = (("<24px", 0.0, 24.0), ("24-35px", 24.0, 35.0), ("35-64px", 35.0, 64.0), (">64px", 64.0, 1e9))


def size_bucket(short_side: float) -> str:
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= short_side < hi:
            return label
    return ">64px"


# ── Pure-NumPy evaluation helpers (unit-tested without GPU/cv2) ──────────────

def match_detections(
    anchor_boxes: Sequence[Sequence[float]],
    cand_boxes: Sequence[Sequence[float]],
    iou_thresh: float = 0.4,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy 1-to-1 IoU matching between anchor (baseline/GT) and candidate boxes."""
    pairs: list[tuple[float, int, int]] = []
    for ai, ab in enumerate(anchor_boxes):
        for ci, cb in enumerate(cand_boxes):
            iou = box_iou_xyxy(ab, cb)
            if iou >= iou_thresh:
                pairs.append((iou, ai, ci))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_a: set[int] = set()
    used_c: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, ai, ci in pairs:
        if ai not in used_a and ci not in used_c:
            used_a.add(ai)
            used_c.add(ci)
            matched.append((ai, ci, float(iou)))
    unmatched_a = [i for i in range(len(anchor_boxes)) if i not in used_a]
    unmatched_c = [i for i in range(len(cand_boxes)) if i not in used_c]
    return matched, unmatched_a, unmatched_c


def landmark_nme(kpts_a: np.ndarray, kpts_b: np.ndarray) -> float:
    """5-point landmark Normalized Mean Error divided by inter-ocular distance."""
    iod = float(np.linalg.norm(kpts_a[1] - kpts_a[0]))
    if iod <= 1e-6:
        return float("nan")
    dists = np.linalg.norm(kpts_a - kpts_b, axis=1)
    return float(np.mean(dists) / iod)


def compute_far_frr_curve(
    gallery_ids: Sequence[str],
    gallery_embs: np.ndarray,
    probe_ids: Sequence[str],
    probe_embs: np.ndarray,
    thresholds: Sequence[float] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    margin: float = 0.05,
) -> dict[str, Any]:
    """Compute genuine/impostor similarity stats and FAR/FRR at each threshold.

    Mirrors FaceRecognizer's open-set matching rule:
      - best score >= threshold, AND
      - best score - best_other_identity_score >= margin.
    Probes whose true ID is not in the gallery (if any) are treated as impostors;
    for closed-set probes in gallery, an accept of a WRONG identity is a false accept
    and a reject ("unknown") of a known identity is a false reject.
    """
    if len(gallery_ids) == 0 or len(probe_ids) == 0:
        return {"n_gallery": len(gallery_ids), "n_probes": len(probe_ids), "operating_points": []}

    g_norm = l2_normalize(gallery_embs)
    p_norm = l2_normalize(probe_embs)
    sim_matrix = p_norm @ g_norm.T  # (P, G)
    g_arr = np.asarray(gallery_ids)
    gallery_set = set(gallery_ids)

    genuine_scores: list[float] = []
    impostor_max_scores: list[float] = []
    per_probe: list[tuple[str, str, float, float]] = []  # (true_id, best_id, best_score, second_best_other_score)

    for pi, true_id in enumerate(probe_ids):
        sims = sim_matrix[pi]
        best_idx = int(np.argmax(sims))
        best_id = str(g_arr[best_idx])
        best_score = float(sims[best_idx])

        other_mask = g_arr != best_id
        second_score = float(np.max(sims[other_mask])) if np.any(other_mask) else -1.0
        per_probe.append((true_id, best_id, best_score, second_score))

        if true_id in gallery_set:
            same_mask = g_arr == true_id
            genuine_scores.append(float(np.max(sims[same_mask])))
            non_self = g_arr != true_id
            if np.any(non_self):
                impostor_max_scores.append(float(np.max(sims[non_self])))
        else:
            impostor_max_scores.append(best_score)

    points = []
    n_known_probes = sum(1 for tid in probe_ids if tid in gallery_set)
    for thr in thresholds:
        false_accepts = 0
        false_rejects = 0
        true_accepts = 0
        for true_id, best_id, best_score, second_score in per_probe:
            accepted = best_score >= thr and (best_score - second_score >= margin if margin > 0 else True)
            if true_id in gallery_set:
                if not accepted:
                    false_rejects += 1
                elif best_id != true_id:
                    false_accepts += 1
                    false_rejects += 1
                else:
                    true_accepts += 1
            elif accepted:
                false_accepts += 1
        points.append(
            {
                "threshold": round(float(thr), 3),
                "margin": margin,
                "far": round(false_accepts / max(1, len(probe_ids)), 4),
                "frr": round(false_rejects / max(1, n_known_probes), 4) if n_known_probes else None,
                "tar": round(true_accepts / max(1, n_known_probes), 4) if n_known_probes else None,
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
            }
        )

    g_np = np.asarray(genuine_scores, dtype=np.float32) if genuine_scores else np.empty(0, dtype=np.float32)
    i_np = np.asarray(impostor_max_scores, dtype=np.float32) if impostor_max_scores else np.empty(0, dtype=np.float32)
    return {
        "n_gallery": len(gallery_ids),
        "n_probes": len(probe_ids),
        "genuine_mean": round(float(np.mean(g_np)), 4) if g_np.size else None,
        "genuine_p10": round(float(np.percentile(g_np, 10)), 4) if g_np.size else None,
        "genuine_min": round(float(np.min(g_np)), 4) if g_np.size else None,
        "impostor_max_mean": round(float(np.mean(i_np)), 4) if i_np.size else None,
        "impostor_max": round(float(np.max(i_np)), 4) if i_np.size else None,
        "operating_points": points,
    }


# ── Dataset Discovery & PGIE Person Extraction ───────────────────────────────

def discover_enrolment_photos(photos_dir: Path) -> list[tuple[str, Path, bool]]:
    """Discover (person_id, image_path, is_primary) matching reregister_all.py layout."""
    items: list[tuple[str, Path, bool]] = []
    if not photos_dir.is_dir():
        return items

    for p in sorted(photos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            items.append((p.stem, p, True))

    multi_dir = photos_dir / "multi"
    if multi_dir.is_dir():
        for person_dir in sorted(multi_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            for p in sorted(person_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    items.append((person_dir.name, p, False))

    # Also support plain <photos_dir>/<person_id>/*.jpg if neither primary nor multi/ exists.
    if not items:
        for person_dir in sorted(photos_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name == "multi":
                continue
            for idx, p in enumerate(sorted(person_dir.iterdir())):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    items.append((person_dir.name, p, idx == 0))
    return items


def discover_probe_photos(probe_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not probe_dir.is_dir():
        return items
    for person_dir in sorted(probe_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for p in sorted(person_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                items.append((person_dir.name, p))
    return items


class PgiePersonExtractor:
    """Optional DeepStream-Yolo PGIE ONNX runner to extract person boxes from frames."""

    def __init__(self, onnx_path: Path, gpu: bool = True, conf_thresh: float = 0.35, input_wh=(640, 640)):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.input_wh = input_wh

    def extract_persons(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        import cv2

        h, w = img_bgr.shape[:2]
        dst_w, dst_h = self.input_wh
        scale, pad_x, pad_y = letterbox_params(w, h, dst_w, dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        raw = self.sess.run(None, {self.input_name: blob})[0]
        arr = raw[0] if raw.ndim == 3 else raw
        if arr.shape[1] < 6:
            return []
        # COCO class 0 == person
        mask = (arr[:, 4] >= self.conf_thresh) & (np.round(arr[:, 5]).astype(int) == 0)
        arr = arr[mask]
        if len(arr) == 0:
            return []
        kept = nms_xyxy(arr[:, :4], arr[:, 4], iou_thresh=0.45)
        out: list[tuple[int, int, int, int]] = []
        for idx in kept:
            ubox = unmap_coords(arr[idx, :4].reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            x1, y1, x2, y2 = [int(round(v)) for v in ubox]
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                out.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        return out


# ── Benchmark Suites ─────────────────────────────────────────────────────────

def run_suite_1_enrolment(
    photos: list[tuple[str, Path, bool]],
    detectors: Sequence[BaseFaceDetector],
    baseline_det: Optional[BaseFaceDetector],
) -> dict[str, Any]:
    import cv2

    logger.info("Suite 1: Evaluating %d enrolment photos across %d detector(s)", len(photos), len(detectors))
    baseline_cache: dict[Path, Optional[DetectedFace]] = {}
    if baseline_det is not None:
        for _pid, path, _prim in photos:
            img = cv2.imread(str(path))
            if img is None:
                baseline_cache[path] = None
                continue
            faces = baseline_det.detect(img)
            faces.sort(key=lambda f: f.width * f.height, reverse=True)
            baseline_cache[path] = faces[0] if faces else None

    results: dict[str, Any] = {}
    for det in detectors:
        detected = 0
        single_face = 0
        plausible = 0
        ious: list[float] = []
        nmes: list[float] = []
        missed: list[str] = []

        for _pid, path, _prim in photos:
            img = cv2.imread(str(path))
            if img is None:
                continue
            faces = det.detect(img)
            if len(faces) == 1:
                single_face += 1
            if not faces:
                missed.append(str(path))
                continue
            detected += 1
            faces.sort(key=lambda f: f.width * f.height, reverse=True)
            top = faces[0]
            if landmarks_plausible(top.landmarks, top.bbox):
                plausible += 1
            base = baseline_cache.get(path)
            if base is not None:
                ious.append(box_iou_xyxy(base.bbox, top.bbox))
                if base.landmarks is not None and top.landmarks is not None:
                    nmes.append(landmark_nme(base.landmarks, top.landmarks))

        total = max(1, len(photos))
        results[det.name] = {
            "total_photos": len(photos),
            "detected": detected,
            "detection_rate": round(detected / total, 4),
            "single_face_rate": round(single_face / total, 4),
            "landmark_plausible_rate": round(plausible / max(1, detected), 4),
            "mean_iou_vs_baseline": round(float(np.mean(ious)), 4) if ious else None,
            "mean_landmark_nme_vs_baseline": round(float(np.mean(nmes)), 4) if nmes else None,
            "missed_sample": missed[:10],
        }
    return results


def _sample_clip_crops(
    clips: Sequence[Path],
    every_s: float,
    max_frames_per_clip: int,
    mux_wh: tuple[int, int],
    pgie: Optional[PgiePersonExtractor],
) -> list[dict[str, Any]]:
    import cv2

    crops: list[dict[str, Any]] = []
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            logger.warning("Cannot open clip: %s", clip)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * every_s)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sampled = 0
        frame_idx = 0
        while sampled < max_frames_per_clip and (total_frames <= 0 or frame_idx < total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            mux = cv2.resize(frame, mux_wh, interpolation=cv2.INTER_LINEAR)
            if pgie is not None:
                boxes = pgie.extract_persons(mux)
            else:
                boxes = [(0, 0, mux_wh[0], mux_wh[1])]
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                crop = mux[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(
                    {
                        "clip": clip.name,
                        "frame_idx": frame_idx,
                        "crop_idx": bi,
                        "crop": crop,
                    }
                )
            sampled += 1
            frame_idx += step
        cap.release()
    return crops


def _draw_contact_sheet(samples: list[dict[str, Any]], out_path: Path) -> None:
    import cv2

    if not samples:
        return
    tile_w, tile_h = 320, 320
    cols = min(4, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, item in enumerate(samples):
        r, c = divmod(idx, cols)
        img = item["crop"].copy()
        ih, iw = img.shape[:2]
        scale = min(tile_w / iw, tile_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        vis = cv2.resize(img, (nw, nh))
        for b, color in ((item["base"], (0, 220, 0)), (item["cand"], (255, 220, 0))):
            if b is None:
                continue
            x1, y1, x2, y2 = [int(round(v * scale)) for v in b.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if b.landmarks is not None:
                for px, py in b.landmarks:
                    cv2.circle(vis, (int(round(px * scale)), int(round(py * scale))), 2, color, -1)
        cv2.putText(vis, item["label"], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        canvas[r * tile_h : r * tile_h + nh, c * tile_w : c * tile_w + nw] = vis
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_suite_2_live_recall(
    clips: Sequence[Path],
    detectors: Sequence[BaseFaceDetector],
    baseline_det: Optional[BaseFaceDetector],
    every_s: float,
    max_frames: int,
    mux_wh: tuple[int, int],
    pgie: Optional[PgiePersonExtractor],
    out_dir: Path,
    contact_sheets: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cv2

    logger.info("Suite 2: Sampling crops from %d clip(s) ...", len(clips))
    crops = _sample_clip_crops(clips, every_s, max_frames, mux_wh, pgie)
    if not crops:
        return {"error": "No frames/crops sampled from clips"}, []

    anchor_det = baseline_det or detectors[0]
    anchors: list[list[DetectedFace]] = [anchor_det.detect(c["crop"]) for c in crops]

    results: dict[str, Any] = {}
    for det in detectors:
        bucket_tot: dict[str, int] = defaultdict(int)
        bucket_hit_op: dict[str, int] = defaultdict(int)
        bucket_hit_low: dict[str, int] = defaultdict(int)
        tot_anchors = 0
        hit_op = 0
        hit_low = 0
        extra_fp = 0
        plausible_cnt = 0
        total_cand = 0
        yaw_pass = 0
        sharp_pass_60 = 0
        sharp_pass_5 = 0
        nmes: list[float] = []
        sheet_samples: list[dict[str, Any]] = []

        for item, anc_faces in zip(crops, anchors):
            cand_op = det.detect(item["crop"])
            cand_low = det.detect(item["crop"], conf_thresh=0.10)

            total_cand += len(cand_op)
            for cf in cand_op:
                if landmarks_plausible(cf.landmarks, cf.bbox):
                    plausible_cnt += 1
                    if face_yaw_from_5pt(cf.landmarks) <= 0.35:
                        yaw_pass += 1
                    aligned = norm_crop_112(item["crop"], cf.landmarks)
                    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sharp >= 60.0:
                        sharp_pass_60 += 1
                    if sharp >= 5.0:
                        sharp_pass_5 += 1

            anc_boxes = [f.bbox for f in anc_faces]
            op_boxes = [f.bbox for f in cand_op]
            low_boxes = [f.bbox for f in cand_low]

            matched_op, _, unm_op = match_detections(anc_boxes, op_boxes, iou_thresh=0.35)
            matched_low, _, _ = match_detections(anc_boxes, low_boxes, iou_thresh=0.35)
            matched_op_set = {ai for ai, _, _ in matched_op}
            matched_low_set = {ai for ai, _, _ in matched_low}
            extra_fp += len(unm_op)

            for ai, ci, _iou in matched_op:
                if anc_faces[ai].landmarks is not None and cand_op[ci].landmarks is not None:
                    nmes.append(landmark_nme(anc_faces[ai].landmarks, cand_op[ci].landmarks))

            for ai, af in enumerate(anc_faces):
                bkt = size_bucket(af.short_side)
                tot_anchors += 1
                bucket_tot[bkt] += 1
                if ai in matched_op_set:
                    hit_op += 1
                    bucket_hit_op[bkt] += 1
                if ai in matched_low_set:
                    hit_low += 1
                    bucket_hit_low[bkt] += 1
                elif contact_sheets > 0 and len(sheet_samples) < contact_sheets:
                    sheet_samples.append(
                        {
                            "crop": item["crop"],
                            "base": af,
                            "cand": cand_op[0] if cand_op else None,
                            "label": f"{det.name} miss {bkt}",
                        }
                    )

        if sheet_samples:
            _draw_contact_sheet(sheet_samples, out_dir / "contact_sheets" / f"{det.name}_misses.jpg")

        by_bucket = {
            lbl: {
                "anchors": bucket_tot[lbl],
                "recall_op": round(bucket_hit_op[lbl] / max(1, bucket_tot[lbl]), 4) if bucket_tot[lbl] else None,
                "recall_0.10": round(bucket_hit_low[lbl] / max(1, bucket_tot[lbl]), 4) if bucket_tot[lbl] else None,
            }
            for lbl, _, _ in SIZE_BUCKETS
        }
        results[det.name] = {
            "crops_evaluated": len(crops),
            "anchor_faces": tot_anchors,
            "recall_operating": round(hit_op / max(1, tot_anchors), 4),
            "recall_floor_0.10": round(hit_low / max(1, tot_anchors), 4),
            "extra_detections": extra_fp,
            "by_size_bucket": by_bucket,
            "landmark_plausible_rate": round(plausible_cnt / max(1, total_cand), 4),
            "barber_yaw_pass_rate": round(yaw_pass / max(1, total_cand), 4),
            "barber_sharp60_pass_rate": round(sharp_pass_60 / max(1, total_cand), 4),
            "barber_sharp5_pass_rate": round(sharp_pass_5 / max(1, total_cand), 4),
            "mean_landmark_nme_vs_anchor": round(float(np.mean(nmes)), 4) if nmes else None,
        }
    return results, crops


def run_suite_3_recognition(
    enrol_photos: list[tuple[str, Path, bool]],
    probe_photos: list[tuple[str, Path]],
    detectors: Sequence[BaseFaceDetector],
    embedders: Sequence[ArcFaceOnnxEmbedder],
) -> dict[str, Any]:
    import cv2

    # If no separate probe_dir is given, hold out multi/ extra angles as probes against primary enrolment photos.
    if probe_photos:
        gallery_items = [(pid, p) for pid, p, _ in enrol_photos]
        probe_items = probe_photos
    else:
        gallery_items = [(pid, p) for pid, p, is_prim in enrol_photos if is_prim]
        probe_items = [(pid, p) for pid, p, is_prim in enrol_photos if not is_prim]

    if not gallery_items or not probe_items:
        return {"skipped": "Need both gallery photos and probe photos (or multi/ angles) for Suite 3."}

    matrix: dict[str, Any] = {}
    for det in detectors:
        for emb in embedders:
            g_ids: list[str] = []
            g_vecs: list[np.ndarray] = []
            for pid, p in gallery_items:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                faces = det.detect(img)
                if not faces:
                    continue
                faces.sort(key=lambda f: f.width * f.height, reverse=True)
                vec = emb.embed_face(img, faces[0])
                if vec is not None:
                    g_ids.append(pid)
                    g_vecs.append(vec)

            p_ids: list[str] = []
            p_vecs: list[np.ndarray] = []
            for pid, p in probe_items:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                faces = det.detect(img)
                if not faces:
                    continue
                faces.sort(key=lambda f: f.width * f.height, reverse=True)
                vec = emb.embed_face(img, faces[0])
                if vec is not None:
                    p_ids.append(pid)
                    p_vecs.append(vec)

            key = f"{det.name} + {emb.name}"
            if g_vecs and p_vecs:
                matrix[key] = compute_far_frr_curve(
                    g_ids, np.stack(g_vecs, axis=0), p_ids, np.stack(p_vecs, axis=0)
                )
            else:
                matrix[key] = {"error": "insufficient detected faces", "n_gallery": len(g_vecs), "n_probes": len(p_vecs)}
    return matrix


def run_suite_4_latency(detectors: Sequence[BaseFaceDetector]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for det in detectors:
        entry: dict[str, Any] = {}
        for bs, key in ((1, "ort_ms_b1"), (16, "ort_ms_b16")):
            try:
                entry[key] = round(det.benchmark_batch(batch_size=bs), 3)
            except ValueError as exc:
                entry[key] = str(exc)
            except Exception as exc:  # noqa: BLE001
                entry[f"{key}_error"] = str(exc)
        out[det.name] = entry
    return out


def format_markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Face Detector & AuraFace Phase 1 Benchmark Report", ""]
    if s1 := report.get("suite_1_enrolment"):
        lines += ["## Suite 1: Enrolment Photo Recall & Landmark Quality", ""]
        lines += ["| Detector | Photos | Detected | Single-Face | Plausible Kpts | Mean IoU | Mean NME |"]
        lines += ["|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in s1.items():
            lines.append(
                f"| `{name}` | {m['total_photos']} | {m['detection_rate']:.1%} | {m['single_face_rate']:.1%} "
                f"| {m['landmark_plausible_rate']:.1%} | {m['mean_iou_vs_baseline']} | {m['mean_landmark_nme_vs_baseline']} |"
            )
        lines.append("")

    if s2 := report.get("suite_2_live_recall"):
        if "error" not in s2:
            lines += ["## Suite 2: Live Person-Crop Recall by Face Size Bucket", ""]
            lines += ["| Detector | Recall (op) | Recall (@0.10) | <24px | 24-35px | 35-64px | >64px | Plausible Kpts | Yaw<=0.35 |"]
            lines += ["|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for name, m in s2.items():
                bk = m["by_size_bucket"]
                fmt = lambda k: f"{bk[k]['recall_op']:.1%}" if bk[k]["recall_op"] is not None else "n/a"
                lines.append(
                    f"| `{name}` | {m['recall_operating']:.1%} | {m['recall_floor_0.10']:.1%} "
                    f"| {fmt('<24px')} | {fmt('24-35px')} | {fmt('35-64px')} | {fmt('>64px')} "
                    f"| {m['landmark_plausible_rate']:.1%} | {m['barber_yaw_pass_rate']:.1%} |"
                )
            lines.append("")

    if s3 := report.get("suite_3_recognition"):
        if "skipped" not in s3:
            lines += ["## Suite 3: Coupled Detector + Embedder FAR/FRR", ""]
            lines += ["| Stack | Genuine Mean | Genuine p10 | Impostor Max | FAR@0.20 | FRR@0.20 | FAR@0.25 | FRR@0.25 |"]
            lines += ["|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, m in s3.items():
                if "operating_points" not in m:
                    continue
                pts = {p["threshold"]: p for p in m["operating_points"]}
                p20, p25 = pts.get(0.20, {}), pts.get(0.25, {})
                lines.append(
                    f"| `{name}` | {m.get('genuine_mean')} | {m.get('genuine_p10')} | {m.get('impostor_max')} "
                    f"| {p20.get('far')} | {p20.get('frr')} | {p25.get('far')} | {p25.get('frr')} |"
                )
            lines.append("")

    if s4 := report.get("suite_4_latency"):
        lines += ["## Suite 4: Inference Latency (ONNX Runtime)", ""]
        lines += ["| Detector | b=1 (ms) | b=16 (ms) |"]
        lines += ["|---|---:|---:|"]
        for name, m in s4.items():
            lines.append(f"| `{name}` | {m.get('ort_ms_b1', m.get('ort_ms_b1_error', 'err'))} | {m.get('ort_ms_b16', m.get('ort_ms_b16_error', 'err'))} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Benchmark commercial face detectors and AuraFace")
    ap.add_argument("--photos-dir", type=Path, default=Path("data/face_photos"))
    ap.add_argument("--probe-dir", type=Path, default=None)
    ap.add_argument("--clips", type=Path, nargs="*", default=[])
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--mux-wh", type=int, nargs=2, default=[1080, 720])
    ap.add_argument("--pgie-onnx", type=Path, default=Path("perception/models/DeepStream-Yolo/yolo11x.onnx"))
    ap.add_argument("--yolov8-face-onnx", type=Path, default=Path("perception/models/DeepStream-Yolo-Face/yolov8n-face.onnx"))
    ap.add_argument("--buffalo-dir", type=Path, default=Path("perception/models/insightface/models/buffalo_l"))
    ap.add_argument("--aurafaces-onnx", type=Path, default=Path("AuraFace/glintr100.onnx"))
    ap.add_argument("--face-det-lite-onnx", type=Path, default=None)
    ap.add_argument("--mediapipe-det-onnx", type=Path, default=None)
    ap.add_argument("--mediapipe-lmk-onnx", type=Path, default=None)
    ap.add_argument("--box-det-onnx", action="append", default=[], help="Extra [name=]path.onnx paired with MediaPipe landmarks")
    ap.add_argument("--fetch-dir", type=Path, default=None, help="Download missing commercial models here")
    ap.add_argument("--cpu", action="store_true", help="Force CPU ExecutionProvider")
    ap.add_argument("--contact-sheets", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/face_det_eval"))
    args = ap.parse_args()

    gpu = not args.cpu
    fetched = fetch_commercial_face_models(args.fetch_dir) if args.fetch_dir else {}
    aura_onnx = args.aurafaces_onnx if args.aurafaces_onnx.is_file() else fetched.get("aurafaces_glintr100")
    fdl_onnx = args.face_det_lite_onnx or fetched.get("face_det_lite")
    mp_det_onnx = args.mediapipe_det_onnx or fetched.get("mediapipe_detector")
    mp_lmk_onnx = args.mediapipe_lmk_onnx or fetched.get("mediapipe_landmark")

    detectors: list[BaseFaceDetector] = []
    baseline_sgie: Optional[BaseFaceDetector] = None
    baseline_enrol: Optional[BaseFaceDetector] = None

    if args.yolov8_face_onnx.is_file():
        baseline_sgie = YoloV8FaceDetector(args.yolov8_face_onnx, gpu=gpu)
        detectors.append(baseline_sgie)
    if args.buffalo_dir.is_dir() and (args.buffalo_dir / "det_10g.onnx").is_file():
        try:
            baseline_enrol = ScrfdDet10gDetector(args.buffalo_dir.parents[1], gpu=gpu)
            detectors.append(baseline_enrol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping buffalo_l det_10g baseline: %s", exc)
    if fdl_onnx and Path(fdl_onnx).is_file():
        detectors.append(QualcommFaceDetLiteDetector(fdl_onnx, gpu=gpu))
    if mp_det_onnx and Path(mp_det_onnx).is_file():
        detectors.append(MediaPipeFaceDetector(mp_det_onnx, mp_lmk_onnx, gpu=gpu))
    for spec in args.box_det_onnx:
        name, path_s = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        if Path(path_s).is_file() and mp_lmk_onnx and Path(mp_lmk_onnx).is_file():
            detectors.append(BoxPlusMediaPipeLandmarkDetector(name, path_s, mp_lmk_onnx, gpu=gpu))

    if not detectors:
        raise SystemExit("No detectors found to benchmark. Pull DVC models or pass --fetch-dir / model paths.")

    embedders: list[ArcFaceOnnxEmbedder] = []
    w600k = args.buffalo_dir / "w600k_r50.onnx"
    if w600k.is_file():
        embedders.append(ArcFaceOnnxEmbedder("w600k_r50", w600k, gpu=gpu))
    if aura_onnx and Path(aura_onnx).is_file():
        embedders.append(ArcFaceOnnxEmbedder("aurafaces_glintr100", aura_onnx, gpu=gpu))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    enrol_photos = discover_enrolment_photos(args.photos_dir)
    if enrol_photos:
        report["suite_1_enrolment"] = run_suite_1_enrolment(enrol_photos, detectors, baseline_enrol)

    existing_clips = [c for c in args.clips if c.is_file()]
    if existing_clips:
        pgie = PgiePersonExtractor(args.pgie_onnx, gpu=gpu) if args.pgie_onnx.is_file() else None
        s2, _ = run_suite_2_live_recall(
            existing_clips,
            detectors,
            baseline_sgie,
            every_s=args.every_s,
            max_frames=args.max_frames,
            mux_wh=(args.mux_wh[0], args.mux_wh[1]),
            pgie=pgie,
            out_dir=args.out_dir,
            contact_sheets=args.contact_sheets,
        )
        report["suite_2_live_recall"] = s2

    probe_photos = discover_probe_photos(args.probe_dir) if args.probe_dir else []
    if enrol_photos and embedders:
        report["suite_3_recognition"] = run_suite_3_recognition(enrol_photos, probe_photos, detectors, embedders)

    report["suite_4_latency"] = run_suite_4_latency(detectors)

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = format_markdown_report(report)
    (args.out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
