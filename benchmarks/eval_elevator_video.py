#!/usr/bin/env python3
"""Evaluate face_det_lite + AuraFace end-to-end on a recorded Attendance video.

Runs the deployed ONNX stack (`yolo11x` PGIE person extractor -> `face_det_lite`
SGIE on person crops -> `auraface/glintr100` 5-pt aligned embedding & open-set
matching) on a clip (default: the server elevator video), writes a structured
`report.json` with per-frame `MM:SS.s` timestamps, hit/miss breakdowns, and
optional annotated keyframes for manual visual inspection.

Usage (inside deepstream-sidecar container or GPU venv):
    python3 benchmarks/eval_elevator_video.py \\
        --video /data/uploads/attendance/syad/syad_6-left_2026-09-10.mkv \\
        --photos-dir data/face_photos \\
        --compare-yolo \\
        --save-dir benchmarks/results/elevator_eval/frames \\
        --out-json benchmarks/results/elevator_eval/report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO / "benchmarks"))

from bench_face_detectors import (  # noqa: E402
    PgiePersonExtractor,
    discover_enrolment_photos,
    match_detections,
)
from face_bench_adapters import (  # noqa: E402
    ArcFaceOnnxEmbedder,
    BaseFaceDetector,
    DetectedFace,
    QualcommFaceDetLiteDetector,
    ScrfdDet10gDetector,
    YoloV8FaceDetector,
    face_yaw_from_5pt,
    l2_normalize,
    landmarks_plausible,
)

logger = logging.getLogger("iris.eval_elevator")

DEFAULT_VIDEO = Path("/data/uploads/attendance/syad/syad_6-left_2026-09-10.mkv")
DEFAULT_DET_ONNX = REPO / "perception" / "models" / "DeepStream-Yolo-Face" / "face_det_lite.onnx"
DEFAULT_YOLO_ONNX = REPO / "perception" / "models" / "DeepStream-Yolo-Face" / "yolov8n-face.onnx"
DEFAULT_REC_ONNX = REPO / "perception" / "models" / "insightface" / "models" / "auraface" / "glintr100.onnx"
DEFAULT_BUFFALO_ONNX = REPO / "perception" / "models" / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
DEFAULT_PGIE_ONNX = REPO / "perception" / "models" / "DeepStream-Yolo" / "yolo11x.onnx"
DEFAULT_OUT_JSON = REPO / "benchmarks" / "results" / "elevator_eval" / "report.json"


def format_timestamp_hms(seconds: float) -> str:
    """Format elapsed seconds as MM:SS.s (or HH:MM:SS.s when >= 1 hour)."""
    seconds = max(0.0, float(seconds))
    mins, rem = divmod(seconds, 60.0)
    hours, mins = divmod(int(mins), 60)
    if hours > 0:
        return f"{hours:02d}:{int(mins):02d}:{rem:04.1f}"
    return f"{int(mins):02d}:{rem:04.1f}"


@dataclass(frozen=True)
class GalleryIndex:
    """L2-normalized gallery vectors paired with identity names."""

    names: list[str]
    matrix: np.ndarray  # shape (N, D), L2-normalized

    @property
    def is_empty(self) -> bool:
        return len(self.names) == 0


def match_face_open_set(
    vec: Optional[np.ndarray],
    gallery: GalleryIndex,
    rec_thresh: float = 0.28,
    rec_margin: float = 0.05,
) -> dict[str, Any]:
    """Match one embedding against the gallery using Attendance's threshold + margin rule."""
    if vec is None or gallery.is_empty:
        return {
            "matched_name": None,
            "best_name": None,
            "similarity": None,
            "margin_gap": None,
            "runner_up_name": None,
            "runner_up_sim": None,
        }
    sims = gallery.matrix @ l2_normalize(vec)
    best_idx = int(np.argmax(sims))
    best_name = gallery.names[best_idx]
    best_sim = float(sims[best_idx])

    runner_up_name: Optional[str] = None
    runner_up_sim: Optional[float] = None
    for i, gname in enumerate(gallery.names):
        if gname == best_name:
            continue
        val = float(sims[i])
        if runner_up_sim is None or val > runner_up_sim:
            runner_up_sim = val
            runner_up_name = gname

    gap = best_sim - runner_up_sim if runner_up_sim is not None else best_sim
    accepted = best_sim >= rec_thresh and (gap >= rec_margin if rec_margin > 0 else True)
    return {
        "matched_name": best_name if accepted else None,
        "best_name": best_name,
        "similarity": round(best_sim, 4),
        "margin_gap": round(float(gap), 4),
        "runner_up_name": runner_up_name,
        "runner_up_sim": round(float(runner_up_sim), 4) if runner_up_sim is not None else None,
    }


def load_gallery_from_photos(
    photos_dir: Path,
    det: BaseFaceDetector,
    emb: ArcFaceOnnxEmbedder,
    det_conf: float = 0.35,
) -> GalleryIndex:
    """Embed enrolment photos from `photos_dir` identically to reregister_all.py."""
    import cv2

    discovered = discover_enrolment_photos(photos_dir)
    names: list[str] = []
    vecs: list[np.ndarray] = []
    for person_id, path, _is_primary in discovered:
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = det.detect(img, conf_thresh=det_conf)
        if not faces:
            logger.warning("Gallery photo %s had no detected face >= %.2f; skipping", path, det_conf)
            continue
        best = max(faces, key=lambda f: f.width * f.height)
        vec = emb.embed_face(img, best)
        if vec is None:
            continue
        names.append(person_id)
        vecs.append(l2_normalize(vec))
    mat = np.stack(vecs, axis=0) if vecs else np.empty((0, 512), dtype=np.float32)
    return GalleryIndex(names=names, matrix=mat)


def load_gallery_from_redis(redis_url: str, gallery_key: str) -> GalleryIndex:
    """Load pre-embedded AuraFace gallery entries directly from Redis."""
    import redis

    client = redis.Redis.from_url(redis_url)
    names: list[str] = []
    vecs: list[np.ndarray] = []
    for _k, raw_v in client.hgetall(gallery_key).items():
        val = json.loads(raw_v.decode() if isinstance(raw_v, bytes) else raw_v)
        embs = val.get("embeddings")
        if embs is None and val.get("embedding") is not None:
            embs = [val["embedding"]]
        for arr in embs or []:
            vec = np.asarray(arr, dtype=np.float32)
            if vec.ndim == 1 and vec.size > 0:
                names.append(str(val.get("name", "unknown")))
                vecs.append(l2_normalize(vec))
    mat = np.stack(vecs, axis=0) if vecs else np.empty((0, 512), dtype=np.float32)
    return GalleryIndex(names=names, matrix=mat)


def _offset_face(face: DetectedFace, ox: int, oy: int) -> DetectedFace:
    x1, y1, x2, y2 = face.bbox
    kpts = face.landmarks.copy() if face.landmarks is not None else None
    if kpts is not None:
        kpts[:, 0] += ox
        kpts[:, 1] += oy
    return DetectedFace(bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy), score=face.score, landmarks=kpts)


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(v), 1) for v in box]


def _should_save_frame(frame_rec: dict[str, Any], save_mode: str) -> bool:
    if save_mode == "all":
        return True
    if save_mode == "people":
        return len(frame_rec["person_boxes"]) > 0
    if save_mode == "faces":
        return len(frame_rec["faces"]) > 0 or len(frame_rec["yolo_only_faces"]) > 0
    if save_mode == "misses":
        return len(frame_rec["yolo_only_faces"]) > 0 or (
            len(frame_rec["person_boxes"]) > 0 and len(frame_rec["faces"]) == 0
        )
    # Default "inspect": save frames with faces, yolo-only misses, or people-with-zero-faces.
    return len(frame_rec["person_boxes"]) > 0 or len(frame_rec["faces"]) > 0 or len(frame_rec["yolo_only_faces"]) > 0


def _annotate_frame(
    mux_bgr: np.ndarray,
    frame_rec: dict[str, Any],
    new_faces: Sequence[DetectedFace],
    yolo_only: Sequence[DetectedFace],
) -> np.ndarray:
    import cv2

    vis = mux_bgr.copy()
    for px1, py1, px2, py2 in frame_rec["person_boxes"]:
        cv2.rectangle(vis, (int(px1), int(py1)), (int(px2), int(py2)), (180, 140, 60), 1)

    for face, meta in zip(new_faces, frame_rec["faces"]):
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
        matched = meta["matched_name"]
        color = (0, 220, 0) if matched else (0, 220, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if face.landmarks is not None:
            for lx, ly in face.landmarks:
                cv2.circle(vis, (int(round(lx)), int(round(ly))), 2, color, -1)
        label = f"{matched} ({meta['similarity']:.2f})" if matched else f"det {meta['det_conf']:.2f}"
        if not matched and meta["similarity"] is not None:
            label += f" [{meta['best_name']}:{meta['similarity']:.2f}]"
        cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    for yface in yolo_only:
        x1, y1, x2, y2 = [int(round(v)) for v in yface.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"YOLO-ONLY {yface.score:.2f}", (x1, min(vis.shape[0] - 6, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)

    header = (
        f"{frame_rec['timestamp_hms']} f={frame_rec['frame_idx']} | "
        f"people={len(frame_rec['person_boxes'])} faces={len(frame_rec['faces'])} "
        f"yolo_only={len(frame_rec['yolo_only_faces'])}"
    )
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(vis, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def _per_identity_stats(frame_records: Sequence[dict[str, Any]], face_key: str) -> tuple[int, dict[str, Any]]:
    per_id_sims: dict[str, list[float]] = defaultdict(list)
    per_id_times: dict[str, list[str]] = defaultdict(list)
    recognized_count = 0
    for fr in frame_records:
        for fc in fr.get(face_key, []):
            name = fc.get("matched_name")
            if name is None:
                continue
            recognized_count += 1
            if fc.get("similarity") is not None:
                per_id_sims[name].append(float(fc["similarity"]))
            if not per_id_times[name] or per_id_times[name][-1] != fr["timestamp_hms"]:
                per_id_times[name].append(fr["timestamp_hms"])
    per_identity = {
        name: {
            "hits": len(per_id_sims[name]) or len(per_id_times[name]),
            "max_sim": round(max(per_id_sims[name]), 4) if per_id_sims[name] else None,
            "median_sim": round(float(np.median(per_id_sims[name])), 4) if per_id_sims[name] else None,
            "timestamps_hms": per_id_times[name],
        }
        for name in sorted(per_id_times)
    }
    return recognized_count, per_identity


def summarize_report(
    frame_records: Sequence[dict[str, Any]],
    video_info: dict[str, Any],
    config_info: dict[str, Any],
    gallery_entries: int,
    raw_faces_total: int,
    yolo_enabled: bool,
    yolo_total_faces: int,
    yolo_both_matched: int,
    buffalo_enabled: bool = False,
    old_gallery_entries: int = 0,
) -> dict[str, Any]:
    """Build top-level summary + flat miss/hit lists for manual visual review."""
    all_faces = [fc for fr in frame_records for fc in fr["faces"]]
    plausible_n = sum(1 for fc in all_faces if fc["plausible_kpts"])
    rec_count, per_identity = _per_identity_stats(frame_records, "faces")

    new_missed_vs_yolo: list[dict[str, Any]] = []
    new_only_extra_hits: list[dict[str, Any]] = []
    people_without_faces: list[dict[str, Any]] = []

    for fr in frame_records:
        for ym in fr["yolo_only_faces"]:
            new_missed_vs_yolo.append({
                "frame_idx": fr["frame_idx"],
                "timestamp_s": fr["timestamp_s"],
                "timestamp_hms": fr["timestamp_hms"],
                **ym,
                "snapshot": fr.get("snapshot"),
            })
        if yolo_enabled:
            for fc in fr["faces"]:
                if fc.get("yolo_iou") is None:
                    new_only_extra_hits.append({
                        "frame_idx": fr["frame_idx"],
                        "timestamp_s": fr["timestamp_s"],
                        "timestamp_hms": fr["timestamp_hms"],
                        "bbox": fc["bbox"],
                        "det_conf": fc["det_conf"],
                        "short_side_px": fc["short_side_px"],
                        "matched_name": fc["matched_name"],
                        "snapshot": fr.get("snapshot"),
                    })
        if fr["person_boxes"] and not fr["faces"]:
            people_without_faces.append({
                "frame_idx": fr["frame_idx"],
                "timestamp_s": fr["timestamp_s"],
                "timestamp_hms": fr["timestamp_hms"],
                "people_count": len(fr["person_boxes"]),
                "yolo_only_count": len(fr["yolo_only_faces"]),
                "snapshot": fr.get("snapshot"),
            })

    confs = [fc["det_conf"] for fc in all_faces]
    conf_hist = {
        "<0.30": sum(1 for c in confs if c < 0.30),
        "0.30-0.40": sum(1 for c in confs if 0.30 <= c < 0.40),
        "0.40-0.60": sum(1 for c in confs if 0.40 <= c < 0.60),
        ">=0.60": sum(1 for c in confs if c >= 0.60),
    }

    summary: dict[str, Any] = {
        "frames_sampled": len(frame_records),
        "frames_with_people": sum(1 for fr in frame_records if fr["person_boxes"]),
        "total_person_boxes": sum(len(fr["person_boxes"]) for fr in frame_records),
        "frames_with_faces": sum(1 for fr in frame_records if fr["faces"]),
        "frames_with_people_but_zero_faces": len(people_without_faces),
        "total_faces_raw": raw_faces_total,
        "total_faces_ge_min_pixels": len(all_faces),
        "confidence_histogram": conf_hist,
        "plausible_kpts": plausible_n,
        "plausible_kpt_pct": round(100.0 * plausible_n / len(all_faces), 1) if all_faces else 0.0,
        "gallery_entries": gallery_entries,
        "recognized_hits": rec_count,
        "unknown_faces": len(all_faces) - rec_count,
        "per_identity": per_identity,
    }
    if yolo_enabled:
        summary["yolo_comparison"] = {
            "yolo_faces_ge_min_pixels": yolo_total_faces,
            "matched_both": yolo_both_matched,
            "new_only_hits": len(new_only_extra_hits),
            "new_missed_vs_yolo": len(new_missed_vs_yolo),
            "recall_vs_yolo": round(yolo_both_matched / yolo_total_faces, 4) if yolo_total_faces else None,
        }
    if buffalo_enabled:
        old_rec_count, old_per_identity = _per_identity_stats(frame_records, "yolo_faces")
        both_rec = new_only_rec = old_only_rec = id_disagreements = 0
        for fr in frame_records:
            new_names = {fc["matched_name"] for fc in fr["faces"] if fc.get("matched_name")}
            old_names = {yf["matched_name"] for yf in fr.get("yolo_faces", []) if yf.get("matched_name")}
            both_rec += len(new_names & old_names)
            new_only_rec += len(new_names - old_names)
            old_only_rec += len(old_names - new_names)
            if new_names and old_names and new_names != old_names:
                id_disagreements += 1
        summary["buffalo_comparison"] = {
            "old_gallery_entries": old_gallery_entries,
            "old_recognized_hits": old_rec_count,
            "old_unknown_faces": yolo_total_faces - old_rec_count,
            "old_per_identity": old_per_identity,
            "frame_identity_both": both_rec,
            "frame_identity_new_only": new_only_rec,
            "frame_identity_old_only": old_only_rec,
            "frames_with_identity_diff": id_disagreements,
        }

    return {
        "config": config_info,
        "video": video_info,
        "summary": summary,
        "misses_and_hits_to_check": {
            "new_missed_vs_yolo": new_missed_vs_yolo,
            "new_only_extra_hits": new_only_extra_hits,
            "frames_with_people_but_zero_faces": people_without_faces,
        },
        "frames": list(frame_records),
    }


def _resolve_buffalo_onnx(requested: Path) -> Path:
    candidates = [
        requested,
        Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"buffalo_l embedder weights not found at {requested} or ~/.insightface/models/buffalo_l/w600k_r50.onnx"
    )


def _load_old_gallery(
    args: argparse.Namespace,
    yolo_det: YoloV8FaceDetector,
    buffalo_emb: ArcFaceOnnxEmbedder,
    gpu: bool,
) -> GalleryIndex:
    if not (args.photos_dir and args.photos_dir.is_dir()):
        return GalleryIndex(names=[], matrix=np.empty((0, 512), dtype=np.float32))
    det_for_old: BaseFaceDetector = yolo_det
    for pack_root in (REPO / "perception" / "models" / "insightface", Path.home() / ".insightface"):
        if (pack_root / "models" / "buffalo_l" / "det_10g.onnx").is_file():
            try:
                det_for_old = ScrfdDet10gDetector(pack_root=pack_root, gpu=gpu, default_conf=0.4)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falling back to yolov8n-face for old gallery enrolment (%s)", exc)
    return load_gallery_from_photos(args.photos_dir, det_for_old, buffalo_emb, det_conf=0.35)


def evaluate_video(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    gpu = not args.cpu
    compare_yolo = bool(args.compare_yolo or args.compare_buffalo)
    pgie = PgiePersonExtractor(args.pgie_onnx, gpu=gpu, conf_thresh=args.pgie_conf)
    det = QualcommFaceDetLiteDetector(args.det_onnx, gpu=gpu, default_conf=args.det_conf)
    emb = ArcFaceOnnxEmbedder("auraface", args.rec_onnx, gpu=gpu)
    yolo_det = (
        YoloV8FaceDetector(args.yolo_onnx, gpu=gpu, default_conf=args.yolo_conf)
        if compare_yolo
        else None
    )
    buffalo_emb = (
        ArcFaceOnnxEmbedder("buffalo_l", _resolve_buffalo_onnx(args.buffalo_onnx), gpu=gpu)
        if args.compare_buffalo
        else None
    )

    if args.redis_url:
        gallery = load_gallery_from_redis(args.redis_url, args.gallery_key)
    elif args.photos_dir and args.photos_dir.is_dir():
        gallery = load_gallery_from_photos(args.photos_dir, det, emb, det_conf=0.35)
    else:
        gallery = GalleryIndex(names=[], matrix=np.empty((0, 512), dtype=np.float32))
    logger.info("Loaded %d AuraFace gallery embeddings", len(gallery.names))

    if buffalo_emb is not None and yolo_det is not None:
        old_gallery = _load_old_gallery(args, yolo_det, buffalo_emb, gpu)
        logger.info("Loaded %d buffalo_l gallery embeddings", len(old_gallery.names))
    else:
        old_gallery = GalleryIndex(names=[], matrix=np.empty((0, 512), dtype=np.float32))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * args.every_s)))
    mux_wh = (args.mux_width, args.mux_height)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, Any]] = []
    raw_faces_total = 0
    yolo_total_faces = 0
    yolo_both_matched = 0
    frame_idx = 0
    sampled = 0
    last_ts = 0.0

    while True:
        if args.max_frames > 0 and sampled >= args.max_frames:
            break
        if frame_idx % step != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        ts_s = pos_msec / 1000.0 if pos_msec > 0.0 else frame_idx / fps
        last_ts = max(last_ts, ts_s)

        mux = cv2.resize(frame, mux_wh, interpolation=cv2.INTER_LINEAR)
        person_boxes = pgie.extract_persons(mux)

        kept_faces: list[DetectedFace] = []
        face_entries: list[dict[str, Any]] = []
        yolo_only_faces: list[DetectedFace] = []
        yolo_only_entries: list[dict[str, Any]] = []
        yolo_all_entries: list[dict[str, Any]] = []

        for pi, (px1, py1, px2, py2) in enumerate(person_boxes):
            crop = mux[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            cand_raw = det.detect(crop, conf_thresh=args.det_conf)
            raw_faces_total += len(cand_raw)
            cand_kept = [f for f in cand_raw if min(f.width, f.height) >= args.min_pixels]

            yolo_iou_by_ci: dict[int, float] = {}
            yolo_box_by_ci: dict[int, list[float]] = {}
            if yolo_det is not None:
                y_raw = yolo_det.detect(crop, conf_thresh=args.yolo_conf)
                y_kept = [f for f in y_raw if min(f.width, f.height) >= args.min_pixels]
                yolo_total_faces += len(y_kept)
                matched, unmatched_y, _ = match_detections(
                    [f.bbox for f in y_kept], [f.bbox for f in cand_kept], iou_thresh=0.4
                )
                yolo_both_matched += len(matched)
                iou_by_yi: dict[int, float] = {}
                for yi, ci, iou in matched:
                    yolo_iou_by_ci[ci] = round(iou, 3)
                    yf_m = _offset_face(y_kept[yi], px1, py1)
                    yolo_box_by_ci[ci] = _round_box(yf_m.bbox)
                    iou_by_yi[yi] = round(iou, 3)
                unmatched_set = set(unmatched_y)
                for yi, yf_crop in enumerate(y_kept):
                    yf_global = _offset_face(yf_crop, px1, py1)
                    y_plaus = landmarks_plausible(yf_global.landmarks, yf_global.bbox)
                    y_yaw = face_yaw_from_5pt(yf_global.landmarks) if (y_plaus and yf_global.landmarks is not None) else None
                    old_match = (
                        match_face_open_set(
                            buffalo_emb.embed_face(mux, yf_global) if not old_gallery.is_empty else None,
                            old_gallery,
                            rec_thresh=args.buffalo_thresh,
                            rec_margin=args.buffalo_margin,
                        )
                        if buffalo_emb is not None
                        else {}
                    )
                    y_entry = {
                        "person_idx": pi,
                        "bbox": _round_box(yf_global.bbox),
                        "det_conf": round(float(yf_global.score), 4),
                        "short_side_px": round(float(min(yf_global.width, yf_global.height)), 1),
                        "plausible_kpts": bool(y_plaus),
                        "yaw": round(float(y_yaw), 3) if y_yaw is not None else None,
                        "matched_new": yi not in unmatched_set,
                        "new_iou": iou_by_yi.get(yi),
                        **old_match,
                    }
                    yolo_all_entries.append(y_entry)
                    if yi in unmatched_set:
                        yolo_only_faces.append(yf_global)
                        yolo_only_entries.append({
                            "person_idx": pi,
                            "bbox": _round_box(yf_global.bbox),
                            "yolo_conf": round(float(yf_global.score), 4),
                            "short_side_px": round(float(min(yf_global.width, yf_global.height)), 1),
                        })

            for ci, fc in enumerate(cand_kept):
                fg = _offset_face(fc, px1, py1)
                plaus = landmarks_plausible(fg.landmarks, fg.bbox)
                yaw = face_yaw_from_5pt(fg.landmarks) if (plaus and fg.landmarks is not None) else None
                vec = emb.embed_face(mux, fg) if not gallery.is_empty else None
                match_info = match_face_open_set(vec, gallery, rec_thresh=args.rec_thresh, rec_margin=args.rec_margin)
                kept_faces.append(fg)
                entry = {
                    "person_idx": pi,
                    "bbox": _round_box(fg.bbox),
                    "det_conf": round(float(fg.score), 4),
                    "short_side_px": round(float(min(fg.width, fg.height)), 1),
                    "plausible_kpts": bool(plaus),
                    "yaw": round(float(yaw), 3) if yaw is not None else None,
                    **match_info,
                }
                if yolo_det is not None:
                    entry["yolo_iou"] = yolo_iou_by_ci.get(ci)
                    if ci in yolo_box_by_ci:
                        entry["yolo_bbox"] = yolo_box_by_ci[ci]
                face_entries.append(entry)

        frame_rec: dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp_s": round(ts_s, 2),
            "timestamp_hms": format_timestamp_hms(ts_s),
            "person_boxes": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in person_boxes],
            "faces": face_entries,
            "yolo_only_faces": yolo_only_entries,
        }
        if yolo_det is not None:
            frame_rec["yolo_faces"] = yolo_all_entries
        if args.save_dir and _should_save_frame(frame_rec, args.save_mode):
            fname = f"t{ts_s:07.2f}s_f{frame_idx:06d}.jpg"
            out_img = args.save_dir / fname
            cv2.imwrite(str(out_img), _annotate_frame(mux, frame_rec, kept_faces, yolo_only_faces))
            frame_rec["snapshot"] = str(out_img)

        frame_records.append(frame_rec)
        sampled += 1
        frame_idx += 1

    cap.release()
    video_info = {
        "path": str(args.video),
        "native_wh": [native_w, native_h],
        "mux_wh": list(mux_wh),
        "fps": round(fps, 3),
        "duration_s": round(last_ts or (total_frames / fps if total_frames > 0 else 0.0), 2),
        "total_frames": total_frames,
        "sampled_frames": sampled,
    }
    config_info = {
        "det_onnx": str(args.det_onnx),
        "det_conf": args.det_conf,
        "min_pixels": args.min_pixels,
        "rec_onnx": str(args.rec_onnx),
        "rec_thresh": args.rec_thresh,
        "rec_margin": args.rec_margin,
        "pgie_onnx": str(args.pgie_onnx),
        "pgie_conf": args.pgie_conf,
        "every_s": args.every_s,
        "compare_yolo": compare_yolo,
        "compare_buffalo": bool(args.compare_buffalo),
        "buffalo_thresh": args.buffalo_thresh if args.compare_buffalo else None,
        "buffalo_margin": args.buffalo_margin if args.compare_buffalo else None,
    }
    return summarize_report(
        frame_records=frame_records,
        video_info=video_info,
        config_info=config_info,
        gallery_entries=len(gallery.names),
        raw_faces_total=raw_faces_total,
        yolo_enabled=compare_yolo,
        yolo_total_faces=yolo_total_faces,
        yolo_both_matched=yolo_both_matched,
        buffalo_enabled=bool(args.compare_buffalo),
        old_gallery_entries=len(old_gallery.names),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate face_det_lite + AuraFace on a video and write report.json")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--photos-dir", type=Path, default=None, help="Optional enrolment photos directory")
    ap.add_argument("--redis-url", default="", help="Optional Redis URL to read pre-enrolled gallery")
    ap.add_argument("--gallery-key", default="attendance:face:gallery")
    ap.add_argument("--every-s", type=float, default=0.5, help="Sample interval in seconds (default 0.5 = ~2 fps)")
    ap.add_argument("--max-frames", type=int, default=0, help="Max sampled frames (0 = full video)")
    ap.add_argument("--mux-width", type=int, default=1280)
    ap.add_argument("--mux-height", type=int, default=720)
    ap.add_argument("--pgie-onnx", type=Path, default=DEFAULT_PGIE_ONNX)
    ap.add_argument("--pgie-conf", type=float, default=0.35)
    ap.add_argument("--det-onnx", type=Path, default=DEFAULT_DET_ONNX)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--min-pixels", type=int, default=24)
    ap.add_argument("--rec-onnx", type=Path, default=DEFAULT_REC_ONNX)
    ap.add_argument("--rec-thresh", type=float, default=0.28)
    ap.add_argument("--rec-margin", type=float, default=0.05)
    ap.add_argument("--compare-yolo", action="store_true", help="Also run yolov8n-face on the same crops for hit/miss diff")
    ap.add_argument("--yolo-onnx", type=Path, default=DEFAULT_YOLO_ONNX)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--compare-buffalo", action="store_true", help="Also embed/match yolov8n-face detections with buffalo_l (implies --compare-yolo)")
    ap.add_argument("--buffalo-onnx", type=Path, default=DEFAULT_BUFFALO_ONNX)
    ap.add_argument("--buffalo-thresh", type=float, default=0.25)
    ap.add_argument("--buffalo-margin", type=float, default=0.05)
    ap.add_argument("--save-dir", type=Path, default=None, help="Optional dir for annotated JPEG inspection frames")
    ap.add_argument("--save-mode", choices=("inspect", "misses", "faces", "people", "all"), default="inspect")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    report = evaluate_video(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
