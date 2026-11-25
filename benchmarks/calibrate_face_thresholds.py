"""Threshold calibration and latency benchmark for deployed commercial face models.

Evaluates the deployed DeepStream ONNX stack:
  - Detector: perception/models/DeepStream-Yolo-Face/face_det_lite.onnx ([B, 3, 480, 640] -> [B, 4800, 15])
  - Embedder: perception/models/insightface/models/auraface/glintr100.onnx ([B, 3, 112, 112] -> [B, 512])

Suites:
  A. Gallery cross-person confusion floor + (threshold x margin) sweep on data/face_photos/
     (primary-only gallery vs multi/ probes, plus multi-embedding Leave-One-Out matching reregister_all.py)
  B. Detector pre-cluster-threshold sweep on Attendance clip person crops (--clips)
  C. Deployed ONNX Runtime latency (batch=1 / batch=16 detector, batch=1 embedder)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO / "benchmarks"))

from bench_face_detectors import (  # noqa: E402
    PgiePersonExtractor,
    _sample_clip_crops,
    compute_far_frr_curve,
    discover_enrolment_photos,
)
from face_bench_adapters import (  # noqa: E402
    ArcFaceOnnxEmbedder,
    DetectedFace,
    QualcommFaceDetLiteDetector,
    _time_ort_session,
    face_yaw_from_5pt,
    l2_normalize,
    landmarks_plausible,
)

logger = logging.getLogger("iris.face_calibrate")

DEFAULT_DET_ONNX = REPO / "perception" / "models" / "DeepStream-Yolo-Face" / "face_det_lite.onnx"
DEFAULT_REC_ONNX = REPO / "perception" / "models" / "insightface" / "models" / "auraface" / "glintr100.onnx"
DEFAULT_PGIE_ONNX = REPO / "perception" / "models" / "DeepStream-Yolo" / "yolo11x.onnx"

REC_THRESHOLDS = (0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35)
REC_MARGINS = (0.00, 0.03, 0.05, 0.08, 0.10)
DET_THRESHOLDS = (0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35)


def _select_largest(faces: Sequence[DetectedFace]) -> Optional[DetectedFace]:
    if not faces:
        return None
    return max(faces, key=lambda f: max(0.0, f.bbox[2] - f.bbox[0]) * max(0.0, f.bbox[3] - f.bbox[1]))


def _embed_photos(
    photos: Sequence[tuple[str, Path, bool]],
    det: QualcommFaceDetLiteDetector,
    emb: ArcFaceOnnxEmbedder,
    det_conf: float = 0.35,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for person_id, path, is_primary in photos:
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = det.detect(img, conf_thresh=det_conf)
        best = _select_largest(faces)
        if best is None:
            records.append({"person_id": person_id, "path": path, "is_primary": is_primary, "emb": None})
            continue
        vec = emb.embed_face(img, best)
        records.append({"person_id": person_id, "path": path, "is_primary": is_primary, "emb": vec})
    return records


def _confusion_floor(ids: Sequence[str], embs: np.ndarray) -> dict[str, Any]:
    if len(ids) < 2:
        return {"pairs": 0, "max": None, "p95": None, "mean": None, "top5": []}
    norm = l2_normalize(embs)
    sim = norm @ norm.T
    top_pairs: list[tuple[float, str, str]] = []
    vals: list[float] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] == ids[j]:
                continue
            s = float(sim[i, j])
            vals.append(s)
            top_pairs.append((s, ids[i], ids[j]))
    if not vals:
        return {"pairs": 0, "max": None, "p95": None, "mean": None, "top5": []}
    top_pairs.sort(key=lambda t: t[0], reverse=True)
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "pairs": len(vals),
        "max": round(float(np.max(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "top5": [{"sim": round(s, 4), "a": a, "b": b} for s, a, b in top_pairs[:5]],
    }


def _leave_one_out_sweep(
    ids: Sequence[str],
    embs: np.ndarray,
    paths: Sequence[Path],
    thresholds: Sequence[float],
    margins: Sequence[float],
) -> dict[str, Any]:
    """Leave-one-out evaluation across all enrolled embeddings (models full Attendance multi/ gallery)."""
    if len(ids) < 2:
        return {"n_samples": len(ids), "grid": [], "top5_impostors": []}
    norm = l2_normalize(embs)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -2.0)
    id_arr = np.asarray(ids)

    per_item: list[tuple[str, str, float, float, Path]] = []
    impostor_hits: list[tuple[float, str, str, float, Path]] = []
    for i, true_id in enumerate(ids):
        row = sim[i]
        best_idx = int(np.argmax(row))
        best_id = str(id_arr[best_idx])
        best_score = float(row[best_idx])
        other_mask = id_arr != best_id
        second_score = float(np.max(row[other_mask])) if np.any(other_mask) else -2.0
        per_item.append((true_id, best_id, best_score, best_score - second_score, paths[i]))

        non_self = id_arr != true_id
        if np.any(non_self):
            imp_idx = int(np.argmax(np.where(non_self, row, -2.0)))
            true_mask = np.copy(id_arr == true_id)
            true_mask[i] = False
            true_best = float(np.max(row[true_mask])) if np.any(true_mask) else -2.0
            impostor_hits.append((float(row[imp_idx]), true_id, str(id_arr[imp_idx]), float(row[imp_idx]) - true_best, paths[i]))

    impostor_hits.sort(key=lambda t: t[0], reverse=True)
    grid: list[dict[str, Any]] = []
    for margin in margins:
        for thr in thresholds:
            ta = fa = fr = 0
            for true_id, best_id, score, gap, _ in per_item:
                if not (score >= thr and gap >= margin):
                    fr += 1
                elif best_id != true_id:
                    fa += 1
                    fr += 1
                else:
                    ta += 1
            n = len(per_item)
            grid.append(
                {
                    "threshold": thr,
                    "margin": margin,
                    "tar": round(ta / n, 4),
                    "frr": round(fr / n, 4),
                    "far": round(fa / n, 4),
                    "false_accepts": fa,
                    "false_rejects": fr,
                }
            )
    return {
        "n_samples": len(ids),
        "grid": grid,
        "top5_impostors": [
            {"sim": round(s, 4), "true_id": tid, "matched_id": mid, "gap_over_true": round(g, 4), "probe": p.name}
            for s, tid, mid, g, p in impostor_hits[:5]
        ],
    }


def _match_embedding(
    vec: Optional[np.ndarray],
    gallery_ids: Sequence[str],
    gallery_norm: np.ndarray,
    rec_thresh: float,
    rec_margin: float,
) -> Optional[str]:
    if vec is None or len(gallery_ids) == 0:
        return None
    sims = gallery_norm @ l2_normalize(vec)
    best_idx = int(np.argmax(sims))
    best_id = gallery_ids[best_idx]
    best_score = float(sims[best_idx])
    other = [sims[i] for i, gid in enumerate(gallery_ids) if gid != best_id]
    second = float(max(other)) if other else -1.0
    if best_score >= rec_thresh and (best_score - second >= rec_margin if rec_margin > 0 else True):
        return best_id
    return None


def run_suite_a(
    photos_dir: Path,
    det: QualcommFaceDetLiteDetector,
    emb: ArcFaceOnnxEmbedder,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    discovered = discover_enrolment_photos(photos_dir)
    records = _embed_photos(discovered, det, emb, det_conf=0.35)
    embedded = [r for r in records if r["emb"] is not None]
    prim = [r for r in embedded if r["is_primary"]]
    probes = [r for r in embedded if not r["is_primary"]]

    prim_ids = [r["person_id"] for r in prim]
    prim_embs = np.stack([r["emb"] for r in prim]) if prim else np.empty((0, 512), dtype=np.float32)
    probe_ids = [r["person_id"] for r in probes]
    probe_embs = np.stack([r["emb"] for r in probes]) if probes else np.empty((0, 512), dtype=np.float32)
    all_ids = [r["person_id"] for r in embedded]
    all_embs = np.stack([r["emb"] for r in embedded]) if embedded else np.empty((0, 512), dtype=np.float32)
    all_paths = [r["path"] for r in embedded]

    primary_to_multi: list[dict[str, Any]] = []
    if len(prim_ids) > 0 and len(probe_ids) > 0:
        for margin in REC_MARGINS:
            curve = compute_far_frr_curve(prim_ids, prim_embs, probe_ids, probe_embs, REC_THRESHOLDS, margin=margin)
            primary_to_multi.extend(curve["operating_points"])

    result = {
        "photos_found": len(discovered),
        "photos_embedded": len(embedded),
        "primary_count": len(prim_ids),
        "multi_count": len(probe_ids),
        "primary_confusion_floor": _confusion_floor(prim_ids, prim_embs),
        "primary_to_multi_grid": primary_to_multi,
        "leave_one_out_all": _leave_one_out_sweep(all_ids, all_embs, all_paths, REC_THRESHOLDS, REC_MARGINS),
    }
    return result, all_ids, all_embs


def run_suite_b(
    clips: Sequence[Path],
    pgie_onnx: Optional[Path],
    det: QualcommFaceDetLiteDetector,
    emb: ArcFaceOnnxEmbedder,
    gallery_ids: Sequence[str],
    gallery_embs: np.ndarray,
    every_s: float,
    max_frames: int,
    gpu: bool,
) -> dict[str, Any]:
    if not clips:
        return {"skipped": True, "reason": "no clips supplied"}
    pgie = PgiePersonExtractor(pgie_onnx, gpu=gpu, conf_thresh=0.35) if pgie_onnx and pgie_onnx.is_file() else None
    crops = _sample_clip_crops(clips, every_s=every_s, max_frames_per_clip=max_frames, mux_wh=(1280, 720), pgie=pgie)
    if not crops:
        return {"skipped": True, "reason": "no crops extracted from clips"}

    g_norm = l2_normalize(gallery_embs) if len(gallery_ids) > 0 else np.empty((0, 512), dtype=np.float32)
    per_crop: list[list[tuple[DetectedFace, bool, float, Optional[str], Optional[str]]]] = []
    for item in crops:
        crop = item["crop"]
        faces = det.detect(crop, conf_thresh=min(DET_THRESHOLDS))
        annotated: list[tuple[DetectedFace, bool, float, Optional[str], Optional[str]]] = []
        for f in faces:
            bw = max(0.0, f.bbox[2] - f.bbox[0])
            bh = max(0.0, f.bbox[3] - f.bbox[1])
            plaus = landmarks_plausible(f.landmarks, f.bbox)
            yaw = face_yaw_from_5pt(f.landmarks) if (plaus and f.landmarks is not None) else 1.0
            vec = emb.embed_face(crop, f) if min(bw, bh) >= 24.0 else None
            match_25 = _match_embedding(vec, gallery_ids, g_norm, 0.25, 0.05)
            match_20 = _match_embedding(vec, gallery_ids, g_norm, 0.20, 0.05)
            annotated.append((f, plaus, yaw, match_25, match_20))
        per_crop.append(annotated)

    rows: list[dict[str, Any]] = []
    for thr in DET_THRESHOLDS:
        total = ge24 = barber_ok = plausible_n = multi_crop = matched_25 = matched_20 = 0
        for faces in per_crop:
            kept = [t for t in faces if t[0].score >= thr]
            if len(kept) > 1:
                multi_crop += 1
            for f, plaus, yaw, m25, m20 in kept:
                total += 1
                if plaus:
                    plausible_n += 1
                short = min(max(0.0, f.bbox[2] - f.bbox[0]), max(0.0, f.bbox[3] - f.bbox[1]))
                if short >= 24.0:
                    ge24 += 1
                    if m25 is not None:
                        matched_25 += 1
                    if m20 is not None:
                        matched_20 += 1
                if short >= 35.0 and plaus and yaw <= 0.35:
                    barber_ok += 1
        rows.append(
            {
                "det_thresh": thr,
                "detections": total,
                "faces_ge_24px": ge24,
                "barber_eligible": barber_ok,
                "plausible_kpt_pct": round(100.0 * plausible_n / max(1, total), 1),
                "multi_face_crops": multi_crop,
                "recognized_at_0_25_m0_05": matched_25,
                "recognized_at_0_20_m0_05": matched_20,
            }
        )
    return {"crops_evaluated": len(crops), "pgie_used": pgie is not None, "operating_points": rows}


def run_suite_c(det: QualcommFaceDetLiteDetector, emb: ArcFaceOnnxEmbedder) -> dict[str, Any]:
    out: dict[str, Any] = {
        "providers_det": det.sess.get_providers(),
        "providers_rec": emb.sess.get_providers(),
        "det_b1_ms": round(det.benchmark_batch(batch_size=1, iters=30, warmup=5), 3),
        "det_b16_ms": round(det.benchmark_batch(batch_size=16, iters=20, warmup=5), 3),
        "rec_b1_ms": round(_time_ort_session(emb.sess, (1, 3, 112, 112), iters=30, warmup=5), 3),
    }
    rec_out_batch = emb.sess.get_outputs()[0].shape[0]
    if isinstance(rec_out_batch, int) and rec_out_batch == 1:
        out["rec_b8_ms"] = "static b=1 output metadata"
    else:
        try:
            out["rec_b8_ms"] = round(_time_ort_session(emb.sess, (8, 3, 112, 112), iters=20, warmup=5), 3)
        except Exception as exc:  # noqa: BLE001
            out["rec_b8_ms"] = f"unsupported ({exc})"
    return out


def _pick_best(grid: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not grid:
        return None
    clean = [g for g in grid if g["false_accepts"] == 0]
    pool = clean if clean else list(grid)
    return max(pool, key=lambda g: (g["tar"], -g["false_accepts"], g["threshold"], g["margin"]))


def format_markdown(report: dict[str, Any]) -> str:
    lines = ["# Commercial Face Stack Calibration Report (`face_det_lite` + `auraface`)", ""]
    a = report["suite_a"]
    fl = a["primary_confusion_floor"]
    lines += [
        "## Suite A: Enrolment Gallery Confusion Floor & `(threshold x margin)` Sweep",
        "",
        f"- **Photos:** {a['photos_embedded']}/{a['photos_found']} embedded ({a['primary_count']} primary, {a['multi_count']} extra `multi/`)",
        f"- **Primary Cross-Person Floor:** max={fl['max']}, p95={fl['p95']}, mean={fl['mean']} (pairs={fl['pairs']})",
        "",
    ]
    if fl["top5"]:
        lines.append("**Top Closest Primary Pairs:** " + ", ".join(f"{t['a']}~{t['b']}={t['sim']}" for t in fl["top5"]))
        lines.append("")

    p2m = a.get("primary_to_multi_grid") or []
    if p2m:
        best_p2m = _pick_best(p2m)
        lines += [
            f"### Primary Gallery ({a['primary_count']}) vs `multi/` Probes ({a['multi_count']}) (best: thresh={best_p2m['threshold']}, margin={best_p2m['margin']} -> TAR={best_p2m['tar']:.1%}, FA={best_p2m['false_accepts']})",
            "",
            "| Margin | Threshold | TAR | FRR | FAR | FA | FR |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in p2m:
            if row["margin"] in (0.0, 0.05, 0.08):
                lines.append(
                    f"| {row['margin']:.2f} | {row['threshold']:.2f} | {row['tar']:.1%} | {row['frr']:.1%} | "
                    f"{row['far']:.1%} | {row['false_accepts']} | {row['false_rejects']} |"
                )
        lines.append("")

    loo = a["leave_one_out_all"]
    if loo["grid"]:
        best_loo = _pick_best(loo["grid"])
        lines += [
            f"### Leave-One-Out Across All Enrolled Photos (n={loo['n_samples']}, best: thresh={best_loo['threshold']}, margin={best_loo['margin']} -> TAR={best_loo['tar']:.1%}, FA={best_loo['false_accepts']})",
            "",
            "| Margin | Threshold | TAR | FRR | FAR | FA | FR |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in loo["grid"]:
            if row["margin"] in (0.0, 0.05, 0.08):
                lines.append(
                    f"| {row['margin']:.2f} | {row['threshold']:.2f} | {row['tar']:.1%} | {row['frr']:.1%} | "
                    f"{row['far']:.1%} | {row['false_accepts']} | {row['false_rejects']} |"
                )
        lines.append("")
    if loo["top5_impostors"]:
        lines.append("**Top 5 Hardest Impostor Hits (LOO):**")
        for imp in loo["top5_impostors"]:
            lines.append(f"- `{imp['probe']}` (true={imp['true_id']} -> matched={imp['matched_id']} @ {imp['sim']}, gap_over_true={imp['gap_over_true']})")
        lines.append("")

    b = report["suite_b"]
    lines.append("## Suite B: Attendance Clip Detector `pre-cluster-threshold` Sweep")
    if b.get("skipped"):
        lines.append(f"- Skipped: {b['reason']}")
    else:
        lines += [
            f"- Crops evaluated: {b['crops_evaluated']} (PGIE person extractor: {b['pgie_used']})",
            "",
            "| `det_thresh` | Detections | `>=24px` (Att) | Barber Eligible | Plausible Kpts | Multi-Face Crops | Rec@0.25/m0.05 | Rec@0.20/m0.05 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in b["operating_points"]:
            lines.append(
                f"| {r['det_thresh']:.2f} | {r['detections']} | {r['faces_ge_24px']} | {r['barber_eligible']} | "
                f"{r['plausible_kpt_pct']}% | {r['multi_face_crops']} | {r['recognized_at_0_25_m0_05']} | {r['recognized_at_0_20_m0_05']} |"
            )
    lines.append("")

    c = report["suite_c"]
    lines += [
        "## Suite C: Deployed ONNX Latency",
        "",
        f"- **Detector (`face_det_lite.onnx`, {c['providers_det'][0]}):** `b=1`: **{c['det_b1_ms']} ms**, `b=16`: **{c['det_b16_ms']} ms**",
        f"- **Embedder (`glintr100.onnx`, {c['providers_rec'][0]}):** `b=1`: **{c['rec_b1_ms']} ms**, `b=8`: **{c['rec_b8_ms']}**",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark and calibrate face_det_lite + auraface thresholds")
    ap.add_argument("--det-onnx", type=Path, default=DEFAULT_DET_ONNX)
    ap.add_argument("--rec-onnx", type=Path, default=DEFAULT_REC_ONNX)
    ap.add_argument("--pgie-onnx", type=Path, default=DEFAULT_PGIE_ONNX)
    ap.add_argument("--photos-dir", type=Path, default=REPO / "data" / "face_photos")
    ap.add_argument("--clips", nargs="*", type=Path, default=[])
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "benchmarks" / "results" / "face_calibration")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gpu = not args.cpu
    det = QualcommFaceDetLiteDetector(args.det_onnx, gpu=gpu, default_conf=0.35)
    emb = ArcFaceOnnxEmbedder("auraface", args.rec_onnx, gpu=gpu)

    suite_a, all_ids, all_embs = run_suite_a(args.photos_dir, det, emb)
    suite_b = run_suite_b(args.clips, args.pgie_onnx, det, emb, all_ids, all_embs, args.every_s, args.max_frames, gpu)
    suite_c = run_suite_c(det, emb)

    report = {"suite_a": suite_a, "suite_b": suite_b, "suite_c": suite_c}
    md = format_markdown(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "calibration_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
