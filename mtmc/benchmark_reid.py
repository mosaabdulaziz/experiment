#!/usr/bin/env python3
"""Benchmark vehicle re-ID models for cross-camera matching on real crops.

Answers, per model, the three questions that decide whether appearance
embedding + cosine matching is paying off:

1. **Retrieval quality** — when a query vehicle instance/crop on camera A is
   matched against everything on the other cameras, where does the true
   same-vehicle match land?  (Rank@1, Rank@2 "within 2 best", Rank@5, mAP)
2. **False positives** — what fraction of cross-camera candidates above a
   similarity threshold are different vehicles?  Precision/recall/FPR over a
   threshold sweep plus the FP count at the production threshold (0.55).
3. **Cost** — throughput (crops/s), single-crop latency, VRAM, params, size.

Ground truth comes from ``chosen crops/vehicle_*`` (see reid_bench_data.py);
every other crop is a gallery distractor / negative candidate.  Unknown
distractors may include unlabeled appearances of curated vehicles, so metrics
that include them are pessimistic.

Usage:
    .venv/bin/python -m mtmc.benchmark_reid --models fastreid-veri-sbs-r50-ibn resnet50-imagenet
    .venv/bin/python -m mtmc.benchmark_reid --models all --batch-size 64
    .venv/bin/python -m mtmc.benchmark_reid --models clip-vit-b32 dinov2-base --no-distractors
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .reid_bench_data import ReidDataset, build_dataset, gt_summary
from .reid_models import REGISTRY, get_model

OUT_ROOT = Path("mtmc/output/reid_bench")
PRODUCTION_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# embedding all crops once per model
# ---------------------------------------------------------------------------


def embed_all_crops(
    ds: ReidDataset,
    model_name: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Embed every crop referenced by the dataset. Returns {crop_key: vec}."""
    import torch
    from .reid_models import get_model

    handle = get_model(model_name)
    keys = sorted(ds.crop_root)
    embs: dict[str, np.ndarray] = {}

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    n_done = 0
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        crops = []
        ok_chunk = []
        for k in chunk:
            im = cv2.imread(str(ds.crop_root[k]))
            if im is None or im.size == 0 or min(im.shape[:2]) < 8:
                continue
            crops.append(im)
            ok_chunk.append(k)
        if not crops:
            continue
        feats = handle.embed_batch(crops)
        for k, v in zip(ok_chunk, feats):
            embs[k] = v
        n_done += len(feats)
        el = time.time() - t0
        if el > 0:
            print(f"      {n_done}/{len(keys)} crops  ({n_done/el:.0f} crops/s)", end="\r")
    print()
    elapsed = time.time() - t0

    stats = {
        "crops_embedded": len(embs),
        "total_s": round(elapsed, 2),
        "crops_per_s": round(len(embs) / elapsed, 1) if elapsed > 0 else 0.0,
        "ms_per_crop": round(elapsed * 1000 / max(len(embs), 1), 2),
    }
    if torch.cuda.is_available():
        stats["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return embs, stats


# ---------------------------------------------------------------------------
# retrieval metrics
# ---------------------------------------------------------------------------


def _inst_embedding(inst, embs: dict[str, np.ndarray]) -> np.ndarray | None:
    vecs = [embs[k] for k in inst.crop_keys if k in embs]
    if not vecs:
        return None
    mean = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 0 else None


def instance_retrieval(
    ds: ReidDataset,
    embs: dict[str, np.ndarray],
    include_distractors: bool,
) -> dict:
    """Rank every labeled multi-cam instance against other-camera instances."""
    # queryable: labeled instances whose identity appears on >= 2 cameras
    cams_by_id: dict[str, set[str]] = {}
    for inst in ds.labeled_instances:
        cams_by_id.setdefault(inst.identity, set()).add(inst.cam)
    queries = [i for i in ds.labeled_instances
               if len(cams_by_id.get(i.identity, set())) >= 2]

    inst_emb: dict[str, np.ndarray] = {}
    for inst in ds.instances:
        v = _inst_embedding(inst, embs)
        if v is not None:
            inst_emb[inst.uid] = v

    # per-query results: best positive rank, reciprocal rank, AP
    per_query = []
    cam_pair_hits: dict[tuple[str, str], list[int]] = {}
    for q in queries:
        if q.uid not in inst_emb:
            continue
        qv = inst_emb[q.uid]
        cands = []
        for inst in ds.instances:
            if inst.cam == q.cam or inst.uid == q.uid:
                continue
            if inst.identity is None and not include_distractors:
                continue
            if inst.uid not in inst_emb:
                continue
            cands.append(inst)
        if not cands:
            continue
        sims = np.array([float(inst_emb[c.uid] @ qv) for c in cands])
        order = np.argsort(-sims, kind="stable")
        ranked = [cands[i] for i in order]

        pos_rank = None
        for r, inst in enumerate(ranked):
            if inst.identity == q.identity:
                pos_rank = r
                break
        # best positive per positive camera (for cam-pair table)
        best_by_cam: dict[str, int] = {}
        for r, inst in enumerate(ranked):
            if inst.identity == q.identity and inst.cam not in best_by_cam:
                best_by_cam[inst.cam] = r
        for pcam, r in best_by_cam.items():
            cam_pair_hits.setdefault((q.cam, pcam), []).append(1 if r == 0 else 0)

        if pos_rank is None:
            continue
        per_query.append({
            "query": q.uid,
            "identity": q.identity,
            "cam": q.cam,
            "rank": pos_rank + 1,
            "rr": 1.0 / (pos_rank + 1),
        })

    if not per_query:
        return {"error": "no evaluable queries"}

    n = len(per_query)
    ranks = np.array([p["rank"] for p in per_query])
    hits = {k: float(np.mean(ranks <= k)) for k in (1, 2, 3, 5)}
    return {
        "n_queries": n,
        "rank1": round(hits[1], 4),
        "rank2_within_2_best": round(hits[2], 4),
        "rank3": round(hits[3], 4),
        "rank5": round(hits[5], 4),
        "mean_reciprocal_rank": round(float(np.mean([p["rr"] for p in per_query])), 4),
        "median_rank": int(np.median(ranks)),
        "max_rank": int(ranks.max()),
        "gallery_size_mean": None,  # filled below by caller if desired
        "top1_wrong_rate": round(1.0 - hits[1], 4),
        "cam_pair_rank1": {
            f"{a}->{b}": round(np.mean(v), 3)
            for (a, b), v in sorted(cam_pair_hits.items())
        },
        "per_query": per_query,
    }


def crop_retrieval(
    ds: ReidDataset,
    embs: dict[str, np.ndarray],
    include_distractors: bool,
) -> dict:
    """Rank every labeled crop against crops from the other cameras."""
    cams_by_id: dict[str, set[str]] = {}
    for inst in ds.labeled_instances:
        cams_by_id.setdefault(inst.identity, set()).add(inst.cam)
    multi = {k for k, v in cams_by_id.items() if len(v) >= 2}

    queries: list[tuple[str, str, str]] = []   # (key, cam, identity)
    for inst in ds.labeled_instances:
        if inst.identity not in multi:
            continue
        for k in inst.crop_keys:
            if k in embs:
                queries.append((k, inst.cam, inst.identity))

    gallery: list[tuple[str, str, str | None]] = []
    for inst in ds.instances:
        for k in inst.crop_keys:
            if k in embs:
                gallery.append((k, inst.cam, inst.identity))

    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    n = 0
    for qk, qcam, qid in queries:
        qv = embs[qk]
        sims, cands = [], []
        for gk, gcam, gid in gallery:
            if gcam == qcam:
                continue
            if gid is None and not include_distractors:
                continue
            cands.append((gk, gid))
            sims.append(float(embs[gk] @ qv))
        if not cands:
            continue
        order = np.argsort(-np.asarray(sims), kind="stable")
        rank = None
        for pos, idx in enumerate(order):
            if cands[idx][1] == qid:
                rank = pos + 1
                break
        if rank is None:
            continue
        n += 1
        for k in hits:
            if rank <= k:
                hits[k] += 1
    if n == 0:
        return {"error": "no evaluable crop queries"}
    return {
        "n_query_crops": n,
        "rank1": round(hits[1] / n, 4),
        "rank2_within_2_best": round(hits[2] / n, 4),
        "rank3": round(hits[3] / n, 4),
        "rank5": round(hits[5] / n, 4),
    }


# ---------------------------------------------------------------------------
# verification / association-style (threshold scan)
# ---------------------------------------------------------------------------


def verification_scan(ds: ReidDataset, embs: dict[str, np.ndarray]) -> dict:
    """All cross-camera labeled instance pairs; positives= same identity."""
    inst_emb = {}
    for inst in ds.instances:
        v = _inst_embedding(inst, embs)
        if v is not None:
            inst_emb[inst.uid] = v

    lab = [i for i in ds.instances if i.identity is not None]
    pos_sims, neg_sims = [], []
    pos_pairs, neg_pairs = [], []
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            a, b = lab[i], lab[j]
            if a.cam == b.cam:
                continue
            if a.uid not in inst_emb or b.uid not in inst_emb:
                continue
            sim = float(inst_emb[a.uid] @ inst_emb[b.uid])
            if a.identity == b.identity:
                pos_sims.append(sim)
                pos_pairs.append((a.uid, b.uid, sim))
            else:
                neg_sims.append(sim)
                neg_pairs.append((a.uid, b.uid, sim))

    # pessimistic negatives: also any pair involving an unknown instance
    unk_pairs = []
    all_insts = list(inst_emb)
    # sample unknown-involving pairs to keep memory sane
    unk_insts = [u for u in ds.unknown_instances if u.uid in inst_emb]
    lab_insts = lab
    rng = np.random.default_rng(0)
    sampled = []
    # unknown x labeled
    for u in unk_insts:
        for l in lab_insts:
            if u.cam != l.cam and l.uid in inst_emb and u.uid in inst_emb:
                sampled.append((u, l))
    # unknown x unknown (cap for speed)
    for a in unk_insts:
        for b in rng.choice(unk_insts, size=min(40, len(unk_insts)), replace=False):
            if a.uid < b.uid and a.cam != b.cam and b.uid in inst_emb:
                sampled.append((a, b))
    unk_sims = [float(inst_emb[a.uid] @ inst_emb[b.uid]) for a, b in sampled]

    pos_arr = np.array(pos_sims)
    neg_arr = np.array(neg_sims)
    unk_arr = np.array(unk_sims) if unk_sims else np.array([])

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.901, 0.05)] + \
                 [PRODUCTION_THRESHOLD]
    thresholds = sorted(set(thresholds))
    rows = []
    for t in thresholds:
        tp = int((pos_arr >= t).sum())
        fp_clean = int((neg_arr >= t).sum())
        fp_pess = fp_clean + int((unk_arr >= t).sum())
        fn = len(pos_arr) - tp
        tn_clean = len(neg_arr) - fp_clean
        prec = tp / (tp + fp_clean) if (tp + fp_clean) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "threshold": t,
            "tp": tp, "fn": fn,
            "fp_clean": fp_clean, "tn_clean": tn_clean,
            "fp_incl_unknown": fp_pess,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr_clean": round(fp_clean / len(neg_arr), 4) if len(neg_arr) else 0.0,
            "fpr_incl_unknown": round(fp_pess / (len(neg_arr) + len(unk_arr)), 4)
            if (len(neg_arr) + len(unk_arr)) else 0.0,
        })

    # best F1 (clean) operating point + stats
    best = max(rows, key=lambda r: r["f1"])
    prod = next(r for r in rows if r["threshold"] == PRODUCTION_THRESHOLD)
    d = {
        "n_positive_pairs": len(pos_arr),
        "n_clean_negative_pairs": len(neg_arr),
        "n_unknown_negative_pairs": len(unk_arr),
        "true_sim_mean": round(float(pos_arr.mean()), 4) if len(pos_arr) else None,
        "true_sim_std": round(float(pos_arr.std()), 4) if len(pos_arr) else None,
        "false_sim_mean": round(float(neg_arr.mean()), 4) if len(neg_arr) else None,
        "false_sim_std": round(float(neg_arr.std()), 4) if len(neg_arr) else None,
        "unknown_sim_mean": round(float(unk_arr.mean()), 4) if len(unk_arr) else None,
        # separation (Cohen's d)
        "separation_d": None,
        "best_f1_point": best,
        "production_0_55": prod,
        "threshold_scan": rows,
        "top_fp_pairs": sorted(neg_pairs, key=lambda x: -x[2])[:15],
    }
    if len(pos_arr) and len(neg_arr):
        pooled = np.sqrt((pos_arr.std() ** 2 + neg_arr.std() ** 2) / 2)
        d["separation_d"] = round(float((pos_arr.mean() - neg_arr.mean()) / pooled), 3) \
            if pooled > 0 else None
    return d


# ---------------------------------------------------------------------------
# FP montage writer
# ---------------------------------------------------------------------------


def _strip(a: np.ndarray, b: np.ndarray, cap_a: str, cap_b: str, h: int = 220):
    def _pad(im, cap):
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        bar = np.full((28, im.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, cap[:60], (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if "TP" in cap else (0, 200, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, im])
    return np.hstack([_pad(a, cap_a), np.full((h + 28, 8, 3), 255, np.uint8), _pad(b, cap_b)])


def write_fp_montage(pairs: list, embs_path_map, ds, title: str, out_path: Path,
                     model: str) -> None:
    """pairs: list of (uid_a, uid_b, sim, note). Render one sheet of strips."""
    from .reid_bench_data import Instance
    insts = {i.uid: i for i in ds.instances}
    out_path.mkdir(parents=True, exist_ok=True)
    cells = []
    for uid_a, uid_b, sim, note in pairs:
        ia, ib = insts.get(uid_a), insts.get(uid_b)
        if ia is None or ib is None or not ia.crop_keys or not ib.crop_keys:
            continue
        ima = cv2.imread(str(ds.crop_root[ia.crop_keys[0]]))
        imb = cv2.imread(str(ds.crop_root[ib.crop_keys[0]]))
        if ima is None or imb is None:
            continue
        cap_a = f"{ia.identity or 'UNK'} {ia.cam} t{ia.tid:04d}"
        cap_b = f"{ib.identity or 'UNK'} {ib.cam} t{ib.tid:04d}"
        cells.append(_strip(ima, imb, cap_a, cap_b))
    if not cells:
        return
    h = max(c.shape[0] for c in cells)
    sheet = np.full((h * len(cells), max(c.shape[1] for c in cells), 3), 30, np.uint8)
    for i, c in enumerate(cells):
        sheet[i * h:i * h + c.shape[0], :c.shape[1]] = c
    p = out_path / f"{title}.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"    FP montage: {p}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_model(
    ds: ReidDataset,
    model_name: str,
    batch_size: int,
    include_distractors: bool,
    out_dir: Path,
) -> dict:
    print(f"\n{'=' * 78}\nMODEL: {model_name}\n{'=' * 78}")
    t_all = time.time()
    handle = get_model(model_name)
    print(f"  loading {model_name} ...")
    t0 = time.time()
    try:
        handle.ensure()
    except Exception as exc:
        print(f"  [ERROR] model failed to load: {exc}")
        return {"model": model_name, "error": str(exc)}
    load_s = time.time() - t0

    print(f"  embedding {len(ds.crop_root)} crops ...")
    embs, estats = embed_all_crops(ds, model_name, batch_size)
    estats["load_s"] = round(load_s, 2)

    # ---- retrieval -----------------------------------------------------
    print("  instance-level retrieval ...")
    inst_res = instance_retrieval(ds, embs, include_distractors)
    print("  crop-level retrieval ...")
    crop_res = crop_retrieval(ds, embs, include_distractors)
    print("  verification threshold scan ...")
    verif = verification_scan(ds, embs)

    # gallery size (mean number of candidates per query)
    if "n_queries" in inst_res and inst_res.get("n_queries"):
        n_cands = []
        for q in inst_res.get("per_query", []):
            pass
        # recompute quickly for the report
        qids = {p["query"] for p in inst_res.get("per_query", [])}
        for q in ds.labeled_instances:
            if q.uid in qids:
                n = sum(1 for i in ds.instances
                        if i.cam != q.cam and (include_distractors or i.identity is not None))
                n_cands.append(n)
        if n_cands:
            inst_res["gallery_size_mean"] = round(float(np.mean(n_cands)), 1)
            inst_res["gallery_size_max"] = int(max(n_cands))

    # ---- FP montages ----------------------------------------------------
    fpdir = out_dir / "fp_montages" / model_name
    top_fp = verif.get("top_fp_pairs", [])[:12]
    # choose pairs at the production threshold for realism
    prod = verif.get("production_0_55", {})
    thr = prod.get("threshold", PRODUCTION_THRESHOLD)
    all_neg = verif.get("top_fp_pairs", [])
    write_fp_montage(
        [(a, b, s, "") for a, b, s in all_neg[:12]],
        None, ds, f"fp_top_sim_{model_name}", fpdir, model_name,
    )

    result = {
        "model": model_name,
        "dim": handle.dim,
        "params": handle.params,
        "device": handle.device,
        "load_s": round(load_s, 2),
        "embed": estats,
        "retrieval_instances": inst_res,
        "retrieval_crops": crop_res,
        "verification": {k: v for k, v in verif.items() if k != "top_fp_pairs"},
        "top_fp_pairs": [
            {"a": a, "b": b, "sim": round(s, 4)} for a, b, s in all_neg[:15]
        ],
        "elapsed_total_s": round(time.time() - t_all, 1),
    }
    return result


def summarize(all_results: dict, ds: ReidDataset) -> str:
    lines = []
    lines.append("# Vehicle Re-ID benchmark — cross-camera crops")
    lines.append("")
    lines.append("Dataset & ground truth: see `mtmc/reid_bench_data.py`.")
    lines.append("")
    lines.append("```")
    import io
    buf = io.StringIO()

    def P(*a):
        print(*a, file=buf)

    summary = gt_summary(ds)
    P("GROUND TRUTH")
    P(f"  cameras: {', '.join(summary['cameras'])}")
    P(f"  instances: {summary['n_instances']} "
      f"(labeled {summary['n_labeled_instances']}, "
      f"unknown/distractor {summary['n_unknown_instances']})")
    P(f"  identities: {summary['n_identities']} "
      f"(multi-camera {summary['n_multi_camera_identities']})")
    P(f"  positive cross-cam instance pairs: {summary['n_positive_cross_cam_pairs']}")
    P(f"  {summary['note']}")
    P("")

    # ---- retrieval table ------------------------------------------------
    P("RETRIEVAL — same-vehicle match must rank within top-k of the other cameras' gallery")
    hdr = (f"{'model':<28} {'q':>4} {'gal':>5} {'R@1':>6} {'R@2':>6} "
           f"{'R@5':>6} {'mRR':>6} {'top1-WR':>8}")
    P(hdr); P("-" * len(hdr))
    for name, r in sorted(all_results.items()):
        ir = r.get("retrieval_instances") or {}
        if "error" in ir:
            P(f"{name:<28} {ir['error']}")
            continue
        P(f"{name:<28} {ir.get('n_queries', 0):>4} "
          f"{ir.get('gallery_size_mean', '?'):>5} "
          f"{ir.get('rank1', 0):>6.3f} {ir.get('rank2_within_2_best', 0):>6.3f} "
          f"{ir.get('rank5', 0):>6.3f} {ir.get('mean_reciprocal_rank', 0):>6.3f} "
          f"{ir.get('top1_wrong_rate', 1):>8.3f}")
    P("")
    P("CROP-LEVEL RETRIEVAL (each curated crop as its own query)")
    P(f"{'model':<28} {'q':>4} {'R@1':>6} {'R@2':>6} {'R@5':>6}")
    for name, r in sorted(all_results.items()):
        cr = r.get("retrieval_crops") or {}
        if "error" in cr:
            P(f"{name:<28} {cr['error']}")
            continue
        P(f"{name:<28} {cr.get('n_query_crops', 0):>4} "
          f"{cr.get('rank1', 0):>6.3f} {cr.get('rank2_within_2_best', 0):>6.3f} "
          f"{cr.get('rank5', 0):>6.3f}")
    P("")

    # ---- similarity separation ------------------------------------------
    P("VERIFICATION — similarity of true same-vehicle pairs vs different-vehicle pairs")
    P(f"{'model':<28} {'TP mean':>8} {'TP std':>7} {'FP mean':>8} {'FP std':>7} "
      f"{'d (sep)':>8} {'FP% @0.55':>10}")
    for name, r in sorted(all_results.items()):
        v = r.get("verification") or {}
        if not v.get("n_positive_pairs"):
            P(f"{name:<28} (no pairs)")
            continue
        prod = v.get("production_0_55") or {}
        P(f"{name:<28} {str(v.get('true_sim_mean')):>8} {str(v.get('true_sim_std')):>7} "
          f"{str(v.get('false_sim_mean')):>8} {str(v.get('false_sim_std')):>7} "
          f"{str(v.get('separation_d')):>8} "
          f"{prod.get('fpr_clean', 0):>9.1%}")
    P("")

    # ---- operating points ------------------------------------------------
    P("OPERATING POINTS (clean labeled pairs; production threshold = 0.55)")
    P(f"{'model':<28} {'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7} {'FP#':>5} "
      f"{'bestF1(thr/P/R)':>20}")
    for name, r in sorted(all_results.items()):
        v = r.get("verification") or {}
        prod = v.get("production_0_55") or {}
        best = v.get("best_f1_point") or {}
        if not v.get("n_positive_pairs"):
            P(f"{name:<28} (no pairs)")
            continue
        P(f"{name:<28} {prod.get('threshold', 0):>5.2f} "
          f"{prod.get('precision', 0):>7.1%} {prod.get('recall', 0):>7.1%} "
          f"{prod.get('f1', 0):>7.3f} {prod.get('fpr_clean', 0):>7.1%} "
          f"{prod.get('fp_clean', 0):>5} "
          f"{best.get('threshold', 0):>10.2f}/{best.get('precision', 0):.2f}/"
          f"{best.get('recall', 0):.2f}")
    P("")

    # ---- efficiency -------------------------------------------------------
    P("EFFICIENCY (params x4B = approx fp32 size; fp16 ~half)")
    P(f"{'model':<28} {'dim':>5} {'params':>9} {'sizeMB':>8} {'load_s':>7} "
      f"{'crops/s':>8} {'ms/crop':>8} {'VRAM MB':>8}")
    for name, r in sorted(all_results.items()):
        e = r.get("embed") or {}
        params = r.get("params", 0)
        P(f"{name:<28} {r.get('dim', 0):>5} {params:>9} {params*4/1e6:>8.1f} "
          f"{e.get('load_s', 0):>7.1f} {e.get('crops_per_s', 0):>8.1f} "
          f"{e.get('ms_per_crop', 0):>8.2f} {e.get('vram_peak_mb', 0):>8.1f}")
    P("```")
    lines.append(buf.getvalue())
    lines.append("")
    lines.append("## Reading the numbers")
    lines.append("")
    lines.append("- **R@1 / R@2**: if production compares a query vehicle against "
                 "all candidates of another camera, R@1 = share of queries where the "
                 "correct vehicle ranks first; R@2 = the \"within 2 best\" the "
                 "association should keep as candidates.")
    lines.append("- **top1-WR**: share of queries whose top-1 candidate is a "
                 "DIFFERENT vehicle — these would be false merges without a "
                 "threshold/time gate.")
    lines.append("- **FP%@0.55**: of all cross-camera candidate pairs above the "
                 "production similarity threshold, the share that are different "
                 "vehicles (false merges).  Distractors excluded in this table.")
    lines.append("- Metrics include unknown/unlabeled crops only where noted; "
                 "they may contain unlabeled appearances of the curated vehicles, "
                 "so any such number is a pessimistic bound.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model names; default all except heavyweight download models, "
                         "or 'all' for everything in the registry")
    ap.add_argument("--crops-root", default="mtmc/output/vehicle_crops")
    ap.add_argument("--distractor-crops", type=int, default=3,
                    help="max crops sampled per unknown tracklet (default 3)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-distractors", action="store_true",
                    help="exclude unknown vehicles from retrieval galleries "
                         "(keeps them in verification as pessimistic negatives)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the combined report from saved result_<model>.json "
                         "files without re-running any model")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        ds = build_dataset(args.crops_root,
                           unknown_crops_per_tracklet=args.distractor_crops)
        all_results = {}
        for jf in sorted(out_dir.glob("result_*.json")):
            name = jf.name[len("result_"):-len(".json")]
            all_results[name] = json.loads(jf.read_text())
        report = summarize(all_results, ds)
        print(report)
        (out_dir / "REPORT.md").write_text(report)
        (out_dir / "results_all.json").write_text(
            json.dumps(all_results, indent=2, default=str))
        print(f"\nReport: {out_dir / 'REPORT.md'}")
        return

    ds = build_dataset(args.crops_root,
                       unknown_crops_per_tracklet=args.distractor_crops)
    summary = gt_summary(ds)
    print("Dataset summary:")
    for k in ("cameras", "n_instances", "n_labeled_instances",
              "n_unknown_instances", "n_identities",
              "n_multi_camera_identities", "n_positive_cross_cam_pairs"):
        print(f"  {k}: {summary[k]}")
    print(f"  multi-camera identities: {summary['multi_camera_identities']}")

    if args.models is None:
        names = ["fastreid-veri-sbs-r50-ibn", "resnet50-imagenet",
                 "clip-vit-b32", "dinov2-base"]
    elif list(args.models) == ["all"]:
        names = sorted(REGISTRY)
    else:
        names = args.models
    for n in names:
        if n not in REGISTRY:
            sys.exit(f"unknown model {n!r}; available: {sorted(REGISTRY)}")

    all_results = {}
    for name in names:
        res = run_model(ds, name, args.batch_size,
                        include_distractors=not args.no_distractors,
                        out_dir=out_dir)
        all_results[name] = res
        with open(out_dir / f"result_{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  -> {out_dir / ('result_' + name + '.json')}")

    report = summarize(all_results, ds)
    print(report)
    (out_dir / "REPORT.md").write_text(report)
    with open(out_dir / "results_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport: {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
