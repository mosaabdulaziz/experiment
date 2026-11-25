#!/usr/bin/env python3
"""Compare vision-encoder-only (embedding + cosine) matching against the VLM
pairwise verdicts from compare_vehicles.py, using the SAME protocol and the
SAME representatives.

For each encoder in the registry (that builds successfully on this machine):
  * within-folder pairs: rep crop vs every other crop of the same vehicle
    (expect 'same')
  * cross-folder pairs: rep vs rep of every other vehicle (expect 'different')
  * explicit self-checks and the curated user pair
then thresholds are swept on cosine similarity and accuracy / F1 /  rank-1
are reported so we can directly compare with the VLM's decision accuracy.

Usage:
    python -m mtmc.benchmark_encoder_vs_vlm
    python -m mtmc.benchmark_encoder_vs_vlm --models resnet50-imagenet fastreid-veri-sbs-r50-ibn
    python -m mtmc.benchmark_encoder_vs_vlm --json-out output/encoder_vs_vlm.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .reid_bench_data import build_dataset, gt_summary
from .reid_models import REGISTRY, get_model

BASE = Path(__file__).parent / "output" / "vehicle_crops" / "chosen crops"
OUT_DEFAULT = Path(__file__).parent / "output" / "encoder_vs_vlm.json"


def images_in(folder: str):
    return sorted((BASE / folder).glob("*.jpg"))


def rel_of(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def representative(folder: str) -> str:
    return rel_of(max(images_in(folder), key=lambda p: p.stat().st_size))


def build_pairs():
    """Recreate the exact within/cross pair protocol of compare_vehicles.py."""
    folders = sorted(p.name for p in BASE.iterdir()
                     if p.is_dir() and p.name.startswith("vehicle_"))
    reps = {f: representative(f) for f in folders}

    within = {}  # folder -> [(rep, other, cam_keys)]
    for f in folders:
        others = [rel_of(p) for p in images_in(f) if rel_of(p) != reps[f]]
        if others:
            within[f] = [(reps[f], o) for o in others]

    cross = []  # (a, b) pairs, dedup unordered
    seen = set()
    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):
            key = tuple(sorted((folders[i], folders[j])))
            if key in seen:
                continue
            seen.add(key)
            a = reps[key[0]]
            b = reps[key[1]]
            cross.append((a, b))

    explicit = [
        ("vehicle_6/t0038_f00726.jpg", "vehicle_5/t0008_f00187.jpg"),
    ]
    return folders, reps, within, cross, explicit


def embed_pairs(pairs, model_name):
    """Embed every distinct referenced crop once, return dict key -> vec."""
    keys = sorted({k for pair in pairs for k in pair})
    handle = get_model(model_name)
    imgs = []
    good = []
    for k in keys:
        im = cv2.imread(str(BASE / k))
        if im is None or im.size == 0 or min(im.shape[:2]) < 8:
            continue
        imgs.append(im)
        good.append(k)
    feats = handle.embed_batch(imgs)
    return dict(zip(good, feats))


def evaluate(embs, within, cross, explicit):
    """Threshold sweep on the pair protocol; returns the result dict."""
    pos_pairs = []  # (sim, expected_same)
    for f, pairs in within.items():
        for a, b in pairs:
            if a in embs and b in embs:
                pos_pairs.append((float(embs[a] @ embs[b]), True))
    neg_pairs = []
    for a, b in cross:
        if a in embs and b in embs:
            neg_pairs.append((float(embs[a] @ embs[b]), False))

    pos = np.array([s for s, _ in pos_pairs])
    neg = np.array([s for s, _ in neg_pairs])

    thresholds = sorted(set(np.round(np.linspace(0.0, 1.0, 201), 3)))
    rows = []
    for t in thresholds:
        tp = int((pos >= t).sum())
        fn = int((pos < t).sum())
        tn = int((neg < t).sum())
        fp = int((neg >= t).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({
            "threshold": t, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
            "f1": round(f1, 4),
        })
    best = max(rows, key=lambda r: r["f1"])

    # rank-1 cross retrieval: for each within-positive rep pair, does the true
    # partner rank above all negative candidates of the same camera pair?
    rank_hits = 0
    rank_total = 0
    for f, pairs in within.items():
        for a, b in pairs:
            if a not in embs or b not in embs:
                continue
            neg_sims = [s for s, _ in neg_pairs]
            sim_a = embs[a] @ embs[b]
            # rank of b among b + negatives for query a
            cands = [sim_a] + [s for s in neg_sims]
            rank_total += 1
            if int(np.sum(np.array(cands) > sim_a)) == 0:
                rank_hits += 1

    return {
        "n_positive_pairs": len(pos_pairs),
        "n_negative_pairs": len(neg_pairs),
        "pos_sim_mean": round(float(pos.mean()), 4) if len(pos) else None,
        "pos_sim_std": round(float(pos.std()), 4) if len(pos) else None,
        "neg_sim_mean": round(float(neg.mean()), 4) if len(neg) else None,
        "neg_sim_std": round(float(neg.std()), 4) if len(neg) else None,
        "separation_d": round(float((pos.mean() - neg.mean()) /
                                   max(pos.std() + neg.std(), 1e-9)), 4)
        if len(pos) and len(neg) else None,
        "best_f1_point": best,
        "accuracy_at_0_55": next((r for r in rows if abs(r["threshold"] - 0.55) < 1e-9), None),
        "rank1_cross_retrieval": {
            "hits": rank_hits, "total": rank_total,
            "rate": round(rank_hits / max(rank_total, 1), 4),
        },
        "explicit_pair": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--json-out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    models = args.models or ["hsv-hist", "fastreid-veri-sbs-r50-ibn",
                             "resnet50-imagenet"]
    for m in models:
        if m not in REGISTRY:
            print(f"unknown model {m!r}; available: {sorted(REGISTRY)}")

    folders, reps, within, cross, explicit = build_pairs()
    print("Representatives:", json.dumps(reps, indent=2))

    # also embed the explicitly requested pair's files
    all_pair_keys = [k for lst in within.values() for k in lst] + list(cross) + list(explicit)

    results = {}
    for name in models:
        print(f"\n=== {name} ===")
        try:
            import time
            t0 = time.time()
            embs = embed_pairs(all_pair_keys, name)
            print(f"  embedded {len(embs)} crops in {time.time()-t0:.1f}s")
            res = evaluate(embs, within, cross, explicit)
            results[name] = res
            best = res["best_f1_point"]
            print(f"  pos={res['n_positive_pairs']} neg={res['n_negative_pairs']} "
                  f"sep_d={res['separation_d']}")
            print(f"  best F1: thr={best['threshold']:.3f} acc={best['accuracy']:.3f} "
                  f"P={best['precision']:.3f} R={best['recall']:.3f}")
            at55 = res["accuracy_at_0_55"]
            if at55:
                print(f"  @0.55:   acc={at55['accuracy']:.3f} P={at55['precision']:.3f} "
                      f"R={at55['recall']:.3f} FPR={at55['fp']}/{res['n_negative_pairs']}")
            r1 = res["rank1_cross_retrieval"]
            print(f"  rank1 cross retrieval: {r1['hits']}/{r1['total']}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ERROR: {e}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()