#!/usr/bin/env python3
"""Offline simulation of the two-tier cascade.

Tier 1 (cheap encoder): for each QUERY crop (= representative of a vehicle),
score every candidate crop of the protocol by cosine similarity and keep a
shortlist (top-k per query, or all candidates above a threshold).

Tier 2 (VLM arbiter): the stored qwen3-vl:4b verdict (mtmc/vehicle_comparisons
.    .json) is "consulted" only for shortlisted pairs. Pairs NOT shortlisted are
    auto-classified 'different' WITHOUT any VLM call.

Why offline: we already have the 4b verdict for every protocol pair, so the
cascade cost/accuracy can be swept over (encoder, k, threshold) instantly.

Metrics per configuration:
  * VLM calls vs the 52-call all-pairs baseline (31 within + 21 cross)
  * within accuracy  (same-vehicle pairs judged 'same'  -> expect all 'same')
  * cross accuracy   (diff-vehicle pairs judged 'different' -> expect all 'different')
    note: pre-screen auto-'different' on unshortlisted cross pairs is a correct
    (and free) verdict, and can even FIX the baseline's v7<->v9 false-same.
  * pre-screen recall: fraction of the 29 baseline-correct within pairs that
    survive into the shortlist (upper bound is the baseline's own 29/31).

Usage:
    python -m mtmc.cascade_matching --models dinov2-base siglip-base siglip2-base
    python -m mtmc.cascade_matching --topk-max 8 --json-out output/cascade_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .reid_models import get_model

BASE = Path(__file__).parent / "output" / "vehicle_crops" / "chosen crops"
COMPARISONS = Path(__file__).parent / "vehicle_comparisons.json"
OUT_DEFAULT = Path(__file__).parent / "output" / "cascade_sweep.json"


def images_in(folder: str):
    return sorted((BASE / folder).glob("*.jpg"))


def rel_of(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def representative(folder: str) -> str:
    return rel_of(max(images_in(folder), key=lambda p: p.stat().st_size))


def build_protocol():
    """Recreate the exact within/cross pair protocol of compare_vehicles.py."""
    folders = sorted(p.name for p in BASE.iterdir()
                     if p.is_dir() and p.name.startswith("vehicle_"))
    reps = {f: representative(f) for f in folders}

    within = {}  # folder -> [(rep, other)]
    for f in folders:
        others = [rel_of(p) for p in images_in(f) if rel_of(p) != reps[f]]
        if others:
            within[f] = [(reps[f], o) for o in others]

    all_files = sorted(rel_of(p) for p in BASE.glob("vehicle_*/*.jpg"))
    return folders, reps, within, all_files


def load_4b_verdicts(reps: dict, within: dict):
    """Map every protocol pair -> stored 4b ('same'|'different'|'uncertain'),
    and rebuild the EXACT cross pair list the VLM actually ran (21 pairs, not
    the full 36-folder matrix)."""
    d = json.loads(COMPARISONS.read_text())
    verdicts = {}
    for f, pairs in within.items():
        for rep, other in pairs:
            v = d["within_vehicle"][f][other]
            verdicts[(rep, other)] = v.get("decision", "error")
    cross = d["cross_vehicle"]
    cross_pairs = []
    for key, v in cross.items():
        a, b = (p.strip() for p in key.split("<->"))
        verdicts[(a, b)] = v.get("decision", "error")
        cross_pairs.append((a, b))
    return verdicts, cross_pairs


def embed_all_models(models: list[str], all_files: list[str]):
    """Embed every distinct crop with each model -> {model: {key: vec}}."""
    imgs, good = [], []
    for k in all_files:
        im = cv2.imread(str(BASE / k))
        if im is None or im.size == 0 or min(im.shape[:2]) < 8:
            continue
        imgs.append(im)
        good.append(k)
    out = {}
    for name in models:
        handle = get_model(name)
        feats = handle.embed_batch(imgs)
        out[name] = dict(zip(good, feats))
    return out


def rank_shortlist(embs: dict, reps: dict, folders: list[str],
                   within: dict, k: int):
    """Top-k per query. Returns (consulted_within, consulted_cross) pair lists
    plus the shortlist itself, mirroring protocol pairs only."""
    consulted_within, consulted_cross = [], []
    shortlists = {}
    for f, rep in reps.items():
        own_others = [o for _, o in within.get(f, [])]
        other_reps = [reps[g] for g in folders if g != f]
        candidates = own_others + other_reps
        candidates = [c for c in candidates if c in embs]
        sims = [(c, float(embs[rep] @ embs[c])) for c in candidates]
        sims.sort(key=lambda t: t[1], reverse=True)
        shortlists[f] = (rep, sims[:k])
        keep = {c for c, _ in sims[:k]}
        consulted_within += [(rep, c) for c in own_others if c in keep]
        consulted_cross += [(rep, r) for r in other_reps if r in keep]
    return shortlists, consulted_within, consulted_cross


def eval_config(verdicts, within_pairs_all, cross_pairs_all,
                consulted_within, consulted_cross):
    """Compute accuracy & call count given which pairs are consulted."""
    correct_within = 0
    for rep, other in within_pairs_all:
        used = (rep, other) in consulted_within
        if used and verdicts.get((rep, other)) == "same":
            correct_within += 1
        elif not used:
            pass  # auto 'different' on a same-vehicle pair -> wrong
    within_acc = correct_within / max(len(within_pairs_all), 1)

    correct_cross = 0
    for rep, other in cross_pairs_all:
        used = (rep, other) in consulted_cross
        if not used:
            correct_cross += 1                          # free, correct
        elif verdicts.get((rep, other)) == "different":
            correct_cross += 1
    cross_acc = correct_cross / max(len(cross_pairs_all), 1)

    calls = len(consulted_within) + len(consulted_cross)
    return {
        "calls": calls,
        "within_acc": within_acc,
        "cross_acc": cross_acc,
        "consulted_within": len(consulted_within),
        "consulted_cross": len(consulted_cross),
    }


def sweep_topk(embs, verdicts, reps, folders, within, k_max, cross_all):
    within_all = [p for lst in within.values() for p in lst]
    n_within, n_cross = len(within_all), len(cross_all)
    baseline_calls = n_within + n_cross

    rows = []
    for k in range(0, k_max + 1):
        _, cw, cc = rank_shortlist(embs, reps, folders, within, k)
        r = eval_config(verdicts, within_all, cross_all, cw, cc)
        rows.append({
            "topk": k,
            **r,
            "reduction": round((baseline_calls - r["calls"]) / baseline_calls, 4),
        })
    return rows


def threshold_decision(embs, verdicts, reps, within, threshold, cross_all):
    """All pairs with sim >= threshold are consulted (per-pair decision)."""
    within_all = [p for lst in within.values() for p in lst]
    cw = [p for p in within_all if float(embs[p[0]] @ embs[p[1]]) >= threshold]
    cc = [p for p in cross_all if float(embs[p[0]] @ embs[p[1]]) >= threshold]
    return within_all, cross_all, cw, cc


def sweep_threshold(embs, verdicts, reps, within, cross_all, steps=11):
    rows = []
    for t in np.linspace(0.5, 1.0, steps):
        within_all, cross_all_, cw, cc = threshold_decision(
            embs, verdicts, reps, within, float(t), cross_all)
        r = eval_config(verdicts, within_all, cross_all, cw, cc)
        n_total = len(within_all) + len(cross_all)
        rows.append({
            "threshold": round(float(t), 3),
            **r,
            "reduction": round((n_total - r["calls"]) / n_total, 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*",
                    default=["dinov2-base", "siglip-base", "siglip2-base"])
    ap.add_argument("--topk-max", type=int, default=8)
    ap.add_argument("--json-out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    folders, reps, within, all_files = build_protocol()
    verdicts, cross_all = load_4b_verdicts(reps, within)
    within_all = [p for lst in within.values() for p in lst]
    n_within, n_cross = len(within_all), len(cross_all)
    baseline_calls = n_within + n_cross
    print(f"protocol: {n_within} within pairs, {n_cross} cross pairs "
          f"({baseline_calls} all-pairs VLM calls)")

    results = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            embs = embed_all_models([name], all_files)[name]
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  EMBED ERROR: {e}")
            continue
        topk_rows = sweep_topk(embs, verdicts, reps, folders, within,
                               args.topk_max, cross_all)
        thr_rows = sweep_threshold(embs, verdicts, reps, within, cross_all)
        results[name] = {
            "n_within": n_within, "n_cross": n_cross,
            "baseline_calls": baseline_calls,
            "topk_sweep": topk_rows,
            "threshold_sweep": thr_rows,
        }
        print(f"  top-k sweep (within_acc / cross_acc / calls / reduction):")
        best = None
        for r in topk_rows:
            flag = ""
            if r["within_acc"] == 1.0 and best is None:
                best = r
                flag = "  <-- perfect within"
            print(f"    k={r['topk']:<2} within={r['within_acc']:.3f} "
                  f"cross={r['cross_acc']:.3f} calls={r['calls']:<3} "
                  f"({r['reduction']*100:.0f}% saved){flag}")
        print(f"  threshold sweep (within / cross / calls):")
        for r in thr_rows:
            print(f"    t={r['threshold']:.2f} within={r['within_acc']:.3f} "
                  f"cross={r['cross_acc']:.3f} calls={r['calls']}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()