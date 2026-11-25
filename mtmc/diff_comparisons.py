#!/usr/bin/env python3
"""Compare pairwise verdicts across different VLM model runs.

Aligns the within/cross/explicit pairs from multiple
vehicle_comparisons_*.json files and reports, against an optional baseline
model (default: qwen3-vl:4b = mtmc/vehicle_comparisons.json):

  * within-vehicle agreement with ground truth (expected 'same')
  * cross-vehicle discrimination (expected 'different')
  * explicit pair verdict
  * per-pair disagreement table vs the baseline

Usage:
    python -m mtmc.diff_comparisons \
        --baseline mtmc/vehicle_comparisons.json \
        mtmc/vehicle_comparisons_qwen25vl3b.json \
        mtmc/vehicle_comparisons_qwen3vl2b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {
        "explicit": d.get("explicit_pairs", {}),
        "within": d.get("within_vehicle", {}),
        "cross": d.get("cross_vehicle", {}),
    }


def flatten_within(within: dict, rep_key: str):
    """Return dict pair_key -> verdict. pair_key mirrors the cross key format."""
    out = {}
    # recover reps: key is 'vehicle_N/crop.jpg', values are other crop keys
    for folder, others in within.items():
        # find the representative from cross keys (rep vs rep) if possible,
        # else reconstruct from within keys by assuming the rep is the key of
        # the cross section; simpler: use the explicit form here as-is.
        for other, verdict in others.items():
            out[f"{folder} <-> {folder}"] = verdict  # placeholder (not used)
    return out


def flat_explicit(explicit: dict):
    return dict(explicit)


def flat_cross(cross: dict):
    return dict(cross)


def flat_within_with_reps(within: dict, cross: dict):
    """Determine each folder's representative from cross keys, then map within
    pairs to (folder, crop) verdicts."""
    reps = {}
    keys = set()
    for k in cross:
        for part in k.split(" <-> "):
            folder = part.split("/")[0]
            reps.setdefault(folder, part)
    out = {}
    for folder, others in within.items():
        rep = reps.get(folder)
        if rep is None:
            continue
        for other, verdict in others.items():
            other_rel = other if "/" in other else f"{folder}/{other}"
            out[f"{rep} <-> {other_rel}"] = verdict
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="comparison JSONs to report")
    ap.add_argument("--baseline", type=Path,
                    default=Path(__file__).parent / "vehicle_comparisons.json")
    args = ap.parse_args()

    baseline = load(args.baseline)
    models = [load(p) for p in args.files]

    cross_base = flat_cross(baseline["cross"])
    within_base = flat_within_with_reps(baseline["within"], baseline["cross"])
    explicit_base = flat_explicit(baseline["explicit"])

    def summary(name, data, show_errors=True):
        cross = flat_cross(data["cross"])
        within = flat_within_with_reps(data["within"], data["cross"])
        explicit = flat_explicit(data["explicit"])

        within_total = len(within)
        within_same = sum(1 for v in within.values() if v.get("decision") == "same")
        within_diff = sum(1 for v in within.values() if v.get("decision") == "different")
        within_unc = sum(1 for v in within.values() if v.get("decision") == "uncertain")
        within_err = sum(1 for v in within.values() if "error" in v)

        cross_total = len(cross)
        cross_diff = sum(1 for v in cross.values() if v.get("decision") == "different")
        cross_same = sum(1 for v in cross.values() if v.get("decision") == "same")
        cross_unc = sum(1 for v in cross.values() if v.get("decision") == "uncertain")
        cross_err = sum(1 for v in cross.values() if "error" in v)

        print(f"\n=== {name} ===")
        print(f"explicit: {[ (k, v.get('decision')) for k, v in explicit.items() ]}")
        print(f"within:  {within_same} same, {within_diff} different, "
              f"{within_unc} uncertain, {within_err} errors (of {within_total}) "
              f"[expect all same]")
        print(f"cross:   {cross_diff} different, {cross_same} same, "
              f"{cross_unc} uncertain, {cross_err} errors (of {cross_total}) "
              f"[expect all different]")

    def diff_vs_baseline(name, data):
        cross = flat_cross(data["cross"])
        within = flat_within_with_reps(data["within"], data["cross"])
        explicit = flat_explicit(data["explicit"])

        rows = []
        for k, v in within.items():
            b = within_base.get(k)
            if b is None:
                continue
            if v.get("decision") != b.get("decision"):
                rows.append(("within", k,
                             f"4b={b.get('decision')} {name}={v.get('decision')}",
                             v.get("explanation", "")))
        for k, v in cross.items():
            b = cross_base.get(k)
            if b is None:
                continue
            if v.get("decision") != b.get("decision"):
                rows.append(("cross", k,
                             f"4b={b.get('decision')} {name}={v.get('decision')}",
                             v.get("explanation", "")))
        for k, v in explicit.items():
            b = explicit_base.get(k)
            if b is None:
                continue
            if v.get("decision") != b.get("decision"):
                rows.append(("explicit", k,
                             f"4b={b.get('decision')} {name}={v.get('decision')}",
                             v.get("explanation", "")))

        if not rows:
            print(f"  [no disagreements with baseline]")
            return
        print(f"  disagreements w/ baseline ({len(rows)}):")
        for kind, k, verdicts, expl in rows:
            print(f"    {kind:<9} {k}")
            print(f"        {verdicts}")
            if expl:
                print(f"        note: {expl[:110]}")
        # also list baseline errors/uncertain that this model resolved differently
        # (handled above via disagreement). Report only disagreements count.

    for model_name, data in zip([str(p) for p in args.files], models):
        summary(model_name, data)
    print("\n-------------------- DISAGREEMENTS vs BASELINE --------------------")
    for model_name, data in zip([str(p) for p in args.files], models):
        diff_vs_baseline(model_name, data)


if __name__ == "__main__":
    main()