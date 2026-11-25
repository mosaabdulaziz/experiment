"""How much do the numbers move when NOTHING changed? (task #228)

Give it the results files of N identical runs (same commit, same machine, nothing else running) and it reports, per
metric, the median, the range and the spread relative to the median, plus a tolerance that is not tighter than that
spread. The golden outputs are checked for determinism too: one distinct hash across the runs, or several.

    python3 -m regress noise run-1.json run-2.json ... [--md NOISE.md] [--tolerances benchmarks/tolerances.json]
"""
import statistics

from . import metrics

SAFETY = 1.5            # the tolerance is this many times the observed spread: N runs rarely show the worst case
MIN_REL = 0.02          # and never below 2%: no measurement on a shared machine is steadier than that
STEP = 0.01             # tolerances are rounded UP to a whole percent


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def analyze(results: list) -> dict:
    """{"metrics": {id: {n, median, min, max, spread_abs, spread_rel, kind, direction, unit}},
        "golden": {cfg: {"distinct": k, "hashes": {sha: count}}}, "runs": N}"""
    flat = [metrics.flatten(r) for r in results]
    ids = sorted({mid for f in flat for mid in f})
    out = {}
    for mid in ids:
        vals = [f[mid].value for f in flat if mid in f and _numeric(f[mid].value)]
        if len(vals) < 2:
            continue
        m = next(f[mid] for f in flat if mid in f)
        med = statistics.median(vals)
        spread = max(vals) - min(vals)
        out[mid] = {"n": len(vals), "median": round(med, 4), "min": min(vals), "max": max(vals),
                    "spread_abs": round(spread, 4), "spread_rel": round(spread / abs(med), 4) if med else None,
                    "stdev": round(statistics.pstdev(vals), 4), "kind": m.kind, "direction": m.direction, "unit": m.unit}
    golden = {}
    for r in results:
        for cfg, g in (r.get("correctness", {}).get("golden") or {}).items():
            if g.get("ran"):
                # judged by the replay's own diff against the repo baseline, not by the raw file's hash: the raw output
                # carries volatile fields, so its hash differs on every run even when the output is identical
                st = (g.get("vs_repo_baseline") or {}).get("status", "unknown")
                golden.setdefault(cfg, {})
                golden[cfg][st] = golden[cfg].get(st, 0) + 1
    return {"runs": len(results), "metrics": out,
            "golden": {c: {"distinct": len(h), "statuses": h} for c, h in golden.items()}}


# a spread this big, relative AND absolute (a counter that moves 0.002 -> 0.004 is not unstable), means the run did not do
# the same thing each time (fire: the face stage ran in some runs and idled in others)
UNSTABLE_REL = 0.35
SMALL = {"ratio": 0.05, "%": 5.0, "": 5.0, "ms": 3.0, "s": 3.0, "fps": 2.0, "count": 1.0, "MiB": 50.0, "bytes": 50000.0}


def unstable(s: dict) -> bool:
    return s["spread_rel"] is not None and s["spread_rel"] > UNSTABLE_REL and s["spread_abs"] > SMALL.get(s["unit"], 0.0)


def suggest(analysis: dict) -> dict:
    """Tolerance rules. A metric that was unstable between identical runs gets its own rule `{"gate": false}` (listed first:
    the first match wins) and is left out of its family's tolerance; the others get the largest spread seen in any member
    of the FAMILY (runtime.*.fps_p10), so a new config is covered the day it is added."""
    fam: dict = {}
    ungated = []
    # a config whose golden output differs between identical runs did not do the same work each time (fire: one run in five
    # took another path and its stage times moved with it), so none of its efficiency numbers can be judged
    flaky = sorted(c for c, g in analysis["golden"].items() if g["distinct"] > 1)
    for cfg in flaky:
        for head in ("runtime", "counters"):
            ungated.append({"match": f"{head}.{cfg}.*", "gate": False, "why": f"{cfg} is flaky: its golden output differs between identical runs"})
    for mid, s in sorted(analysis["metrics"].items()):
        if s["kind"] != "efficiency" or s["direction"] == "equal":
            continue
        if mid.split(".")[1:2] and mid.split(".")[1] in flaky and mid.split(".")[0] in ("runtime", "counters"):
            continue
        if unstable(s):
            ungated.append({"match": mid, "gate": False,
                            "why": f"{100 * s['spread_rel']:.0f}% apart between identical runs ({s['min']:g} to {s['max']:g})"})
            continue
        f = metrics.family(mid)
        cur = fam.setdefault(f, {"rel": 0.0, "abs": 0.0, "members": 0})
        cur["members"] += 1
        cur["abs"] = max(cur["abs"], s["spread_abs"] * SAFETY)
        if s["spread_rel"] is not None:
            cur["rel"] = max(cur["rel"], s["spread_rel"] * SAFETY)
    rules = []
    for f, c in sorted(fam.items()):
        rel = max(MIN_REL, c["rel"])
        rel = -(-rel // STEP) * STEP                                   # round up to a whole percent
        rules.append({"match": f, "rel": round(rel, 2), "abs": round(c["abs"], 3), "members": c["members"]})
    return {"default": {"rel": 0.10, "abs": 0.0}, "rules": ungated + rules}


def render(analysis: dict, tolerances: dict) -> str:
    lines = [f"noise over {analysis['runs']} identical runs", ""]
    lines.append("golden outputs against the repo baseline, per run (one status = the replay behaves the same every time):")
    for cfg, g in sorted(analysis["golden"].items()):
        st = ", ".join(f"{k} x{v}" for k, v in sorted(g["statuses"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  {cfg:<14} {st}" + ("" if g["distinct"] == 1 else "   <- differs between identical runs: FLAKY"))
    lines += ["", f"{'metric':<52}{'median':>12}{'min':>12}{'max':>12}{'spread%':>9}"]
    for mid, s in sorted(analysis["metrics"].items()):
        if s["spread_abs"] == 0:
            continue
        rel = "-" if s["spread_rel"] is None else f"{100 * s['spread_rel']:.1f}"
        lines.append(f"{mid:<52}{s['median']:>12g}{s['min']:>12g}{s['max']:>12g}{rel:>9}")
    steady = sum(1 for s in analysis["metrics"].values() if s["spread_abs"] == 0)
    lines.append(f"({steady} metrics did not move at all and are not listed)")
    gated = [r for r in tolerances["rules"] if r.get("gate", True)]
    off = [r for r in tolerances["rules"] if not r.get("gate", True)]
    if off:
        lines += ["", f"not gated, unstable between identical runs ({len(off)}):"]
        lines += [f"  {r['match']:<52} {r['why']}" for r in off]
    lines += ["", "proposed tolerances (largest spread in the family x %.1f, at least %d%%):" % (SAFETY, MIN_REL * 100)]
    for r in gated:
        lines.append(f"  {r['match']:<52} rel {r['rel']:.2f}  abs {r['abs']:g}")
    return "\n".join(lines)
