"""Record this machine's baseline: N runs of the same clean commit, the median of each number (task #228).

    python3 -m regress baseline [--runs 3] [--allow-dirty] [--allow-noisy] [-- <regress run options>]

Refuses to record from a dirty tree (a baseline must name a commit), from a run that was not idle, or from a run that
fails correctness: a baseline of broken code would make the broken behaviour the reference. The result is written to
benchmarks/baselines/<machine_id>/baseline.json together with the golden replays' outputs (golden/<config>.jsonl), which
is what the compare command diffs a new run against.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from . import metrics, noise


def build(results: list) -> tuple:
    """(baseline document, {config: path of the golden output to keep}) from N results files of one machine and commit."""
    first = results[0]
    flats = [metrics.flatten(r) for r in results]
    # a metric counts when at least two runs have it (one run when there is only one): a metric added to the harness later is recorded by the
    # runs made after it, and demanding it of every earlier run would drop it. The median is over the runs that have it.
    need = min(2, len(flats))
    common = {mid for mid in set().union(*[set(f) for f in flats]) if sum(1 for f in flats if mid in f) >= need}
    doc_metrics = {}
    for mid in sorted(common):
        vals = [f[mid].value for f in flats if mid in f]
        m = next(f[mid] for f in flats if mid in f)
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        doc_metrics[mid] = {"value": round(statistics.median(vals), 4) if numeric else vals[0], "kind": m.kind,
                            "direction": m.direction, "unit": m.unit}
    golden, keep = {}, {}
    an = noise.analyze(results)
    for cfg, g in an["golden"].items():
        modal = max(g["statuses"], key=g["statuses"].get)
        src = next(r["correctness"]["golden"][cfg] for r in results
                   if (r.get("correctness", {}).get("golden", {}).get(cfg, {}).get("vs_repo_baseline") or {}).get("status", "unknown") == modal)
        deterministic = g["distinct"] == 1
        if modal == "no baseline":
            # no repo baseline to give each run a status: the raw outputs are all there is. Found with PPE, fall and shelf (2026-11-18):
            # byte-different between identical runs (other frames get analysed, alert ids are random), so the output is not an oracle.
            shas = {r["correctness"]["golden"][cfg].get("output_sha256") for r in results
                    if r.get("correctness", {}).get("golden", {}).get(cfg, {}).get("ran")}
            deterministic = len(shas) == 1
        golden[cfg] = {"sha256": src["output_sha256"], "output": f"golden/{cfg}.jsonl", "entries": src["entries"],
                       "deterministic": deterministic, "statuses": g["statuses"]}
        keep[cfg] = src["output_file"]
    doc = {"schema": "regress.baseline.v1", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "machine_id": first["machine_id"], "fingerprint": first["fingerprint"], "runs": len(results), "noisy": False,
           "metrics": doc_metrics, "golden": golden}
    return doc, keep


def check(results: list, allow_noisy: bool) -> list:
    """Reasons not to record a baseline from these runs ([] = fine)."""
    reasons = []
    commits = {r["fingerprint"]["git"]["commit"] for r in results}
    if len(commits) > 1:
        reasons.append(f"the runs are from {len(commits)} different commits")
    if any(r["fingerprint"]["git"]["dirty"] for r in results):
        reasons.append("the working tree was dirty (commit first: a baseline must name a commit)")
    if not allow_noisy and any(r.get("noisy") for r in results):
        reasons.append("a run was not on an idle machine (--allow-noisy records it anyway, marked noisy)")
    for i, r in enumerate(results, 1):
        if not r.get("verdict", {}).get("pass", False):
            reasons.append(f"run {i} fails correctness: {'; '.join(r.get('verdict', {}).get('reasons', []))[:160]}")
    return reasons


def write(repo: str, doc: dict, keep: dict) -> str:
    dest = os.path.join(repo, "benchmarks", "baselines", doc["machine_id"])
    os.makedirs(os.path.join(dest, "golden"), exist_ok=True)
    for cfg, src in keep.items():
        shutil.copyfile(src, os.path.join(dest, "golden", f"{cfg}.jsonl"))
    path = os.path.join(dest, "baseline.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


def record(repo: str, runs: int, run_args: list, allow_dirty: bool, allow_noisy: bool) -> int:
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty and not allow_dirty:
        print("refusing to record a baseline from a dirty tree:\n" + dirty[:400], file=sys.stderr)
        return 2
    tmp = tempfile.mkdtemp(prefix="regress-baseline-")
    results = []
    try:
        for i in range(1, runs + 1):
            out = os.path.join(tmp, f"run-{i}.json")
            print(f"--- baseline run {i} of {runs}", flush=True)
            env = {**os.environ, "PYTHONPATH": os.path.join(repo, "tools")}
            rc = subprocess.run([sys.executable, "-m", "regress", "run", *run_args, "--out", out], cwd=repo, env=env).returncode
            if not os.path.exists(out):
                print(f"run {i} wrote no results (exit {rc}); nothing recorded", file=sys.stderr)
                return 2
            with open(out) as f:
                results.append(json.load(f))
        reasons = check(results, allow_noisy)
        if allow_dirty:
            reasons = [r for r in reasons if "dirty" not in r]
        if reasons:
            print("not recording the baseline:", file=sys.stderr)
            for r in reasons:
                print(f"  - {r}", file=sys.stderr)
            return 2
        doc, keep = build(results)
        doc["noisy"] = any(r.get("noisy") for r in results)
        path = write(repo, doc, keep)
        print(f"baseline written: {path}\n  {len(doc['metrics'])} metrics, golden outputs: {', '.join(sorted(doc['golden'])) or 'none'}")
        flips = [c for c, g in doc["golden"].items() if not g["deterministic"]]
        if flips:
            print(f"  note: {', '.join(flips)} did not behave the same in every run (the most frequent outcome's output is kept; "
                  "compare only WARNs on these)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
