"""The results file: one JSON document per run, the same shape on every machine (stdlib only).

    {schema, created, machine_id, fingerprint, guardrails, noisy,
     correctness {unit {suite: ...}, golden {config: ...}},
     runtime     (per golden config: fps, GPU % per stage, drops, boot time, engine rebuilt, GPU memory),
     build       {images, frontend_bundle, lock_packages},
     verdict     {pass, reasons, known_failures}}

`machine_id`, `fingerprint` and `verdict{pass, reasons}` are the fields issue #140 asks of every result. The verdict
here is about CORRECTNESS only (tests pass, the replays ran); whether a number got worse is decided against a stored
baseline by the compare command (task #230), with tolerances measured from noise (task #228).
"""
import json
import os
from datetime import datetime, timezone

SCHEMA = "regress.v1"
KNOWN_FAILURES = os.path.join("benchmarks", "known_failures.txt")


def load_known_failures(repo: str) -> dict:
    """{test id: reason} from benchmarks/known_failures.txt (`classname::name  # why`). A failure listed there does not
    fail the verdict, but it is printed on every run so it is not forgotten."""
    path = os.path.join(repo, KNOWN_FAILURES)
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                tid, _, why = line.partition("#")
                out[tid.strip()] = why.strip()
    return out


def new_result(fingerprint: dict, guardrails: dict) -> dict:
    return {"schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "machine_id": fingerprint["machine_id"], "fingerprint": fingerprint, "guardrails": guardrails,
            "noisy": not guardrails["idle"], "correctness": {}, "runtime": {}, "build": {}, "verdict": {}}


def finish(result: dict, known: dict) -> dict:
    """Fill `runtime` from the golden section and compute the correctness verdict."""
    reasons, known_hit = [], []
    for suite, s in result["correctness"].get("unit", {}).items():
        if not s.get("ran"):
            reasons.append(f"unit suite {suite} did not run: {s.get('error', '?')[:120]}")
            continue
        unexpected = [t for t in s["failed"] if t not in known]
        known_hit += [t for t in s["failed"] if t in known]
        if unexpected:
            reasons.append(f"unit suite {suite}: {len(unexpected)} failing test(s), e.g. {unexpected[0]}")
    for name, g in result["correctness"].get("golden", {}).items():
        if not g.get("ran"):
            reasons.append(f"golden {name} did not run: {g.get('error', '?')[:120]}")
        elif not g.get("entries"):
            reasons.append(f"golden {name} recorded nothing")
        else:
            result["runtime"][name] = {**g["runtime"], "wall_s": g["wall_s"], "gpu_memory": g["gpu_memory"],
                                       "entries": g["entries"], "gpu_counters": g.get("gpu_counters")}
    result["verdict"] = {"pass": not reasons, "reasons": reasons, "known_failures": sorted(set(known_hit))}
    return result


def write(result: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def _mb(n):
    if n is None:
        return "n/a"
    return f"{n / 1e6:,.1f} MB" if n < 1e8 else f"{n / 1e6:,.0f} MB"


def render(r: dict) -> str:
    fp = r["fingerprint"]
    out = [f"{r['machine_id']}  {fp['gpu']}  driver {fp.get('driver', '?')}  commit {fp['git']['short']}"
           f"{' (dirty tree)' if fp['git']['dirty'] else ''}  {r['created']}"]
    out.append(f"idle machine: {'yes' if not r['noisy'] else 'NO -> NOISY, efficiency numbers are advisory'}")
    for reason in r["guardrails"].get("reasons", []):
        out.append(f"   - {reason}")
    unit = r["correctness"].get("unit", {})
    if unit:
        out.append("\nunit suites")
        for name, s in unit.items():
            if s.get("ran"):
                out.append(f"  {name:<8} {s['tests']:>5} tests  {s['failures'] + s['errors']:>3} failing  {s['skipped']:>3} skipped  "
                           f"{s['wall_s']:>6.1f} s")
            else:
                out.append(f"  {name:<8} did not run: {s.get('error', '')[:90]}")
    golden = r["correctness"].get("golden", {})
    if golden:
        out.append("\ngolden replays (vs the baseline recorded in the repo; informational off the reference GPU)")
        out.append(f"  {'config':<14}{'entries':>8}  {'stages Σ%':>10}  {'boot s':>7}  {'fps p10':>8}  {'drops':>6}  {'engine':>14}  "
                   f"{'+GPU MiB':>9}  vs repo baseline")
        for name, g in golden.items():
            if not g.get("ran"):
                out.append(f"  {name:<14} did not run: {g.get('error', '')[:80]}")
                continue
            rt = g["runtime"]
            p10 = min((c["p10"] for c in rt["fps"].values()), default=None)
            drops = int(sum(rt["drops"].values()))
            eng = rt["engine"]
            engine = ("rebuilt " + ",".join(eng["rebuilt_elements"])) if eng["rebuilt"] else ("cached" if eng["cached"] else "?")
            out.append(f"  {name:<14}{g['entries']:>8}  {rt['gpu_total_busy']:>10.1f}  {str(rt['boot_to_first_fps_s']):>7}  "
                       f"{str(p10):>8}  {drops:>6}  {engine:>14}  {str(g['gpu_memory']['added_mib']):>9}  {g['vs_repo_baseline']['status']}")
    if golden:
        out.append("  stages Σ% = the sidecar's GPU stage timers added up; they overlap, so it can pass 100: a size of the work, "
                   "not a utilisation")
    if any(g.get("gpu_counters") for g in golden.values()):
        out.append("\nGPU hardware counters while the pipeline ran (median; NVML = driver, DCGM = profiling; '-' = not available)")
        out.append(f"  {'config':<14}{'GPU time%':>10}{'SM act%':>9}{'SM occ%':>9}{'tensor%':>9}{'DRAM%':>7}{'NVDEC%':>8}{'NVJPG%':>8}"
                   f"{'SM MHz':>8}{'throttled':>10}")
        f = lambda d, k, scale=1.0: (f"{d[k]['median'] * scale:.0f}" if d.get(k) else "-")        # noqa: E731
        for name, g in golden.items():
            c = g.get("gpu_counters")
            if not c:
                continue
            n, d = c["nvml"], c["dcgm"]
            if not c.get("dcgm_valid", True):
                out.append(f"  {name:<14} DCGM SM counters INVALID: {c['dcgm_invalid_reason'][:110]}")
            thr = f"{n['throttled_fraction'] * 100:.0f}%" if "throttled_fraction" in n else "-"
            out.append(f"  {name:<14}{f(n, 'gpu_util'):>10}{f(d, 'sm_active', 100):>9}{f(d, 'sm_occupancy', 100):>9}"
                       f"{f(d, 'tensor_active', 100):>9}{f(d, 'dram_active', 100):>7}{f(n, 'nvdec_util'):>8}{f(n, 'nvjpg_util'):>8}"
                       f"{f(n, 'sm_clock_mhz'):>8}{thr:>10}")
        for name, g in golden.items():
            if (g.get("gpu_counters") or {}).get("retried"):
                out.append(f"  note {name}: its first attempt had invalid DCGM counters, so it was run again on a fresh exporter")
            for h in (g.get("gpu_counters") or {}).get("hints", []):
                out.append(f"  hint {name}: {h}")
        status = next((g["gpu_counters"]["dcgm_status"] for g in golden.values() if g.get("gpu_counters")), {})
        if not status.get("supported"):
            out.append(f"  DCGM profiling counters were not available: {status.get('reason', '')}")
    b = r["build"]
    if b:
        out.append("\nbuild and size")
        for img, size in b.get("images", {}).items():
            out.append(f"  image {img:<44} {_mb(size)}")
        fb = b.get("frontend_bundle") or {}
        if "total_bytes" in fb:
            out.append(f"  frontend bundle {_mb(fb['total_bytes'])} ({fb['files']} files; largest {fb['largest_chunk']['file']} "
                       f"{_mb(fb['largest_chunk']['bytes'])})")
        elif fb:
            out.append(f"  frontend bundle: {fb.get('skipped') or fb.get('error')}")
        for lock, n in b.get("lock_packages", {}).items():
            out.append(f"  {lock:<34} {n} packages")
    v = r["verdict"]
    out.append(f"\ncorrectness verdict: {'PASS' if v['pass'] else 'FAIL'}")
    for reason in v["reasons"]:
        out.append(f"   - {reason}")
    for tid in v["known_failures"]:
        out.append(f"   (known failure, listed in {KNOWN_FAILURES}: {tid})")
    return "\n".join(out)
