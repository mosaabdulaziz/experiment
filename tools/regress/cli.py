"""`just regress`: measure this checkout on this machine and write a results file."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from . import backfill, baseline, capacity, collectors, compare, gpucounters, hostsense, machine, noise, parsers, plan, results, soak, soak_analyze, tolerances

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_SLIM = os.environ.get("REGRESS_SLIM_IMAGE", "iris-deepstream-sidecar:latest")
DEFAULT_MTMC = os.environ.get("REGRESS_MTMC_IMAGE", "iris-deepstream-sidecar-mtmc:latest")
SECTIONS = ("unit", "build", "golden")


def cmd_run(a) -> int:
    only = [s for s in a.only.split(",") if s] if a.only else list(SECTIONS)
    bad = [s for s in only if s not in SECTIONS]
    if bad:
        sys.exit(f"--only takes {', '.join(SECTIONS)}; got {', '.join(bad)}")
    fp = machine.fingerprint(REPO)
    counters = None
    try:
        # only a run that replays on the GPU needs counters and an idle GPU; a CPU-only run (CI) has nothing to check
        if "golden" in only:
            print(f"GPU counters: NVML{'' if a.no_dcgm else ' + DCGM profiling'} (the DCGM exporter takes up to 45 s to start) ...", flush=True)
            counters = gpucounters.GpuCounters(REPO, use_dcgm=not a.no_dcgm)
            st = counters.dcgm_status
            print(f"  DCGM profiling: {'available' if st.get('supported') else 'not available: ' + st.get('reason', '')}", flush=True)
            bg = counters.background()
            guard = machine.idle_report(own_prefix=collectors.PROJECT, background=bg["pct"], source=bg["source"])
            guard["gpu"]["nvidia_smi_util_pct"] = bg["nvidia_smi_util_pct"]
            if not guard["idle"] and not a.allow_noisy:
                print("this machine is not idle, so the efficiency numbers would not be comparable:", file=sys.stderr)
                for reason in guard["reasons"]:
                    print(f"  - {reason}", file=sys.stderr)
                print("stop what is running, or pass --allow-noisy to record the run marked NOISY (efficiency advisory)", file=sys.stderr)
                return 2
        else:
            guard = {"idle": True, "reasons": [], "gpu": {}, "note": "not checked: this run has no GPU section"}
        res = results.new_result(fp, guard)
        print(f"{fp['machine_id']} ({fp['gpu']}), commit {fp['git']['short']}{' dirty' if fp['git']['dirty'] else ''}; "
              f"{'idle' if guard['idle'] else 'NOT idle (noisy)'}; running: {', '.join(only)}", flush=True)
        if "unit" in only:
            print("unit suites ...", flush=True)
            junit = dict(item.split("=", 1) for item in a.unit_junit)
            res["correctness"]["unit"] = collectors.collect_unit(REPO, tuple(a.unit_suites.split(",")), junit)
        if "build" in only:
            print("build and size ...", flush=True)
            res["build"] = collectors.collect_build(REPO, [a.slim_image, a.mtmc_image, a.app_image],
                                                    build_frontend=not a.no_frontend_build)
        if "golden" in only:
            if a.configs:
                names = a.configs.split(",")
            else:
                names, skipped = collectors.default_golden_configs(REPO)
                for cfg, why in skipped.items():
                    print(f"  golden {cfg}: left out, {why}", flush=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = os.path.join(REPO, "benchmarks", "runs", fp["machine_id"], f"{stamp}-{fp['git']['short']}-golden")
            print(f"golden replays: {', '.join(names)} (about 2 to 4 minutes each) ...", flush=True)
            res["correctness"]["golden"] = collectors.collect_golden(
                REPO, names, out_dir, a.slim_image, a.mtmc_image, parsers.load_gpu_budget_parse(REPO), counters)
            res["dcgm_status"] = counters.dcgm_status
    finally:
        if counters is not None:
            counters.stop()
    results.finish(res, results.load_known_failures(REPO))
    path = a.out or os.path.join(REPO, "benchmarks", "runs", fp["machine_id"],
                                 f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{fp['git']['short']}.json")
    results.write(res, path)
    print("\n" + results.render(res) + f"\n\nwritten: {path}")
    return 0 if res["verdict"]["pass"] else 1


def cmd_gpu_probe() -> int:
    """Sample for a few seconds and say which counter layers answered: the thing to run when a machine is onboarded."""
    import time
    fp = machine.fingerprint(REPO)
    print(f"{fp['machine_id']}  {fp['gpu']}  driver {fp.get('driver')}")
    gc = gpucounters.GpuCounters(REPO, use_dcgm=True)
    t0 = time.time()
    time.sleep(6)
    w = gc.window(t0, time.time())
    gc.stop()
    for layer in ("nvml", "jetson"):
        if w[layer]:
            print(f"\n{layer.upper()} counters ({w['samples'][layer]} samples):")
            for k, v in w[layer].items():
                print(f"  {k:<18} " + (f"median {v['median']}  max {v['max']}" if isinstance(v, dict) else str(v)))
    st = w["dcgm_status"]
    print(f"\nDCGM profiling counters: {'AVAILABLE' if st.get('supported') else 'NOT available'}  {st.get('reason', '')}")
    if st.get("supported"):
        print("  " + ", ".join(st["counters"]))
        for k, v in w["dcgm"].items():
            print(f"  {k:<18} median {v['median']}  max {v['max']}")
    return 0 if w["nvml"] or w["jetson"] else 1


def cmd_noise(a) -> int:
    runs = [json.load(open(f)) for f in a.files]
    ids = {r.get("machine_id") for r in runs}
    commits = {r.get("fingerprint", {}).get("git", {}).get("commit") for r in runs}
    if len(runs) < 2:
        sys.exit("noise needs at least 2 results files (5 or more is what a tolerance should rest on)")
    if len(ids) > 1 or len(commits) > 1:
        print(f"warning: these runs are from {len(ids)} machine(s) and {len(commits)} commit(s); noise is only meaningful "
              "for identical runs", file=sys.stderr)
    an = noise.analyze(runs)
    tol = noise.suggest(an)
    print(noise.render(an, tol))
    if a.tolerances:
        with open(a.tolerances, "w") as f:
            json.dump({**tol, "note": f"proposed by `regress noise` from {len(runs)} identical runs on {sorted(ids)}",
                       "runs": len(runs)}, f, indent=1)
            f.write("\n")
        print(f"\ntolerances written: {a.tolerances}")
    return 0


def cmd_baseline(a) -> int:
    return baseline.record(REPO, a.runs, a.run_args, a.allow_dirty, a.allow_noisy)


def cmd_compare(a) -> int:
    new = json.load(open(a.file))
    path = a.baseline or os.path.join(REPO, "benchmarks", "baselines", new["machine_id"], "baseline.json")
    if not os.path.exists(path):
        print(f"no baseline for {new['machine_id']} at {path}\n(record one with `just regress-baseline`, or pass --baseline "
              "<results file> to compare with a single earlier run)", file=sys.stderr)
        return 2
    base = compare.load_baseline(path)
    per_machine = os.path.join(REPO, "benchmarks", "baselines", new["machine_id"], "tolerances.json")
    tol = tolerances.load(a.tolerances or (per_machine if os.path.exists(per_machine)
                                           else os.path.join(REPO, "benchmarks", "tolerances.json")))

    def golden_diff(cfg, before, after):
        d = collectors.diff_outputs(REPO, cfg, before, after)
        return bool(d.get("ok")), d["status"]

    rep = compare.compare(new, base, tol, golden_diff=golden_diff, strict=a.strict, outputs_root=a.repo or REPO)
    print(f"{new['machine_id']}: {os.path.relpath(a.file)} against {os.path.relpath(path)}"
          f"{' (tolerances: ' + os.path.relpath(tol['source']) + ')' if tol['source'] else ' (default tolerances: no benchmarks/tolerances.json)'}\n")
    print(compare.render(rep, a.all))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return rep["exit"]


def _pairs(items: list, what: str) -> dict:
    out = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            sys.exit(f"{what} takes label=path, got {item!r}")
        out[label] = path
    return out


def cmd_soak(a) -> int:
    fp = machine.fingerprint(REPO)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = a.out or os.path.join(REPO, "benchmarks", "runs", fp["machine_id"], f"{stamp}-soak")
    cfg = soak.SoakConfig(project=a.project, interval=a.interval, redis_container=a.redis_container or f"{a.project}-redis-1",
                          api=a.api, watch=_pairs(a.watch, "--watch"), pidfiles=_pairs(a.pidfile, "--pidfile"),
                          du_every=a.du_every)
    print(f"soak: project {cfg.project}, every {cfg.interval:.0f} s for {a.duration}, writing {out_dir}\n"
          f"  GPU counters: {'off' if a.no_gpu else ('NVML' if a.no_dcgm else 'NVML + DCGM')}; Ctrl-C stops and keeps what was sampled",
          flush=True)
    path = soak.run(cfg, out_dir, soak.parse_duration(a.duration), REPO, use_gpu=not a.no_gpu, use_dcgm=not a.no_dcgm)
    print(f"samples: {path}")
    return soak_analyze.main_analyze(out_dir, a.warmup)


def cmd_soak_analyze(a) -> int:
    return soak_analyze.main_analyze(a.path, a.warmup, a.json, a.top)


def cmd_hostsense(a) -> int:
    projects = [p for p in a.projects.split(",") if p]
    print(f"hostsense: every {a.interval:.0f} s for {a.duration}, containers of {projects or 'all projects'}, writing {a.out}", flush=True)
    path = hostsense.run(projects, a.out, soak.parse_duration(a.duration), a.interval, logs=not a.no_logs)
    print(f"samples: {path}")
    return 0


def cmd_hostsense_summary(a) -> int:
    s = hostsense.summarize(hostsense.load(a.path), a.t0, a.t1)
    print(hostsense.render(s))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1)
    return 0


def cmd_soak_compare(a) -> int:
    new = json.load(open(a.new))
    ref_path = a.reference or os.path.join(REPO, "benchmarks", "baselines", machine.fingerprint(REPO)["machine_id"], "soak-reference.json")
    if not os.path.exists(ref_path):
        print(f"no reference soak at {ref_path} (pass --reference <soak-analysis.json>)", file=sys.stderr)
        return 2
    rep = soak_analyze.compare_soak(new, json.load(open(ref_path)), strict=a.strict)
    print(f"{os.path.relpath(a.new)} against {os.path.relpath(ref_path)} (reference: one 3 h soak; factors calibrated on two identical soaks)\n")
    print(soak_analyze.render_compare(rep))
    return rep["exit"]


def cmd_plan(a) -> int:
    rules = plan.load_rules(a.rules or os.path.join(REPO, "benchmarks", "tiers.json"))
    files = a.files or plan.changed_files(a.base, REPO)
    if not files:
        print(f"no changed files against {a.base}: nothing to check")
        return 0
    result = plan.classify(files, rules)
    steps = plan.commands(result, a.out)
    if a.json:
        print(json.dumps({"tier": result["tier"], "configs": result["configs"], "steps": steps, "notes": result["notes"]}, indent=1))
    else:
        print(plan.render(result, steps))
    return 0


def cmd_backfill(a) -> int:
    return backfill.run(a.log, a.runs, a.out)


def cmd_capacity(a) -> int:
    return capacity.main(os.path.abspath(a.repo) if a.repo else REPO, a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="regress", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fingerprint", help="what this machine and checkout are, as JSON")
    sub.add_parser("idle", help="is the GPU free of other work? exit 0 when idle")
    sub.add_parser("gpu-probe", help="which GPU counter layers does this machine support? (about 45 s)")
    sh = sub.add_parser("show", help="print a results file as a table")
    sh.add_argument("file")
    sk = sub.add_parser("soak", help="sample a running stack for hours: CPU, memory (anon vs cache), GPU, disk, Redis, pipeline; then analyse growth")
    sk.add_argument("--project", default="safety", help="compose project whose containers are sampled")
    sk.add_argument("--duration", default="3h")
    sk.add_argument("--interval", type=float, default=30.0, help="seconds between samples")
    sk.add_argument("--warmup", default="15m", help="cut from the analysis (models and engines load early)")
    sk.add_argument("--redis-container", default="", help="default <project>-redis-1")
    sk.add_argument("--api", default="", help="the app's base URL (events and incident counts), e.g. http://localhost:58000")
    sk.add_argument("--watch", action="append", default=[], metavar="LABEL=DIR", help="a directory whose size is tracked (clips, alert_svr)")
    sk.add_argument("--pidfile", action="append", default=[], metavar="LABEL=FILE", help="a host process (fall service, controller) by its pid file")
    sk.add_argument("--du-every", type=int, default=4, help="directory sizes every N samples")
    sk.add_argument("--no-gpu", action="store_true")
    sk.add_argument("--no-dcgm", action="store_true", help="NVML only (no dcgm-exporter container)")
    sk.add_argument("--out", default="")
    pl = sub.add_parser("plan", help="which checks a change needs: changed files -> tier -> configs and the exact commands, with durations")
    pl.add_argument("files", nargs="*", help="changed files (default: git diff --name-only against --base)")
    pl.add_argument("--base", default="origin/main")
    pl.add_argument("--rules", default="", help="default benchmarks/tiers.json")
    pl.add_argument("--out", default="/tmp/regress-plan", help="where the printed commands write their results")
    pl.add_argument("--json", action="store_true")
    sc = sub.add_parser("soak-compare", help="coarse gate: a soak's growth against the reference soak (anon memory, disk, threads, fds, Redis)")
    sc.add_argument("new", help="the new soak's analysis (`regress soak-analyze <dir> --json out.json`)")
    sc.add_argument("--reference", default="", help="default benchmarks/baselines/<machine>/soak-reference.json")
    sc.add_argument("--strict", action="store_true")
    bf = sub.add_parser("backfill", help="add the CPU-side stage timers and flight lines to results files from a sidecar log saved while they were recorded")
    bf.add_argument("--log", required=True)
    bf.add_argument("--out", required=True)
    bf.add_argument("runs", nargs="+")
    cp = sub.add_parser("capacity", help="the machine's limits: un-paced ceiling, real-time cameras per use case, detector sizes, GPU time per model")
    cp.add_argument("--repo", default="", help="the checkout that has the models, clips and engines (default: this one)")
    cp.add_argument("--configs", default="", help="golden configs for ceiling and cameras (comma list, or 'all')")
    cp.add_argument("--cameras", default="1,2,4", help="camera counts to replay in real time (1 is always included)")
    cp.add_argument("--models", default="", help="attendance detector variants (comma list, or 'all'): ceiling and cameras for each")
    cp.add_argument("--engines", action="store_true", help="trtexec GPU time per detector ONNX")
    cp.add_argument("--analysis-fps", type=int, default=0, help="analyse this many frames a second per camera (the story's 5 fps cells); 0 = every frame")
    cp.add_argument("--resume", action="store_true", help="reuse the per-step results files already in --out")
    cp.add_argument("--out", default="")
    hs = sub.add_parser("hostsense", help="record what else the machine was doing: pressure, temperatures, frequency, power, I/O, container events and logs")
    hs.add_argument("--projects", default="regress,safety", help="comma list of compose projects whose containers are tracked (empty = all)")
    hs.add_argument("--duration", default="12h")
    hs.add_argument("--interval", type=float, default=10.0)
    hs.add_argument("--no-logs", action="store_true")
    hs.add_argument("--out", required=True)
    hss = sub.add_parser("hostsense-summary", help="summarise a hostsense recording (optionally a time window, epoch seconds)")
    hss.add_argument("path")
    hss.add_argument("--t0", type=float, default=None)
    hss.add_argument("--t1", type=float, default=None)
    hss.add_argument("--json", default="")
    sa = sub.add_parser("soak-analyze", help="growth per series of a soak run")
    sa.add_argument("path", help="a soak run directory or its samples.jsonl")
    sa.add_argument("--warmup", default="15m")
    sa.add_argument("--json", default="")
    sa.add_argument("--top", type=int, default=60)
    n = sub.add_parser("noise", help="how much do the numbers move between identical runs? proposes tolerances")
    n.add_argument("files", nargs="+")
    n.add_argument("--tolerances", default="", help="write the proposed tolerance table here (benchmarks/tolerances.json)")
    bl = sub.add_parser("baseline", help="record this machine's baseline: N runs, the median of each number")
    bl.add_argument("--runs", type=int, default=3)
    bl.add_argument("--allow-dirty", action="store_true")
    bl.add_argument("--allow-noisy", action="store_true")
    bl.add_argument("run_args", nargs=argparse.REMAINDER, help="options passed to every `run` (after --)")
    c = sub.add_parser("compare", help="PASS / WARN / FAIL of a results file against the baseline")
    c.add_argument("file")
    c.add_argument("--baseline", default="", help="baseline.json or any results file (default benchmarks/baselines/<machine_id>/baseline.json)")
    c.add_argument("--tolerances", default="")
    c.add_argument("--repo", default="", help="the checkout the results file was written in (its golden outputs are relative to it); "
                                              "default: this one")
    c.add_argument("--strict", action="store_true", help="efficiency warnings fail too")
    c.add_argument("--all", action="store_true", help="list passing rows as well")
    c.add_argument("--json", default="", help="also write the comparison here")
    r = sub.add_parser("run", help="measure and write a results file")
    r.add_argument("--only", default="", help=f"comma list of {', '.join(SECTIONS)} (default: all)")
    r.add_argument("--configs", default="", help="golden configs (default: the six the repo records baselines for)")
    r.add_argument("--unit-suites", default="core,barber", help="core = tests/ and test_streamer/; others are products/<name>")
    r.add_argument("--unit-junit", action="append", default=[], metavar="SUITE=FILE",
                   help="read this suite's results from an existing junit XML instead of running it (CI: core=report.xml)")
    r.add_argument("--no-dcgm", action="store_true", help="skip the DCGM profiling counters (no dcgm-exporter container)")
    r.add_argument("--allow-noisy", action="store_true", help="run on a machine that is not idle; the result is marked NOISY")
    r.add_argument("--no-frontend-build", action="store_true", help="measure frontend/dist as it is instead of building it")
    r.add_argument("--slim-image", default=DEFAULT_SLIM)
    r.add_argument("--mtmc-image", default=DEFAULT_MTMC)
    r.add_argument("--app-image", default=os.environ.get("REGRESS_APP_IMAGE", "iris-iris:latest"))
    r.add_argument("--out", default="", help="results file (default benchmarks/runs/<machine_id>/<time>-<commit>.json)")
    a = ap.parse_args(argv)
    if a.cmd == "fingerprint":
        print(json.dumps(machine.fingerprint(REPO), indent=2))
        return 0
    if a.cmd == "idle":
        rep = machine.idle_report(own_prefix=collectors.PROJECT)
        print(json.dumps(rep, indent=2))
        return 0 if rep["idle"] else 1
    if a.cmd == "gpu-probe":
        return cmd_gpu_probe()
    if a.cmd == "show":
        print(results.render(json.load(open(a.file))))
        return 0
    if a.cmd == "soak":
        return cmd_soak(a)
    if a.cmd == "soak-analyze":
        return cmd_soak_analyze(a)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "soak-compare":
        return cmd_soak_compare(a)
    if a.cmd == "backfill":
        return cmd_backfill(a)
    if a.cmd == "capacity":
        return cmd_capacity(a)
    if a.cmd == "hostsense":
        return cmd_hostsense(a)
    if a.cmd == "hostsense-summary":
        return cmd_hostsense_summary(a)
    if a.cmd == "noise":
        return cmd_noise(a)
    if a.cmd == "baseline":
        a.run_args = [x for x in a.run_args if x != "--"]
        return cmd_baseline(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    return cmd_run(a)
