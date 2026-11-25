"""Which check for which change: `regress plan` reads the changed files and prints the cheapest set of checks that can notice what they touched.

    python3 -m regress plan [--base origin/main] [--json] [files ...]

Rules live in benchmarks/tiers.json (first match wins per file; a file no rule matches is treated as a shared change, tier 2, and says so):

  tier 0  no GPU: unit suites and build sizes (docs, frontend, app code that the golden replays do not run)
  tier 1  the golden replay and the 1-camera ceiling of the use case(s) the change belongs to
  tier 2  every golden replay and every ceiling (shared engine, pipeline, models, images)
  tier 3  tier 2 plus a soak smoke of the safety stack gated by `soak-compare` (recording, alert clips, incidents, anything lifecycle- or memory-related)

The highest tier of any changed file wins; the configs of all matching rules are united. Durations are what the laptop reference measured.
"""
import fnmatch
import json
import os
import subprocess

# measured on the RTX 5050 laptop reference (minutes): one golden config ~2.5 min + ~1 min of harness start-up per invocation; a 1-camera ceiling ~3.5 min
GOLDEN_MIN_PER_CONFIG, HARNESS_OVERHEAD_MIN, CEILING_MIN_PER_CONFIG, UNIT_BUILD_MIN, SOAK_SMOKE_MIN = 2.5, 1.0, 3.5, 2.0, 75.0
ALL_CONFIGS = ["attendance", "zones", "fire", "multicam", "barber", "barber-faces", "ppe", "fall_detection", "shelf", "proximity"]
CEILING_CONFIGS = ["attendance", "zones", "fire", "barber", "ppe", "fall_detection", "shelf", "proximity"]       # multicam and barber-faces have no ceiling run


def load_rules(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _matches(path: str, patterns: list) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def classify(paths: list, rules: dict) -> dict:
    """{"tier", "configs", "files": [{path, tier, configs, why}], "notes"} for a list of changed paths."""
    files, notes = [], []
    for path in paths:
        rule = next((r for r in rules["rules"] if _matches(path, r["match"])), None)
        if rule is None:
            files.append({"path": path, "tier": 2, "configs": "all", "why": "no rule matches: treated as a shared change"})
            continue
        files.append({"path": path, "tier": rule["tier"], "configs": rule.get("configs", []), "why": rule.get("why", "")})
        if rule.get("note") and rule["note"] not in notes:
            notes.append(rule["note"])
    tier = max((f["tier"] for f in files), default=0)
    configs, everything = set(), False
    for f in files:
        if f["tier"] == 0:
            continue
        if f["configs"] == "all":
            everything = True
        else:
            configs.update(f["configs"])
    if tier >= 2:
        everything = True
    return {"tier": tier, "configs": ALL_CONFIGS if everything else sorted(configs), "everything": everything, "files": files, "notes": notes}


def commands(plan: dict, out_dir: str = "/tmp/regress-plan") -> list:
    """The exact commands, in order, and what each costs: [{"cmd", "minutes", "what"}]."""
    tier, cfgs = plan["tier"], plan["configs"]
    py = "python3 -m regress"
    steps = [{"cmd": f"{py} run --only unit,build --unit-suites core --out {out_dir}/unit.json && {py} compare {out_dir}/unit.json",
              "minutes": UNIT_BUILD_MIN, "what": "unit suites and build sizes against the baseline"}]
    if tier >= 1 and cfgs:
        n = len(cfgs)
        steps.append({"cmd": f"{py} run --only golden --configs {','.join(cfgs)} --out {out_dir}/golden.json && {py} compare {out_dir}/golden.json",
                      "minutes": round(HARNESS_OVERHEAD_MIN + GOLDEN_MIN_PER_CONFIG * n, 1),
                      "what": f"golden replay of {n} config{'s' if n != 1 else ''}: output, fps, drops, stage timers, resources, GPU counters"})
        ceil = [c for c in cfgs if c in CEILING_CONFIGS]
        if ceil:
            steps.append({"cmd": f"{py} capacity --configs {','.join(ceil)} --cameras 1 --out {out_dir}/cap && {py} compare {out_dir}/cap/capacity-results.json",
                          "minutes": round(CEILING_MIN_PER_CONFIG * len(ceil), 1), "what": f"1-camera un-paced ceiling of {len(ceil)} config{'s' if len(ceil) != 1 else ''}"})
    if tier >= 3:
        steps.append({"cmd": f"{py} soak --project safety --duration 1h --warmup 15m --out {out_dir}/soak && {py} soak-analyze {out_dir}/soak --json {out_dir}/soak-analysis.json "
                             f"&& {py} soak-compare {out_dir}/soak-analysis.json",
                      "minutes": SOAK_SMOKE_MIN, "what": "1 h soak of the running safety stack (see benchmarks/README.md: start it with `just safety-up` and `just safety-demo`), growth against the reference"})
    return steps


def changed_files(base: str, repo: str = ".") -> list:
    """What this branch changed: the diff of the working tree against the merge-base with `base`, so what landed on `base` since the branch
    was cut does not count as ours."""
    try:
        mb = subprocess.run(["git", "-C", repo, "merge-base", base, "HEAD"], capture_output=True, text=True).stdout.strip() or base
        r = subprocess.run(["git", "-C", repo, "diff", "--name-only", mb], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("git is not installed here: pass the changed files as arguments (regress plan path1 path2 ...)")
    if r.returncode != 0:
        raise SystemExit(f"git diff against {base} failed: {r.stderr.strip()[:200]}")
    return [l for l in r.stdout.splitlines() if l.strip()]


def render(plan: dict, steps: list) -> str:
    n = len(plan["files"])
    lines = [f"{n} changed file{'s' if n != 1 else ''} -> tier {plan['tier']}"
             + (f", configs: {', '.join(plan['configs'])}" if plan["tier"] >= 1 and plan["configs"] else "")]
    shown = sorted(plan["files"], key=lambda f: -f["tier"])
    for f in shown[:12]:
        lines.append(f"  tier {f['tier']}  {f['path']}" + (f"   ({f['why']})" if f["why"] else ""))
    if len(shown) > 12:
        lines.append(f"  ... and {len(shown) - 12} more")
    total = sum(s["minutes"] for s in steps)
    lines.append("")
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. ~{s['minutes']:g} min  {s['what']}\n   {s['cmd']}")
    lines.append(f"\ntotal about {total:g} minutes" + (" (the soak needs the safety stack up first)" if plan["tier"] >= 3 else ""))
    for note in plan["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)
