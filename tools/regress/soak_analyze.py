"""Does anything grow? Turn a soak run's samples.jsonl into a table of growth per series.

    python3 -m regress soak-analyze benchmarks/runs/<machine>/<run>-soak [--warmup 15m]

For every numeric series in the rows (a container's anon memory, a process's RSS, fds, threads, Redis memory, a stream's
length, a directory's bytes, GPU memory ...) it fits an ordinary least-squares line over the run after the warm-up (engines and
models load in the first minutes: growth then is not a leak) and compares the first and last third of the run. A series is

    growing   its fitted growth over the run is at least the unit's floor (20 MiB, 10 items, 0.1 core, ...) AND 5% of where it
              started, the line explains the data (R^2 >= 0.5) and the last third really is higher than the first
    noisy     it moved by that much but not along a line (a sawtooth, a cache that fills and empties)
    flat      it did not move by that much

Cumulative counters (CPU time, network and disk bytes) are turned into rates first (cores, MiB/s): a rate that climbs over hours is
a symptom too. Page cache and tmpfs are separate series from anon: only anon (and process RSS) says "leak"; a recording backlog
fills `shmem` and `file` and is a different problem. GPU, pipeline timers and the app's own counts are listed for reading, not
judged (an event counter is supposed to grow).
"""
import json
import os
import statistics

from .soak import parse_duration

GROW_FLOOR = {"MiB": 20.0, "count": 10.0, "cores": 0.1, "MiB/s": 0.5}
GROW_REL = 0.05
MIN_POINTS = 6
CUMULATIVE = {"cpu_usec": ("cores", 1e-6), "cpu_ticks": ("cores", 0.01), "cpu_total_ticks": (None, 1), "cpu_idle_ticks": (None, 1),
              "rx_bytes": ("MiB/s", 1 / 1048576), "tx_bytes": ("MiB/s", 1 / 1048576),
              "read_bytes": ("MiB/s", 1 / 1048576), "write_bytes": ("MiB/s", 1 / 1048576)}
BYTES_LEAVES = {"anon", "file", "shmem", "kernel", "bytes", "used_memory", "used_memory_rss", "anon_plus_swap", "rss_anon_bytes_plus_swap"}
COUNT_LEAVES = {"threads", "fds", "pids", "n", "files", "keys", "connected_clients"}
INFO_ROOTS = ("gpu", "telemetry", "app")          # listed, not judged
# MAXLEN of the streams the engine writes (perception/engine/gst_build.py, modules.py, config.py): a stream filling up to its cap is
# bounded, not growing; trimming is approximate, so a stream is only suspect above CAP_SLACK times its cap.
STREAM_CAPS = {"meta": 64, "alert": 256, "overlay": 1500, "svr_request": 32, "svr_done": 64, "zone_alert": 64, "overcrowd": 64}
CAP_SLACK = 3


def flatten(obj, prefix: str = "") -> dict:
    """Nested dict -> {"a.b.c": number} for every numeric leaf (booleans and strings dropped)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("t", "errors") and not prefix:
                continue
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def unit_of(path: str) -> "tuple[str, float]":
    """(unit, factor to apply to the stored value) of a series, from its name."""
    leaf = path.rsplit(".", 1)[-1]
    if leaf in CUMULATIVE:
        return ("rate", 1.0)
    if leaf in BYTES_LEAVES or leaf.endswith("_bytes"):
        return ("MiB", 1 / 1048576)
    if leaf.endswith("_mib") or path.startswith("gpu_procs_mib"):
        return ("MiB", 1.0)
    if leaf in COUNT_LEAVES or path.startswith("redis.stream_len") or path.startswith("redis.key_prefixes"):
        return ("count", 1.0)
    return ("other", 1.0)


def load_rows(path: str) -> list:
    if os.path.isdir(path):
        path = os.path.join(path, "samples.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass                                   # a half-written last line
    return rows


def derive(flat: dict) -> dict:
    """Series that only exist as a sum. Under memory pressure a process's pages move to swap: its RSS and cgroup anon FALL while
    what it holds does not, so a leak can hide there. `anon_plus_swap` and `rss_plus_swap_bytes` are what it really holds;
    `host.used_plus_swap_bytes` is the machine's total demand."""
    out = dict(flat)
    for path, v in flat.items():
        if path.endswith(".cgroup.anon"):
            out[path + "_plus_swap"] = v + flat.get(path[:-len("anon")] + "swap_bytes", 0.0)
        elif path.endswith(".procs.main.rss_bytes"):
            out[path[:-len("rss_bytes")] + "rss_plus_swap_bytes"] = v + flat.get(path[:-len("rss_bytes")] + "swap_bytes", 0.0)
        elif path.endswith(".procs.main.rss_anon_bytes"):                   # the process's own heap, swap included
            out[path + "_plus_swap"] = v + flat.get(path[:-len("rss_anon_bytes")] + "swap_bytes", 0.0)
    if "host.mem_total_bytes" in flat and "host.mem_available_bytes" in flat:
        out["host.used_plus_swap_bytes"] = (flat["host.mem_total_bytes"] - flat["host.mem_available_bytes"]
                                            + flat.get("host.swap_used_bytes", 0.0))
    return out


def series_of(rows: list) -> dict:
    """{path: ([seconds since the first row], [values])}; a series missing from a row is skipped for that row."""
    if not rows:
        return {}
    t0 = rows[0]["t"]
    out: dict = {}
    for row in rows:
        for path, v in derive(flatten(row)).items():
            ts, vs = out.setdefault(path, ([], []))
            ts.append(row["t"] - t0)
            vs.append(v)
    return out


def to_rate(ts: list, vs: list, scale: float) -> "tuple[list, list]":
    """A cumulative counter -> per-second rate between consecutive rows (a reset, i.e. a negative step, is skipped)."""
    rt, rv = [], []
    for i in range(1, len(vs)):
        dt = ts[i] - ts[i - 1]
        d = vs[i] - vs[i - 1]
        if dt > 0 and d >= 0:
            rt.append(ts[i])
            rv.append(d / dt * scale)
    return rt, rv


def ols(ts: list, vs: list) -> "tuple[float, float]":
    """(slope per second, R^2) of the least-squares line through the points."""
    n = len(ts)
    if n < 2:
        return 0.0, 0.0
    mt, mv = sum(ts) / n, sum(vs) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx == 0:
        return 0.0, 0.0
    sxy = sum((t - mt) * (v - mv) for t, v in zip(ts, vs))
    slope = sxy / sxx
    ss_tot = sum((v - mv) ** 2 for v in vs)
    if ss_tot == 0:
        return slope, 1.0
    ss_res = sum((v - (mv + slope * (t - mt))) ** 2 for t, v in zip(ts, vs))
    return slope, max(0.0, 1.0 - ss_res / ss_tot)


def analyze_series(ts: list, vs: list, unit: str, warmup_s: float, judged: bool = True, cap: "int | None" = None) -> dict:
    """Stats and a verdict for one series (values already in `unit`)."""
    keep = [(t, v) for t, v in zip(ts, vs) if t >= warmup_s]
    if len(keep) < MIN_POINTS:
        keep = list(zip(ts, vs))                           # a run shorter than the warm-up: use it all, say so
    t = [x[0] for x in keep]
    v = [x[1] for x in keep]
    if len(v) < MIN_POINTS:
        return {"n": len(v), "unit": unit, "verdict": "too short"}
    third = max(1, len(v) // 3)
    first, last = statistics.median(v[:third]), statistics.median(v[-third:])
    slope, r2 = ols(t, v)
    span_h = (t[-1] - t[0]) / 3600.0
    growth = slope * 3600.0 * span_h
    res = {"n": len(v), "unit": unit, "start": round(v[0], 3), "end": round(v[-1], 3), "first_third": round(first, 3),
           "last_third": round(last, 3), "slope_per_h": round(slope * 3600.0, 4), "r2": round(r2, 3),
           "growth": round(growth, 3), "span_h": round(span_h, 3), "max": round(max(v), 3), "median": round(statistics.median(v), 3)}
    if not judged or unit not in GROW_FLOOR:
        res["verdict"] = "info"
        return res
    if unit == "count" and cap is not None and max(v) <= cap * CAP_SLACK:
        res["verdict"] = "bounded"                         # a stream with a MAXLEN, filling up to it
        res["cap"] = cap
        return res
    floor = max(GROW_FLOOR[unit], GROW_REL * abs(first))
    moved = growth >= floor and (last - first) >= 0.5 * floor
    res["verdict"] = ("growing" if r2 >= 0.5 else "noisy") if moved else "flat"
    return res


def _stream_cap(path: str):
    """The MAXLEN of a `redis.stream_len.iris:<kind>:<id>` series, if the kind has one."""
    if not path.startswith("redis.stream_len."):
        return None
    parts = path[len("redis.stream_len."):].split(":")
    return STREAM_CAPS.get(parts[1]) if len(parts) > 1 else None


def analyze(rows: list, warmup_s: float = 900.0) -> dict:
    """{series: result} for the whole run plus run facts under the key '_run'."""
    out = {}
    for path, (ts, vs) in series_of(rows).items():
        unit, factor = unit_of(path)
        if unit == "rate":
            leaf = path.rsplit(".", 1)[-1]
            unit2, scale = CUMULATIVE[leaf]
            if unit2 is None:
                continue                                    # host CPU ticks only feed host.cpu_busy_pct below
            ts, vs = to_rate(ts, vs, scale)
            unit = unit2
        else:
            vs = [v * factor for v in vs]
        out[path] = analyze_series(ts, vs, unit, warmup_s, judged=path.split(".", 1)[0] not in INFO_ROOTS, cap=_stream_cap(path))
    S = series_of(rows)
    if "host.cpu_total_ticks" in S and "host.cpu_idle_ticks" in S:                 # the host's busy % from the tick counters
        ts, tot = S["host.cpu_total_ticks"]
        _t, idle = S["host.cpu_idle_ticks"]
        bt, bv = [], []
        for i in range(1, min(len(tot), len(idle))):
            d_tot, d_idle = tot[i] - tot[i - 1], idle[i] - idle[i - 1]
            if d_tot > 0 and d_idle >= 0:
                bt.append(ts[i])
                bv.append(100.0 * (1.0 - d_idle / d_tot))
        out["host.cpu_busy_pct"] = analyze_series(bt, bv, "pct", warmup_s, judged=False)
    if rows:
        out["_run"] = {"rows": len(rows), "hours": round((rows[-1]["t"] - rows[0]["t"]) / 3600.0, 3), "warmup_s": warmup_s}
    return out


def render(results: dict, top: int = 60) -> str:
    run = results.get("_run", {})
    lines = [f"soak analysis: {run.get('rows', 0)} samples over {run.get('hours', 0)} h (warm-up cut: {run.get('warmup_s', 0):.0f} s)", ""]
    judged = {k: v for k, v in results.items() if k != "_run" and v.get("verdict") in ("growing", "noisy", "flat")}
    order = {"growing": 0, "noisy": 1, "flat": 2}
    rows = sorted(judged.items(), key=lambda kv: (order[kv[1]["verdict"]], -abs(kv[1].get("growth", 0))))
    grow = [k for k, v in rows if v["verdict"] == "growing"]
    bounded = [k for k, v in results.items() if k != "_run" and v.get("verdict") == "bounded"]
    lines.append(f"{len(grow)} growing, {sum(1 for _k, v in rows if v['verdict'] == 'noisy')} noisy, "
                 f"{sum(1 for _k, v in rows if v['verdict'] == 'flat')} flat (of {len(rows)} judged series), "
                 f"{len(bounded)} streams filling to their cap")
    lines.append("")
    lines.append(f"{'series':<66}{'unit':>7}{'start':>11}{'end':>11}{'slope/h':>11}{'R2':>6}  verdict")
    shown = [r for r in rows if r[1]["verdict"] != "flat"][:top]
    for k, v in shown:
        lines.append(f"{k:<66}{v['unit']:>7}{v['first_third']:>11.4g}{v['last_third']:>11.4g}{v['slope_per_h']:>11.4g}{v['r2']:>6.2f}  {v['verdict']}")
    if not shown:
        lines.append("(nothing grew)")
    short = [k for k, v in results.items() if k != "_run" and v.get("verdict") == "too short"]
    if short:
        lines += ["", f"{len(short)} series had too few points to judge (run longer than the warm-up)"]
    return "\n".join(lines)


def main_analyze(path: str, warmup: str = "15m", json_out: str = "", top: int = 60) -> int:
    rows = load_rows(path)
    if not rows:
        print(f"no samples in {path}")
        return 2
    results = analyze(rows, parse_duration(warmup))
    print(render(results, top))
    out = json_out or os.path.join(path if os.path.isdir(path) else os.path.dirname(path) or ".", "analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwritten: {out}")
    return 0


# ---------------------------------------------------------------- the coarse gate against a reference soak
# (series pattern, unit, factor, floor): a watched series is a WARN when it grows and the reference did not, or when its slope is above
# `factor` x the reference's and more than `floor` (in the series' unit, per hour) above it. Calibrated on two identical 3 h soaks of the same
# stack on the same laptop (2026-11-17/18): their slopes differed by at most 1.20x for anon memory (7 MiB/h), 1.18x for disk bytes (23 MiB/h) and
# 1.16x for disk files (5 files/h); the factors sit above that. Two soaks are still two: re-check after a machine or stack change.
# Page cache (`cgroup.file`) is not watched: it grows with I/O and is reclaimed.
WATCH = (
    ("*.cgroup.anon", "MiB", 2.0, 25.0), ("*.cgroup.anon_plus_swap", "MiB", 2.0, 25.0),
    ("*.procs.main.rss_anon_bytes", "MiB", 2.0, 25.0), ("*.procs.main.rss_anon_bytes_plus_swap", "MiB", 2.0, 25.0),
    ("disk.*.bytes", "MiB", 1.5, 100.0), ("disk.*.files", "count", 1.5, 25.0),
    ("*.procs.main.threads", "count", 2.0, 2.0), ("*.procs.main.fds", "count", 2.0, 2.0),
    ("redis.used_memory", "MiB", 2.0, 5.0),
)


def _watch_rule(series_id: str):
    import fnmatch
    return next((w for w in WATCH if fnmatch.fnmatchcase(series_id, w[0])), None)


def compare_soak(new: dict, ref: dict, strict: bool = False) -> dict:
    """{"rows": [{id, status, reference, new, note}], "counts": {...}, "verdict", "exit"} of two soak analyses (`regress soak-analyze --json`).
    WARN only; with `strict` a WARN fails."""
    rows = []
    for sid in sorted(set(new) | set(ref)):
        rule = _watch_rule(sid)
        if not rule:
            continue
        _pat, _unit, factor, floor = rule
        n, r = new.get(sid), ref.get(sid)
        if n is None:
            continue                                                    # the series is not in this soak (a role that did not run)
        ns = n.get("slope_per_h") or 0.0
        rs = (r or {}).get("slope_per_h") or 0.0
        n_grows = n.get("verdict") == "growing"
        r_grows = (r or {}).get("verdict") == "growing"
        if n_grows and not r_grows:
            status, note = ("WARN", "grows now, was not growing in the reference") if ns >= floor else ("INFO", f"grows, but under the {floor} {n.get('unit')}/h floor")
        elif n_grows and r_grows:
            if ns > factor * rs and ns - rs > floor:
                status, note = "WARN", f"slope more than {factor}x the reference's (and {ns - rs:.0f} {n.get('unit')}/h more)"
            else:
                status, note = "PASS", "growth as in the reference"
        elif r_grows:
            if n.get("verdict") in ("flat", "bounded"):
                status, note = "IMPROVED", "was growing in the reference, is not now"
            else:                                                       # noisy or too short to judge: not evidence of an improvement
                status, note = "INFO", f"verdict {n.get('verdict')!r} now (growing in the reference): slope {ns:.1f} against {rs:.1f}"
        else:
            status, note = "PASS", ""
        if status != "PASS" or n_grows:
            rows.append({"id": sid, "status": status, "reference": round(rs, 2) if r else None, "new": round(ns, 2), "unit": n.get("unit", "") + "/h", "note": note})
    if strict:
        for row in rows:
            if row["status"] == "WARN":
                row["status"], row["note"] = "FAIL", (row["note"] + " (strict)").strip()
    counts = {s: sum(1 for x in rows if x["status"] == s) for s in ("PASS", "IMPROVED", "INFO", "WARN", "FAIL")}
    verdict = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"
    return {"rows": rows, "counts": counts, "verdict": verdict, "exit": 1 if counts["FAIL"] else 0}


def render_compare(rep: dict) -> str:
    order = {"FAIL": 0, "WARN": 1, "IMPROVED": 2, "INFO": 3, "PASS": 4}
    lines = [f"{'':<9}{'series':<62}{'reference':>11}{'this soak':>11}  unit / note"]
    for r in sorted(rep["rows"], key=lambda r: (order[r["status"]], r["id"])):
        lines.append(f"{r['status']:<9}{r['id']:<62}{str(r['reference']):>11}{str(r['new']):>11}  {r['unit']} {r['note']}")
    c = rep["counts"]
    lines += ["", f"{c['PASS']} pass, {c['IMPROVED']} improved, {c['INFO']} info, {c['WARN']} warn, {c['FAIL']} FAIL  ->  {rep['verdict']}"]
    return "\n".join(lines)
