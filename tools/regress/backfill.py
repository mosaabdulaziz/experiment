"""Add what a results file did not record yet, from a sidecar log that was saved while it was recorded.

`regress run` reads the sidecar's CPU-side stage timers (`cpu_stages`) and flight lines (`flight`) from its log. A results file written before that
existed lacks them, but if the container's log was kept (the host recorder, `regress hostsense`, saves it with docker timestamps) they can be read
afterwards, per config, over the same steady window the file already records. Nothing is re-run.

A config without a recorded steady window (the harness takes it from the end-of-stream line, which the sidecar does not always log) gets one
inferred from the log: between its neighbours' windows, from the first fps line after the sidecar's last start marker to its last probe line. Such
a window is stored as `backfill_window`, so it is never mistaken for a recorded one.

    python3 -m regress backfill --log <sidecar log> --out <dir> run-1.json run-2.json ...
"""
import copy
import json
import os
import re

from . import parsers

_PROBE = re.compile(r"\[telemetry\] probe n=")


def index_log(text: str) -> dict:
    """Sorted epoch times of the log's markers: sidecar starts, fps lines, probe-stage lines (docker timestamps only)."""
    text = parsers.clean(text)
    day, starts, fps, probes = [0, None], [], [], []
    for raw in text.splitlines():
        t, msg = parsers._stamp(raw, day)
        if t is None or t < 1e9:
            continue
        if parsers.STARTING in msg:
            starts.append(t)
        elif parsers._FPS.search(msg):
            fps.append(t)
        elif _PROBE.search(msg):
            probes.append(t)
    return {"starts": sorted(starts), "fps": sorted(fps), "probes": sorted(probes)}


def infer_window(idx: dict, after: float, next_lo: float):
    """The replay that ran between `after` and the next config's first fps line `next_lo` (epoch): [its first fps line, its last probe line], or None.
    The start marker that belongs to it is the latest one that has fps lines after it: the next config's own start (its boot is the last thing
    before `next_lo`) has none, and a retried replay leaves an earlier start with fps lines but a later one that counts."""
    gap = [t for t in idx["starts"] if after < t < next_lo]
    mine = [t for t in gap if any(t < f < next_lo for f in idx["fps"])]
    if not mine:
        return None
    start = mine[-1]
    later = [t for t in gap if t > start]
    before = later[0] if later else next_lo
    fps = [t for t in idx["fps"] if start < t < before]
    probes = [t for t in idx["probes"] if start < t < before]
    if not fps or not probes or probes[-1] <= fps[0]:
        return None
    return [fps[0], probes[-1]]


def backfill(result: dict, log_text: str, idx: dict = None) -> tuple:
    """(a copy of `result` with cpu_stages and flight added to every config that has, or can be given, a window; how many configs got them)."""
    idx = idx or index_log(log_text)
    out = copy.deepcopy(result)
    runtime = out.get("runtime") or {}
    names = list(runtime)
    done = 0
    for i, cfg in enumerate(names):
        rt = runtime[cfg]
        win = rt.get("steady_window")
        if not win:
            prev_hi = max((runtime[n]["steady_window"][1] for n in names[:i] if runtime[n].get("steady_window")), default=0.0)
            next_lo = min((runtime[n]["steady_window"][0] for n in names[i + 1:] if runtime[n].get("steady_window")), default=float("inf"))
            wall = ((out.get("correctness") or {}).get("golden", {}).get(cfg, {}).get("wall_s")) or 150.0
            # both ends must be anchored, or the search could land on any later replay in the log: the first and the last config of a run
            # get their missing side from the config's own wall time (a replay starts within about two minutes of the previous one's end)
            if prev_hi and next_lo == float("inf"):
                next_lo = prev_hi + 120.0 + wall
            elif not prev_hi and next_lo != float("inf"):
                prev_hi = next_lo - wall - 240.0
            elif not prev_hi:
                continue
            win = infer_window(idx, prev_hi, next_lo)
            if not win:
                continue
            rt["backfill_window"] = win
        lo, hi = win
        rt["cpu_stages"] = parsers.parse_cpu_stages(log_text, lo, hi)
        rt["flight"] = parsers.parse_flight(log_text, lo, hi)
        done += 1
    return out, done


def run(log_path: str, run_paths: list, out_dir: str) -> int:
    with open(log_path, errors="replace") as f:
        text = f.read()
    idx = index_log(text)
    os.makedirs(out_dir, exist_ok=True)
    for path in run_paths:
        with open(path) as f:
            res = json.load(f)
        new, n = backfill(res, text, idx)
        dest = os.path.join(out_dir, os.path.basename(path))
        with open(dest, "w") as f:
            json.dump(new, f)
        inferred = [c for c, rt in (new.get("runtime") or {}).items() if rt.get("backfill_window")]
        stages = sum(len(rt.get("cpu_stages") or {}) for rt in (new.get("runtime") or {}).values())
        print(f"{os.path.basename(path)}: {n} of {len(new.get('runtime') or {})} configs, {stages} stage series"
              + (f"; window inferred for {', '.join(inferred)}" if inferred else "") + f" -> {dest}")
    return 0
