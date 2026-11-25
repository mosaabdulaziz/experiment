#!/usr/bin/env python3
"""What each module and model costs on the GPU: reads the sidecar's [telemetry] lines and prints, per GPU stage
(`gpu.infer.<gie name>`, `gpu.decode.<camera>`, `gpu.jpeg`), how busy it kept the GPU and how long one batch took.

Run a config to completion with tools/engine_gpu_budget.sh (which feeds this), or on any log:

    docker compose logs --no-color deepstream-sidecar | python3 tools/engine_gpu_budget.py -
    python3 tools/engine_gpu_budget.py sidecar.log --max-total 80     # exit 1 when the GPU stages add up past 80%

Stages are averaged over the run after the first window (model load and engine build make it unrepresentative).
Several `gpu.decode.<camera>` rows are summed into one `gpu.decode`. busy% is the share of the telemetry window the
stage was executing; on one GPU the shares add up, so the total says how much room a new model has.
"""
import argparse
import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[telemetry\] (?P<name>gpu\.\S+) n=(?P<n>\d+) total=(?P<total>[\d.]+)ms busy=(?P<busy>[\d.]+)% "
                  r"avg=(?P<avg>[\d.]+)ms max=(?P<max>[\d.]+)ms .*window=(?P<window>[\d.]+)s")


def stage_of(name: str) -> str:
    """gpu.decode.cam-1 and gpu.decode.cam-2 are one stage."""
    return "gpu.decode" if name.startswith("gpu.decode.") else name


def parse(text: str, skip_first_window: bool = True) -> dict:
    """{stage: {"busy": mean busy% per window (stages of one window summed), "avg_ms", "max_ms", "n", "windows"}}."""
    per_window = defaultdict(lambda: defaultdict(lambda: {"busy": 0.0, "n": 0, "total": 0.0, "max": 0.0}))
    seen_windows, names = 0, set()
    for line in text.splitlines():
        m = LINE.search(line)
        if not m:
            continue
        # a flush writes every stage in one burst: a new window starts when a stage name repeats
        if m["name"] in names:
            seen_windows, names = seen_windows + 1, set()
        names.add(m["name"])
        a = per_window[seen_windows][stage_of(m["name"])]
        a["busy"] += float(m["busy"])
        a["n"] += int(m["n"])
        a["total"] += float(m["total"])
        a["max"] = max(a["max"], float(m["max"]))
    windows = sorted(per_window)
    if skip_first_window and len(windows) > 1:
        windows = windows[1:]
    out = {}
    for w in windows:
        for stage, a in per_window[w].items():
            o = out.setdefault(stage, {"busy": 0.0, "n": 0, "total": 0.0, "max_ms": 0.0, "windows": 0})
            o["busy"] += a["busy"]
            o["n"] += a["n"]
            o["total"] += a["total"]
            o["max_ms"] = max(o["max_ms"], a["max"])
            o["windows"] += 1
    for o in out.values():
        o["busy"] = o["busy"] / o["windows"]
        o["avg_ms"] = o["total"] / o["n"] if o["n"] else 0.0
    return out


def render(stages: dict) -> str:
    rows = sorted(stages.items(), key=lambda kv: -kv[1]["busy"])
    width = max([len(k) for k in stages] + [5])
    lines = [f"{'stage':<{width}}  busy%   avg ms   max ms   batches"]
    for name, o in rows:
        lines.append(f"{name:<{width}}  {o['busy']:5.1f}  {o['avg_ms']:7.2f}  {o['max_ms']:7.2f}  {o['n']:8d}")
    lines.append(f"{'total':<{width}}  {total(stages):5.1f}")
    return "\n".join(lines)


def total(stages: dict) -> float:
    return sum(o["busy"] for o in stages.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="sidecar log file, or - for stdin")
    ap.add_argument("--max-total", type=float, default=None, help="exit 1 when the stages' busy%% adds up past this")
    args = ap.parse_args(argv)
    text = sys.stdin.read() if args.log == "-" else open(args.log, encoding="utf-8", errors="replace").read()
    stages = parse(text)
    if not stages:
        print("no gpu.* [telemetry] lines found (is telemetry_enabled on, and did the run last a few windows?)",
              file=sys.stderr)
        return 2
    print(render(stages))
    if args.max_total is not None and total(stages) > args.max_total:
        print(f"GPU stages add up to {total(stages):.1f}% > {args.max_total:.1f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
