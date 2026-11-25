#!/usr/bin/env python3
"""A/B tests of what the GPU core is spent on, run against the live local scale stack.

Each test changes ONE setting in the DEPLOYED sidecar config (scale-local/products/multicam/sidecar.yaml, never
the repo), restarts the sidecar, waits for 15 cameras plus a warm-up, samples the GPU once a second with
`nvidia-smi dmon`, reads the rest from Prometheus, and writes one table. The original config is always
put back and the sidecar restarted at the end, also on failure or Ctrl-C.

    python3 scripts/scale_gpu_experiment.py                  # all tests (~35 min)
    python3 scripts/scale_gpu_experiment.py --tests T0,T1    # some of them
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

TESTS = {
    "T0": ("baseline (current settings)", {}),
    "T1": ("previews off (pipeline_preview: false)", {"pipeline_preview": "false"}),
    "T2": ("inference every 4th batch (process_fps: 2.5)", {"process_fps": "2.5"}),
    "T3": ("frames dropped before the mux (analysis_fps: 10)", {"analysis_fps": "10"}),
    # The sidecar derives the inference interval from stream_fps / process_fps (10/5 = every 2nd batch), so
    # changing stream_fps alone also changes how often inference runs. T4 keeps that ratio at 2.
    "T4": ("preview rate 5 fps (stream_fps: 5, process_fps: 2.5)", {"stream_fps": "5", "process_fps": "2.5"}),
    "T5": ("inference every 3rd batch (process_fps: 3.33)", {"process_fps": "3.33"}),
}
CONTAINER = "scale-deepstream-sidecar-1"


def apply_overrides(text: str, changes: dict[str, str]) -> str:
    """Set top-level ``key: value`` lines of a yaml text, keeping a trailing comment; add missing keys."""
    for key, value in changes.items():
        pat = re.compile(rf"^({re.escape(key)}:\s*)([^#\n]*?)(\s*(?:#.*)?)$", re.M)
        if pat.search(text):
            text = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
        else:
            text = text.rstrip("\n") + f"\n{key}: {value}\n"
    return text


@contextlib.contextmanager
def overridden(path: str, changes: dict[str, str]):
    """Apply ``changes`` to the file for the duration of the block; the exact original bytes come back
    afterwards whatever happens inside."""
    with open(path, "rb") as f:
        original = f.read()
    try:
        with open(path, "wb") as f:
            f.write(apply_overrides(original.decode(), changes).encode())
        yield
    finally:
        with open(path, "wb") as f:
            f.write(original)


def parse_dmon(text: str) -> list[dict]:
    """Rows of `nvidia-smi dmon -s um`: gpu sm mem enc dec jpg ofa fb bar1 ccpm."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 6:
            continue
        try:
            rows.append({"sm": int(parts[1]), "mem": int(parts[2]), "enc": int(parts[3]),
                         "dec": int(parts[4]), "jpg": int(parts[5])})
        except ValueError:
            continue
    return rows


def gpu_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    sm = sorted(r["sm"] for r in rows)
    return {"sm_avg": statistics.mean(sm), "sm_p95": sm[min(len(sm) - 1, int(len(sm) * 0.95))],
            "sm_ge90_pct": 100.0 * sum(1 for v in sm if v >= 90) / len(sm),
            "mem_avg": statistics.mean(r["mem"] for r in rows), "dec_avg": statistics.mean(r["dec"] for r in rows),
            "samples": len(rows)}


def render_table(results: list[dict]) -> str:
    cols = [("test", "test"), ("core avg %", "sm_avg"), ("core p95 %", "sm_p95"), ("time >=90 %", "sm_ge90_pct"),
            ("mem ctrl %", "mem_avg"), ("NVDEC %", "dec_avg"), ("fps min", "fps_min"), ("queues full /15", "queues_full"),
            ("shed/min", "shed_min"), ("lost/min", "lost_min"), ("sidecar CPU %", "cpu"), ("probe ms", "probe_ms"),
            ("pgie ms", "pgie_ms"), ("stalls", "stalls")]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        cells = []
        for _c, k in cols:
            v = r.get(k)
            cells.append("-" if v is None else (v if isinstance(v, str) else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prom(base: str, expr: str):
    url = base + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)["data"]["result"]


def prom_one(base: str, expr: str):
    r = prom(base, expr)
    return float(r[0]["value"][1]) if r else None


def wait_cameras(base: str, want: int = 15, timeout: float = 600.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if (prom_one(base, "count(iris_camera_meta_age_seconds<30)") or 0) >= want:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def run_test(name: str, desc: str, changes: dict, cfg_path: str, base: str, warmup: int, sample: int) -> dict:
    print(f"\n=== {name}: {desc}", flush=True)
    r: dict = {"test": f"{name} {desc.split(' (')[0]}"}
    with overridden(cfg_path, changes):
        sh("docker", "restart", CONTAINER)
        if not wait_cameras(base):
            r["test"] += " (cameras did not all come back)"
            return r
        print(f"    15 cameras up, warming up {warmup} s", flush=True)
        time.sleep(warmup)
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        print(f"    sampling {sample} s", flush=True)
        out = sh("nvidia-smi", "dmon", "-s", "um", "-c", str(sample)).stdout
        r.update(gpu_stats(parse_dmon(out)))
        w = f"{sample}s"
        q = lambda e: prom_one(base, e)           # noqa: E731
        fps = prom(base, f'avg_over_time(iris_gauge{{name=~"fps_cam-.*"}}[{w}])')
        r["fps_min"] = min((float(x["value"][1]) for x in fps), default=None)
        r["queues_full"] = q('sum(avg_over_time((iris_gauge{name=~"mux_queue_[0-9]+"} >= bool scalar(iris_gauge{name="mux_queue_cap"}))[%s:5s]))' % w)
        r["shed_min"] = q(f'sum(rate(iris_gauge{{name=~"pub_shed_total_cam-.*"}}[{w}]))*60')
        r["lost_min"] = q(f'sum(rate(iris_gauge{{name=~"pub_drops_total_cam-.*"}}[{w}]))*60')
        r["cpu"] = q(f'avg_over_time(iris_gauge{{name="flight_cpu_pct"}}[{w}])')
        r["probe_ms"] = q(f'avg_over_time(iris_stage_avg_ms{{stage="probe"}}[{w}])')
        r["pgie_ms"] = q(f'avg_over_time(iris_stage_avg_ms{{stage="gpu.infer.pgie"}}[{w}])')
        logs = sh("docker", "logs", "--since", since + "Z", CONTAINER)
        r["stalls"] = (logs.stdout + logs.stderr).count("[source-stall] START")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/URCA/IRIS/scale-local"))
    ap.add_argument("--tests", default=",".join(TESTS))
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--sample", type=int, default=240)
    ap.add_argument("--prom", default="http://localhost:39090")
    a = ap.parse_args()
    cfg = os.path.join(a.dest, "products", "multicam", "sidecar.yaml")
    results: list[dict] = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(a.dest, "data", f"gpu-experiment-{stamp}.md")
    try:
        for name in [t for t in a.tests.split(",") if t in TESTS]:
            desc, changes = TESTS[name]
            results.append(run_test(name, desc, changes, cfg, a.prom, a.warmup, a.sample))
    except KeyboardInterrupt:
        print("interrupted: restoring the original config", flush=True)
    finally:
        print("\nrestoring the original config and restarting the sidecar", flush=True)
        sh("docker", "restart", CONTAINER)
        wait_cameras(a.prom)
    table = render_table(results)
    body = (f"# GPU experiment {stamp}\n\nFootage restarts at each test; {a.warmup} s warm-up, {a.sample} s sampled "
            f"(1 s `nvidia-smi dmon`).\n\n{table}\n")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(body)
    print("\n" + table + f"\n\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
