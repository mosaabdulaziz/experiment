"""Pure parsers of the regression harness: text in, numbers out. No docker, no GPU, no network, so every one of them is
tested against real log excerpts (tests/fixtures/regress/).

What they read (all of it already exists; nothing in the sidecar was changed for this):
  * the sidecar's container log: the `sidecar starting` marker, `[cam] 20.0 fps` lines, `[telemetry] ...` stage and gauge
    lines, TensorRT's engine messages;
  * a pytest junit XML;
  * a uv.lock;
  * a directory of build output.
"""
import importlib.util
import os
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# `docker compose logs --timestamps` puts an RFC3339 time first; the sidecar's own lines carry HH:MM:SS
_DOCKER_TS = re.compile(r"^(?:\S+\s+\|\s+)?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$")
_CLOCK = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) \[")
_FPS = re.compile(r"\[([\w.-]+)\] ([\d.]+) fps")
_ELEMENT = re.compile(r"<(\w+)>")          # nvinfer's log lines name the element: ...gst_nvinfer_logger:<sgie_face> NvDsInferContext...
_GAUGE = re.compile(r"\[telemetry\] gauge ([\w.-]+)=(-?[\d.]+)")
# `[telemetry] probe n=100 total=233.8ms busy=4.7% avg=2.34ms max=5.90ms thread=Dummy-3 window=5.0s`
_STAGE = re.compile(r"\[telemetry\] ([\w.]+) n=(\d+) total=([\d.]+)ms busy=([\d.]+)% avg=([\d.]+)ms max=([\d.]+)ms thread=\S+ window=([\d.]+)s")
# `[flight] cpu 65% in 109 threads: python 48%, Dummy 6% | main-loop lag 0.0s, sched delay 0ms`
_FLIGHT = re.compile(r"\[flight\] cpu (\d+)% in (\d+) threads: (.*?) \| main-loop lag ([\d.]+)s, sched delay (\d+)ms")
STARTING = "sidecar starting"
EOS = ("EOS \u2014 pipeline reached end of stream", "pipeline reached end of stream")
# the rules benchmarks/run_bench.py already used for these two
ENGINE_CACHED = ("deserialized trt engine from", "Use deserialized engine")
ENGINE_REBUILT = ("Building the TensorRT Engine", "Trying to create engine")
WARMUP_READINGS = 2           # the first fps readings of a camera are the ramp-up, not the steady state


def clean(text: str) -> str:
    return _ANSI.sub("", text)


def _stamp(line: str, day_offset: list) -> tuple:
    """(seconds since the first line, message). A docker RFC3339 time is absolute; an HH:MM:SS clock wraps at midnight."""
    m = _DOCKER_TS.match(line)
    if m:
        # docker writes nanoseconds; fromisoformat (before 3.11) takes at most microseconds
        stamp = re.sub(r"(\.\d{6})\d+", r"\1", m.group(1)).replace("Z", "+00:00")
        dt = datetime.fromisoformat(stamp).astimezone(timezone.utc)
        return dt.timestamp(), m.group(2)
    m = _CLOCK.match(line)
    if m:
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if day_offset[1] is not None and secs + day_offset[0] < day_offset[1] - 1:
            day_offset[0] += 86400            # the clock went past midnight
        day_offset[1] = secs + day_offset[0]
        return secs + day_offset[0], line
    return None, line


def _in_window(t, lo, hi) -> bool:
    return lo is None or (t is not None and lo <= t <= hi)


def parse_cpu_stages(text: str, lo: float = None, hi: float = None) -> dict:
    """The sidecar's non-GPU stage timers (probe, publish, publish.draw, publish.redis, rotation, upload, ...) over a window of epoch
    seconds (docker timestamps; the whole log when `lo` is None): {stage: {avg_ms, busy, max_ms, n}}.
    avg_ms = total time / calls; busy = time spent in the stage as a percentage of the window (several threads can add up past 100).
    The GPU stages (gpu.*) are not here: tools/engine_gpu_budget.parse reads those."""
    text = clean(text)
    day, acc, longest = [0, None], {}, 0.0
    for raw in text.splitlines():
        t, msg = _stamp(raw, day)
        m = _STAGE.search(msg)
        if not m or m.group(1).startswith("gpu.") or not _in_window(t, lo, hi):
            continue
        a = acc.setdefault(m.group(1), {"n": 0, "total": 0.0, "max": 0.0, "win": 0.0})
        a["n"] += int(m.group(2))
        a["total"] += float(m.group(3))
        a["max"] = max(a["max"], float(m.group(6)))
        a["win"] += float(m.group(7))
        longest = max(longest, a["win"])
    duration = (hi - lo) if lo is not None and hi is not None and hi > lo else longest
    return {name: {"avg_ms": round(a["total"] / a["n"], 3) if a["n"] else None,
                   "busy": round(100.0 * a["total"] / (duration * 1000.0), 2) if duration else None,
                   "max_ms": round(a["max"], 2), "n": a["n"]} for name, a in sorted(acc.items())}


def parse_flight(text: str, lo: float = None, hi: float = None) -> dict:
    """The sidecar's flight-recorder lines (every 30 s) over a window: the process's CPU %, the busiest thread's share, main-loop lag and
    scheduler delay. {} when there is no line in the window (a replay of about a minute has two)."""
    text = clean(text)
    day, cpu, top, threads, lag, sched = [0, None], [], [], [], [], []
    for raw in text.splitlines():
        t, msg = _stamp(raw, day)
        m = _FLIGHT.search(msg)
        if not m or not _in_window(t, lo, hi):
            continue
        cpu.append(int(m.group(1)))
        threads.append(int(m.group(2)))
        first = re.match(r"\s*[\w.-]+(?: [\w.-]+)* (\d+)%", m.group(3))
        top.append(int(first.group(1)) if first else 0)
        lag.append(float(m.group(4)))
        sched.append(int(m.group(5)))
    if not cpu:
        return {}
    return {"samples": len(cpu), "cpu_pct_max": max(cpu), "top_thread_pct_max": max(top), "threads_max": max(threads),
            "main_loop_lag_max_s": max(lag), "sched_delay_max_ms": max(sched)}


def parse_runtime(text: str, gpu_budget_parse=None) -> dict:
    """Everything the harness records about one replay from its sidecar log:

      boot_to_first_fps_s   from the last `sidecar starting` to the first fps line after it (None when either is missing)
      engine                {cached, rebuilt, cached_elements, rebuilt_elements}: a rebuilt engine makes the boot time
                            unrepresentative, and the elements (pgie, sgie_face, ...) say which one it was
      fps                   {camera: {"median", "p10", "min", "n"}} over the steady state (the first readings dropped)
      drops                 last value of qos_dropped_total, publisher_drops, pub_shed_total (summed), stall_episodes (summed)
      gpu_stages            {stage: {"busy", "avg_ms", "max_ms", "n"}} from tools/engine_gpu_budget.parse, first window dropped
      gpu_total_busy        sum of the stages' busy %
    """
    text = clean(text)
    day = [0, None]
    boot_at, first_fps_at, eos_at = None, None, None
    cached_by, rebuilt_by = set(), set()
    fps: dict = {}
    gauges: dict = {}
    for raw in text.splitlines():
        t, msg = _stamp(raw, day)
        if STARTING in msg:
            boot_at, first_fps_at = t, None           # a restart inside the log: measure the last boot
            fps = {}
        element = (_ELEMENT.search(raw) or [None, "?"])[1]
        if any(s in raw for s in ENGINE_CACHED):
            cached_by.add(element)
        if any(s in raw for s in ENGINE_REBUILT):
            rebuilt_by.add(element)
        if eos_at is None and any(e in raw for e in EOS):
            eos_at = t
        m = _FPS.search(msg)
        if m:
            if first_fps_at is None:
                first_fps_at = t
            fps.setdefault(m.group(1), []).append(float(m.group(2)))
        g = _GAUGE.search(msg)
        if g:
            gauges[g.group(1)] = float(g.group(2))
    boot = None
    if boot_at is not None and first_fps_at is not None:
        boot = round(first_fps_at - boot_at, 1)
    cams = {}
    for cam, xs in fps.items():
        steady = xs[WARMUP_READINGS:] if len(xs) > WARMUP_READINGS + 1 else xs
        srt = sorted(steady)
        cams[cam] = {"median": round(statistics.median(steady), 2), "p10": round(srt[max(0, int(len(srt) * 0.1))], 2),
                     "min": round(srt[0], 2), "n": len(steady)}
    drops = {
        "qos_dropped_total": gauges.get("qos_dropped_total", 0.0),
        "publisher_drops": gauges.get("publisher_drops", 0.0),
        "pub_shed_total": sum(v for k, v in gauges.items() if k.startswith("pub_shed_total_")),
        "stall_episodes": sum(v for k, v in gauges.items() if k.startswith("source_stall_episodes_total_")),
    }
    stages = {}
    if gpu_budget_parse is not None:
        for name, o in gpu_budget_parse(text).items():
            stages[name] = {"busy": round(o["busy"], 2), "avg_ms": round(o["avg_ms"], 2),
                            "max_ms": round(o["max_ms"], 2), "n": o["n"]}
    engine = {"cached": bool(cached_by), "rebuilt": bool(rebuilt_by),
              "cached_elements": sorted(cached_by), "rebuilt_elements": sorted(rebuilt_by)}
    # the stretch in which the pipeline really ran, as wall-clock epoch seconds (docker timestamps), for the counter window
    window = [first_fps_at, eos_at] if first_fps_at and eos_at and first_fps_at > 1e9 and eos_at > first_fps_at else None
    lo, hi = window if window else (None, None)
    return {"boot_to_first_fps_s": boot, "steady_window": window, "engine": engine, "fps": cams, "drops": drops,
            "gpu_stages": stages, "gpu_total_busy": round(sum(s["busy"] for s in stages.values()), 2),
            "cpu_stages": parse_cpu_stages(text, lo, hi), "flight": parse_flight(text, lo, hi)}


def load_gpu_budget_parse(repo: str):
    """tools/engine_gpu_budget.py's parse(), loaded by path (it is a script, not a package)."""
    path = os.path.join(repo, "tools", "engine_gpu_budget.py")
    spec = importlib.util.spec_from_file_location("engine_gpu_budget", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


def parse_junit(xml_text: str, slowest: int = 10) -> dict:
    """{tests, failures, errors, skipped, time_s, failed: [ids], slowest: [[id, s], ...]} of a pytest --junitxml file."""
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time_s": 0.0, "failed": [], "slowest": []}
    cases = []
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            out[k] += int(s.get(k, 0))
        out["time_s"] += float(s.get("time", 0.0))
        for c in s.iter("testcase"):
            cid = f"{c.get('classname', '')}::{c.get('name', '')}"
            cases.append((float(c.get("time", 0.0)), cid))
            if c.find("failure") is not None or c.find("error") is not None:
                out["failed"].append(cid)
    out["time_s"] = round(out["time_s"], 1)
    out["failed"] = sorted(out["failed"])
    out["slowest"] = [[cid, round(t, 2)] for t, cid in sorted(cases, reverse=True)[:slowest]]
    return out


def count_lock_packages(lock_text: str) -> int:
    """Packages a uv.lock resolved (its own project excluded)."""
    import tomllib                                       # 3.11+; only this function needs it (the Jetson has 3.10)
    pkgs = tomllib.loads(lock_text).get("package", [])
    return sum(1 for p in pkgs if "registry" in p.get("source", {}) or "git" in p.get("source", {}) or "url" in p.get("source", {}))


def dir_files(path: str) -> list:
    """[(relative path, bytes)] of every file under `path`."""
    out = []
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            full = os.path.join(dirpath, f)
            out.append((os.path.relpath(full, path), os.path.getsize(full)))
    return out


def bundle_summary(files: list) -> dict:
    """The frontend build output: total bytes, the JS and CSS bytes, and the largest chunk."""
    js = sum(n for p, n in files if p.endswith(".js"))
    css = sum(n for p, n in files if p.endswith(".css"))
    biggest = max(files, key=lambda f: f[1], default=("", 0))
    return {"total_bytes": sum(n for _p, n in files), "js_bytes": js, "css_bytes": css,
            "files": len(files), "largest_chunk": {"file": biggest[0], "bytes": biggest[1]}}
