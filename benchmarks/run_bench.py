#!/usr/bin/env python3
"""benchmarks/run_bench.py — before/after benchmark driver for the DeepStream
sidecar enhancements.

Modes:
  before      materialize code from 32aaf69 (pre-enhancement) + FP32 pgie config
  after       restore HEAD code (enhancements) + INT8 pgie config (committed)
  after_fp32  restore HEAD code but flip the pgie to FP32 (isolates INT8's delta)
  compare     print the markdown comparison table from results/*.json

Any run mode accepts --telemetry to enable the per-stage [telemetry] timing
(telemetry_enabled: true in sidecar.yaml for the run; the yaml is restored
afterwards) and folds the stage busy%/avg table into results/<mode>.json
and the compare output. NOTE: it only produces lines on code versions that
have the instrumentation (HEAD) — 'before' (32aaf69) will simply show no
[telemetry] rows.

Each run:
  1. materializes the requested code/config into the working tree
     (the container bind-mounts this repo at /workspace/iris/...),
  2. injects [bench] timing wrappers (benchmarks/instrument.py),
  3. restarts the deepstream-sidecar container,
  4. waits for the first fps line, samples GPU utilization + logs,
  5. writes benchmarks/results/<mode>.json.

Run from the repo root on the host (needs docker + nvidia-smi access).
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEFORE = "32aaf69"  # parent of commit 015b069 "Deepstream enhancements"
# Resolved lazily: the compose project name derives from the checkout directory,
# so the sidecar container name differs between machines. Never hardcode it.
SIDECAR = None
SWAP_FILES = [
    "perception/pipeline/sidecar.py",
    "perception/modules/face_recognition/recognizer.py",
    "perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt",
]
RESULTS = Path(__file__).resolve().parent / "results"
SAMPLE_SECONDS = 90  # GPU/log sampling window after fps starts
GPU_STEP = 3

# Docker's --timestamps prefix (RFC3339Nano) carries the authoritative time;
# the sidecar's own formatter only prints HH:MM:SS.
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def log_epoch(line: str):
    m = TS_RE.search(line)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def sidecar_container() -> str:
    """Auto-detect the deepstream-sidecar container (compose project name varies
    with the checkout directory name, so the container can't be hardcoded)."""
    global SIDECAR
    if SIDECAR:
        return SIDECAR
    r = sh("docker", "compose", "ps", "-q", "deepstream-sidecar", cwd=str(ROOT))
    if r.returncode == 0 and r.stdout.strip():
        SIDECAR = r.stdout.splitlines()[0]
    else:
        r = sh("docker", "ps", "--format", "{{.Names}}")
        SIDECAR = next((n for n in r.stdout.split() if "sidecar" in n), None)
    if not SIDECAR:
        raise SystemExit("deepstream sidecar container not found - is the stack up?")
    return SIDECAR


def logs_since(ts: int) -> str:
    r = sh("docker", "logs", "--timestamps", "--since", str(ts), sidecar_container())
    return r.stdout + r.stderr


def materialize(mode: str):
    # NOTE: `git restore` on SWAP_FILES is DISABLED (commented out) so uncommitted
    # working-tree changes (e.g. perception/pipeline/sidecar.py analysis_fps) survive the run.
    # `after` mode now uses the working tree exactly as-is (HEAD code + any local
    # edits + the committed INT8 pgie config). If a previous run left
    # instrumentation in the tree, run:
    #   git checkout -- perception/pipeline/sidecar.py perception/modules/face_recognition/recognizer.py
    # after committing/stashing your changes, or strip the [bench] wrappers by
    # restoring from a backup. `before` mode still materializes 32aaf69 code via
    # `git show` (which OVERWRITES the working tree) — back up your edits first.
    # r = sh("git", "-C", str(ROOT), "restore", *SWAP_FILES)
    # if r.returncode != 0:
    #     raise SystemExit(f"git restore failed: {r.stderr}")
    if mode == "before":
        for f in SWAP_FILES:
            out = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{BEFORE}:{f}"])
            (ROOT / f).write_bytes(out)
            print(f"[{mode}] materialized {f} from {BEFORE}")
    elif mode == "after_fp32":
        cfg = ROOT / "perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt"
        text = cfg.read_text()
        # Line-anchored: the config's prose comments also mention
        # "network-mode=1", so a plain replace would hit the comment first.
        text, n = re.subn(r"(?m)^network-mode=1\s*$", "network-mode=0", text)
        cfg.write_text(text)
        print(f"[{mode}] pgie config flipped to FP32 ({n} network-mode line(s) set to 0)")
    else:  # after (HEAD INT8, already restored)
        print(f"[{mode}] HEAD code + INT8 config")


def instrument():
    r = sh("python3", str(ROOT / "benchmarks/instrument.py"),
           str(ROOT / "perception/pipeline/sidecar.py"),
           str(ROOT / "perception/modules/face_recognition/recognizer.py"))
    print(r.stdout, end="")
    if r.returncode != 0:
        raise SystemExit(f"instrument failed: {r.stderr}")


def restart() -> None:
    global SIDECAR
    r = sh("docker", "restart", sidecar_container())
    if r.returncode != 0:
        # The cached id can go stale if the container was recreated between runs
        # (docker compose up --build); clear the cache and resolve it fresh once.
        SIDECAR = None
        r = sh("docker", "restart", sidecar_container())
    if r.returncode != 0:
        raise SystemExit(f"docker restart failed: {r.stderr}")


def wait_for_fps(ts0: int, timeout: float = 150.0):
    """Return (first_fps_epoch, boot_start_epoch, logs-from-boot) once the NEW
    boot — identified by its 'sidecar starting' marker — reaches its first fps
    line. Anchoring on the marker (not wall-clock) makes the measurement immune
    to stale pre-restart fps lines and host/container clock skew."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        lines = logs_since(ts0).splitlines()
        boot_at = None
        first_fps = None
        boot_idx = None
        for i, line in enumerate(lines):
            if boot_at is None and "sidecar starting" in line:
                boot_at = log_epoch(line)
                boot_idx = i
                continue
            if boot_at is not None and " fps" in line and "bench" not in line:
                first_fps = log_epoch(line)
                break
        if first_fps is not None:
            trimmed = "\n".join(lines[boot_idx:]) if boot_idx is not None else "\n".join(lines)
            return first_fps, boot_at, trimmed
        time.sleep(2)
    return None, None, logs_since(ts0)


def sample_gpu(seconds: int):
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        r = sh("nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
               "--format=csv,noheader,nounits")
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) >= 2:
                try:
                    samples.append({"util": float(parts[0].strip()),
                                    "mem_mib": int(parts[1].strip())})
                except ValueError:
                    pass
        time.sleep(GPU_STEP)
    return samples


def parse_logs(text: str, first_fps_epoch):
    out = {
        "fps": [], "probe_busy": [], "probe_avg_ms": [], "rotation_ms": [],
        "recognizer_ms": [], "faces_per_call": [],
        "telemetry": [],
        "engine_cached": False, "engine_rebuilt": False,
        "pgie_engine_path": None, "sgie_engine_path": None,
    }
    for line in text.splitlines():
        # fps lines look like "[cam-1] 30.0 fps" — match any source id.
        m = re.search(r"\[[^\]]+\] ([\d.]+) fps", line)
        if m:
            out["fps"].append(float(m.group(1)))
        m = re.search(r"\[bench\] probe busy=([\d.]+)% frames=\d+ avg=([\d.]+)ms", line)
        if m:
            out["probe_busy"].append(float(m.group(1)))
            out["probe_avg_ms"].append(float(m.group(2)))
        m = re.search(r"\[bench\].*rotation_ms=([\d.]+)", line)
        if m:
            out["rotation_ms"].append(float(m.group(1)))
        m = re.search(r"\[bench\].*recognizer_ms=([\d.]+) faces=(\d+)", line)
        if m:
            out["recognizer_ms"].append(float(m.group(1)))
            out["faces_per_call"].append(int(m.group(2)))
        # [telemetry] per-stage lines (perception/engine/telemetry.py benchmarking mode):
        #   [telemetry] probe n=1250 total=9123.4ms busy=24.1% avg=7.30ms max=48.2ms thread=streaming window=38.0s
        m = re.search(
            # stage names may carry camera ids with hyphens (gpu.decode.cam-01)
            r"\[telemetry\] ([\w.-]+) n=(\d+) total=([\d.]+)ms busy=([\d.]+)% "
            r"avg=([\d.]+)ms max=([\d.]+)ms thread=(\S+) window=([\d.]+)s", line)
        if m:
            out["telemetry"].append({
                "stage": m.group(1), "n": int(m.group(2)),
                "total_ms": float(m.group(3)), "busy_pct": float(m.group(4)),
                "avg_ms": float(m.group(5)), "max_ms": float(m.group(6)),
                "thread": m.group(7), "window_s": float(m.group(8))})
        if "deserialized trt engine from" in line or "Use deserialized engine" in line:
            out["engine_cached"] = True
        if "Building the TensorRT Engine" in line or "Trying to create engine" in line:
            out["engine_rebuilt"] = True
        m = re.search(r"TRT engine path: (\S+)", line)
        if m:
            out["pgie_engine_path"] = m.group(1)
        m = re.search(r"sgie_face enabled.*engine=(\S+)", line)
        if m:
            out["sgie_engine_path"] = m.group(1)
    return out


def summarize(name, xs):
    if not xs:
        return None
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "p95": round(sorted(xs)[int(len(xs) * 0.95) - 1], 3),
            "max": round(max(xs), 3)}


def run_mode(mode: str):
    RESULTS.mkdir(exist_ok=True)
    materialize(mode)
    instrument()

    print(f"[{mode}] restarting {sidecar_container()} ...")
    restart()
    # Capture the log window AFTER the restart returns, so the dying pre-restart
    # process's tail (which can contain stale fps lines) is not counted.
    ts0 = int(time.time()) - 1

    first_fps_epoch, boot_start_epoch, logs = wait_for_fps(ts0)
    boot_s = ((first_fps_epoch - boot_start_epoch)
              if (first_fps_epoch and boot_start_epoch) else None)
    print(f"[{mode}] boot->first fps: {boot_s}s"
          + ("" if first_fps_epoch else "  (NO FPS — run failed?)"))

    print(f"[{mode}] sampling GPU for {SAMPLE_SECONDS}s ...")
    gpu = sample_gpu(SAMPLE_SECONDS)

    # Small tail to catch bench lines emitted during the final seconds; trim to
    # this boot (wait_for_fps already trimmed, so keep the newest full view).
    time.sleep(4)
    full = logs_since(ts0)
    if "sidecar starting" in full:
        idx = full.splitlines().index(next(l for l in full.splitlines() if "sidecar starting" in l))
        logs = "\n".join(full.splitlines()[idx:])
    else:
        logs = full

    parsed = parse_logs(logs, first_fps_epoch)
    # Only bench lines from the sampled window are meaningful (the video may
    # have ended); rotation/recognizer lines are sparse, so keep all of them.
    # [telemetry] lines: merge per-flush-window rows into per-stage medians.
    telemetry = {}
    for t in parsed["telemetry"]:
        s = telemetry.setdefault(t["stage"], {"n": 0, "total_ms": 0.0,
                                              "busy": [], "avg": [],
                                              "thread": t["thread"]})
        s["n"] += t["n"]
        s["total_ms"] += t["total_ms"]
        s["busy"].append(t["busy_pct"])
        s["avg"].append(t["avg_ms"])
    for s in telemetry.values():
        s["busy_pct"] = round(statistics.median(s["busy"]), 1)
        s["avg_ms"] = round(s["total_ms"] / s["n"], 2) if s["n"] else None
        s.pop("busy")
        s.pop("avg")
    result = {
        "mode": mode,
        "boot_to_fps_s": boot_s,
        "engine_cached": parsed["engine_cached"],
        "engine_rebuilt": parsed["engine_rebuilt"],
        "pgie_engine_path": parsed["pgie_engine_path"],
        "sgie_engine_path": parsed["sgie_engine_path"],
        "fps": summarize("fps", parsed["fps"]),
        "probe_busy_pct": summarize("busy", parsed["probe_busy"]),
        "probe_avg_ms": summarize("avg", parsed["probe_avg_ms"]),
        "rotation_ms": summarize("rot", parsed["rotation_ms"]),
        "recognizer_ms": summarize("rec", parsed["recognizer_ms"]),
        "faces_per_call": summarize("faces", parsed["faces_per_call"]),
        "telemetry": telemetry or None,
        "gpu_util_pct": summarize("util", [g["util"] for g in gpu]),
        "gpu_mem_mib": summarize("mem", [g["mem_mib"] for g in gpu]),
    }
    out = RESULTS / f"{mode}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[{mode}] wrote {out.relative_to(ROOT)}")
    return result


def compare():
    rows = []
    for mode in ("before", "after", "after_fp32"):
        p = RESULTS / f"{mode}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if len(rows) < 2:
        print("need at least before.json and after.json — run: run_bench.py before; run_bench.py after")
        return
    by = {r["mode"]: r for r in rows}
    b, a = by["before"], by["after"]

    def cell(v, unit=""):
        if v is None:
            return "n/a"
        return f"{v}{unit}"

    def num(r, k, agg="median"):
        v = r.get(k)
        return v.get(agg) if v else None

    print("| metric | before (FP32, old code) | after (INT8, new code) |")
    print("|---|---|---|")
    print(f"| boot to first fps | {cell(b['boot_to_fps_s'], 's')} | {cell(a['boot_to_fps_s'], 's')} |")
    print(f"| face engine | {'rebuilt' if b['engine_rebuilt'] else 'cached'} | {'rebuilt' if a['engine_rebuilt'] else 'cached'} |")
    print(f"| pipeline fps | {cell(num(b,'fps'),'')} | {cell(num(a,'fps'),'')} |")
    print(f"| probe busy % (streaming thread) | {cell(num(b,'probe_busy_pct'),'%')} | {cell(num(a,'probe_busy_pct'),'%')} |")
    print(f"| probe avg per buffer | {cell(num(b,'probe_avg_ms'),'ms')} | {cell(num(a,'probe_avg_ms'),'ms')} |")
    print(f"| segment rotation callback | {cell(num(b,'rotation_ms'),'ms')} | {cell(num(a,'rotation_ms'),'ms')} |")
    print(f"| recognizer per frame | {cell(num(b,'recognizer_ms'),'ms')} | {cell(num(a,'recognizer_ms'),'ms')} |")
    print(f"| GPU util % | {cell(num(b,'gpu_util_pct'),'%')} | {cell(num(a,'gpu_util_pct'),'%')} |")
    print(f"| GPU mem | {cell(num(b,'gpu_mem_mib'),'MiB')} | {cell(num(a,'gpu_mem_mib'),'MiB')} |")
    if "after_fp32" in by:
        c = by["after_fp32"]
        print()
        print("INT8 isolation (after code): FP32 vs INT8")
        print(f"- GPU util: {cell(num(c,'gpu_util_pct'),'%')} (FP32) -> {cell(num(a,'gpu_util_pct'),'%')} (INT8)")
        print(f"- GPU mem:  {cell(num(c,'gpu_mem_mib'),'MiB')} (FP32) -> {cell(num(a,'gpu_mem_mib'),'MiB')} (INT8)")
        print(f"- engine:   fp32 39.5MB / int8 16.6MB (measured on disk)")
    if a.get("telemetry"):
        print()
        print("per-stage consumption (after, from [telemetry] benchmarking mode):")
        print(f"| stage | n | busy% | avg_ms | total_ms | thread |")
        print("|---|---|---|---|---|---|")
        for name in sorted(a["telemetry"]):
            t = a["telemetry"][name]
            print(f"| {name} | {t['n']} | {t['busy_pct']} | {t['avg_ms']} | "
                  f"{round(t['total_ms'], 1)} | {t['thread']} |")


def set_telemetry(enabled: bool):
    """Toggle telemetry_enabled in perception/pipeline/sidecar.yaml (backed up/restored by
    main when --telemetry is passed). Mirror of write_sources' handling."""
    p = ROOT / "sidecar" / "sidecar.yaml"
    text = p.read_text()
    val = str(bool(enabled)).lower()
    if re.search(r"(?m)^telemetry_enabled:.*$", text):
        text = re.sub(r"(?m)^telemetry_enabled:.*$", f"telemetry_enabled: {val}", text)
    else:
        text = re.sub(r"(?m)^(stream_fps:.*)$", f"\\1\ntelemetry_enabled: {val}", text)
    p.write_text(text)


def main(argv):
    args = argv[1:] if argv else []
    if not args or args[0] not in ("before", "after", "after_fp32", "compare"):
        print(__doc__)
        return 2
    if args[0] == "compare":
        compare()
        return 0
    yaml_backup = (ROOT / "sidecar" / "sidecar.yaml").read_text()
    try:
        if "--telemetry" in args:
            set_telemetry(True)
            print("[run_bench] telemetry ENABLED for this run")
        run_mode(args[0])
    finally:
        if "--telemetry" in args:
            (ROOT / "sidecar" / "sidecar.yaml").write_text(yaml_backup)
            print("[run_bench] sidecar.yaml restored")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
