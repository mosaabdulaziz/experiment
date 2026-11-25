"""Stream-capacity benchmark for the IRIS DeepStream sidecar.

Sweeps N sources x source frame rate (via transcoded H.264 fixtures) and
measures how far the host GPU can go before
it saturates (~95% util). For each cell:

  1. rewrites sidecar.yaml: sources = N copies of a pre-transcoded <fps>-fps
     H.264 fixture (benchmarks/fixtures/fps{fps}.mp4) so decode AND inference
     genuinely run at the target frame rate; stream_fps is set to match
     (stream_fps alone only throttles the JPEG/analysis cadence, NOT the
     DeepStream pipeline -- see RESULTS.md for the root-cause note)
  2. optionally sets analysis_fps=<target> (the videorate pre-mux cap on the
     analysis branch). The measured per-source fps is the ACHIEVED analysis
     rate: videorate quantizes non-integer source/analysis ratios to the
     nearest submultiple (25 fps source -> af=10 measures ~8.4, af=15 ~12.5),
     which is accepted behavior (the user chose videorate over the exact
     wall-clock throttle for its native-PTS pacing). The saturation check
     tolerates this with a 0.75x target floor.
  3. restarts the sidecar container
  4. waits for the first "[cam-*] .. fps" log (anchored on 'sidecar starting'
     so stale pre-restart lines can't pollute the window), tolerating engine
     rebuilds
  5. samples host GPU util/VRAM + host CPU every `sample_every` s for
     `window_s` s, and collects per-source fps + saturation/error lines
  6. writes benchmarks/results/capacity_<n>s_<fps>fps.json and appends the
     summary to benchmarks/results/capacity_summary.jsonl

Usage:
  python3 benchmarks/stream_capacity.py fixtures --source <video>   # build fps fixtures
  python3 benchmarks/stream_capacity.py run --n 4 --fps 10 [--analysis-fps 5]
  python3 benchmarks/stream_capacity.py sweep            # full grid
  python3 benchmarks/stream_capacity.py summary          # print the grid
  python3 benchmarks/stream_capacity.py restore          # restore original yaml

--source points at any real camera/street video used to build the fixtures;
it defaults to the test clip in data/uploads (gitignored, so drop your own
video there on a fresh machine). Restores the original sidecar.yaml at the
end of run/sweep (or on failure).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR_YAML = os.path.join(REPO, "perception", "pipeline", "sidecar.yaml")
RESULTS_DIR = os.path.join(REPO, "benchmarks", "results")
SOURCE_VIDEO = os.path.join(REPO, "data", "uploads", "attendance", "camera_5_2023-06-08-12-29-08.avi")
FIXTURES_DIR = os.path.join(REPO, "benchmarks", "fixtures")
CONTAINER_PREFIX = "file:///workspace/iris/"


def fixture_uri(fps: int) -> str:
    """In-container file:// URI of the pre-transcoded <fps>-fps fixture."""
    return f"{CONTAINER_PREFIX}benchmarks/fixtures/fps{fps}.mp4"


def start_pushers(n: int, fps: int):
    """Spawn one host ffmpeg per source, looping the fps fixture into go2rtc's
    RTSP ingest as raw-cam-N. `-c copy` keeps the original codec — the sidecar
    decodes the same stream a production camera would send. Must run BEFORE the
    sidecar restart so every source has a live producer at boot."""
    global PUSHERS
    stop_pushers()
    for i in range(1, n + 1):
        path = os.path.join(FIXTURES_DIR, f"fps{fps}.mp4")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
               "-stream_loop", "-1", "-i", path, "-c", "copy",
               "-rtsp_transport", "tcp", "-f", "rtsp",
               f"rtsp://localhost:8554/raw-cam-{i}"]
        PUSHERS.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL))
    # Let the publishes land before the sidecar boots against them.
    time.sleep(4)


def stop_pushers():
    global PUSHERS
    for p in PUSHERS:
        try:
            p.terminate()
        except Exception:
            pass
    for p in PUSHERS:
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    PUSHERS = []
    # Drop the now-orphaned RTSP sessions from go2rtc before the next cell.
    time.sleep(1)


def ensure_fixtures():
    """Transcode the source test video to every fps in FPS_GRID once, so each
    cell exercises a real <fps>-fps camera stream (decode + inference scale
    down with fps, unlike stream_fps which only trims the JPEG cadence)."""
    if not os.path.exists(SOURCE_VIDEO):
        raise RuntimeError(f"source video not found: {SOURCE_VIDEO}")
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    for fps in FPS_GRID:
        out = os.path.join(FIXTURES_DIR, f"fps{fps}.mp4")
        if os.path.exists(out) and os.path.getsize(out) > 100_000:
            continue
        # Temp name must still end in .mp4 or ffmpeg can't infer the muxer;
        # the atomic rename then lands the finished file at its real name.
        tmp = out + ".part.mp4"
        cmd = (f"ffmpeg -y -v error -i {SOURCE_VIDEO} -vf fps={fps} "
               f"-c:v libx264 -preset veryfast -crf 23 -an {tmp}")
        r = sh(cmd)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 100_000:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise RuntimeError(f"fixture transcode failed for fps={fps}: {r.stderr[-400:]}")
        os.replace(tmp, out)  # atomic: a killed run never leaves a bogus accepted file
        print(f"transcoded {out}")

SAMPLE_EVERY = 2.0      # s between GPU/CPU samples
WINDOW_S = 60.0         # sampling window per cell (longer => p95 is a real level, not one spike)
MAX_BOOT_WAIT = 240.0   # s to wait for first fps line (engine rebuilds)
N_GRID = [1, 2, 4, 6, 8, 10, 12]
FPS_GRID = [25, 15, 10, 5]

_SOURCE_BLOCK_RE = re.compile(r"(?m)^sources:\n(?:[ \t]+-[^\n]*\n|[ \t]*#[^\n]*\n|  [ \t]*\n)+")

BACKUP = os.path.join(RESULTS_DIR, "_sidecar.yaml.bak")
NO_RECORD = False   # set from --no-record; gates record_enabled in write_sources
SHARDS = 1          # set from --shards; N sidecar replicas share the cell's sources
TELEMETRY = False   # set from --telemetry; gates telemetry_enabled + [telemetry] parsing
RTSP = False        # set from --rtsp; sources are rtsp://go2rtc:8554/raw-cam-N (SVR path) instead of file:// fixtures

# ffmpeg pusher processes feeding go2rtc's RTSP ingest in RTSP mode (one per
# source, looping the fps fixture with -c copy so the sidecar sees a genuine
# <fps>-fps RTSP camera, original codecs — the exact production ingest path).
PUSHERS = []

# Per-stage [telemetry] log lines emitted by perception/engine/telemetry.py every
# telemetry_interval s (one line per stage per flush window, merged across the
# stage's threads):
#   [telemetry] probe n=1250 total=9123.4ms busy=24.1% avg=7.30ms max=48.2ms thread=streaming window=38.0s
# plus gauge samples:
#   [telemetry] gauge publisher_pending=2
TELEMETRY_STAGE_RE = re.compile(
    # stage names may carry camera ids with hyphens (gpu.decode.cam-01)
    r"\[telemetry\] ([\w.-]+) n=(\d+) total=([\d.]+)ms busy=([\d.]+)% "
    r"avg=([\d.]+)ms max=([\d.]+)ms thread=(\S+) window=([\d.]+)s")
TELEMETRY_GAUGE_RE = re.compile(r"\[telemetry\] gauge (\w+)=(-?\d+)")


def parse_telemetry(logs):
    """Aggregate every [telemetry] flush line in `logs` into a cell-level summary.

    busy% is per flush window (share of that stage's thread wall-clock); across
    the cell it becomes a median + count. avg_ms is the mean of the window
    averages. Gauges keep the last sampled value.
    """
    stages, gauges = {}, {}
    for m in TELEMETRY_STAGE_RE.finditer(logs):
        name = m.group(1)
        entry = stages.setdefault(name, {"n": 0, "total_ms": 0.0, "busy": [],
                                        "avg": [], "thread": m.group(7),
                                        "window_s": float(m.group(8))})
        entry["n"] += int(m.group(2))
        entry["total_ms"] += float(m.group(3))
        entry["busy"].append(float(m.group(4)))
        entry["avg"].append(float(m.group(5)))
        entry["max_ms"] = max(entry.get("max_ms", 0.0), float(m.group(6)))
    out = {}
    for name, e in stages.items():
        busy_sorted = sorted(e["busy"])
        out[name] = {
            "n": e["n"],
            "total_ms": round(e["total_ms"], 1),
            "busy_pct": {"median": busy_sorted[len(busy_sorted) // 2],
                          "n": len(busy_sorted)},
            # total/n is the exact per-call average (mean-of-window-means would
            # skew toward whatever windows had few calls)
            "avg_ms": round(e["total_ms"] / e["n"], 2) if e["n"] else None,
            "max_ms": round(e["max_ms"], 2),
            "calls_per_s": round(e["n"] / e["window_s"], 1) if e["window_s"] > 0 else None,
            "thread": e["thread"],
        }
    for m in TELEMETRY_GAUGE_RE.finditer(logs):
        gauges[m.group(1)] = int(m.group(2))
    return {"stages": out, "gauges": gauges} if (out or gauges) else None


def print_telemetry_table(tlm):
    """Print a per-stage busy%/avg table from a parsed telemetry dict."""
    if not tlm or not tlm.get("stages"):
        print("  (no [telemetry] lines — run with --telemetry)")
        return
    print("  per-stage consumption (busy% = share of that stage's thread wall-clock):")
    print(f"    {'stage':<20}{'calls':>8}{'busy%':>8}{'avg_ms':>8}{'max_ms':>9}{'calls/s':>9}  thread")
    for name in sorted(tlm["stages"]):
        e = tlm["stages"][name]
        bp = e["busy_pct"]["median"] if e["busy_pct"] else None
        print(f"    {name:<20}{e['n']:>8}{str(bp):>8}{e['avg_ms']:>8}"
              f"{e['max_ms']:>9}{str(e['calls_per_s']):>9}  {e['thread']}")
    if tlm.get("gauges"):
        gs = " ".join(f"{k}={v}" for k, v in sorted(tlm["gauges"].items()))
        print(f"    gauges: {gs}")


def shard_service_names() -> list:
    """Compose service names for the sidecar replicas.

    deepstream-sidecar (shard 0) always exists; shard 1/2 are behind the
    'shards' compose profile. Return the names of the replicas that are
    actually running, oldest-compose-defined first.
    """
    out = sh("docker compose ps --format '{{.Service}}' | grep '^deepstream-sidecar' || true",
             cwd=REPO).stdout.split()
    return out or ["deepstream-sidecar"]


def sidecar_containers() -> list:
    """Container IDs of every running sidecar replica (1 in single-process mode)."""
    ids = []
    for svc in shard_service_names():
        out = sh(f"docker compose ps -q {svc}", cwd=REPO).stdout.split()
        ids.extend(out)
    return ids


def shard_source_count(n: int) -> int:
    """Per-shard source count when `n` sources are split across SHARDS replicas.

    sidecar.py slices sources[shard_id::shard_count]; with an even split every
    shard owns exactly n // SHARDS. The benchmark requires n % SHARDS == 0 so
    the per-shard engine batch (and thus int8_calib_batch_size) is uniform.
    """
    if SHARDS <= 1:
        return n
    if n % SHARDS != 0:
        raise RuntimeError(f"--shards {SHARDS} requires n divisible by {SHARDS} "
                           f"(got n={n})")
    return n // SHARDS


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def original_yaml():
    if os.path.exists(BACKUP):
        return open(BACKUP).read()
    return open(SIDECAR_YAML).read()


def write_sources(n, fps, analysis_fps=None):
    """Rewrite sidecar.yaml with n sources + stream_fps; keep everything else.

    When analysis_fps is not None it is written explicitly (videorate cap on
    the analysis branch). The analysis branch only sees analysis_fps frames/s,
    so process_fps's infer_interval would compound with the cap -- the sidecar
    forces infer_interval=1 whenever analysis_fps>0, but we zero process_fps
    here too so the yaml states the intent.

    With SHARDS > 1 every replica reads the SAME sidecar.yaml and slices
    sources[IRIS_SHARD_ID::IRIS_SHARD_COUNT]; the engine batch is the
    per-shard source count (n // SHARDS), so int8_calib_batch_size mirrors
    that, not n."""
    text = original_yaml()
    # Replace stream_fps (used for the analysis/JPEG cadence).
    text = re.sub(r"(?m)^stream_fps:.*$", f"stream_fps: {fps}", text)
    # INT8 engine builds key on the engine batch == per-shard sources, and the
    # custom YOLO calibrator segfaults during calibration if its batch size
    # (int8_calib_batch_size) doesn't match the engine batch. Mirror it.
    per_shard = shard_source_count(n)
    text = re.sub(r"(?m)^#?\s*int8_calib_batch_size:.*$",
                  f"int8_calib_batch_size: {per_shard}", text)
    # Each source's recording branch opens a concurrent NVENC session; past
    # ~15 sources the GPU/driver caps sessions and the pipeline stalls at
    # PLAYING with zero frames (observed at n=16). Capacity sweeps measure the
    # inference pipeline, so sweep with recording off; the recording cap is
    # itself a reported finding. Record mode is set EXPLICITLY both ways so a
    # record-on cell never silently inherits record_enabled: false from the
    # committed config (the merged config has recording off).
    rec_val = "true" if not NO_RECORD else "false"
    rec_re = re.compile(r"(?m)^record_enabled:.*$")
    if rec_re.search(text):
        text = rec_re.sub(f"record_enabled: {rec_val}", text)
    else:
        text = re.sub(r"(?m)^(stream_fps:.*)$", f"\\1\nrecord_enabled: {rec_val}", text)
    if analysis_fps is not None:
        if analysis_fps > 0:
            text = re.sub(r"(?m)^analysis_fps:.*$", f"analysis_fps: {analysis_fps}", text)
            text = re.sub(r"(?m)^process_fps:.*$", "process_fps: 0", text)
        else:
            text = re.sub(r"(?m)^analysis_fps:.*$", "analysis_fps: 0", text)
    # Per-stage timing telemetry. Set EXPLICITLY both ways (like record_enabled)
    # so a cell never silently inherits the other setting from the committed yaml.
    tlm_val = "true" if TELEMETRY else "false"
    tlm_re = re.compile(r"(?m)^telemetry_enabled:.*$")
    if tlm_re.search(text):
        text = tlm_re.sub(f"telemetry_enabled: {tlm_val}", text)
    else:
        text = re.sub(r"(?m)^(stream_fps:.*)$", f"\\1\ntelemetry_enabled: {tlm_val}", text)
    # Replace the sources block (current header line is 'sources:'). In RTSP
    # mode each source is the go2rtc RTSP ingest (raw-cam-N) — the production
    # SVR path — instead of a file:// fixture; the pushers must already be up
    # (start_pushers) so the sidecar finds a live producer at boot.
    if RTSP:
        block = "sources:\n" + "".join(
            f'    - {{id: cam-{i}, uri: "rtsp://go2rtc:8554/raw-cam-{i}"}}\n'
            for i in range(1, n + 1))
    else:
        block = "sources:\n" + "".join(
            f'    - {{id: cam-{i}, uri: "{fixture_uri(fps)}"}}\n' for i in range(1, n + 1))
    if _SOURCE_BLOCK_RE.search(text):
        text = _SOURCE_BLOCK_RE.sub(block, text, count=1)
    else:
        text = block + text
    open(SIDECAR_YAML, "w").write(text)


def restart_sidecars() -> list:
    names = sidecar_containers()
    if not names:
        raise RuntimeError("no sidecar containers found")
    for name in names:
        sh(f"docker restart {name}")
    return names


def wait_for_fps(names, n, fps, timeout=MAX_BOOT_WAIT, analysis_fps=None):
    """Return when ALL n sources report their first fps line after boot, so the
    sampling window starts only once every source is live (not when the first
    one appears — with many RTSP sources the reconnect ramp would pollute the
    window's min-fps). Anchored on each container's StartedAt so stale
    pre-restart lines can't win. With SHARDS > 1, every replica must reach its
    full share of cam ids before we start sampling.

    NOTE: `docker logs --since` filtering is UNRELIABLE on this docker daemon
    once the json-file log rotates (any real timestamp returns nothing), so we
    fetch a bounded --timestamps --tail and time-filter in Python instead.

    With analysis_fps set in sidecar.yaml, the probe only sees analysis_fps
    frames/s per source (e.g. 5) regardless of the fixture's native rate, so
    we don't compare against `fps` anymore — any fps line from the new boot is
    enough. The per-source reading still proves the analysis branch is live
    and the mux is batching sources.
    """
    def _anchor(name):
        started = sh(f"docker inspect -f '{{{{.State.StartedAt}}}}' {name}").stdout.strip()
        try:
            return datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return time.time() - 60

    fps_re = re.compile(r"\[(cam-\d+)\] ([\d.]+) fps")
    # Each replica owns sources[shard_id::shard_count]; shard 0 is
    # deepstream-sidecar, shard k is deepstream-sidecar-k.
    def _want(name):
        idx = shard_service_names().index(name) if len(names) > 1 else 0
        return {f"cam-{i}" for i in range(idx + 1, n + 1, len(names))}
    want = {name: _want(name) for name in names}
    # A cam that logged only a ramp-up "0.1 fps" line is NOT live — with many
    # RTSP sources connecting at once, the mux (sync-inputs=1) waits for the
    # slowest per batch, so steady state takes a while. Require TWO readings at
    # least 90 % of the target rate before a cam counts as up: a single 3.3 fps
    # reading passes a 0.5x floor but still drags the whole batch below target.
    steady_floor = 0.9 * min(fps, (analysis_fps if analysis_fps else fps))
    steady_count = {name: {} for name in names}  # name -> cam -> readings seen
    anchors = {name: _anchor(name) for name in names}
    seen = {name: set() for name in names}
    t0 = time.time()
    while time.time() - t0 < timeout:
        for name in names:
            if seen[name] == want[name]:
                continue
            out = sh(f"docker logs --timestamps --tail 1500 {name} 2>&1").stdout
            for ln in out.splitlines():
                ts = ln[:30].strip()
                try:
                    lt = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
                if lt >= anchors[name]:
                    m = fps_re.search(ln)
                    if m and float(m.group(2)) >= steady_floor:
                        cid = m.group(1)
                        steady_count[name][cid] = steady_count[name].get(cid, 0) + 1
                        if steady_count[name][cid] >= 2:
                            seen[name].add(cid)
        if all(seen[name] == want[name] for name in names):
            return time.time() - t0
        time.sleep(3)
    missing = [n for n, s in seen.items() if not s]
    raise TimeoutError(f"no fps reading within {timeout}s (shards never live: {missing})")


def sample_gpu_cpu(window_s, every):
    """Sample host GPU util/VRAM and CPU% during the window."""
    gpu_util, gpu_mem, cpu = [], [], []
    t0 = time.time()
    while time.time() - t0 < window_s:
        g = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
        parts = re.findall(r"[\d.]+", g.stdout)
        if len(parts) >= 2:
            gpu_util.append(float(parts[0]))
            gpu_mem.append(float(parts[1]))
        c = sh("top -bn1 | grep -E '^%Cpu' | head -1")
        m = re.search(r"([\d.]+)\s*id", c.stdout)
        if m:
            cpu.append(100.0 - float(m.group(1)))
        time.sleep(every)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    p95 = lambda xs: sorted(xs)[int(len(xs) * 0.95) - 1] if xs else None
    p99 = lambda xs: sorted(xs)[int(len(xs) * 0.99) - 1] if xs else None
    return {"gpu_util": {"median": med(gpu_util), "p95": p95(gpu_util), "p99": p99(gpu_util),
                          "n": len(gpu_util), "series": gpu_util},
            "gpu_mem_mib": {"median": med(gpu_mem), "p95": p95(gpu_mem), "n": len(gpu_mem),
                            "series": gpu_mem},
            "cpu_pct": {"median": med(cpu), "n": len(cpu)}}


def collect_logs(names, anchor_ts):
    """Gather per-camera fps (median of ALL readings after the restart anchor),
    saturation warnings and error lines strictly from THIS cell's window.

    Merges logs across every sidecar replica (SHARDS > 1): each shard logs its
    own cam-* ids, so the union is the full source set for the cell. Without
    the anchor stale readings from a previous cell (with a different source
    count) would pollute the stats.

    `docker logs --since` is unreliable on this daemon (see wait_for_fps), so
    we pull a bounded --timestamps --tail and drop lines older than the anchor
    in Python.
    """
    if isinstance(names, str):
        names = [names]
    kept = []
    for name in names:
        out = sh(f"docker logs --timestamps --tail 5000 {name} 2>&1").stdout
        for ln in out.splitlines():
            ts = ln[:30].strip()
            try:
                lt = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if lt >= anchor_ts:
                kept.append(ln)
    logs = "\n".join(kept)
    fps_readings: dict[str, list[float]] = {}
    for m in re.finditer(r"\[(cam-\d+)\] ([\d.]+) fps", logs):
        fps_readings.setdefault(m.group(1), []).append(float(m.group(2)))
    per_source = {sid: sorted(xs)[len(xs) // 2] for sid, xs in fps_readings.items()}
    saturated = len(re.findall(r"frame publisher saturated", logs))
    errors = [ln for ln in logs.splitlines()
              if re.search(r"ERROR|bus error|FATAL", ln)
              and "GStreamer-WARNING" not in ln]
    result = {"per_source_fps": per_source,
              "fps_line_count": len(fps_readings),
              "publisher_saturated_warnings": saturated,
              "error_lines": errors[:10]}
    if TELEMETRY:
        result["telemetry"] = parse_telemetry(logs)
    return result


def ensure_shard_containers():
    """Reconcile running sidecar replicas with SHARDS.

    SHARDS > 1: enable the compose 'shards' profile and start ONLY the wanted
    replicas by name (IRIS_SHARD_COUNT is exported so every replica slices for
    the ACTUAL shard count), reusing the existing image (--no-build: all code
    is on the shared volume, so a stale image is never the problem). SHARDS ==
    1: plain `up` (no profile) then stop any replicas left over from an
    earlier sharded run, preserving the pre-M1 restart-only behavior. Always
    stops replicas beyond SHARDS.
    """
    env = dict(os.environ, IRIS_SHARD_COUNT=str(SHARDS))
    # Start ONLY the wanted replicas by name (profile enables shard-1/2, but
    # naming the services keeps an extra replica with an out-of-range shard id
    # from crash-looping before we stop it). Unwanted leftovers are stopped.
    want = ["deepstream-sidecar"] + [f"deepstream-sidecar-{i}" for i in range(1, SHARDS)]
    profile = ["--profile", "shards"] if SHARDS > 1 else []
    sh(f"docker compose {' '.join(profile)} up -d --no-build {' '.join(want)}",
       cwd=REPO, env=env)
    for idx in range(SHARDS, 3):
        sh(f"docker compose stop deepstream-sidecar-{idx}", cwd=REPO, env=env)
    for svc in want:
        if not sh(f"docker compose ps -q {svc}", cwd=REPO, env=env).stdout.split():
            raise RuntimeError(f"sidecar shard service {svc} not running")


def run_cell(n, fps, analysis_fps=None):
    write_sources(n, fps, analysis_fps)
    if RTSP:
        start_pushers(n, fps)
    try:
        names = restart_sidecars()
        # Anchor log collection on the LATEST restart so earlier cells can't
        # leak in (all shards restart together, so use the max StartedAt).
        started_ts = []
        for name in names:
            started = sh(f"docker inspect -f '{{{{.State.StartedAt}}}}' {name}").stdout.strip()
            try:
                started_ts.append(datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp())
            except ValueError:
                started_ts.append(time.time())
        anchor_ts = max(started_ts)
        boot_s = wait_for_fps(names, n, fps, analysis_fps=analysis_fps)
        metrics = sample_gpu_cpu(WINDOW_S, SAMPLE_EVERY)
        logs = collect_logs(names, anchor_ts)
        result = {"n": n, "fps": fps, "analysis_fps": analysis_fps,
                  "record": not NO_RECORD, "shards": SHARDS,
                  "boot_s": round(boot_s, 1), **metrics, **logs}
        af = "" if analysis_fps is None else f"_af{analysis_fps}"
        rec = "_norec" if NO_RECORD else "_rec"
        shards = "" if SHARDS <= 1 else f"_s{SHARDS}"
        path = os.path.join(RESULTS_DIR, f"capacity_{n}s_{fps}fps{af}{rec}{shards}.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        json.dump(result, open(path, "w"), indent=2)
        summary = {k: result[k] for k in
                   ("n", "fps", "analysis_fps", "record", "shards", "boot_s", "gpu_util",
                    "gpu_mem_mib", "cpu_pct", "publisher_saturated_warnings")}
        if result["per_source_fps"]:
            vals = sorted(result["per_source_fps"].values())
            summary["median_per_stream_fps"] = vals[len(vals) // 2]
            summary["min_per_stream_fps"] = vals[0]
            summary["mean_per_stream_fps"] = round(sum(vals) / len(vals), 2)
        else:
            summary["median_per_stream_fps"] = None
            summary["min_per_stream_fps"] = None
            summary["mean_per_stream_fps"] = None
        summary["fps_line_count"] = result["fps_line_count"]
        with open(os.path.join(RESULTS_DIR, "capacity_summary.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        return result
    finally:
        # Always restore the original yaml so the repo is left as the user had it.
        open(SIDECAR_YAML, "w").write(original_yaml())
        if RTSP:
            stop_pushers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sweep", "summary", "restore", "fixtures", "telemetry"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--analysis-fps", type=int, default=None,
                    help="set analysis_fps (analysis-branch throttle) for the "
                         "cell; the measured per-source fps is the achieved "
                         "analysis rate")
    ap.add_argument("--source", default=None,
                    help="source video used to build the fps fixtures "
                         "(default: data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi)")
    ap.add_argument("--ns", default=None,
                    help="comma-separated stream-count grid for sweep "
                         "(default: %s)" % ",".join(map(str, N_GRID)))
    ap.add_argument("--fpss", default=None,
                    help="comma-separated fps grid for sweep (default: %s)" % ",".join(map(str, FPS_GRID)))
    ap.add_argument("--shards", type=int, default=1,
                    help="run N sidecar replicas (process sharding, M1); the cell's "
                         "sources are split across them, so --n must be divisible by "
                         "--shards (default 1 = single process)")
    ap.add_argument("--saturate", type=float, default=90.0,
                    help="stop a sweep row once median GPU util >= this %% "
                         "or min per-stream fps < 75%% of target (default 90; "
                         "the 75%% floor tolerates videorate's PTS quantization "
                         "of non-integer analysis_fps ratios)")
    ap.add_argument("--no-record", action="store_true",
                    help="set record_enabled: false per cell (required past ~15 "
                         "sources: NVENC caps concurrent recording sessions and "
                         "the pipeline stalls at PLAYING with no frames)")
    ap.add_argument("--rtsp", action="store_true",
                    help="sources are rtsp://go2rtc:8554/raw-cam-N (the production "
                         "SVR ingest path) instead of file:// fixtures; spawns one "
                         "ffmpeg pusher per source looping the fps fixture. Required "
                         "for record-on cells (Smart Video Record is RTSP-only). "
                         "Requires go2rtc.yaml raw-cam placeholders + go2rtc running.")
    ap.add_argument("--telemetry", action="store_true",
                    help="enable per-stage timing telemetry for the cell "
                         "(telemetry_enabled: true in sidecar.yaml) and collect the "
                         "[telemetry] log lines into the cell JSON; print the per-stage "
                         "busy%% breakdown after each cell. The breakdown is also "
                         "available later via 'stream_capacity.py telemetry <cell.json>'")
    ap.add_argument("cell", nargs="?", default=None,
                    help="cell JSON path — only used by the 'telemetry' subcommand")
    args = ap.parse_args()

    global SOURCE_VIDEO, NO_RECORD, SHARDS, TELEMETRY, RTSP
    if args.source:
        SOURCE_VIDEO = args.source
    NO_RECORD = args.no_record
    SHARDS = max(1, args.shards or 1)
    TELEMETRY = args.telemetry
    RTSP = args.rtsp
    if SHARDS > 3:
        sys.exit("only 3 sidecar services are defined; --shards must be 1, 2 or 3")

    if args.cmd == "fixtures":
        ensure_fixtures()
        sys.exit(0)

    if args.cmd == "run":
        if not args.n or not args.fps:
            sys.exit("run requires --n and --fps")
        ensure_fixtures()
        shard_source_count(args.n)  # validates n % SHARDS == 0 before any side effects
        ensure_shard_containers()
        if not os.path.exists(BACKUP):
            os.makedirs(RESULTS_DIR, exist_ok=True)
            sh(f"cp {SIDECAR_YAML} {BACKUP}")
        r = run_cell(args.n, args.fps, args.analysis_fps)
        print(json.dumps({k: r[k] for k in ("n", "fps", "analysis_fps", "record", "shards", "boot_s",
                                            "gpu_util", "gpu_mem_mib", "cpu_pct",
                                            "per_source_fps",
                                            "publisher_saturated_warnings")}, indent=2))
        if r.get("telemetry"):
            print_telemetry_table(r["telemetry"])

    elif args.cmd == "sweep":
        ensure_fixtures()
        shard_source_count(1)  # validate --shards before any side effects
        if args.shards and args.shards > 3:
            sys.exit("only 3 sidecar services are defined; --shards must be 1, 2 or 3")
        ensure_shard_containers()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        sh(f"cp {SIDECAR_YAML} {BACKUP}")
        ns = [int(x) for x in args.ns.split(",")] if args.ns else N_GRID
        fpss = [int(x) for x in args.fpss.split(",")] if args.fpss else FPS_GRID
        def _run_cell(n, fps):
            """Run one cell; returns True when the row is saturated (stop)."""
            r = run_cell(n, fps, args.analysis_fps)
            gu = r['gpu_util']['median']
            psc = r['per_source_fps']
            rec = "rec" if r['record'] else "no-rec"
            print(f"n={n} fps={fps} af={args.analysis_fps} [{rec}]: "
                  f"gpu_util_med={gu}% vram={r['gpu_mem_mib']['median']}MiB "
                  f"stream_fps={psc}")
            if TELEMETRY and r.get("telemetry"):
                print_telemetry_table(r["telemetry"])
            minsrc = min(psc.values()) if psc else None
            # The per-stream ceiling is the analysis_fps cap when set,
            # NOT the fixture rate: with af=5 the probe reports ~5 fps,
            # so 85% of 25 would false-saturate every af row. The floor is
            # 0.75x of target rather than 0.85x to accommodate videorate's
            # PTS quantization of non-integer ratios (~0.84x for af=10 on a
            # 25 fps source) without masking real saturation (GPU util is the
            # primary signal; this is only the per-stream keep-up guard).
            target = (min(fps, args.analysis_fps)
                      if (args.analysis_fps and args.analysis_fps > 0) else fps)
            saturated = (gu is not None and gu >= args.saturate) or \
                        (minsrc is not None and minsrc < 0.75 * target)
            if saturated:
                print(f"  -> n={n} @ {fps}fps SATURATED "
                      f"(gpu%={gu}, min_src_fps={minsrc}); stopping row")
            return saturated

        for fps in fpss:
            for n in ns:
                try:
                    if _run_cell(n, fps):
                        break
                except Exception as exc:
                    # One retry: cells occasionally fail on a teardown/restart
                    # race (observed at low N with recording on) and pass on
                    # the second attempt. A second failure is a real limit.
                    print(f"n={n} fps={fps}: FAILED {exc} (retrying once)", file=sys.stderr)
                    open(SIDECAR_YAML, "w").write(original_yaml())
                    try:
                        if _run_cell(n, fps):
                            break
                    except Exception as exc2:
                        print(f"n={n} fps={fps}: FAILED {exc2}", file=sys.stderr)
                        open(SIDECAR_YAML, "w").write(original_yaml())
                        print(f"  -> n={n} @ {fps}fps pipeline failed; stopping row")
                        break
        # Restore the containers to the original single-source config.
        ensure_shard_containers()
        restart_sidecars()
        print("sweep done; containers restarted on restored yaml")

    elif args.cmd == "summary":
        rows = [json.loads(ln) for ln in open(os.path.join(RESULTS_DIR, "capacity_summary.jsonl"))]
        print(f"{'n':>3} {'fps':>4} {'af':>4} {'sh':>2} {'rec':>4} {'gpu%':>6} {'vramMiB':>8} {'cpu%':>6} {'perSrcFps':>9} sat")
        for r in sorted(rows, key=lambda r: (r.get("shards", 1), r["fps"], r.get("analysis_fps"), r["n"])):
            gu = r["gpu_util"] or {}
            af = r.get("analysis_fps")
            rec = {True: "on", False: "off"}.get(r.get("record"), "-")
            shards = r.get("shards", 1)
            print(f"{r['n']:>3} {r['fps']:>4} {str(af):>4} {shards:>2} {rec:>4} {gu.get('median'):>6} "
                  f"{(r['gpu_mem_mib'] or {}).get('median'):>8} "
                  f"{(r['cpu_pct'] or {}).get('median'):>6} "
                  f"{str(r['median_per_stream_fps']):>9} {r['publisher_saturated_warnings']}")

    elif args.cmd == "telemetry":
        # python3 benchmarks/stream_capacity.py telemetry <cell.json>
        if not args.cell:
            sys.exit("usage: stream_capacity.py telemetry <results/capacity_*.json>")
        with open(args.cell) as f:
            data = json.load(f)
        if not data.get("telemetry"):
            print(f"{args.cell}: no telemetry (cell was not run with --telemetry)")
        else:
            print(f"{args.cell}: n={data.get('n')} fps={data.get('fps')} "
                  f"af={data.get('analysis_fps')} shards={data.get('shards')}")
            print_telemetry_table(data["telemetry"])

    elif args.cmd == "restore":
        if os.path.exists(BACKUP):
            sh(f"cp {BACKUP} {SIDECAR_YAML}")
            print("restored sidecar.yaml from backup")
        stop_pushers()
        ensure_shard_containers()
        names = restart_sidecars()
        for n in names:
            print(f"restarted {n}")


if __name__ == "__main__":
    main()
