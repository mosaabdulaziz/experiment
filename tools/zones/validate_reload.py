"""GPU validation of zones live reload: latency vs the 2 s target, only the owning worker reloads, and whether a ROI
section removed from the nvdsanalytics config stops applying. Opt-in and GPU-only; see tools/zones/README.md.

    uv run python tools/zones/validate_reload.py --check        # preflight only: files, images, GPU, ports
    uv run python tools/zones/validate_reload.py                # the run (needs the GPU to itself)

Two shard replicas of tests/configs/zones-reload-gpu.yaml each run one clip: cam-a on deepstream-sidecar (shard 0),
cam-b on deepstream-sidecar-1 (shard 1). Zones are saved through the app's API exactly as the zone editor does, and
the effect is read from the zones module's roi_probe stream (iris:zones:roi_probe:<camera>: the ROI labels
nvdsanalytics put on each analyzed frame).

Zone ids are positional (z0, z1, ...), so an edit is made observable by alternating two shapes on cam-a:
  A = [full frame]                     -> applied once a frame WITH objects carries "z0" and no "z1"
  B = [tiny corner box, full frame]    -> applied once any object carries "z1" (the full-frame zone is now z1)
Every object is inside the full-frame zone, so either test settles on the first frame with a person in it -- which is
why every timed step first waits for people in cam-a's frames (the clips have long empty stretches).

  a. latency: N edits (A/B alternating), each timed from just before the POST to (1) the owning worker's per-camera
     ack (iris:analytics:zones_applied:cam-a) and (2) the first probe frame showing the new ROIs.
  b. only the owner reloads: shard 1's rendered file is byte-identical and unmodified across the run, its log says
     "analytics config unchanged" for every version and never "reloaded", while it still acknowledges each version.
  c. removed section: cam-a at B (sections present), then all its zones deleted -- shard 0's file is then
     [property] only. Any label on cam-a's objects after the ack means the element kept the removed section live.
     Then A is saved again: its z0 must come back, i.e. the element still reloads a section it dropped.

A timeout in any step (people, ack, effect, the re-save) fails the run: exit 2, like a failed check.

Writes a JSON report (default data/zones-reload/report-<time>.json) and prints a summary. The zones files the app
uses (data/zones_config.txt, data/filter_zones.json) are backed up first and restored on exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = "tests/configs/zones-reload-gpu.yaml"
CAM_A, CAM_B = "cam-a", "cam-b"
OWNER, BYSTANDER = "deepstream-sidecar", "deepstream-sidecar-1"          # shard 0 runs cam-a, shard 1 cam-b
WORKER_FILES = {OWNER: "data/nvdsanalytics_config_worker_single.txt",
                BYSTANDER: "data/nvdsanalytics_config_worker_single_shard1.txt"}
ZONE_FILES = ("data/zones_config.txt", "data/filter_zones.json")
NEEDED_FILES = ("data/uploads/attendance/enter-3.mp4", "data/uploads/attendance/enter-4.mp4",
                "perception/models/DeepStream-Yolo/yolo11s.onnx", "perception/models/DeepStream-Yolo/labels.txt",
                "perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt")
APPLIED = "iris:analytics:zones_applied"
PROBE = "iris:zones:roi_probe:"

FULL = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}]
CORNER = [{"x": 0.0, "y": 0.0}, {"x": 0.02, "y": 0.0}, {"x": 0.02, "y": 0.02}, {"x": 0.0, "y": 0.02}]


def _zone(name, points):
    return {"name": name, "points": points, "intrusion": {"mode": "restricted", "classes": ["person"]}}


SHAPES = {"A": [_zone("full", FULL)], "B": [_zone("corner", CORNER), _zone("full", FULL)]}


def applied(shape: str, labels: set, objects: int) -> bool:
    """Whether a probe frame shows `shape` in effect (see the module docstring). A needs its own label z0 as well as
    the absence of z1: after (c) removes cam-a's section, frames carry no labels at all, and "no z1" alone would call a
    re-saved A applied even if the element never put the section back."""
    if shape == "B":
        return "z1" in labels
    return objects > 0 and "z0" in labels and "z1" not in labels


# B is unambiguous (z1 exists only once B is loaded); A is an absence, so it must hold on this many object frames in a
# row -- one frame whose only person stands on the frame edge must not end the measurement early.
CONFIRM = {"A": 2, "B": 1}


# ── plumbing ─────────────────────────────────────────────────────────────────────────────────────────────────

def compose(*args, env=None, check=True, capture=False):
    cmd = ["docker", "compose", *args]
    out = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}, check=check, text=True,
                         stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    return out.stdout if capture else None


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def save_zones(self, cam, zones):
        req = urllib.request.Request(f"{self.base}/api/streams/{cam}/zones", method="POST",
                                     data=json.dumps({"zones": zones}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
        if "zones_version" not in body:
            raise RuntimeError(f"the save of {cam} announced no zones version: {body}")
        return int(body["zones_version"])


def _int(v):
    return int(v) if v is not None else 0


def _sha(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _mtime(path):
    p = ROOT / path
    return p.stat().st_mtime_ns if p.exists() else None


def watch(r, cam, shape, version, t0, timeout):
    """Follow cam's probe stream and its ack key from just before a save at t0 until `shape` is in effect.
    Returns {"ack_s", "effect_s", "effect_frame_s", "first_object_frame_after_ack_s", "frames", "timed_out"}."""
    stream, key = PROBE + cam, f"{APPLIED}:{cam}"
    last = watch.cursor.get(cam, "$")
    out = {"ack_s": None, "effect_s": None, "effect_frame_s": None, "first_object_frame_after_ack_s": None,
           "frames": 0, "timed_out": False}
    t_ack, run, first_hit = None, 0, None
    while time.time() - t0 < timeout:
        if t_ack is None and _int(r.get(key)) >= version:
            t_ack = time.time()
            out["ack_s"] = round(t_ack - t0, 3)
        for _stream, entries in r.xread({stream: last}, block=20, count=200) or []:
            for entry_id, f in entries:
                last = entry_id
                out["frames"] += 1
                at = float(f.get(b"at", b"0") or 0)
                objects = int(f.get(b"objects", b"0"))
                labels = {x for x in f.get(b"labels", b"").decode().split(",") if x}
                if t_ack is not None and objects and out["first_object_frame_after_ack_s"] is None and at >= t_ack:
                    out["first_object_frame_after_ack_s"] = round(at - t0, 3)
                if at < t0 or not objects:
                    continue
                if not applied(shape, labels, objects):
                    run, first_hit = 0, None
                    continue
                run += 1
                first_hit = first_hit or (time.time(), at)
                if run >= CONFIRM[shape]:
                    out["effect_s"] = round(first_hit[0] - t0, 3)
                    out["effect_frame_s"] = round(first_hit[1] - t0, 3)
                    watch.cursor[cam] = last
                    return out
    watch.cursor[cam] = last
    out["timed_out"] = True
    return out


watch.cursor = {}

# The clips are not full of people (enter-3.mp4 has nobody from ~65 s to its end at 78 s and in its first ~17 s, so a
# ~30 s empty stretch around each loop). Every effect is read off frames with objects, so a step started in an empty
# stretch would time out for want of people, not for want of a reload. Each timed step therefore waits first for
# PEOPLE_RUN object frames in a row, and starts only then.
PEOPLE_RUN = 5


def wait_for_people(r, cam, timeout, start="$"):
    """Block until cam's probe stream shows PEOPLE_RUN consecutive new frames with objects; returns the wait in s.
    Raises TimeoutError if the scene stays empty (longer than one clip loop means the pipeline is not detecting)."""
    stream, last, run, t = PROBE + cam, start, 0, time.time()
    while time.time() - t < timeout:
        for _stream, entries in r.xread({stream: last}, block=200, count=500) or []:
            for entry_id, f in entries:
                last = entry_id
                run = run + 1 if int(f.get(b"objects", b"0")) else 0
                if run >= PEOPLE_RUN:
                    watch.cursor[cam] = last
                    return round(time.time() - t, 2)
    raise TimeoutError(f"timed out after {timeout:.0f} s waiting for people in {cam}'s frames")


def wait_for(what, fn, timeout, every=1.0):
    t = time.time()
    while time.time() - t < timeout:
        try:
            v = fn()
            if v:
                return v
        except Exception:                                   # noqa: BLE001 -- not up yet
            pass
        time.sleep(every)
    raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")


def logs_since(service, since_iso):
    return compose("logs", "--no-color", "--since", since_iso, service, capture=True, check=False) or ""


# the zones module's per-version log line (perception/modules/zones/__init__.py, _on_zones_version)
VERSION_LINE = re.compile(r"zones version (\d+): analytics config (reloaded|unchanged)")


def version_outcomes(log_text: str, versions) -> dict:
    """{version: [(outcome, line), ...]} for each of `versions`, outcome "reloaded" or "unchanged", from a worker's
    log. Lines for any other version (the setup save, cam-b's baseline save) are ignored, so the result does not depend
    on where a log time window happens to start."""
    out = {v: [] for v in versions}
    for line in log_text.splitlines():
        m = VERSION_LINE.search(line)
        if m and int(m.group(1)) in out:
            out[int(m.group(1))].append((m.group(2), line))
    return out


def reload_check(owner_log: str, by_log: str, versions, *, bystander_file_unchanged: bool,
                 bystander_acked_last_version: bool) -> dict:
    """Check (b), counted per edit version: the owner logs exactly one "reloaded" for each edit's version, and the
    bystander logs each of those versions and never "reloaded" for any of them."""
    owner, by = version_outcomes(owner_log, versions), version_outcomes(by_log, versions)
    owner_reloads = {v: sum(o == "reloaded" for o, _ in lines) for v, lines in owner.items()}
    b = {
        "bystander_file_unchanged": bystander_file_unchanged,
        "bystander_reload_lines": [ln for lines in by.values() for o, ln in lines if o == "reloaded"],
        "bystander_version_lines": sum(bool(lines) for lines in by.values()),   # edit versions it logged
        "bystander_acked_last_version": bystander_acked_last_version,
        "owner_reload_lines": sum(owner_reloads.values()),
        "owner_reloads_by_version": owner_reloads,
        "edits": len(versions),
    }
    b["ok"] = (b["bystander_file_unchanged"] and not b["bystander_reload_lines"]
               and b["bystander_version_lines"] == len(versions) and b["bystander_acked_last_version"]
               and all(n == 1 for n in owner_reloads.values()))
    return b


# ── safety guards: a shared machine must not have its running stack or its Redis clobbered ─────────────────────

def running_iris_services(env) -> list:
    """The iris app and any deepstream-sidecar* services already running in the target compose project. The run would
    replace them, so any match refuses it (a second sidecar replica, deepstream-sidecar-2, counts too)."""
    running = (compose("ps", "--status", "running", "--services", env=env, capture=True, check=False) or "").split()
    return sorted(s for s in running if s == "iris" or s.startswith("deepstream-sidecar"))


def has_iris_status(r) -> bool:
    """Whether the Redis holds any iris:status:* key -- the sign that an iris stack is (or was) live on it."""
    return next(iter(r.scan_iter(match="iris:status:*", count=100)), None) is not None


def redis_iris_status_problem(env) -> "str | None":
    """Refuse if the Redis at REDIS_HOST_PORT already holds iris:status:* keys: another iris stack is using it, and
    this run clears the iris namespace at start. An unreachable Redis means nothing to clash with."""
    port = os.environ.get("REDIS_HOST_PORT", "6379")
    try:
        import redis
        r = redis.Redis.from_url(f"redis://localhost:{port}/0", socket_connect_timeout=2)
        if r.ping() and has_iris_status(r):
            return (f"the Redis at REDIS_HOST_PORT={port} already holds iris:status:* keys: another iris stack is "
                    "using it, and this run clears the iris namespace -- point REDIS_HOST_PORT at a Redis of your own")
    except Exception:                                   # noqa: BLE001 -- not reachable here: no clash
        return None
    return None


def scan_delete(r, pattern: str) -> int:
    """Delete every key matching `pattern`, in batches, via SCAN. Returns the count removed."""
    removed, batch = 0, []
    for key in r.scan_iter(match=pattern, count=500):
        batch.append(key)
        if len(batch) >= 500:
            removed += r.delete(*batch)
            batch = []
    if batch:
        removed += r.delete(*batch)
    return removed


def run_key_patterns() -> list:
    """The Redis keys THIS run creates, and only those: the app's zones version, the global and per-camera applied
    acks, and the roi_probe streams for exactly the two cameras it drives. Deliberately narrow -- the pre-run cleanup
    deletes these and nothing else, so another IRIS product's iris:* keys on a shared Redis survive. FLUSHALL wiped the
    whole server, and even a blanket `iris:*` delete would take every product's keys if the running-stack guard ever
    missed a case; this cannot."""
    patterns = ["iris:ctrl:zones_version", APPLIED]
    for cam in (CAM_A, CAM_B):
        patterns += [f"{APPLIED}:{cam}", f"{PROBE}{cam}"]
    return patterns


def clear_run_keys(r) -> int:
    """SCAN-delete only this run's own keys (run_key_patterns), leaving every other key -- including other products'
    iris:* keys -- untouched. Returns the count removed."""
    return sum(scan_delete(r, pattern) for pattern in run_key_patterns())


def compose_project(env) -> str:
    """This run's compose project name: COMPOSE_PROJECT_NAME if set, else compose's default (the project directory
    basename, lowercased with every character outside [a-z0-9_-] removed)."""
    name = env.get("COMPOSE_PROJECT_NAME") or os.environ.get("COMPOSE_PROJECT_NAME")
    return name or re.sub(r"[^a-z0-9_-]", "", ROOT.name.lower())


def redis_container_project(port: str) -> "str | None":
    """The com.docker.compose.project label of the container publishing host `port`, or None -- e.g. an external Redis
    that compose did not start, or no container on that port."""
    try:
        ids = subprocess.run(["docker", "ps", "-q", "--filter", f"publish={port}"],
                             capture_output=True, text=True, check=True).stdout.split()
        for cid in ids:
            label = subprocess.run(
                ["docker", "inspect", "-f", '{{index .Config.Labels "com.docker.compose.project"}}', cid],
                capture_output=True, text=True, check=True).stdout.strip()
            if label:
                return label
    except (OSError, subprocess.CalledProcessError):
        return None
    return None


def redis_ownership_problem(project: str, port: str, owner: "str | None", dbsize: "int | None") -> "str | None":
    """Refuse the run unless the Redis at REDIS_HOST_PORT is this run's OWN redis container -- its
    com.docker.compose.project label is this project -- AND it is empty at start. run_key_patterns includes global keys
    a real stack also writes (iris:ctrl:zones_version, iris:analytics:zones_applied), so even the scoped cleanup must
    only ever touch a Redis that is provably ours and fresh. This ownership + DBSIZE==0 gate is the first layer; the
    scoped delete is the second."""
    if owner != project:
        return (f"the Redis at REDIS_HOST_PORT={port} is not this run's own redis container: it belongs to compose "
                f"project {owner!r}, not {project!r} -- point REDIS_HOST_PORT at a Redis of your own")
    if dbsize:
        return (f"the Redis at REDIS_HOST_PORT={port} is not empty (DBSIZE={dbsize}) at start: this run needs a fresh "
                "Redis -- stop whatever is using it, or start from a clean one")
    return None


def check_redis_ownership(env, r) -> "str | None":
    """Resolve this run's project and the Redis container's owner from docker, then apply redis_ownership_problem."""
    port = os.environ.get("REDIS_HOST_PORT", "6379")
    try:
        dbsize = r.dbsize()
    except Exception:                                   # noqa: BLE001 -- treated as unknown, not empty
        dbsize = None
    return redis_ownership_problem(compose_project(env), port, redis_container_project(port), dbsize)


# ── preflight ────────────────────────────────────────────────────────────────────────────────────────────────

def _parse_int(x) -> "int | None":
    """int(x) or None -- nvidia-smi reports memory as [N/A] on Tegra (integrated GPU), which int() cannot parse."""
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def is_tegra() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


def meminfo_available_mib() -> "int | None":
    """Host MemAvailable in MiB from /proc/meminfo, or None. On Tegra the GPU shares this memory with the CPU."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024          # kB -> MiB
    except (OSError, ValueError, IndexError):
        return None
    return None


def memory_problems(smi_text, need_mib: int, tegra: bool, meminfo_mib: "int | None") -> "tuple[list, str]":
    """Decide the preflight memory problems from nvidia-smi output (index,name,memory.used,memory.total CSV), whether
    this is a Tegra, and the host MemAvailable. Returns (problems, source). Never raises on an unparsable or [N/A]
    memory field: a [N/A] GPU figure on Tegra falls back to MemAvailable (shared memory), elsewhere it is reported."""
    frees = []
    for line in (smi_text or "").strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        used, total = _parse_int(parts[2]), _parse_int(parts[3])
        frees.append(None if used is None or total is None else total - used)
    parseable = [(i, f) for i, f in enumerate(frees) if f is not None]
    if parseable:
        return ([f"GPU {i} has {f} MiB free; the run wants ~{need_mib} MiB" for i, f in parseable if f < need_mib],
                "nvidia-smi")
    # nvidia-smi gave no usable figure (e.g. [N/A] on Tegra, or produced no output).
    if tegra:
        source = "/proc/meminfo MemAvailable (Tegra: the GPU shares host memory with the CPU)"
        if meminfo_mib is None:
            return ["could not read MemAvailable from /proc/meminfo"], source
        if meminfo_mib < need_mib:
            return [f"only {meminfo_mib} MiB available (MemAvailable); the run wants ~{need_mib} MiB"], source
        return [], source
    return (["nvidia-smi reported no parseable GPU memory and this is not a Tegra; cannot check free memory"],
            "nvidia-smi")


def check_memory(need_mib: int = 4096) -> list:
    """Preflight's memory check: query nvidia-smi, fall back to host MemAvailable on Tegra, print the source used, and
    return any problems. Standalone-callable so it can be run on its own (tools/zones/README.md)."""
    try:
        smi_text = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        if not is_tegra():
            return ["nvidia-smi failed: no GPU visible here"]
        smi_text = ""                                   # Tegra without nvidia-smi: fall through to MemAvailable
    if smi_text.strip():
        print("GPU:", smi_text.strip().replace("\n", " | "))
    problems, source = memory_problems(smi_text, need_mib, is_tegra(), meminfo_available_mib())
    print(f"memory check source: {source}")
    return problems


def preflight(env) -> list:
    problems = []
    for f in NEEDED_FILES:
        if not (ROOT / f).exists():
            problems.append(f"missing {f} (`just data-pull` / dvc pull it)")
    images = (compose("config", "--images", env=env, capture=True, check=False) or "").split()
    for image in {i for i in images if "deepstream-sidecar" in i or i.endswith("-iris") or "-iris:" in i}:
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
            problems.append(f"image {image} is not built (docker compose build {'iris' if 'iris' in image and 'sidecar' not in image else 'deepstream-sidecar'})")
    clash = running_iris_services(env)
    if clash:
        problems.append(f"compose project {env.get('COMPOSE_PROJECT_NAME') or '(default)'} already runs {clash}: "
                        "this run would replace them -- use a separate COMPOSE_PROJECT_NAME (README)")
    redis_problem = redis_iris_status_problem(env)
    if redis_problem:
        problems.append(redis_problem)
    problems += check_memory()
    return problems


# ── the run ──────────────────────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="preflight only; start nothing")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--target", type=float, default=2.0, help="seconds from save to effect")
    ap.add_argument("--timeout", type=float, default=15.0, help="per edit")
    ap.add_argument("--settle", type=float, default=1.5, help="pause between edits")
    ap.add_argument("--observe", type=float, default=10.0, help="how long (c) watches for a stale section")
    ap.add_argument("--boot-timeout", type=float, default=900.0, help="per worker; the first boot builds the engine")
    ap.add_argument("--people-timeout", type=float, default=120.0,
                    help="how long a step waits for people in frame before it starts (one clip loop is 78 s)")
    ap.add_argument("--stop-timeout", type=int, default=30,
                    help="seconds `docker compose stop` gives each service to exit before killing it")
    ap.add_argument("--keep-up", action="store_true", help="leave the app and both workers running afterwards")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = {"SIDECAR_CONFIG": CONFIG, "IRIS_SHARD_COUNT": "2"}
    problems = preflight(env)
    if problems or args.check:
        print("\n".join(f"  - {p}" for p in problems) or "preflight: ok")
        return 1 if problems else 0

    import redis                                           # the project environment has it (uv run)
    api = Api(f"http://localhost:{os.environ.get('IRIS_HOST_PORT', '18000')}")
    r = redis.Redis.from_url(f"redis://localhost:{os.environ.get('REDIS_HOST_PORT', '6379')}/0")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = ROOT / "data" / "zones-reload" / f"backup-{stamp}"
    report = {"started": stamp, "config": CONFIG, "iterations": args.iterations, "target_s": args.target}
    out_path = pathlib.Path(args.out) if args.out else ROOT / "data" / "zones-reload" / f"report-{stamp}.json"

    backup.mkdir(parents=True)
    saved = [f for f in ZONE_FILES if (ROOT / f).exists()]
    for f in saved:
        shutil.copy2(ROOT / f, backup / pathlib.Path(f).name)
    try:
        for f in ZONE_FILES:                               # start from no zones at all
            (ROOT / f).unlink(missing_ok=True)

        # stack: redis, an empty Redis, the app, then shard 0 alone (it builds the TensorRT engine on a cold machine)
        # and shard 1 once that is done, reusing the cached engine. No MinIO: the config records nothing and both the
        # app (config.yaml segments.storage_backend: local) and the sidecar store segments locally, so every service
        # starts with --no-deps (compose would otherwise start the minio service they declare a dependency on).
        compose("up", "-d", "--no-build", "--no-deps", "redis", env=env)
        wait_for("redis to answer", r.ping, 30, 0.5)
        owned = check_redis_ownership(env, r)           # first layer: it must be OUR redis container and empty...
        if owned:
            raise RuntimeError(owned)
        removed = clear_run_keys(r)                     # ...second layer: delete only this run's own keys
        print(f"cleared {removed} leftover key(s) from a prior run")
        compose("up", "-d", "--no-build", "--no-deps", "iris", env=env)
        t_boot = time.time()
        for service, cam in ((OWNER, CAM_A), (BYSTANDER, CAM_B)):
            compose("--profile", "shards", "up", "-d", "--no-build", "--no-deps", service, env=env)
            wait_for(f"{service} to analyze {cam}", lambda cam=cam: r.xlen(PROBE + cam) > 0, args.boot_timeout, 2.0)
            print(f"{service}: {cam} is being analyzed ({time.time() - t_boot:.0f} s)")
        report["boot_s"] = round(time.time() - t_boot, 1)
        wait_for("the app's zones status", lambda: api.get("/api/cameras").get("zones", {}).get("enabled"), 120)

        # baseline: cam-b gets a zone of its own (shard 1 then has a section to keep), cam-a starts at A
        v = api.save_zones(CAM_B, SHAPES["A"])
        wait_for("cam-b's zones to apply", lambda: _int(r.get(f"{APPLIED}:{CAM_B}")) >= v, 30, 0.1)
        def step(shape, timeout):
            """Wait for people, save `shape` on cam-a, watch it apply; the result carries the people wait."""
            waited = wait_for_people(r, CAM_A, args.people_timeout)
            t0 = time.time()
            v = api.save_zones(CAM_A, SHAPES[shape])
            res = watch(r, CAM_A, shape, v, t0, timeout)
            res.update(shape=shape, version=v, people_wait_s=waited)
            return res

        first = step("A", 60)
        if first["timed_out"]:
            raise RuntimeError(f"cam-a never showed its first zone: {first}")
        time.sleep(args.settle)

        # (a) + (b)
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1))
        by_sha, by_mtime = _sha(WORKER_FILES[BYSTANDER]), _mtime(WORKER_FILES[BYSTANDER])
        runs, versions = [], []
        for i in range(args.iterations):
            shape = "B" if i % 2 == 0 else "A"
            res = step(shape, args.timeout)
            res.update(i=i)
            runs.append(res)
            versions.append(res["version"])
            print(f"edit {i + 1}/{args.iterations} ({shape}, v{res['version']}): people after {res['people_wait_s']} s,"
                  f" ack {res['ack_s']} s, effect {res['effect_s']} s" + (" TIMEOUT" if res["timed_out"] else ""))
            time.sleep(args.settle)
        effects = [x["effect_s"] for x in runs if x["effect_s"] is not None]
        acks = [x["ack_s"] for x in runs if x["ack_s"] is not None]

        def _stats(xs):
            return ({"min": min(xs), "median": round(statistics.median(xs), 3), "max": max(xs)} if xs else None)

        report["a_latency"] = {"runs": runs, "ack_s": _stats(acks), "effect_s": _stats(effects),
                               "timeouts": sum(x["timed_out"] for x in runs),
                               "meets_target": bool(effects) and len(effects) == len(runs)
                               and max(effects) <= args.target}

        time.sleep(2)                                       # let the last log lines land
        # the window only bounds the log volume: the count is per edit version, so a line it catches from the setup
        # save (or the baseline cam-b save) is never counted
        owner_log, by_log = logs_since(OWNER, since), logs_since(BYSTANDER, since)
        report["b_only_owner_reloads"] = reload_check(
            owner_log, by_log, versions,
            bystander_file_unchanged=_sha(WORKER_FILES[BYSTANDER]) == by_sha
            and _mtime(WORKER_FILES[BYSTANDER]) == by_mtime,
            bystander_acked_last_version=_int(r.get(f"{APPLIED}:{CAM_B}")) >= versions[-1])

        # (c) removed section
        before = step("B", args.timeout)
        if before["timed_out"]:
            raise RuntimeError(f"(c) cam-a never showed B before the removal: {before}")
        time.sleep(args.settle)
        removal_people_wait = wait_for_people(r, CAM_A, args.people_timeout)   # the observation reads object frames
        v = api.save_zones(CAM_A, [])                      # every zone of cam-a deleted: its section goes
        wait_for("the removal's ack", lambda: _int(r.get(f"{APPLIED}:{CAM_A}")) >= v, args.timeout, 0.05)
        t_ack = time.time()
        owner_text = (ROOT / WORKER_FILES[OWNER]).read_text()
        stale, clean, last = 0, 0, watch.cursor.get(CAM_A, "$")
        stale_labels: set = set()
        while time.time() - t_ack < args.observe:
            for _s, entries in r.xread({PROBE + CAM_A: last}, block=100, count=500) or []:
                for entry_id, f in entries:
                    last = entry_id
                    if float(f.get(b"at", b"0") or 0) < t_ack + 0.5 or not int(f.get(b"objects", b"0")):
                        continue                            # in flight at the reload, or nobody in the frame
                    labels = {x for x in f.get(b"labels", b"").decode().split(",") if x}
                    stale += bool(labels)
                    clean += not labels
                    stale_labels |= labels
        watch.cursor[CAM_A] = last
        after = step("A", args.timeout)                     # the re-save: z0 must come back (applied() needs it)
        report["c_removed_section"] = {
            "sections_before_applied": not before["timed_out"],
            "removal_people_wait_s": removal_people_wait,
            "owner_file_after_removal": owner_text,
            "frames_with_objects_after_ack": stale + clean,
            "frames_still_labelled": stale, "stale_labels": sorted(stale_labels),
            "verdict": ("no frames with objects observed: inconclusive" if stale + clean == 0 else
                        "REMOVED SECTION STAYS LIVE" if stale else "removed section stops applying"),
            "reload_after_removal": after,
            "resaved_zones_come_back": not after["timed_out"],
        }
    except (TimeoutError, RuntimeError) as e:              # a step that never completed: a failed check, not a crash
        report["error"] = f"{type(e).__name__}: {e}"
    finally:
        for f in ZONE_FILES:
            (ROOT / f).unlink(missing_ok=True)
        for f in saved:
            shutil.copy2(backup / pathlib.Path(f).name, ROOT / f)
        if not args.keep_up:
            compose("--profile", "shards", "stop", "-t", str(args.stop_timeout), OWNER, BYSTANDER, "iris",
                    env=env, check=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"report: {out_path}")

    return summarize(report, args.target)


def summarize(report: dict, target: float) -> int:
    """Print the verdicts; 0 only if every step completed (no timeout anywhere) and all three checks pass, else 2."""
    if "error" in report:
        print(f"\nFAILED, the run did not complete: {report['error']}")
        return 2
    a, b, c = report["a_latency"], report["b_only_owner_reloads"], report["c_removed_section"]
    after = c["reload_after_removal"]
    print(f"\n(a) save -> effect: {a['effect_s']} s (target {target} s: {'met' if a['meets_target'] else 'NOT met'});"
          f" save -> ack: {a['ack_s']} s; timeouts {a['timeouts']}")
    print(f"(b) only the owner reloads: {'yes' if b['ok'] else 'NO'} (owner reloads {b['owner_reload_lines']}/"
          f"{b['edits']}, bystander reloads {len(b['bystander_reload_lines'])}, file unchanged "
          f"{b['bystander_file_unchanged']})")
    print(f"(c) {c['verdict']} ({c['frames_still_labelled']}/{c['frames_with_objects_after_ack']} frames labelled"
          f" {c['stale_labels']}); re-saved zones come back: "
          + (f"yes (ack {after['ack_s']} s, effect {after['effect_s']} s)" if c["resaved_zones_come_back"]
             else f"NO (timed out; ack {after['ack_s']} s, {after['frames']} frames)"))
    ok = (a["meets_target"] and a["timeouts"] == 0 and b["ok"]
          and c["verdict"] == "removed section stops applying" and c["resaved_zones_come_back"])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
