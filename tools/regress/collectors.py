"""The collectors: run the existing tools and turn what they leave behind into the numbers of one results file.

Nothing here measures anything new; it wires together what the repo already has:
  unit     pytest with --junitxml, as CI's `core-tests` runs it (and the barber product's own suite)
  golden   tools/engine_golden_run.sh (replay a config on its clip, record what it publishes), tools/engine_golden.py diff,
           and the sidecar's own telemetry through tools/engine_gpu_budget.py: GPU % per stage, fps, drops, boot time,
           whether TensorRT rebuilt an engine, the GPU memory it peaked at
  build    image sizes, the frontend bundle, the number of packages in every uv.lock

The golden replays run in their own compose project (`regress`) on their own host ports, so a stack that is already
running is never touched, and the project is removed at the end.
"""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import gpucounters, machine, parsers, soak

GOLDEN_CONFIGS = ["attendance", "zones", "fire", "multicam", "barber", "barber-faces"]
# Replayed by default only when its clip and baseline are in the checkout: both are DVC files `just data-pull` fetches
# (the `golden` group), and a machine that has not pulled them should still be able to run the other six.
OPTIONAL_GOLDEN = {"proximity": ("data/uploads/golden/proximity.mp4", "tests/baselines/golden-proximity.before.jsonl")}


def default_golden_configs(repo: str) -> tuple:
    """(configs to replay by default, {config: why it was left out})."""
    names, skipped = list(GOLDEN_CONFIGS), {}
    for name, files in OPTIONAL_GOLDEN.items():
        missing = [f for f in files if not os.path.exists(os.path.join(repo, f))]
        if missing:
            skipped[name] = "not pulled: " + ", ".join(missing) + " (just data-pull golden)"
        else:
            names.append(name)
    return names, skipped
# the comparison modes tools/engine_golden_check.sh uses (and TESTING.md documents for proximity)
SEMANTIC = {"fire", "barber", "barber-faces", "proximity"}
TOLERANCE = {"barber": "0.08", "barber-faces": "0.08"}
MTMC_CONFIGS = {"multicam"}              # needs the sidecar image that carries CUDA torch
OK_STATUS = ("IDENTICAL", "SEMANTICALLY EQUAL")
PROJECT = "regress"
PORTS = {"REDIS_HOST_PORT": "6391", "MINIO_API_HOST_PORT": "19100", "MINIO_CONSOLE_HOST_PORT": "19101",
         "SIDECAR_TELEMETRY_PORT": "9191"}


def sh(cmd, cwd=None, env=None, timeout=None) -> subprocess.CompletedProcess:
    """Run a command; a tool that is not installed (docker in CI's python image) is exit 127, not an exception, so the
    section reports "not available" and the rest of the run goes on."""
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", f"{cmd[0]}: not found ({exc})")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── unit suites ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _pytest_suite(cmd: list, cwd: str, junit: str) -> dict:
    t0 = time.time()
    r = sh(cmd, cwd=cwd)
    wall = round(time.time() - t0, 1)
    if not os.path.exists(junit):
        return {"ran": False, "error": (r.stdout + r.stderr)[-400:], "wall_s": wall, "exit_code": r.returncode}
    with open(junit) as f:
        out = parsers.parse_junit(f.read())
    out.update({"ran": True, "wall_s": wall, "exit_code": r.returncode})
    return out


def collect_unit(repo: str, suites=("core", "barber"), junit_files=None) -> dict:
    """{suite: {tests, failures, errors, skipped, time_s, failed, slowest, wall_s}}. `core` is CI's core-tests job.
    `junit_files` {suite: path} reads a junit XML that already exists (CI's core-tests artifact) instead of running it."""
    out = {}
    for name, path in (junit_files or {}).items():
        with open(path) as f:
            out[name] = {**parsers.parse_junit(f.read()), "ran": True, "source": "junit file", "exit_code": None}
        out[name]["wall_s"] = out[name]["time_s"]
    suites = [s for s in suites if s not in out]
    tmp = tempfile.mkdtemp(prefix="regress-unit-")
    try:
        if "core" in suites:
            junit = os.path.join(tmp, "core.xml")
            out["core"] = _pytest_suite([sys.executable, "-m", "pytest", "tests", "test_streamer", "-q",
                                         "-p", "no:cacheprovider", f"--junitxml={junit}"], repo, junit)
        for name in [s for s in suites if s != "core"]:
            product = os.path.join(repo, "products", name)
            junit = os.path.join(tmp, f"{name}.xml")
            if not os.path.isdir(product):
                out[name] = {"ran": False, "error": f"no such product folder products/{name}"}
            elif not shutil.which("uv"):
                out[name] = {"ran": False, "error": "uv is not installed (the product has its own environment)"}
            else:
                env_cmd = ["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           f"--junitxml={junit}"]
                out[name] = _pytest_suite(env_cmd, product, junit)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ── build and size ───────────────────────────────────────────────────────────────────────────────────────────────────

def collect_build(repo: str, images=None, build_frontend=True) -> dict:
    """Image sizes, frontend bundle, uv.lock package counts. Machine independent (given the same base image)."""
    out = {"images": {}, "frontend_bundle": None, "lock_packages": {}}
    for img in images or []:
        r = sh(["docker", "image", "inspect", img, "--format", "{{.Size}}"])
        out["images"][img] = int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None
    front = os.path.join(repo, "frontend")
    dist = os.path.join(front, "dist")
    if build_frontend and os.path.isdir(os.path.join(front, "node_modules")) and shutil.which("npm"):
        t0 = time.time()
        r = sh(["npm", "run", "build"], cwd=front)
        out["frontend_build_s"] = round(time.time() - t0, 1)
        if r.returncode != 0:
            out["frontend_bundle"] = {"error": (r.stdout + r.stderr)[-300:]}
            dist = ""
    if dist and os.path.isdir(dist):
        out["frontend_bundle"] = parsers.bundle_summary(parsers.dir_files(dist))
    elif out["frontend_bundle"] is None:
        out["frontend_bundle"] = {"skipped": "frontend/dist is not built and frontend/node_modules is missing"}
    for lock in sorted(glob.glob(os.path.join(repo, "**", "uv.lock"), recursive=True)):
        rel = os.path.relpath(lock, repo)
        if any(part in rel.split(os.sep) for part in ("node_modules", ".venv", "helper-repos")):
            continue
        with open(lock) as f:
            out["lock_packages"][rel] = parsers.count_lock_packages(f.read())
    return out


# ── golden replays: correctness and runtime ──────────────────────────────────────────────────────────────────────────

class MemorySampler(threading.Thread):
    """The highest GPU memory in use while the replay ran (once a second), and what was in use before it started."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.before = machine.gpu_sample()["mem_used_mib"]
        self.peak = self.before
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(1.0):
            used = machine.gpu_sample()["mem_used_mib"]
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used

    def finish(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        return {"before_mib": self.before, "peak_mib": self.peak,
                "added_mib": (self.peak - self.before) if None not in (self.peak, self.before) else None}


class ResourceSampler(threading.Thread):
    """The replay's sidecar container, sampled every few seconds with the soak sampler's own readers (cgroup memory split, CPU time, threads,
    fds): so the numbers are the ones `regress soak` records, taken only while the replay runs."""

    def __init__(self, interval: float = 3.0, sampler=None) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.rows = []
        self._halt = threading.Event()
        self._sampler = sampler or soak.Sampler(soak.SoakConfig(project=PROJECT, redis_container=""), soak.RealIO())

    def run(self) -> None:
        while not self._halt.is_set():
            now = time.time()
            try:
                containers, _ = self._sampler._containers(now)
                for name, c in containers.items():
                    if "deepstream-sidecar" in name and c.get("cgroup"):
                        self.rows.append((now, c))
            except Exception:                                  # noqa: BLE001 -- a failed sample is a missing sample, never a failed replay
                pass
            self._halt.wait(self.interval)

    def finish(self, lo: float = None, hi: float = None) -> dict:
        self._halt.set()
        self.join(timeout=10)
        return summarize_resources(self.rows, lo, hi)


def summarize_resources(rows: list, lo: float = None, hi: float = None) -> dict:
    """[(t, container entry)] -> the replay's CPU cores (time-averaged over the window), peak anon and cgroup memory (MiB), threads, fds."""
    rs = [(t, c) for t, c in rows if lo is None or lo <= t <= hi]
    if len(rs) < 2:
        return {}
    cpu = [(t, c["cgroup"].get("cpu_usec")) for t, c in rs if c["cgroup"].get("cpu_usec") is not None]
    out = {}
    if len(cpu) >= 2 and cpu[-1][0] > cpu[0][0]:
        out["cpu_cores"] = round((cpu[-1][1] - cpu[0][1]) / 1e6 / (cpu[-1][0] - cpu[0][0]), 3)
    anon = [c["cgroup"].get("anon") for _, c in rs if c["cgroup"].get("anon") is not None]
    peak = [c["cgroup"].get("peak_bytes") for _, c in rs if c["cgroup"].get("peak_bytes") is not None]
    main = [(c.get("procs") or {}).get("main") or {} for _, c in rs]
    if anon:
        out["rss_anon_mib"] = round(max(anon) / 2 ** 20, 1)
    if peak:
        out["cgroup_peak_mib"] = round(max(peak) / 2 ** 20, 1)
    for key, src in (("threads", "threads"), ("fds", "fds")):
        vals = [m.get(src) for m in main if m.get(src) is not None]
        if vals:
            out[key] = max(vals)
    out["samples"] = len(rs)
    return out


def golden_env() -> dict:
    """The environment of a replay. tools/engine_golden_run.sh starts its recorder as plain `python3`, which must be the
    interpreter that has `redis`: on a machine where the harness runs from a virtualenv (the Orin), the system python3 has
    none and the recorder died on `import redis`. So the harness's own interpreter comes first on PATH."""
    path = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + os.environ.get("PATH", "")
    return {**os.environ, "PATH": path, "COMPOSE_PROJECT_NAME": PROJECT, **PORTS,
            "GOLDEN_IDLE": os.environ.get("GOLDEN_IDLE", "40"), "GOLDEN_MAX": os.environ.get("GOLDEN_MAX", "600")}


def diff_vs_repo_baseline(repo: str, name: str, after: str) -> dict:
    """tools/engine_golden.py diff against tests/baselines/golden-<name>.before.jsonl, as engine_golden_check.sh does.
    That baseline was recorded on the GPU server; on another GPU it can differ without anything being wrong, so this is
    information, not the gate (the gate is the per-machine baseline, task #228)."""
    before = os.path.join(repo, "tests", "baselines", f"golden-{name}.before.jsonl")
    if not os.path.exists(before):
        return {"status": "no baseline", "detail": f"{before} is not in the checkout (just data-pull)"}
    return diff_outputs(repo, name, before, after)


def diff_outputs(repo: str, name: str, before: str, after: str) -> dict:
    """tools/engine_golden.py diff of two recorded outputs of golden config `name`, in the mode that config is compared in."""
    cmd = [sys.executable, os.path.join(repo, "tools", "engine_golden.py"), "diff"]
    if name in SEMANTIC:
        cmd.append("--semantic")
    if name in TOLERANCE:
        cmd += ["--tol", TOLERANCE[name]]
    r = sh(cmd + [before, after], cwd=repo)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    status = next((l for l in reversed(lines) if l in OK_STATUS or "difference" in l), lines[-1] if lines else "no output")
    return {"status": status, "mode": "semantic" if name in SEMANTIC else "exact", "ok": status in OK_STATUS,
            "failing": [l for l in lines if l.startswith("FAIL")][:6]}


def _replay(repo, name, out_dir, image, env, budget_parse, counters) -> dict:
    """One golden config: replay it, record what it published, read the sidecar log, take the counter window."""
    cfg = os.path.join("tests", "configs", f"golden-{name}.yaml")      # relative: the container mounts the repo at /workspace/iris
    sh(["docker", "tag", image, f"{PROJECT}-deepstream-sidecar:latest"])
    out_file = os.path.join(out_dir, f"{name}.jsonl")
    if os.path.exists(out_file):
        os.unlink(out_file)
    mem = MemorySampler()
    mem.start()
    res = ResourceSampler()
    res.start()
    t0 = time.time()
    try:
        run = sh(["tools/engine_golden_run.sh", cfg, out_file], cwd=repo, env=env, timeout=int(env["GOLDEN_MAX"]) + 180)
    except subprocess.TimeoutExpired:
        run = subprocess.CompletedProcess([], 124, "", f"the replay did not finish in {env['GOLDEN_MAX']} s")
    wall = round(time.time() - t0, 1)
    memory = mem.finish()
    logs = sh(["docker", "compose", "logs", "--no-color", "--timestamps", "deepstream-sidecar"], cwd=repo, env=env)
    entry = {"ran": run.returncode == 0 and os.path.exists(out_file), "wall_s": wall, "image": image, "gpu_memory": memory}
    if not entry["ran"]:
        entry["error"] = (run.stdout + run.stderr)[-500:]
        res.finish()
        return entry
    with open(out_file) as f:
        entry["entries"] = sum(1 for _ in f)
    entry["output_sha256"] = sha256_file(out_file)
    entry["output_file"] = os.path.relpath(out_file, repo) if out_file.startswith(repo) else out_file
    entry["runtime"] = parsers.parse_runtime(logs.stdout + logs.stderr, budget_parse)
    lo_w, hi_w = entry["runtime"]["steady_window"] or (None, None)
    entry["runtime"]["resources"] = res.finish(lo_w, hi_w)
    if counters is not None:
        # only while the pipeline ran (first fps line to end of stream), not the recorder's idle tail
        lo, hi = entry["runtime"]["steady_window"] or (t0, t0 + wall)
        entry["gpu_counters"] = counters.window(lo, hi)
        entry["gpu_counters"]["hints"] += gpucounters.stage_hints(entry["gpu_counters"]["dcgm"], entry["runtime"]["gpu_stages"])
    entry["vs_repo_baseline"] = diff_vs_repo_baseline(repo, name, out_file)
    return entry


def collect_golden(repo: str, names, out_dir: str, slim_image: str, mtmc_image: str, budget_parse, counters=None) -> dict:
    """For each golden config: replay it on its clip, record what it published, read the sidecar log for the cost. A replay
    whose DCGM profiling counters came back invalid is run once more on a fresh exporter (and says so)."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    env = golden_env()
    try:
        for name in names:
            if not os.path.exists(os.path.join(repo, "tests", "configs", f"golden-{name}.yaml")):
                results[name] = {"ran": False, "error": f"no tests/configs/golden-{name}.yaml"}
                continue
            image = mtmc_image if name in MTMC_CONFIGS else slim_image
            if sh(["docker", "image", "inspect", image]).returncode != 0:
                results[name] = {"ran": False, "error": f"image {image} is not on this machine"}
                continue
            entry = _replay(repo, name, out_dir, image, env, budget_parse, counters)
            gcn = entry.get("gpu_counters")
            if gcn is not None and not gcn.get("dcgm_valid", True) and counters is not None:
                first_reason = gcn["dcgm_invalid_reason"]
                counters.restart_dcgm()                                  # a fresh profiling session, then the same replay again
                entry = _replay(repo, name, out_dir, image, env, budget_parse, counters)
                if entry.get("gpu_counters") is not None:
                    entry["gpu_counters"]["retried"] = {"first_attempt": first_reason}
            results[name] = entry
    finally:
        sh(["docker", "compose", "down", "-v"], cwd=repo, env=env)
    return results
