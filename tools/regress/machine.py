"""Which machine a result came from, and whether that machine was idle while it ran (stdlib only).

The fingerprint says what a number was measured on (GPU, driver, CUDA, CPU, RAM, git commit and whether the tree was
dirty, which sidecar image); the machine_id is the key a baseline is stored under (the ids of issue #140). The idle check
is the set of guardrails of issue #149 that a host can check: a GPU another process is using makes every efficiency
number worth less, so the run is refused, or recorded as NOISY. (The open-files limit of #149 is the sidecar container's
own, set by docker-compose, so it is not a host check.)

x86 machines are read through nvidia-smi; a Jetson through tegrastats and sysfs (no nvidia-smi there).
"""
import os
import platform
import re
import subprocess
import threading

# first match wins; the ids are the ones of issue #140
MACHINE_IDS = [
    (re.compile(r"RTX PRO 6000", re.I), "rtx-pro-6000-maxq"),
    (re.compile(r"RTX 5090", re.I), "rtx-5090"),
    (re.compile(r"RTX 5050", re.I), "rtx-5050-laptop"),
    (re.compile(r"Orin", re.I), "jetson-agx-orin-64gb"),
]
IDLE_GPU_UTIL_PCT = 10.0          # a GPU busier than this before the run is not idle (REGRESS_IDLE_GPU_UTIL overrides it)
BACKGROUND_SAMPLES = 5            # readings, one second apart: the median is the background load


def run(*cmd: str, timeout: float = 15) -> str:
    """stdout of a command, '' when it is missing or fails (this module only describes; it never raises)."""
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_id(gpu_name: str) -> str:
    env = os.environ.get("IRIS_MACHINE_ID")
    if env:
        return env
    if not gpu_name and os.environ.get("CI"):
        return "ci-cpu"                       # a CI job: no GPU, and the numbers are filed apart from every real machine
    for pattern, ident in MACHINE_IDS:
        if pattern.search(gpu_name or ""):
            return ident
    return re.sub(r"[^a-z0-9]+", "-", (gpu_name or "unknown-gpu").lower()).strip("-")


def parse_tegrastats(line: str) -> dict:
    """`RAM 2115/62828MB ... GR3D_FREQ 0% cpu@39.75C ...` -> {ram_used_mb, ram_total_mb, gpu_util_pct, temps_c}."""
    out = {}
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        out["ram_used_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        out["gpu_util_pct"] = int(m.group(1))
    out["temps_c"] = {k: float(v) for k, v in re.findall(r"(\w+)@(-?[\d.]+)C", line)}
    return out


def parse_compute_apps(csv_text: str) -> list:
    """`pid, process_name, used_memory` rows of nvidia-smi --query-compute-apps."""
    apps = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "mem_mib": int(re.sub(r"\D", "", parts[2]) or 0) if len(parts) > 2 else 0})
    return apps


def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or "tegra" in platform.release()


def _tegrastats_line(timeout: float = 5.0) -> str:
    """The first line tegrastats prints. It streams until killed, so it is read from a process, not run to the end."""
    try:
        p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return ""
    try:
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        return p.stdout.readline().strip()
    finally:
        timer.cancel()
        p.kill()
        p.wait()


def gpu_sample() -> dict:
    """One reading of the GPU: {util_pct, mem_used_mib, mem_total_mib, source}."""
    if is_jetson():
        t = parse_tegrastats(_tegrastats_line())
        return {"util_pct": t.get("gpu_util_pct"), "mem_used_mib": t.get("ram_used_mb"),
                "mem_total_mib": t.get("ram_total_mb"), "source": "tegrastats (unified memory)"}
    row = run("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits").strip()
    parts = [p.strip() for p in row.splitlines()[0].split(",")] if row else []
    if len(parts) == 3 and all(p.replace(".", "").isdigit() for p in parts):
        return {"util_pct": float(parts[0]), "mem_used_mib": int(float(parts[1])), "mem_total_mib": int(float(parts[2])),
                "source": "nvidia-smi"}
    return {"util_pct": None, "mem_used_mib": None, "mem_total_mib": None, "source": "unavailable"}


def _first(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip().strip("\x00")
    except OSError:
        return ""


def fingerprint(repo: str) -> dict:
    """What a result was measured on and from."""
    jetson = is_jetson()
    if jetson:
        gpu = _first("/proc/device-tree/model") or "NVIDIA Jetson"
        rel = re.search(r"R(\d+) .*REVISION: ([\d.]+)", _first("/etc/nv_tegra_release"))
        driver = f"L4T R{rel.group(1)}.{rel.group(2)}" if rel else _first("/etc/nv_tegra_release")[:60]
        power = run("nvpmodel", "-q").strip().splitlines()
        extra = {"power_mode": power[0].replace("NV Power Mode: ", "") if power else None}
        mem = None
    else:
        smi = run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").strip()
        parts = [p.strip() for p in smi.splitlines()[0].split(",")] if smi else ["", "", ""]
        gpu, mem, driver = parts[0], (int(float(parts[1])) if parts[1:2] and parts[1].replace(".", "").isdigit() else None), \
            (parts[2] if len(parts) > 2 else "")
        cuda = re.search(r"CUDA Version: ([\d.]+)", run("nvidia-smi"))
        extra = {"cuda": cuda.group(1) if cuda else None, "vram_mib": mem}
    cpu = next((l.split(":", 1)[1].strip() for l in _first("/proc/cpuinfo").splitlines() if l.startswith("model name")), "") \
        or platform.processor()
    mem_kb = next((int(re.sub(r"\D", "", l)) for l in _first("/proc/meminfo").splitlines() if l.startswith("MemTotal")), 0)
    git = lambda *a: run("git", "-C", repo, *a).strip()      # noqa: E731
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {
        "machine_id": machine_id(gpu), "hostname": platform.node(), "arch": platform.machine(), "jetson": jetson,
        "gpu": gpu, "driver": driver, **extra, "cpu": cpu, "cpu_count": os.cpu_count(), "ram_gb": round(mem_kb / 1048576, 1),
        "os": platform.platform(), "python": platform.python_version(),
        "git": {"commit": git("rev-parse", "HEAD"), "short": git("rev-parse", "--short", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty},
    }


def on_battery() -> bool:
    for p in ("BAT0", "BAT1"):
        s = _first(f"/sys/class/power_supply/{p}/status")
        if s:
            return s.lower() == "discharging"
    return False


def idle_threshold() -> float:
    try:
        return float(os.environ.get("REGRESS_IDLE_GPU_UTIL", IDLE_GPU_UTIL_PCT))
    except ValueError:
        return IDLE_GPU_UTIL_PCT


def background_util(samples: int = BACKGROUND_SAMPLES, interval: float = 1.0, sample=None, sleep=None):
    """The median GPU utilisation over a few seconds: one reading is too easily a moment of someone's video call."""
    import statistics
    import time
    sample, sleep = sample or gpu_sample, sleep or time.sleep
    xs = []
    for i in range(samples):
        u = sample()["util_pct"]
        if u is not None:
            xs.append(float(u))
        if i + 1 < samples:
            sleep(interval)
    return statistics.median(xs) if xs else None


def idle_report(own_prefix: str = "regress", background=None, source: str = "GPU utilisation") -> dict:
    """{idle, reasons, gpu, foreign_compute, other_stacks}: the guardrails of #149 that this machine can check.
    `own_prefix` is the compose project the harness itself uses; its containers are not "other stacks"."""
    reasons = []
    gpu = gpu_sample()
    limit = idle_threshold()
    util = background_util() if background is None else background
    gpu["background_util_pct"], gpu["idle_limit_pct"], gpu["background_source"] = util, limit, source
    if util is None:
        reasons.append("GPU load unreadable")
    elif util > limit:
        reasons.append(f"GPU is {util:.0f}% busy before the run ({source}; limit {limit:.0f}%"
                       f", REGRESS_IDLE_GPU_UTIL changes it)")
    apps = [] if is_jetson() else parse_compute_apps(
        run("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"))
    if apps:
        reasons.append("other processes compute on the GPU: " + ", ".join(f"{a['name']}({a['pid']})" for a in apps[:4]))
    names = [n for n in run("docker", "ps", "--format", "{{.Names}}").split() if n and not n.startswith(own_prefix)]
    stacks = [n for n in names if re.search(r"sidecar|deepstream|go2rtc|redis|minio|mlflow", n)]
    if stacks:
        reasons.append("other stacks are running: " + ", ".join(stacks[:5]))
    if on_battery():
        reasons.append("laptop is on battery")
    return {"idle": not reasons, "reasons": reasons, "gpu": gpu, "foreign_compute": apps, "other_stacks": stacks}
