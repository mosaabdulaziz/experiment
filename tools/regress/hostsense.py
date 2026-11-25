"""Host-level sensors that the soak sampler and the golden runs do not read.

Why: a benchmark number is only a reference if what else was happening on the machine is recorded next to it. The soak sampler
(`soak.py`) records the containers' memory, CPU, I/O, the pipeline's own timers and the GPU; the golden runs record the GPU and the
stage timers. Neither records:

* pressure: PSI stall time for CPU / memory / IO (the host and each container), swap-in/out and major-fault rates, OOM kills;
* the CPU time split (user / system / iowait / steal) and per-core load, CPU frequency and governor;
* temperatures (CPU, NVMe, board, iGPU), fan speed, AC / battery state and iGPU power: what throttles a laptop;
* block-device I/O (the host's NVMe);
* per container: restarts, exit code, OOM-killed, health, cgroup `memory.events` / pressure / CPU throttling;
* the containers' logs (a teardown deletes them);
* a provenance snapshot at the start and the end: kernel, cmdline, governor, power profile, docker version, image ids and creation
  dates, `nvidia-smi -q`, the lscpu / meminfo text.

Everything is read from /proc, /sys and the docker CLI; one sample costs a few milliseconds (run it `nice`d).
Pure parsers first (tested on synthetic text), then the sampler with an injectable I/O layer, then the summary.
"""
import glob
import json
import os
import re
import signal
import subprocess
import threading
import time

from .soak import cgroup_paths, parse_kv_text

VMSTAT_KEYS = ("pswpin", "pswpout", "pgmajfault", "pgfault", "oom_kill", "allocstall_normal", "allocstall_movable", "pgscan_kswapd",
               "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct", "workingset_refault_anon", "workingset_refault_file",
               "thp_fault_alloc", "compact_stall")
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
CONTAINER_FORMAT = ("{{.Name}}\t{{.RestartCount}}\t{{.State.Status}}\t{{.State.OOMKilled}}\t{{.State.ExitCode}}\t"
                    "{{if .State.Health}}{{.State.Health.Status}}{{end}}\t{{.State.Pid}}\t{{.Config.Image}}\t{{.State.StartedAt}}\t"
                    "{{index .Config.Labels \"com.docker.compose.project\"}}")


# ---------------------------------------------------------------- parsers
def parse_psi(text: str) -> dict:
    """/proc/pressure/* or a cgroup *.pressure file: `some avg10=0.16 avg60=0.15 avg300=0.14 total=130505899720` (+ a `full` line)
    -> {"some": {"avg10": .., "avg60": .., "avg300": .., "total": int µs}, "full": {...}}."""
    out = {}
    for line in (text or "").splitlines():
        kind, _, rest = line.partition(" ")
        if kind not in ("some", "full"):
            continue
        fields = dict(p.split("=", 1) for p in rest.split() if "=" in p)
        try:
            out[kind] = {"avg10": float(fields["avg10"]), "avg60": float(fields["avg60"]), "avg300": float(fields["avg300"]),
                         "total": int(fields["total"])}
        except (KeyError, ValueError):
            continue
    return out


def parse_vmstat(text: str, keys=VMSTAT_KEYS) -> dict:
    out = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(" ")
        if k in keys and v.strip().isdigit():
            out[k] = int(v)
    return out


def parse_cpu_stat(text: str) -> dict:
    """/proc/stat -> {"all": {user..steal: ticks}, "cores": [[total, idle], ...]} (idle includes iowait)."""
    out = {"all": {}, "cores": []}
    for line in (text or "").splitlines():
        if not line.startswith("cpu"):
            continue
        name, *vals = line.split()
        try:
            nums = [int(v) for v in vals]
        except ValueError:
            continue
        if name == "cpu":
            out["all"] = dict(zip(CPU_FIELDS, nums))
        else:
            out["cores"].append([sum(nums[:8]), nums[3] + (nums[4] if len(nums) > 4 else 0)])
    return out


def parse_diskstats(text: str, dev: str) -> dict:
    """/proc/diskstats line of `dev` -> reads, sectors read, ms reading, writes, sectors written, ms writing, ms busy, ms weighted."""
    for line in (text or "").splitlines():
        p = line.split()
        if len(p) >= 14 and p[2] == dev:
            n = [int(x) for x in p[3:14]]
            return {"reads": n[0], "rd_sectors": n[2], "rd_ms": n[3], "writes": n[4], "wr_sectors": n[6], "wr_ms": n[7],
                    "io_busy_ms": n[9], "io_weighted_ms": n[10]}
    return {}


def parse_containers(text: str) -> dict:
    """`docker inspect --format CONTAINER_FORMAT` lines -> {name: {restarts, status, oom_killed, exit_code, health, pid, image, started, project}}."""
    out = {}
    for line in (text or "").splitlines():
        p = line.split("\t")
        if len(p) < 10:
            continue
        out[p[0].lstrip("/")] = {"restarts": int(p[1]) if p[1].isdigit() else None, "status": p[2], "oom_killed": p[3] == "true",
                                 "exit_code": int(p[4]) if p[4].lstrip("-").isdigit() else None, "health": p[5],
                                 "pid": int(p[6]) if p[6].isdigit() else None, "image": p[7], "started": p[8], "project": p[9]}
    return out


class Sys:
    """The file system and process boundary; tests replace it. `root` re-bases /proc and /sys (a fake tree in tests)."""

    def __init__(self, root: str = "/") -> None:
        self.root = root

    def path(self, p: str) -> str:
        return os.path.join(self.root, p.lstrip("/")) if self.root != "/" else p

    def read(self, p: str) -> str:
        try:
            with open(self.path(p), errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def glob(self, pattern: str) -> list:
        """Matches as paths relative to the root (so they can be passed straight back to read/listdir)."""
        hits = sorted(glob.glob(self.path(pattern)))
        return [h if self.root == "/" else "/" + os.path.relpath(h, self.root) for h in hits]

    def listdir(self, p: str) -> list:
        try:
            return sorted(os.listdir(self.path(p)))
        except OSError:
            return []

    def run(self, cmd: list, timeout: float = 20) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.stdout + (r.stderr if "logs" in cmd else "")
        except (OSError, subprocess.SubprocessError):
            return ""


def read_hwmon(sys_: Sys) -> dict:
    """Every hwmon temperature (°C), fan (rpm) and power (W) as `<chip>:<label|index>`."""
    out = {}
    for h in sys_.glob("/sys/class/hwmon/hwmon*"):
        name = sys_.read(os.path.join(h, "name")).strip() or os.path.basename(h)
        for f in sys_.listdir(h):
            m = re.fullmatch(r"(temp|fan|power|in)(\d+)_(input|average)", f)
            if not m:
                continue
            raw = sys_.read(os.path.join(h, f)).strip()
            if not re.fullmatch(r"-?\d+", raw):
                continue
            label = sys_.read(os.path.join(h, f"{m.group(1)}{m.group(2)}_label")).strip() or m.group(2)
            v = int(raw)
            key = f"{name}:{m.group(1)}:{label}"
            out[key] = round(v / 1000.0, 1) if m.group(1) == "temp" else (round(v / 1e6, 2) if m.group(1) == "power" else
                                                                             (v if m.group(1) == "fan" else round(v / 1000.0, 3)))
    return out


def read_cpufreq(sys_: Sys) -> dict:
    fs = [int(x) for x in (sys_.read(p).strip() for p in sys_.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")) if x.isdigit()]
    if not fs:
        return {}
    return {"min_khz": min(fs), "max_khz": max(fs), "mean_khz": round(sum(fs) / len(fs))}


def read_power_supply(sys_: Sys) -> dict:
    out = {}
    for d in sys_.glob("/sys/class/power_supply/*"):
        n = os.path.basename(d)
        entry = {}
        for k in ("online", "status", "capacity", "power_now", "current_now", "voltage_now"):
            v = sys_.read(os.path.join(d, k)).strip()
            if v:
                entry[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
        if entry:
            out[n] = entry
    return out


# ---------------------------------------------------------------- sampler
class HostSense:
    def __init__(self, projects: list, sys_: Sys = None, disk_dev: str = "nvme0n1", log_dir: str = "", log_every: float = 60.0,
                 inspect_every: int = 3) -> None:
        self.projects = projects
        self.s = sys_ or Sys()
        self.disk_dev = disk_dev
        self.log_dir = log_dir
        self.log_every = log_every
        self.inspect_every = max(1, inspect_every)
        self.n = 0
        self._cont = {}
        self._log_t = {}

    def _containers(self, now: float) -> dict:
        if self.n % self.inspect_every == 0 or not self._cont:
            ids = self.s.run(["docker", "ps", "-aq"]).split()
            text = self.s.run(["docker", "inspect", "--format", CONTAINER_FORMAT] + ids) if ids else ""
            self._cont = {k: v for k, v in parse_containers(text).items() if not self.projects or v["project"] in self.projects}
        out = {}
        for name, c in self._cont.items():
            entry = {k: c[k] for k in ("restarts", "status", "oom_killed", "exit_code", "health", "started", "image")}
            if c["status"] == "running" and c["pid"]:
                p = cgroup_paths(self.s.read(f"/proc/{c['pid']}/cgroup"))
                if p.get("v") == 2:
                    base = "/sys/fs/cgroup" + p["path"]
                    entry["memory_events"] = parse_kv_text(self.s.read(base + "/memory.events"))
                    entry["cpu_throttle"] = {k: v for k, v in parse_kv_text(self.s.read(base + "/cpu.stat")).items()
                                             if k in ("nr_periods", "nr_throttled", "throttled_usec")}
                    for what in ("memory", "cpu", "io"):
                        psi = parse_psi(self.s.read(f"{base}/{what}.pressure"))
                        if psi:
                            entry.setdefault("psi", {})[what] = psi
            out[name] = entry
        return out

    def _logs(self, now: float, force: bool = False) -> None:
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        for name, c in self._cont.items():
            if c["status"] != "running":
                continue
            last = self._log_t.get(name)
            if last is not None and not force and now - last < self.log_every:
                continue
            args = ["docker", "logs", "--timestamps"] + (["--since", f"{last:.3f}"] if last else []) + ["--until", f"{now:.3f}", name]
            text = self.s.run(args, timeout=60)
            self._log_t[name] = now
            if text:
                with open(os.path.join(self.log_dir, f"{name}.log"), "a") as f:
                    f.write(text if text.endswith("\n") else text + "\n")

    def tick(self) -> dict:
        now = time.time()
        row, errors = {"t": now}, {}

        def probe(name, fn, *a):
            try:
                return fn(*a)
            except Exception as exc:                              # noqa: BLE001 -- one broken probe never stops the run
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None
        row["psi"] = probe("psi", lambda: {w: parse_psi(self.s.read(f"/proc/pressure/{w}")) for w in ("cpu", "memory", "io")}) or {}
        row["vmstat"] = parse_vmstat(self.s.read("/proc/vmstat"))
        row["cpu"] = probe("cpu", parse_cpu_stat, self.s.read("/proc/stat")) or {}
        row["freq"] = probe("freq", read_cpufreq, self.s) or {}
        row["hwmon"] = probe("hwmon", read_hwmon, self.s) or {}
        row["power"] = probe("power", read_power_supply, self.s) or {}
        row["disk"] = parse_diskstats(self.s.read("/proc/diskstats"), self.disk_dev)
        row["containers"] = probe("containers", self._containers, now) or {}
        probe("logs", self._logs, now)
        if errors:
            row["errors"] = errors
        self.n += 1
        return row

    def flush_logs(self) -> None:
        self._logs(time.time(), force=True)


# ---------------------------------------------------------------- provenance
def snapshot(sys_: Sys, out_dir: str, tag: str) -> None:
    """Provenance text files at the start and at the end of a recording."""
    os.makedirs(out_dir, exist_ok=True)
    cmds = {"nvidia-smi-q": ["nvidia-smi", "-q"], "lscpu": ["lscpu"], "docker-version": ["docker", "version", "--format", "{{json .}}"],
            "docker-images": ["docker", "images", "--no-trunc", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"],
            "docker-ps": ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"], "sensors": ["sensors"],
            "nvidia-smi-clocks": ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,pstate,"
                                  "clocks_event_reasons.supported,gpu_bus_id,vbios_version", "--format=csv"],
            "ps-top": ["ps", "-eo", "pid,user,pcpu,pmem,rss,etime,comm", "--sort=-pcpu"]}
    files = {"cmdline": "/proc/cmdline", "meminfo": "/proc/meminfo", "os-release": "/etc/os-release", "version": "/proc/version",
             "governor": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "amd-pstate": "/sys/devices/system/cpu/amd_pstate/status",
             "boost": "/sys/devices/system/cpu/cpufreq/boost", "swappiness": "/proc/sys/vm/swappiness", "swaps": "/proc/swaps",
             "thp": "/sys/kernel/mm/transparent_hugepage/enabled", "loadavg": "/proc/loadavg"}
    with open(os.path.join(out_dir, f"provenance-{tag}.txt"), "w") as f:
        f.write(f"# captured {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        for k, p in files.items():
            f.write(f"\n## {k} ({p})\n{sys_.read(p).strip()}\n")
        for k, c in cmds.items():
            f.write(f"\n## {k} ({' '.join(c)[:60]})\n{sys_.run(c, 30).strip()}\n")


def run(projects: list, out_dir: str, duration_s: float, interval: float, sys_: Sys = None, logs: bool = True) -> str:
    sys_ = sys_ or Sys()
    os.makedirs(out_dir, exist_ok=True)
    hs = HostSense(projects, sys_, log_dir=os.path.join(out_dir, "logs") if logs else "")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    # its own pid, written by the process itself: a driver that backgrounds it from a subshell gets the wrong `$!`
    with open(os.path.join(out_dir, "hostsense.pid"), "w") as f:
        f.write(str(os.getpid()))
    snapshot(sys_, out_dir, "start")
    json.dump({"started": time.time(), "interval_s": interval, "projects": projects, "duration_s": duration_s},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    path = os.path.join(out_dir, "hostsense.jsonl")
    end = time.time() + duration_s
    with open(path, "a") as f:
        while not stop.is_set() and time.time() < end:
            t0 = time.time()
            f.write(json.dumps(hs.tick()) + "\n")
            f.flush()
            stop.wait(max(0.0, interval - (time.time() - t0)))
    hs.flush_logs()
    snapshot(sys_, out_dir, "end")
    try:
        os.remove(os.path.join(out_dir, "hostsense.pid"))
    except OSError:
        pass
    return path


# ---------------------------------------------------------------- summary
def _stats(vals: list) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    return {"min": round(v[0], 3), "mean": round(sum(v) / len(v), 3), "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 3),
            "max": round(v[-1], 3), "n": len(v)}


def _rate(rows: list, get) -> list:
    """Per-second rate of a monotonic counter between consecutive rows (a decrease = restart of the counter, skipped)."""
    out, prev = [], None
    for r in rows:
        v = get(r)
        if v is not None and prev is not None and v >= prev[1] and r["t"] > prev[0]:
            out.append((v - prev[1]) / (r["t"] - prev[0]))
        if v is not None:
            prev = (r["t"], v)
    return out


def summarize(rows: list, t0: float = None, t1: float = None) -> dict:
    """What the host was doing in [t0, t1] (epoch seconds; default all rows)."""
    rows = [r for r in rows if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
    out = {"samples": len(rows)}
    if not rows:
        return out
    out["seconds"] = round(rows[-1]["t"] - rows[0]["t"], 1)
    psi = {}
    for what in ("cpu", "memory", "io"):
        for kind in ("some", "full"):
            tot = _rate(rows, lambda r, w=what, k=kind: (r.get("psi", {}).get(w, {}).get(k) or {}).get("total"))
            if tot:
                psi[f"{what}_{kind}_stall_pct"] = _stats([x / 1e4 for x in tot])        # µs per s -> % of wall time
    out["psi"] = psi
    out["vm_per_s"] = {k: _stats(_rate(rows, lambda r, k=k: r.get("vmstat", {}).get(k))) for k in ("pswpin", "pswpout", "pgmajfault")}
    out["oom_kill_total"] = max((r.get("vmstat", {}).get("oom_kill", 0) for r in rows), default=0) - (rows[0].get("vmstat", {}).get("oom_kill", 0))
    split = {}
    for fld in CPU_FIELDS:
        pct = []
        for a, b in zip(rows, rows[1:]):
            ta, tb = a.get("cpu", {}).get("all"), b.get("cpu", {}).get("all")
            if ta and tb:
                tot = sum(tb.values()) - sum(ta.values())
                if tot > 0:
                    pct.append(100.0 * (tb[fld] - ta[fld]) / tot)
        if pct:
            split[fld] = _stats(pct)
    out["cpu_pct"] = split
    busiest = []
    for a, b in zip(rows, rows[1:]):
        ca, cb = a.get("cpu", {}).get("cores"), b.get("cpu", {}).get("cores")
        if ca and cb and len(ca) == len(cb):
            loads = [100.0 * (1 - (y[1] - x[1]) / (y[0] - x[0])) for x, y in zip(ca, cb) if y[0] > x[0]]
            if loads:
                busiest.append(max(loads))
    out["busiest_core_pct"] = _stats(busiest)
    out["freq_mean_khz"] = _stats([r.get("freq", {}).get("mean_khz") for r in rows])
    out["freq_min_khz"] = _stats([r.get("freq", {}).get("min_khz") for r in rows])
    keys = sorted({k for r in rows for k in r.get("hwmon", {})})
    out["hwmon"] = {k: _stats([r.get("hwmon", {}).get(k) for r in rows]) for k in keys}
    out["on_ac_fraction"] = round(sum(1 for r in rows if any(v.get("online") == 1 for k, v in r.get("power", {}).items()
                                                            if not k.startswith("BAT"))) / len(rows), 3)
    out["battery_status"] = sorted({v.get("status") for r in rows for k, v in r.get("power", {}).items() if k.startswith("BAT") and v.get("status")})
    d = {k: _stats(_rate(rows, lambda r, k=k: r.get("disk", {}).get(k))) for k in ("rd_sectors", "wr_sectors", "io_busy_ms")}
    out["disk_per_s"] = d
    cont = {}
    names = sorted({n for r in rows for n in r.get("containers", {})})
    for n in names:
        seq = [r["containers"][n] for r in rows if n in r.get("containers", {})]
        ev = lambda k: [s.get("memory_events", {}).get(k) for s in seq]                  # noqa: E731
        cont[n] = {"restarts_start_end": [seq[0].get("restarts"), seq[-1].get("restarts")],
                   "oom_killed_ever": any(s.get("oom_killed") for s in seq), "last_status": seq[-1].get("status"),
                   "last_exit_code": seq[-1].get("exit_code"), "unhealthy_samples": sum(1 for s in seq if s.get("health") not in ("", "healthy", None)),
                   "memory_events_delta": {k: (max(v for v in ev(k) if v is not None) - min(v for v in ev(k) if v is not None))
                                           for k in ("high", "max", "oom", "oom_kill") if any(v is not None for v in ev(k))},
                   "cpu_throttled_periods_delta": ((seq[-1].get("cpu_throttle", {}).get("nr_throttled") or 0)
                                                   - (seq[0].get("cpu_throttle", {}).get("nr_throttled") or 0))}
    out["containers"] = cont
    errs = {}
    for r in rows:
        for k, v in r.get("errors", {}).items():
            errs[k] = errs.get(k, 0) + 1
    out["probe_errors"] = errs
    return out


def load(path: str) -> list:
    p = os.path.join(path, "hostsense.jsonl") if os.path.isdir(path) else path
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def render(s: dict) -> str:
    lines = [f"{s['samples']} samples over {s.get('seconds', 0):.0f} s"]
    if not s.get("seconds"):
        return lines[0]

    def row(label, st, unit=""):
        if st:
            lines.append(f"  {label:<34} mean {st['mean']:>10.2f}  p95 {st['p95']:>10.2f}  max {st['max']:>10.2f} {unit}")
    lines.append("pressure (stall as % of wall time):")
    for k, st in s["psi"].items():
        row(k, st, "%")
    lines.append("memory movement:")
    for k, st in s["vm_per_s"].items():
        row(k + " /s", st)
    lines.append(f"  oom kills during the window: {s['oom_kill_total']}")
    lines.append("CPU time split (% of all cores):")
    for k, st in s["cpu_pct"].items():
        row(k, st, "%")
    row("busiest single core", s["busiest_core_pct"], "%")
    row("frequency, mean over cores", {k: (v / 1000 if k != "n" else v) for k, v in s["freq_mean_khz"].items()} if s["freq_mean_khz"] else {}, "MHz")
    lines.append("temperatures / fan / power (hwmon):")
    for k, st in s["hwmon"].items():
        row(k, st)
    lines.append(f"  on AC for {100 * s['on_ac_fraction']:.0f}% of samples; battery {s['battery_status']}")
    lines.append("containers (restarts start->end, OOM, throttling):")
    for n, c in s["containers"].items():
        lines.append(f"  {n:<44} restarts {c['restarts_start_end']}  oom_killed {c['oom_killed_ever']}  status {c['last_status']}  "
                     f"mem_events {c['memory_events_delta']}  cpu_throttled {c['cpu_throttled_periods_delta']}")
    if s["probe_errors"]:
        lines.append(f"probe errors: {s['probe_errors']}")
    return "\n".join(lines)
