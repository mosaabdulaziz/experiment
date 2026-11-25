"""Long-run resource sampler: what does each part of a stack use, and what grows? (task: long-run resource growth, #174)

    python3 -m regress soak --project safety --duration 3h --interval 30 \
        --watch alert_svr=scratch/safety/data/alert_svr --pidfile fall=scratch/safety/fall.pid

Every `interval` seconds it appends one JSON row to `<out>/samples.jsonl` covering, for the compose project's containers and
the host processes next to them: CPU, memory split into what the process really holds (anon) and what is only cache or
tmpfs (file, shmem), process RSS / peak / threads / fds, network and block I/O, GPU (the counters `regress run` uses:
NVML, DCGM profiling, tegrastats), Redis memory and stream lengths, directory sizes (disk growth), the sidecars' own
`:9100` stage timers and gauges, and the app's event counts. `soak_analyze` turns the file into growth slopes.

Read from OUTSIDE the processes only: cgroup files, the host's /proc (a container's processes are visible there), `docker top`,
`redis-cli` through `docker exec`, the GPU tools. Nothing is signalled or restarted; containers are addressed by their exact names
and host processes only through a pid file that still names the process that wrote it (the rule of scripts/safety_pids.sh).
Every source degrades to "missing" on failure: one broken probe never stops the run. Pure parsers first, so they are tested on
real excerpts without docker or a GPU.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field

from . import gpucounters, machine

PROMETHEUS_PORT = 9100
STREAM_PREFIXES = ("meta", "alert", "overlay", "svr_request", "svr_done", "zone_alert", "overcrowd")


# ── pure parsers ─────────────────────────────────────────────────────────────────────────────────────────────────────
def parse_kv_text(text: str) -> dict:
    """`key value` lines (cgroup memory.stat, cpu.stat) -> {key: int}."""
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"-?\d+", parts[1]):
            out[parts[0]] = int(parts[1])
    return out


def cgroup_paths(proc_cgroup: str) -> dict:
    """Where a process's cgroup lives under /sys/fs/cgroup, from /proc/<pid>/cgroup: v2 `0::/system.slice/docker-<id>.scope`
    or v1 `N:memory:/docker/<id>` lines. {"v": 2, "path"} or {"v": 1, "memory", "cpuacct", "pids"}."""
    v1 = {}
    for line in (proc_cgroup or "").splitlines():
        hid, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hid == "0" and controllers == "":
            return {"v": 2, "path": path}
        for c in controllers.split(","):
            if c in ("memory", "cpuacct", "pids"):
                v1[c] = path
    return {"v": 1, **v1} if v1 else {}


def memory_split(stat: dict, version: int) -> dict:
    """What a cgroup's memory is made of, in bytes: `anon` (heap and stacks: what a leak grows), `file` (page cache),
    `shmem` (tmpfs and shared memory: counted in `file` by v2 as well, so reported separately), `kernel`."""
    if version == 2:
        return {"anon": stat.get("anon"), "file": stat.get("file"), "shmem": stat.get("shmem"),
                "kernel": stat.get("kernel", (stat.get("slab") or 0) + (stat.get("kernel_stack") or 0))}
    return {"anon": stat.get("total_rss", stat.get("rss")), "file": stat.get("total_cache", stat.get("cache")),
            "shmem": stat.get("total_shmem", stat.get("shmem")), "kernel": None}


def parse_io_stat(text: str) -> dict:
    """cgroup v2 io.stat (`8:0 rbytes=1 wbytes=2 ...` per device) -> total {read_bytes, write_bytes}."""
    r = w = 0
    for line in (text or "").splitlines():
        for key, val in re.findall(r"(rbytes|wbytes)=(\d+)", line):
            if key == "rbytes":
                r += int(val)
            else:
                w += int(val)
    return {"read_bytes": r, "write_bytes": w}


def parse_proc_status(text: str) -> dict:
    """/proc/<pid>/status -> {rss_bytes, peak_rss_bytes, swap_bytes, rss_anon_bytes, rss_file_bytes, rss_shmem_bytes, threads}
    (kB fields converted to bytes). RSS alone mixes anonymous memory with file-backed pages: under memory pressure the kernel evicts
    the latter, so RSS can fall while a leak keeps growing; RssAnon is the part to watch."""
    out = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        num = re.match(r"\s*(\d+)", rest)
        if not num:
            continue
        v = int(num.group(1))
        if key == "VmRSS":
            out["rss_bytes"] = v * 1024
        elif key == "VmHWM":
            out["peak_rss_bytes"] = v * 1024
        elif key == "VmSwap":
            out["swap_bytes"] = v * 1024
        elif key == "RssAnon":
            out["rss_anon_bytes"] = v * 1024         # heap and stacks: the part a leak grows
        elif key == "RssFile":
            out["rss_file_bytes"] = v * 1024         # mapped files and libraries: the kernel evicts these under pressure
        elif key == "RssShmem":
            out["rss_shmem_bytes"] = v * 1024
        elif key == "Threads":
            out["threads"] = v
    return out


def parse_net_dev(text: str) -> dict:
    """/proc/<pid>/net/dev -> total {rx_bytes, tx_bytes} of every interface but lo."""
    rx = tx = 0
    for line in (text or "").splitlines()[2:]:
        name, _, rest = line.partition(":")
        fields = rest.split()
        if name.strip() == "lo" or len(fields) < 9:
            continue
        rx += int(fields[0])
        tx += int(fields[8])
    return {"rx_bytes": rx, "tx_bytes": tx}


def parse_proc_stat_cpu(stat_text: str) -> "tuple | None":
    """/proc/<pid>/stat -> (utime + stime ticks, pgrp, starttime ticks); the command name may hold spaces and ')'."""
    tail = (stat_text or "").rsplit(")", 1)
    if len(tail) != 2:
        return None
    f = tail[1].split()
    try:
        return int(f[11]) + int(f[12]), int(f[2]), int(f[19])        # fields 14, 15 (u/s time), 5 (pgrp), 22 (start)
    except (IndexError, ValueError):
        return None


def pid_is_the_writer(pid: int, stat_text: str, pidfile_mtime: float, btime: float, clk_tck: int) -> bool:
    """The rule of scripts/safety_pids.sh: a pid in a pid file is trusted only while it is still the process that wrote the file:
    above 1, leading its own process group, started no later than the file was written (2 s of rounding)."""
    parsed = parse_proc_stat_cpu(stat_text)
    if pid <= 1 or parsed is None or clk_tck <= 0:
        return False
    _ticks, pgrp, start = parsed
    return pgrp == pid and (btime + start / clk_tck) <= pidfile_mtime + 2


def parse_redis_info(text: str) -> dict:
    """redis-cli INFO output -> {field: number} for the numeric fields (memory, clients, ...)."""
    out = {}
    for line in (text or "").splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.strip().partition(":")
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            pass
    return out


def key_prefix_counts(keys: list) -> dict:
    """`iris:meta:cam@zones` -> counted under `iris:meta`: how many keys of each kind exist."""
    out = {}
    for k in keys:
        p = ":".join(k.split(":")[:2])
        out[p] = out.get(p, 0) + 1
    return out


def parse_prom_text(text: str) -> dict:
    """The sidecar's :9100/metrics -> {"gauges": {name: v}, "stage_busy": {stage: %}, "stage_avg_ms": {stage: ms}}."""
    gauges, busy, avg = {}, {}, {}
    for line in (text or "").splitlines():
        m = re.match(r'^(iris_gauge|iris_stage_busy_percent|iris_stage_avg_ms)\{(?:name|stage)="([^"]+)"\}\s+(\S+)', line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        {"iris_gauge": gauges, "iris_stage_busy_percent": busy, "iris_stage_avg_ms": avg}[m.group(1)][m.group(2)] = v
    return {"gauges": gauges, "stage_busy": busy, "stage_avg_ms": avg}


def tree_size(path: str) -> dict:
    """{bytes, files} of a directory tree (symlinks not followed); missing = zeros."""
    total = files = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.lstat(os.path.join(root, n)).st_size
                files += 1
            except OSError:
                pass
    return {"bytes": total, "files": files}


# ── the real data sources (everything below can be replaced in a test) ──────────────────────────────────────────────
def _run(cmd: list, timeout: float = 15, stdin: str = None) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


class RealIO:
    """Docker, /proc, cgroup, Redis, HTTP and the GPU tools. Each method returns '' / {} / [] when its source is not there."""

    def containers(self, project: str) -> list:
        """[(name, full id, init host pid)] of the compose project's running containers (exact names)."""
        names = _run(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.Names}}"]).split()
        if not names:
            return []
        out = _run(["docker", "inspect", "-f", "{{.Name}} {{.Id}} {{.State.Pid}}", *names]).splitlines()
        rows = []
        for line in out:
            parts = line.split()
            if len(parts) == 3 and parts[2].isdigit():
                rows.append((parts[0].lstrip("/"), parts[1], int(parts[2])))
        return rows

    def container_pids(self, name: str) -> list:
        return [int(x) for x in _run(["docker", "top", name, "-eo", "pid"]).split()[1:] if x.isdigit()]

    def read(self, path: str) -> str:
        return _read(path)

    def fd_count(self, pid: int):
        try:
            return len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            return None

    def mtime(self, path: str):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def redis(self, container: str, commands: list) -> str:
        return _run(["docker", "exec", "-i", container, "redis-cli"], stdin="\n".join(commands) + "\n")

    def redis_scan(self, container: str, pattern: str) -> list:
        return _run(["docker", "exec", container, "redis-cli", "--scan", "--pattern", pattern], timeout=30).split()

    def container_http(self, name: str, port: int = PROMETHEUS_PORT) -> str:
        script = f"import urllib.request;print(urllib.request.urlopen('http://localhost:{port}/metrics',timeout=5).read().decode())"
        return _run(["docker", "exec", name, "python3", "-c", script], timeout=20)

    def http_json(self, url: str):
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                return json.loads(r.read())
        except Exception:                                   # noqa: BLE001 -- the app may be down: that is a data point, not an error
            return None

    def gpu_processes(self) -> list:
        out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout=10)
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                rows.append((int(parts[0]), int(parts[1])))
        return rows

    def clk_tck(self) -> int:
        try:
            return os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError):
            return 100


@dataclass
class SoakConfig:
    project: str
    interval: float = 30.0
    redis_container: str = ""
    api: str = ""
    watch: dict = field(default_factory=dict)            # label -> directory whose size is tracked
    pidfiles: dict = field(default_factory=dict)         # label -> pid file of a host process
    telemetry_match: str = r"(safety-(proximity|zones|ppe|fall|fire)|deepstream-sidecar)"
    du_every: int = 4                                     # directory sizes every N ticks (walking big trees is not free)


class Sampler:
    """One `tick()` = one row. Holds the previous cumulative readings only to keep the row self-contained for the analyzer."""

    def __init__(self, cfg: SoakConfig, io=None, gpu=None) -> None:
        self.cfg, self.io, self.gpu = cfg, io or RealIO(), gpu
        self.n = 0
        self._last_t = time.time()
        self._disk_cache: dict = {}

    # each probe returns a dict; a failure is recorded in row["errors"] and never raised
    def _containers(self, now: float) -> tuple:
        out, gpu_pid_to_name = {}, {}
        rows = self.io.containers(self.cfg.project)
        pid_owner = {}
        for name, _cid, init_pid in rows:
            entry = {}
            paths = cgroup_paths(self.io.read(f"/proc/{init_pid}/cgroup"))
            cg = {}
            if paths.get("v") == 2:
                base = "/sys/fs/cgroup" + paths["path"]
                stat = parse_kv_text(self.io.read(base + "/memory.stat"))
                cg.update(memory_split(stat, 2))
                cur = self.io.read(base + "/memory.current").strip()
                peak = self.io.read(base + "/memory.peak").strip()
                swap = self.io.read(base + "/memory.swap.current").strip()
                cg["memory_bytes"] = int(cur) if cur.isdigit() else None
                cg["peak_bytes"] = int(peak) if peak.isdigit() else None
                cg["swap_bytes"] = int(swap) if swap.isdigit() else None
                cg["cpu_usec"] = parse_kv_text(self.io.read(base + "/cpu.stat")).get("usage_usec")
                pids = self.io.read(base + "/pids.current").strip()
                cg["pids"] = int(pids) if pids.isdigit() else None
                cg.update(parse_io_stat(self.io.read(base + "/io.stat")))
            elif paths.get("v") == 1 and "memory" in paths:
                base = "/sys/fs/cgroup/memory" + paths["memory"]
                cg.update(memory_split(parse_kv_text(self.io.read(base + "/memory.stat")), 1))
                usage = self.io.read(base + "/memory.usage_in_bytes").strip()
                cg["memory_bytes"] = int(usage) if usage.isdigit() else None
            entry["cgroup"] = {k: v for k, v in cg.items() if v is not None}
            n_procs, tot_rss, main, tot_anon, tot_file = 0, 0, None, 0, 0
            for pid in self.io.container_pids(name):
                st = parse_proc_status(self.io.read(f"/proc/{pid}/status"))
                if "rss_bytes" not in st:
                    continue
                pid_owner[pid] = name
                n_procs += 1
                tot_rss += st["rss_bytes"]
                tot_anon += st.get("rss_anon_bytes", 0)
                tot_file += st.get("rss_file_bytes", 0)
                if main is None or st["rss_bytes"] > main["rss_bytes"]:
                    main = {"pid": pid, **st, "fds": self.io.fd_count(pid)}
                    cpu = parse_proc_stat_cpu(self.io.read(f"/proc/{pid}/stat"))
                    main["cpu_ticks"] = cpu[0] if cpu else None
                    main["comm"] = self.io.read(f"/proc/{pid}/comm").strip()
            entry["procs"] = {"n": n_procs, "rss_bytes_total": tot_rss, "rss_anon_bytes_total": tot_anon, "rss_file_bytes_total": tot_file,
                              "main": {k: v for k, v in (main or {}).items() if v is not None}}
            entry["net"] = parse_net_dev(self.io.read(f"/proc/{init_pid}/net/dev"))
            out[name] = entry
        return out, pid_owner

    def _host_procs(self) -> dict:
        out = {}
        btime = 0.0
        for line in self.io.read("/proc/stat").splitlines():
            if line.startswith("btime"):
                btime = float(line.split()[1])
        tck = self.io.clk_tck()
        for label, pidfile in self.cfg.pidfiles.items():
            mtime = self.io.mtime(pidfile)
            raw = self.io.read(pidfile).strip()
            if mtime is None or not raw.isdigit():
                continue
            pid = int(raw)
            stat = self.io.read(f"/proc/{pid}/stat")
            if not pid_is_the_writer(pid, stat, mtime, btime, tck):
                out[label] = {"stale_pidfile": 1}
                continue
            st = parse_proc_status(self.io.read(f"/proc/{pid}/status"))
            cpu = parse_proc_stat_cpu(stat)
            out[label] = {"pid": pid, **st, "fds": self.io.fd_count(pid), "cpu_ticks": cpu[0] if cpu else None}
            out[label] = {k: v for k, v in out[label].items() if v is not None}
        return out

    def _host(self) -> dict:
        mem = parse_kv_text("\n".join(l.replace(":", "").replace(" kB", "") for l in self.io.read("/proc/meminfo").splitlines()))
        load = self.io.read("/proc/loadavg").split()
        cpu = self.io.read("/proc/stat").splitlines()[:1]
        ticks = [int(x) for x in cpu[0].split()[1:]] if cpu and cpu[0].startswith("cpu ") else []
        out = {"mem_available_bytes": mem.get("MemAvailable", 0) * 1024, "mem_total_bytes": mem.get("MemTotal", 0) * 1024,
               "swap_used_bytes": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) * 1024,
               "shmem_bytes": mem.get("Shmem", 0) * 1024}
        if load:
            out["load1"], out["load5"] = float(load[0]), float(load[1])
        if ticks:
            out["cpu_total_ticks"] = sum(ticks)
            out["cpu_idle_ticks"] = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
        return out

    def _redis(self) -> dict:
        c = self.cfg.redis_container
        if not c:
            return {}
        keys = self.io.redis_scan(c, "iris:*")
        streams = [k for k in keys if k.split(":")[1:2] and k.split(":")[1] in STREAM_PREFIXES][:60]
        cmds = ["INFO memory", "INFO clients", "DBSIZE"] + [f"XLEN {k}" for k in streams]
        text = self.io.redis(c, cmds)
        info = parse_redis_info(text)
        # the XLEN answers are the last len(streams) integer lines, in order (a tty prints `(integer) 5`, a pipe prints `5`)
        ints = [m.group(1) for m in (re.fullmatch(r"(?:\(integer\) )?(\d+)", l.strip()) for l in text.splitlines()) if m]
        lens = ints[-len(streams):] if streams else []
        out = {k: info[k] for k in ("used_memory", "used_memory_rss", "mem_fragmentation_ratio", "connected_clients") if k in info}
        out["keys"] = len(keys)
        out["key_prefixes"] = key_prefix_counts(keys)
        out["stream_len"] = {k: int(v) for k, v in zip(streams, lens) if str(v).isdigit()}
        return out

    def _disk(self) -> dict:
        out = {}
        for label, path in self.cfg.watch.items():
            if self.n % max(1, self.cfg.du_every) == 0 or label not in self._disk_cache:
                self._disk_cache[label] = tree_size(path)
            entry = dict(self._disk_cache[label])
            try:
                entry["free_bytes"] = shutil.disk_usage(path if os.path.exists(path) else ".").free
            except OSError:
                pass
            out[label] = entry
        return out

    def _telemetry(self, names: list) -> dict:
        out = {}
        rx = re.compile(self.cfg.telemetry_match)
        for name in names:
            if rx.search(name):
                parsed = parse_prom_text(self.io.container_http(name))
                if parsed["gauges"] or parsed["stage_busy"]:
                    out[name] = parsed
        return out

    def _app(self) -> dict:
        if not self.cfg.api:
            return {}
        out = {}
        events = self.io.http_json(self.cfg.api.rstrip("/") + "/api/events?limit=1000")
        if events is not None:
            ev = events if isinstance(events, list) else events.get("events", [])
            out["events_total"] = len(ev)
            out["incidents_ongoing"] = sum(1 for e in ev if e.get("incident_status") == "ongoing")
        cams = self.io.http_json(self.cfg.api.rstrip("/") + "/api/safety/cameras")
        if isinstance(cams, dict):
            fps = {}
            for cam in cams.get("cameras", []):
                for s in cam.get("streams", []):
                    if s.get("fps") is not None:
                        fps[s["id"]] = s["fps"]
            out["stream_fps"] = fps
        return out

    def tick(self) -> dict:
        now = time.time()
        row, errors = {"t": now}, {}

        def probe(name, fn, *a):
            try:
                return fn(*a)
            except Exception as exc:                          # noqa: BLE001 -- one broken probe never stops the run
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None
        got = probe("containers", self._containers, now)
        containers, pid_owner = got if got else ({}, {})
        row["containers"] = containers
        row["host"] = probe("host", self._host) or {}
        row["host_procs"] = probe("host_procs", self._host_procs) or {}
        row["redis"] = probe("redis", self._redis) or {}
        row["disk"] = probe("disk", self._disk) or {}
        row["telemetry"] = probe("telemetry", self._telemetry, list(containers)) or {}
        row["app"] = probe("app", self._app) or {}
        if self.gpu is not None:
            w = probe("gpu", self.gpu.window, self._last_t, now) or {}
            row["gpu"] = {k: w[k] for k in ("nvml", "dcgm", "jetson") if w.get(k)}
            procs = {}
            for pid, mib in probe("gpu_procs", self.io.gpu_processes) or []:
                owner = pid_owner.get(pid)
                if owner:
                    procs[owner] = procs.get(owner, 0) + mib
            row["gpu_procs_mib"] = procs
        if errors:
            row["errors"] = errors
        self._last_t = now
        self.n += 1
        return row


def parse_duration(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", str(text).strip())
    if not m:
        raise ValueError(f"not a duration: {text!r} (examples: 90s, 30m, 3h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def run(cfg: SoakConfig, out_dir: str, duration_s: float, repo: str, use_gpu: bool = True, use_dcgm: bool = True,
        io=None) -> str:
    """Sample until `duration_s` has passed or SIGINT/SIGTERM; returns the samples file. Writes meta.json first."""
    os.makedirs(out_dir, exist_ok=True)
    samples = os.path.join(out_dir, "samples.jsonl")
    gpu = None
    if use_gpu:
        try:
            gpu = gpucounters.GpuCounters(repo, use_dcgm=use_dcgm)
        except Exception as exc:                              # noqa: BLE001
            print(f"GPU counters unavailable: {exc}", flush=True)
    halt = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: halt.set())
        except ValueError:                                    # not the main thread (tests)
            pass
    meta = {"started": time.time(), "interval_s": cfg.interval, "duration_s": duration_s, "project": cfg.project,
            "fingerprint": machine.fingerprint(repo), "on_battery": machine.on_battery(),
            "background_gpu_util_pct": machine.background_util(samples=3),
            "gpu_counters": None if gpu is None else {"dcgm": gpu.dcgm_status},
            "config": {k: v for k, v in cfg.__dict__.items()}}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    sampler = Sampler(cfg, io=io, gpu=gpu)
    t_end = time.time() + duration_s
    try:
        with open(samples, "a") as f:
            while not halt.is_set():
                row = sampler.tick()
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                if time.time() + cfg.interval > t_end:
                    break
                halt.wait(cfg.interval)
    finally:
        if gpu is not None:
            gpu.stop()
    return samples
