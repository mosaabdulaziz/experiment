"""iris-exporter: Prometheus metrics for everything around the sidecar.

Reads the GPU through NVML (core, decoder, encoder, memory, temperature, power, throttle
reasons, Xid/ECC), the stack's containers through the Docker API, Redis (memory, camera
stream ages, the truck-feed backlog), go2rtc (which camera inputs are alive), the footage
players (ffmpeg processes), disk space and the truck-tracking platform. Everything is
optional: a source that is not reachable only turns its own `iris_probe_up{source}` to 0.

    python exporter.py            # serves :9200/metrics, polls every POLL_S seconds
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.request
from typing import Callable

from prometheus_client import Counter, Gauge, start_http_server

import events

log = logging.getLogger("iris.exporter")

POLL_S = float(os.environ.get("POLL_S", "5"))
CONTAINER_POLL_S = float(os.environ.get("CONTAINER_POLL_S", "15"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
FEED_KEY = os.environ.get("FEED_QUEUE_KEY", "iris:camera-feed")
FEED_DB = int(os.environ.get("FEED_DB", "1"))
GO2RTC_URL = os.environ.get("GO2RTC_URL", "http://go2rtc:1984")
UI_URL = os.environ.get("UI_URL", "http://ui:5173")
PROJECT = os.environ.get("COMPOSE_PROJECT", "scale")
DISK_PATHS = [p for p in os.environ.get("DISK_PATHS", "/data:data").split(",") if p]  # path:label
PLAYER_PROC = os.environ.get("PLAYER_PROC", "/host/proc")   # host /proc, read-only
CAMERAS = [c for c in os.environ.get("CAMERAS", "").split(",") if c]

EVENTS_LOG = os.environ.get("EVENTS_LOG", "/events/events.log")
PROM_URL = os.environ.get("PROM_URL", "http://prometheus:9090")
SYS_ROOT = os.environ.get("SYS_ROOT", "/host/sys")

NS = "iris_"


def _g(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
    return Gauge(NS + name, doc, labels)


# --- metrics -----------------------------------------------------------------------------------
probe_up = _g("probe_up", "1 when the last poll of this source worked", ("source",))
probe_s = _g("probe_seconds", "Duration of the last poll of this source", ("source",))

gpu_util = _g("gpu_util_percent", "GPU core utilisation", ("gpu",))
gpu_dec = _g("gpu_decoder_util_percent", "NVDEC utilisation", ("gpu",))
gpu_enc = _g("gpu_encoder_util_percent", "NVENC utilisation", ("gpu",))
gpu_mem_used = _g("gpu_memory_used_bytes", "GPU memory in use", ("gpu",))
gpu_mem_total = _g("gpu_memory_total_bytes", "GPU memory size", ("gpu",))
gpu_temp = _g("gpu_temperature_celsius", "GPU temperature", ("gpu",))
gpu_power = _g("gpu_power_watts", "GPU power draw", ("gpu",))
gpu_power_limit = _g("gpu_power_limit_watts", "GPU power cap in force", ("gpu",))
gpu_sm_clock = _g("gpu_sm_clock_mhz", "SM clock", ("gpu",))
gpu_sm_clock_max = _g("gpu_sm_clock_max_mhz", "Highest SM clock", ("gpu",))
gpu_mem_clock = _g("gpu_mem_clock_mhz", "Memory clock", ("gpu",))
gpu_throttle = _g("gpu_throttle", "1 while this throttle reason is active", ("gpu", "reason"))
gpu_pcie_tx = _g("gpu_pcie_tx_kbps", "PCIe transmit (KB/s)", ("gpu",))
gpu_pcie_rx = _g("gpu_pcie_rx_kbps", "PCIe receive (KB/s)", ("gpu",))
gpu_enc_sessions = _g("gpu_encoder_sessions", "Open NVENC sessions", ("gpu",))
gpu_enc_fps = _g("gpu_encoder_avg_fps", "Average FPS of the open NVENC sessions", ("gpu",))
gpu_enc_latency = _g("gpu_encoder_avg_latency_us", "Average NVENC latency", ("gpu",))
gpu_dec_sessions = _g("gpu_decoder_sessions", "Open NVDEC sessions (when the driver reports them)", ("gpu",))
gpu_ecc = _g("gpu_ecc_errors", "Uncorrected ECC errors", ("gpu",))
gpu_xid = _g("gpu_last_xid", "Most recent Xid error code seen (0 none)", ("gpu",))
gpu_procs = _g("gpu_processes", "Processes using the GPU", ("gpu",))

gpu_violation = _g("gpu_violation_seconds_total", "Cumulative time the GPU spent held back by a policy (NVML violation status)", ("gpu", "policy"))
gpu_throttle_events = Counter(NS + "gpu_throttle_events_total", "Throttles that lasted 3 s or more, by reason", ("gpu", "reason"))
gpu_throttle_blips = Counter(NS + "gpu_throttle_blips_total", "Throttle flickers shorter than 3 s (not logged), by reason", ("gpu", "reason"))
ac_online = _g("host_ac_online", "1 when the laptop's AC adapter is connected (-1 unknown)")
cpu_temp = _g("host_cpu_temperature_celsius", "CPU package temperature")

ctr_up = _g("container_up", "1 while the container runs", ("name",))
ctr_restarts = _g("container_restarts", "Restart count", ("name",))
ctr_oom = _g("container_oom_killed", "1 if the container was OOM-killed", ("name",))
ctr_cpu = _g("container_cpu_percent", "CPU use (100 = one core)", ("name",))
ctr_mem = _g("container_memory_bytes", "Memory in use", ("name",))
ctr_mem_limit = _g("container_memory_limit_bytes", "Memory limit (0 = none)", ("name",))
ctr_health = _g("container_healthy", "1 healthy, 0 unhealthy, -1 no healthcheck", ("name",))

redis_mem = _g("redis_memory_bytes", "Redis memory in use")
redis_mem_max = _g("redis_memory_max_bytes", "Redis maxmemory (0 = none)")
redis_clients = _g("redis_clients", "Connected Redis clients")
redis_evicted = _g("redis_evicted_keys", "Keys Redis evicted")
redis_ops = _g("redis_ops_per_second", "Redis commands per second")
cam_meta_age = _g("camera_meta_age_seconds", "Seconds since the sidecar last published this camera", ("cam",))
stream_len = _g("redis_stream_length", "Entries in a Redis stream", ("stream",))
feed_backlog = _g("truck_feed_backlog", "Batches waiting for the platform to consume")

go2rtc_producer = _g("go2rtc_input_alive", "1 when go2rtc has a live producer for the camera", ("cam",))
go2rtc_consumers = _g("go2rtc_consumers", "Readers (the sidecar, viewers) of the camera", ("cam",))
go2rtc_bytes = _g("go2rtc_bytes_received", "Bytes received from the camera's source", ("cam",))
go2rtc_rate = _g("go2rtc_bitrate_bps", "Incoming bitrate of the camera", ("cam",))
go2rtc_sent_rate = _g("go2rtc_sent_bps", "Bitrate go2rtc sends on to the camera's readers (the sidecar)", ("cam",))

player_alive = _g("player_alive", "1 while a footage ffmpeg feeds the camera", ("cam",))
players_total = _g("players", "Footage ffmpeg processes running")

disk_free = _g("disk_free_ratio", "Free fraction of the disk", ("label",))
disk_free_b = _g("disk_free_bytes", "Free bytes", ("label",))

plat_ready = _g("platform_ready", "1 when the truck-tracking API answers ready")
plat_events = _g("platform_feed_events_5m", "Feed events in the last 5 minutes by status", ("status",))
plat_sighting_age = _g("platform_newest_sighting_age_seconds", "Age of the newest stored sighting")
plat_sightings = _g("platform_sightings_total", "Sightings stored")
plat_trucks = _g("platform_trucks_total", "Trucks stored")


# --- helpers (pure, unit tested) ---------------------------------------------------------------
THROTTLE_BITS = {
    "gpu_idle": 0x1, "app_clocks": 0x2, "sw_power_cap": 0x4, "hw_slowdown": 0x8,
    "sync_boost": 0x10, "sw_thermal": 0x20, "hw_thermal": 0x40, "hw_power_brake": 0x80,
}
# the benign ones: idle and "clocks pinned by the app" are not trouble
THROTTLE_BAD = {"sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"}


def decode_throttle(mask: int) -> dict[str, int]:
    return {name: int(bool(mask & bit)) for name, bit in THROTTLE_BITS.items()}


def stream_entry_age(last_id: str | bytes | None, now: float | None = None) -> float | None:
    """Seconds since a Redis stream's last entry (`<ms>-<seq>`), None for an empty stream."""
    if not last_id:
        return None
    s = last_id.decode() if isinstance(last_id, bytes) else str(last_id)
    try:
        ms = int(s.split("-", 1)[0])
    except ValueError:
        return None
    return max(0.0, (time.time() if now is None else now) - ms / 1000.0)


def cam_name(n: str | int) -> str:
    """cam-1 / cam-01 / 1 -> cam-01 (go2rtc names are not zero-padded, the sidecar's are)."""
    return "cam-%02d" % int(str(n).rsplit("-", 1)[-1])


def parse_go2rtc(streams: dict) -> dict[str, dict]:
    """go2rtc /api/streams -> {cam-NN: {alive, consumers, bytes}} for the raw camera inputs.

    A raw-cam stream is alive while a footage player (or a real camera) is connected and
    pushing: a producer that carries a remote address / received bytes."""
    out = {}
    for name, st in (streams or {}).items():
        m = re.fullmatch(r"raw-cam-(\d+)", name)
        if not m:
            continue
        prods = (st or {}).get("producers") or []
        cons = (st or {}).get("consumers") or []
        recv = sum(int(p.get("bytes_recv") or 0) for p in prods)
        # what go2rtc has sent on to its readers (the sidecar): it keeps rising while the sidecar reads
        sent = sum(int(sd.get("bytes") or 0) for c in cons for sd in (c.get("senders") or []))
        alive = any(p.get("remote_addr") or int(p.get("bytes_recv") or 0) > 0 for p in prods)
        out[cam_name(m.group(1))] = {"alive": alive, "consumers": len(cons), "bytes": recv, "sent": sent}
    return out


def player_cams(cmdlines: list[str]) -> dict[str, int]:
    """ffmpeg command lines -> {cam: count} for the footage players (they push to raw-cam-NN)."""
    out: dict[str, int] = {}
    for c in cmdlines:
        if "ffmpeg" not in c:
            continue
        m = re.search(r"raw-cam-(\d+)", c)
        if m:
            k = cam_name(m.group(1))
            out[k] = out.get(k, 0) + 1
    return out


def cpu_percent(stats: dict) -> float:
    """Docker stats JSON -> CPU % (100 = one core)."""
    try:
        cpu = stats["cpu_stats"]
        pre = stats["precpu_stats"]
        d_cpu = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
        d_sys = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        n = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or [1])
        return d_cpu / d_sys * n * 100.0 if d_sys > 0 and d_cpu >= 0 else 0.0
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0


def mem_used(stats: dict) -> tuple[int, int]:
    m = stats.get("memory_stats") or {}
    used = int(m.get("usage") or 0)
    cache = int((m.get("stats") or {}).get("inactive_file") or (m.get("stats") or {}).get("cache") or 0)
    return max(0, used - cache), int(m.get("limit") or 0)


# --- pollers -----------------------------------------------------------------------------------
_nvml = None


def poll_gpu() -> None:
    global _nvml
    import pynvml as nv   # nvidia-ml-py
    if _nvml is None:
        nv.nvmlInit()
        _nvml = nv
    n = nv.nvmlDeviceGetCount()
    for i in range(n):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        lab = str(i)

        def safe(fn, *a):
            try:
                return fn(*a)
            except nv.NVMLError:
                return None

        u = safe(nv.nvmlDeviceGetUtilizationRates, h)
        if u is not None:
            gpu_util.labels(lab).set(u.gpu)
        d = safe(nv.nvmlDeviceGetDecoderUtilization, h)
        if d is not None:
            gpu_dec.labels(lab).set(d[0])
        e = safe(nv.nvmlDeviceGetEncoderUtilization, h)
        if e is not None:
            gpu_enc.labels(lab).set(e[0])
        mem = safe(nv.nvmlDeviceGetMemoryInfo, h)
        if mem is not None:
            gpu_mem_used.labels(lab).set(mem.used)
            gpu_mem_total.labels(lab).set(mem.total)
        t = safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
        if t is not None:
            gpu_temp.labels(lab).set(t)
        p = safe(nv.nvmlDeviceGetPowerUsage, h)
        if p is not None:
            gpu_power.labels(lab).set(p / 1000.0)
        pl = safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
        if pl is not None:
            gpu_power_limit.labels(lab).set(pl / 1000.0)
        for gauge, kind in ((gpu_sm_clock, nv.NVML_CLOCK_SM), (gpu_mem_clock, nv.NVML_CLOCK_MEM)):
            c = safe(nv.nvmlDeviceGetClockInfo, h, kind)
            if c is not None:
                gauge.labels(lab).set(c)
        cm = safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM)
        if cm is not None:
            gpu_sm_clock_max.labels(lab).set(cm)
        mask = safe(nv.nvmlDeviceGetCurrentClocksEventReasons, h)
        if mask is None:
            mask = safe(nv.nvmlDeviceGetCurrentClocksThrottleReasons, h)
        if mask is not None:
            for name, v in decode_throttle(int(mask)).items():
                gpu_throttle.labels(lab, name).set(v)
        tx = safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_TX_BYTES)
        rx = safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_RX_BYTES)
        if tx is not None:
            gpu_pcie_tx.labels(lab).set(tx)
        if rx is not None:
            gpu_pcie_rx.labels(lab).set(rx)
        es = safe(nv.nvmlDeviceGetEncoderStats, h)
        if es is not None:
            gpu_enc_sessions.labels(lab).set(es[0])
            gpu_enc_fps.labels(lab).set(es[1])
            gpu_enc_latency.labels(lab).set(es[2])
        ds = safe(nv.nvmlDeviceGetDecoderUtilization, h)   # sessions are not exposed; keep the label set
        if ds is not None and hasattr(nv, "nvmlDeviceGetDecoderSessions"):
            sess = safe(nv.nvmlDeviceGetDecoderSessions, h)
            if sess is not None:
                gpu_dec_sessions.labels(lab).set(len(sess))
        ecc = safe(nv.nvmlDeviceGetTotalEccErrors, h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC)
        if ecc is not None:
            gpu_ecc.labels(lab).set(ecc)
        procs = safe(nv.nvmlDeviceGetComputeRunningProcesses, h) or []
        gpu_procs.labels(lab).set(len(procs))


def emit_event(component: str, text: str) -> None:
    """One line to the events log file and to the container log (-> Loki, tagged [event])."""
    line = events.event_line(component, text)
    events.append_event(EVENTS_LOG, line)
    log.warning("%s", line)


_tracker = events.ThrottleTracker()
_VIOLATION_POLICIES = {"power": 0, "thermal": 1, "sync_boost": 2, "board_limit": 3, "low_util": 4,
                       "reliability": 5, "total_app_clocks": 10, "total_base_clocks": 11}


def poll_throttle() -> None:
    """Every second: watch the throttle reasons and write each start / end to the events log with the
    context that explains a laptop's throttling (power limit vs draw, clocks, temperatures, AC)."""
    global _nvml
    import pynvml as nv
    if _nvml is None:
        nv.nvmlInit()
        _nvml = nv
    ac = events.read_ac_online(f"{SYS_ROOT}/class/power_supply")
    cpu_t = events.read_cpu_temp(f"{SYS_ROOT}/class/hwmon")
    ac_online.set(-1 if ac is None else int(ac))
    if cpu_t is not None:
        cpu_temp.set(cpu_t)
    now = time.time()
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)

        def safe(fn, *a):
            try:
                return fn(*a)
            except nv.NVMLError:
                return None

        mask = safe(nv.nvmlDeviceGetCurrentClocksEventReasons, h)
        if mask is None:
            mask = safe(nv.nvmlDeviceGetCurrentClocksThrottleReasons, h)
        if mask is None:
            continue
        p = safe(nv.nvmlDeviceGetPowerUsage, h)
        pl = safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
        ctx = {"power_w": p / 1000.0 if p is not None else None,
               "power_limit_w": pl / 1000.0 if pl is not None else None,
               "sm_mhz": safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
               "sm_max_mhz": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM),
               "temp_c": safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
               "cpu_temp_c": cpu_t, "ac_online": ac}
        for ev in _tracker.update(now, decode_throttle(int(mask)), ctx):
            if ev["kind"] == "blip":
                gpu_throttle_blips.labels(str(i), ev["reason"]).inc()
                continue
            if ev["kind"] == "start":
                gpu_throttle_events.labels(str(i), ev["reason"]).inc()
            emit_event("gpu", events.format_throttle(ev))
        for name, kind in _VIOLATION_POLICIES.items():
            v = safe(nv.nvmlDeviceGetViolationStatus, h, kind)
            if v is not None:
                gpu_violation.labels(str(i), name).set(v.violationTime / 1e9)


_alerts_state: dict = {}
_alerts_primed = False


def poll_alerts() -> None:
    """Write every alert that starts or stops firing (Prometheus rules) to the events log."""
    global _alerts_state, _alerts_primed
    data = _get_json(PROM_URL + "/api/v1/alerts")["data"]["alerts"]
    _alerts_state, evs = events.diff_alerts(_alerts_state, data, time.time())
    if not _alerts_primed:
        _alerts_primed = True
        evs = [e for e in evs if e["kind"] == "firing"]       # on first look: what is already firing
    for ev in evs:
        emit_event("alert", events.format_alert(ev))


def poll_containers() -> None:
    import docker
    cli = docker.from_env()
    seen = set()
    for c in cli.containers.list(all=True, filters={"label": f"com.docker.compose.project={PROJECT}"}):
        name = c.name
        seen.add(name)
        attrs = c.attrs
        st = attrs.get("State", {})
        up = st.get("Running", False) and not st.get("Restarting", False)
        ctr_up.labels(name).set(int(up))
        ctr_restarts.labels(name).set(attrs.get("RestartCount", 0))
        ctr_oom.labels(name).set(int(bool(st.get("OOMKilled"))))
        h = (st.get("Health") or {}).get("Status")
        ctr_health.labels(name).set({"healthy": 1, "unhealthy": 0}.get(h, -1 if h is None else 0))
        if up:
            try:
                s = c.stats(stream=False)
                ctr_cpu.labels(name).set(cpu_percent(s))
                used, limit = mem_used(s)
                ctr_mem.labels(name).set(used)
                ctr_mem_limit.labels(name).set(limit)
            except Exception:
                pass


_redis = {}


def _r(db: int):
    import redis
    if db not in _redis:
        _redis[db] = redis.Redis.from_url(REDIS_URL, db=db, socket_timeout=3)
    return _redis[db]


def poll_redis() -> None:
    r0 = _r(0)
    info = r0.info()
    redis_mem.set(info.get("used_memory", 0))
    redis_mem_max.set(info.get("maxmemory", 0))
    redis_clients.set(info.get("connected_clients", 0))
    redis_evicted.set(info.get("evicted_keys", 0))
    redis_ops.set(info.get("instantaneous_ops_per_sec", 0))
    cams = CAMERAS or [k.decode().rsplit(":", 1)[-1] for k in r0.scan_iter("iris:meta:cam-*", count=200)]
    for cam in sorted(cams):
        try:
            last = r0.xinfo_stream(f"iris:meta:{cam}").get("last-generated-id")
        except Exception:
            last = None
        age = stream_entry_age(last)
        cam_meta_age.labels(cam).set(age if age is not None else 1e9)
    for s in ("iris:mtmc:assigned", "iris:mtmc:exited", "iris:mtmc:gate", "iris:mtmc:plates"):
        try:
            stream_len.labels(s).set(r0.xlen(s))
        except Exception:
            pass
    feed_backlog.set(_r(FEED_DB).llen(FEED_KEY))


def _get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


_g2_prev: dict[str, tuple[float, int]] = {}
_g2_sent_prev: dict[str, tuple[float, int]] = {}


def poll_go2rtc() -> None:
    parsed = parse_go2rtc(_get_json(GO2RTC_URL + "/api/streams"))
    now = time.time()
    for cam in sorted(set(CAMERAS) | set(parsed)):
        st = parsed.get(cam, {"alive": False, "consumers": 0, "bytes": 0, "sent": 0})
        go2rtc_producer.labels(cam).set(int(st["alive"]))
        go2rtc_consumers.labels(cam).set(st["consumers"])
        go2rtc_bytes.labels(cam).set(st["bytes"])
        prev = _g2_prev.get(cam)
        if prev and now > prev[0] and st["bytes"] >= prev[1]:
            go2rtc_rate.labels(cam).set((st["bytes"] - prev[1]) * 8 / (now - prev[0]))
        _g2_prev[cam] = (now, st["bytes"])
        sprev = _g2_sent_prev.get(cam)
        if sprev and now > sprev[0] and st["sent"] >= sprev[1]:
            go2rtc_sent_rate.labels(cam).set((st["sent"] - sprev[1]) * 8 / (now - sprev[0]))
        _g2_sent_prev[cam] = (now, st["sent"])


def poll_players() -> None:
    cmds = []
    try:
        for pid in os.listdir(PLAYER_PROC):
            if not pid.isdigit():
                continue
            try:
                with open(f"{PLAYER_PROC}/{pid}/cmdline", "rb") as f:
                    cmds.append(f.read().replace(b"\0", b" ").decode(errors="ignore"))
            except OSError:
                continue
    except OSError:
        raise
    cams = player_cams(cmds)
    players_total.set(sum(cams.values()))
    for cam in sorted(set(CAMERAS) | set(cams)):
        player_alive.labels(cam).set(int(cams.get(cam, 0) > 0))


def poll_disk() -> None:
    for item in DISK_PATHS:
        path, _, label = item.partition(":")
        u = shutil.disk_usage(path)
        disk_free.labels(label or path).set(u.free / u.total)
        disk_free_b.labels(label or path).set(u.free)


def poll_platform() -> None:
    try:
        # the UI's dev server only answers its own host names, not the compose service name
        req = urllib.request.Request(UI_URL + "/health/ready", headers={"Host": "localhost"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            plat_ready.set(int(resp.status == 200))
    except Exception:
        plat_ready.set(0)
        raise
    dsn = os.environ.get("PLATFORM_DB_DSN")
    if not dsn:
        return
    import psycopg
    with psycopg.connect(dsn, connect_timeout=3, autocommit=True) as con:
        cur = con.cursor()
        cur.execute("select processing_status::text, count(*) from feed_events_raw "
                    "where received_at > now() - interval '5 minutes' group by 1")
        seen = {k: v for k, v in cur.fetchall()}
        for status in ("ok", "rejected", "failed", "duplicate"):
            plat_events.labels(status).set(seen.get(status, 0))
        cur.execute("select extract(epoch from now() - max(seen_at)), count(*) from sightings")
        age, n = cur.fetchone()
        plat_sighting_age.set(age if age is not None else 1e9)
        plat_sightings.set(n)
        cur.execute("select count(*) from trucks")
        plat_trucks.set(cur.fetchone()[0])


POLLERS: list[tuple[str, Callable[[], None], float]] = [
    ("gpu", poll_gpu, POLL_S),
    ("redis", poll_redis, POLL_S),
    ("go2rtc", poll_go2rtc, POLL_S),
    ("players", poll_players, POLL_S),
    ("disk", poll_disk, 30.0),
    ("containers", poll_containers, CONTAINER_POLL_S),
    ("platform", poll_platform, 10.0),
    ("throttle", poll_throttle, 1.0),
    ("alerts", poll_alerts, 5.0),
]


def _run(name: str, fn: Callable[[], None], every: float) -> None:
    while True:
        t0 = time.monotonic()
        try:
            fn()
            probe_up.labels(name).set(1)
        except Exception as exc:
            probe_up.labels(name).set(0)
            log.warning("poll %s failed: %s", name, exc)
        probe_s.labels(name).set(time.monotonic() - t0)
        time.sleep(max(0.5, every - (time.monotonic() - t0)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    start_http_server(int(os.environ.get("PORT", "9200")))
    for name, fn, every in POLLERS:
        threading.Thread(target=_run, args=(name, fn, every), daemon=True, name=name).start()
    log.info("iris-exporter on :%s", os.environ.get("PORT", "9200"))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
