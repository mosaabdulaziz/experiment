"""The soak sampler and its growth analysis (tools/regress/soak.py, soak_analyze.py). No docker, no GPU, no network: the parsers
run on realistic excerpts, the analyzer on synthetic series, and the whole sampler through a fake data source."""
import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import cli, machine, soak, soak_analyze  # noqa: E402


# ── parsers ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cgroup_v2_and_v1_paths():
    assert soak.cgroup_paths("0::/system.slice/docker-abc123.scope\n") == {"v": 2, "path": "/system.slice/docker-abc123.scope"}
    v1 = soak.cgroup_paths("12:memory:/docker/abc\n11:cpuacct,cpu:/docker/abc\n3:pids:/docker/abc\n")
    assert v1 == {"v": 1, "memory": "/docker/abc", "cpuacct": "/docker/abc", "pids": "/docker/abc"}
    assert soak.cgroup_paths("") == {}


def test_memory_split_keeps_anon_apart_from_cache_and_tmpfs():
    stat = soak.parse_kv_text("anon 1000\nfile 5000\nshmem 800\nkernel 40\nslab 7\n")
    assert soak.memory_split(stat, 2) == {"anon": 1000, "file": 5000, "shmem": 800, "kernel": 40}
    v1 = soak.parse_kv_text("total_rss 11\ntotal_cache 22\ntotal_shmem 3\n")
    assert soak.memory_split(v1, 1) == {"anon": 11, "file": 22, "shmem": 3, "kernel": None}
    assert soak.parse_kv_text("garbage\nkey notanumber\nok 5\n") == {"ok": 5}


def test_io_stat_sums_devices():
    text = "8:0 rbytes=100 wbytes=50 rios=1 wios=2\n259:0 rbytes=1 wbytes=2 rios=0 wios=0\n"
    assert soak.parse_io_stat(text) == {"read_bytes": 101, "write_bytes": 52}


def test_proc_status_converts_kb_to_bytes():
    text = "Name:\tpython3\nVmPeak:\t 900 kB\nVmHWM:\t 2048 kB\nVmRSS:\t 1024 kB\nVmSwap:\t 0 kB\nThreads:\t35\n"
    assert soak.parse_proc_status(text) == {"rss_bytes": 1024 * 1024, "peak_rss_bytes": 2048 * 1024, "swap_bytes": 0, "threads": 35}


def test_net_dev_ignores_loopback():
    text = ("Inter-|   Receive\n face |bytes\n"
            "    lo: 999 1 0 0 0 0 0 0 999 1 0 0 0 0 0 0\n"
            "  eth0: 100 1 0 0 0 0 0 0 200 1 0 0 0 0 0 0\n  eth1: 5 1 0 0 0 0 0 0 7 1 0 0 0 0 0 0\n")
    assert soak.parse_net_dev(text) == {"rx_bytes": 105, "tx_bytes": 207}


def _stat(pid, comm, pgrp, utime, stime, start):
    # fields after the command: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime ... starttime
    rest = ["S", "1", str(pgrp), "0", "0", "0", "0", "0", "0", "0", "0", str(utime), str(stime), "0", "0", "20", "0", "1", "0", str(start)]
    return f"{pid} ({comm}) " + " ".join(rest) + " 0 0\n"


def test_proc_stat_cpu_survives_odd_command_names():
    assert soak.parse_proc_stat_cpu(_stat(7, "a b) c", 7, 10, 5, 4242)) == (15, 7, 4242)
    assert soak.parse_proc_stat_cpu("garbage") is None


def test_a_pid_file_is_trusted_only_while_it_names_the_process_that_wrote_it():
    btime, tck = 1_000_000.0, 100
    stat = _stat(500, "python3", 500, 1, 1, 6000)               # started 60 s after boot
    written = btime + 61
    assert soak.pid_is_the_writer(500, stat, written, btime, tck)
    assert not soak.pid_is_the_writer(500, stat, btime + 10, btime, tck)             # file older than the process: a recycled pid
    assert not soak.pid_is_the_writer(500, _stat(500, "x", 499, 1, 1, 6000), written, btime, tck)   # not leading its own group
    assert not soak.pid_is_the_writer(1, stat, written, btime, tck)
    assert not soak.pid_is_the_writer(500, "", written, btime, tck)


def test_redis_info_prefix_counts_prometheus_and_duration():
    assert soak.parse_redis_info("# Memory\r\nused_memory:100\r\nmem_fragmentation_ratio:1.5\r\nrole:master\r\n") == \
        {"used_memory": 100, "mem_fragmentation_ratio": 1.5}
    assert soak.key_prefix_counts(["iris:meta:a@z", "iris:meta:b", "iris:alert:a", "x"]) == {"iris:meta": 2, "iris:alert": 1, "x": 1}
    prom = ('# HELP x\niris_gauge{name="publisher_pending"} 3.0\niris_stage_busy_percent{stage="gpu.decode.cam"} 41.5\n'
            'iris_stage_avg_ms{stage="probe"} 7.25\niris_stage_busy_percent_created{stage="x"} 1e9\n')
    assert soak.parse_prom_text(prom) == {"gauges": {"publisher_pending": 3.0}, "stage_busy": {"gpu.decode.cam": 41.5},
                                          "stage_avg_ms": {"probe": 7.25}}
    assert soak.parse_duration("90s") == 90 and soak.parse_duration("3h") == 10800 and soak.parse_duration("45") == 45
    with pytest.raises(ValueError):
        soak.parse_duration("soon")


def test_tree_size(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.mkv").write_bytes(b"1" * 1000)
    (tmp_path / "y.mkv").write_bytes(b"1" * 500)
    assert soak.tree_size(str(tmp_path)) == {"bytes": 1500, "files": 2}
    assert soak.tree_size(str(tmp_path / "missing")) == {"bytes": 0, "files": 0}


# ── analysis ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_ols_recovers_a_line_and_its_fit():
    ts = list(range(0, 600, 30))
    slope, r2 = soak_analyze.ols(ts, [5 + 0.01 * t for t in ts])
    assert slope == pytest.approx(0.01) and r2 == pytest.approx(1.0)
    assert soak_analyze.ols([1, 2, 3], [4, 4, 4]) == (0.0, 1.0)


def _verdict(values, unit="MiB", step=30, warmup=0.0):
    ts = [i * step for i in range(len(values))]
    return soak_analyze.analyze_series(ts, values, unit, warmup)["verdict"]


def test_an_injected_leak_is_flagged_and_noise_and_a_warmup_step_are_not():
    n = 240                                                       # 2 h at 30 s
    leak = [500 + 0.01 * i * 30 / 3600 * 1000 * 3 for i in range(n)]       # a steady climb of about 90 MiB/h
    assert _verdict(leak) == "growing"
    rnd = random.Random(7)
    flat = [500 + rnd.uniform(-3, 3) for _ in range(n)]
    assert _verdict(flat) == "flat"
    sawtooth = [500 + 60 * ((i % 40) / 40) for i in range(n)]      # a cache that fills and empties
    assert _verdict(sawtooth) in ("noisy", "flat") and _verdict(sawtooth) != "growing"
    step_then_flat = [100] * 20 + [900] * (n - 20)                 # models load in the first 10 minutes
    assert _verdict(step_then_flat, warmup=900) == "flat"
    assert _verdict(step_then_flat, warmup=0) == "flat"           # a load inside the first third is absorbed even without the cut
    late_step = [500] * 140 + [900] * (n - 140)                    # something grew by 400 MiB well after start-up: worth a look
    assert _verdict(late_step) == "growing"
    assert _verdict([1, 2, 3]) == "too short"


def test_small_absolute_growth_is_not_a_leak():
    tiny = [500 + 0.001 * i for i in range(240)]                   # 0.24 MiB in 2 h
    assert _verdict(tiny) == "flat"
    assert _verdict([100 + 0.5 * i for i in range(60)], unit="count") == "growing"      # +30 fds in 30 min is
    assert _verdict([100 + 0.01 * i for i in range(60)], unit="count") == "flat"


def test_cumulative_counters_become_rates_and_info_roots_are_not_judged():
    rows = []
    for i in range(40):
        rows.append({"t": 1000.0 + 30 * i,
                     "containers": {"c": {"cgroup": {"cpu_usec": 500_000 * 30 * i, "anon": (200 + 0.2 * i) * 1048576},
                                          "procs": {"main": {"fds": 12}}}},
                     "app": {"events_total": 10 * i}, "gpu": {"nvml": {"mem_used_mib": {"median": 1000.0 + 5 * i}}}})
    res = soak_analyze.analyze(rows, warmup_s=0)
    cpu = res["containers.c.cgroup.cpu_usec"]
    assert cpu["unit"] == "cores" and cpu["median"] == pytest.approx(0.5, rel=0.01) and cpu["verdict"] == "flat"
    assert res["containers.c.procs.main.fds"]["verdict"] == "flat"
    assert res["app.events_total"]["verdict"] == "info" and res["gpu.nvml.mem_used_mib.median"]["verdict"] == "info"
    assert res["containers.c.cgroup.anon"]["unit"] == "MiB" and res["_run"]["rows"] == 40


def test_unit_and_flatten_rules():
    assert soak_analyze.unit_of("containers.c.cgroup.anon") == ("MiB", 1 / 1048576)
    assert soak_analyze.unit_of("redis.used_memory_rss")[0] == "MiB"
    assert soak_analyze.unit_of("gpu_procs_mib.safety-zones-1") == ("MiB", 1.0)
    assert soak_analyze.unit_of("redis.stream_len.iris:meta:cam@zones")[0] == "count"
    assert soak_analyze.unit_of("host.load1")[0] == "other"
    flat = soak_analyze.flatten({"t": 5, "a": {"b": 1, "c": True, "d": "x", "e": {"f": 2.5}}, "errors": {"x": "y"}})
    assert flat == {"a.b": 1.0, "a.e.f": 2.5}


# ── the sampler end to end, through a fake data source ───────────────────────────────────────────────────────────────
class FakeIO:
    """One zones container whose anon memory grows by 1 MiB per tick and one flat Redis container."""

    def __init__(self):
        self.step = 0
        self.broken = set()

    def containers(self, project):
        return [("safety-safety-zones-1", "aaa", 100), ("safety-redis-1", "bbb", 200)]

    def container_pids(self, name):
        return [101] if "zones" in name else [201]

    def read(self, path):
        grow = (300 + self.step) * 1048576
        files = {
            "/proc/100/cgroup": "0::/system.slice/docker-aaa.scope\n",
            "/proc/200/cgroup": "0::/system.slice/docker-bbb.scope\n",
            "/sys/fs/cgroup/system.slice/docker-aaa.scope/memory.stat": f"anon {grow}\nfile 5000\nshmem 99\nkernel 10\n",
            "/sys/fs/cgroup/system.slice/docker-aaa.scope/memory.current": str(grow + 5000),
            "/sys/fs/cgroup/system.slice/docker-aaa.scope/cpu.stat": f"usage_usec {self.step * 15_000_000}\n",
            "/sys/fs/cgroup/system.slice/docker-aaa.scope/pids.current": "35",
            "/sys/fs/cgroup/system.slice/docker-bbb.scope/memory.stat": "anon 50000000\nfile 1000\nshmem 0\nkernel 5\n",
            "/proc/101/status": f"VmHWM:\t{(300 + self.step) * 1024 + 10} kB\nVmRSS:\t{(300 + self.step) * 1024} kB\nThreads:\t35\n",
            "/proc/201/status": "VmHWM:\t60000 kB\nVmRSS:\t50000 kB\nThreads:\t4\n",
            "/proc/101/stat": _stat(101, "python3", 100, self.step * 100, 0, 5000),
            "/proc/101/comm": "python3\n",
            "/proc/100/net/dev": "h\nh\n eth0: 1000 1 0 0 0 0 0 0 2000 1 0 0 0 0 0 0\n",
            "/proc/meminfo": "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\nShmem: 1000 kB\n",
            "/proc/loadavg": "0.5 0.4 0.3 1/200 999\n",
            "/proc/stat": "cpu  100 0 100 800 0 0 0 0 0 0\nbtime 1000000\n",
        }
        if path in self.broken:
            raise OSError("boom")
        return files.get(path, "")

    def fd_count(self, pid):
        return 12

    def mtime(self, path):
        return None

    def redis(self, container, commands):
        return "# Memory\r\nused_memory:1048576\r\nused_memory_rss:2097152\r\n# Clients\r\nconnected_clients:3\r\n42\r\n64\r\n7\r\n"

    def redis_scan(self, container, pattern):
        return ["iris:meta:office@zones", "iris:alert:office@zones", "iris:status:office@zones"]

    def container_http(self, name, port=9100):
        return 'iris_gauge{name="publisher_pending"} 2.0\niris_stage_busy_percent{stage="probe"} 12.0\n'

    def http_json(self, url):
        return {"events": [{"incident_status": "ongoing"}, {"incident_status": "resolved"}]} if "events" in url else \
            {"cameras": [{"id": "office", "streams": [{"id": "office@zones", "fps": 19.5}]}]}

    def gpu_processes(self):
        return [(101, 900)]

    def clk_tck(self):
        return 100


def _config(tmp_path):
    return soak.SoakConfig(project="safety", interval=0.0, redis_container="safety-redis-1", api="http://app",
                           watch={"alert_svr": str(tmp_path)}, du_every=1)


def test_one_tick_covers_every_aspect(tmp_path):
    io = FakeIO()
    row = soak.Sampler(_config(tmp_path), io=io).tick()
    c = row["containers"]["safety-safety-zones-1"]
    assert c["cgroup"]["anon"] == 300 * 1048576 and c["cgroup"]["shmem"] == 99 and c["cgroup"]["pids"] == 35
    assert c["procs"]["main"]["rss_bytes"] == 300 * 1024 * 1024 and c["procs"]["main"]["fds"] == 12 and c["procs"]["n"] == 1
    assert c["net"] == {"rx_bytes": 1000, "tx_bytes": 2000}
    assert row["redis"]["used_memory"] == 1048576 and row["redis"]["keys"] == 3
    assert row["redis"]["stream_len"] and row["redis"]["key_prefixes"]["iris:meta"] == 1
    assert row["host"]["mem_available_bytes"] == 8000000 * 1024 and row["host"]["load1"] == 0.5
    assert row["disk"]["alert_svr"]["bytes"] == 0
    assert row["telemetry"]["safety-safety-zones-1"]["gauges"] == {"publisher_pending": 2.0}
    assert "safety-redis-1" not in row["telemetry"]                    # only the sidecars are scraped
    assert row["app"] == {"events_total": 2, "incidents_ongoing": 1, "stream_fps": {"office@zones": 19.5}}
    assert "errors" not in row


def test_one_broken_probe_does_not_stop_the_row(tmp_path):
    io = FakeIO()
    io.redis = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("redis down"))
    row = soak.Sampler(_config(tmp_path), io=io).tick()
    assert "redis" in row["errors"] and row["redis"] == {}
    assert row["containers"]["safety-safety-zones-1"]["cgroup"]["anon"]          # everything else is still there


def test_a_leaking_container_is_found_end_to_end(tmp_path):
    io = FakeIO()
    sampler = soak.Sampler(_config(tmp_path), io=io)
    rows = []
    for i in range(120):
        io.step = i
        row = sampler.tick()
        row["t"] = 5000.0 + 30 * i                                      # a 1-hour run at 30 s
        rows.append(row)
    res = soak_analyze.analyze(rows, warmup_s=300)
    grow = {k for k, v in res.items() if k != "_run" and v.get("verdict") == "growing"}
    assert "containers.safety-safety-zones-1.cgroup.anon" in grow
    assert "containers.safety-safety-zones-1.procs.main.rss_bytes" in grow
    assert "containers.safety-redis-1.cgroup.anon" not in grow
    assert res["containers.safety-safety-zones-1.cgroup.anon"]["slope_per_h"] == pytest.approx(120.0, rel=0.05)   # 1 MiB / 30 s
    text = soak_analyze.render(res)
    assert "growing" in text and "containers.safety-safety-zones-1.cgroup.anon" in text


def test_run_writes_meta_and_samples_and_the_cli_analyses_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(machine, "fingerprint", lambda repo: {"machine_id": "test-box"})
    monkeypatch.setattr(machine, "background_util", lambda **k: 3.0)
    monkeypatch.setattr(machine, "on_battery", lambda: False)
    io = FakeIO()
    out = tmp_path / "run"
    path = soak.run(_config(tmp_path), str(out), duration_s=0.0, repo=str(tmp_path), use_gpu=False, io=io)
    assert os.path.exists(path) and json.load(open(out / "meta.json"))["fingerprint"]["machine_id"] == "test-box"
    rows = soak_analyze.load_rows(str(out))
    assert len(rows) >= 1 and "containers" in rows[0]
    assert cli.main(["soak-analyze", str(out), "--warmup", "0s"]) in (0, 2)
    assert math.isfinite(1.0)                                            # the CLI path runs without docker or GPU


def test_swapped_out_memory_cannot_hide_a_leak():
    """Under memory pressure the kernel swaps a leaking process out: RSS and cgroup anon fall while what it holds keeps growing."""
    rows = []
    for i in range(120):
        held = (300 + i) * 1048576                               # +1 MiB per 30 s: a real leak
        swapped = min(held - 100 * 1048576, i * 3 * 1048576)     # and the kernel pushes it to swap faster than it grows
        rows.append({"t": 1000.0 + 30 * i, "containers": {"c": {"cgroup": {"anon": held - swapped, "swap_bytes": swapped},
                                                                 "procs": {"main": {"rss_bytes": held - swapped, "swap_bytes": swapped}}}},
                     "host": {"mem_total_bytes": 16e9, "mem_available_bytes": 4e9, "swap_used_bytes": swapped}})
    res = soak_analyze.analyze(rows, warmup_s=300)
    assert res["containers.c.cgroup.anon"]["verdict"] != "growing"              # RSS-style view: looks fine (it even falls)
    assert res["containers.c.cgroup.anon_plus_swap"]["verdict"] == "growing"      # what it holds: leaking
    assert res["containers.c.procs.main.rss_plus_swap_bytes"]["verdict"] == "growing"
    assert "host.used_plus_swap_bytes" in res


def test_a_stream_filling_to_its_maxlen_is_bounded_not_growing():
    ts = [i * 30.0 for i in range(120)]
    filling = [min(1500, 100 + 6 * i) for i in range(120)]                      # the overlay stream: capped at 1500
    assert soak_analyze.analyze_series(ts, filling, "count", 0, cap=1500)["verdict"] == "bounded"
    runaway = [100 + 40 * i for i in range(120)]                                # far past the cap: not bounded
    assert soak_analyze.analyze_series(ts, runaway, "count", 0, cap=64)["verdict"] == "growing"
    assert soak_analyze._stream_cap("redis.stream_len.iris:overlay:office@zones") == 1500
    assert soak_analyze._stream_cap("redis.stream_len.iris:unknown:x") is None
    assert soak_analyze._stream_cap("redis.keys") is None


def test_process_anon_is_kept_apart_from_file_backed_memory():
    """Under pressure the kernel evicts a process's file-backed pages: RSS falls while its heap (RssAnon) keeps growing."""
    text = "VmRSS:\t 2048 kB\nRssAnon:\t 1024 kB\nRssFile:\t 900 kB\nRssShmem:\t 124 kB\nVmSwap:\t 10 kB\nThreads:\t5\n"
    st = soak.parse_proc_status(text)
    assert st["rss_anon_bytes"] == 1024 * 1024 and st["rss_file_bytes"] == 900 * 1024 and st["rss_shmem_bytes"] == 124 * 1024
    rows = []
    for i in range(120):
        anon, filebacked = (200 + 0.5 * i) * 1048576, max(50.0, 800 - 8 * i) * 1048576      # heap +60 MiB/h, files evicted
        rows.append({"t": 1000.0 + 30 * i, "containers": {"c": {"procs": {"main": {"rss_bytes": anon + filebacked, "rss_anon_bytes": anon,
                                                                                      "swap_bytes": 0}}}}})
    res = soak_analyze.analyze(rows, warmup_s=300)
    assert res["containers.c.procs.main.rss_bytes"]["verdict"] != "growing"                 # RSS falls: hides the growth
    assert res["containers.c.procs.main.rss_anon_bytes"]["verdict"] == "growing"           # the heap does not
    assert res["containers.c.procs.main.rss_anon_bytes_plus_swap"]["verdict"] == "growing"



def series(slope, verdict, unit="MiB"):
    return {"n": 300, "unit": unit, "slope_per_h": slope, "r2": 0.7, "verdict": verdict}


def test_soak_gate_passes_the_same_growth_and_warns_on_a_new_or_a_much_steeper_one():
    ref = {"containers.z.cgroup.anon": series(42.0, "growing"), "containers.p.cgroup.anon": series(1.0, "flat"),
           "containers.z.cgroup.file": series(125.0, "growing"), "disk.alert_clips.bytes": series(3600.0, "growing")}
    same = {k: dict(v) for k, v in ref.items()}
    assert soak_analyze.compare_soak(same, ref)["verdict"] == "PASS"
    leak = dict(same, **{"containers.p.cgroup.anon": series(80.0, "growing")})                        # was flat, now grows
    rep = soak_analyze.compare_soak(leak, ref)
    assert rep["verdict"] == "WARN" and [r["id"] for r in rep["rows"] if r["status"] == "WARN"] == ["containers.p.cgroup.anon"]
    steeper = dict(same, **{"containers.z.cgroup.anon": series(120.0, "growing")})                      # 2.9x and +78 MiB/h
    assert [r["id"] for r in soak_analyze.compare_soak(steeper, ref)["rows"] if r["status"] == "WARN"] == ["containers.z.cgroup.anon"]
    assert soak_analyze.compare_soak(steeper, ref, strict=True)["exit"] == 1
    cache = dict(same, **{"containers.z.cgroup.file": series(900.0, "growing")})                      # page cache is not watched
    assert soak_analyze.compare_soak(cache, ref)["verdict"] == "PASS"


def test_the_disk_factor_sits_above_the_variability_of_two_identical_soaks():
    ref = {"disk.alert_svr.bytes": series(129.0, "growing"), "disk.alert_clips.bytes": series(3618.0, "growing")}
    second_soak = {"disk.alert_svr.bytes": series(152.0, "growing"), "disk.alert_clips.bytes": series(3599.0, "growing")}     # what the 2nd soak showed
    assert soak_analyze.compare_soak(second_soak, ref)["verdict"] == "PASS"
    much_more = {"disk.alert_svr.bytes": series(129.0, "growing"), "disk.alert_clips.bytes": series(6000.0, "growing")}          # 1.66x and +2.4 GB/h
    assert [r["id"] for r in soak_analyze.compare_soak(much_more, ref)["rows"] if r["status"] == "WARN"] == ["disk.alert_clips.bytes"]


def test_soak_gate_floors_and_improvements():
    ref = {"containers.z.cgroup.anon": series(42.0, "growing"), "containers.q.cgroup.anon": series(1.0, "flat")}
    small = {"containers.z.cgroup.anon": series(60.0, "growing"), "containers.q.cgroup.anon": series(12.0, "growing")}
    rep = soak_analyze.compare_soak(small, ref)
    st = {r["id"]: r["status"] for r in rep["rows"]}
    assert st["containers.z.cgroup.anon"] == "PASS"                    # under 2x
    assert st["containers.q.cgroup.anon"] == "INFO"                    # grows, but under the 25 MiB/h floor
    better = {"containers.z.cgroup.anon": series(0.5, "flat"), "containers.q.cgroup.anon": series(1.0, "flat")}
    assert {r["status"] for r in soak_analyze.compare_soak(better, ref)["rows"]} == {"IMPROVED"}
    assert soak_analyze.compare_soak({"containers.new.cgroup.anon": series(90.0, "growing")}, ref)["verdict"] == "WARN"   # a series not in the reference


def test_soak_compare_command_reads_the_reference_and_exits_by_the_verdict(tmp_path, capsys):
    ref = {"containers.z.cgroup.anon": series(42.0, "growing")}
    new = {"containers.z.cgroup.anon": series(200.0, "growing")}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "new.json").write_text(json.dumps(new))
    assert cli.main(["soak-compare", str(tmp_path / "new.json"), "--reference", str(tmp_path / "ref.json")]) == 0        # a WARN does not fail
    assert "WARN" in capsys.readouterr().out
    assert cli.main(["soak-compare", str(tmp_path / "new.json"), "--reference", str(tmp_path / "ref.json"), "--strict"]) == 1


def test_a_series_that_is_noisy_now_is_not_called_improved():
    ref = {"containers.z.cgroup.anon": series(42.0, "growing")}
    new = {"containers.z.cgroup.anon": series(72.0, "noisy")}
    rep = soak_analyze.compare_soak(new, ref)
    assert [r["status"] for r in rep["rows"]] == ["INFO"] and rep["verdict"] == "PASS"
