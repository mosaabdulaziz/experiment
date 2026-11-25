"""The host sensor recorder (tools/regress/hostsense.py). No docker, no real /proc: parsers on realistic excerpts, the sampler on a fake
file tree and a fake docker, the summary on synthetic series."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import cli, hostsense  # noqa: E402

PSI = "some avg10=0.16 avg60=0.15 avg300=0.14 total=1000000\nfull avg10=0.01 avg60=0.02 avg300=0.03 total=50000\n"
CPU_STAT = ("cpu  100 0 50 800 40 0 10 5 0 0\ncpu0 50 0 25 400 20 0 5 2 0 0\ncpu1 50 0 25 400 20 0 5 3 0 0\nintr 1 2 3\n")
DISKSTATS = "   259       0 nvme0n1 100 0 8000 50 200 0 16000 90 0 70 140 0 0 0 0 0 0\n   7  0 loop0 1 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
CONT = ("/safety-redis-1\t0\trunning\tfalse\t0\t\t4242\tredis:7\t2026-11-17T10:00:00Z\tsafety\n"
        "/safety-fire-1\t2\trunning\ttrue\t137\thealthy\t4343\tiris-sidecar\t2026-11-17T10:01:00Z\tsafety\n"
        "/other-db-1\t0\trunning\tfalse\t0\t\t4444\tpostgres\t2026-11-17T09:00:00Z\tother\n")


def test_parse_psi_some_and_full():
    p = hostsense.parse_psi(PSI)
    assert p["some"] == {"avg10": 0.16, "avg60": 0.15, "avg300": 0.14, "total": 1000000}
    assert p["full"]["total"] == 50000
    assert hostsense.parse_psi("") == {}
    assert hostsense.parse_psi("some avg10=x") == {}                 # damaged line: skipped, no crash


def test_parse_vmstat_only_wanted_keys():
    v = hostsense.parse_vmstat("pswpin 5\npswpout 9\nnr_free_pages 123\noom_kill 1\npgmajfault 77\n")
    assert v == {"pswpin": 5, "pswpout": 9, "oom_kill": 1, "pgmajfault": 77}


def test_parse_cpu_stat_all_and_cores():
    c = hostsense.parse_cpu_stat(CPU_STAT)
    assert c["all"]["user"] == 100 and c["all"]["iowait"] == 40 and c["all"]["steal"] == 5
    assert len(c["cores"]) == 2 and c["cores"][0] == [502, 420]        # total of the 8 fields; idle includes iowait


def test_parse_diskstats_picks_the_device():
    d = hostsense.parse_diskstats(DISKSTATS, "nvme0n1")
    assert d["reads"] == 100 and d["rd_sectors"] == 8000 and d["wr_sectors"] == 16000 and d["io_busy_ms"] == 70
    assert hostsense.parse_diskstats(DISKSTATS, "sda") == {}


def test_parse_containers():
    c = hostsense.parse_containers(CONT)
    assert c["safety-fire-1"]["restarts"] == 2 and c["safety-fire-1"]["oom_killed"] is True and c["safety-fire-1"]["exit_code"] == 137
    assert c["safety-redis-1"]["health"] == "" and c["other-db-1"]["project"] == "other"


# ── the sampler on a fake tree ───────────────────────────────────────────────────────────────────────────────────────
def _write(root, rel, text):
    p = os.path.join(root, rel.lstrip("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


class FakeSys(hostsense.Sys):
    def __init__(self, root):
        super().__init__(root)
        self.calls = []

    def run(self, cmd, timeout=20):
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return "a\nb\nc\n"
        if cmd[:2] == ["docker", "inspect"]:
            return CONT
        if cmd[:2] == ["docker", "logs"]:
            return f"2026-11-17T10:00:00Z line from {cmd[-1]}\n"
        return ""


def _fake_host(root):
    _write(root, "/proc/pressure/cpu", PSI)
    _write(root, "/proc/pressure/memory", PSI)
    _write(root, "/proc/pressure/io", PSI)
    _write(root, "/proc/vmstat", "pswpin 5\npswpout 9\noom_kill 0\npgmajfault 77\n")
    _write(root, "/proc/stat", CPU_STAT)
    _write(root, "/proc/diskstats", DISKSTATS)
    _write(root, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "2000000\n")
    _write(root, "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "3000000\n")
    _write(root, "/sys/class/hwmon/hwmon6/name", "k10temp\n")
    _write(root, "/sys/class/hwmon/hwmon6/temp1_input", "61500\n")
    _write(root, "/sys/class/hwmon/hwmon6/temp1_label", "Tctl\n")
    _write(root, "/sys/class/hwmon/hwmon1/name", "acpi_fan\n")
    _write(root, "/sys/class/hwmon/hwmon1/fan1_input", "3200\n")
    _write(root, "/sys/class/power_supply/ACAD/online", "1\n")
    _write(root, "/sys/class/power_supply/BAT1/status", "Not charging\n")
    _write(root, "/sys/class/power_supply/BAT1/capacity", "76\n")
    _write(root, "/proc/4343/cgroup", "0::/system.slice/docker-fire.scope\n")
    _write(root, "/sys/fs/cgroup/system.slice/docker-fire.scope/memory.events", "low 0\nhigh 4\nmax 2\noom 1\noom_kill 1\n")
    _write(root, "/sys/fs/cgroup/system.slice/docker-fire.scope/cpu.stat", "usage_usec 5\nnr_periods 10\nnr_throttled 3\nthrottled_usec 99\n")
    _write(root, "/sys/fs/cgroup/system.slice/docker-fire.scope/memory.pressure", PSI)


def test_tick_reads_every_sensor(tmp_path):
    root = str(tmp_path / "fake")
    _fake_host(root)
    hs = hostsense.HostSense(["safety"], FakeSys(root), log_dir=str(tmp_path / "logs"))
    row = hs.tick()
    assert row["psi"]["memory"]["some"]["total"] == 1000000
    assert row["vmstat"]["pswpout"] == 9
    assert row["freq"] == {"min_khz": 2000000, "max_khz": 3000000, "mean_khz": 2500000}
    assert row["hwmon"]["k10temp:temp:Tctl"] == 61.5 and row["hwmon"]["acpi_fan:fan:1"] == 3200
    assert row["power"]["ACAD"]["online"] == 1 and row["power"]["BAT1"]["capacity"] == 76
    assert row["disk"]["wr_sectors"] == 16000
    assert set(row["containers"]) == {"safety-redis-1", "safety-fire-1"}          # the other project is not tracked
    fire = row["containers"]["safety-fire-1"]
    assert fire["restarts"] == 2 and fire["memory_events"]["oom_kill"] == 1 and fire["cpu_throttle"]["nr_throttled"] == 3
    assert fire["psi"]["memory"]["some"]["avg10"] == 0.16
    assert "errors" not in row


def test_logs_are_appended_without_gaps_or_repeats(tmp_path):
    root = str(tmp_path / "fake")
    _fake_host(root)
    sys_ = FakeSys(root)
    hs = hostsense.HostSense(["safety"], sys_, log_dir=str(tmp_path / "logs"), log_every=0)
    hs.tick()
    hs.tick()
    log_calls = [c for c in sys_.calls if c[:2] == ["docker", "logs"] and c[-1] == "safety-fire-1"]
    assert "--since" not in log_calls[0] and "--since" in log_calls[1]
    # the second window starts exactly where the first ended
    assert log_calls[1][log_calls[1].index("--since") + 1] == log_calls[0][log_calls[0].index("--until") + 1]
    text = open(tmp_path / "logs" / "safety-fire-1.log").read()
    assert text.count("line from safety-fire-1") == 2


def test_one_broken_probe_does_not_stop_the_row(tmp_path):
    root = str(tmp_path / "fake")
    _fake_host(root)

    class Bad(FakeSys):
        def run(self, cmd, timeout=20):
            raise RuntimeError("docker is gone")
    row = hostsense.HostSense(["safety"], Bad(root)).tick()
    assert "containers" in row["errors"] and row["psi"]["cpu"]["some"]["total"] == 1000000


def test_run_writes_files_and_provenance(tmp_path):
    root = str(tmp_path / "fake")
    _fake_host(root)
    out = str(tmp_path / "out")
    path = hostsense.run(["safety"], out, duration_s=0.5, interval=0.2, sys_=FakeSys(root))
    rows = hostsense.load(path)
    assert len(rows) >= 2 and os.path.exists(os.path.join(out, "provenance-start.txt")) and os.path.exists(os.path.join(out, "provenance-end.txt"))
    assert "## cmdline" in open(os.path.join(out, "provenance-start.txt")).read()
    assert os.path.exists(os.path.join(out, "logs", "safety-fire-1.log"))
    assert json.load(open(os.path.join(out, "meta.json")))["projects"] == ["safety"]
    assert not os.path.exists(os.path.join(out, "hostsense.pid"))        # removed again on a clean stop


# ── the summary ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _rows(n=11, stall_us_per_s=100000, swap_per_s=0, temp=60.0, oom_at=None):
    rows = []
    for i in range(n):
        t = 1000.0 + 10 * i
        rows.append({"t": t,
                     "psi": {"memory": {"some": {"avg10": 0, "avg60": 0, "avg300": 0, "total": int(stall_us_per_s * 10 * i)}}},
                     "vmstat": {"pswpin": 0, "pswpout": swap_per_s * 10 * i, "pgmajfault": 5 * 10 * i, "oom_kill": 1 if oom_at is not None and i >= oom_at else 0},
                     "cpu": {"all": dict(zip(hostsense.CPU_FIELDS, [100 * i, 0, 50 * i, 800 * i, 10 * i, 0, 0, 0])), "cores": [[1000 * i, 800 * i], [1000 * i, 400 * i]]},
                     "freq": {"mean_khz": 2500000 - 1000 * i, "min_khz": 2000000},
                     "hwmon": {"k10temp:temp:Tctl": temp + i},
                     "power": {"ACAD": {"online": 1}, "BAT1": {"status": "Not charging"}},
                     "disk": {"wr_sectors": 2000 * 10 * i, "io_busy_ms": 100 * 10 * i, "rd_sectors": 0},
                     "containers": {"safety-fire-1": {"restarts": 0 if i < 5 else 1, "status": "running", "oom_killed": False, "exit_code": 0,
                                                      "health": "healthy", "memory_events": {"high": 0, "max": i // 4, "oom": 0, "oom_kill": 0},
                                                      "cpu_throttle": {"nr_throttled": 2 * i}}}})
    return rows


def test_summary_rates_and_events():
    s = hostsense.summarize(_rows(stall_us_per_s=100000, swap_per_s=50, oom_at=7))
    assert s["samples"] == 11 and s["seconds"] == 100.0
    assert abs(s["psi"]["memory_some_stall_pct"]["mean"] - 10.0) < 0.01        # 100 ms stalled per second = 10 %
    assert abs(s["vm_per_s"]["pswpout"]["mean"] - 50.0) < 1e-6
    assert s["oom_kill_total"] == 1
    assert s["busiest_core_pct"]["max"] > 59                                   # core 1 was ~60 % busy
    assert s["hwmon"]["k10temp:temp:Tctl"]["max"] == 70.0
    assert s["on_ac_fraction"] == 1.0
    c = s["containers"]["safety-fire-1"]
    assert c["restarts_start_end"] == [0, 1] and c["memory_events_delta"]["max"] == 2 and c["cpu_throttled_periods_delta"] == 20
    assert abs(s["cpu_pct"]["iowait"]["mean"] - 100 * 10 / 960) < 0.01


def test_summary_window_and_empty():
    rows = _rows()
    w = hostsense.summarize(rows, t0=1030.0, t1=1060.0)
    assert w["samples"] == 4 and w["seconds"] == 30.0
    assert hostsense.summarize([]) == {"samples": 0}
    assert hostsense.render(hostsense.summarize(rows)).startswith("11 samples over 100 s")


def test_counter_reset_is_skipped_not_negative():
    rows = _rows(n=4)
    rows[2]["vmstat"]["pswpout"] = 0                                          # counter restarted (machine reboot would do this)
    assert all(v >= 0 for v in hostsense._rate(rows, lambda r: r["vmstat"]["pswpout"]))


def test_cli_has_the_commands():
    assert callable(cli.cmd_hostsense) and callable(cli.cmd_hostsense_summary)


def test_it_writes_its_own_pid_while_running(tmp_path):
    import signal
    root = str(tmp_path / "fake")
    _fake_host(root)
    out = str(tmp_path / "out")
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    seen = []
    orig_tick = hostsense.HostSense.tick

    def spy(self):
        seen.append(open(os.path.join(out, "hostsense.pid")).read())
        return orig_tick(self)
    hostsense.HostSense.tick = spy
    try:
        hostsense.run(["safety"], out, duration_s=0.4, interval=0.2, sys_=FakeSys(root))
    finally:
        hostsense.HostSense.tick = orig_tick
        for s_, h in old.items():
            signal.signal(s_, h)
    assert seen and seen[0] == str(os.getpid())
