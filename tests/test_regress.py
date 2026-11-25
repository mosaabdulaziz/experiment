"""The regression harness (tools/regress): its parsers on real log excerpts, the machine ids, the verdict. No GPU, no docker."""
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from regress import machine, parsers, results  # noqa: E402

LOG = (REPO / "tests" / "fixtures" / "regress" / "sidecar-attendance.log").read_text()


@pytest.fixture(scope="module")
def runtime():
    return parsers.parse_runtime(LOG, parsers.load_gpu_budget_parse(str(REPO)))


# ── the sidecar log (a real golden attendance replay) ────────────────────────────────────────────────────────────────

def test_boot_time_runs_from_the_starting_marker_to_the_first_fps_line(runtime):
    assert runtime["boot_to_first_fps_s"] == 43            # 12:44:17 -> 12:45:00 in the log


def test_a_cached_engine_is_not_a_rebuild(runtime):
    assert runtime["engine"]["cached"] is True and runtime["engine"]["rebuilt"] is False
    assert runtime["engine"]["cached_elements"] == ["pgie_fire", "sgie_face"]


def test_a_building_message_marks_the_engine_rebuilt_and_names_the_element():
    line = "0:00:05.1 78 0x1 INFO nvinfer gstnvinfer.cpp:725:gst_nvinfer_logger:<pgie> NvDsInferContext[UID 1]: Trying to create engine from model files"
    r = parsers.parse_runtime(LOG + "\n" + line + "\n")
    assert r["engine"]["rebuilt"] is True and r["engine"]["rebuilt_elements"] == ["pgie"]
    assert r["engine"]["cached"] is True                     # the secondary engines were still found


def test_fps_is_the_steady_state_per_camera(runtime):
    cam = runtime["fps"]["cam-1"]
    assert cam["median"] == 20.0 and cam["p10"] == 20.0 and cam["n"] >= 10


def test_the_ramp_up_readings_are_not_counted():
    text = "\n".join(f"12:00:{i:02d} [INFO] iris.sidecar: [cam-1] {v} fps" for i, v in enumerate([3.0, 9.0, 20.0, 20.0, 20.0]))
    assert parsers.parse_runtime(text)["fps"]["cam-1"] == {"median": 20.0, "p10": 20.0, "min": 20.0, "n": 3}


def test_drops_are_the_last_gauge_values(runtime):
    assert runtime["drops"] == {"qos_dropped_total": 0.0, "publisher_drops": 0.0, "pub_shed_total": 0.0, "stall_episodes": 0.0}
    r = parsers.parse_runtime("12:00:00 [INFO] iris.sidecar: [telemetry] gauge pub_shed_total_cam-1=4\n"
                              "12:00:01 [INFO] iris.sidecar: [telemetry] gauge pub_shed_total_cam-2=3\n"
                              "12:00:05 [INFO] iris.sidecar: [telemetry] gauge qos_dropped_total=7\n")
    assert r["drops"]["pub_shed_total"] == 7 and r["drops"]["qos_dropped_total"] == 7


def test_gpu_stages_come_from_engine_gpu_budget(runtime):
    stages = runtime["gpu_stages"]
    assert {"gpu.decode", "gpu.infer.pgie", "gpu.jpeg"} <= set(stages)          # decode rows of all cameras are one stage
    assert stages["gpu.infer.pgie"]["busy"] > 0 and stages["gpu.infer.pgie"]["avg_ms"] > 0
    assert runtime["gpu_total_busy"] == round(sum(s["busy"] for s in stages.values()), 2)


def test_docker_timestamps_give_the_same_boot_time():
    lines = []
    for i, line in enumerate(LOG.splitlines()):
        m = re.match(r"(\d\d):(\d\d):(\d\d) ", line)
        stamp = f"2026-11-05T{m.group(1)}:{m.group(2)}:{m.group(3)}.123456789Z" if m else "2026-11-05T12:44:18.000000000Z"
        lines.append(f"regress-deepstream-sidecar-1  | {stamp} {line}")
    assert parsers.parse_runtime("\n".join(lines))["boot_to_first_fps_s"] == 43


def test_a_clock_that_passes_midnight_does_not_give_a_negative_boot_time():
    text = ("23:59:50 [INFO] iris.sidecar: sidecar starting shard=0/1\n"
            "00:00:20 [INFO] iris.sidecar: [cam-1] 20.0 fps\n")
    assert parsers.parse_runtime(text)["boot_to_first_fps_s"] == 30


def test_ansi_colours_are_ignored():
    assert parsers.parse_runtime("\x1b[31m12:00:00 [INFO] iris.sidecar: sidecar starting\x1b[0m\n"
                                 "12:00:05 [INFO] iris.sidecar: [cam-1] 9.0 fps\n")["boot_to_first_fps_s"] == 5


# ── junit, locks, build output ───────────────────────────────────────────────────────────────────────────────────────

JUNIT = """<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="2" tests="6" time="3.5">
<testcase classname="tests.test_a" name="test_ok" time="0.10"/>
<testcase classname="tests.test_a" name="test_slow" time="2.50"/>
<testcase classname="tests.test_b" name="test_bad" time="0.20"><failure message="x"/></testcase>
<testcase classname="tests.test_b" name="test_err" time="0.10"><error message="y"/></testcase>
<testcase classname="tests.test_c" name="test_skip" time="0.0"><skipped/></testcase>
<testcase classname="tests.test_c" name="test_skip2" time="0.0"><skipped/></testcase>
</testsuite></testsuites>"""


def test_junit_counts_failures_and_the_slowest_tests():
    j = parsers.parse_junit(JUNIT, slowest=2)
    assert (j["tests"], j["failures"], j["errors"], j["skipped"]) == (6, 1, 1, 2)
    assert j["failed"] == ["tests.test_b::test_bad", "tests.test_b::test_err"]
    assert j["slowest"][0] == ["tests.test_a::test_slow", 2.5] and len(j["slowest"]) == 2 and j["time_s"] == 3.5


def test_lock_packages_exclude_the_project_itself():
    lock = 'version = 1\n[[package]]\nname = "app"\nversion = "0"\nsource = { virtual = "." }\n' \
           '[[package]]\nname = "a"\nversion = "1"\nsource = { registry = "https://pypi.org/simple" }\n' \
           '[[package]]\nname = "b"\nversion = "2"\nsource = { registry = "https://pypi.org/simple" }\n'
    assert parsers.count_lock_packages(lock) == 2


def test_the_repos_own_locks_parse():
    for lock in REPO.glob("products/*/uv.lock"):
        assert parsers.count_lock_packages(lock.read_text()) > 5


def test_bundle_summary_finds_the_largest_chunk():
    files = [("index.html", 500), ("assets/app.js", 9000), ("assets/vendor.js", 40000), ("assets/app.css", 2000)]
    b = parsers.bundle_summary(files)
    assert b["total_bytes"] == 51500 and b["js_bytes"] == 49000 and b["css_bytes"] == 2000
    assert b["largest_chunk"] == {"file": "assets/vendor.js", "bytes": 40000}


# ── the machine ──────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gpu, ident", [
    ("NVIDIA GeForce RTX 5050 Laptop GPU", "rtx-5050-laptop"), ("NVIDIA GeForce RTX 5090", "rtx-5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "rtx-pro-6000-maxq"),
    ("NVIDIA Jetson AGX Orin Developer Kit", "jetson-agx-orin-64gb"), ("Tesla T4", "tesla-t4"), ("", "unknown-gpu")])
def test_machine_ids(gpu, ident, monkeypatch):
    monkeypatch.delenv("IRIS_MACHINE_ID", raising=False)
    monkeypatch.delenv("CI", raising=False)              # a CI job without a GPU is `ci-cpu` (next test), whatever it is named
    assert machine.machine_id(gpu) == ident


def test_a_ci_job_without_a_gpu_is_filed_as_ci_cpu(monkeypatch):
    monkeypatch.delenv("IRIS_MACHINE_ID", raising=False)
    monkeypatch.setenv("CI", "true")
    assert machine.machine_id("") == "ci-cpu"
    assert machine.machine_id("NVIDIA GeForce RTX 5090") == "rtx-5090"       # a GPU runner keeps its own id


def test_the_machine_id_can_be_set_by_hand(monkeypatch):
    monkeypatch.setenv("IRIS_MACHINE_ID", "lab-box")
    assert machine.machine_id("NVIDIA GeForce RTX 5090") == "lab-box"


def test_tegrastats_line_of_the_orin_is_read():
    line = ("11-10-2026 10:49:30 RAM 2115/62828MB (lfb 2x4MB) SWAP 0/31414MB (cached 0MB) CPU [1%@729,0%@729] GR3D_FREQ 0% "
            "cpu@39.75C soc2@36.5C")
    t = machine.parse_tegrastats(line)
    assert t["ram_used_mb"] == 2115 and t["ram_total_mb"] == 62828 and t["gpu_util_pct"] == 0
    assert t["temps_c"]["cpu"] == 39.75


def test_compute_apps_are_listed():
    apps = machine.parse_compute_apps("1234, python3, 2048\n99, /usr/bin/Xorg, 10\n")
    assert [a["pid"] for a in apps] == [1234, 99] and apps[0]["mem_mib"] == 2048
    assert machine.parse_compute_apps("") == []


# ── the verdict ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _result(unit=None, golden=None):
    fp = {"machine_id": "m", "gpu": "g", "git": {"short": "abc", "dirty": False}}
    r = results.new_result(fp, {"idle": True, "reasons": []})
    r["correctness"] = {"unit": unit or {}, "golden": golden or {}}
    return r


def _suite(failed=()):
    return {"ran": True, "tests": 10, "failures": len(failed), "errors": 0, "skipped": 0, "time_s": 1.0, "wall_s": 2.0,
            "failed": list(failed), "slowest": []}


def test_a_green_run_passes():
    assert results.finish(_result(unit={"core": _suite()}), {})["verdict"]["pass"] is True


def test_a_new_failing_test_fails_the_verdict():
    v = results.finish(_result(unit={"core": _suite(["tests.t::test_x"])}), {})["verdict"]
    assert v["pass"] is False and "tests.t::test_x" in v["reasons"][0]


def test_a_known_failure_does_not_fail_it_but_is_reported():
    known = {"tests.t::test_x": "stale golden"}
    v = results.finish(_result(unit={"core": _suite(["tests.t::test_x"])}), known)["verdict"]
    assert v["pass"] is True and v["known_failures"] == ["tests.t::test_x"]


def test_a_suite_or_replay_that_did_not_run_fails_it():
    r = _result(unit={"barber": {"ran": False, "error": "uv is not installed"}},
                golden={"zones": {"ran": False, "error": "image missing"}})
    v = results.finish(r, {})["verdict"]
    assert v["pass"] is False and len(v["reasons"]) == 2


def test_a_replay_that_recorded_nothing_fails_it():
    g = {"ran": True, "entries": 0, "wall_s": 1, "gpu_memory": {}, "runtime": {}}
    assert results.finish(_result(golden={"zones": g}), {})["verdict"]["pass"] is False


def test_the_known_failures_file_is_read(tmp_path):
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "known_failures.txt").write_text("# header\ntests.a::test_b  # why it fails\n\n")
    assert results.load_known_failures(str(tmp_path)) == {"tests.a::test_b": "why it fails"}


def test_a_results_file_round_trips_and_renders(tmp_path, runtime):
    r = _result(unit={"core": _suite()})
    r["fingerprint"].update({"driver": "1", "gpu": "GPU"})
    r["build"] = {"images": {"img": 1_000_000}, "frontend_bundle": {"skipped": "not built"}, "lock_packages": {"uv.lock": 3}}
    results.finish(r, {})
    path = results.write(r, str(tmp_path / "x.json"))
    back = json.loads(Path(path).read_text())
    assert back["schema"] == "regress.v1" and back["machine_id"] == "m"
    text = results.render(back)
    assert "PASS" in text and "unit suites" in text and "uv.lock" in text


# ── the idle check ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_background_util_is_the_median_so_one_spike_does_not_decide():
    readings = iter([5, 95, 6, 4, 5])
    assert machine.background_util(5, 0, sample=lambda: {"util_pct": next(readings)}, sleep=lambda s: None) == 5


def test_background_util_skips_unreadable_samples():
    readings = iter([None, 8, None])
    assert machine.background_util(3, 0, sample=lambda: {"util_pct": next(readings)}, sleep=lambda s: None) == 8
    assert machine.background_util(2, 0, sample=lambda: {"util_pct": None}, sleep=lambda s: None) is None


def test_a_busy_gpu_is_not_idle_and_the_limit_can_be_raised(monkeypatch):
    monkeypatch.setattr(machine, "run", lambda *a, **k: "")
    monkeypatch.setattr(machine, "is_jetson", lambda: False)
    monkeypatch.setattr(machine, "on_battery", lambda: False)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 40, "mem_used_mib": 56, "mem_total_mib": 8000, "source": "t"})
    monkeypatch.delenv("REGRESS_IDLE_GPU_UTIL", raising=False)
    rep = machine.idle_report(background=40.0)
    assert rep["idle"] is False and "40% busy" in rep["reasons"][0]
    monkeypatch.setenv("REGRESS_IDLE_GPU_UTIL", "60")                    # a laptop that drives its display
    rep = machine.idle_report(background=40.0)
    assert rep["idle"] is True and rep["gpu"]["background_util_pct"] == 40.0 and rep["gpu"]["idle_limit_pct"] == 60.0


def test_other_stacks_and_a_battery_make_it_not_idle(monkeypatch):
    monkeypatch.setattr(machine, "run", lambda *a, **k: "iris-deepstream-sidecar-1\nregress-redis-1\nsome-db\n" if a[0] == "docker" else "")
    monkeypatch.setattr(machine, "is_jetson", lambda: False)
    monkeypatch.setattr(machine, "on_battery", lambda: True)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 0, "mem_total_mib": 1, "source": "t"})
    rep = machine.idle_report(own_prefix="regress", background=0.0)
    assert rep["idle"] is False and rep["other_stacks"] == ["iris-deepstream-sidecar-1"]      # regress-* is ours
    assert any("battery" in r for r in rep["reasons"])


# ── the golden collector (docker and the GPU faked) ──────────────────────────────────────────────────────────────────

def test_the_replay_gets_a_config_path_relative_to_the_repo_root(tmp_path, monkeypatch):
    """The sidecar container sees the repo at /workspace/iris, so a host path (/home/...) makes it crash and the recorder
    wait out its whole timeout. The first real run found this."""
    from regress import collectors
    (tmp_path / "tests" / "configs").mkdir(parents=True)
    (tmp_path / "tests" / "configs" / "golden-zones.yaml").write_text("x: 1\n")
    calls = []

    def fake_sh(cmd, cwd=None, env=None, timeout=None):
        calls.append(list(cmd))
        if cmd[0].endswith("engine_golden_run.sh"):
            Path(cmd[2]).write_text('{"a": 1}\n{"a": 2}\n')
        out = LOG if cmd[:3] == ["docker", "compose", "logs"] else ""
        return __import__("subprocess").CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(collectors, "sh", fake_sh)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 100, "mem_total_mib": 1, "source": "t"})
    res = collectors.collect_golden(str(tmp_path), ["zones"], str(tmp_path / "out"), "slim:img", "mtmc:img",
                                    parsers.load_gpu_budget_parse(str(REPO)))
    run = next(c for c in calls if c[0].endswith("engine_golden_run.sh"))
    assert run[1] == "tests/configs/golden-zones.yaml" and not run[1].startswith("/")
    assert res["zones"]["ran"] and res["zones"]["entries"] == 2 and res["zones"]["runtime"]["boot_to_first_fps_s"] == 43
    assert any(c[:3] == ["docker", "compose", "down"] for c in calls)            # the project is always removed


def test_multicam_is_replayed_on_the_mtmc_image(tmp_path, monkeypatch):
    from regress import collectors
    (tmp_path / "tests" / "configs").mkdir(parents=True)
    for n in ("multicam", "zones"):
        (tmp_path / "tests" / "configs" / f"golden-{n}.yaml").write_text("x: 1\n")
    tagged = []

    def fake_sh(cmd, cwd=None, env=None, timeout=None):
        if cmd[:2] == ["docker", "tag"]:
            tagged.append(cmd[2])
        if cmd[0].endswith("engine_golden_run.sh"):
            Path(cmd[2]).write_text("{}\n")
        return __import__("subprocess").CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(collectors, "sh", fake_sh)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 1, "mem_total_mib": 1, "source": "t"})
    collectors.collect_golden(str(tmp_path), ["multicam", "zones"], str(tmp_path / "o"), "slim:img", "mtmc:img", None)
    assert tagged == ["mtmc:img", "slim:img"]


def test_a_missing_tool_is_exit_127_not_a_crash(tmp_path):
    """CI's python:3.11-slim has no docker: the build section must report the images as unavailable and carry on."""
    from regress import collectors
    r = collectors.sh(["definitely-not-a-real-tool-xyz", "--version"])
    assert r.returncode == 127 and "not found" in r.stderr
    out = collectors.collect_build(str(tmp_path), images=["img:latest"], build_frontend=False)
    assert out["images"] == {"img:latest": None}
    assert "skipped" in out["frontend_bundle"]


# ── the hardware counters (tools/regress/gpucounters.py) ─────────────────────────────────────────────────────────────

from regress import gpucounters as gc  # noqa: E402

# the formats are those the real tools printed on the RTX 5050 laptop (2026-11-10)
NVML_LINE = "39, 14, 0, 0, 0, 0, 570, 3090, 810, 8.78, 41, 0x0000000000000000, 56"
DCGM_TEXT = '''# HELP DCGM_FI_PROF_SM_ACTIVE SM active ratio
# TYPE DCGM_FI_PROF_SM_ACTIVE gauge
DCGM_FI_PROF_SM_ACTIVE{gpu="0",UUID="GPU-168e",pci_bus_id="00000000:01:00.0",device="nvidia0",modelName="NVIDIA GeForce RTX 5050 Laptop GPU"} 0.2511
DCGM_FI_PROF_SM_OCCUPANCY{gpu="0",UUID="GPU-168e"} 0.1042
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",UUID="GPU-168e"} 0
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="GPU-168e"} 0.31
DCGM_FI_PROF_SM_ACTIVE{gpu="1",UUID="GPU-other"} 0.9
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-168e"} 40
'''


def test_an_nvml_row_is_read_with_its_engines_and_clocks():
    r = gc.parse_nvml_row(NVML_LINE)
    assert r["gpu_util"] == 39 and r["mem_ctrl_util"] == 14 and r["nvdec_util"] == 0 and r["nvjpg_util"] == 0
    assert r["sm_clock_mhz"] == 570 and r["sm_clock_max_mhz"] == 3090 and r["power_w"] == 8.78 and r["throttle_mask"] == 0


def test_not_available_values_are_none_not_zero():
    r = gc.parse_nvml_row("39, 14, [N/A], 0, 0, [Not Supported], 570, 3090, 810, [N/A], 41, 0x0, 56")
    assert r["nvenc_util"] is None and r["ofa_util"] is None and r["power_w"] is None and r["gpu_util"] == 39
    assert gc.parse_nvml_row("not a row") == {}


def test_dmon_pcie_lines():
    assert gc.parse_dmon_pcie("# gpu   rxpci   txpci") == {}
    assert gc.parse_dmon_pcie("    0     12      3") == {"pcie_rx_mb_s": 12.0, "pcie_tx_mb_s": 3.0}


def test_dcgm_metrics_are_read_for_one_gpu_only():
    m = gc.parse_dcgm_metrics(DCGM_TEXT, gpu="0")
    assert m == {"sm_active": 0.2511, "sm_occupancy": 0.1042, "tensor_active": 0.0, "dram_active": 0.31}
    assert gc.parse_dcgm_metrics(DCGM_TEXT, gpu="1") == {"sm_active": 0.9}
    assert gc.parse_dcgm_metrics("") == {}


def test_a_jetson_line_gives_the_media_engines_and_memory_load():
    line = ("11-10-2026 10:49:30 RAM 2115/62828MB (lfb 2x4MB) SWAP 0/31414MB CPU [1%@729,0%@729] EMC_FREQ 3%@2133 GR3D_FREQ 12% "
            "NVDEC 729 NVENC off NVJPG 0 cpu@39.75C VDD_GPU_SOC 4023mW/4023mW VDD_CPU_CV 1200mW/1200mW")
    t = gc.parse_tegrastats_line(line)
    assert t["gr3d_pct"] == 12 and t["emc_pct"] == 3 and t["nvdec_mhz"] == 729 and t["nvenc_mhz"] == 0 and t["nvjpg_mhz"] == 0
    assert t["power_mw"]["VDD_GPU_SOC"] == 4023 and t["power_vdd_gpu_soc_mw"] == 4023 and t["ram_used_mb"] == 2115


def test_summaries_give_median_p95_max_and_the_throttled_share():
    rows = [{"t": i, "gpu_util": v, "throttle_mask": m} for i, (v, m) in enumerate(
        [(10, 0), (20, 0), (30, 0x4), (40, 0x1), (100, 0x20)])]
    s = gc.summarize(rows)
    assert s["gpu_util"]["median"] == 30 and s["gpu_util"]["max"] == 100 and s["gpu_util"]["n"] == 5
    assert s["throttled_fraction"] == 0.4            # 0x4 and 0x20 count; 0x1 (GPU idle) does not


def _s(**kv):
    return {k: {"median": v, "p95": v, "max": v, "mean": v, "n": 5} for k, v in kv.items()}


def test_hint_busy_gpu_with_few_active_sms():
    h = gc.hints(_s(gpu_util=70), _s(sm_active=0.2), {})
    assert any("overstates" in x and "20%" in x for x in h)


def test_hint_memory_bound_and_idle_tensor_cores():
    h = gc.hints(_s(gpu_util=90), _s(sm_active=0.7, dram_active=0.8, tensor_active=0.0), {})
    assert any("memory-bound" in x for x in h) and any("tensor cores are idle" in x for x in h)


def test_hint_decoder_near_its_limit():
    assert any("NVDEC" in x for x in gc.hints(_s(nvdec_util=92), {}, {}))
    assert gc.hints(_s(nvdec_util=40, gpu_util=30), _s(sm_active=0.6), {}) == []      # nothing to point at: no hint


def test_hint_throttling_and_low_clocks():
    n = {**_s(gpu_util=80, sm_clock_mhz=900, sm_clock_max_mhz=3000), "throttled_fraction": 0.2}
    h = gc.hints(n, {}, {})
    assert any("throttled" in x for x in h) and any("clocks stayed low" in x for x in h)


def test_the_steady_window_runs_from_the_first_fps_line_to_the_end_of_stream():
    lines = ["svc  | 2026-11-05T12:44:17.123456789Z 12:44:17 [INFO] iris.sidecar: sidecar starting shard=0/1",
             "svc  | 2026-11-05T12:45:00.000000000Z 12:45:00 [INFO] iris.sidecar: [cam-1] 11.6 fps",
             "svc  | 2026-11-05T12:46:13.000000000Z 12:46:13 [WARNING] iris.sidecar: EOS — pipeline reached end of stream"]
    w = parsers.parse_runtime("\n".join(lines))["steady_window"]
    assert w is not None and round(w[1] - w[0]) == 73
    assert parsers.parse_runtime(LOG)["steady_window"] is None        # a clock-only log has no absolute time


def test_the_dcgm_image_is_the_one_the_compose_file_pins():
    img = gc.DcgmSampler._image(str(REPO))
    assert img.startswith("nvidia/dcgm-exporter:") and "@sha256:" in img


def test_a_replay_gets_its_counter_window_from_the_log(tmp_path, monkeypatch):
    from regress import collectors
    (tmp_path / "tests" / "configs").mkdir(parents=True)
    (tmp_path / "tests" / "configs" / "golden-zones.yaml").write_text("x: 1\n")
    logtext = "\n".join(["svc | 2026-11-05T12:44:17.000000000Z 12:44:17 [INFO] iris.sidecar: sidecar starting",
                         "svc | 2026-11-05T12:45:00.000000000Z 12:45:00 [INFO] iris.sidecar: [cam-1] 20.0 fps",
                         "svc | 2026-11-05T12:46:13.000000000Z 12:46:13 [WARNING] iris.sidecar: EOS — pipeline reached end of stream"])

    def fake_sh(cmd, cwd=None, env=None, timeout=None):
        if cmd[0].endswith("engine_golden_run.sh"):
            Path(cmd[2]).write_text("{}\n")
        out = logtext if cmd[:3] == ["docker", "compose", "logs"] else ""
        return __import__("subprocess").CompletedProcess(cmd, 0, out, "")

    class FakeCounters:
        def window(self, lo, hi):
            return {"asked": [lo, hi], "hints": [], "dcgm": {}, "nvml": {}}

    monkeypatch.setattr(collectors, "sh", fake_sh)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 1, "mem_total_mib": 1, "source": "t"})
    res = collectors.collect_golden(str(tmp_path), ["zones"], str(tmp_path / "o"), "s", "m", None, FakeCounters())
    lo, hi = res["zones"]["gpu_counters"]["asked"]
    assert round(hi - lo) == 73                          # not the recorder's idle tail, not the boot


def test_a_stage_busy_by_its_timer_while_the_sms_idle_is_pointed_out():
    stages = {"gpu.infer.sgie_plate": {"busy": 89.3, "avg_ms": 1, "max_ms": 1, "n": 5},
              "gpu.decode": {"busy": 52.0, "avg_ms": 1, "max_ms": 1, "n": 5},
              "gpu.infer.pgie": {"busy": 4.2, "avg_ms": 1, "max_ms": 1, "n": 5}}
    h = gc.stage_hints({"sm_active": {"median": 0.0, "p95": 0.43, "max": 0.49, "mean": 0.1, "n": 85}}, stages)
    assert len(h) == 1 and "sgie_plate" in h[0] and "43%" in h[0]            # decode is a hardware engine, pgie is not busy
    assert gc.stage_hints({}, stages) == []                                  # no SM counter: nothing to compare with
    busy = {"sm_active": {"median": 0.7, "p95": 0.8, "max": 0.9, "mean": 0.7, "n": 5}}
    assert gc.stage_hints(busy, stages) == []                                # the SMs really were busy


def test_the_background_uses_engine_activity_when_dcgm_has_it(monkeypatch):
    c = gc.GpuCounters.__new__(gc.GpuCounters)
    c.window = lambda lo, hi: {"nvml": {"gpu_util": {"median": 39.5}}, "dcgm": {"gr_engine_active": {"median": 0.036}}, "jetson": {}}
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    bg = gc.GpuCounters.background(c, 0)
    assert bg["pct"] == 3.6 and "DCGM" in bg["source"] and bg["nvidia_smi_util_pct"] == 39.5     # the display: 40% vs 3.6%
    c.window = lambda lo, hi: {"nvml": {"gpu_util": {"median": 39.5}}, "dcgm": {}, "jetson": {}}
    assert gc.GpuCounters.background(c, 0)["pct"] == 39.5


def test_the_idle_reason_names_where_the_number_came_from(monkeypatch):
    monkeypatch.setattr(machine, "run", lambda *a, **k: "")
    monkeypatch.setattr(machine, "is_jetson", lambda: False)
    monkeypatch.setattr(machine, "on_battery", lambda: False)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 0, "mem_total_mib": 1, "source": "t"})
    monkeypatch.delenv("REGRESS_IDLE_GPU_UTIL", raising=False)
    rep = machine.idle_report(background=45.0, source="DCGM graphics-engine activity")
    assert not rep["idle"] and "DCGM graphics-engine activity" in rep["reasons"][0]


# ── a stalled DCGM profiling session (seen on the RTX 5050: SM active 0.0 in all 73 samples, engine 68% active) ──────

def _ser(median, mx=None):
    return {"median": median, "p95": mx if mx is not None else median, "max": mx if mx is not None else median,
            "mean": median, "n": 73}


def test_sm_counters_at_zero_while_the_engine_was_busy_are_invalid():
    stalled = {"gr_engine_active": _ser(0.68), "sm_active": _ser(0.0), "sm_occupancy": _ser(0.0), "dram_active": _ser(0.096)}
    ok, why = gc.validate_dcgm(stalled)
    assert not ok and "68%" in why and "stalled" in why


def test_bursty_work_with_some_sm_activity_is_valid():
    multicam = {"gr_engine_active": _ser(0.142), "sm_active": _ser(0.0, mx=0.494)}
    assert gc.validate_dcgm(multicam)[0] is True                      # median 0 but the SMs did work in bursts
    assert gc.validate_dcgm({"gr_engine_active": _ser(0.03), "sm_active": _ser(0.0)})[0] is True       # an idle GPU, genuinely
    assert gc.validate_dcgm({})[0] is True                            # nothing measured: nothing to doubt


def test_an_invalid_window_drops_the_sm_group_but_keeps_the_rest():
    c = gc.GpuCounters.__new__(gc.GpuCounters)
    c.streams, c.jetson, c.dcgm_status = {}, False, {"supported": True}

    class D:
        samples = [{"t": i, "gr_engine_active": 0.7, "sm_active": 0.0, "sm_occupancy": 0.0, "tensor_active": 0.0,
                    "dram_active": 0.1} for i in range(10)]
    c.dcgm = D()
    w = c.window(0, 100)
    assert w["dcgm_valid"] is False and "stalled" in w["dcgm_invalid_reason"]
    assert set(w["dcgm"]) == {"gr_engine_active", "dram_active"}      # sm_active, occupancy, tensor are not reported
    assert not any("overstates" in h for h in w["hints"])             # and no hint is built on them


def test_a_replay_with_invalid_counters_is_run_again_on_a_fresh_exporter(tmp_path, monkeypatch):
    from regress import collectors
    (tmp_path / "tests" / "configs").mkdir(parents=True)
    (tmp_path / "tests" / "configs" / "golden-zones.yaml").write_text("x: 1\n")
    runs = []

    def fake_sh(cmd, cwd=None, env=None, timeout=None):
        if cmd[0].endswith("engine_golden_run.sh"):
            runs.append(cmd[1])
            Path(cmd[2]).write_text("{}\n")
        return __import__("subprocess").CompletedProcess(cmd, 0, "", "")

    class Counters:
        restarts = 0

        def window(self, lo, hi):
            first = self.restarts == 0
            return {"hints": [], "dcgm": {}, "nvml": {}, "dcgm_valid": not first,
                    "dcgm_invalid_reason": "SM counters read 0 ... stalled" if first else ""}

        def restart_dcgm(self):
            self.restarts += 1

    monkeypatch.setattr(collectors, "sh", fake_sh)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 1, "mem_total_mib": 1, "source": "t"})
    c = Counters()
    res = collectors.collect_golden(str(tmp_path), ["zones"], str(tmp_path / "o"), "s", "m", None, c)
    assert len(runs) == 2 and c.restarts == 1                          # replayed once more, after a restart
    assert res["zones"]["gpu_counters"]["dcgm_valid"] is True
    assert "stalled" in res["zones"]["gpu_counters"]["retried"]["first_attempt"]


def test_valid_counters_are_not_retried(tmp_path, monkeypatch):
    from regress import collectors
    (tmp_path / "tests" / "configs").mkdir(parents=True)
    (tmp_path / "tests" / "configs" / "golden-zones.yaml").write_text("x: 1\n")
    runs = []

    def fake_sh(cmd, cwd=None, env=None, timeout=None):
        if cmd[0].endswith("engine_golden_run.sh"):
            runs.append(1)
            Path(cmd[2]).write_text("{}\n")
        return __import__("subprocess").CompletedProcess(cmd, 0, "", "")

    class Counters:
        def window(self, lo, hi):
            return {"hints": [], "dcgm": {}, "nvml": {}, "dcgm_valid": True, "dcgm_invalid_reason": ""}

    monkeypatch.setattr(collectors, "sh", fake_sh)
    monkeypatch.setattr(machine, "gpu_sample", lambda: {"util_pct": 0, "mem_used_mib": 1, "mem_total_mib": 1, "source": "t"})
    collectors.collect_golden(str(tmp_path), ["zones"], str(tmp_path / "o"), "s", "m", None, Counters())
    assert len(runs) == 1


def test_golden_env_puts_the_running_interpreter_first_on_path():
    """The replay script starts its recorder as `python3`; inside a virtualenv that must be the venv's (it has redis)."""
    import os
    import sys

    from regress import collectors
    env = collectors.golden_env()
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(os.path.abspath(sys.executable))



def test_proximity_is_replayed_only_when_its_clip_and_baseline_are_pulled(tmp_path):
    from regress import collectors
    names, skipped = collectors.default_golden_configs(str(tmp_path))
    assert names == collectors.GOLDEN_CONFIGS and "just data-pull golden" in skipped["proximity"]
    for rel in collectors.OPTIONAL_GOLDEN["proximity"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    names, skipped = collectors.default_golden_configs(str(tmp_path))
    assert names == collectors.GOLDEN_CONFIGS + ["proximity"] and skipped == {}
    assert "proximity" in collectors.SEMANTIC             # the replay's diff mode, the same as engine_golden_check.sh's


def test_the_check_script_compares_proximity_semantically():
    import os
    text = open(os.path.join(os.path.dirname(__file__), "..", "tools", "engine_golden_check.sh")).read()
    assert 'SEMANTIC="${SEMANTIC:-fire barber barber-faces proximity}"' in text


def test_a_partly_stalled_dcgm_window_is_invalid():
    """2026-11-11: fire's SM median read 0 while the graphics engine was 95% active (some samples up to 0.6)."""
    from regress import gpucounters
    stat = lambda m, mx: {"median": m, "p95": mx, "max": mx, "mean": m, "n": 70}   # noqa: E731
    assert not gpucounters.validate_dcgm({"gr_engine_active": stat(0.95, 0.99), "sm_active": stat(0.0, 0.6)})[0]
    assert not gpucounters.validate_dcgm({"gr_engine_active": stat(0.95, 0.99), "sm_active": stat(0.4, 0.6)})[0]
    assert gpucounters.validate_dcgm({"gr_engine_active": stat(0.68, 0.75), "sm_active": stat(0.61, 0.67)})[0]      # attendance
    assert gpucounters.validate_dcgm({"gr_engine_active": stat(0.13, 0.18), "sm_active": stat(0.0, 0.6)})[0]       # multicam: bursty, GPU mostly idle
