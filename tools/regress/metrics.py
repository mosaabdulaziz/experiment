"""A results file as a flat list of named metrics, so that two files (or ten) can be compared metric by metric.

Every metric has an id (`runtime.attendance.fps_p10`), a value, a kind and a direction:

  kind       correctness  the gate: a change here FAILS a merge request
             efficiency   a change here WARNS (FAILS only with --strict)
  direction  lower        a bigger number is worse (times, sizes, memory, drops)
             higher       a smaller number is worse (fps, tests run)
             both         a change either way is notable (utilisation counters: the same work using fewer SMs may be an
                          optimisation or the pipeline starving, only a person can say)
             equal        any change is notable (package counts)
             zero         must be 0 (failures)

Nothing here runs anything: it only reads the file `regress run` wrote (schema regress.v1).
"""
from dataclasses import dataclass

from . import gpucounters

# the hardware counters compared between runs (the rest are in the file for reading, not for gating)
DCGM_COMPARED = ("sm_active", "sm_occupancy", "tensor_active", "dram_active", "gr_engine_active")
NVML_COMPARED = ("gpu_util", "mem_ctrl_util", "nvdec_util", "nvjpg_util", "power_w")


@dataclass(frozen=True)
class Metric:
    id: str
    value: object
    kind: str
    direction: str
    unit: str = ""


def _add(out: dict, mid: str, value, kind: str, direction: str, unit: str = "") -> None:
    if value is not None:
        out[mid] = Metric(mid, value, kind, direction, unit)


def flatten(result: dict) -> dict:
    """{metric id: Metric} of one results file. A section that did not run simply contributes nothing."""
    out: dict = {}
    for suite, s in (result.get("correctness", {}).get("unit") or {}).items():
        if not s.get("ran", True) or "tests" not in s:
            continue
        _add(out, f"unit.{suite}.tests", s["tests"], "correctness", "higher", "tests")
        _add(out, f"unit.{suite}.failing", s["failures"] + s["errors"], "correctness", "zero", "tests")
        _add(out, f"unit.{suite}.time_s", s.get("wall_s", s.get("time_s")), "efficiency", "lower", "s")

    for cfg, g in (result.get("correctness", {}).get("golden") or {}).items():
        _add(out, f"golden.{cfg}.ran", bool(g.get("ran")), "correctness", "equal")
        if g.get("ran"):
            _add(out, f"golden.{cfg}.entries", g.get("entries"), "correctness", "equal", "entries")

    for cfg, rt in (result.get("runtime") or {}).items():
        eng = rt.get("engine", {})
        if not eng.get("rebuilt"):                    # a rebuilt TensorRT engine makes the boot time meaningless
            _add(out, f"runtime.{cfg}.boot_s", rt.get("boot_to_first_fps_s"), "efficiency", "lower", "s")
        fps = rt.get("fps") or {}
        if fps:
            _add(out, f"runtime.{cfg}.fps_p10", min(c["p10"] for c in fps.values()), "efficiency", "higher", "fps")
            _add(out, f"runtime.{cfg}.fps_median", min(c["median"] for c in fps.values()), "efficiency", "higher", "fps")
        _add(out, f"runtime.{cfg}.drops", sum((rt.get("drops") or {}).values()), "efficiency", "lower", "count")
        _add(out, f"runtime.{cfg}.stages_sum", rt.get("gpu_total_busy"), "efficiency", "lower", "%")
        _add(out, f"runtime.{cfg}.gpu_added_mib", (rt.get("gpu_memory") or {}).get("added_mib"), "efficiency", "lower", "MiB")
        for stage, v in (rt.get("gpu_stages") or {}).items():
            _add(out, f"runtime.{cfg}.stage.{stage}.busy", v["busy"], "efficiency", "lower", "%")
            _add(out, f"runtime.{cfg}.stage.{stage}.avg_ms", v["avg_ms"], "efficiency", "lower", "ms")
        for stage, v in (rt.get("cpu_stages") or {}).items():
            _add(out, f"runtime.{cfg}.cpu_stage.{stage}.avg_ms", v.get("avg_ms"), "efficiency", "lower", "ms")
            _add(out, f"runtime.{cfg}.cpu_stage.{stage}.busy", v.get("busy"), "efficiency", "lower", "%")
        fl = rt.get("flight") or {}
        _add(out, f"runtime.{cfg}.flight_cpu_pct", fl.get("cpu_pct_max"), "efficiency", "lower", "%")
        _add(out, f"runtime.{cfg}.main_loop_lag_s", fl.get("main_loop_lag_max_s"), "efficiency", "lower", "s")
        res = rt.get("resources") or {}
        for key, unit in (("cpu_cores", "cores"), ("rss_anon_mib", "MiB"), ("cgroup_peak_mib", "MiB"), ("threads", "threads"), ("fds", "fds")):
            _add(out, f"runtime.{cfg}.res.{key}", res.get(key), "efficiency", "lower", unit)
        gc = rt.get("gpu_counters") or {}
        # re-checked here, not only trusted from the file: results written before a rule existed carry the stalled windows.
        # Only the SM group is dropped (SM active, occupancy, the pipes): graphics-engine and DRAM activity were fine.
        sm_ok = gc.get("dcgm_valid", True) and gpucounters.validate_dcgm(gc.get("dcgm") or {})[0]
        for k in DCGM_COMPARED:
            s = (gc.get("dcgm") or {}).get(k)
            if s and (sm_ok or k not in gpucounters.SM_GROUP):
                _add(out, f"counters.{cfg}.{k}.median", s["median"], "efficiency", "both", "ratio")
                _add(out, f"counters.{cfg}.{k}.p95", s["p95"], "efficiency", "both", "ratio")
        for k in NVML_COMPARED:
            s = (gc.get("nvml") or {}).get(k)
            if s:
                _add(out, f"counters.{cfg}.{k}.median", s["median"], "efficiency", "both", "")
        for k, s in (gc.get("jetson") or {}).items():          # tegrastats (GR3D, EMC, rails): present on a Jetson only
            if isinstance(s, dict) and "median" in s and k.startswith(("gr3d", "emc")):
                _add(out, f"counters.{cfg}.{k}.median", s["median"], "efficiency", "both", "%")

    for cfg, c in (result.get("capacity") or {}).items():          # `regress capacity`
        if "@af" in cfg:
            # at a fixed analysis rate the pipeline is capped at that rate, so the ceiling says nothing; the cameras kept up with do. That count
            # depends on how far the ladder went, so it is only comparable between runs with the same camera counts (the 5 fps cells: 1, 2, 4, 8)
            _add(out, f"capacity.{cfg}.realtime_cameras", c.get("realtime_cameras_tested_ok"), "efficiency", "higher", "cameras")
        else:
            # the un-paced 1-camera ceiling is one number, comparable whatever ladder a run climbed; the cameras kept up with are not gated
            _add(out, f"capacity.{cfg}.ceiling_fps", c.get("ceiling_fps_1cam"), "efficiency", "higher", "fps")

    b = result.get("build") or {}
    for img, size in (b.get("images") or {}).items():
        _add(out, f"build.image.{img}", size, "efficiency", "lower", "bytes")
    fe = b.get("frontend_bundle") or {}
    for k in ("total_bytes", "js_bytes", "css_bytes"):
        _add(out, f"build.frontend.{k}", fe.get(k), "efficiency", "lower", "bytes")
    if fe.get("largest_chunk"):
        _add(out, "build.frontend.largest_chunk_bytes", fe["largest_chunk"]["bytes"], "efficiency", "lower", "bytes")
    for lock, n in (b.get("lock_packages") or {}).items():
        _add(out, f"build.lock.{lock}", n, "efficiency", "equal", "packages")
    return out


def family(metric_id: str) -> str:
    """`runtime.attendance.fps_p10` -> `runtime.*.fps_p10`: the metric without the config / suite / image it belongs to.
    A tolerance is set per family, so a new golden config is covered the day it is added."""
    parts = metric_id.split(".")
    head = parts[0]
    if head in ("runtime", "counters", "golden", "unit", "capacity") and len(parts) >= 3:
        return ".".join([head, "*"] + parts[2:])
    if head == "build" and len(parts) >= 3 and parts[1] in ("image", "lock"):
        return f"build.{parts[1]}.*"
    return metric_id
