"""Hardware GPU counters, sampled once a second while a replay runs (stdlib only).

Why this exists: nvidia-smi's `utilization.gpu` is the share of time in which AT LEAST ONE kernel was running. A GPU can
read 90% while most of its SMs sit idle (one small kernel at a time), so it says nothing about how well the hardware is
used. The sidecar's own stage timers (`gpu.infer.pgie busy=...`) are time based in the same way. Three layers say more:

  NVML     any NVIDIA GPU (read through nvidia-smi): GPU and memory-controller time, and the hardware engines separately:
           NVDEC (decode), NVENC, NVJPG (JPEG), OFA; clocks against their maximum, power, temperature, the clock throttle
           reasons, PCIe throughput.
  DCGM     the profiling counters, which look inside the GPU: SM_ACTIVE (how many SMs had a warp resident), SM_OCCUPANCY,
           PIPE_TENSOR_ACTIVE / PIPE_FP16_ACTIVE / PIPE_FP32_ACTIVE (which execution pipes ran), DRAM_ACTIVE (memory
           bandwidth), GR_ENGINE_ACTIVE. Read from the repo's own pinned dcgm-exporter image. Where the GPU or driver does
           not support them the result says "unsupported" and why, never a made-up number.
  Jetson   tegrastats: GR3D load, EMC (memory) load, NVDEC / NVENC / NVJPG / VIC / OFA, power rails. DCGM does not run there.

`hints()` turns the numbers into "look here first" statements. They are heuristics with documented thresholds, to point a
profiler (Nsight Systems or Compute, installed with CUDA) at the right place, not verdicts.
"""
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request

from . import machine

NVML_FIELDS = ["utilization.gpu", "utilization.memory", "utilization.encoder", "utilization.decoder", "utilization.jpeg",
               "utilization.ofa", "clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw", "temperature.gpu",
               "clocks_event_reasons.active", "memory.used"]
NVML_NAMES = ["gpu_util", "mem_ctrl_util", "nvenc_util", "nvdec_util", "nvjpg_util", "ofa_util", "sm_clock_mhz",
              "sm_clock_max_mhz", "mem_clock_mhz", "power_w", "temp_c", "throttle_mask", "mem_used_mib"]
# bits of the clock event reasons that mean the GPU was held back (idle, application clocks and display bits do not)
THROTTLE_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80      # SW power cap, HW slowdown, SW thermal, HW thermal, power brake
DCGM_PORT = int(os.environ.get("REGRESS_DCGM_PORT", "19400"))
# the profiling counters and the engine utilisations dcgm-exporter is asked for (a ratio 0..1 unless noted)
DCGM_COUNTERS = [
    ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", "gr_engine_active"), ("DCGM_FI_PROF_SM_ACTIVE", "sm_active"),
    ("DCGM_FI_PROF_SM_OCCUPANCY", "sm_occupancy"), ("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "tensor_active"),
    ("DCGM_FI_PROF_PIPE_FP16_ACTIVE", "fp16_active"), ("DCGM_FI_PROF_PIPE_FP32_ACTIVE", "fp32_active"),
    ("DCGM_FI_PROF_DRAM_ACTIVE", "dram_active"), ("DCGM_FI_PROF_PCIE_TX_BYTES", "pcie_tx_bytes_s"),
    ("DCGM_FI_PROF_PCIE_RX_BYTES", "pcie_rx_bytes_s"),
]
SUMMARY_KEYS = ("median", "p95", "max", "mean")


# ── parsers (pure) ───────────────────────────────────────────────────────────────────────────────────────────────────

def _num(text: str):
    text = text.strip()
    if text.lower().startswith("0x"):
        try:
            return int(text, 16)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None                       # "[N/A]", "[Not Supported]"


def parse_nvml_row(line: str) -> dict:
    """One `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` line -> {name: number or None}."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(NVML_NAMES):
        return {}
    return {name: _num(v) for name, v in zip(NVML_NAMES, parts)}


def parse_dmon_pcie(line: str) -> dict:
    """`nvidia-smi dmon -s t` data line `  0   12    3` -> {pcie_rx_mb_s, pcie_tx_mb_s} (headers start with #)."""
    if line.lstrip().startswith("#"):
        return {}
    parts = line.split()
    if len(parts) >= 3 and parts[0].isdigit():
        return {"pcie_rx_mb_s": _num(parts[1]), "pcie_tx_mb_s": _num(parts[2])}
    return {}


def parse_dcgm_metrics(text: str, gpu: str = "0") -> dict:
    """The exporter's /metrics text -> {our name: value} for one GPU, only the counters that were returned."""
    wanted = {dcgm: ours for dcgm, ours in DCGM_COUNTERS}
    out = {}
    for line in text.splitlines():
        m = re.match(r'^(DCGM_FI_\w+)\{([^}]*)\}\s+(\S+)', line)
        if not m or f'gpu="{gpu}"' not in m.group(2):
            continue
        if m.group(1) in wanted:
            v = _num(m.group(3))
            if v is not None:
                out[wanted[m.group(1)]] = v
    return out


def parse_tegrastats_line(line: str) -> dict:
    """A Jetson tegrastats line -> the numbers that matter: GR3D %, memory, EMC %, the media engines' clocks (0 = off),
    power rails (mW), temperatures."""
    out = machine.parse_tegrastats(line)
    out["gr3d_pct"] = out.pop("gpu_util_pct", None)
    m = re.search(r"EMC_FREQ (\d+)%", line)
    out["emc_pct"] = int(m.group(1)) if m else None
    for eng in ("NVDEC", "NVENC", "NVJPG", "VIC", "OFA"):
        m = re.search(rf"{eng} (off|\d+)", line)
        out[eng.lower() + "_mhz"] = (0 if m.group(1) == "off" else int(m.group(1))) if m else None
    out["power_mw"] = {k: int(v) for k, v in re.findall(r"(\w+) (\d+)mW/\d+mW", line)}
    for rail, mw in out["power_mw"].items():                     # numbers, so they are summarised like every other metric
        out[f"power_{rail.lower()}_mw"] = mw
    return out


# ── summaries (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────

def percentile(xs: list, q: float):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize_series(values: list):
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return {"median": round(statistics.median(xs), 3), "p95": round(percentile(xs, 0.95), 3), "max": round(max(xs), 3),
            "mean": round(statistics.fmean(xs), 3), "n": len(xs)}


def summarize(samples: list) -> dict:
    """{metric: {median, p95, max, mean, n}} over a list of sample dicts, plus `throttled_fraction` for the NVML rows."""
    keys = sorted({k for s in samples for k, v in s.items() if k not in ("t", "power_mw") and not isinstance(v, dict)})
    out = {}
    for k in keys:
        s = summarize_series([x.get(k) for x in samples])
        if s:
            out[k] = s
    masks = [int(x["throttle_mask"]) for x in samples if x.get("throttle_mask") is not None]
    if masks:
        out["throttled_fraction"] = round(sum(1 for m in masks if m & THROTTLE_BITS) / len(masks), 3)
    return out


SM_GROUP = ("sm_active", "sm_occupancy", "tensor_active", "fp16_active", "fp32_active")


def validate_dcgm(dcgm: dict):
    """(ok, reason). The SM counters cannot read exactly zero for a whole window while the graphics engine was busy: a stalled
    profiling session does that (seen once on the RTX 5050: SM active 0.0 in all 73 samples with the engine 68% active, and
    61% when the same replay ran again). Such numbers are dropped, never reported as 'the SMs were idle'."""
    gr, sm = (dcgm.get("gr_engine_active") or {}), (dcgm.get("sm_active") or {})
    if gr and sm and gr["median"] >= 0.4 and sm["max"] < 0.01:
        return False, ("the SM counters read 0 in every sample while the graphics engine was %.0f%% active: the DCGM profiling "
                       "session had stalled, so SM active, occupancy and the pipe counters are not reported" % (gr["median"] * 100))
    # A partly stalled session (seen on 2026-11-11: fire's SM median 0 with the engine 95% active, yet some samples up to 0.6):
    # SM active cannot be far under graphics-engine active, because an SM works only while the engine does.
    if gr and sm and gr["median"] >= 0.5 and sm["median"] < 0.5 * gr["median"]:
        return False, ("the SM active median (%.2f) is under half of the graphics-engine median (%.2f), which cannot be: the DCGM "
                       "profiling session had stalled for part of the window, so the SM counters are not reported"
                       % (sm["median"], gr["median"]))
    return True, ""


def hints(nvml: dict, dcgm: dict, jetson: dict, probe: dict = None) -> list:
    """Where to look first, from the counters. Heuristics (thresholds in the text), for pointing a profiler, not verdicts.
    `nvml`, `dcgm`, `jetson` are summarize() outputs (or empty); `probe` is the sidecar's time-based total busy %."""
    out = []
    g = lambda d, k, f="median": (d.get(k) or {}).get(f)      # noqa: E731
    if g(nvml, "nvdec_util", "p95") is not None and g(nvml, "nvdec_util", "p95") >= 80:
        out.append("decode: NVDEC reached %.0f%% (p95): the hardware decoder is close to its limit" % g(nvml, "nvdec_util", "p95"))
    if g(nvml, "nvjpg_util", "p95") is not None and g(nvml, "nvjpg_util", "p95") >= 80:
        out.append("JPEG: NVJPG reached %.0f%% (p95): the JPEG engine is close to its limit" % g(nvml, "nvjpg_util", "p95"))
    if g(nvml, "nvenc_util", "p95") is not None and g(nvml, "nvenc_util", "p95") >= 80:
        out.append("encode: NVENC reached %.0f%% (p95)" % g(nvml, "nvenc_util", "p95"))
    util, sm = g(nvml, "gpu_util"), g(dcgm, "sm_active")
    if util is not None and sm is not None and util >= 50 and sm <= 0.30:
        out.append("utilisation overstates the load: the GPU was 'busy' %.0f%% of the time but only %.0f%% of the SMs were "
                   "active (small batches, or kernels launched one after another)" % (util, sm * 100))
    dram, tensor = g(dcgm, "dram_active"), g(dcgm, "tensor_active")
    if dram is not None and dram >= 0.60:
        out.append("memory-bound: DRAM was active %.0f%% of the cycles" % (dram * 100))
    if sm is not None and sm >= 0.50 and tensor is not None and tensor < 0.05:
        out.append("the tensor cores are idle (%.1f%% active) while the SMs are busy: the models run on the FP32 path" % (tensor * 100))
    if (nvml.get("throttled_fraction") or 0) >= 0.05:
        out.append("throttled: the GPU was held back (power cap or thermal) in %.0f%% of the samples" % (nvml["throttled_fraction"] * 100))
    clk, cmax = g(nvml, "sm_clock_mhz"), g(nvml, "sm_clock_max_mhz")
    if clk and cmax and util and util >= 50 and clk < 0.7 * cmax:
        out.append("clocks stayed low: SM %.0f MHz of %.0f MHz while busy" % (clk, cmax))
    if g(jetson, "gr3d_pct", "p95") is not None and g(jetson, "gr3d_pct", "p95") >= 90:
        out.append("GPU (GR3D) reached %.0f%% (p95) on the Jetson" % g(jetson, "gr3d_pct", "p95"))
    if g(jetson, "emc_pct", "p95") is not None and g(jetson, "emc_pct", "p95") >= 80:
        out.append("memory-bound: the Jetson's memory controller (EMC) reached %.0f%% (p95)" % g(jetson, "emc_pct", "p95"))
    return out


def stage_hints(dcgm: dict, stages: dict) -> list:
    """The sidecar's stage timers say how long a stage was EXECUTING; the SM counters say how much of the GPU that used.
    A stage that is busy most of the time while the SMs mostly idle is launching many small kernels or waiting on the CPU
    or on copies: that is where batching or pre-processing is worth a look."""
    out = []
    sm = (dcgm.get("sm_active") or {})
    if not sm or not stages:
        return out
    for name, st in sorted(stages.items(), key=lambda kv: -kv[1]["busy"]):
        if st["busy"] >= 50 and sm["p95"] <= 0.5 and not name.startswith("gpu.decode"):
            out.append("%s is busy %.0f%% of the time by its timer, but the SMs were active at most %.0f%% (p95, median %.0f%%): "
                       "many small launches, or waiting for the CPU or a copy; look at its batching and pre-processing"
                       % (name, st["busy"], sm["p95"] * 100, sm["median"] * 100))
    return out


# ── the samplers (side effects) ──────────────────────────────────────────────────────────────────────────────────────

class _Stream(threading.Thread):
    """Reads lines from a command that prints until killed (nvidia-smi -l 1, dmon, tegrastats) into samples."""

    def __init__(self, cmd: list, parse, extra: dict = None) -> None:
        super().__init__(daemon=True)
        self.cmd, self.parse, self.samples, self.error = cmd, parse, [], None
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as exc:
            self.proc, self.error = None, str(exc)

    def run(self) -> None:
        if self.proc is None:
            return
        for line in self.proc.stdout:
            row = self.parse(line.strip())
            if row:
                row["t"] = time.time()
                self.samples.append(row)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()


class DcgmSampler(threading.Thread):
    """dcgm-exporter (the pinned image of docker-compose.yml) with the profiling counters, polled once a second."""

    def __init__(self, repo: str) -> None:
        super().__init__(daemon=True)
        self.samples, self.status = [], {"supported": False, "reason": "not started"}
        self._halt = threading.Event()
        self.name_ = f"regress-dcgm-{os.getpid()}"
        self.image = self._image(repo)
        self.dir = tempfile.mkdtemp(prefix="regress-dcgm-")
        with open(os.path.join(self.dir, "counters.csv"), "w") as f:
            for dcgm, _ours in DCGM_COUNTERS:
                f.write(f"{dcgm}, gauge, {dcgm}\n")

    @staticmethod
    def _image(repo: str) -> str:
        try:
            text = open(os.path.join(repo, "docker-compose.yml")).read()
            m = re.search(r"image:\s*(nvidia/dcgm-exporter:\S+)", text)
            return m.group(1) if m else "nvidia/dcgm-exporter:latest"
        except OSError:
            return "nvidia/dcgm-exporter:latest"

    def start_container(self, wait_s: float = 45.0) -> dict:
        """Start the exporter and wait until a profiling counter shows up. The first scrape comes ~30 s after the start."""
        r = subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name_, "--gpus", "all", "--cap-add", "SYS_ADMIN",
                            "-p", f"127.0.0.1:{DCGM_PORT}:9400", "-v", f"{self.dir}/counters.csv:/etc/dcgm-exporter/c.csv:ro",
                            self.image, "-f", "/etc/dcgm-exporter/c.csv", "--collect-interval", "1000"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.status = {"supported": False, "reason": (r.stderr or "docker run failed").strip()[-160:]}
            return self.status
        deadline = time.time() + wait_s
        while time.time() < deadline:
            got = self._scrape()
            if got:
                self.status = {"supported": True, "reason": "", "image": self.image, "counters": sorted(got)}
                self.start()
                return self.status
            time.sleep(2)
        logs = subprocess.run(["docker", "logs", "--tail", "6", self.name_], capture_output=True, text=True)
        self.status = {"supported": False, "image": self.image,
                       "reason": "no profiling counter was returned in %.0f s: the GPU or the driver does not expose "
                                 "them to DCGM (or profiling is restricted to admin users). %s" % (wait_s, (logs.stdout + logs.stderr)[-140:].strip())}
        self.close()
        return self.status

    def _scrape(self) -> dict:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DCGM_PORT}/metrics", timeout=3) as resp:
                return parse_dcgm_metrics(resp.read().decode())
        except (OSError, ValueError):
            return {}

    def run(self) -> None:
        while not self._halt.wait(1.0):
            row = self._scrape()
            if row:
                row["t"] = time.time()
                self.samples.append(row)

    def close(self) -> None:
        self._halt.set()
        subprocess.run(["docker", "rm", "-f", self.name_], capture_output=True)
        try:
            for f in os.listdir(self.dir):
                os.unlink(os.path.join(self.dir, f))
            os.rmdir(self.dir)
        except OSError:
            pass


def nvml_query_fields() -> list:
    """NVML_FIELDS, with the older name of the throttle-reasons field on drivers that do not know the new one."""
    ok = machine.run("nvidia-smi", "--query-gpu=clocks_event_reasons.active", "--format=csv,noheader")
    return NVML_FIELDS if ok.strip() else [f.replace("clocks_event_reasons", "clocks_throttle_reasons") for f in NVML_FIELDS]


class GpuCounters:
    """Start once per run; `window(t0, t1)` summarises the samples taken while one replay ran."""

    def __init__(self, repo: str, use_dcgm: bool = True) -> None:
        self.jetson = machine.is_jetson()
        self._repo = repo
        self.streams, self.dcgm = {}, None
        if self.jetson:
            self.streams["jetson"] = _Stream(["tegrastats", "--interval", "1000"], parse_tegrastats_line)
        else:
            self.streams["nvml"] = _Stream(["nvidia-smi", f"--query-gpu={','.join(nvml_query_fields())}",
                                            "--format=csv,noheader,nounits", "-l", "1"], parse_nvml_row)
            self.streams["pcie"] = _Stream(["nvidia-smi", "dmon", "-s", "t", "-d", "1"], parse_dmon_pcie)
        for s in self.streams.values():
            if s.proc is not None:
                s.start()
        self.dcgm_status = {"supported": False, "reason": "not requested (--no-dcgm)" if not use_dcgm else "Jetson: DCGM does not run there"}
        if use_dcgm and not self.jetson:
            self.dcgm = DcgmSampler(repo)
            self.dcgm_status = self.dcgm.start_container()

    def window(self, t0: float, t1: float) -> dict:
        def cut(samples):
            return [s for s in samples if t0 <= s["t"] <= t1]
        nvml = {}
        if "nvml" in self.streams:
            rows = cut(self.streams["nvml"].samples)
            pcie = cut(self.streams["pcie"].samples)
            for r in rows:                              # attach the nearest PCIe reading
                near = min(pcie, key=lambda p: abs(p["t"] - r["t"]), default=None)
                if near:
                    r.update({k: v for k, v in near.items() if k != "t"})
            nvml = summarize(rows)
        dcgm = summarize(cut(self.dcgm.samples)) if self.dcgm and self.dcgm_status.get("supported") else {}
        ok, why = validate_dcgm(dcgm)
        if not ok:
            dcgm = {k: v for k, v in dcgm.items() if k not in SM_GROUP}
        jet = summarize(cut(self.streams["jetson"].samples)) if "jetson" in self.streams else {}
        return {"nvml": nvml, "dcgm": dcgm, "jetson": jet, "dcgm_status": self.dcgm_status,
                "dcgm_valid": ok, "dcgm_invalid_reason": why,
                "samples": {"nvml": len(cut(self.streams["nvml"].samples)) if "nvml" in self.streams else 0,
                            "dcgm": len(cut(self.dcgm.samples)) if self.dcgm else 0,
                            "jetson": len(cut(self.streams["jetson"].samples)) if "jetson" in self.streams else 0},
                "hints": hints(nvml, dcgm, jet)}

    def restart_dcgm(self) -> dict:
        """A fresh exporter, hence a fresh DCGM profiling session (after validate_dcgm said the last one stalled)."""
        if self.dcgm is not None:
            self.dcgm.close()
        self.dcgm = DcgmSampler(self._repo)
        self.dcgm_status = self.dcgm.start_container()
        return self.dcgm_status

    def background(self, seconds: float = 5.0) -> dict:
        """How busy the GPU is before anything runs: the graphics-engine activity DCGM measures when it is available, else
        nvidia-smi's utilisation. The two differ a lot: a laptop that draws its display reads 40% on nvidia-smi and 3.6% on
        the engine counter, and only the second one says something else could contend with a benchmark."""
        t0 = time.time()
        time.sleep(seconds)
        w = self.window(t0, time.time())
        if w["dcgm"].get("gr_engine_active"):
            return {"pct": round(w["dcgm"]["gr_engine_active"]["median"] * 100, 1), "source": "DCGM graphics-engine activity",
                    "nvidia_smi_util_pct": (w["nvml"].get("gpu_util") or {}).get("median")}
        util = (w["nvml"].get("gpu_util") or w["jetson"].get("gr3d_pct") or {}).get("median")
        return {"pct": util, "source": "GPU utilisation (nvidia-smi / tegrastats)", "nvidia_smi_util_pct": util}

    def stop(self) -> None:
        for s in self.streams.values():
            s.stop()
        if self.dcgm:
            self.dcgm.close()
