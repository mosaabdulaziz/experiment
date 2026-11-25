"""Unit tests for perception/engine/telemetry.py (per-stage timing, benchmarking mode)
and for the [telemetry] log-line parser in benchmarks/stream_capacity.py."""

import importlib.util
import logging
import threading
import time
from pathlib import Path

import pytest

from perception.engine.telemetry import (
    configure, enabled, stage, timeit, register_gauge, flush, reset,
    gpu_enter, gpu_exit, fresh_or_zero,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_telemetry():
    reset()
    yield
    reset()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


@pytest.fixture
def capture():
    logger = logging.getLogger("iris.telemetry")
    h = _Capture()
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    yield h
    logger.removeHandler(h)


# ── disabled behavior ────────────────────────────────────────────────────────

def test_disabled_by_default_no_output(capture):
    assert enabled() is False
    with stage("probe"):
        time.sleep(0.005)
    flush(force=True)
    assert capture.records == []


def test_disabled_decorator_still_calls_through(capture):
    calls = []

    @timeit("deco")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(21) == 42
    assert calls == [21]
    flush(force=True)
    assert capture.records == []


# ── enabled behavior ─────────────────────────────────────────────────────────

def test_stage_accumulates_and_flushes(capture):
    configure(enabled=True, interval=60.0)  # interval long enough to never auto-flush
    with stage("probe"):
        time.sleep(0.01)
    flush(force=True)
    assert len(capture.records) == 1
    msg = capture.records[0]
    assert "[telemetry] probe n=1 total=" in msg
    assert "busy=" in msg and "avg=" in msg and "max=" in msg
    assert "thread=" in msg


def test_decorator_stage_reported(capture):
    configure(enabled=True, interval=60.0)

    @timeit("recognizer.embed")
    def embed():
        time.sleep(0.005)

    embed()
    embed()
    flush(force=True)
    msgs = [m for m in capture.records if "recognizer.embed" in m]
    assert len(msgs) == 1
    assert "n=2" in msgs[0]


def test_nested_stages_both_reported(capture):
    configure(enabled=True, interval=60.0)
    with stage("publish"):
        with stage("publish.draw"):
            time.sleep(0.005)
        time.sleep(0.005)
    flush(force=True)
    names = {m.split(" ")[1] for m in capture.records}
    assert {"publish", "publish.draw"} <= names


def test_multithread_stages_merged_with_thread_names(capture):
    configure(enabled=True, interval=60.0)

    def worker(name):
        with stage("publish"):
            time.sleep(0.01)

    ts = [threading.Thread(target=worker, args=(f"w{i}",), name=f"w{i}") for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    flush(force=True)
    msgs = [m for m in capture.records if m.startswith("[telemetry] publish ")]
    assert len(msgs) == 1
    assert "n=3" in msgs[0]  # three threads merged into one row
    # Thread names joined (order is set-iteration order, so compare as a set)
    names = msgs[0].split("thread=")[1].split(" ")[0].split(",")
    assert set(names) == {"w0", "w1", "w2"}


def test_busy_pct_tracks_wall_clock(capture):
    configure(enabled=True, interval=60.0)
    with stage("probe"):
        time.sleep(0.1)
    flush(force=True)
    msg = next(m for m in capture.records if m.startswith("[telemetry] probe "))
    # 0.1s of work over a ~0.1s window => busy near 100% (tolerate timer slop)
    busy = float(msg.split("busy=")[1].split("%")[0])
    assert busy > 50.0


def test_gauge_sampled_on_flush(capture):
    configure(enabled=True, interval=60.0)
    register_gauge("publisher_pending", lambda: 7)
    flush(force=True)
    assert "[telemetry] gauge publisher_pending=7" in capture.records


def test_auto_flush_with_short_interval(capture):
    configure(enabled=True, interval=0.05)
    for _ in range(20):
        with stage("probe"):
            time.sleep(0.01)
    # Short interval => at least one automatic flush happened.
    assert any("[telemetry] probe " in m for m in capture.records)


# ── GPU-element stages (pad-probe timing) ───────────────────────────────────

def test_gpu_stage_reported_with_thread_gpu(capture):
    configure(enabled=True, interval=60.0)
    gpu_enter("gpu.infer.pgie")
    time.sleep(0.01)
    gpu_exit("gpu.infer.pgie")
    flush(force=True)
    msgs = [m for m in capture.records if m.startswith("[telemetry] gpu.infer.pgie ")]
    assert len(msgs) == 1
    assert "n=1" in msgs[0]
    assert "thread=gpu" in msgs[0]
    assert "busy=" in msgs[0] and "avg=" in msgs[0] and "max=" in msgs[0]


def test_gpu_stage_avg_and_busy(capture):
    configure(enabled=True, interval=60.0)
    gpu_enter("gpu.jpeg")
    time.sleep(0.02)
    gpu_exit("gpu.jpeg")
    flush(force=True)
    msg = next(m for m in capture.records if m.startswith("[telemetry] gpu.jpeg "))
    assert float(msg.split("avg=")[1].split("ms")[0]) > 10.0
    assert float(msg.split("busy=")[1].split("%")[0]) > 50.0


def test_gpu_stage_interleaved_pairing_degrades_gracefully(capture):
    configure(enabled=True, interval=60.0)
    # enter/enter/exit/exit simulates two probes racing: the unmatched exit is
    # dropped, the surviving pair still times — never a crash, never a
    # double-count, never a negative.
    gpu_enter("gpu.infer.pgie")
    gpu_enter("gpu.infer.pgie")
    time.sleep(0.01)
    gpu_exit("gpu.infer.pgie")
    gpu_exit("gpu.infer.pgie")
    flush(force=True)
    msg = next(m for m in capture.records if m.startswith("[telemetry] gpu.infer.pgie "))
    assert "n=1" in msg
    total = float(msg.split("total=")[1].split("ms")[0])
    assert total >= 0.0


def test_gpu_stage_missing_exit_no_crash(capture):
    configure(enabled=True, interval=60.0)
    gpu_enter("gpu.decode.cam-1")  # never exited (probe dropped the frame)
    flush(force=True)
    assert capture.records == []


def test_gpu_stage_disabled_noop(capture):
    configure(enabled=False)
    gpu_enter("gpu.infer.pgie")
    gpu_exit("gpu.infer.pgie")
    flush(force=True)
    assert capture.records == []


def test_gpu_stages_cleared_by_reset(capture):
    configure(enabled=True, interval=60.0)
    gpu_enter("gpu.jpeg")
    gpu_exit("gpu.jpeg")
    reset()
    configure(enabled=True, interval=60.0)
    flush(force=True)
    assert capture.records == []


# ── harness parser ───────────────────────────────────────────────────────────

def _load_stream_capacity():
    spec = importlib.util.spec_from_file_location(
        "stream_capacity", ROOT / "benchmarks" / "stream_capacity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_telemetry_lines():
    sc = _load_stream_capacity()
    logs = (
        "[telemetry] probe n=1250 total=9123.4ms busy=24.1% avg=7.30ms "
        "max=48.2ms thread=streaming window=38.0s\n"
        "[telemetry] probe n=1180 total=8801.2ms busy=23.6% avg=7.46ms "
        "max=45.1ms thread=streaming window=37.2s\n"
        "[telemetry] publish n=310 total=1420.8ms busy=63.0% avg=4.58ms "
        "max=31.9ms thread=frame-publisher-cam-1 window=38.0s\n"
        "[telemetry] gauge publisher_pending=2\n"
        "[telemetry] gauge mux_queue_3=1\n"
        "[telemetry] gpu.infer.pgie n=100 total=8000.0ms busy=95.0% "
        "avg=80.00ms max=120.00ms thread=gpu window=38.0s\n"
        "[telemetry] gpu.decode.cam-1 n=500 total=1000.0ms busy=13.0% "
        "avg=2.00ms max=4.50ms thread=gpu window=38.0s\n"
        "[cam-1] 5.0 fps\n"
    )
    t = sc.parse_telemetry(logs)
    assert t is not None
    # hyphenated camera ids in gpu.decode.* stages must parse
    assert set(t["stages"]) == {"probe", "publish", "gpu.infer.pgie",
                                 "gpu.decode.cam-1"}
    g = t["stages"]["gpu.infer.pgie"]
    assert g["thread"] == "gpu"
    assert g["avg_ms"] == 80.0
    d = t["stages"]["gpu.decode.cam-1"]
    assert d["n"] == 500 and d["avg_ms"] == 2.0
    p = t["stages"]["probe"]
    assert p["n"] == 1250 + 1180
    assert p["busy_pct"]["median"] == 24.1
    assert p["busy_pct"]["n"] == 2
    assert p["calls_per_s"] is not None and p["calls_per_s"] > 0
    pub = t["stages"]["publish"]
    assert pub["thread"] == "frame-publisher-cam-1"
    assert pub["avg_ms"] > 0
    assert t["gauges"] == {"publisher_pending": 2, "mux_queue_3": 1}


def test_parse_telemetry_empty():
    sc = _load_stream_capacity()
    assert sc.parse_telemetry("[cam-1] 5.0 fps\n") is None


def test_a_rate_with_no_recent_sample_reads_zero():
    """2026-09-26: two barber cameras stopped delivering, and their fps gauges
    kept printing the last window's 30.6 for an hour -- the logs looked
    healthy. A rate whose last sample is older than max_age is 0."""
    assert fresh_or_zero(30.6, last_at=100.0, now=104.0, max_age=5.0) == 30.6
    assert fresh_or_zero(30.6, last_at=100.0, now=105.1, max_age=5.0) == 0.0


def test_a_rate_never_sampled_reads_zero():
    assert fresh_or_zero(12.0, last_at=None, now=1.0, max_age=5.0) == 0.0
