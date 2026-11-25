"""Per-stage timing telemetry for the IRIS sidecar pipeline (benchmarking mode).

When enabled (sidecar.yaml ``telemetry_enabled: true`` or env ``IRIS_TELEMETRY=1``),
the hot-path pipeline stages accumulate wall-clock time in per-thread counters
and periodically log ``[telemetry]`` lines:

  [telemetry] probe n=1250 total=9123.4ms busy=24.1% avg=7.30ms max=48.2ms thread=streaming window=38.0s
  [telemetry] publish n=310 total=1420.8ms busy=63.0% avg=4.58ms max=31.9ms thread=frame-publisher-cam-1 window=38.0s
  [telemetry] gauge publisher_pending=2

Semantics:

  - ``n``     calls in the window
  - ``total`` accumulated wall-clock time of those calls (ms)
  - ``busy``  ``total`` as a share of the thread's wall-clock window — the
              pipeline-thread utilization number to compare across stages
  - ``avg``/``max`` per-call cost

Threads own their accumulators (thread-local; no lock on the hot path). The
registry only takes a lock when a thread first touches a stage and at flush
time (once per interval). ``busy`` is per-thread, so stages running on
different threads are compared by busy% (each is that thread's utilization);
the absolute ``total`` ms lets you eyeball cross-thread cost.

Stages wired in sidecar.py / face_recognizer.py:

  probe          streaming thread, per batch  (sub: probe.surface_copy,
                 probe.frame_objects, probe.dets)
  publish        frame-publisher worker, per job (sub: publish.draw, publish.redis)
  recognizer     recognition thread, per frame (sub: recognizer.embed,
                 recognizer.stabilize, recognizer.publish)
  upload         segment uploader thread, per fragment
  rotation       recording streaming thread, per splitmuxsink rotation

GPU-element stages (``gpu.*``) are timed with GStreamer pad probes attached
by the sidecar around the element's sink+src pads; the probe callbacks just
call :func:`gpu_enter` / :func:`gpu_exit`, so this module stays pure stdlib
and off-GPU importable. busy% for a ``gpu.*`` stage is the element's
processing time as a share of the wall-clock window (its utilization):

  gpu.infer.pgie     nvinfer inference, per batch (batches in -> one out)
  gpu.infer.sgie     secondary face nvinfer, per batch
  gpu.decode.<cam>   the decoder inside each source bin (nvv4l2decoder /
                     avdec_h264/265), per frame
  gpu.jpeg           nvjpegenc in the publisher's mini-pipeline, per encode

Gauges are backpressure signals for the Python-side queues. Aggregate GPU
cost across all elements still comes from host-side ``nvidia-smi`` sampling.

Prometheus export
-----------------
:func:`start_prometheus` serves the same numbers over HTTP at
``/metrics`` (text format) for a Prometheus scrape: every flushed stage row
becomes ``iris_stage_busy_percent`` / ``iris_stage_avg_ms`` /
``iris_stage_total_ms`` / ``iris_stage_samples_total`` labelled by stage, and
every sampled gauge becomes ``iris_gauge{name=...}`` (backpressure queues,
per-stream ``fps_<id>``, ``sgie_mode``). The exporter runs in a daemon thread
started by the sidecar when telemetry is enabled; the busy% numbers are
identical to the ``[telemetry]`` log lines, so a dashboard matches the
benchmark harness exactly. The import of ``prometheus_client`` is deferred to
:func:`start_prometheus`, so the module stays importable off-GPU without the
dependency installed (the exporter just logs a warning and stays off).

Pure stdlib, no gi — importable off-GPU (unit-tested on the host).
"""

import functools
import logging
import os
import threading
import time

_logger = logging.getLogger("iris.telemetry")
_active = False
_interval = 5.0
_t0 = time.monotonic()
_next_flush = _t0 + _interval
_lock = threading.Lock()
_threads: set = set()          # Thread objects that touched a stage
_gauges: dict = {}             # name -> zero-arg callable returning a number
_gpu_entered: dict = {}        # gpu stage -> last entry time (monotonic s)
_gpu_accs: dict = {}           # gpu stage -> _Accum (pad-probe timing)
# Optional Prometheus export (set by start_prometheus; None = exporter off).
_prom = None                   # (registry, busy, avg, total, count, gauge)


class _Accum:
    __slots__ = ("n", "ms", "max")
    def __init__(self):
        self.n = 0
        self.ms = 0.0
        self.max = 0.0


def _accs() -> dict:
    """This thread's {stage: _Accum}, creating + registering it on first use.

    The dict lives on the Thread object so any thread (including the one doing
    the flush) can read it back under the registry lock.
    """
    t = threading.current_thread()
    d = getattr(t, "_iris_tlm", None)
    if d is None:
        d = {}
        try:
            t._iris_tlm = d
        except (AttributeError, TypeError):
            pass
        with _lock:
            _threads.add(t)
    return d


def configure(enabled: bool | None = None, interval: float | None = None,
              logger=None) -> bool:
    """(Re)configure telemetry. Returns whether it ended up active.

    ``enabled`` defaults to env ``IRIS_TELEMETRY`` (any value except empty /
    "0" / "false"). Must be called before stages run; the sidecar calls it
    from SidecarConfig with the yaml value.
    """
    global _active, _interval, _t0, _next_flush, _logger
    if enabled is None:
        env = os.environ.get("IRIS_TELEMETRY", "")
        enabled = env not in ("", "0", "false", "False")
    _active = bool(enabled)
    if interval is not None:
        _interval = float(interval)
    if logger is not None:
        _logger = logger
    _t0 = time.monotonic()
    _next_flush = _t0 + _interval
    return _active


def enabled() -> bool:
    return _active


class _NullStage:
    __slots__ = ()
    def __enter__(self):
        return self
    def __exit__(self, *_exc):
        return False


_NULL_STAGE = _NullStage()


class _TimingStage:
    __slots__ = ("name", "t0")
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.monotonic()
    def __enter__(self):
        return self
    def __exit__(self, *_exc):
        add(self.name, time.monotonic() - self.t0)
        return False


def stage(name: str):
    """Context manager timing ``name``; returns a no-op when disabled."""
    if not _active:
        return _NULL_STAGE
    return _TimingStage(name)


def timeit(name: str):
    """Decorator timing one function/method call as stage ``name``.

    When disabled the wrapper still delegates to the wrapped function (the
    decision is made per call, not at decoration time), so the decorators can
    be applied unconditionally with zero meaningful overhead when off.
    """
    def _deco(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            if not _active:
                return fn(*args, **kwargs)
            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                add(name, time.monotonic() - t0)
        return _wrapper
    return _deco


def add(name: str, seconds: float) -> None:
    """Accumulate one timed call. Hot path: no lock, one float compare."""
    if not _active:
        return
    acc = _accs()
    a = acc.get(name)
    if a is None:
        a = acc[name] = _Accum()
    a.n += 1
    a.ms += seconds * 1000.0
    if seconds * 1000.0 > a.max:
        a.max = seconds * 1000.0
    if time.monotonic() >= _next_flush:
        flush()


def interval() -> float:
    """Seconds between flushes (and so between gauge samples)."""
    return _interval


def fresh_or_zero(value, last_at, now, max_age):
    """A rate gauge's value, or 0.0 when nothing fed it for max_age seconds.

    A per-window rate is only recomputed when a new sample arrives, so a
    source that stops dead would keep reporting its last window forever
    (2026-09-26: two cameras of one deployment read 30.6 fps for an hour while dead).
    """
    if last_at is None or now - last_at > max_age:
        return 0.0
    return value


def register_gauge(name: str, fn) -> None:
    """Register a zero-arg callable sampled on every flush (queue depths, ...)."""
    with _lock:
        _gauges[name] = fn


def gpu_enter(stage_name: str) -> None:
    """Record the entry time of one buffer into a GPU element (pad probe).

    Called from a sink-pad BUFFER probe attached by the sidecar. Matched by
    :func:`gpu_exit` from the src-pad probe; the delta is the element's
    processing latency for that buffer (or batch, for nvinfer). Cheap dict
    write on the hot path; no-op when disabled.
    """
    if _active:
        _gpu_entered[stage_name] = time.monotonic()


def gpu_exit(stage_name: str) -> None:
    """Accumulate one GPU-element stage sample (src-pad probe callback).

    delta = exit - last entry for this stage. Serialized elements (a decoder
    or nvinfer instance handles one buffer at a time) make this exact; if two
    probes interleave the pairing degrades gracefully — one sample ~0, the
    next ~2x — bounded telemetry error, never a crash.
    """
    if not _active:
        return
    t0 = _gpu_entered.pop(stage_name, None)
    if t0 is None:
        return
    ms = (time.monotonic() - t0) * 1000.0
    with _lock:
        a = _gpu_accs.get(stage_name)
        if a is None:
            a = _gpu_accs[stage_name] = _Accum()
        a.n += 1
        a.ms += ms
        if ms > a.max:
            a.max = ms


def start_prometheus(port: int = 9100) -> bool:
    """Serve the telemetry as Prometheus metrics at http://<host>:<port>/metrics.

    Idempotent: a second call keeps the first server and returns True. The
    ``prometheus-client`` import is deferred to this call so the module loads
    (and its unit tests run) on hosts without the package; when it is missing
    this logs a warning and returns False — the ``[telemetry]`` log lines keep
    working either way. Call once after :func:`configure`, before stages run.
    """
    global _prom
    if _prom is not None:
        return True
    try:
        from prometheus_client import (CollectorRegistry, Counter, Gauge,
                                       start_http_server)
    except Exception:
        _logger.warning("prometheus-client not installed — telemetry /metrics "
                        "exporter disabled (install prometheus-client)")
        return False
    reg = CollectorRegistry()
    busy = Gauge("iris_stage_busy_percent",
                 "Stage wall-clock busy percent (share of the telemetry window)",
                 ["stage"], registry=reg)
    avg = Gauge("iris_stage_avg_ms",
                "Stage average wall-clock per call (ms)", ["stage"], registry=reg)
    total = Gauge("iris_stage_total_ms",
                  "Stage accumulated wall-clock in the last window (ms)",
                  ["stage"], registry=reg)
    count = Counter("iris_stage_samples_total",
                    "Total stage call count", ["stage"], registry=reg)
    gauge = Gauge("iris_gauge",
                  "Sampled telemetry gauges (backpressure queues, per-stream fps, mode flags)",
                  ["name"], registry=reg)
    _prom = (reg, busy, avg, total, count, gauge)
    try:
        start_http_server(port, registry=reg)
    except Exception as exc:
        _logger.warning("could not start Prometheus /metrics server on :%d — %s",
                        port, exc)
        _prom = None
        return False
    _logger.info("telemetry Prometheus exporter on :%d/metrics (stage rows + "
                 "gauges pushed every flush)", port)
    return True


def _export_prometheus(merged, gpu_stages, gauge_values, window):
    """Push one flush window's stage rows + gauges into the Prometheus registry.

    Called from :func:`flush` under the registry lock; no-op until
    :func:`start_prometheus` has run. busy% uses the same formula as the
    ``[telemetry]`` log line (share of the window), so dashboard numbers match
    the benchmark harness exactly. Gauge values are sampled once here and
    shared with the log output.
    """
    p = _prom
    if p is None:
        return
    _reg, busy, avg, total, count, gauge = p
    for name, (n, ms, _mx, _tn) in merged.items():
        if n == 0:
            continue
        busy.labels(stage=name).set(
            (100.0 * ms / (window * 1000.0)) if window > 0 else 0.0)
        avg.labels(stage=name).set(ms / n)
        total.labels(stage=name).set(ms)
        count.labels(stage=name).inc(n)
    for name, a in gpu_stages.items():
        if a.n == 0:
            continue
        busy.labels(stage=name).set(
            (100.0 * a.ms / (window * 1000.0)) if window > 0 else 0.0)
        avg.labels(stage=name).set(a.ms / a.n)
        total.labels(stage=name).set(a.ms)
        count.labels(stage=name).inc(a.n)
    for gname, val in gauge_values:
        try:
            gauge.labels(name=gname).set(float(val))
        except (TypeError, ValueError):
            continue


def flush(now: float | None = None, force: bool = False) -> None:
    """Merge every thread's accumulators, log them, reset, sample gauges.

    Runs under the registry lock (once per interval across all threads).
    """
    global _next_flush, _t0, _gpu_accs
    if not _active:
        return
    if now is None:
        now = time.monotonic()
    with _lock:
        if not force and now < _next_flush:
            return
        _next_flush = now + _interval
        window = now - _t0
        _t0 = now
        merged: dict = {}
        for t in list(_threads):
            # Swap the thread's dict out (under the lock) rather than clearing
            # it in place: the owning thread may be mid-add() on the same dict,
            # and iterating a dict another thread is writing can raise
            # RuntimeError. A call racing the swap is simply counted next
            # window — bounded measurement loss, never a crash.
            old = getattr(t, "_iris_tlm", None)
            if not old:
                continue
            t._iris_tlm = {}
            for name, a in list(old.items()):
                m = merged.get(name)
                if m is None:
                    merged[name] = [a.n, a.ms, a.max, t.name]
                else:
                    m[0] += a.n
                    m[1] += a.ms
                    m[2] = max(m[2], a.max)
                    m[3] += f",{t.name}"
        for name, (n, ms, mx, tn) in sorted(merged.items()):
            if n == 0:
                continue
            busy = (100.0 * ms / (window * 1000.0)) if window > 0 else 0.0
            _logger.info(
                "[telemetry] %s n=%d total=%.1fms busy=%.1f%% avg=%.2fms "
                "max=%.2fms thread=%s window=%.1fs",
                name, n, ms, busy, ms / n, mx, tn, window)
        # GPU-element stages (pad-probe timing; not tied to one thread).
        # Swap the dict out under the lock (gpu_exit mutates it under the
        # same lock) and emit one row per element with thread=gpu.
        old_gpu = _gpu_accs
        _gpu_accs = {}
        for name, a in sorted(old_gpu.items()):
            if a.n == 0:
                continue
            busy = (100.0 * a.ms / (window * 1000.0)) if window > 0 else 0.0
            _logger.info(
                "[telemetry] %s n=%d total=%.1fms busy=%.1f%% avg=%.2fms "
                "max=%.2fms thread=gpu window=%.1fs",
                name, a.n, a.ms, busy, a.ms / a.n, a.max, window)
        gauge_values = []
        for gname, fn in sorted(_gauges.items()):
            try:
                val = fn()
            except Exception:
                continue
            if val is None:
                continue
            _logger.info("[telemetry] gauge %s=%s", gname, val)
            gauge_values.append((gname, val))
        _export_prometheus(merged, old_gpu, gauge_values, window)


def reset() -> None:
    """Test helper: disable + drop all accumulators/gauges/thread registries."""
    global _active, _t0, _next_flush
    with _lock:
        _active = False
        _gauges.clear()
        _threads.clear()
        _gpu_entered.clear()
        _gpu_accs.clear()
        _t0 = time.monotonic()
        _next_flush = _t0 + _interval
        for t in list(threading.enumerate()):
            # hasattr, NOT truthiness: after a flush the dict is empty ({}) but
            # still present, and _accs() only re-registers the thread when the
            # attribute is absent — an empty dict left behind would silently
            # drop every later stage (see tests/test_telemetry.py).
            if hasattr(t, "_iris_tlm"):
                try:
                    delattr(t, "_iris_tlm")
                except (AttributeError, TypeError):
                    pass
