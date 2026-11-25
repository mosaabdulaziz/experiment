"""Flight recorder for the sidecar: evidence for "the cameras went silent and nobody knows why".

Two losses of every camera happened after hours of running (one source after another stopped
being read, the stall watchdog's own GLib timer stopped ticking, the main thread sat in a futex).
This module watches the process from OUTSIDE the GLib main loop, in a plain thread, so it keeps
working when the loop is stuck:

  * main-loop lag: the loop calls ``beat()`` every second; if the beat stops, the Python stack of
    every thread is written to ``<dir>/stall-<time>.txt`` (and the main thread's to the log), so
    the exact call the loop is blocked in is on record;
  * scheduling delay: how late this thread wakes from a 2 s sleep (CPU / GIL starvation shows here);
  * CPU by thread group every 30 s ("frame-publisher-*" 780 %, "gpu" 120 %, ...), logged as
    ``[flight] cpu`` -- who is eating the machine when sources start starving;
  * bus events counted per camera and kind (no data from source, could not write to resource,
    errors, watchdog restarts), exposed as gauges next to the other sidecar metrics.

Everything is read-only and cheap; it can be left on.
"""
from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

_TICK = 2.0


def classify_bus(text: str, is_error: bool = False) -> str:
    """Bucket a bus message by what it says."""
    t = (text or "").lower()
    if "no data from source" in t:
        return "nodata"
    if "could not write to resource" in t:
        return "writefail"
    if is_error:
        return "error"
    return "warn"


def cam_of(src_name: str, index_to_id: dict) -> str:
    """'src_4' -> the camera id of source 4; anything else -> 'other'."""
    m = re.fullmatch(r"src_(\d+)", src_name or "")
    if m:
        return str(index_to_id.get(int(m.group(1)), f"src_{m.group(1)}"))
    return "other"


def census_from(tasks: list[tuple[str, str, str]], prefix: str = "dec_que") -> dict:
    """tasks: (thread name, wchan, state) of every native thread. Counts the decoder threads
    (``dec_que:src``) by what they are doing: a healthy idle one waits on a futex / condition; one
    stuck in nvv4l2decoder's polling loop sleeps in nanosleep (the signature of the earlier hang)."""
    dec = [(w, st) for name, w, st in tasks if name.startswith(prefix)]
    sleeping = [w for w, _st in dec if "nanosleep" in w or "hrtimer" in w]
    running = [1 for _w, st in dec if st == "R"]
    return {"decoders": len(dec), "sleeping": len(sleeping), "running": len(running),
            "text": f"{len(dec)} decoder threads: {len(dec) - len(sleeping)} waiting normally, "
                    f"{len(sleeping)} in a nanosleep loop, {len(running)} running"}


def read_tasks(root: str = "/proc/self/task") -> list[tuple[str, str, str]]:
    out = []
    try:
        tids = os.listdir(root)
    except OSError:
        return out
    for tid in tids:
        try:
            comm = open(f"{root}/{tid}/comm").read().strip()
            wchan = open(f"{root}/{tid}/wchan").read().strip()
            raw = open(f"{root}/{tid}/stat").read()
            out.append((comm, wchan, raw[raw.rindex(")") + 2:].split()[0]))
        except (OSError, IndexError, ValueError):
            continue
    return out


def _call_with_timeout(fn, secs: float = 1.0):
    """Run fn() in a helper thread and give up after ``secs``: the recorder must never block on a
    pipeline object that is itself stuck. Returns (ok, value)."""
    box: list = []
    t = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    t.start()
    t.join(secs)
    return (True, box[0]) if box else (False, None)


def _event_line(component: str, text: str) -> str:
    """Same one-line format as the exporter's events log: '2026-10-04T08:01:03Z [event] component: text'."""
    return f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [event] {component}: {text}"


class FlightRecorder:
    def __init__(self, out_dir: str, logger, stall_s: float = 10.0, describe=None, native_dump: bool = True):
        self.out_dir = out_dir
        self.log = logger
        self.stall_s = float(stall_s)
        self.native_dump = native_dump    # gdb on a stall; off where gdb takes minutes and freezes every stream
        self.describe = describe          # () -> list[str], one line per source
        self._beat = time.monotonic()
        self.main_lag_s = 0.0
        self.sched_delay_ms = 0.0
        self._sched_max = 0.0
        self.threads = 0
        self.process_cpu_pct = 0.0
        self._events: dict[tuple[str, str], int] = collections.defaultdict(int)
        self._lock = threading.Lock()
        self._last_dump = 0.0
        self._cpu_prev: dict[int, int] = {}
        self._cpu_prev_t = time.monotonic()
        self.stalls = 0
        self._last_native = 0.0
        # components that must keep up with the others; see watch_drops / watch_depth
        self._drop_watch: list[dict] = []
        self._depth_watch: list[dict] = []
        self.pressure_events = 0
        self._qos: dict[str, int] = {}
        # per-camera episodes of "no frame for a while" (see watch_sources)
        self._ages = None
        self._evidence = None
        self.stall_after_s = 4.0
        self._episodes: dict[str, float] = {}       # cam -> monotonic start of the open episode
        self._episode_count: dict[str, int] = collections.defaultdict(int)
        self.decoders_sleeping = 0
        os.makedirs(out_dir, exist_ok=True)

    def event(self, component: str, text: str) -> None:
        """Append one line to <dir>/events.log (shared with the exporter's alert and GPU events)."""
        try:
            with open(os.path.join(self.out_dir, "events.log"), "a") as f:
                f.write(_event_line(component, text) + "\n")
        except OSError:
            pass

    # -- fed by the sidecar ---------------------------------------------------
    def beat(self) -> bool:
        """Called from a 1 s GLib timer: proves the main loop is turning."""
        self._beat = time.monotonic()
        return True

    def note(self, cam: str, kind: str) -> None:
        with self._lock:
            self._events[(cam, kind)] += 1

    def events(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._events)

    def count(self, cam: str, kind: str) -> int:
        with self._lock:
            return self._events.get((cam, kind), 0)

    def watch_drops(self, name: str, total, per_s: float, hint: str) -> None:
        """Warn (``[backpressure]`` in the log) when the cumulative counter ``total()`` rises faster
        than ``per_s`` per second for ~10 s: something downstream is discarding work."""
        self._drop_watch.append({"name": name, "fn": total, "limit": per_s, "hint": hint,
                                 "prev": None, "hot": 0, "last": 0.0})

    def watch_depth(self, name: str, depth, cap: float, hint: str, frac: float = 0.9, secs: float = 30.0) -> None:
        """Warn when ``depth()`` stays at or above ``frac`` of ``cap`` for ``secs`` seconds."""
        self._depth_watch.append({"name": name, "fn": depth, "cap": cap, "frac": frac, "secs": secs,
                                  "hint": hint, "since": None, "last": 0.0})

    def watch_sources(self, ages, evidence=None, after_s: float = 4.0) -> None:
        """``ages()`` -> {camera: seconds since its last frame, or None before the first one}. A camera
        silent for ``after_s`` seconds opens an episode (logged with ``evidence(cam)``, a string of
        what its pipeline looks like right then); frames returning closes it. This is earlier than
        nvurisrcbin's own 10 s reset, so the evidence is taken while the source is still stuck."""
        self._ages, self._evidence, self.stall_after_s = ages, evidence, after_s

    def episodes(self, cam: str) -> int:
        return self._episode_count.get(cam, 0)

    def open_episodes(self) -> int:
        return len(self._episodes)

    def _check_sources(self, now: float) -> None:
        if self._ages is None:
            return
        ok, ages = _call_with_timeout(self._ages)
        if not ok or not ages:
            return
        census = census_from(read_tasks())
        self.decoders_sleeping = census["sleeping"]
        for cam, age in ages.items():
            if age is None:
                continue
            if cam not in self._episodes and age >= self.stall_after_s:
                self._episodes[cam] = now - age
                self._episode_count[cam] += 1
                ev = "n/a"
                if self._evidence is not None:
                    got, text = _call_with_timeout(lambda c=cam: self._evidence(c), 1.0)
                    ev = text if got else "evidence unavailable: reading the pipeline blocked for 1 s"
                msg = f"{cam}: no frame for {age:.0f} s -- {ev}; {census['text']}"
                self.log.warning("[source-stall] START %s", msg)
                self.event("source", "STALL START " + msg)
            elif cam in self._episodes and age < 1.5:
                dur = now - self._episodes.pop(cam)
                self.log.warning("[source-stall] END %s: frames are back after %.0f s", cam, dur)
                self.event("source", f"STALL END {cam}: frames are back after {dur:.0f} s")

    def note_qos(self, element: str, dropped: int) -> None:
        """A GStreamer QoS message: cumulative frames this element has dropped as too late."""
        if dropped > self._qos.get(element, 0) and (element in self._qos or len(self._qos) < 60):
            self._qos[element] = dropped

    def qos_dropped(self) -> int:
        return sum(self._qos.values())

    def _check_pressure(self, now: float, dt: float) -> None:
        for w in self._drop_watch:
            try:
                total = float(w["fn"]())
            except Exception:
                continue
            if w["prev"] is not None and dt > 0:
                rate = (total - w["prev"]) / dt
                w["hot"] = w["hot"] + 1 if rate > w["limit"] else 0
                if w["hot"] >= 5 and now - w["last"] > 60.0:
                    w["last"] = now
                    self.pressure_events += 1
                    self.log.warning("[backpressure] %s: %.1f dropped/s (%d so far) -- %s",
                                     w["name"], rate, int(total), w["hint"])
                    self.event("backpressure", f"{w['name']}: {rate:.1f} dropped/s ({int(total)} so far) -- {w['hint']}")
            w["prev"] = total
        for w in self._depth_watch:
            try:
                d = float(w["fn"]())
            except Exception:
                continue
            if d >= w["cap"] * w["frac"]:
                if w["since"] is None:
                    w["since"] = now
                if now - w["since"] >= w["secs"] and now - w["last"] > 120.0:
                    w["last"] = now
                    self.pressure_events += 1
                    self.log.warning("[backpressure] %s: %.0f of %.0f for %.0fs -- %s",
                                     w["name"], d, w["cap"], now - w["since"], w["hint"])
                    self.event("backpressure", f"{w['name']}: {d:.0f} of {w['cap']:.0f} for {now - w['since']:.0f}s -- {w['hint']}")
            else:
                w["since"] = None

    # -- evidence ----------------------------------------------------------------
    def dump_stacks(self, reason: str) -> str:
        """Write every Python thread's stack to a file; return its path."""
        names = {t.ident: t.name for t in threading.enumerate()}
        main_id = threading.main_thread().ident
        lines = [f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {reason}", f"# main-loop lag {self.main_lag_s:.1f}s, "
                 f"sched delay {self.sched_delay_ms:.0f}ms, threads {self.threads}", ""]
        if self.describe:
            try:
                lines += ["## sources"] + list(self.describe()) + [""]
            except Exception as exc:   # never let the description hide the stacks
                lines.append(f"## sources: unavailable ({exc})")
        frames = sys._current_frames()
        for ident in sorted(frames, key=lambda i: (i != main_id, names.get(i, ""))):
            lines.append(f"## thread {names.get(ident, '?')} ({ident}){' MAIN' if ident == main_id else ''}")
            lines += [l.rstrip() for l in traceback.format_stack(frames[ident])]
            lines.append("")
        stem = os.path.join(self.out_dir, f"stall-{time.strftime('%Y%m%d-%H%M%S')}")
        path = stem + ".txt"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        if self.native_dump and (self.stalls <= 3 or time.monotonic() - self._last_native > 900):   # gdb pauses the process: not every minute
            self._last_native = time.monotonic()
            self.dump_native(stem)
        main = frames.get(main_id)
        if main is not None:
            tail = [l.strip().replace("\n", " | ") for l in traceback.format_stack(main)[-4:]]
            self.log.error("[flight] %s -- stacks in %s; main thread blocked in: %s", reason, path, " <- ".join(reversed(tail)))
        self.event("sidecar", f"STALL {reason} -- stacks in {path}")
        return path

    def dump_native(self, stem: str) -> None:
        """Native (C) backtraces of every thread, with gdb, in the background: the Python stacks show
        only that the main thread is inside g_main_loop_run, never which C call it is blocked in.
        Needs the container to have SYS_PTRACE (docker-compose.scale.yml); skipped quietly if not.
        The process is paused while gdb reads the threads (some seconds), which is fine: it is stuck."""
        if shutil.which("gdb") is None:
            return

        def work():
            path = f"{stem}.native.txt"
            cmd = ["gdb", "-p", str(os.getpid()), "-batch", "-ex", "set pagination off",
                   "-ex", "thread 1", "-ex", "frame 0", "-ex", "info registers rdi",
                   "-ex", "x/6dw $rdi",    # a futex wait: [lock, count, OWNER tid, ...] of the mutex it waits on
                   "-ex", "thread apply all bt 25"]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                with open(path, "w") as f:
                    f.write("# main thread: frame 0 + the mutex it waits on (3rd word = owner thread id), then all threads\n")
                    f.write(out.stdout + out.stderr)
                owner = re.search(r"^0x[0-9a-f]+:\s+-?\d+\s+-?\d+\s+(\d+)", out.stdout, re.M)
                self.log.error("[flight] native backtraces in %s (main thread waits on a lock owned by thread %s)",
                               path, owner.group(1) if owner else "?")
            except Exception as exc:
                self.log.warning("[flight] native dump failed: %s", exc)

        threading.Thread(target=work, name="flight-native", daemon=True).start()

    def _cpu_by_group(self) -> tuple[str, float]:
        """CPU % per thread group since the last call (100 = one core), and the process total."""
        names = {t.native_id: t.name for t in threading.enumerate() if getattr(t, "native_id", None)}
        now = time.monotonic()
        dt = max(now - self._cpu_prev_t, 1e-3)
        hz = os.sysconf("SC_CLK_TCK")
        cur: dict[int, int] = {}
        groups: dict[str, float] = collections.defaultdict(float)
        try:
            for tid in os.listdir("/proc/self/task"):
                try:
                    with open(f"/proc/self/task/{tid}/stat") as f:
                        raw = f.read()
                except OSError:
                    continue
                comm = raw[raw.index("(") + 1:raw.rindex(")")]
                fields = raw[raw.rindex(")") + 2:].split()
                ticks = int(fields[11]) + int(fields[12])    # utime + stime
                tid_i = int(tid)
                cur[tid_i] = ticks
                d = ticks - self._cpu_prev.get(tid_i, 0 if self._cpu_prev else ticks)   # a thread born since the last look counts from 0
                if d <= 0:
                    continue
                name = names.get(tid_i) or comm
                groups[re.sub(r"[-_]?\d+$", "", re.sub(r"cam-\d+", "*", name)) or name] += d / hz / dt * 100.0
        except OSError:
            pass
        self._cpu_prev, self._cpu_prev_t = cur, now
        total = sum(groups.values())
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:6]
        return ", ".join(f"{n} {v:.0f}%" for n, v in top if v >= 5), total

    # -- the watcher thread ----------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, name="flight-recorder", daemon=True).start()

    def _run(self) -> None:
        n = 0
        while True:
            t0 = time.monotonic()
            time.sleep(_TICK)
            now = time.monotonic()
            delay = max(0.0, now - t0 - _TICK) * 1000.0
            self._sched_max = max(self._sched_max * 0.9, delay)
            self.sched_delay_ms = self._sched_max
            self.main_lag_s = max(0.0, now - self._beat - 1.0)
            n += 1
            try:
                self._check_pressure(now, _TICK + delay / 1000.0)
                self._check_sources(now)
                if self.main_lag_s > self.stall_s and now - self._last_dump > 60.0:
                    self._last_dump = now
                    self.stalls += 1
                    self.dump_stacks(f"main loop silent for {self.main_lag_s:.0f}s")
                if n % 15 == 0:      # every 30 s
                    top, total = self._cpu_by_group()
                    self.process_cpu_pct = total
                    self.threads = len(os.listdir("/proc/self/task"))
                    self.log.info("[flight] cpu %.0f%% in %d threads: %s | main-loop lag %.1fs, sched delay %.0fms",
                                  total, self.threads, top or "-", self.main_lag_s, self.sched_delay_ms)
            except Exception:
                self.log.exception("[flight] recorder step failed")
