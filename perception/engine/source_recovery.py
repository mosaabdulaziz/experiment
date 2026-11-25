"""Which bus messages mean a source is dead, and when to restart it.

Pure Python: importable off-GPU, unit-tested, no gi/pyds.
"""

import re
import threading
from typing import NamedTuple

_SOURCE_RE = re.compile(r"GstDsNvUriSrcBin:src_(\d+)")

# GStreamer appends "streaming stopped, reason <flow>" when an element pauses its
# task because a downstream push failed. Only these three are unrecoverable: the
# element will not resume on its own, so the source is finished until something
# restarts it. `eos` and `flushing` are ordinary shutdown and must not match.
_FATAL_FLOW_REASONS = ("not-negotiated", "error", "not-linked")


def dead_source_index(debug):
    """The nvurisrcbin index this bus message reports as fatally stopped, else None.

    Deliberately narrow. Transient RTSP read failures and nvurisrcbin's own
    "Trying reconnection" watchdog also name a source, and both recover without
    help -- acting on those would tear down a fleet that was healing itself.
    """
    if not debug:
        return None
    if not any(f"streaming stopped, reason {r}" in debug
               for r in _FATAL_FLOW_REASONS):
        return None
    found = _SOURCE_RE.search(debug)
    if found is None:
        return None
    return int(found.group(1))


class SourceRestartPolicy:
    """Rate-limits source restarts. One instance for the whole sidecar.

    A single failure is reported several times -- rtspsrc stops the instant the
    push fails, the record bin's queue reports the same thing again when the
    recording's duration expires, and nvurisrcbin's own reconnect keeps posting
    after that. Restarting on each would tear the source down repeatedly while it
    was still coming back up, so the first report wins and the rest are absorbed.

    The cooldown is per source: one camera failing must never mask another's.
    """

    def __init__(self, cooldown_seconds):
        self._cooldown = max(float(cooldown_seconds), 0.0)
        self._last_restart = {}

    def observe(self, now, index):
        """True if this source should be restarted now, False while cooling down."""
        previous = self._last_restart.get(index)
        if previous is not None and now - previous <= self._cooldown:
            return False
        self._last_restart[index] = now
        return True


class StallCheck(NamedTuple):
    """What one SourceStallWatch poll found."""
    restart: list        # source indices to restart, in index order
    downstream: bool     # every source went silent at once: not the sources' fault
    # Sources whose last max_failed_restarts restarts brought no frame back:
    # in-process recovery has failed, the caller should end the process.
    hopeless: tuple = ()
    # Every source silent for max_failed_restarts + 1 reports in a row.
    hopeless_downstream: bool = False


def escalation_due(strikes, now, max_failed, window_seconds):
    """True when max_failed (> 0) of these failure times fall within the window."""
    if max_failed <= 0:
        return False
    return sum(1 for t in strikes if now - t <= window_seconds) >= max_failed


class SourceStallWatch:
    """Finds sources that stopped delivering frames without saying so.

    A source can go silent with its RTSP session still up: rtspsrc keeps
    reading (so nvurisrcbin's "No data from source" watchdog, which looks
    upstream of the decoder, never fires) while none of its frames come out
    of the mux -- either the decoder is starved of surfaces, or the mux takes
    its buffers and never batches them. Nothing is posted on the bus, so
    dead_source_index never sees it -- observed 2026-09-24 on all three cameras of one deployment
    cameras in turn, each lost for the life of the process. The only evidence
    is the missing frames, so the caller notes every frame the batch probe
    sees (after the mux, which catches both kinds) and polls stalled().

    * A source that has never delivered is not stalled -- it is still
      connecting, which nvurisrcbin's own reconnect handles.
    * A reported source waits twice as long before its next report (capped at
      max_backoff_seconds), so a camera that is really gone is not restarted
      every period forever; its next frame resets the wait.
    * When every tracked source (two or more) is silent at once, the stall is
      downstream of them (inference, probe): nothing is restarted, the watch
      re-arms and says so.
    * reset() re-arms everything -- for a pipeline that was paused on purpose.
    * Escalation (max_failed_restarts > 0; 0, the default, is off): a source
      restarted that many times within failed_restart_window_seconds with no
      frame since is not restarted again on its next report but returned in
      StallCheck.hopeless -- 2026-09-26, after the test streams restarted, two
      cameras stayed dead through every restart until the whole sidecar
      restarted. The same count applies to consecutive downstream reports
      (hopeless_downstream); any frame from any source clears that one. A
      source that never delivers after the process restarts is never tracked,
      so a camera that is really gone does not cause a restart loop.

    stall_seconds <= 0 disables. note_buffer runs on the streaming thread and
    stalled() on the main loop, hence the lock.
    """

    def __init__(self, stall_seconds, max_backoff_seconds=300.0,
                 max_failed_restarts=0, failed_restart_window_seconds=600.0):
        self._stall = float(stall_seconds)
        self._max_wait = max(float(max_backoff_seconds), self._stall)
        self._max_failed = int(max_failed_restarts)
        self._window = float(failed_restart_window_seconds)
        self._last = {}
        self._wait = {}
        self._strikes = {}       # index -> restart times since its last frame
        self._down_strikes = []  # downstream report times since any frame
        self._lock = threading.Lock()

    @property
    def stall_seconds(self):
        return self._stall

    @property
    def max_failed_restarts(self):
        return self._max_failed

    def note_buffer(self, index, now):
        """Called from the streaming thread for every frame: keep it trivial."""
        with self._lock:
            self._last[index] = now
            if index in self._wait:
                del self._wait[index]
                self._strikes.pop(index, None)
            if self._down_strikes:
                self._down_strikes = []

    def reset(self, now):
        """Treat every tracked source as having just delivered."""
        with self._lock:
            for index in self._last:
                self._last[index] = now
            self._wait.clear()
            self._strikes.clear()
            self._down_strikes = []

    def stalled(self, now):
        """Poll: which sources to restart now (see the class docstring)."""
        if self._stall <= 0:
            return StallCheck([], False)
        with self._lock:
            if len(self._last) >= 2 and all(now - last > self._stall
                                            for last in self._last.values()):
                for index in self._last:
                    self._last[index] = now
                if escalation_due(self._down_strikes, now, self._max_failed,
                                  self._window):
                    return StallCheck([], False, (), True)
                self._down_strikes.append(now)
                return StallCheck([], True)
            out = []
            hopeless = []
            for index, last in sorted(self._last.items()):
                wait = self._wait.get(index, self._stall)
                if now - last > wait:
                    self._last[index] = now
                    self._wait[index] = min(wait * 2, self._max_wait)
                    strikes = self._strikes.setdefault(index, [])
                    if escalation_due(strikes, now, self._max_failed, self._window):
                        hopeless.append(index)
                    else:
                        out.append(index)
                        strikes.append(now)
            return StallCheck(out, False, tuple(hopeless))


class MainLoopWatchdog:
    """Notices a wedged GLib main loop, from outside it.

    nvurisrcbin's own reconnect runs on the main loop, and on 2026-09-24 it
    deadlocked there: resetting a source whose decoder was starved, rtspsrc's
    pad deactivation waited forever for a streaming thread stuck behind that
    decoder. The stall watch, the bus handler and the ctrl poll are all
    main-loop callbacks, so they all stopped with it and the camera was lost
    silently. The loop beats once a second; a separate thread polls wedged()
    and, since nothing in-process can break that deadlock, ends the process
    for the container's restart policy. timeout_seconds <= 0 disables.
    """

    def __init__(self, timeout_seconds, now):
        self._timeout = float(timeout_seconds)
        self._last = now

    @property
    def timeout_seconds(self):
        return self._timeout

    @property
    def poll_seconds(self):
        """How often the watching thread checks: a tenth of the timeout,
        1 to 5 s, so a short timeout isn't stretched by a slow poll."""
        return min(5.0, max(1.0, self._timeout / 10.0))

    def beat(self, now):
        self._last = now

    def silent_for(self, now):
        return now - self._last

    def wedged(self, now):
        return self._timeout > 0 and now - self._last > self._timeout
