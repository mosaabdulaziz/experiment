"""Pure, loop-only metrics for the iris pipeline. All mutations happen on the
asyncio event loop (analyzer, scheduler, stream_manager worker, sweep) — no
threads write here, so there is no locking. No I/O, no asyncio: unit-testable."""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class StreamRow:
    stream: str
    last_s: float | None
    avg_s: float | None
    cpm: int
    age_s: float | None


class _StreamMetrics:
    def __init__(self):
        self._durations: deque[float] = deque(maxlen=10)   # last 10 call durations
        self._call_ts: deque[float] = deque(maxlen=240)    # recent call timestamps
        self._last_event_t: float | None = None

    def record(self, duration_s: float, now: float) -> None:
        self._durations.append(float(duration_s))
        self._call_ts.append(float(now))

    def note_event(self, now: float) -> None:
        self._last_event_t = float(now)

    def row(self, stream: str, now: float) -> StreamRow:
        last = self._durations[-1] if self._durations else None
        avg = sum(self._durations) / len(self._durations) if self._durations else None
        cpm = sum(1 for t in self._call_ts if now - t <= 60.0)
        age = (now - self._last_event_t) if self._last_event_t is not None else None
        return StreamRow(stream, last, avg, cpm, age)


class MetricsRegistry:
    def __init__(self):
        self._streams: dict[str, _StreamMetrics] = {}

    def _get(self, stream: str) -> _StreamMetrics:
        m = self._streams.get(stream)
        if m is None:
            m = self._streams[stream] = _StreamMetrics()
        return m

    def record_analyze(self, stream: str, duration_s: float, now: float | None = None) -> None:
        self._get(stream).record(duration_s, time.monotonic() if now is None else now)

    def note_event(self, stream: str, now: float | None = None) -> None:
        self._get(stream).note_event(time.monotonic() if now is None else now)

    def rows(self, now: float) -> list[StreamRow]:
        return [self._streams[s].row(s, now) for s in sorted(self._streams)]


def _fmt_dur(v: float | None) -> str:
    return f"{v:.1f}s" if v is not None else "-"


def _fmt_age(v: float | None) -> str:
    return f"{int(v)}s ago" if v is not None else "-"


def _row(stream: str, q: str, last: str, avg: str, cpm: str, age: str) -> str:
    return f" {stream:<8}{q:<3}{last:<7}{avg:<7}{cpm:<9}{age}"


def render_table(rows, queue_depths, worker, sweeps, now_str) -> str:
    lines = [
        f"----- analysis status @{now_str} -----",
        _row("stream", "q", "last", "avg", "calls/m", "last-event"),
    ]
    for r in rows:
        lines.append(_row(
            r.stream, str(queue_depths.get(r.stream, 0)),
            _fmt_dur(r.last_s), _fmt_dur(r.avg_s), str(r.cpm), _fmt_age(r.age_s),
        ))
    worker_str = f"busy({worker})" if worker else "idle"
    lines.append(f" VLM worker: {worker_str}  chat-sweeps: {sweeps} active")
    return "\n".join(lines)
