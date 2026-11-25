"""The footage's clock for a run on recordings (2026-10-18, user: a run playing 16 Oct 09:30-19:00 reports that date
and those times, not the device's). A site's footage_start is when its clips' first frame was filmed. The run's rows
are in device time from the lap's start -- the clips play together and clear_on_loop keeps one lap of rows -- so
footage time = device time + (footage_start - the lap's start). The lap is the one the consumer saw (its clip_run:
set at the run's start and at each loop), never one counted from the wall clock: a run that ended without looping
keeps its footage times after its end. Live cameras, or no footage_start: no footage clock (device time)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FootageClock:
    start: datetime          # the footage's first frame, footage time
    shift: timedelta         # footage - device, for this lap
    duration_s: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)

    def footage(self, device: datetime) -> datetime:
        return device + self.shift

    def device(self, footage: datetime) -> datetime:
        return footage - self.shift


def footage_clock(footage_start: datetime | None, run: dict | None) -> FootageClock | None:
    """`run`: the consumer's clip_run, {camera: [start_ts, clip_duration_s, lap]}. The clips are synced: the earliest
    lap start and the first clip's length place the footage."""
    if footage_start is None or not run:
        return None
    laps = [datetime.fromtimestamp(ts + lap * length) for ts, length, lap in run.values()]
    length = next(iter(run.values()))[1]
    return FootageClock(footage_start, footage_start - min(laps), float(length))
