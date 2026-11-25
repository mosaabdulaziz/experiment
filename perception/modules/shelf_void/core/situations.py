# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/situations.py. Copied unchanged below this header: fix it there first, then copy it again.
"""When to send event.raised / event.cleared for one (pipeline, camera).

IRIS keeps one open alert per (pipelineId, cameraId, correlationKey) (guide §7.1). We use one
correlationKey per camera, so one Situation per camera:

  raised    confirmed spots go 0 -> >=1                  (new alert, people notified)
  raised    count rises above the last reported count    (same alert updated, no re-notify)
  raised    every `reraise_seconds` while open           (fresh eventId, current confidence)
  cleared   zero confirmed spots for `clear_seconds`, counted only while the zone is person-free
  cleared   close(): camera removed / pipeline paused while open
  nothing   count drops but stays >= 1; anything while closed with zero spots

`now` is a monotonic clock in seconds (stream time) used for all timers; `wall` is the aware
UTC datetime of the same sample, used only for occurredAt.
"""
from dataclasses import dataclass, field
from datetime import datetime

from .tracker import EPS


@dataclass(frozen=True)
class Spot:
    bbox: list            # [x1, y1, x2, y2] normalised 0-1
    detector_score: float  # score of the spot's latest detection (tracker overwrites it on each hit)
    stability: float      # hit ratio since first seen, 0-1


@dataclass(frozen=True)
class Intent:
    """An event to send. messages.build_event turns it into the IRIS body."""
    state: str            # "raised" | "cleared"
    reason: str           # new | more | periodic | gone | paused | camera_removed | removed
    occurred_at: datetime
    spots: tuple = field(default_factory=tuple)
    clear_seconds: float = 0.0


class Situation:
    def __init__(self, clear_seconds, reraise_seconds):
        self.clear_seconds, self.reraise_seconds = clear_seconds, reraise_seconds
        self.open = False
        self.reported_count = 0
        self.last_raise = None
        self.clear_acc = 0.0           # person-free seconds with zero spots, while open
        self._prev = None              # (now, counted) of the previous sample

    def _raise(self, reason, spots, now, wall):
        self.open, self.reported_count, self.last_raise = True, len(spots), now
        self.clear_acc = 0.0
        return [Intent("raised", reason, wall, tuple(spots))]

    def update(self, spots, active, now, wall):
        """One sample: `spots` = confirmed spots in the zone, `active` = zone is person-free
        (always True when pauseOnPerson is off). Returns a list of Intents (usually empty)."""
        if wall.tzinfo is None:
            raise ValueError("wall must be timezone-aware (UTC)")
        prev, self._prev = self._prev, None
        n = len(spots)
        if n:
            self._prev = (now, False)
            self.clear_acc = 0.0  # a spot is back: the clear hold starts again
            if not self.open:
                return self._raise("new", spots, now, wall)
            if n > self.reported_count:
                return self._raise("more", spots, now, wall)
            if now - self.last_raise >= self.reraise_seconds - EPS:
                return self._raise("periodic", spots, now, wall)
            return []
        if not self.open:
            return []
        # Open, zero spots: only person-free time between two zero samples counts towards clearing.
        counted = bool(active)
        if counted and prev is not None and prev[1]:
            self.clear_acc += now - prev[0]
        self._prev = (now, counted)
        if self.clear_acc >= self.clear_seconds - EPS:
            return self.close(wall, "gone")
        return []

    def close(self, wall, reason):
        """Clear an open situation now (reason: gone / paused / camera_removed / removed)."""
        if not self.open:
            return []
        self.open, self.reported_count, self.last_raise, self.clear_acc = False, 0, None, 0.0
        self._prev = None
        return [Intent("cleared", reason, wall, (), self.clear_seconds)]
