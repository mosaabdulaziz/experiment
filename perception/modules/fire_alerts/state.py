"""Full-frame fire/smoke alert state: one sustained latch per fire class, per stream.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU, same convention
as overcrowding_dedup.py and engine_path.py.

Unlike every zone behaviour, this is deliberately NOT scoped to a drawn region. A zone
exists to answer "is this object inside the polygon", via an nvdsanalytics ROI and each
object's roiStatus. Fire has no such question -- any fire anywhere in frame matters --
so this path needs no ROI, no roiStatus, and no analytics_enabled.

The per-class split is not cosmetic: steam, vapour, exhaust and dust trigger SMOKE and
never FIRE, so the two classes need different sustain windows. Sharing one would force
the unambiguous signal (fire) to wait out the ambiguous one's debounce.
"""

import logging
from dataclasses import dataclass, field

from perception.lib.overcrowding_dedup import OvercrowdingLatch

logger = logging.getLogger("iris.fire_alerts")

_DEFAULT_CLASSES = ("fire", "smoke")
# Fire is visually unambiguous -- confirm fast. Smoke needs longer to separate from
# steam/vapour. Both are reasoned starting points, NOT measured; tune against real
# footage before trusting them.
_DEFAULT_SUSTAIN = {"fire": 3.0, "smoke": 8.0}
# Fallback for a configured class with no explicit sustain entry (see FireAlertState):
# the most conservative (longest) of the known defaults, i.e. smoke's -- never 0.0,
# which would fire on the very first detection and defeat this phase's entire purpose.
_MOST_CONSERVATIVE_DEFAULT_SUSTAIN = max(_DEFAULT_SUSTAIN.values())


@dataclass(frozen=True)
class FireIncidentEvent:
    """One incident-lifecycle transition for one fire/smoke class on one stream.

    "started" fires the instant a class's sustain window confirms (same moment
    the pre-incident-tracking code fired on) -- `onset_age` is how much earlier
    than `now` the condition ACTUALLY began (>= sustain_seconds; more, if dropout
    grace tolerated gaps), so a wall-clock-only consumer can back-date the true
    onset instead of using the confirming moment.

    "resolved" fires once, the instant the same class re-arms (confirmed absent
    for release_seconds) -- `duration` is the incident's true lifetime, onset to
    the moment it actually dropped out, NOT including the release_seconds debounce
    that follows it. Both timings come from stream pts (OvercrowdingLatch's own
    clock), never wall time, so they need no cross-container clock sync -- see
    this module's caller for how they're turned into wall-clock fields.
    """
    cls: str
    kind: str  # "started" | "resolved"
    count: int = 0             # qualifying detections on the frame that started this
    onset_age: "float | None" = None    # "started" only: now - true onset
    duration: "float | None" = None     # "resolved" only: true onset to true end


@dataclass(frozen=True)
class FireAlertSettings:
    """Read once at process start."""
    enabled: bool = False
    classes: tuple = _DEFAULT_CLASSES
    sustain_seconds: dict = field(default_factory=lambda: dict(_DEFAULT_SUSTAIN))
    release_seconds: float = 30.0
    min_confidence: float = 0.4
    dropout_grace_seconds: float = 1.0


class FireAlertState:
    """One instance per stream. Feed it every frame's detections; it returns
    incident lifecycle events -- "started" when a class's sustain window just
    confirmed, "resolved" when a previously-started class's release window just
    cleared it. Exactly one "resolved" follows each "started", in order, per
    class (the latch cannot re-fire until it re-arms) -- see FireIncidentEvent.

    Threshold is fixed at 1 -- one confident detection is the condition. "Two
    simultaneous fires" is not a meaningful alert rule, so this is not configurable.
    """

    def __init__(self, settings: FireAlertSettings):
        self.settings = settings
        self._latches = {}
        # Confirmed incidents' true onset, by class -- captured at "started" time
        # and consumed at "resolved" time. Cannot be re-read off the latch itself
        # at resolve time: dropout_grace_seconds independently clears the latch's
        # own run_started_at well before release_seconds re-arms it (grace is
        # meant to be short; release is the long debounce), so by the time a
        # "resolved" transition is even possible the latch has usually already
        # forgotten where the run began.
        self._incident_started_at = {}
        for name in settings.classes:
            if name in settings.sustain_seconds:
                sustain = float(settings.sustain_seconds[name])
            else:
                # A configured class with no explicit sustain entry is under-specified,
                # not incoherent -- fall back to the most conservative known default
                # rather than 0.0 (which would fire on the very first detection).
                sustain = _MOST_CONSERVATIVE_DEFAULT_SUSTAIN
                logger.warning(
                    "fire_alert class %r has no configured sustain_seconds entry; "
                    "falling back to the most conservative default (%.1fs). Set an "
                    "explicit sustain value for this class.", name, sustain)
            self._latches[name] = OvercrowdingLatch(
                release_seconds=settings.release_seconds,
                sustain_seconds=sustain,
                dropout_grace_seconds=settings.dropout_grace_seconds,
            )

    @property
    def has_open_incident(self) -> bool:
        """True while any class on this stream is between "started" and
        "resolved" -- i.e. there's an incident a caller might want to keep
        recording coverage for."""
        return bool(self._incident_started_at)

    def observe(self, now: float, detections) -> list:
        """`detections` is [(label, confidence), ...] for this frame. Returns
        FireIncidentEvent entries for classes that started or resolved on this
        frame -- usually empty, occasionally one entry, never more than one per
        class (a class cannot both start and resolve on the same frame)."""
        if not self.settings.enabled:
            return []
        events = []
        for name, latch in self._latches.items():
            count = sum(1 for label, conf in detections
                        if label == name and conf >= self.settings.min_confidence)
            was_armed = latch.armed
            fired = latch.observe(now, count, 1)
            if fired:
                onset = latch.run_started_at
                self._incident_started_at[name] = onset
                events.append(FireIncidentEvent(
                    cls=name, kind="started", count=count,
                    onset_age=(now - onset) if onset is not None else None))
            elif (not was_armed) and latch.armed:
                # Just re-armed: the incident that started earlier just resolved.
                # ended_at is this latch's own pts clock; started_at is whatever
                # this class's "started" event captured, however long ago.
                started_at = self._incident_started_at.pop(name, None)
                ended_at = latch.run_ended_at
                duration = (ended_at - started_at
                           if started_at is not None and ended_at is not None else None)
                events.append(FireIncidentEvent(cls=name, kind="resolved", duration=duration))
        return events
