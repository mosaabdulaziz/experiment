"""Zone-intrusion alert dedup that survives tracker re-IDs.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU.
"""

import math


class ZoneReentryGrace:
    """Per-stream "one alert per person" bookkeeping for zone intrusions.

    The naive rule -- alert once per tracker object_id -- re-fires when the
    detector briefly drops someone (posture change, partial occlusion) and the
    tracker re-badges them with a new id on re-detection. This keeps the
    once-per-id rule but adds a spatial grace: a NEW id appearing in-zone close
    to where an already-alerted id vanished less than ``grace_seconds`` ago
    inherits that id's alerted status instead of firing again. A genuinely new
    person entering elsewhere in the zone still alerts immediately.
    """

    def __init__(self, grace_seconds: float = 5.0, match_radius: float = 1.5):
        self.grace_seconds = float(grace_seconds)
        self.match_radius = float(match_radius)
        self._alerted: set = set()   # ids that ever fired; never pruned (same id never re-fires)
        self._last_seen: dict = {}   # alerted id -> (t, cx, cy, diag); pruned after the grace window

    def observe(self, now: float, in_zone: list) -> list:
        """Process one frame's zone-intruding tracked objects.

        ``in_zone`` is this frame's ``[(object_id, (left, top, width, height)), ...]``
        (bboxes in a consistent coordinate space, e.g. mux resolution). Returns
        the object_ids an alert should fire for, in input order.
        """
        present = {}
        for oid, (left, top, w, h) in in_zone:
            present[oid] = (left + w / 2.0, top + h / 2.0, math.hypot(w, h))

        # Refresh last-seen for already-alerted ids still in the zone, so their
        # grace window is measured from when they actually vanished.
        for oid, (cx, cy, diag) in present.items():
            if oid in self._alerted:
                self._last_seen[oid] = (now, cx, cy, diag)

        fire = []
        for oid, (cx, cy, diag) in present.items():
            if oid in self._alerted:
                continue
            donor = self._match_vanished(now, cx, cy, diag, present)
            self._alerted.add(oid)
            if donor is not None:
                # Same person re-badged by the tracker: transfer, don't re-alert.
                del self._last_seen[donor]
            else:
                fire.append(oid)
            self._last_seen[oid] = (now, cx, cy, diag)

        # Entries unseen for longer than the grace window can no longer absolve
        # a new id; the ids themselves stay in _alerted forever.
        stale = [oid for oid, entry in self._last_seen.items()
                 if now - entry[0] > self.grace_seconds]
        for oid in stale:
            del self._last_seen[oid]

        return fire

    def _match_vanished(self, now: float, cx: float, cy: float, diag: float,
                        present: dict):
        """Nearest recently-vanished alerted id within the match radius, or None."""
        best = None
        best_dist = None
        for oid, (t, ox, oy, odiag) in self._last_seen.items():
            if oid in present or now - t > self.grace_seconds:
                continue
            dist = math.hypot(cx - ox, cy - oy)
            if dist > self.match_radius * max(diag, odiag):
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = oid, dist
        return best
