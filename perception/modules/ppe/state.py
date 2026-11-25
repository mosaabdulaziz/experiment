"""One stream's PPE rule state: the evaluation-rate gate, the temporal filter and the duplicate-alert hint. Pure Python,
no engine types.

Each evaluated frame: the frame's objects -> adapter -> the vendored per-frame rules (evaluate_frame) -> track ids
back -> the vendored TemporalFilter (window per track and item, episodes, cooldown, forgetting). Frames arriving
faster than eval_fps are skipped, so the window counts the same number of evaluated frames per second on every
camera. Time is the engine's monotonic `now`, not pts (which restarts when a file loops).

Smoothing is keyed by the tracker's id, or, with the optional re-ID grace (reid.py), by a canonical id that a new
tracker id can inherit from a recently lost one. Either way the results handed back carry the tracker's CURRENT id:
alerts and the overlay name the box actually seen; the canonical id never leaves the filter.

Duplicate hint (always on): when the tracker loses a worker and re-finds them under a new id, the same violation
alerts again. An alert that may repeat an earlier one names it (`possible_duplicate_of`) instead of being dropped:
grouping, not suppression, as ppe-compliance does (its issue #15). The rule (IRIS; it replaces
ppe-compliance's loose `find_possible_duplicate`, which named the wrong worker in 8 of 14 hints on test2.mp4): the most
recent earlier alert, for a shared item, of a different track in the same chain of track links (reid.py TrackLinks).

Alert hold (`hold_alert` / `due_alerts`): a new violation is held for `alert_hold_s` before it is raised, so an item
that starts a moment later on the same track joins it: one alert "without helmet and vest", not two.
"""
from .adapter import frame_input
from .compliance.compliance import evaluate_frame
from .compliance.config import ComplianceConfig, TemporalConfig
from .compliance.temporal import TemporalFilter
from .compliance.types import ItemStatus
from .reid import LinkConfig, ReidConfig, ReidGrace, TrackLinks

_EPS = 1e-6                              # s; frame times that are "on" a due time but off by float rounding


class StreamRules:
    def __init__(self, compliance: ComplianceConfig, temporal: TemporalConfig, eval_fps: float,
                 reid: ReidConfig = None, links: LinkConfig = None) -> None:
        reid = reid or ReidConfig()
        self.compliance = compliance
        self.temporal = temporal
        self.period = 1.0 / eval_fps
        self.filter = TemporalFilter(temporal, compliance.required_items)
        self.reid = ReidGrace(reid, temporal.forget_after_s)
        self.links = TrackLinks(links or LinkConfig(), temporal.forget_after_s)  # for the duplicate hint
        self.alerts: list = []           # (now, pts, tracker id, chain, items) of this stream's recent alerts
        self.held: dict = {}             # tracker id -> a new alert held for alert_hold_s (hold_alert)
        self._due = None                 # monotonic time of the next frame to evaluate
        self.evaluated = 0
        # {tracker id: [missing items, in required_items order]} as of the last evaluated frame: who is in violation
        # now (smoothed ABSENT), kept between evaluated frames for drawing.
        self.violators: dict = {}

    def _take(self, now: float) -> bool:
        """True when this frame is due. Due times advance by one period, so a 30 fps stream is evaluated at 20 fps
        on average; after a gap longer than a period the schedule restarts from this frame."""
        if self._due is not None and now < self._due - _EPS:
            return False
        if self._due is None or now - self._due >= self.period:
            self._due = now + self.period
        else:
            self._due += self.period
        return True

    def evaluate(self, objects, mux_size, now: float):
        """The smoothed persons of this frame (list of SmoothedPerson, each result carrying the tracker's current
        id), or None when the frame is skipped. An evaluated frame with no person still runs the filter, so tracks
        that left are forgotten on time."""
        if not self._take(now):
            return None
        inp = frame_input(objects, mux_size)
        results, _unassigned = evaluate_frame(inp.detections, self.compliance, inp.image_size)
        inp.attach_track_ids(results)
        tracker_ids = [r.track_id for r in results]
        self.links.observe(now, [(tid, r.person.bbox) for tid, r in zip(tracker_ids, results)])
        canonical = self.reid.observe(now, list(zip(tracker_ids, (r.person.bbox for r in results))),
                                      inp.image_size[1])
        for r in results:
            r.track_id = canonical[r.track_id]
        self.evaluated += 1
        smoothed = self.filter.update(results, now)
        for tid, r in zip(tracker_ids, results):  # the filter has read the canonical ids; hand back the current ones
            r.track_id = tid
        required = self.compliance.required_items
        self.violators = {tid: m for tid, sp in zip(tracker_ids, smoothed) if (m := missing_items(sp, required))}
        return smoothed

    def hold_alert(self, now: float, pts: float, tracker_id, new, missing) -> bool:
        """Hold a new violation of this track; an item starting while one is held joins it. True when this opened
        a new held alert (not a join)."""
        required = self.compliance.required_items
        held = self.held.get(tracker_id)
        if held is None:
            self.held[tracker_id] = {"now": now, "pts": pts, "track_id": tracker_id,
                                     "new": [i for i in required if i in new],
                                     "missing": [i for i in required if i in missing]}
            return True
        held["new"] = [i for i in required if i in held["new"] or i in new]
        held["missing"] = [i for i in required if i in held["missing"] or i in missing]
        return False

    def due_alerts(self, now: float, hold_s: float) -> list:
        """The held alerts held for hold_s by `now`, oldest first; they are no longer held."""
        due = sorted((h for h in self.held.values() if now - h["now"] >= hold_s - _EPS), key=lambda h: h["now"])
        for h in due:
            del self.held[h["track_id"]]
        return due

    def note_alert(self, now: float, pts: float, tracker_id, items) -> "dict | None":
        """Record an alert, and return the hint that it may repeat an earlier one of this stream: the most recent
        alert, for a shared item, of a different track in this track's chain of links. {"possible_duplicate_of":
        tracker id, "pts", "items", "seconds_since"} plus, when this track continues another, the "gap_s" and
        "distance_heights" of that link; or None."""
        self.alerts = [a for a in self.alerts if now - a[0] <= self.temporal.forget_after_s]
        chain = self.links.chain(tracker_id)
        hint = None
        for t, a_pts, a_tid, a_chain, a_items in reversed(self.alerts):
            shared = [i for i in a_items if i in items]
            if chain is None or a_chain != chain or a_tid == tracker_id or not shared:
                continue
            hint = {"possible_duplicate_of": a_tid, "pts": a_pts, "items": shared, "seconds_since": round(now - t, 1)}
            link = self.links.link(tracker_id)
            if link is not None:
                hint.update(gap_s=round(link[1], 3), distance_heights=round(link[2], 3))
            break
        self.alerts.append((now, pts, tracker_id, chain, list(items)))
        return hint


def missing_items(sp, required) -> list:
    """The items this smoothed person is missing now (smoothed ABSENT), in required_items order."""
    return [name for name in required if sp.item_status.get(name) is ItemStatus.ABSENT]


def join_items(items) -> str:
    """["helmet"] -> "helmet"; ["helmet", "vest"] -> "helmet and vest"; three -> "a, b and c"."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
