"""Per-stream, app-measured durations for every tracked object.

The VLM sees 2 fps of frames and cannot tell whether a truck has been parked for
one minute or eleven, so any duration it reports is a guess. This module counts
instead: it consumes the detections already flowing on `iris:meta`
(`(cls, conf, track_id, bbox)`, bbox in mux pixels) and renders a TRACKED OBJECTS
block of exact presence/stationary durations that the prompt tells the model to
read rather than re-estimate.

Pure state and text - no Redis, no config, no IO - so it is testable off-GPU,
the same shape as `app/severity.py` and `app/chat_select.py`.

Identity is the tracker's: a record dropped for staleness and then reappearing
under the same id starts fresh, and a tracker re-ID mid-dwell is a brand-new
object here. That is the same accepted limitation `TrackHistory` carries in the
sidecar (see CLAUDE.md's two-layer re-ID note), not a bug to paper over - a
duration attributed across a re-ID would be exactly the false precision this
module exists to remove.
"""
import math
from collections import deque
from dataclasses import dataclass, field

_HEADING = "## TRACKED OBJECTS (measured — these durations are exact, not estimates)"
_LABEL_WIDTH = 8
_PRESENT_WIDTH = 16


def format_duration(seconds: float) -> str:
    """`<M>m<SS>s` under an hour, `<H>h<MM>m` at or above. Never negative."""
    total = max(int(seconds), 0)
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    return f"{total // 60}m{total % 60:02d}s"


def duration_seconds(seconds: float) -> float:
    """`format_duration`'s numeric twin, for the API: 1dp, never negative.

    Both halves of the memory answer the HTTP route with these, so the rendered
    block and the JSON can never disagree about a duration's sign.
    """
    return round(max(float(seconds), 0.0), 1)


@dataclass
class TrackRecord:
    oid: int
    label: str
    first_seen: float
    last_seen: float
    positions: deque = field(default_factory=deque)  # (t, cx, cy) in mux pixels, trimmed to window_seconds
    anchor: tuple | None = None                      # (cx, cy) the stationary test measures against
    stationary_since: float | None = None            # None until the first positioned sample


class TrackObserver:
    """One per stream. Fed detections by the caller's sampler; renders the block."""

    def __init__(self, *, frame_width: int, frame_height: int,
                 window_seconds: float = 120.0, stale_seconds: float = 30.0,
                 stationary_radius: float = 0.02, max_rendered: int = 20):
        self._window_seconds = float(window_seconds)
        self._stale_seconds = float(stale_seconds)
        self._max_rendered = int(max_rendered)
        # One pixel threshold off the frame diagonal, computed once: normalising each
        # axis separately would make 0.02 a different real distance horizontally than
        # vertically on a non-square frame.
        self._stationary_px = float(stationary_radius) * math.hypot(frame_width, frame_height)
        self._tracks: dict[int, TrackRecord] = {}

    def observe(self, now: float, dets: list[tuple]) -> None:
        for cls, _conf, track_id, bbox in dets:
            if track_id is None:
                continue  # no identity across frames, so every duration would be a lie
            rec = self._tracks.get(track_id)
            if rec is None:
                rec = TrackRecord(oid=track_id, label=str(cls), first_seen=now, last_seen=now)
                self._tracks[track_id] = rec
            rec.label = str(cls)
            rec.last_seen = now
            if bbox is None:
                continue  # present, but no position sample and no anchor decision
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            rec.positions.append((now, cx, cy))
            cutoff = now - self._window_seconds
            while rec.positions and rec.positions[0][0] < cutoff:
                rec.positions.popleft()
            if rec.anchor is None or math.hypot(cx - rec.anchor[0], cy - rec.anchor[1]) > self._stationary_px:
                rec.anchor = (cx, cy)
                rec.stationary_since = now
        self._prune(now)

    def live_ids(self) -> set:
        """The ids currently tracked, as of the last `observe()` (which prunes).

        Read-only, no state of its own. It exists because `SceneMemory` keys entity
        notes on track id but owns no tracks: without this set it cannot tell a note
        about someone still on camera from one about a track that vanished ten
        minutes ago, and would render the latter forever.
        """
        return set(self._tracks)

    def present_for(self, oid: int, now: float) -> float | None:
        rec = self._tracks.get(oid)
        return None if rec is None else now - rec.first_seen

    def stationary_for(self, oid: int, now: float) -> float | None:
        rec = self._tracks.get(oid)
        if rec is None or rec.stationary_since is None:
            return None
        return now - rec.stationary_since

    def speed(self, oid: int, now: float, window_seconds: float = 5.0) -> float | None:
        """Pixels/second between the ends of the last `window_seconds` of samples.

        Not rendered: mux pixels have no calibrated relationship to m/s, and a bare
        number would be read as authoritative. Kept as a query for a later phase.
        """
        rec = self._tracks.get(oid)
        if rec is None:
            return None
        window = [p for p in rec.positions if p[0] >= now - window_seconds]
        if len(window) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = window[0], window[-1]
        if t1 <= t0:
            return None
        return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)

    def render(self, now: float) -> str:
        """The prompt block, longest-present first, capped. `""` when nothing is tracked.

        Wrapped in newlines so the analyzer's `{tracked}` slot spaces correctly and
        collapses to today's exact bytes when this returns `""`.
        """
        records = sorted(self._tracks.values(), key=lambda r: r.first_seen)[:self._max_rendered]
        if not records:
            return ""
        rows = []
        for rec in records:
            present = f"present {format_duration(now - rec.first_seen)}"
            still = self.stationary_for(rec.oid, now)
            if still is None:
                # No positioned sample ever arrived. parse_dets deliberately tolerates a
                # sidecar too old to publish bboxes, and against one EVERY track lands
                # here - so a "moving" default would assert motion for the whole frame
                # having measured none of it. The row stops at what was actually counted.
                rows.append(f"  #{rec.oid} {rec.label.ljust(_LABEL_WIDTH)} {present}")
                continue
            # Below a whole second there is no duration worth naming, so the honest word
            # is "moving" - which is what a track whose anchor was just reset gets.
            state = "moving" if int(still) < 1 else f"stationary {format_duration(still)}"
            rows.append(f"  #{rec.oid} {rec.label.ljust(_LABEL_WIDTH)} {present.ljust(_PRESENT_WIDTH)}{state}")
        return "\n" + _HEADING + "\n" + "\n".join(rows) + "\n"

    def snapshot(self, now: float) -> dict:
        """`render`'s structured twin for the HTTP route: same order, same cap, no
        formatting. Read-only — it must not perturb what the next prompt renders.

        Durations, never `first_seen`: that stamp is `time.monotonic()`, an arbitrary
        process-local number no client can turn into a wall clock.
        """
        records = sorted(self._tracks.values(), key=lambda r: r.first_seen)[:self._max_rendered]
        return {"tracked": [{
            "id": rec.oid,
            "label": rec.label,
            "present_for_s": duration_seconds(now - rec.first_seen),
            # None, not 0, when no position was ever measured - the same distinction
            # `render` makes by dropping the state column. 0 would assert stillness.
            "stationary_for_s": (None if (still := self.stationary_for(rec.oid, now)) is None
                                 else duration_seconds(still)),
        } for rec in records]}

    def _prune(self, now: float) -> None:
        cutoff = now - self._stale_seconds
        for oid in [oid for oid, rec in self._tracks.items() if rec.last_seen < cutoff]:
            del self._tracks[oid]
