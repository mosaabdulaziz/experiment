"""stability: tracker -- the accepted shelf-demo's per-region logic for one camera, without DeepStream. The parts are
the demo's own code, vendored unchanged into core/ (tracker.py, person_gate.py, situations.py, from shelf-demo
service/ at the commit their headers name); this file only wires them the way the demo's CameraWorker does
(service/camera.py: gate_active, apply_detections), once per named region:

- each void box of at least TrackerConfig.conf belongs to the FIRST region whose polygon contains its centre, so
  overlapping regions never count it twice;
- PersonGate: a region a (padded) person box touches is paused until it has been person-free for `resume_s`;
- StableTracker, per region: an active region learns from its boxes (update), a paused one keeps its confirmed gaps
  and drops its candidates (freeze); split() gives the confirmed gaps ("spots");
- Situation, per region: spots 0 -> >=1 is `raised` (reason new), more spots than last reported is `raised` (more),
  every `reraise_s` while open is `raised` (periodic), zero spots for `clear_s` of person-free time is `cleared`
  (gone). The module turns these into shelf_empty / shelf_restocked alerts: one open situation per region.

Times are seconds on the engine's monotonic clock; every frame stands for 1/fps seconds (the trackers' sample
interval), as the demo's video loop does with frame_index / fps.
"""
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .core.person_gate import PersonGate, region_masks
from .core.situations import Situation, Spot
from .core.tracker import StableTracker


@dataclass
class TrackerConfig:
    conf: float = 0.08            # the demo's --conf / catalog `confidence`: minimum void confidence
    confirm_s: float = 2.0        # --confirm-sec / catalog `confirmSeconds`
    max_miss_s: float = 1.0       # --max-miss-sec (service/camera.py MAX_MISS_S)
    min_iou: float = 0.3          # StableTracker min_iou
    min_hit_ratio: float = 0.5    # StableTracker min_hit_ratio
    fps: float = 0.0              # frames per second (each frame is 1/fps s of samples); 0 = measured from the pts


@dataclass
class PersonConfig:
    enabled: bool = True          # the demo always pauses (catalog `pauseOnPerson`)
    conf: float = 0.35            # --person-conf
    pad: float = 0.15             # --person-pad: fraction of the box's width/height added per side
    resume_s: float = 0.4         # service/camera.py RESUME_S: the demo's 10 frames at 25 fps


@dataclass
class SituationConfig:
    clear_s: float = 30.0         # catalog `clearSeconds` default: person-free seconds with zero spots to clear
    reraise_s: float = 300.0      # service RERAISE_SECONDS: re-send raised this often while open


@dataclass
class Event:
    kind: str                     # "raised" | "cleared"
    region: str
    intent: object                # core.situations.Intent
    alert_id: str                 # one id per open situation, raised ... cleared
    tracks: list = field(default_factory=list)   # the confirmed StableTracker tracks behind a raised intent
    opened_at: float = 0.0        # engine time the situation opened


class RegionTrackers:
    """One camera: named regions (polygons, in the pixels of the boxes), each with its tracker and situation, and one
    person gate over all of them."""

    def __init__(self, regions: list, cfg: TrackerConfig, fps: float, size, person: "PersonConfig | None" = None,
                 situation: "SituationConfig | None" = None):
        self.names = [name for name, _ in regions]
        self.polys = [np.asarray(poly, np.float32) for _, poly in regions]   # scaled, so never rounded
        self.cfg, self.fps, self.interval = cfg, fps, 1.0 / fps
        self.size = (int(size[0]), int(size[1]))
        self.person = person or PersonConfig(enabled=False)
        self.situation_cfg = situation or SituationConfig()
        self.trackers = [StableTracker(cfg.confirm_s, cfg.max_miss_s, self.interval, cfg.min_iou, cfg.min_hit_ratio)
                         for _ in regions]
        self.situations = [Situation(self.situation_cfg.clear_s, self.situation_cfg.reraise_s) for _ in regions]
        self.alert_ids = [None] * len(regions)
        self.opened_at = [0.0] * len(regions)
        w, h = self.size
        # The demo service's masks: the polygon rounded to whole pixels (zone.to_pixels), filled by region_masks.
        self.gate = PersonGate(region_masks([np.round(p).astype(np.int32) for p in self.polys], w, h),
                               self.person.resume_s)
        self.active = [True] * len(regions)
        self.samples = 0
        self.paused_samples = 0           # samples with at least one region paused (the demo's end-of-run count)

    def owner(self, box) -> "int | None":
        """Index of the first region whose polygon contains the box centre."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return next((i for i, p in enumerate(self.polys) if cv2.pointPolygonTest(p, (cx, cy), False) >= 0), None)

    def step(self, boxes, persons, now: float, wall) -> list:
        """One frame at engine time `now` (s) and aware UTC `wall`: its void boxes (x1, y1, x2, y2, conf) and its
        padded person boxes (int x1, y1, x2, y2). Returns this frame's Events."""
        n = len(self.trackers)
        active = self.gate.update(persons or [], now) if self.person.enabled else [True] * n
        self.active = active
        self.samples += 1
        if not all(active):
            self.paused_samples += 1
        mine = [[] for _ in range(n)]
        for b in boxes:
            if b[4] < self.cfg.conf:
                continue
            i = self.owner(b)
            if i is not None:
                mine[i].append(tuple(b))
        events = []
        w, h = self.size
        for i, tr in enumerate(self.trackers):
            if active[i]:
                tr.update(mine[i], now)
            else:
                tr.freeze(now)
            confirmed, _ = tr.split(now)
            spots = [Spot(_normalise(t["box"], w, h), float(t["conf"]), tr.stability(t, now)) for t in confirmed]
            for intent in self.situations[i].update(spots, active[i], now, wall):
                events.append(self._event(i, intent, confirmed, now))
        return events

    def close(self, wall, reason: str) -> list:
        """Clear every open situation now (reason as in Situation.close)."""
        return [self._event(i, intent, [], 0.0)
                for i, s in enumerate(self.situations) for intent in s.close(wall, reason)]

    def _event(self, i: int, intent, tracks: list, now: float) -> Event:
        if intent.state == "raised" and self.alert_ids[i] is None:
            self.alert_ids[i], self.opened_at[i] = uuid.uuid4().hex[:12], now
        ev = Event(intent.state, self.names[i], intent, self.alert_ids[i], list(tracks), self.opened_at[i])
        if intent.state == "cleared":
            self.alert_ids[i] = None
        return ev

    def confirmed(self) -> list:
        """(region name, alert id or None, track) of every confirmed gap now."""
        return [(self.names[i], self.alert_ids[i], t) for i, tr in enumerate(self.trackers)
                for t in tr.tracks if t.get("confirmed")]

    def seconds(self, frames: int) -> float:
        return frames / self.fps


def _normalise(box, w, h, ndigits: int = 4) -> list:
    """(x1, y1, x2, y2) pixels -> [x1, y1, x2, y2] in 0-1 (shelf-demo service/zone.py normalise_box)."""
    x1, y1, x2, y2 = box
    return [round(min(max(v, 0.0), 1.0), ndigits) for v in (x1 / w, y1 / h, x2 / w, y2 / h)]


class FpsProbe:
    """The stream's frame rate from its pts: the median of the first `samples` positive frame intervals (a looping
    file restarts its pts; that interval is skipped)."""

    def __init__(self, samples: int = 10):
        self.samples, self.last, self.dts = samples, None, []

    def feed(self, pts: float) -> "float | None":
        if self.last is not None and pts > self.last:
            self.dts.append(pts - self.last)
        self.last = pts
        if len(self.dts) < self.samples:
            return None
        return 1.0 / float(np.median(self.dts))
