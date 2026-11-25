"""Empty shelf space (voids) on supermarket shelves: a full-frame void detector, a stability check over time, alerts.

Adds a void detector (nvinfer "pgie_shelf", see Settings.config) that runs on the FULL frame, placed AFTER the tracker
(see ShelfVoid.gies), so its boxes are never tracked: they reach this module exactly as the detector made them, down
to its own low floor (pre-cluster-threshold 0.02). One frame of them is noise; `core/` keeps a decaying heat map per
camera, so a place that stays empty frame after frame becomes a stable void region, and the alerter turns those into
`shelf_empty` (the place stayed empty for confirm_s) and `shelf_restocked` (it has been gone for clear_s) alerts.

`stability: tracker` replaces the heat map with the accepted shelf-demo's per-region logic (stability.py; core/
tracker.py, person_gate.py and situations.py are vendored unchanged from shelf-demo service/): every frame, boxes of
at least tracker.conf go to the first region containing their centre; a gap that stays in place for
tracker.confirm_s is a confirmed spot; each region has one situation, raised as `shelf_empty` when its first spot is
confirmed (again, same alert id, when more spots appear and every situation.reraise_s) and cleared as
`shelf_restocked` after situation.clear_s person-free seconds without spots. A region a person (the shared pgie's
`person`, setting `person`) stands in front of is paused like the demo's: its confirmed spots are held, its
candidates dropped, and it resumes after person.resume_s person-free seconds. The heat-map settings are then unused.

The module is model-agnostic: it reads boxes labelled `label` (default "empty") from the GIE named pgie_shelf and never
a class id, so another model is a new ONNX plus its labels file and nvinfer config.

    modules:
      shelf_void:
        config: /workspace/iris/perception/models/DeepStream-Yolo-Shelf/config_infer_shelf.txt
        interval: 0              # MUST stay 0 (see Settings.interval)
        label: empty
        eval_fps: 30             # the heat map is updated at most this often per camera
        min_area: 400            # mux px^2
        decay: 0.85
        show_thresh: 0.5
        close_kernel: [40, 9]    # mux px (width, height)
        roi: {}                  # {camera id: [[x, y], ...]} in mux pixels; voids outside are ignored. With
                                 # stability: tracker, also {camera id: {size: [w, h], regions: [...] | {name: [...]}}}
        confirm_s: 2.0
        clear_s: 3.0
        match_iou: 0.3
        stability: heatmap       # or tracker
        tracker: {conf: 0.08, confirm_s: 2.0, max_miss_s: 1.0, min_iou: 0.3, min_hit_ratio: 0.5, fps: 0}
        person: {enabled: true, conf: 0.35, pad: 0.15, resume_s: 0.4}     # stability: tracker only
        situation: {clear_s: 30.0, reraise_s: 300.0}                      # stability: tracker only

Every heat-map and alert default is the standalone project's (shelf-monitoring 02538da), which was tuned on 1280x720
frames at 30 fps; here the boxes are in mux pixels (1080x720 by default). See README.md, "Tuning".
"""
import configparser
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2

from perception.engine.engine_path import gie_engine_path
from perception.engine.frame import FrameOutput, FrameView
from perception.engine.modules import GieSpec, Module, ModuleConfigError

from .adapter import EMPTY_LABEL, SHELF_GIE, person_boxes, void_boxes
from .core import AlertConfig, DetectorConfig, EmptyShelfAlerter, ShelfVoidDetector
from .stability import FpsProbe, PersonConfig, RegionTrackers, SituationConfig, TrackerConfig

logger = logging.getLogger("iris.sidecar")

# BGR. A confirmed void: violet, distinct from the person green, zone coral, PPE yellow/sky blue, fire orange and the
# proposed fall magenta (perception/NEXT_MODULES.md colour table). Drawn by this module (out.draw), not as an object.
BOX_VOID = (211, 0, 148)
RAW_KIND = "shelf_void_raw"              # ObjectView.kind of the detector's own boxes, set in transform
STABILITY = ("heatmap", "tracker")
MAX_SPOTS = 20                           # spots listed on one shelf_empty (shelf-demo MAX_SPOTS_IN_METADATA)


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo-Shelf/config_infer_shelf.txt"
    # Inference interval of the void detector. On a batch where `interval` skips inference no void metadata is
    # produced (the boxes are untracked, so nothing carries them over): a skipped frame would cool the heat map as if
    # the shelf were full. Keep it 0 (every frame).
    interval: int = 0
    # The detector's class that means "empty shelf space", by its name in the labels file.
    label: str = EMPTY_LABEL
    # Updates of the heat map per second, per camera. `decay` is per update, so the standalone tuning (30 fps video)
    # holds on any camera at or above this rate; a slower camera warms and cools the heat map more slowly.
    eval_fps: float = 30.0
    # The heat map (core/detector.py DetectorConfig), in mux pixels.
    min_area: int = 400
    decay: float = 0.85
    show_thresh: float = 0.5
    close_kernel: list = field(default_factory=lambda: [40, 9])
    # Per camera: only voids inside count on that camera; a camera not named here uses the whole frame. Either
    #   {camera id: polygon [[x, y], ...] in mux pixels}       one region, named "roi"
    # or, with stability: tracker only, several named regions drawn on a frame of `size` (e.g. the demo's roi.json,
    # in the source video's pixels), scaled to the mux size:
    #   {camera id: {size: [w, h], regions: [polygon, ...]}}   named R1, R2, ... (the demo's labels)
    #   {camera id: {size: [w, h], regions: {name: polygon}}}  named; the order is the order of the mapping
    # A void belongs to the FIRST region containing its box centre.
    roi: dict = field(default_factory=dict)
    # The alerter (core/alerts.py AlertConfig).
    confirm_s: float = 2.0
    clear_s: float = 3.0
    match_iou: float = 0.3
    # "heatmap": the decaying heat map and alerter above. "tracker": the accepted shelf-demo's per-region tracker,
    # person pause and situation, every frame (stability.py); the heat-map and alerter settings above are then unused.
    stability: str = "heatmap"
    # {TrackerConfig field: value} for stability: tracker: conf (0.08), confirm_s (2), max_miss_s (1), min_iou (0.3),
    # min_hit_ratio (0.5), fps (0 = measured from the stream's pts; each frame counts as 1/fps s of samples).
    tracker: dict = field(default_factory=dict)
    # {PersonConfig field: value} for stability: tracker: enabled (true), conf (0.35), pad (0.15 of the box per side),
    # resume_s (0.4). Persons are the shared pgie's `person` boxes; a region they touch is paused.
    person: dict = field(default_factory=dict)
    # {SituationConfig field: value} for stability: tracker: clear_s (30: person-free seconds without spots before
    # shelf_restocked), reraise_s (300: shelf_empty again, same alert id, while the region stays empty).
    situation: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ModuleConfigError("modules.shelf_void: interval must be >= 0")
        if self.eval_fps <= 0:
            raise ModuleConfigError(f"modules.shelf_void: eval_fps must be > 0, got {self.eval_fps!r}")
        if self.min_area < 0:
            raise ModuleConfigError(f"modules.shelf_void: min_area must be >= 0, got {self.min_area!r}")
        if not 0 < self.decay < 1 or not 0 < self.show_thresh < 1:
            raise ModuleConfigError("modules.shelf_void: decay and show_thresh must be between 0 and 1")
        k = self.close_kernel
        if not (isinstance(k, (list, tuple)) and len(k) == 2 and all(isinstance(v, int) and v > 0 for v in k)):
            raise ModuleConfigError(f"modules.shelf_void: close_kernel must be [width, height] > 0, got {k!r}")
        if self.confirm_s < 0 or self.clear_s < 0 or not 0 < self.match_iou <= 1:
            raise ModuleConfigError("modules.shelf_void: confirm_s and clear_s must be >= 0, match_iou in (0, 1]")
        if self.stability not in STABILITY:
            raise ModuleConfigError(f"modules.shelf_void: stability must be one of {list(STABILITY)}, "
                                    f"got {self.stability!r}")
        if not isinstance(self.roi, dict):
            raise ModuleConfigError(f"modules.shelf_void: roi must map a camera id to a polygon, got {self.roi!r}")
        for cam, roi in self.roi.items():
            if isinstance(roi, dict):
                _check_regions(cam, roi, self.stability)
            elif not _is_polygon(roi):
                raise ModuleConfigError(f"modules.shelf_void: roi.{cam} must be a polygon of >= 3 [x, y] points")
        self.tracker_config()             # build them once, so a bad key or value refuses startup
        self.person_config()
        self.situation_config()

    def tracker_config(self) -> TrackerConfig:
        cfg = _sub(TrackerConfig, "tracker", self.tracker)
        if not (0 <= cfg.conf <= 1 and cfg.confirm_s >= 0 and cfg.max_miss_s >= 0 and 0 < cfg.min_iou <= 1
                and 0 <= cfg.min_hit_ratio <= 1 and cfg.fps >= 0):
            raise ModuleConfigError(f"modules.shelf_void: tracker {self.tracker!r}: conf, min_hit_ratio in [0, 1], "
                                    "min_iou in (0, 1], confirm_s, max_miss_s and fps >= 0")
        return cfg

    def person_config(self) -> PersonConfig:
        cfg = _sub(PersonConfig, "person", self.person)
        if not (isinstance(cfg.enabled, bool) and 0 <= cfg.conf <= 1 and cfg.pad >= 0 and _num(cfg.resume_s)
                and cfg.resume_s >= 0):
            raise ModuleConfigError(f"modules.shelf_void: person {self.person!r}: enabled is true/false, conf in "
                                    "[0, 1], pad >= 0, resume_s >= 0")
        return cfg

    def situation_config(self) -> SituationConfig:
        cfg = _sub(SituationConfig, "situation", self.situation)
        if not (_num(cfg.clear_s) and _num(cfg.reraise_s) and cfg.clear_s >= 0 and cfg.reraise_s > 0):
            raise ModuleConfigError(f"modules.shelf_void: situation {self.situation!r}: clear_s >= 0, reraise_s > 0")
        return cfg

    def regions(self, stream_id: str, mux_size) -> list:
        """[(name, polygon)] of this camera for stability: tracker, in mux pixels (floats, not rounded: the centre
        test is then the one the polygons were drawn for); [] when none (the whole frame)."""
        roi = self.roi.get(stream_id)
        if not roi:
            return []
        if not isinstance(roi, dict):
            return [("roi", [(float(x), float(y)) for x, y in roi])]
        sx, sy = mux_size[0] / roi["size"][0], mux_size[1] / roi["size"][1]
        named = roi["regions"].items() if isinstance(roi["regions"], dict) else \
            ((f"R{i + 1}", poly) for i, poly in enumerate(roi["regions"]))
        return [(str(name), [(x * sx, y * sy) for x, y in poly]) for name, poly in named]

    def detector_config(self, stream_id: str) -> DetectorConfig:
        roi = self.roi.get(stream_id)
        return DetectorConfig(min_area=self.min_area, decay=self.decay, show_thresh=self.show_thresh,
                              close_kernel=tuple(self.close_kernel),
                              roi=[tuple(p) for p in roi] if roi else None)

    def alert_config(self, stream_id: str) -> AlertConfig:
        return AlertConfig(camera_id=stream_id, confirm_s=self.confirm_s, clear_s=self.clear_s,
                           match_iou=self.match_iou)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _sub(cls, key: str, raw):
    """A nested setting (tracker, person, situation) as its dataclass; unknown keys refuse startup."""
    if not isinstance(raw, dict):
        raise ModuleConfigError(f"modules.shelf_void: {key} must be a mapping, got {raw!r}")
    known = {f.name for f in dataclasses.fields(cls)}
    bad = sorted(set(raw) - known)
    if bad:
        raise ModuleConfigError(f"modules.shelf_void: {key} has unknown field(s) {bad}; known: {sorted(known)}")
    return cls(**raw)


def _is_polygon(poly) -> bool:
    return (isinstance(poly, list) and len(poly) >= 3
            and all(isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p) for p in poly))


def _check_regions(cam, roi: dict, stability: str) -> None:
    """roi.<cam> as {size: [w, h], regions: [polygon, ...] | {name: polygon}}."""
    if stability != "tracker":
        raise ModuleConfigError(f"modules.shelf_void: roi.{cam}: named regions need stability: tracker; the heat map "
                                "takes one polygon in mux pixels")
    if sorted(roi) != ["regions", "size"]:
        raise ModuleConfigError(f"modules.shelf_void: roi.{cam} must have exactly `size` and `regions`, "
                                f"got {sorted(roi)}")
    size, regions = roi["size"], roi["regions"]
    if not (isinstance(size, list) and len(size) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in size)):
        raise ModuleConfigError(f"modules.shelf_void: roi.{cam}.size must be [width, height] > 0, got {size!r}")
    polys = regions.values() if isinstance(regions, dict) else regions
    if not isinstance(regions, (dict, list)) or not regions or not all(_is_polygon(p) for p in polys):
        raise ModuleConfigError(f"modules.shelf_void: roi.{cam}.regions must be a non-empty list or mapping of "
                                "polygons of >= 3 [x, y] points")


def _require_model(config: str) -> None:
    """Refuse to start, saying where to get it, when the detector config or the ONNX it names is missing (a clone that
    has not run `just data-pull`); without this check nvinfer would fail later with a TensorRT build error that does not
    say which file is wrong or where it comes from. Also refuse a config that does not run on the full frame."""
    cp = configparser.ConfigParser()
    if not cp.read(config) or not cp.has_option("property", "onnx-file"):
        raise ModuleConfigError(f"modules.shelf_void: detector config {config} is missing or names no onnx-file")
    if cp.getint("property", "process-mode", fallback=1) != 1:
        raise ModuleConfigError(f"modules.shelf_void: detector config {config} must run on the full frame "
                                "(process-mode=1)")
    onnx = os.path.join(os.path.dirname(os.path.abspath(config)), cp.get("property", "onnx-file"))
    if not os.path.isfile(onnx):
        raise ModuleConfigError(
            f"modules.shelf_void: the shelf model {onnx} is missing. It is versioned with DVC: run "
            "`just data-pull models-shelf`, or export it as perception/models/DeepStream-Yolo-Shelf/README.md "
            "('Getting the model') describes, and put it at that path.")


class ShelfVoid(Module):
    name = "shelf_void"
    Settings = Settings

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        if settings.interval > 0:
            logger.warning("modules.shelf_void.interval is %d (> 0); skipped batches carry no void detections. "
                           "Set it to 0.", settings.interval)

    def gies(self) -> list:
        # role="secondary" PLACES the detector after the tracker, so nvtracker never deletes its low-confidence boxes
        # (it drops everything under its own floor); its config's process-mode=1 runs it on the full frame, with the
        # mux's batch, as the PPE README describes for a full-frame detector.
        s = self.settings
        _require_model(s.config)
        return [GieSpec(name=SHELF_GIE, config_path=s.config, role="secondary", interval=s.interval,
                        engine_path=gie_engine_path(s.config, self.services.batch_size), batch_from_mux=True)]

    def on_built(self, pipeline) -> None:
        s = self.settings
        for sid in self.services.stream_ids:
            st = self.services.state(self.name, sid)
            if s.stability == "tracker":
                st["regions"] = None                       # RegionTrackers, made once the frame rate is known
                st["fps_probe"] = FpsProbe()
                continue
            st["detector"] = ShelfVoidDetector(s.detector_config(sid))
            st["alerter"] = EmptyShelfAlerter(s.alert_config(sid))
            st["last_eval"] = None

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        if self.settings.stability == "tracker":
            self._track(frame)
        else:
            self._heatmap(frame)

    def _track(self, frame: FrameView) -> None:
        """stability: tracker. Every frame, like the demo: the persons into the gate, this frame's void boxes into the
        region trackers, their confirmed spots into each region's situation; its intents become alerts."""
        st = self.services.state(self.name, frame.stream_id)
        if "fps_probe" not in st:
            return
        rt = st["regions"]
        if rt is None:
            s = self.settings
            cfg = s.tracker_config()
            fps = cfg.fps or st["fps_probe"].feed(frame.pts)
            if fps is None:
                return
            regions = s.regions(frame.stream_id, frame.mux_size) or [("frame", _whole(frame.mux_size))]
            person, situation = s.person_config(), s.situation_config()
            rt = st["regions"] = RegionTrackers(regions, cfg, fps, frame.mux_size, person, situation)
            logger.info("[%s] shelf_void tracker at %.2f fps: conf %.2f, confirm %.1f s, forget after %.1f s, clear "
                        "after %.0f s, re-raise every %.0f s; regions %s; person pause %s", frame.stream_id, fps,
                        cfg.conf, cfg.confirm_s, cfg.max_miss_s, situation.clear_s, situation.reraise_s,
                        ", ".join(rt.names),
                        f"on (resume after {person.resume_s:g} s)" if person.enabled else "off")
        persons = None
        if rt.person.enabled:
            p = rt.person
            persons = person_boxes(frame.objects, frame.mux_size, p.conf, p.pad)
        for ev in rt.step(void_boxes(frame.objects, label=self.settings.label), persons, frame.now, _utc_now()):
            self._track_alert(frame, ev)
        if rt.samples % 100 == 0:                         # the demo's progress line
            counts = {}
            for name, _aid, _t in rt.confirmed():
                counts[name] = counts.get(name, 0) + 1
            logger.info("[%s] shelf_void frame %d: %s; %d frame(s) with a region paused", frame.stream_id,
                        rt.samples, "  ".join(f"{n}={counts.get(n, 0)}{'' if a else '(P)'}"
                                              for n, a in zip(rt.names, rt.active)), rt.paused_samples)

    def _track_alert(self, frame: FrameView, ev) -> None:
        """One situation intent of a region as an alert. raised -> shelf_empty (reason new / more / periodic; the
        best spot's box, the spot count, every spot normalised), cleared -> shelf_restocked; the same alert_id from
        the first raise to the clear. `occurred_at` is the intent's own UTC time (what IRIS calls occurredAt)."""
        it = ev.intent
        extra = {"alert_id": ev.alert_id, "region": ev.region, "reason": it.reason,
                 "occurred_at": _iso(it.occurred_at)}
        if ev.kind == "raised":
            spots = sorted(it.spots, key=lambda sp: (-sp.stability, -sp.detector_score))
            best = max(ev.tracks, key=lambda t: t["conf"])
            x1, y1, x2, y2 = (int(round(v)) for v in best["box"])
            n = len(spots)
            extra.update(bbox=f"{x1},{y1},{x2},{y2}", score=f"{max(sp.detector_score for sp in spots):.3f}",
                         stability=f"{max(sp.stability for sp in spots):.3f}",
                         spots=json.dumps([{"bbox": sp.bbox, "detectorScore": round(sp.detector_score, 4),
                                            "stability": round(sp.stability, 4)} for sp in spots[:MAX_SPOTS]],
                                          separators=(",", ":")))
            if it.reason == "new":                        # one clip per situation, not per re-raise
                self.services.request_clip(frame.stream_id, frame.now)
            message = (f"Empty shelf space on {frame.stream_id} in region {ev.region}: {n} spot{'s' if n != 1 else ''}"
                       f" at ({x1},{y1})-({x2},{y2})")
            self.services.alert(frame.stream_id, "shelf_empty", message=message, severity="warning",
                                category="operations", pts=frame.pts, count=n, **extra)
        else:
            empty_for = max(0.0, frame.now - ev.opened_at)
            message = f"Shelf restocked on {frame.stream_id} in region {ev.region} (empty for {empty_for:.0f} s)"
            self.services.alert(frame.stream_id, "shelf_restocked", message=message, severity="info",
                                category="operations", pts=frame.pts, count=0, empty_for_s=f"{empty_for:.1f}",
                                clear_s=f"{it.clear_seconds:g}", **extra)
        logger.info("[%s] shelf %s (%s) on source frame %d: %s (alert %s)", frame.stream_id, ev.kind, it.reason,
                    frame.frame_num, message, ev.alert_id)

    def _heatmap(self, frame: FrameView) -> None:
        """Every frame, ahead of the publish gate, at most eval_fps per camera: this frame's void boxes into the heat
        map, its stable regions into the alerter, which says when a place became empty or was restocked. Times are
        the engine's monotonic `now` (pts restarts when a file loops); the alert's pts is the frame's own."""
        st = self.services.state(self.name, frame.stream_id)
        if "detector" not in st:
            return
        last = st["last_eval"]
        if last is not None and frame.now - last < 1.0 / self.settings.eval_fps - 1e-6:
            return
        st["last_eval"] = frame.now
        boxes = void_boxes(frame.objects, label=self.settings.label)
        regions = st["detector"].update_boxes(boxes, frame.mux_size)
        for ev in st["alerter"].update(regions, frame.now):
            self._alert(frame, ev)

    def _alert(self, frame: FrameView, ev) -> None:
        x1, y1, x2, y2 = ev.bbox
        where = f"at ({x1},{y1})-({x2},{y2})"
        extra = {"alert_id": ev.alert_id, "bbox": f"{x1},{y1},{x2},{y2}", "score": f"{ev.score:.3f}"}
        if ev.event == "RAISED":
            # Like ppe and fire_alerts: the clip comes from a smart-record session started when the alert is raised.
            self.services.request_clip(frame.stream_id, frame.now)
            message = f"Empty shelf space on {frame.stream_id} {where}"
            self.services.alert(frame.stream_id, "shelf_empty", message=message, severity="warning",
                                category="operations", pts=frame.pts, **extra)
        else:
            empty_for = ev.timestamp - ev.first_seen
            message = f"Shelf restocked on {frame.stream_id} {where} (empty for {empty_for:.0f} s)"
            self.services.alert(frame.stream_id, "shelf_restocked", message=message, severity="info",
                                category="operations", pts=frame.pts, empty_for_s=f"{empty_for:.1f}", **extra)
        logger.info("[%s] shelf %s: %s (alert %s)", frame.stream_id, ev.event.lower(), message, ev.alert_id)

    def transform(self, frame: FrameView) -> None:
        """The detector's own boxes are evidence for the heat map, not detections: never drawn (one frame of them is
        mostly noise at a 0.02 floor) and never published, whatever the product's relevant_classes say."""
        for o in frame.objects:
            if o.gie_name == SHELF_GIE:
                o.kind, o.draw, o.publish = RAW_KIND, False, False

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        """Draw every void with an open alert (the alerter's last box for it) and, on meta frames, publish them as the
        `shelf_void` field of iris:meta."""
        st = self.services.state(self.name, frame.stream_id)
        if st.get("regions") is not None:
            active = [(aid, tuple(int(round(v)) for v in t["box"]), name)
                      for name, aid, t in st["regions"].confirmed()]
        elif "alerter" in st:
            active = [(ev.alert_id, ev.bbox, None) for ev in st["alerter"].active()]
        else:
            return
        if not active:
            return
        if frame.need_meta:
            out.fields[self.name] = [{"alert_id": aid, "bbox": list(box), **({"region": r} if r else {})}
                                     for aid, box, r in active]

        def draw(img, sx, sy, boxes=active):
            for _aid, (x1, y1, x2, y2), _region in boxes:
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(img, p1, p2, BOX_VOID, 2)
                cv2.putText(img, "empty", (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_VOID, 2)

        out.draw(draw)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """Aware datetime -> '2026-11-12T10:00:47.080Z' (shelf-demo service/messages.py utc_iso)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _whole(size) -> list:
    w, h = int(size[0]), int(size[1])
    return [(0, 0), (w, 0), (w, h), (0, h)]


MODULE = ShelfVoid
