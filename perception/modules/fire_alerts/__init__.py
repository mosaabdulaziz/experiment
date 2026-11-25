"""Full-frame fire/smoke detection and alerting.

Adds a second full-frame primary detector (a D-Fire-trained YOLO) and raises an incident when fire or smoke persists:
one sustained latch per class per stream, a clip request while an incident is open, and a resolve message when it
clears. Deliberately not scoped to a drawn region: any fire anywhere in frame matters, so it needs no zone, no ROI
status and no analytics element.

    modules:
      fire_alerts:
        interval: 0                      # MUST stay 0 (see Settings.interval)
        fire_sustain_seconds: 2.0
        smoke_sustain_seconds: 4.0

Alerts go to iris:overcrowd_alert:<cam> with kind=fire_smoke / fire_smoke_resolved (the app's count_alerts reads
that stream); fire and smoke boxes are drawn in the hazard colour, which is deliberately not the zone-violation red.
"""
import logging
from dataclasses import dataclass, field

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import GieSpec, Module, ModuleConfigError
from perception.lib.overlay_boxes import box
from perception.engine.engine_path import gie_engine_path

from .state import FireAlertSettings, FireAlertState
from perception.lib.feature_zones import LiveFeatureZones

logger = logging.getLogger("iris.sidecar")

# BGR. Vivid orange (#ff8c00). Deliberately NOT the zone-violation red: red already means "person violating a zone",
# and an operator scanning the wall must be able to tell a trespasser from a fire at a glance.
BOX_HAZARD = (0, 140, 255)
HAZARD_CLASSES = ("fire", "smoke")
_HAZARD_PRIORITY = 10                    # beats any zone colouring, whichever module runs first


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo-Fire/config_infer_fire_box.txt"
    # Inference interval of the fire detector. It runs BEFORE the tracker, so on a batch where `interval` skips
    # inference no fire metadata is produced at all: the alert would count a skipped frame as "no fire" and, at
    # interval >= ~10, the gap outlasts dropout_grace_seconds on every cycle, so the sustain window could never
    # accumulate and no alert could ever fire. Keep it 0 (every frame).
    interval: int = 0
    classes: list = field(default_factory=lambda: list(HAZARD_CLASSES))
    # How long a class must persist before it alerts: the primary false-positive control. Fire is visually
    # unambiguous so it confirms fast; smoke shares its appearance with steam, vapour, exhaust and dust and needs
    # the longer window to tell them apart.
    fire_sustain_seconds: float = 2.0
    smoke_sustain_seconds: float = 4.0
    # How long the condition must stay clear before the incident resolves and a second alert can fire. Tuned
    # against the 61 s bucketfire sample, whose fire-absence window repeats at ~27 s: the release must be shorter
    # than the gap or the alert stays latched through the whole loop.
    release_seconds: float = 10.0
    # A detection below this does not count. Must be positive: a non-positive threshold would admit DeepStream's
    # -0.1 shadow-track sentinel, letting a tracker ghost with no detector support satisfy the sustain window.
    min_confidence: float = 0.45
    # Detectors flicker, smoke especially; this long a gap does not reset the sustain window.
    dropout_grace_seconds: float = 1.0
    # The shared zones file. A camera with zones whose `fire` block is on counts only fire/smoke whose box centre is
    # inside one, and the alert names it; a camera with none covers the whole frame (perception/lib/feature_zones).
    zones_path: str = "data/zones_config.txt"

    def __post_init__(self) -> None:
        if not self.classes:
            raise ModuleConfigError("modules.fire_alerts: classes is empty; no class would ever be counted and no "
                                    "alert could ever fire")
        if self.min_confidence <= 0:
            raise ModuleConfigError(
                f"modules.fire_alerts: min_confidence is {self.min_confidence!r}; a non-positive threshold admits "
                "the -0.1 shadow-track sentinel, letting a tracker ghost satisfy the sustain window indefinitely")
        if self.interval < 0:
            raise ModuleConfigError("modules.fire_alerts: interval must be >= 0")


class FireAlerts(Module):
    name = "fire_alerts"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        return [ModelRef(settings.config, "config", "nvinfer")]

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.alert_settings = FireAlertSettings(
            enabled=True, classes=tuple(settings.classes),
            sustain_seconds={"fire": settings.fire_sustain_seconds, "smoke": settings.smoke_sustain_seconds},
            release_seconds=settings.release_seconds, min_confidence=settings.min_confidence,
            dropout_grace_seconds=settings.dropout_grace_seconds)
        if settings.interval > 0:
            logger.warning(
                "modules.fire_alerts.interval is %d (> 0); fire alerting expects every-frame inference from the "
                "fire detector, and a nonzero interval can prevent fire/smoke alerts from ever firing. Set it to 0.",
                settings.interval)
        self._zones = LiveFeatureZones("fire", settings.zones_path, services.redis)

    def control(self) -> dict:
        return self._zones.control()

    def gies(self) -> list:
        s = self.settings
        return [GieSpec(name="pgie_fire", config_path=s.config, role="primary", interval=s.interval,
                        engine_path=gie_engine_path(s.config, self.services.batch_size))]

    def on_built(self, pipeline) -> None:
        for sid in self.services.stream_ids:
            state = self.services.state(self.name, sid)
            state["alerts"] = FireAlertState(self.alert_settings)
            state["recheck_at"] = 0.0
            state["zone_by_class"] = {}
        self._zones.start()

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        """Runs on every frame, not gated on the analysis throttle: this is an alert path and must see them all."""
        st = self.services.state(self.name, frame.stream_id)
        alerts = st["alerts"]
        emit = self.services.emit
        detections, zone_by_class = [], st.setdefault("zone_by_class", {})
        boxes = []
        best: dict = {}
        for o in frame.objects:
            if o.label not in self.alert_settings.classes:
                continue
            covered, zone = self._zones.covering(frame.stream_id, o.left + o.width / 2.0, o.top + o.height / 2.0)
            if not covered:
                continue                    # outside every fire zone of this camera: does not count
            detections.append((o.label, o.conf))
            if o.conf >= self.alert_settings.min_confidence:          # only what counts toward an alert is drawn
                boxes.append(box(o.bbox, o.label if o.label in HAZARD_CLASSES else "fire",
                                 label=f"{o.label} {o.conf:.2f}"))
            if o.conf > best.get(o.label, -1.0):
                best[o.label] = o.conf
                zone_by_class[o.label] = zone   # the zone of the class's most confident detection, kept for "started"
        # the hazards' boxes for the alerts' clips, drawn whatever else the clip shows (app/overlay_track.py)
        self.overlay_boxes.write(frame.stream_id, frame.pts, boxes, at=frame.arrived)
        for ev in alerts.observe(frame.now, detections):
            if ev.kind == "started":
                fields = {"frame_num": frame.frame_num, "pts": f"{frame.pts:.3f}", "kind": "fire_smoke",
                          "class": ev.cls, "count": ev.count, "onset_age": f"{ev.onset_age:.3f}"}
                if zone_by_class.get(ev.cls):           # only with fire zones: the entry is otherwise as it was
                    fields["zone"] = zone_by_class[ev.cls]
                emit(f"iris:overcrowd_alert:{self.services.key_id(frame.stream_id)}", fields)
                logger.info("[%s] %s detected (count=%d, confirmed, onset %.1fs ago)",
                            frame.stream_id, ev.cls, ev.count, ev.onset_age)
                self.services.request_clip(frame.stream_id, frame.now)
            else:  # "resolved": no frame or count to report, no clip to start
                emit(f"iris:overcrowd_alert:{self.services.key_id(frame.stream_id)}",
                     {"pts": f"{frame.pts:.3f}", "kind": "fire_smoke_resolved", "class": ev.cls,
                      "duration": f"{ev.duration:.3f}"})
                logger.info("[%s] %s cleared (duration %.1fs)", frame.stream_id, ev.cls, ev.duration)
        if alerts.has_open_incident:
            # A long-burning incident can outlast one smart-record session (fixed duration). Keep re-arming while
            # it is open: a safe no-op while a session is active (the trigger coalesces) and a fresh session the
            # instant the previous one closes, so one incident gets back-to-back clips instead of coverage
            # stopping after the first. Throttled to ~1/s: the status read is a real (locked) property access.
            if frame.now >= st["recheck_at"]:
                st["recheck_at"] = frame.now + 1.0
                self.services.request_clip(frame.stream_id, frame.now)

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        for o in frame.objects:
            if o.draw and o.label in HAZARD_CLASSES:
                out.annotate(o, color=BOX_HAZARD, priority=_HAZARD_PRIORITY)


MODULE = FireAlerts
