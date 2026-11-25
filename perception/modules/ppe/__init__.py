"""Helmet and high-visibility vest detection on people: a PPE detector, per-person rules and candidate violation alerts.

Adds a PPE detector (VoxDroid YOLOv8s, Construction Site Safety dataset) that runs on each tracked person's crop of
the COCO pgie, placed AFTER the tracker (see Ppe.gies), so its boxes are never tracked: they reach modules exactly as
the detector made them, in frame coordinates, untracked, with gie_name "pgie_ppe" and labels Hardhat, NO-Hardhat,
Safety Vest, NO-Safety Vest. Persons still come from the pgie, tracked. The config file filters out the model's other classes (Person, Mask, NO-Mask, Safety Cone,
machinery, vehicle). PPE objects carry no track id (DeepStream's untracked id): never key anything on `o.id` for them.

    modules:
      ppe:
        config: /workspace/iris/perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt
        interval: 0                      # MUST stay 0 (see Settings.interval)
        required_items: [helmet, vest]
        items: {vest: {absent_conf: 0.5}}  # per-item overrides of compliance/config.py PPEItemConfig, by item name
        min_person_height_px: 32         # mux pixels; every item, unless `items` overrides it
        edge_margin_px: 2                # mux pixels
        eval_fps: 20
        temporal: {window: 8, min_absent_frames: 6}   # compliance/config.py TemporalConfig fields
        reid: {enabled: false}           # reid.py ReidConfig fields; the re-ID grace is off by default
        links: {max_gap_s: 4, max_dist: 1.2}   # reid.py LinkConfig fields: the duplicate hint's track links
        alert_hold_s: 1                  # a new violation waits this long for a second item of the same track

The model keeps its own class ids 0-9, which overlap COCO's (Hardhat = 0 = person, Safety Vest = 7 = truck). The
tracker spike (2026-10-26) found no effect of that overlap; it found that the tracker drops PPE boxes instead
(docs/PPE_MODULE_PLAN.md, section 10).
"""
import configparser
import dataclasses
import logging
import os
from dataclasses import dataclass, field

from perception.engine.engine_path import gie_engine_path
from perception.engine.frame import FrameOutput, FrameView
from perception.engine.modules import GieSpec, Module, ModuleConfigError
from perception.lib.overlay_boxes import corners_box

from .adapter import PERSON_GIE, PERSON_LABEL, PPE_GIE, PPE_LABELS, UNTRACKED
from .compliance.config import ComplianceConfig, PPEItemConfig, TemporalConfig
from .reid import LinkConfig, ReidConfig
from .compliance.types import ItemStatus
from .state import StreamRules, join_items, missing_items
from perception.lib.feature_zones import LiveFeatureZones
DEFER_FORGET_S = 10.0     # a violation begun outside every PPE zone is forgotten once its track is unseen this long

logger = logging.getLogger("iris.sidecar")

# BGR. A person in violation: yellow at priority 8 (perception/NEXT_MODULES.md colour table), above the zone
# colouring (1) and below the fire hazard orange (10), whichever module runs first.
BOX_PPE_VIOLATION = (0, 255, 255)
_VIOLATION_PRIORITY = 8
# The PPE item boxes themselves: sky blue, distinct from the person green, zone coral, PPE yellow and fire orange.
# Low priority, so any module that colours these boxes for its own reason wins.
BOX_PPE_ITEM = (255, 191, 0)
_ITEM_PRIORITY = 2
PPE_KIND = "ppe_item"                    # ObjectView.kind of the PPE detector's boxes, set in transform
CLOSE_AFTER_PRESENT_S = 2.0              # the module's default for temporal.close_after_present_s (Settings.temporal)


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt"
    # Inference interval of the PPE detector. On a batch where `interval` skips inference no PPE metadata is produced
    # at all (the boxes are untracked, so nothing carries them over): a skipped frame would read as "no helmet".
    # Keep it 0 (every frame).
    interval: int = 0
    # Items every person must wear, by name (the vendored defaults define "helmet" and "vest").
    required_items: list = field(default_factory=lambda: ["helmet", "vest"])
    # {item name: {PPEItemConfig field: value}}: patches the vendored default items field by field, e.g.
    # {vest: {absent_conf: 0.5}}. Thresholds and geometry not named here keep their evaluated values.
    items: dict = field(default_factory=dict)
    # The two pixel thresholds of the rules, in MUX pixels (1080x720 by default), where the boxes are. The evaluated
    # values were 48 px and 3 px on 1080-high frames; scaled to the mux height: 48 * 720 / 1080 = 32, 3 -> 2.
    # Every item gets min_person_height_px unless `items` overrides it for that item.
    min_person_height_px: int = 32
    edge_margin_px: int = 2
    # Frames per second the rules run at, per stream. The temporal window counts evaluated frames (8, "absent in 6 of
    # the last 8") and was tuned at 20 fps; a fixed rate keeps it about 0.4 s on every camera, whatever its fps.
    eval_fps: float = 20.0
    # {TemporalConfig field: value}: window, min_absent_frames, min_present_frames, cooldown_s, forget_after_s,
    # close_after_present_s. The module's default for close_after_present_s is CLOSE_AFTER_PRESENT_S (2 s), not the
    # vendored 0: an episode closes only after 2 s of unbroken PRESENT, so a cap read as a hardhat for a second does
    # not close it (test2.mp4: all 6 same-track repeat alerts came from 0.4-1.45 s of such false PRESENT).
    temporal: dict = field(default_factory=dict)
    # {ReidConfig field: value}: the re-ID grace, OFF by default (enabled: false). On, a new tracker id close to where
    # a recently lost one was (max_gap_s 2, max_speed 0.7 height/s, max_height_ratio 1.5, the lost worker's tallest
    # box of their last recent_s 1 s, never boxes taller than max_box_height 0.5 of the frame) takes over its
    # smoothing, open episode and cooldown. Off because a wrong match names or silences the wrong worker; repeat
    # alerts are grouped by the duplicate hint instead (state.py).
    reid: dict = field(default_factory=dict)
    # {LinkConfig field: value}: the duplicate hint's track links (reid.py): a (re)appearing track continues a lost
    # one last seen at most max_gap_s (4) ago, at most max_dist (1.2) of its last box's heights away. Grouping only.
    links: dict = field(default_factory=dict)
    # Seconds a new violation is held before it is raised, so an item starting on the same track meanwhile joins it
    # (one alert "without helmet and vest"). The clip is requested when the violation starts, not when it is raised.
    # 0 = raise at once.
    alert_hold_s: float = 1.0
    # The shared zones file. A camera with zones whose `ppe` block is on alerts only for workers standing (box
    # bottom centre) inside one, and names it; a camera with none covers the whole frame (perception/lib/feature_zones).
    # A zone's block may name the equipment checked there ({"enabled": true, "items": ["helmet"]}): a worker in it is
    # held only to those of required_items; without `items`, and on a whole-frame camera, to all of them.
    zones_path: str = "data/zones_config.txt"

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ModuleConfigError("modules.ppe: interval must be >= 0")
        if self.eval_fps <= 0:
            raise ModuleConfigError(f"modules.ppe: eval_fps must be > 0, got {self.eval_fps!r}")
        if self.min_person_height_px < 0 or self.edge_margin_px < 0:
            raise ModuleConfigError("modules.ppe: min_person_height_px and edge_margin_px must be >= 0")
        if self.alert_hold_s < 0:
            raise ModuleConfigError(f"modules.ppe: alert_hold_s must be >= 0, got {self.alert_hold_s!r}")
        self.compliance_config()          # build them once, so a bad item, field or key refuses startup
        self.temporal_config()
        self.reid_config()
        self.link_config()

    def compliance_config(self) -> ComplianceConfig:
        """The vendored ComplianceConfig with this module's settings: persons are the pgie's label, the pixel
        thresholds are in mux pixels, and `items` patches the default items by name."""
        defaults = ComplianceConfig()
        by_name = {it.name: it for it in defaults.items}
        unknown = sorted(set(self.items) - set(by_name))
        if unknown:
            raise ModuleConfigError(f"modules.ppe: items has unknown item(s) {unknown}; known: {sorted(by_name)}")
        known_fields = {f.name for f in dataclasses.fields(PPEItemConfig)} - {"name"}
        items = []
        for name, item in by_name.items():
            patch = self.items.get(name) or {}
            if not isinstance(patch, dict):
                raise ModuleConfigError(f"modules.ppe: items.{name} must be a mapping, got {type(patch).__name__}")
            bad = sorted(set(patch) - known_fields)
            if bad:
                raise ModuleConfigError(f"modules.ppe: items.{name} has unknown field(s) {bad}; "
                                        f"known: {sorted(known_fields)}")
            items.append(dataclasses.replace(item, **{"min_person_height_px": self.min_person_height_px, **patch}))
        missing = sorted(set(self.required_items) - set(by_name))
        if missing or not self.required_items:
            raise ModuleConfigError(f"modules.ppe: required_items {self.required_items!r} must name items from "
                                    f"{sorted(by_name)}")
        return dataclasses.replace(defaults, person_classes=[PERSON_LABEL], edge_margin_px=self.edge_margin_px,
                                   required_items=list(self.required_items), items=items)

    def temporal_config(self) -> TemporalConfig:
        known = {f.name for f in dataclasses.fields(TemporalConfig)}
        bad = sorted(set(self.temporal) - known)
        if bad:
            raise ModuleConfigError(f"modules.ppe: temporal has unknown field(s) {bad}; known: {sorted(known)}")
        cfg = TemporalConfig(**{"close_after_present_s": CLOSE_AFTER_PRESENT_S, **self.temporal})
        if cfg.close_after_present_s < 0:
            raise ModuleConfigError("modules.ppe: temporal.close_after_present_s must be >= 0")
        return cfg

    def reid_config(self) -> ReidConfig:
        known = {f.name for f in dataclasses.fields(ReidConfig)}
        bad = sorted(set(self.reid) - known)
        if bad:
            raise ModuleConfigError(f"modules.ppe: reid has unknown field(s) {bad}; known: {sorted(known)}")
        cfg = ReidConfig(**self.reid)
        if not isinstance(cfg.enabled, bool) or cfg.max_gap_s < 0 or cfg.max_speed < 0 or cfg.min_dist < 0 \
                or cfg.recent_s < 0 or cfg.max_height_ratio < 1 or not 0 < cfg.max_box_height <= 1:
            raise ModuleConfigError(f"modules.ppe: reid {self.reid!r}: enabled is true/false, gap, speed, distance "
                                    "and recent_s >= 0, max_height_ratio >= 1, max_box_height in (0, 1]")
        return cfg

    def link_config(self) -> LinkConfig:
        known = {f.name for f in dataclasses.fields(LinkConfig)}
        bad = sorted(set(self.links) - known)
        if bad:
            raise ModuleConfigError(f"modules.ppe: links has unknown field(s) {bad}; known: {sorted(known)}")
        cfg = LinkConfig(**self.links)
        if cfg.max_gap_s < 0 or cfg.max_dist < 0:
            raise ModuleConfigError(f"modules.ppe: links {self.links!r}: max_gap_s and max_dist must be >= 0")
        return cfg


def _crop_mode(config: str) -> tuple:
    """(True, its batch-size) when the detector config runs on objects (process-mode=2), else (False, 0)."""
    cp = configparser.ConfigParser()
    cp.read(config)
    if cp.getint("property", "process-mode", fallback=1) != 2:
        return False, 0
    return True, cp.getint("property", "batch-size", fallback=1)


def _require_model(config: str) -> None:
    """Refuse to start, saying where to get it, when the detector config or the ONNX it names is missing (a clone that
    has not run `just data-pull`); without this check nvinfer would fail later with a TensorRT build error that does
    not say which file is wrong or where it comes from."""
    cp = configparser.ConfigParser()
    if not cp.read(config) or not cp.has_option("property", "onnx-file"):
        raise ModuleConfigError(f"modules.ppe: detector config {config} is missing or names no onnx-file")
    onnx = os.path.join(os.path.dirname(os.path.abspath(config)), cp.get("property", "onnx-file"))
    if not os.path.isfile(onnx):
        raise ModuleConfigError(
            f"modules.ppe: the PPE model {onnx} is missing. It is versioned with DVC: run `just data-pull models-ppe`, "
            "or rebuild it as perception/models/DeepStream-Yolo-PPE/README.md ('Getting the model') describes, and put "
            "it at that path.")


def _missing_label(items: list) -> "str | None":
    """What a worker's box says on a clip: "no helmet", "no helmet and vest"; None when nothing is missing."""
    return f"no {join_items(items)}" if items else None


class Ppe(Module):
    name = "ppe"
    Settings = Settings

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        if settings.interval > 0:
            logger.warning("modules.ppe.interval is %d (> 0); skipped batches carry no PPE detections. Set it to 0.",
                           settings.interval)
        self._zones = LiveFeatureZones("ppe", settings.zones_path, services.redis)
        self._warned_zones: set = set()      # (camera, zone id) of zones whose items name none of required_items

    def gies(self) -> list:
        s = self.settings
        # role="secondary" PLACES the detector after the tracker; what it runs on is its config's process-mode (the
        # engine never reads or overrides it). The default config says process-mode=2: each tracked person's crop.
        # nvinfer maps the boxes back to frame coordinates and makes the person their parent (the link does not reach
        # the module on this pipeline; the rules match boxes to persons by position). It sees a variable number of
        # crops per batch, so its batch size and engine are its config's. A process-mode=1 config runs on the full
        # frame instead, with the mux's batch (batch_from_mux), like a primary detector.
        # Not a "primary" module GIE: that sits before the tracker, and nvtracker deletes every object it does not
        # track: all boxes below its minDetectorConfidence (0.19) and new targets in probation or refused as
        # duplicates, ~19% of the confident PPE boxes on test.mp4 (docs/PPE_MODULE_PLAN.md sections 10-11). The
        # compliance logic needs every PPE box down to the detector's own 0.10 floor, so they must not be tracked.
        _require_model(s.config)
        on_crops, crop_batch = _crop_mode(s.config)
        batch = crop_batch if on_crops else self.services.batch_size
        return [GieSpec(name="pgie_ppe", config_path=s.config, role="secondary", interval=s.interval,
                        engine_path=gie_engine_path(s.config, batch), batch_from_mux=not on_crops)]

    def on_built(self, pipeline) -> None:
        s = self.settings
        compliance, temporal = s.compliance_config(), s.temporal_config()
        for sid in self.services.stream_ids:
            state = self.services.state(self.name, sid)
            state["rules"] = StreamRules(compliance, temporal, s.eval_fps, s.reid_config(), s.link_config())
            state["zone_by_track"] = {}
        self._zones.start()

    def control(self) -> dict:
        return self._zones.control()

    def _scope(self, stream_id: str, person) -> "tuple[bool, str, tuple]":
        """Where this worker stands (box bottom centre): whether that is in PPE scope, the zone ("" on a whole-frame
        camera; the first, where zones overlap), and the items checked there -- of required_items, every one any of
        the zones holding them checks (a zone without `items` checks them all)."""
        x1, _y1, x2, y2 = person.bbox
        covered, holding = self._zones.zones_at(stream_id, (x1 + x2) / 2.0, y2)
        required = self.settings.required_items
        if not covered:
            return False, "", ()
        if not holding:
            return True, "", tuple(required)
        for zone in holding:
            if zone.items is not None and not set(zone.items) & set(required) \
                    and (stream_id, zone.zone_id) not in self._warned_zones:
                self._warned_zones.add((stream_id, zone.zone_id))
                logger.warning("[%s] PPE zone %s checks none of %s (its items: %s): nothing alerts there",
                               stream_id, zone.zone_id, ", ".join(required), ", ".join(zone.items) or "none")
        return True, holding[0].zone_id, tuple(
            i for i in required if any(zone.items is None or i in zone.items for zone in holding))

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        """Every frame, ahead of the publish gate: the rules gate themselves to eval_fps. An evaluated frame leaves
        its smoothed persons (compliance/temporal.py SmoothedPerson) in frame.scratch["ppe"] for the later hooks.
        A new violation is held for alert_hold_s and raised on the first frame after (any frame, evaluated or not)."""
        state = self.services.state(self.name, frame.stream_id)
        rules = state["rules"]
        smoothed = rules.evaluate(frame.objects, frame.mux_size, frame.now)
        if smoothed is not None:
            frame.scratch[self.name] = smoothed
            # what each worker is held to where they stand now; kept for the highlight between evaluated frames
            checked = state["checked_by_track"] = {}
            missing = {}
            for sp in smoothed:
                tid = sp.result.track_id
                covered, zone, checked[tid] = self._scope(frame.stream_id, sp.result.person)
                missing[tid] = [i for i in missing_items(sp, rules.compliance.required_items) if i in checked[tid]]
                pending = self._in_scope_violations(state, sp, frame.now, covered, zone, checked[tid])
                if not pending:
                    continue
                if rules.hold_alert(frame.now, frame.pts, tid, pending, missing[tid]):
                    # Like fire_alerts: the alert's clip comes from a smart-record session started here, when the
                    # violation starts (coalesced with one already running), not when the held alert is raised.
                    # One request per violation episode; the episode stays open without repeat alerts.
                    self.services.request_clip(frame.stream_id, frame.now)
            self._forget_gone_tracks(state, {sp.result.track_id for sp in smoothed}, frame.now)
            # the workers' boxes for the alerts' clips: the clip draws its alert's worker (app/overlay_track.py)
            self.overlay_boxes.write(frame.stream_id, frame.pts, [
                corners_box(sp.result.person.bbox, "alert", sp.result.track_id,
                            _missing_label(missing[sp.result.track_id]))
                for sp in smoothed if sp.result.track_id != UNTRACKED], at=frame.arrived)
        for held in rules.due_alerts(frame.now, self.settings.alert_hold_s):
            self._alert(frame.stream_id, rules, held)

    @staticmethod
    def _in_scope_violations(state: dict, sp, now: float, covered: bool, zone: str, checked: tuple) -> list:
        """The items this worker's alert should name now: violations starting on this frame, plus ones that started
        where they were not checked (outside every PPE zone, or in a zone that does not check that item) and are
        still missing now that the worker stands where they are. The rules flag a missing item only on the frame it
        starts, so an unchecked one is kept per track (`deferred`) until the worker stands where it is checked
        (alert), the item is seen again, or the track is gone DEFER_FORGET_S."""
        deferred = state.setdefault("deferred", {})
        tid = sp.result.track_id
        kept = deferred.get(tid, {}).get("items", ())
        pending = list(sp.new_violations) + [item for item in kept if item not in sp.new_violations
                                             and sp.item_status.get(item) is ItemStatus.ABSENT]
        due = [item for item in pending if covered and item in checked]
        later = {item for item in pending if item not in due}
        if later:
            deferred[tid] = {"items": later, "seen": now}            # not checked here: not a violation, yet
        else:
            deferred.pop(tid, None)
        if due:
            state.setdefault("zone_by_track", {})[tid] = zone
        return due

    @staticmethod
    def _forget_gone_tracks(state: dict, present: set, now: float) -> None:
        """Bounded memory: a deferred violation is dropped once its track has been unseen DEFER_FORGET_S; the zone map
        is emptied as each held alert is raised, and pruned to the tracks in view if it ever grows past 256."""
        deferred = state.get("deferred") or {}
        for tid, entry in list(deferred.items()):
            if tid in present:
                entry["seen"] = now
            elif now - entry["seen"] > DEFER_FORGET_S:
                deferred.pop(tid)
        zones = state.get("zone_by_track") or {}
        if len(zones) > 256:
            for tid in [t for t in zones if t not in present]:
                zones.pop(tid)

    def stop(self) -> None:
        """Shutdown: raise the alerts still held, rather than lose them."""
        for sid in self.services.stream_ids:
            rules = self.services.state(self.name, sid).get("rules")
            if rules is not None:
                for held in rules.due_alerts(float("inf"), 0.0):
                    self._alert(sid, rules, held)

    def _alert(self, stream_id: str, rules: StreamRules, held: dict) -> None:
        """One alert per held violation (StreamRules.hold_alert), naming every item that started while it was held.
        The app reads only kind/severity/category/message/zone/count/pts, so what the reviewer must see (who, what is
        missing, and that it may repeat an earlier alert) is in `message`; track_id, new, missing and the
        possible_duplicate_* fields stay on the stream entry for other consumers. track_id is always the tracker's
        current id, the box seen; pts is the frame the violation started on."""
        tid, new, missing = held["track_id"], held["new"], held["missing"]
        message = f"Person #{tid} without {join_items(missing)} on {stream_id}"
        dup = rules.note_alert(held["now"], held["pts"], tid, new)
        extra = {}
        if dup:
            # Grouping, not suppression: the alert is raised; the reviewer is told which earlier one it may repeat.
            message += f" (possibly #{dup['possible_duplicate_of']} again, alerted {dup['seconds_since']:.0f} s earlier)"
            extra = {"possible_duplicate_of": dup["possible_duplicate_of"],
                     "possible_duplicate_of_pts": f"{dup['pts']:.3f}", "possible_duplicate_gap_s": dup.get("gap_s"),
                     "possible_duplicate_distance": dup.get("distance_heights")}
        zone = self.services.state(self.name, stream_id).get("zone_by_track", {}).pop(tid, "")
        self.services.alert(stream_id, "ppe_violation", message=message, severity="warning",
                            category="safety", pts=held["pts"], zone=zone, count=len(missing), track_id=tid,
                            new=",".join(new), missing=",".join(missing), **extra)
        logger.info("[%s] PPE violation: %s (new: %s)", stream_id, message, ", ".join(new))

    def transform(self, frame: FrameView) -> None:
        """PPE boxes are drawn but never published: they are evidence on the overlay, not detections of their own
        (the engine hides them first, since they are not in relevant_classes). Drawn above the person boxes."""
        for o in frame.objects:
            if o.gie_name == PPE_GIE and o.label in PPE_LABELS:
                o.kind, o.draw, o.publish, o.layer = PPE_KIND, o.detected, False, 2

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        """Highlight every person in violation now (from the last evaluated frame, so the highlight holds on skipped
        frames and clears once the smoothed item is no longer ABSENT), and caption the PPE boxes without the
        untracked id the overlay would otherwise print."""
        state = self.services.state(self.name, frame.stream_id)
        checked = state.get("checked_by_track") or {}
        violators = {tid: shown for tid, items in state["rules"].violators.items()
                     if (shown := [i for i in items if i in checked.get(tid, items)])}
        for o in frame.objects:
            if o.kind == PPE_KIND and o.draw:
                out.annotate(o, color=BOX_PPE_ITEM, text=f"{o.label} {o.conf:.2f}", priority=_ITEM_PRIORITY)
            elif o.gie_name == PERSON_GIE and o.label == PERSON_LABEL and o.id in violators:
                out.annotate(o, color=BOX_PPE_VIOLATION, text=f"#{o.id} no {join_items(violators[o.id])}",
                             priority=_VIOLATION_PRIORITY)


MODULE = Ppe
