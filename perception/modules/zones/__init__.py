"""zones: restricted/valid regions drawn in the UI, with intrusion and overcrowding alerts, and filter regions.

Three things share one set of per-camera polygons, so they are one module:

- intrusion alerts: an nvdsanalytics element flags which ROIs each tracked object is inside; one alert per object
  entering a restricted zone (or leaving a valid one), tolerant of tracker re-IDs (ZoneReentryGrace)
- overcrowding alerts: the count of chosen classes inside a zone crossing a threshold, with a debounced release
- filter zones: detections whose centre is outside every filter polygon of a camera are dropped before anything
  downstream (tracking, MTMC, the preview) sees them

The app writes the zone files and announces a new generation on iris:ctrl:zones_version. Every zones process sees
every bump, re-renders its own nvdsanalytics config from its own streammux slots, and reloads the element only if that
render changed; it rebuilds a stream's zone state only if that stream's zones changed, so an edit on one camera leaves
the alert state of the others alone. It then acknowledges the version on iris:analytics:zones_applied and, per camera
it runs, on iris:analytics:zones_applied:<camera>, so the UI can show "applying" vs "active" for the camera it saved.
"""
import logging
import os
import threading
from dataclasses import dataclass, field

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.modules import ElementSpec, Module, ModuleConfigError
from perception.engine.nvds import analytics_roi

from .analytics import read_zones, render_analytics_config, roi_lines, slot_map, worker_config_path, write_worker_config
from .colors import box_color
from .config import load_stream_detection_configs, load_stream_filter_zones, point_in_polygon
from .dedup import ZoneReentryGrace
from .runtime import ZoneRuntime
from perception.lib.overcrowding_dedup import OvercrowdingLatch
from perception.lib.overlay_boxes import box
from perception.lib.feature_zones import ZONES_VERSION_KEY
from perception.lib.versioned import VersionedConfig

_HAZARD_CLASSES_DEFAULT = ("fire", "smoke")


def hazard_overlap_warnings(zones_by_stream: dict, hazard_classes) -> list:
    """One warning per (stream, zone) whose enabled intrusion classes overlap the full-frame hazard classes.
    Guidance only, not enforcement: `fire`/`smoke` are legitimate zone intrusion classes, but where a hazard
    detector alerts on them too, a single fire raises both a zone_intrusion event and a fire_smoke event with no
    dedup between them. Zones hot-reload, so this is meant to be called once at boot."""
    warnings = []
    hazard_classes = set(hazard_classes or ())
    if not hazard_classes:
        return warnings
    for stream_id, zones in zones_by_stream.items():
        for zone in zones:
            if not zone.intrusion_enabled:
                continue
            overlap = zone.intrusion_classes & hazard_classes
            if overlap:
                warnings.append(
                    f"stream {stream_id!r} zone {(zone.name or zone.label)!r} has intrusion class(es) "
                    f"{sorted(overlap)} overlapping modules.fire_alerts.classes; a single fire/smoke event may "
                    f"raise both a zone_intrusion alert and a fire_smoke alert, with no dedup between them.")
    return warnings

logger = logging.getLogger("iris.sidecar")

VERSION_KEY = ZONES_VERSION_KEY                     # perception/lib/feature_zones.py: the one name of it
APPLIED_KEY = "iris:analytics:zones_applied"
_ZONE_COLOR_PRIORITY = 1            # below a hazard colour (fire), above the engine's default green


@dataclass
class Settings:
    # Where the app writes the per-zone detection settings; the filter zones file is found beside it by name.
    config_path: str = "data/zones_config.txt"
    # The nvdsanalytics ROI file the app renders; lives under data/ so the app container's shared ./data mount
    # carries the write out to the host without a new bind mount.
    analytics_config_path: str = "data/nvdsanalytics_config.txt"
    # Object classes that count as an intrusion when inside a zone; one alert per tracked object entering, not per
    # frame it stays. Per zone these are set in the UI; this is the default for a zone that does not say.
    alert_classes: list = field(default_factory=lambda: ["person"])
    # Re-ID grace: a new tracker id appearing in-zone near a spot where an already-alerted id vanished less than
    # this many seconds ago inherits its alerted status instead of re-firing the alert.
    grace_seconds: float = 3.0
    # Spatial match radius for that grace, as a multiple of the larger of the two bbox diagonals.
    match_radius: float = 0.5
    # Read by the app (whether it listens for overcrowding alerts); the zones themselves say per zone whether
    # overcrowding is on.
    overcrowding_enabled: bool = True
    overcrowding_threshold: int = 5                    # default object-count threshold for a zone that has none
    overcrowding_classes: list = field(default_factory=lambda: ["person"])
    # The count must stay below the threshold this long, continuously, before a new overcrowding alert can fire
    # (debounces flapping when the count hovers right at the threshold).
    overcrowding_release_seconds: float = 5.0
    # Say when an incident ENDS, on the same streams (the safety product turns intrusion and overcrowding into
    # incidents with an end): `kind: overcrowding_resolved` when a zone's overcrowding latch releases, and
    # `kind: intrusion_cleared` once a zone that alerted has had nobody violating it for intrusion_clear_seconds.
    # Off by default: an app that does not know these kinds would read them as new alerts.
    end_markers: bool = False
    intrusion_clear_seconds: float = 5.0
    # Diagnostic, off by default: on every analyzed frame of every stream (zoned or not), write which nvdsanalytics
    # ROI labels its objects are inside to `iris:zones:roi_probe:<camera>` (fields: pts, at = arrival wall time,
    # objects, labels). For validating live reloads (tools/zones/validate_reload.py); never on in a product.
    roi_probe: bool = False

    def __post_init__(self) -> None:
        if not self.intrusion_clear_seconds > 0:
            raise ModuleConfigError(
                f"modules.zones: intrusion_clear_seconds must be > 0, got {self.intrusion_clear_seconds!r}")


class Zones(Module):
    name = "zones"
    Settings = Settings

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self._analytics = None                          # the nvdsanalytics element, set in on_built
        self._versions = VersionedConfig(services.redis, VERSION_KEY, APPLIED_KEY)
        # {streammux slot: camera id}: the engine's pad map (sources in pad order when it has none); #173 moves it
        # via slots_changed
        self._slots = dict(services.index_to_id) if services.index_to_id else slot_map(services.sources)
        self._config_text = None                        # the ROI config the element last loaded; None: unknown
        self._built_from = {}                           # stream id -> what its zone state was built from
        # slots_changed (the engine's attach/detach thread) and a zones version bump (the control thread) both
        # re-render the ROI config, reload the element and rewrite stream state; this serialises them so a slot move
        # and a save cannot interleave into a mixed render or a half-reset stream.
        self._lock = threading.Lock()

    # build ------------------------------------------------------------------------------------------------
    def elements(self) -> list:
        path, self._config_text = self._write_analytics_config(self._render())
        logger.info("nvdsanalytics enabled, config=%s classes=%s", path, sorted(self.settings.alert_classes))
        return [ElementSpec(name="analytics", factory="nvdsanalytics", properties={"config-file": path})]

    def on_built(self, pipeline) -> None:
        self._analytics = pipeline.get("analytics")
        # A zone saved between elements() rendering the config the pipeline was built from and here must be applied,
        # not just acked: re-render and reload the element if that window changed the ROIs. An unchanged render
        # reloads nothing (no extra set_property).
        self._reload_analytics()
        self._load()
        fire = (self.services.config.get("modules") or {}).get("fire_alerts")
        if fire is not None:
            classes = fire.get("classes", _HAZARD_CLASSES_DEFAULT) if isinstance(fire, dict) else _HAZARD_CLASSES_DEFAULT
            for msg in hazard_overlap_warnings(
                    {sid: self.services.state(self.name, sid)["zones"] for sid in self.services.stream_ids}, classes):
                logger.warning("%s", msg)
        # The pipeline was just built from the current config file, so whatever version exists now is already
        # applied. This also self-heals zones saved while the sidecar was down.
        self._versions.acknowledge()
        self._acknowledge_cameras(self._versions.version())

    def control(self) -> dict:
        return self._versions.control(self._on_zones_version)

    def slots_changed(self, slots: dict) -> None:
        """The hook for attach/detach (#173): `slots` is the new {streammux slot: camera id}. Re-renders and reloads
        the ROI config if it changed, ends a detached camera's open incidents and drops its zones, and gives every
        camera that gained, lost or changed its slot fresh zone state (a reused slot must not inherit the previous
        camera's). Cameras that kept their slot keep their state."""
        with self._lock:
            old = self._slots
            moved = ({cam for slot, cam in slots.items() if old.get(slot) != cam}
                     | {cam for slot, cam in old.items() if slots.get(slot) != cam})
            self._slots = dict(slots)
            self._reload_analytics()
            for sid in moved:
                self._built_from.pop(sid, None)
                if sid not in self._slots.values():
                    state = self.services.state(self.name, sid)
                    if self.settings.end_markers:
                        self._end_open_incidents(sid, state.get("zones") or [])
                    state["zones"], state["filters"] = [], []
            self._load()
            logger.info("streammux slots changed (%s); zone state reset for %s", self._slots, sorted(moved))

    # nvdsanalytics config ---------------------------------------------------------------------------------
    def _render(self) -> str:
        """This process's nvdsanalytics config, from the zones file and its own slot map: [roi-filtering-stream-N] is
        numbered by this process's streammux slots, so no global index is involved."""
        return render_analytics_config(read_zones(self.settings.config_path), self._slots, *self.services.mux_size)

    def _write_analytics_config(self, text: str) -> "tuple[str, str | None]":
        """Write this process's render to disk; return (path it landed at, the config now applied for this process).
        The second value is `text` when the write reached this process's own file -- what the element loads -- or
        None after a fallback, where the applied config is unknown so the next bump reloads whatever it renders. The
        caller records it as _config_text; a live reload does so only after set_property succeeds, so a failed
        reload leaves the applied config as it was and is retried, not acknowledged."""
        s, svc = self.settings, self.services
        path = write_worker_config(s.analytics_config_path, text, svc.role, *svc.mux_size, shard_id=svc.shard_id)
        own = worker_config_path(s.analytics_config_path, svc.role, svc.shard_id)
        return path, (text if path == own else None)

    def _reload_analytics(self) -> bool:
        """Re-render, and reload the element only if this process's ROIs changed. True when it reloaded.

        The applied config (_config_text) is recorded only after set_property returns: if the reload raises, the
        applied config is left untouched, so _on_zones_version acks nothing and the next poll re-renders and retries.
        Recording it before set_property would make the retry see nothing to do and ack a config the element never
        loaded."""
        text = self._render()
        if self._analytics is None or text == self._config_text:
            return False
        # Re-setting the config-file property makes the plugin reload all configurations live -- no pipeline state
        # change, other elements/streams unaffected (forums.developer.nvidia.com/t/270161).
        path, applied = self._write_analytics_config(text)
        self._analytics.set_property("config-file", path)
        self._config_text = applied
        return True

    def _acknowledge_cameras(self, version: int) -> None:
        """Per camera this process runs, the version its zones are applied at: the UI waits on the camera it saved,
        so another process's ack cannot announce this one's camera as active."""
        for sid in self._slots.values():
            self.services.redis.set(f"{APPLIED_KEY}:{sid}", version)

    # state ------------------------------------------------------------------------------------------------
    def _load(self) -> list:
        """Rebuild the zone list of each stream whose zones changed since it was last built, from zones_config.txt;
        return those stream ids. Streams without saved zones get an empty list. Called at startup and on every zones
        version bump (hot-reload). A rebuilt stream's zones get fresh dedup state, so re-drawing geometry re-alerts
        under the new zones; a stream whose zones (settings and ROI geometry) are unchanged keeps its state, so an edit
        on another camera does not re-alert or end incidents here. Also loads filter-zone polygons (stateless).

        Zone state follows this process's streammux slots, not stream_ids: a camera this process no longer owns (a
        detach whose pad lingers in stream_ids) keeps no zones and no filters, matching its absent ROI section, even
        though its saved zones are still in the file."""
        s = self.settings
        configs = load_stream_detection_configs(
            s.config_path, set(s.alert_classes), set(s.overcrowding_classes), s.overcrowding_threshold)
        stored = read_zones(s.config_path)
        # Filter-zone polygons (0-1 mux coords) per stream
        filter_path = s.config_path.replace("zones_config", "filter_zones")
        if not filter_path.endswith(".json"):
            filter_path = os.path.splitext(filter_path)[0] + ".json"
        mux_w, mux_h = self.services.mux_size
        filter_cfgs = load_stream_filter_zones(filter_path, mux_w, mux_h)
        if filter_cfgs:
            logger.info("Loaded filter zones for streams: %s", list(filter_cfgs.keys()))
        owned = set(self._slots.values())
        rebuilt = []
        for sid in self.services.stream_ids:
            state = self.services.state(self.name, sid)
            if sid not in owned:
                # Not one of this process's streams any more: no zones, no filters (its ROI section is gone too).
                state["filters"] = []
                if state.get("zones"):
                    if s.end_markers:
                        self._end_open_incidents(sid, state["zones"])
                    state["zones"] = []
                self._built_from.pop(sid, None)
                continue
            state["filters"] = filter_cfgs.get(sid, [])
            built_from = (configs.get(sid, []), roi_lines(stored.get(sid)))
            if "zones" in state and self._built_from.get(sid) == built_from:
                continue
            self._built_from[sid] = built_from
            rebuilt.append(sid)
            if s.end_markers:
                self._end_open_incidents(sid, state.get("zones") or [])
            # Rebinding the whole list is an atomic swap vs. the probe thread, and someone inside a re-drawn zone
            # should re-alert under the new geometry anyway.
            state["zones"] = [
                ZoneRuntime(
                    label=z["label"], name=z["name"], mode=z["mode"],
                    intrusion_classes=z["intrusion_classes"],
                    overcrowding_enabled=z["overcrowding_enabled"],
                    overcrowding_threshold=z["overcrowding_threshold"],
                    overcrowding_classes=z["overcrowding_classes"],
                    _zone_grace=ZoneReentryGrace(s.grace_seconds, s.match_radius),
                    _overcrowd_latch=OvercrowdingLatch(s.overcrowding_release_seconds),
                )
                for z in configs.get(sid, [])
            ]
        return rebuilt

    def _on_zones_version(self, version: int) -> None:
        """The app announced a new generation (zone saved or deleted in the UI). Raising here means no ack and a
        retry on the next tick; a persistent failure keeps applied < version, which is what drives the UI warning."""
        with self._lock:
            reloaded = self._reload_analytics()
            rebuilt = self._load()
            self._acknowledge_cameras(version)
        logger.info("zones version %d: analytics config %s; zone state rebuilt for %s", version,
                    "reloaded" if reloaded else "unchanged", rebuilt or "no stream")

    # hot path ---------------------------------------------------------------------------------------------
    def extract(self, frame: FrameView):
        """Which analytics regions each object is inside, as attrs["roi"] (a walk per object, so only on analyzed
        frames of a camera that has zones)."""
        if not (frame.analyze and (self.settings.roi_probe or self.services.state(self.name, frame.stream_id).get("zones"))):
            return None
        return lambda o: o.attrs.__setitem__("roi", analytics_roi(o.raw))

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        """Alerting: runs on every frame, ahead of the engine's publish gating."""
        st = self.services.state(self.name, frame.stream_id)
        if self.settings.roi_probe and frame.analyze:
            self._probe(frame)
        zones = st.get("zones") or []
        if frame.analyze and zones:
            # Raw per-frame set of object_ids currently violating ANY of this stream's intrusion zones (inside a
            # restricted ROI / outside a valid one) -- drives live box coloring. Distinct from the deduped alert
            # sets (which fire once per entry); coloring must reflect current status every frame.
            violating_ids: set = set()
            any_intrusion = False
            intrusion_union: set = set()
            emit, now = self.services.emit, frame.now
            frame_objs = [(o.id, o.label, o.bbox, o.attrs.get("roi", frozenset())) for o in frame.objects]
            subjects: dict = {}             # oid -> its box, for the alerts' clips (app/overlay_track.py)
            for zone in zones:
                if zone.mode != "off":
                    any_intrusion = True
                    intrusion_union |= zone.intrusion_classes
                    # Non-inverted membership: entered iff the zone's label is in roi_labels.
                    # restricted alerts on entry (inside), valid alerts on exit (outside).
                    violators = [
                        (oid, label, bbox) for oid, label, bbox, roi_labels in frame_objs
                        if label in zone.intrusion_classes
                        and ((zone.label in roi_labels) == (zone.mode == "restricted"))
                    ]
                    # ZoneReentryGrace.observe's documented input is [(object_id, bbox)] -- keep that contract
                    # intact and carry the class label alongside, so the alert can name what violated ("Car #4")
                    # instead of assuming a person.
                    in_zone = [(oid, bbox) for oid, _label, bbox in violators]
                    for oid, _label, bbox in violators:
                        subjects.setdefault(oid, box(bbox, "alert", oid, zone.name))
                    labels = {oid: label for oid, label, _bbox in violators}
                    violating_ids.update(oid for oid, _bbox in in_zone)
                    # One alert per tracked object entering the zone, not per frame it stays. The per-zone
                    # ZoneReentryGrace also swallows tracker re-IDs (a new object_id near where an alerted one
                    # just vanished inherits its alerted status).
                    for oid in zone._zone_grace.observe(now, in_zone):
                        if emit(f"iris:zone_alert:{self.services.key_id(frame.stream_id)}",
                                {"frame_num": frame.frame_num, "pts": f"{frame.pts:.3f}",
                                 "object_id": oid, "object_label": labels.get(oid, ""), "zone": zone.label}):
                            logger.info("[%s] zone intrusion detected (zone=%s object_id=%s label=%s)",
                                        frame.stream_id, zone.label, oid, labels.get(oid, ""))
                        self.services.request_clip(frame.stream_id, now)
                        if zone._intrusion_open_at is None:
                            zone._intrusion_open_at = now
                    if self.settings.end_markers:
                        self._intrusion_end(frame, zone, bool(in_zone))
                if zone.overcrowding_enabled:
                    counted = [(oid, bbox) for oid, label, bbox, roi_labels in frame_objs
                               if label in zone.overcrowding_classes and zone.label in roi_labels]
                    count = len(counted)
                    for oid, bbox in counted:
                        subjects.setdefault(oid, box(bbox, "alert", oid, zone.name))
                    latch = zone._overcrowd_latch
                    was_armed = latch.armed
                    if not latch.observe(now, count, zone.overcrowding_threshold):
                        if self.settings.end_markers and not was_armed and latch.armed:
                            self._overcrowding_end(frame, zone, count)     # the latch just released: it ended
                    else:
                        zone._overcrowd_open_at = latch.run_started_at if latch.run_started_at is not None else now
                        if emit(f"iris:overcrowd_alert:{self.services.key_id(frame.stream_id)}",
                                {"frame_num": frame.frame_num, "pts": f"{frame.pts:.3f}",
                                 "count": count, "zone": zone.label}):
                            logger.info("[%s] overcrowding detected (zone=%s count=%d threshold=%d)",
                                        frame.stream_id, zone.label, count, zone.overcrowding_threshold)
                        # Same source-level trigger as intrusion: an overcrowding alert during a running session
                        # coalesces onto its clip.
                        self.services.request_clip(frame.stream_id, now)
            self.overlay_boxes.write(frame.stream_id, frame.pts, list(subjects.values()), at=frame.arrived)
            # Cache the last analyzed violation state so skipped (non-inferred) frames keep the same box coloring
            # between inference cycles.
            st["last"] = (violating_ids, any_intrusion, intrusion_union)
        elif frame.analyze:
            st["last"] = (set(), False, set())      # no zones any more: nothing to colour

    def _probe(self, frame: FrameView) -> None:
        """roi_probe: the ROI labels nvdsanalytics put on this frame's objects, whether or not Python has zones for the
        stream -- so a section the element still applies after it was removed from the file shows up here."""
        labels = sorted(set().union(*(o.attrs.get("roi", frozenset()) for o in frame.objects)))
        self.services.emit(f"iris:zones:roi_probe:{self.services.key_id(frame.stream_id)}",
                           {"pts": f"{frame.pts:.3f}", "at": f"{frame.arrived:.3f}",
                            "objects": len(frame.objects), "labels": ",".join(labels)}, maxlen=512)

    def _end_open_incidents(self, sid: str, zones: list) -> None:
        """A rebuild replaces a stream's zones with fresh state, so the incidents still open under the old ones end here
        (reason zones_reloaded) -- else the app would keep them ongoing and add the next entry, hours later, to them.
        Someone still inside a zone opens a new incident under the new geometry."""
        for zone in zones:
            ends = []
            if zone._intrusion_open_at is not None:
                ends.append(("iris:zone_alert", "intrusion_cleared"))
            if zone._overcrowd_open_at is not None:
                ends.append(("iris:overcrowd_alert", "overcrowding_resolved"))
            for stream, kind in ends:
                self.services.emit(f"{stream}:{self.services.key_id(sid)}",
                                   {"kind": kind, "pts": "", "zone": zone.label, "duration": "",
                                    "reason": "zones_reloaded"})
                logger.info("[%s] zones reloaded: %s in zone %s ended", sid, kind, zone.label)

    def _intrusion_end(self, frame: FrameView, zone: ZoneRuntime, occupied: bool) -> None:
        """A zone that alerted ends its intrusion once nobody has violated it for intrusion_clear_seconds."""
        if zone._intrusion_open_at is None:
            return
        if occupied:
            zone._intrusion_clear_since = None
            return
        if zone._intrusion_clear_since is None:
            zone._intrusion_clear_since = frame.now
            return
        if frame.now - zone._intrusion_clear_since < self.settings.intrusion_clear_seconds:
            return
        duration = zone._intrusion_clear_since - zone._intrusion_open_at
        self.services.emit(f"iris:zone_alert:{self.services.key_id(frame.stream_id)}",
                           {"kind": "intrusion_cleared", "pts": f"{frame.pts:.3f}", "zone": zone.label,
                            "duration": f"{duration:.3f}"})
        logger.info("[%s] zone intrusion cleared (zone=%s, %.1fs)", frame.stream_id, zone.label, duration)
        zone._intrusion_open_at = zone._intrusion_clear_since = None

    def _overcrowding_end(self, frame: FrameView, zone: ZoneRuntime, count: int) -> None:
        """The zone's overcrowding latch released (the count stayed below the threshold for the release time)."""
        ended = zone._overcrowd_latch.run_ended_at
        ended = frame.now if ended is None else ended
        opened = ended if zone._overcrowd_open_at is None else zone._overcrowd_open_at
        self.services.emit(f"iris:overcrowd_alert:{self.services.key_id(frame.stream_id)}",
                           {"kind": "overcrowding_resolved", "pts": f"{frame.pts:.3f}", "zone": zone.label,
                            "count": count, "duration": f"{max(0.0, ended - opened):.3f}"})
        logger.info("[%s] overcrowding resolved (zone=%s, %.1fs)", frame.stream_id, zone.label, ended - opened)
        zone._overcrowd_open_at = None

    def transform(self, frame: FrameView) -> None:
        """Filter zones: drop detections whose centre falls outside ALL of the camera's filter polygons. Runs before
        anything that follows (plate matching, MTMC) so their results match the surviving detections."""
        polygons = self.services.state(self.name, frame.stream_id).get("filters")
        if not polygons:
            return
        mux_w, mux_h = frame.mux_size
        published = sum(1 for o in frame.objects if o.publish)
        for o in frame.objects:
            if (o.draw or o.publish) and not any(
                    point_in_polygon((o.left + o.width / 2.0) / mux_w, (o.top + o.height / 2.0) / mux_h, poly)
                    for poly in polygons):
                o.draw = o.publish = False
        kept = sum(1 for o in frame.objects if o.publish)
        if kept < published:
            logger.info("[%s] filter zones: %d→%d detections", frame.stream_id, published, kept)

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        """Colour every drawn box by this frame's zone status: violating red, other tracked classes green, the
        rest gray. Only while some zone has intrusion detection on; otherwise the engine's default applies."""
        violating_ids, any_intrusion, intrusion_union = self.services.state(self.name, frame.stream_id).get(
            "last", (set(), False, set()))
        if not any_intrusion:
            return
        for o in frame.objects:
            if o.draw:
                out.annotate(o, color=box_color(o.id, o.label, violating_ids, True, intrusion_union),
                             priority=_ZONE_COLOR_PRIORITY)


MODULE = Zones
