"""proximity: a person too close to a moving vehicle inside a zone -> one safety alert per episode.

Model-free: it reads the primary detector's persons and vehicles with their instance masks (a YOLO-seg model, see
products/proximity/sidecar.yaml) and raises `services.alert(kind="proximity", category="safety")`. The rule is
camera.py; the settings, the zone block and the alert fields are in README.md; the design and the measurements behind
every number are in RESEARCH_pedestrian-vehicle-proximity_2026-10-26.md (branch analysis/ped-vehicle-proximity).

Per frame the detector ran on (nvinfer's own flag: the engine's `analyze` counter can disagree with nvinfer's
`interval`, and objects on the other frames are tracker predictions), in `on_frame`: for each zone with
proximity on, a person whose foot (recovered when hidden) is inside, a moving vehicle whose ground reference point is
inside, not its rider, and the foot within max_gap person heights of the lowest part of its mask for persist_s -> an
alert, once per episode (once more if the episode turns more severe). The depth check (per zone) measures those pairs'
frames in a worker thread. The overlay draws each close pair's distance line and tags parked vehicles.

Zone settings live in the app's zones file (a "proximity" block per zone) and reload when the app bumps
iris:ctrl:proximity_version; the module acknowledges on iris:analytics:proximity_applied. Each camera's person-scale
map is kept in Redis (iris:proximity:scale:<camera>) across restarts.

attrs: reads and fills `attrs["mask"]` (the instance mask, as person_masks does) for the vehicles of a camera with
proximity zones, on inferred frames, and for its persons only while a zone holds a vehicle that is moving or not yet
known to be parked (the costly part: until then a person's box is enough for the scale map and the foot memory).
A vehicle always gets its mask, so its reference point keeps coming from the same place.
"""
import json
import logging
import math
from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import Module, ModuleConfigError
from perception.engine.nvds import infer_done, instance_mask
from perception.lib.versioned import VersionedConfig

from .camera import Person, ProximityCamera, Vehicle
from .depth import DepthModel, DepthNotOnGpu, DepthWorker, depth_crop, ensure_on_gpu
from .rules import HIDDEN_FEET, load_zone_rules
from .scale import PersonScale
from .steady import SteadyBoxes
from .shapes import (blob_of, outline_polygon, person_shape, person_shape_from_box, vehicle_shape,
                     vehicle_shape_from_box)
from .zone_defaults import DEFAULTS as ZONE_DEFAULTS

logger = logging.getLogger("iris.sidecar")

VERSION_KEY = "iris:ctrl:proximity_version"
APPLIED_KEY = "iris:analytics:proximity_applied"
SCALE_KEY = "iris:proximity:scale:{}"
PREVIEW_KEY = "iris:ctrl:proximity_preview"   # the live preview's layers, set from the UI: README.md
DEPTH_MAP_HOLD_S = 2.0        # s: the preview shows a depth map at most this old
OVERLAY_HOLD_S = 0.5          # the overlay keeps the last analysed frame's lines this long (frames between inferences)
DEPTH_FRAME_SLACK_S = 0.25    # a frame's pixels serve a depth request made at most this long before it
DRIFT_WARN = 0.25             # recent people this much taller/shorter than the map: the camera may have moved
DRIFT_WARN_EVERY_S = 600.0    # ... said at most this often per camera
SCALE_SAVE_EVERY_S = 60.0     # a refitted map is saved at most this often per camera (and once more on stop)
NO_PTS = (1 << 64) - 1        # GST_CLOCK_TIME_NONE: a buffer without a timestamp
NO_PTS_WARN_EVERY_S = 60.0
LINE_LATCHED = (0, 0, 255)    # BGR red: close for persist_s
LINE_CLOSE = (0, 165, 255)    # orange: close, not yet for persist_s
LINE_FAR = (200, 255, 170)    # pale mint (not the people's green): further than the max gap, up to the line range
LINE_SHADOW = (25, 25, 25)    # the dark outline under every line and gap, so they read on bright ground
DEPTH_SAME = (80, 200, 80)      # BGR: the depth check says same depth (a confirmed near case)
DEPTH_APART = (60, 60, 230)     # ... says one is well behind the other: the line is dashed, no alert
DEPTH_WAIT = (170, 170, 170)    # ... has not answered yet, or did not in time
RING_R = 6                      # px on the preview: the ring at an estimated (hidden) foot
PARKED = (160, 160, 160)
# masks and boxes (app/overlay_track.py draws the clips in the same colours)
PERSON_COLOR = (80, 220, 80)                                        # green
VEHICLE_COLORS = {"moving": (230, 60, 230), "parked": PARKED, "unknown": (255, 150, 30)}   # magenta, grey, blue
MASK_ALPHA = 0.18


@dataclass
class Settings:
    # The zones file the app writes; a zone's "proximity" block turns the rule on there and overrides the defaults
    # below (README.md).
    config_path: str = "data/zones_config.txt"
    person_classes: list = field(default_factory=lambda: ["person"])
    # COCO classes; a forklift comes through as truck or car (RESEARCH 4.1).
    vehicle_classes: list = field(default_factory=lambda: list(ZONE_DEFAULTS["vehicle_classes"]))
    # The rule's defaults for a zone that does not set them: zone_defaults.json (RESEARCH 5.4), tuned in Ticket 3.
    max_gap: float = ZONE_DEFAULTS["max_gap"]                # person heights, foot to the lowest vehicle_band of its mask
    stationary_window_s: float = ZONE_DEFAULTS["stationary_window_s"]
    # person heights; a vehicle displaced less than this over the window is parked
    stationary_motion: float = ZONE_DEFAULTS["stationary_motion"]
    persist_s: float = ZONE_DEFAULTS["persist_s"]
    grace_s: float = ZONE_DEFAULTS["grace_s"]
    cooldown_s: float = ZONE_DEFAULTS["cooldown_s"]
    # hidden feet and no depth check: "unconfirmed" (alert, flagged) | "suppress"
    hidden_feet: str = ZONE_DEFAULTS["hidden_feet"]
    min_confidence: float = 0.4           # a person detected below this is not evidence
    # a pair's gap is the median of its last this many analysed frames; 1 = each frame as measured (Ticket 3: no
    # gain from smoothing on held-out footage)
    gap_smoothing_frames: int = 1
    # the drawn line's end on the vehicle follows where it lay over this many analysed frames: it no longer jumps
    # along the vehicle's edge (display only)
    line_steady_frames: int = 7
    # a person's gap to a moving vehicle runs to the nearest part of the lowest this share of its mask (0 = its lower
    # outline only; Ticket 3: 0.25 beat the outline and the whole mask on held-out footage)
    vehicle_band: float = 0.25
    # a vehicle moves only if its box moves too, by more than this share of the zone's stationary_motion: a parked
    # vehicle whose mask merges with a neighbour now and then moves its reference point, not its box (0 = off;
    # Ticket 3: removed the C189 parked-truck alerts, ~8 false alarms in 8.7 labelled camera-hours, 5 slow-truck catches)
    box_motion: float = 0.5
    # s a moving vehicle must look parked in a row before it reads parked (reading moving again is immediate): a
    # manoeuvring forklift no longer blinks parked -- and out of the rule -- between moves (Ticket 5; 0 = no hold).
    # 0.75 against 1.0 on the labelled records: 2-3 fewer false alarms each (the C189 parked-truck critical in all
    # three), one incident fewer on one record; 0.5 lost it on two, 0.25 lost real bay 4 alerts on the demo footage
    parked_hold_s: float = 0.75
    # The depth check, for zones with "depth_check": true (the default since Ticket 6). Empty: no model, those zones
    # behave as if it were off.
    depth_model: str = ""
    # relative depth gap at or under which a pair is at the same depth: the default of a zone that sets none
    depth_max_gap: float = 0.25
    depth_every_s: float = 0.5            # at most one depth frame per camera this often; older verdicts are refreshed
    # The person-scale map: fitted once this many clear standing people were seen, refitted every refit_every more.
    scale_min_samples: int = 50
    scale_refit_every: int = 100
    # Maps learned beforehand ({camera id: {"a", "b", "n"}}, products/proximity/scale_seed.json): a camera with none
    # saved in Redis starts from its seed instead of from nothing. Empty: no seed.
    scale_seed: str = ""
    # The live preview's layers until the UI sets them (iris:ctrl:proximity_preview): the distance lines, the masks of
    # every person and vehicle (person masks are then read on every analysed frame), the boxes, and how far apart
    # (person heights) a person and a moving vehicle still get a line beyond the zone's max gap.
    preview_lines: bool = True
    preview_masks: bool = False
    preview_boxes: bool = True
    preview_depth: bool = False   # the live preview blends the latest depth map (cameras with a depth-check zone)
    preview_line_range: float = 2.0
    # The preview's person and vehicle boxes, steadied (display only, steady.py): each follows its detections with
    # this time constant (s; 0 = the detection's own box), and one whose object vanished stays this long (s; 0 = none)
    preview_box_smoothing_s: float = 0.08
    preview_box_hold_s: float = 0.5

    def __post_init__(self) -> None:
        problems = []
        if not self.person_classes or not self.vehicle_classes:
            problems.append("person_classes and vehicle_classes must not be empty")
        for name in ("max_gap", "stationary_window_s", "stationary_motion", "persist_s", "min_confidence",
                     "depth_max_gap", "depth_every_s", "preview_line_range"):
            if not getattr(self, name) > 0:
                problems.append(f"{name} must be > 0")
        for name in ("grace_s", "cooldown_s"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be >= 0")
        if self.hidden_feet not in HIDDEN_FEET:
            problems.append(f"hidden_feet must be one of {HIDDEN_FEET}")
        if self.gap_smoothing_frames < 1 or self.line_steady_frames < 1:
            problems.append("gap_smoothing_frames and line_steady_frames must be >= 1")
        if not 0.0 <= self.vehicle_band < 1.0:
            problems.append("vehicle_band must be in [0, 1)")
        if self.box_motion < 0:
            problems.append("box_motion must be >= 0")
        for name in ("parked_hold_s", "preview_box_smoothing_s", "preview_box_hold_s"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be >= 0")
        if self.scale_min_samples < 10 or self.scale_refit_every < 1:
            problems.append("scale_min_samples must be >= 10 and scale_refit_every >= 1")
        if problems:
            raise ModuleConfigError("modules.proximity: " + "; ".join(problems))


def _load_depth(path: str):
    model = DepthModel(path)
    model(np.zeros((720, 1280, 3), np.uint8))       # the first run sets CUDA up: at start, not on a candidate frame
    ensure_on_gpu(model.session)
    return model


def _dashed(img, a, b, color, width, on=6, off=4, pad=None) -> None:
    """A dashed segment a -> b (same arguments as cv2.line). Each gap is widened by `pad` (default: the width): a thick
    segment's round caps would close it. An outline passes its line's width, so its dashes stay in step."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 1:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    s = 0.0
    while s < length:
        e = min(s + on, length)
        cv2.line(img, (int(a[0] + ux * s), int(a[1] + uy * s)), (int(a[0] + ux * e), int(a[1] + uy * e)), color, width)
        s += on + off + (width if pad is None else pad)


def _text(img, text, at, color) -> None:
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, LINE_SHADOW, 3, cv2.LINE_AA)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _depth_text(depth):
    """(text, colour) of a line's depth part, or None: the depth check's reading after the gap."""
    if depth is None:
        return None
    state, gap = depth
    if state in ("same", "apart"):
        return f"d {gap:.2f}", DEPTH_SAME if state == "same" else DEPTH_APART
    return ("d ..." if state == "pending" else "d -"), DEPTH_WAIT


def _draw_held(boxes, img, sx, sy) -> None:
    """Boxes whose objects vanished a moment ago, drawn as the engine draws a box: (box, colour, caption) each."""
    for (left, top, w, h), colour, caption in boxes:
        x1, y1 = int(left * sx), int(top * sy)
        cv2.rectangle(img, (x1, y1), (int((left + w) * sx), int((top + h) * sy)), colour, 2)
        cv2.putText(img, caption, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw(lines, img, sx, sy) -> None:
    """Each line and its gap over a dark outline (all outlines first, so lines that cross stay whole). Where the feet
    are hidden, a ring at the estimated foot, joined by a dashed leg to the person's lowest visible point, and the line
    starts at the ring; else a dot at the foot. In a depth zone the depth check's reading follows the gap, and a line
    it says is "apart" is dashed."""
    drawn = []
    for ln in lines:
        a = (int(ln.foot[0] * sx), int(ln.foot[1] * sy))
        b = (int(ln.outline_point[0] * sx), int(ln.outline_point[1] * sy))
        seen = (int(ln.seen_foot[0] * sx), int(ln.seen_foot[1] * sy)) if ln.foot_hidden and ln.seen_foot else None
        start = a
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if seen and length > RING_R:                              # the line leaves from the ring's edge
            start = (int(a[0] + (b[0] - a[0]) * RING_R / length), int(a[1] + (b[1] - a[1]) * RING_R / length))
        color = {"latched": LINE_LATCHED, "close": LINE_CLOSE}.get(ln.style, LINE_FAR)
        width = 3 if ln.close else 2
        line = partial(_dashed, pad=width) if ln.depth is not None and ln.depth[0] == "apart" else cv2.line
        drawn.append((a, start, b, seen, color, width, line, ln))
    for a, start, b, seen, _color, width, line, _ln in drawn:
        line(img, start, b, LINE_SHADOW, width + 2)
        if seen:
            _dashed(img, seen, (a[0], a[1] - RING_R), LINE_SHADOW, 3, pad=1)
            cv2.circle(img, a, RING_R, LINE_SHADOW, 4)
        else:
            cv2.circle(img, a, 4, LINE_SHADOW, -1)
    for a, start, b, seen, color, width, line, ln in drawn:
        line(img, start, b, color, width)
        if seen:
            _dashed(img, seen, (a[0], a[1] - RING_R), color, 1)
            cv2.circle(img, a, RING_R, color, 2)
        else:
            cv2.circle(img, a, 3, color, -1)
        at = ((a[0] + b[0]) // 2 + 4, (a[1] + b[1]) // 2)
        _text(img, f"{ln.gap:.2f}", at, color)
        part = _depth_text(ln.depth)
        if part:
            shift = cv2.getTextSize(f"{ln.gap:.2f} ", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            _text(img, part[0], (at[0] + shift, at[1]), part[1])


def _draw_depth_map(depth_map, img, sx, sy) -> None:
    """The depth map, scaled to the preview, blended half and half under the drawings that follow."""
    h, w = img.shape[:2]
    m = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    img[:, :, :3] = cv2.addWeighted(m, 0.5, img[:, :, :3], 0.5, 0)   # BGR or BGRA: a channel slice is no cv::Mat


def _draw_masks(masks, img, sx, sy) -> None:
    """Each (outline, colour) as a see-through fill with its outline, scaled from mux pixels to the preview's."""
    h, w = img.shape[:2]
    for outline, color in masks:
        pts = np.round(np.array(outline, np.float32) * (sx, sy)).astype(np.int32)
        x0, y0 = np.maximum(pts.min(axis=0), 0)
        x1, y1 = np.minimum(pts.max(axis=0) + 1, (w, h))
        if x1 <= x0 or y1 <= y0:
            continue
        region = img[y0:y1, x0:x1]
        filled = region.copy()
        cv2.fillPoly(filled, [pts - (x0, y0)], color)
        cv2.addWeighted(filled, MASK_ALPHA, region, 1 - MASK_ALPHA, 0, dst=region)
        cv2.polylines(region, [pts - (x0, y0)], True, color, 1, cv2.LINE_AA)


def _colour(kind: str, oid: int, states: dict) -> tuple:
    return PERSON_COLOR if kind == "person" else VEHICLE_COLORS[states.get(oid, "unknown")]


def _shapes(lines, states: dict, outlines: dict) -> list:
    """This frame's measured pairs for the overlay stream: each line with its gap, style and pair, and each person's and
    vehicle's box once, labelled with its class and track id, with the vehicle's state and the mask's outline when
    there is one (a clip draws its incident's pair from them, the snapshot too)."""
    shapes, boxes = [], {}
    for ln in lines:
        line = {"type": "line", "p1": [round(ln.foot[0], 1), round(ln.foot[1], 1)],
                "p2": [round(ln.outline_point[0], 1), round(ln.outline_point[1], 1)],
                "label": f"{ln.gap:.2f}", "style": ln.style, "person": ln.person_id, "vehicle": ln.vehicle_id}
        if ln.foot_hidden and ln.seen_foot:
            line["hidden"], line["seen"] = True, [round(ln.seen_foot[0], 1), round(ln.seen_foot[1], 1)]
        if ln.depth is not None:
            line["depth"] = {"state": ln.depth[0], "gap": None if ln.depth[1] is None else round(ln.depth[1], 3)}
        shapes.append(line)
        boxes[("person", ln.person_id)] = (ln.person_box, f"person {ln.person_id}")
        boxes[("vehicle", ln.vehicle_id)] = (ln.vehicle_box, f"{ln.vehicle_label or 'vehicle'} {ln.vehicle_id}")
    for (style, oid), (rect, label) in boxes.items():
        box = {"type": "box", "rect": [round(float(v), 1) for v in rect], "style": style, "id": oid, "label": label}
        if style == "vehicle":
            box["state"] = states.get(oid, "unknown")
        if outlines.get((style, oid)):
            box["mask"] = outlines[(style, oid)]
        shapes.append(box)
    return shapes


class Proximity(Module):
    name = "proximity"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        # the detector is the sidecar's pgie_config (the engine lists it); the depth check is this module's own
        return [ModelRef(settings.depth_model, "depth_model")] if settings.depth_model else []

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self._versions = VersionedConfig(services.redis, VERSION_KEY, APPLIED_KEY)
        # what the live preview draws: replaced whole (never mutated) by the control thread
        self._preview = self._preview_defaults()
        self._depth = DepthWorker(partial(_load_depth, settings.depth_model)) if settings.depth_model else None

    # build ------------------------------------------------------------------------------------------------
    def on_built(self, pipeline) -> None:
        seed = self._seed()
        for sid in self.services.stream_ids:
            self.services.state(self.name, sid)["camera"] = ProximityCamera(
                self.services.mux_size, self._restore_scale(sid, seed),
                self.settings.depth_every_s, self.settings.gap_smoothing_frames, self.settings.line_steady_frames,
                self.settings.vehicle_band, self.settings.box_motion, self.settings.parked_hold_s)
        self._load()
        self._versions.acknowledge()
        try:                                    # a setting saved before this start (the poller treats it as seen)
            self._set_preview(self.services.redis.get(PREVIEW_KEY) if self.services.redis is not None else None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("proximity: preview setting unreadable at start (%s); defaults apply", exc)

    def start(self) -> None:
        if self._depth is None:
            return
        try:
            self._depth.start()
        except DepthNotOnGpu as exc:
            # A missing or broken model refuses startup (the error propagates); one that only runs on the CPU would
            # take every core of the host, so the check goes off and the zones that ask for it fall back to the image.
            logger.error("proximity: the depth model is NOT on the GPU (%s); the depth check is OFF", exc)
            self._depth = None

    def stop(self) -> None:
        if self._depth is not None:
            self._depth.stop()
        for sid in self.services.stream_ids:
            cam = self.services.state(self.name, sid).get("camera")
            if cam is not None and cam.scale.ready:
                self._write_scale(sid, cam.scale)

    def control(self) -> dict:
        handlers = self._versions.control(lambda _version: self._load())
        handlers[PREVIEW_KEY] = self._set_preview
        return handlers

    def _preview_defaults(self) -> dict:
        s = self.settings
        return {"lines": s.preview_lines, "masks": s.preview_masks, "boxes": s.preview_boxes,
                "depth": s.preview_depth, "line_range": s.preview_line_range}

    def _set_preview(self, raw) -> None:
        """The live preview's layers from the UI's setting (JSON {"masks", "lines", "boxes": bool, "line_range": person
        heights}); this product's defaults for what it leaves out or gets wrong, or when there is none."""
        try:
            chosen = json.loads(raw) if raw else {}
        except ValueError:
            chosen = None
        if not isinstance(chosen, dict):
            logger.warning("proximity: preview setting %r unreadable; defaults apply", raw)
            chosen = {}
        preview = self._preview_defaults()
        for layer in ("lines", "masks", "boxes", "depth"):
            if isinstance(chosen.get(layer), bool):
                preview[layer] = chosen[layer]
        reach = chosen.get("line_range")
        if isinstance(reach, (int, float)) and not isinstance(reach, bool) and math.isfinite(reach) and reach > 0:
            preview["line_range"] = float(reach)
        self._preview = preview

    def gauges(self) -> dict:
        if self._depth is None:
            return {}
        return {"proximity_depth_dropped": lambda: self._depth.dropped,
                "proximity_depth_failed": lambda: self._depth.failed}

    # state ------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        """(Re)read the zones file: each camera's proximity zones and the classes whose masks they need."""
        s = self.settings
        rules = load_zone_rules(s.config_path, s)
        for sid in self.services.stream_ids:
            st = self.services.state(self.name, sid)
            mine = rules.get(sid, [])
            st["persons"] = frozenset().union(*(r.person_classes for r in mine))
            st["vehicles"] = frozenset().union(*(r.vehicle_classes for r in mine))
            st["rules"] = mine                  # last: the streaming thread checks it first
        if self._depth is None and any(r.depth_check for zs in rules.values() for r in zs):
            logger.warning("proximity: a zone asks for the depth check but modules.proximity.depth_model is not "
                           "set; the check is off there")
        logger.info("proximity zones: %s", {sid: [r.zone_id for r in zs] for sid, zs in rules.items()})

    def _seed(self) -> dict:
        """The seed file's maps, {} when there is none or it cannot be read (a warning; never fatal)."""
        if not self.settings.scale_seed:
            return {}
        try:
            with open(self.settings.scale_seed) as f:
                seed = json.load(f)
            if not isinstance(seed, dict):
                raise ValueError("not a mapping of camera id -> fit")
            return seed
        except Exception as exc:  # noqa: BLE001
            logger.warning("proximity: scale seed %s unusable (%s); cameras learn from nothing",
                           self.settings.scale_seed, exc)
            return {}

    def _restore_scale(self, sid: str, seed: dict) -> PersonScale:
        """The camera's map as saved in Redis, else its seed, else a fresh one."""
        s = self.settings
        scale = PersonScale(s.scale_min_samples, s.scale_refit_every)
        try:
            raw = self.services.redis.get(SCALE_KEY.format(sid)) if self.services.redis is not None else None
            if raw:
                scale.restore(json.loads(raw))
                return scale
        except Exception as exc:  # noqa: BLE001 -- a bad saved map is learned again, never fatal
            logger.warning("[%s] proximity: saved scale map unusable (%s)", sid, exc)
        if sid in seed:
            try:
                scale.restore(seed[sid])
                logger.info("[%s] proximity: scale map from the seed file", sid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] proximity: seeded scale map unusable (%s); learning it again", sid, exc)
        return scale

    def _refitted(self, frame: FrameView, st: dict, scale: PersonScale) -> None:
        """A refit: warn when recent people no longer match the map (rarely), save it (at most once a minute)."""
        sid, now = frame.stream_id, frame.now
        if abs(scale.drift) > DRIFT_WARN and now - st.get("drift_warned", float("-inf")) >= DRIFT_WARN_EVERY_S:
            st["drift_warned"] = now
            logger.warning("[%s] proximity: recent people are %+.0f%% the height the scale map expects; has the "
                           "camera moved? (the map re-learns on its own)", sid, 100 * scale.drift)
        if now - st.get("scale_saved", float("-inf")) >= SCALE_SAVE_EVERY_S:
            st["scale_saved"] = now
            self._write_scale(sid, scale)

    def _write_scale(self, sid: str, scale: PersonScale) -> None:
        try:
            if self.services.redis is not None:
                self.services.redis.set(SCALE_KEY.format(sid), json.dumps(scale.to_dict()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] proximity: scale map not saved: %s", sid, exc)

    # hot path ---------------------------------------------------------------------------------------------
    @staticmethod
    def _inferred(frame: FrameView) -> bool:
        """The detector ran on this frame (nvinfer's flag; `analyze` when there is no frame meta, as in tests)."""
        return frame.analyze if frame.frame_meta is None else infer_done(frame.frame_meta)

    def _timed(self, frame: FrameView, st: dict) -> bool:
        """The frame has a timestamp: one without would make the rule's clock jump (and then go back)."""
        if frame.buf_pts != NO_PTS:
            return True
        if frame.now - st.get("no_pts_warned", float("-inf")) >= NO_PTS_WARN_EVERY_S:
            st["no_pts_warned"] = frame.now
            logger.warning("[%s] proximity: frames without a timestamp are skipped", frame.stream_id)
        return False

    def extract(self, frame: FrameView):
        """The instance mask of each person and vehicle as attrs["mask"], on the frames the detector ran on, of a
        camera with proximity zones (person_masks may have read it already)."""
        if not self._inferred(frame):
            return None
        st = self.services.state(self.name, frame.stream_id)
        if not st.get("rules"):
            return None
        # person masks while a zone held a vehicle that may be moving, up to a stationary window ago
        persons_too = self._preview["masks"] or frame.pts <= st.get("person_masks_until", float("inf"))
        wanted = (st["persons"] | st["vehicles"]) if persons_too else st["vehicles"]

        def read(o) -> None:
            if o.kind == "object" and o.detected and o.label in wanted and "mask" not in o.attrs:
                o.attrs["mask"] = instance_mask(o.raw)
        return read

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        """The rule, on every frame the detector ran on (ahead of the publish gating, like the zones' alerts)."""
        if not self._inferred(frame):
            return
        st = self.services.state(self.name, frame.stream_id)
        rules = st.get("rules")
        if not rules:
            self._expire_unruled(frame, st.get("camera"))
            return
        if not self._timed(frame, st):
            return
        cam = st["camera"]
        persons, vehicles = self._objects(frame, st)
        depth = None if self._depth is None else partial(self._depth.latest, frame.stream_id)
        step = cam.step(frame.pts, persons, vehicles, rules, depth)
        for alert in step.alerts:
            self._raise(frame, alert)
        for held in step.suppressed:
            self._quiet(frame, held)
        if step.refit:
            self._refitted(frame, st, cam.scale)
        for e in step.closed:
            self._resolve(frame, e)
        masked = self._preview["masks"]
        paired = {("person", ln.person_id) for ln in step.lines} | {("vehicle", ln.vehicle_id) for ln in step.lines}
        outlines = {(kind, o.id): outline_polygon(o.shape.blob)
                    for kind, group in (("person", persons), ("vehicle", vehicles)) for o in group
                    if o.shape.blob is not None and (masked or (kind, o.id) in paired)}
        masks = [(outline, _colour(kind, oid, step.states)) for (kind, oid), outline in outlines.items()
                 if outline] if masked else []
        st["states"] = step.states
        st["overlay"] = (frame.pts, step.lines, step.parked, masks)
        if step.lines:
            self.services.overlay(frame.stream_id, self.name, frame.pts, self.services.mux_size,
                                  _shapes(step.lines, step.states, outlines), at=frame.arrived)
        if step.active:
            st["person_masks_until"] = frame.pts + max(r.stationary_window_s for r in rules)
        live_map = (self._preview["depth"] and self._depth is not None and any(r.depth_check for r in rules))
        pending = st.get("depth_wanted")
        if step.depth_requests:
            st["depth_wanted"] = (frame.pts, step.depth_requests)
        elif live_map and not (pending and pending[1] and 0 <= frame.pts - pending[0] <= DEPTH_FRAME_SLACK_S):
            st["depth_wanted"] = (frame.pts, [])          # a map refresh for the preview; never over pairs waiting

    def _objects(self, frame: FrameView, st: dict) -> tuple:
        mux_h = frame.mux_size[1]
        persons, vehicles = [], []
        for o in frame.objects:
            if o.kind != "object" or not o.detected:
                continue
            if o.label in st["persons"]:
                if o.conf < self.settings.min_confidence:
                    continue
                blob = blob_of(o.attrs.get("mask"), o.bbox)
                shape = person_shape(blob, mux_h) if blob is not None else person_shape_from_box(o.bbox, mux_h)
                persons.append(Person(o.id, o.label, o.conf, o.bbox, shape))
            elif o.label in st["vehicles"]:
                blob = blob_of(o.attrs.get("mask"), o.bbox)
                shape = vehicle_shape(blob) if blob is not None else vehicle_shape_from_box(o.bbox)
                vehicles.append(Vehicle(o.id, o.label, o.bbox, shape))
        return persons, vehicles

    def _raise(self, frame: FrameView, a) -> None:
        unconfirmed = a.confirmation == "none"
        message = (f"Person within {a.gap:.2f} person heights of a moving {a.vehicle.label} in {a.zone.name}"
                   + (" (feet not visible, unconfirmed)" if unconfirmed else "")
                   + (" -- now more severe" if a.escalated else ""))
        depth = {}
        if a.depth_gap is not None:
            depth = {"depth_gap": f"{a.depth_gap:.3f}", "depth_max_gap": f"{a.zone.depth_max_gap:.2f}"}
            if a.confirmation == "depth" and self._depth is not None:
                got = self._depth.latest_map(frame.stream_id)
                if got is not None:
                    depth["depth_image"] = depth_crop(got[1], self.services.mux_size, a.person, a.vehicle)
        self.services.alert(
            frame.stream_id, "proximity", message=message, severity=a.severity, category="safety", pts=frame.pts,
            zone=a.zone.zone_id, person_ids=",".join(map(str, a.episode.person_ids)),
            vehicle_ids=",".join(map(str, a.episode.vehicle_ids)), vehicle_class=a.vehicle.label,
            gap=f"{a.gap:.2f}", gap_unit="ph", duration_s=f"{a.held_s:.1f}",
            feet_visible="0" if a.foot.hidden else "1", foot_source=a.foot.source, confirmation=a.confirmation,
            escalation="1" if a.escalated else "0", incident=a.episode.incident,
            incident_state="escalated" if a.escalated else "opened", min_gap_pts=f"{a.episode.min_gap_t:.3f}",
            ntp_ts=frame.ntp_ts or None, **depth)
        self.services.request_clip(frame.stream_id, frame.now)
        logger.info("[%s] proximity alert (zone=%s person=%s %s=%s gap=%.2f ph held=%.1f s %s, %s%s)",
                    frame.stream_id, a.zone.zone_id, a.person.id, a.vehicle.label, a.vehicle.id, a.gap, a.held_s,
                    a.confirmation, a.severity, ", escalated" if a.escalated else "")

    def _quiet(self, frame: FrameView, q) -> None:
        """A pair the depth check held back (Ticket 6): one record the app keeps but never shows live -- info, marked
        suppressed, with what it would have alerted at, the reading against the zone's cut-off, the depth image and a
        clip. No incident: nothing opens or resolves."""
        message = (f"Held back by the depth check: person within {q.gap:.2f} person heights of a moving "
                   f"{q.vehicle.label} in {q.zone.name} (depth gap {q.depth_gap:.2f} > {q.zone.depth_max_gap:.2f})")
        depth = {"depth_gap": f"{q.depth_gap:.3f}", "depth_max_gap": f"{q.zone.depth_max_gap:.2f}"}
        if self._depth is not None:
            got = self._depth.latest_map(frame.stream_id)
            if got is not None:
                depth["depth_image"] = depth_crop(got[1], self.services.mux_size, q.person, q.vehicle)
        self.services.alert(
            frame.stream_id, "proximity", message=message, severity="info", category="safety", pts=frame.pts,
            zone=q.zone.zone_id, person_ids=",".join(map(str, q.episode.person_ids)),
            vehicle_ids=",".join(map(str, q.episode.vehicle_ids)), vehicle_class=q.vehicle.label,
            gap=f"{q.gap:.2f}", gap_unit="ph", duration_s=f"{q.held_s:.1f}",
            feet_visible="0" if q.foot.hidden else "1", foot_source=q.foot.source, suppressed="depth",
            would_be=q.severity, min_gap_pts=f"{q.episode.min_gap_t:.3f}", ntp_ts=frame.ntp_ts or None, **depth)
        self.services.request_clip(frame.stream_id, frame.now)
        logger.info("[%s] proximity pair held back by the depth check (zone=%s person=%s %s=%s gap=%.2f ph, depth "
                    "%.2f > %.2f, would be %s)", frame.stream_id, q.zone.zone_id, q.person.id, q.vehicle.label,
                    q.vehicle.id, q.gap, q.depth_gap, q.zone.depth_max_gap, q.severity)

    def _expire_unruled(self, frame: FrameView, cam) -> None:
        """A camera whose proximity zones were all turned off still closes its open episodes once their hold runs out,
        so their incidents resolve."""
        if cam is not None and cam.episodes.open and frame.buf_pts != NO_PTS:
            for e in cam.episodes.expire(frame.pts):
                self._resolve(frame, e)

    def _resolve(self, frame: FrameView, e) -> None:
        """An episode is over: one more entry tells the app its incident is resolved (it makes no event)."""
        self.services.alert(
            frame.stream_id, "proximity", severity=e.severity, category="safety", pts=e.end, zone=e.zone_id,
            incident=e.incident, incident_state="resolved", duration_s=f"{e.end - e.start:.1f}",
            min_gap=f"{e.min_gap:.2f}", min_gap_pts=f"{e.min_gap_t:.3f}",
            person_ids=",".join(map(str, e.person_ids)), vehicle_ids=",".join(map(str, e.vehicle_ids)))
        logger.info("[%s] proximity episode over (zone=%s, %.1f s, persons %s, vehicles %s, min gap %.2f ph)",
                    frame.stream_id, e.zone_id, e.end - e.start, e.person_ids, e.vehicle_ids, e.min_gap)

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        """The live preview, as switched from the UI (PREVIEW_KEY): the boxes in the colour of what they are, or none;
        the masks; the distance line of each close pair, and of each person and moving vehicle within the line range;
        a "parked" tag. What the last analysed frame found is held until the next one."""
        st = self.services.state(self.name, frame.stream_id)
        preview = self._preview
        self._boxes(frame, out, st, preview["boxes"])
        if preview["depth"] and self._depth is not None:
            got = self._depth.latest_map(frame.stream_id)
            if got is not None and 0 <= frame.pts - got[0] <= DEPTH_MAP_HOLD_S:
                out.draw(partial(_draw_depth_map, got[1]))
        overlay = st.get("overlay")
        if overlay is None or frame.buf_pts == NO_PTS or not 0 <= frame.pts - overlay[0] <= OVERLAY_HOLD_S:
            return
        _t, lines, parked, masks = overlay
        if masks and preview["masks"]:
            out.draw(partial(_draw_masks, masks))
        shown = [ln for ln in lines
                 if ln.close or (ln.vehicle_state == "moving" and ln.gap <= preview["line_range"])]
        if shown and preview["lines"]:
            out.draw(partial(_draw, shown))
        if parked:
            for o in frame.objects:
                if o.draw and o.id in parked:
                    out.annotate(o, extra=("parked", PARKED, 0.5, 1))

    def _boxes(self, frame: FrameView, out: FrameOutput, st: dict, shown: bool) -> None:
        """Every box hidden, or on a camera with proximity zones each person's and vehicle's in its colour (a default
        any other module's colour replaces: a zone violation, a hazard), steadied, and drawn on for a moment after its
        object vanished (steady.py; display only)."""
        if not shown:
            for o in frame.objects:
                if o.draw:
                    out.annotate(o, hidden=True)
            return
        if not st.get("rules"):
            return
        states = st.get("states", {})
        styles = st.setdefault("box_styles", {})    # object id -> (colour, caption) as last drawn
        drawn = {}
        for o in frame.objects:
            if not o.draw or o.kind != "object":
                continue
            if o.label in st["persons"]:
                colour = PERSON_COLOR
            elif o.label in st["vehicles"]:
                colour = _colour("vehicle", o.id, states)
            else:
                continue
            drawn[o.id] = o
            styles[o.id] = (colour, f"{o.label} {o.shown_conf:.2f} #{o.id}")
        if frame.buf_pts == NO_PTS:
            for o in drawn.values():
                out.annotate(o, color=styles[o.id][0], priority=-1)
            return
        steadier = st.get("steady")
        if steadier is None:
            steadier = st["steady"] = SteadyBoxes(self.settings.preview_box_smoothing_s, self.settings.preview_box_hold_s)
        steady, held = steadier.update(frame.pts, {oid: o.bbox for oid, o in drawn.items()})
        for oid, o in drawn.items():
            out.annotate(o, color=styles[oid][0], priority=-1, box=steady[oid])
        if held:
            out.draw(partial(_draw_held, [(box, *styles[k]) for k, box in held.items() if k in styles]))
        for k in [k for k in styles if k not in drawn and k not in held]:
            del styles[k]

    def wants_pixels(self, frame: FrameView) -> bool:
        """A depth frame is due on this camera: a close pair waits for its check, none is in flight, and the last
        was depth_every_s ago (on the monotonic clock, which a looping file or a restarted source does not set back).
        A map refresh for the preview (the Depth map switch, no pair waiting) keeps its own clock and goes only when the
        worker is idle, so it never delays or crowds out a pair's check (final review, Ticket 4)."""
        if self._depth is None or frame.buf_pts == NO_PTS:
            return False
        st = self.services.state(self.name, frame.stream_id)
        wanted = st.get("depth_wanted")
        if wanted is None or not 0 <= frame.pts - wanted[0] <= DEPTH_FRAME_SLACK_S:
            return False
        if wanted[1]:
            return (frame.now - st.get("depth_sent", float("-inf")) >= self.settings.depth_every_s
                    and not self._depth.busy(frame.stream_id))
        return frame.now - st.get("map_sent", float("-inf")) >= self.settings.depth_every_s and self._depth.idle()

    def on_pixels(self, frame: FrameView, bgr) -> None:
        if not self.wants_pixels(frame):                    # another module asked for the pixels
            return
        st = self.services.state(self.name, frame.stream_id)
        _t, requests = st.pop("depth_wanted")
        if self._depth.submit(frame.stream_id, frame.pts, bgr.copy(), frame.capture_scale(), requests):
            st["depth_sent" if requests else "map_sent"] = frame.now


MODULE = Proximity
