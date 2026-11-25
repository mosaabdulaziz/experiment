"""plates: find the licence plate on each vehicle, keep one steady plate per vehicle, and say where it is.

A secondary detector looks at each vehicle's crop (the detector runs on every few frames only), so every plate comes
with the vehicle it was found on (its parent). The module then, in `transform`:

- drops what cannot be a plate: too small or oddly shaped to be read (`min_*_px`, `min_aspect`/`max_aspect`: the
  reader makes characters up from a 6 x 6 box), cut by the frame edge, floating off its vehicle, or big next to it;
- gives a plate whose vehicle is gone (a dropped nested box, outside the filter zones) to the smallest surviving
  vehicle box that holds its centre, and keeps ONE plate per vehicle, the one nearest the bottom-centre of its box;
- holds each plate on its vehicle for `hold_seconds` between detections, as a position relative to the vehicle's box,
  so it moves with the vehicle instead of blinking;
- logs one summary line per vehicle once it has been out of sight a few seconds: was a plate found, kept, dropped and why.

It leaves two kinds of object in the frame: kind "plate" (one per vehicle per detection, not drawn: the plate as
detected, with its vehicle's track id in `parent_id`, which a module that follows vehicles uses) and kind
"plate_shown" (the held box that is drawn, labelled "plate"). The `plates` meta field is the shown boxes:
[[left, top, width, height, confidence, vehicle id], ...].

List it after the modules that decide which vehicles count (box_stabilizer, zones) and before `mtmc`.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import GieSpec, Module, ModuleConfigError

from .hold import PlateDiag, PlateHold, one_plate_per_vehicle, resolve_parent

logger = logging.getLogger("iris.sidecar")

DETECTION_LABEL = "plate_det"      # what the detector's config names its class


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo/config_sgie_plate.txt"
    engine: str = ""                           # the engine nvinfer deserializes; empty: from the config
    confidence_threshold: float = 0.3
    hold_seconds: float = 1.5
    # A plate box smaller than this (pixels of the mux frame) or with a width/height ratio outside the range is dropped
    # before it is chosen, drawn or read: it cannot be read, and the reader makes characters up from it.
    min_width_px: float = 28
    min_height_px: float = 12
    min_aspect: float = 1.0
    max_aspect: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ModuleConfigError("modules.plates: confidence_threshold must be within 0..1")
        if self.min_aspect > self.max_aspect:
            raise ModuleConfigError("modules.plates: min_aspect is larger than max_aspect")
        if self.hold_seconds < 0:
            raise ModuleConfigError("modules.plates: hold_seconds must be >= 0")


def _vehicle_box(o: ObjectView) -> tuple:
    return (o.left, o.top, o.width, o.height, o.label, o.shown_conf, o.id)


class Plates(Module):
    name = "plates"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        return [ModelRef(settings.config, "config", "nvinfer")]

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.diag = PlateDiag(emit=logger.info)
        self._drops_logged: set = set()          # one log line per camera, vehicle and reason

    def gies(self) -> list:
        s = self.settings
        return [GieSpec(name="sgie_plate", config_path=s.config, role="secondary", engine_path=s.engine or None,
                        batch_from_mux=False)]

    def transform(self, frame: FrameView) -> None:
        s = self.settings
        mux = frame.mux_size
        detections = [o for o in frame.objects if o.label == DETECTION_LABEL]
        raw, plates = [], []                     # (parent id, conf, box) of every plate seen; the ones that count
        for o in detections:
            o.kind, o.draw, o.publish = "plate_det", False, False      # never a vehicle of its own
            parent = o.parent_id if o.parent_id is not None else -1
            raw.append((parent, o.conf, o.bbox))
            if o.conf >= s.confidence_threshold:
                plates.append((o.left, o.top, o.width, o.height, round(o.conf, 3), parent))
        state = self.services.state(self.name, frame.stream_id)
        hold: Optional[PlateHold] = state.get("hold")
        if hold is None:
            hold = state["hold"] = PlateHold(s.hold_seconds)
        vehicles = [_vehicle_box(o) for o in frame.objects if o.draw and o.kind == "object"]
        lost: list = []
        if plates:
            # A plate belongs to the vehicle it was found on. Only vehicles that survived the filters above count; a
            # plate whose vehicle is gone goes to the smallest surviving vehicle box that holds its centre.
            veh_ids = {b[6] for b in vehicles}
            kept = []
            for pl, pt_, pw, ph, pc, pid in plates:
                if pid not in veh_ids:
                    cx, cy = pl + pw / 2.0, pt_ + ph / 2.0
                    inside = [b for b in vehicles if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]]
                    if not inside:
                        continue
                    pid = min(inside, key=lambda b: b[2] * b[3])[6]
                kept.append((pl, pt_, pw, ph, pc, pid))
            # One plate per vehicle: the one nearest the bottom-centre of its box; a plate that is not on its vehicle
            # is dropped instead of floating beside it.
            plates = one_plate_per_vehicle(
                kept, vehicles, remembered=hold.centres(), dropped=lost, min_size=(s.min_width_px, s.min_height_px),
                aspect=(s.min_aspect, s.max_aspect), frame=mux)
            for pl, v, why in lost:
                key = (frame.stream_id, v[6] if v else -1, why)
                if key not in self._drops_logged:
                    self._drops_logged.add(key)
                    logger.info("[plate] %s: a plate (conf %.2f, box %s) on vehicle #%s (box %s) was dropped: %s",
                                frame.stream_id, pl[4], [round(x) for x in pl[:4]], pl[5],
                                [round(x) for x in v[:4]] if v else None, why)
        diag_vehicles = [b for b in (_vehicle_box(o) for o in frame.objects if o.draw) if b[4] not in ("face", "person")]
        self.diag.frame(frame.stream_id, frame.pts, {b[6] for b in diag_vehicles},
                        [(resolve_parent(pid, bx, diag_vehicles), c) for pid, c, bx in raw], plates, lost)
        # What is drawn and published is steadier than what was detected: the detector runs on every few frames only,
        # so each plate is held on its vehicle in between.
        shown = hold.update(plates, vehicles, frame.pts)
        frame.objects.extend(
            ObjectView(id=-1, label="plate", conf=pc, tracker_conf=pc, left=pl, top=pt_, width=pw, height=ph,
                       parent_id=pid, kind="plate", draw=False, publish=False)
            for pl, pt_, pw, ph, pc, pid in plates)
        frame.objects.extend(
            ObjectView(id=-1, label="plate", conf=pc, tracker_conf=pc, left=pl, top=pt_, width=pw, height=ph,
                       parent_id=pid, kind="plate_shown", publish=False)
            for pl, pt_, pw, ph, pc, pid in shown)

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        shown = [[round(o.left, 1), round(o.top, 1), round(o.width, 1), round(o.height, 1), o.conf, o.parent_id]
                 for o in frame.objects if o.kind == "plate_shown"]
        if shown:
            out.fields["plates"] = shown


MODULE = Plates
