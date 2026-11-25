"""box_stabilizer: steady, whole-object boxes in place of the detector's flicker.

A tall one-class detector (the vehicle model) sometimes alternates between a box on the whole object and a tight box
on part of it (a tractor's cab). This module drops the nested part-box when the whole-object box is present, keeps
each object's full-size box stable across that flicker (a much smaller update translates the held box instead of
shrinking it), and keeps a vanished object's box alive for `hold_seconds`, so a held box also stops its cab surfacing
as a NEW tracker id. It mirrors the offline annotator (mtmc/annotate_videos.py), so the preview and the meta feed
show what the review tool shows.

It replaces the objects it stabilizes with new ones, and publishes every box it emits, shadow tracks included,
because whatever follows (tracking across cameras) needs consistent detections frame to frame. List it before any
module that filters or matches detections.
"""
import logging
from dataclasses import dataclass, field

from perception.engine.frame import FrameView, ObjectView
from perception.engine.modules import Module, ModuleConfigError

from .stabilizer import BoxStabilizer

logger = logging.getLogger("iris.sidecar")


@dataclass
class Settings:
    # Only these labels are stabilized; every other label passes through untouched.
    labels: list = field(default_factory=lambda: ["vehicle"])
    # Drop any box whose area is at least this much inside another box of the same label.
    containment_ratio: float = 0.85
    # How long a missing object's canonical box stays drawable (matches the annotator's 15 frames at 20 fps).
    hold_seconds: float = 0.75

    def __post_init__(self) -> None:
        if not self.labels:
            raise ModuleConfigError("modules.box_stabilizer: labels is empty; nothing would be stabilized")
        if not 0.0 < self.containment_ratio <= 1.0:
            raise ModuleConfigError(f"modules.box_stabilizer: containment_ratio {self.containment_ratio!r} is not in (0, 1]")
        if self.hold_seconds < 0:
            raise ModuleConfigError("modules.box_stabilizer: hold_seconds must be >= 0")


class BoxStabilizerModule(Module):
    name = "box_stabilizer"
    Settings = Settings

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.stabilizer = BoxStabilizer(labels=settings.labels, ratio=settings.containment_ratio,
                                        hold_seconds=settings.hold_seconds)

    def transform(self, frame: FrameView) -> None:
        objects = [o for o in frame.objects if o.draw and o.kind == "object"]
        boxes = [(o.left, o.top, o.width, o.height, o.label, o.shown_conf, o.id) for o in objects]
        try:
            # stabilize() always emits still-held virtual boxes, even when this frame has zero raw detections
            # (a flicker frame).
            stabilized = self.stabilizer.stabilize(frame.stream_id, boxes, frame.pts)
        except Exception as exc:  # noqa: BLE001 -- on a failure the raw boxes stay as they were
            logger.debug("[%s] box stabilization failed (using raw boxes): %s", frame.stream_id, exc)
            return
        for o in objects:
            o.draw = o.publish = False
        frame.objects.extend(
            ObjectView(id=b[6], label=b[4], conf=b[5], tracker_conf=b[5], left=b[0], top=b[1], width=b[2],
                       height=b[3]) for b in stabilized)


MODULE = BoxStabilizerModule
