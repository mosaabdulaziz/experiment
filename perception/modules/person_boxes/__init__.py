"""person_boxes: the tracked people's boxes in iris:meta, for consumers that reason about how people relate.

Published as `person_boxes`: [[left, top, width, height, object id], ...] in mux coordinates, one per person the
tracker reports and the engine draws (a shadow track included when the engine draws shadow tracks), on frames that
feed iris:meta; omitted when nobody is in frame. Consumers use them to check, for example, whether two person
tracks overlap (attendance's collision guard) or which person a face belongs to.

The people are noted in `transform`, where the module is listed, and published in `analyze`: list it BEFORE a module
that filters detections (the filter zones) and people that filter drops are still reported, as a consumer reasoning
about who is near whom needs; list it after and they are not.
"""
from perception.engine.frame import FrameOutput, FrameView
from perception.engine.modules import Module


class PersonBoxes(Module):
    name = "person_boxes"

    def transform(self, frame: FrameView) -> None:
        if frame.need_meta:
            frame.scratch[self.name] = [[o.left, o.top, o.width, o.height, o.id] for o in frame.objects
                                        if o.kind == "object" and o.label == "person" and o.draw]

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        boxes = frame.scratch.get(self.name)
        if boxes:
            out.fields["person_boxes"] = boxes


MODULE = PersonBoxes
