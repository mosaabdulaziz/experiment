"""person_masks: each person's segmentation mask as a compact outline in iris:meta.

For a segmentation detector (the chair engine's YOLO-seg model) DeepStream attaches an instance mask to every
object. The module turns each drawn person's mask into the largest blob's contour, simplified to at most 40
corners and normalised 0..1 like the zones -- about 0.5 KB per person in the stream instead of a full mask.

Published as `person_masks`: [[object id, confidence, outline | None], ...] on frames that feed iris:meta; the
outline is None for an object without a readable mask (a shadow track, a detection-only model).

Like person_boxes, the people are read in `transform` (list it before a module that filters detections to keep the
people that filter drops) and published in `analyze`.
"""
from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import Module
from perception.engine.nvds import instance_mask

from .outline import outline


def mask_outline_of(o: ObjectView, mux_w: float, mux_h: float):
    """The object's instance mask as an outline, or None: no mask on this object (shadow-tracked, or a
    detection-only model) or one that cannot be read. A bad mask must never cost the frame."""
    try:
        mask = o.attrs.get("mask")
        if mask is None:
            return None
        scores = mask.values.reshape(mask.height, mask.width)
        return outline(scores, mask.threshold, o.bbox, (mux_w, mux_h))
    except Exception:  # noqa: BLE001
        return None


class PersonMasks(Module):
    name = "person_masks"

    def extract(self, frame: FrameView):
        """Each person's instance mask as attrs["mask"], on the frames that feed iris:meta."""
        if not frame.need_meta:
            return None

        def read(o: ObjectView) -> None:
            if o.kind == "object" and o.label == "person":
                o.attrs["mask"] = instance_mask(o.raw)
        return read

    def transform(self, frame: FrameView) -> None:
        if not frame.need_meta:
            return
        mux_w, mux_h = frame.mux_size
        frame.scratch[self.name] = [
            [o.id, round(o.conf, 3), mask_outline_of(o, mux_w, mux_h)]
            for o in frame.objects if o.kind == "object" and o.label == "person" and o.draw and o.raw is not None]

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        masks = frame.scratch.get(self.name)
        if masks:
            out.fields["person_masks"] = masks


MODULE = PersonMasks
