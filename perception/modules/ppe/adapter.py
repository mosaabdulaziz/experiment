"""From the engine's ObjectViews to the vendored rules' inputs, and track ids back onto their results.

The only IRIS-specific glue between a frame and `compliance/` (which knows nothing of DeepStream):
- persons: the pgie's tracked, detected `person` boxes; a shadow track (the detector missed it this frame) or an
  untracked person is left out, because the rules judge only what the detector saw and smoothing needs a track id;
- PPE: the PPE detector's boxes of the four PPE labels, all of them (they are untracked by design, see
  `Ppe.gies`); its other labels are filtered in its nvinfer config already, and VoxDroid's own `Person` class never
  counts as a person;
- boxes: ObjectView's (left, top, width, height) in mux pixels -> the rules' (x1, y1, x2, y2), same pixels; the
  frame size the rules check truncation against is the mux size.
"""
from dataclasses import dataclass, field

from .compliance.types import Detection

PERSON_GIE = "pgie"
PERSON_LABEL = "person"
PPE_GIE = "pgie_ppe"
PPE_LABELS = ("Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest")
UNTRACKED = 0xFFFFFFFFFFFFFFFF          # DeepStream's object id for an object the tracker never saw


def to_bbox(o) -> tuple:
    """ObjectView box (left, top, width, height) -> (x1, y1, x2, y2), in the same (mux) pixels."""
    return (o.left, o.top, o.left + o.width, o.top + o.height)


@dataclass
class FrameInput:
    """One frame, ready for `compliance.evaluate_frame`."""
    detections: list                     # persons first, then PPE boxes, each in metadata order
    image_size: tuple                    # (width, height) of the frame the boxes are in: the mux
    track_ids: dict = field(default_factory=dict)   # id(person Detection) -> tracker id

    def attach_track_ids(self, results: list) -> list:
        """Set `track_id` on each PersonResult from the person Detection it was built from (the rules keep the
        Detection objects they are given, so identity is the key)."""
        for r in results:
            r.track_id = self.track_ids[id(r.person)]
        return results


def frame_input(objects, mux_size) -> FrameInput:
    persons, ppe, track_ids = [], [], {}
    for o in objects:
        if not o.detected:
            continue
        if o.gie_name == PERSON_GIE and o.label == PERSON_LABEL and o.id != UNTRACKED:
            d = Detection(o.label, o.conf, to_bbox(o))
            track_ids[id(d)] = o.id
            persons.append(d)
        elif o.gie_name == PPE_GIE and o.label in PPE_LABELS:
            ppe.append(Detection(o.label, o.conf, to_bbox(o)))
    return FrameInput(persons + ppe, tuple(mux_size), track_ids)

