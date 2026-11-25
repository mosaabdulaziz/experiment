"""From the engine's ObjectViews to the heat-map detector's input.

The only IRIS-specific glue between a frame and `core/` (which knows nothing of DeepStream):
- voids: every box of the shelf detector (gie "pgie_shelf") with the label `empty`, down to the detector's own floor
  (its nvinfer config's pre-cluster-threshold); the boxes are untracked (the detector sits after the tracker), and the
  heat map is what filters them over time. Any model whose labels file names a class `empty` works: the module never
  reads a class id;
- persons (stability: tracker): the shared pgie's `person` boxes detected on this frame, grown and clamped as the
  shelf-demo does, for the person pause;
- boxes: ObjectView's (left, top, width, height) in mux pixels -> (x1, y1, x2, y2, conf), same pixels; the frame the
  heat map covers is the mux frame.
"""
SHELF_GIE = "pgie_shelf"
EMPTY_LABEL = "empty"
PERSON_GIE = "pgie"                      # the shared pipeline's COCO detector (tracked)
PERSON_LABEL = "person"


def void_boxes(objects, gie: str = SHELF_GIE, label: str = EMPTY_LABEL) -> list:
    """This frame's void boxes as (x1, y1, x2, y2, conf), in metadata order."""
    return [(o.left, o.top, o.left + o.width, o.top + o.height, o.conf)
            for o in objects if o.gie_name == gie and o.label == label and o.detected]


def person_boxes(objects, size, conf: float = 0.35, pad: float = 0.15, gie: str = PERSON_GIE,
                 label: str = PERSON_LABEL) -> list:
    """This frame's persons of at least `conf`, each grown by `pad` of its width/height per side and clamped to the
    frame of `size` (width, height), as int (x1, y1, x2, y2): the shelf-demo's person boxes (void_roi.py step 3a).
    A shadow track (detector missed it) is not a detection this frame, as in the demo."""
    w, h = size
    out = []
    for o in objects:
        if o.gie_name == gie and o.label == label and o.detected and o.conf >= conf:
            x1, y1, x2, y2 = o.left, o.top, o.left + o.width, o.top + o.height
            pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
            out.append((max(int(x1 - pw), 0), max(int(y1 - ph), 0), min(int(x2 + pw), w), min(int(y2 + ph), h)))
    return out
