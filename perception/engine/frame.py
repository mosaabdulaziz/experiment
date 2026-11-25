"""What a module sees of a frame, and how it influences what is published.

The engine builds one `FrameView` per frame in the probe (objects are extracted once, lazily) and hands it to
every module's hooks. Modules never touch pyds metadata themselves, and never see another module's state: data
flows between modules only through `ObjectView` fields and `FrameOutput`.

Coordinates are always mux coordinates (the resolution nvstreammux scales every source to), as (left, top,
width, height). `capture_scale()` converts to the capture resolution used for pixel work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(eq=False)            # compared and hashed by identity: annotations and flags belong to THIS object
class ObjectView:
    """One object nvinfer/nvtracker reported on a frame, or a box a module added or replaced.

    The engine's per-frame object list is the single place detections live. `attrs` carries whatever a module
    attached to an object (an analytics region it is in, facial landmarks); the engine does not know the keys.
    Four fields say what becomes of an object, and modules change them in `transform`:
      kind     what the object is: "object" is an ordinary detection; modules use their own kinds
               for objects they take over, so another module can leave them alone
      draw     the overlay draws it
      publish  it goes into the detections of iris:meta
      layer    draw order, lower first (so a name written on a face box is not painted over by a person box)"""
    id: int                              # tracker object id (unique per stream while tracked)
    label: str
    conf: float                          # detector confidence; negative for a shadow track the detector missed
    tracker_conf: float
    left: float
    top: float
    width: float
    height: float
    gie_name: str = ""                   # unique name of the inference element that produced it
    parent_id: Optional[int] = None      # tracker id of the object this one was found inside (secondary GIEs)
    raw: Any = None                      # the pyds object; only valid during the probe, never kept by a module
    attrs: dict = field(default_factory=dict)   # what modules attach to the object: an extractor (Module.extract)
                                         # reads it off the metadata, a module's own code adds the rest. Keys are
                                         # a module's to document; the engine never reads them
    kind: str = "object"
    draw: bool = True
    publish: bool = True
    layer: int = 1

    @property
    def bbox(self) -> tuple:
        return (self.left, self.top, self.width, self.height)

    @property
    def detected(self) -> bool:
        """True when the detector (not just the tracker) saw it this frame."""
        return self.conf >= 0

    @property
    def shown_conf(self) -> float:
        """Confidence to display: the tracker's own for a shadow track, so no negative number is shown."""
        return self.conf if self.conf >= 0 else self.tracker_conf


@dataclass
class FrameView:
    stream_id: str
    state: Any                           # the stream's StreamState (engine-owned, per stream)
    pad_index: int
    frame_num: int
    pts: float                           # seconds
    now: float                           # time.monotonic() when the probe saw the frame
    ntp_ts: int = 0                      # camera clock in ns (0 = none)
    buf_pts: int = 0                     # the buffer's presentation time in ns, as the stream clock gives it
    arrived: float = 0.0                 # wall time the frame entered nvstreammux (segment_index.arrival_wall); 0 = unknown
    analyze: bool = True                 # False on frames skipped by the process_fps throttle
    need_meta: bool = False              # this frame feeds iris:meta
    need_stream: bool = False            # this frame feeds the preview
    mux_size: tuple = (0, 0)
    capture_size: tuple = (0, 0)
    bgr: Any = None                      # CPU copy of the frame (capture resolution), set only if a module asked
    scratch: dict = field(default_factory=dict)   # per-frame notes, keyed by module name: what an earlier hook saw
    dets_with_boxes: bool = False        # a module asks for the detections of iris:meta to carry each box (7 fields)
    frame_meta: Any = None               # the pyds frame meta; only valid during the probe
    _objects_loader: Optional[Callable[[], list]] = field(default=None, repr=False)
    _objects: Optional[list] = field(default=None, repr=False)

    @property
    def objects(self) -> list:
        """Every object on the frame, extracted on first use and shared by all modules."""
        if self._objects is None:
            self._objects = self._objects_loader() if self._objects_loader else []
        return self._objects

    @objects.setter
    def objects(self, value: list) -> None:
        self._objects = value

    def capture_scale(self) -> tuple:
        mw, mh = self.mux_size
        cw, ch = self.capture_size
        return (cw / mw if mw else 1.0, ch / mh if mh else 1.0)


@dataclass
class Annotation:
    """How the overlay should draw one object. Unset fields keep the engine's default.

    `color`/`text` each remember the priority of whoever set them: a later annotation only replaces them with an
    equal or higher priority, so the result does not depend on the order modules run in. `text=""` suppresses the
    default label (a module that owns a kind of object and labels it itself does that at priority -1). `extra` is more text drawn under the box: (text, color, scale, thickness).
    `box` (left, top, width, height in mux px) is drawn in place of the object's own box -- a steadier one for the
    preview; the object itself, and what is published of it, keep the detection's."""
    color: Optional[tuple] = None        # BGR, as OpenCV draws it
    text: Optional[str] = None
    hidden: bool = False
    extra: list = field(default_factory=list)
    box: Optional[tuple] = None
    color_priority: int = -1             # below any annotate() call's default priority 0, so a module can set a default
    text_priority: int = -1              # (priority=-1) that every other annotation replaces, whatever the module order
    box_priority: int = -1


class FrameOutput:
    """What modules add to the frame's published result. Created by the engine per frame."""

    def __init__(self) -> None:
        self.annotations: dict[ObjectView, Annotation] = {}   # object -> overlay change
        self.fields: dict[str, Any] = {}               # extra top-level iris:meta fields (JSON-dumped by the engine)
        self.publish_hooks: list = []                  # callables run on the publisher thread, see after_publish
        self.draws: list = []                          # callables run on the publisher thread, see draw
        self.need_pixels = False

    def annotate(self, obj: ObjectView, *, color: Optional[tuple] = None, text: Optional[str] = None,
                 hidden: bool = False, extra: Optional[tuple] = None, priority: int = 0,
                 box: Optional[tuple] = None) -> None:
        a = self.annotations.setdefault(obj, Annotation())
        if color is not None and priority >= a.color_priority:
            a.color, a.color_priority = color, priority
        if text is not None and priority >= a.text_priority:
            a.text, a.text_priority = text, priority
        if box is not None and priority >= a.box_priority:
            a.box, a.box_priority = tuple(box), priority
        if hidden:
            a.hidden = True
        if extra is not None:
            a.extra.append(extra)

    def force_pixels(self) -> None:
        self.need_pixels = True

    def draw(self, fn: Callable[[Any, float, float], None]) -> None:
        """Run `fn(img, sx, sy)` on the publisher thread after the boxes are drawn, to paint something of the module's
        own on the preview frame (a zone outline, a distance line between two objects). `img` is the BGR frame at
        capture resolution; multiply mux coordinates by `sx`, `sy` to get its pixels. Runs only on frames that are
        drawn, and a failure in it costs the frame its preview, so keep it simple."""
        self.draws.append(fn)

    def after_publish(self, fn: Callable[[Any], None]) -> None:
        """Run `fn(clean_bgr)` on the publisher thread once this frame is published. `clean_bgr` is the frame as
        captured, before boxes and labels were drawn on it (a copy when the frame is drawn), so a module can crop
        faces or bodies out of it. It needs the frame's pixels (FrameView.bgr must have been set: ask for them in
        wants_pixels); with no pixels, `fn` is not called. Keep it quick or hand the work to a thread of your own:
        the publisher thread is shared by every module and stream."""
        self.publish_hooks.append(fn)


def detection_rows(objects, full: bool) -> list:
    """The `dets` of iris:meta: one row per published object, in list order.

    `full` adds the box ([label, conf, id, left, top, width, height], conf as shown on the overlay); without it the
    row is [label, conf, id]. Which one a camera feed carries is the legacy wire shape: consumers unpack exactly
    the three or the seven fields."""
    rows = []
    for o in objects:
        if not o.publish:
            continue
        if full:
            rows.append([o.label, round(o.shown_conf, 3), o.id,
                         round(o.left, 1), round(o.top, 1), round(o.width, 1), round(o.height, 1)])
        else:
            rows.append([o.label, round(o.conf, 3), o.id])
    return rows
