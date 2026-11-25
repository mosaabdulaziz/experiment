"""Run a secondary GIE only on the frames whose metadata is published, and only on the objects a module allows.

The decision logic is pure stdlib (no gi / GStreamer imports at module level), so it is unit-testable on the host;
`GieGate.install` adds the two pad probes that apply it.

Why: the probe reads a secondary GIE's output only on meta frames (need_meta, one per sample_interval per source),
but the GIE infers on every object of every frame -- at 25-30 fps and sample_fps 12 more than half of that work is
never read (2026-09-30: one such GIE was ~40% of the sidecar's GPU time, the largest single stage).

How: nvinfer's `interval` does not gate a secondary GIE, and `infer-on-class-ids` is element-wide, not per frame.
The per-object gate that is per frame is should_infer_object()'s first check (gstnvinfer.cpp, DeepStream 9.0): an
object whose unique_component_id is not the GIE's operate-on-gie-id is skipped. So a probe on the GIE's sink pad
moves the objects of an unsampled frame (or outside the allowed area) out of that id (hidden_id) and a probe on its
src pad moves them back (restored_id) before anything else reads them.

One decision, two readers: the sink-pad probe decides (decide) and the main probe downstream takes that same
decision as its need_meta (take). Were the main probe to keep its own clock, one frame it skips (a failed surface
map) would put the two a frame apart for good -- every published frame without the GIE's output.
"""

from collections import OrderedDict

# unique_component_id is a gint; gie ids are small (a handful per pipeline),
# so an offset this size can't collide with a real one.
HIDE_OFFSET = 1000


def hidden_id(uid: int, operate_on: int) -> int:
    """The id that makes the SGIE skip an object it would otherwise infer on."""
    return uid + HIDE_OFFSET if uid == operate_on else uid


def restored_id(uid: int) -> int:
    """Undo hidden_id (a no-op for any id hidden_id didn't move)."""
    return uid - HIDE_OFFSET if uid >= HIDE_OFFSET else uid


class SampleGate:
    """Per-source sampling clock with the probe's need_meta rule: the first
    frame, then the first frame at least `sample_interval` after the last
    sampled one. Decisions wait keyed by (source, buf_pts) until the main
    probe takes them; `maxlen` bounds them per source should one never be."""

    def __init__(self, sample_interval: float, maxlen: int = 64):
        self.sample_interval, self.maxlen = sample_interval, maxlen
        self._last: dict[int, float] = {}
        self._pending: dict[int, OrderedDict] = {}

    def decide(self, source: int, buf_pts: int, pts: float) -> bool:
        last = self._last.get(source)
        due = last is None or pts - last >= self.sample_interval
        if due:
            self._last[source] = pts
        pending = self._pending.setdefault(source, OrderedDict())
        pending[buf_pts] = due
        while len(pending) > self.maxlen:
            pending.popitem(last=False)
        return due

    def take(self, source: int, buf_pts: int) -> bool | None:
        """The decision made for this frame, once; None if there is none."""
        return self._pending.get(source, {}).pop(buf_pts, None)


class GieGate:
    """The sampling clock plus the two pad probes that steer one secondary GIE with it.

    `object_allowed(camera, left, top, width, height)` (box as fractions of the mux frame) lets a module keep the
    GIE off objects it will not read, e.g. outside an area; None allows all."""

    def __init__(self, element, name: str, sample_interval: float, mux_size: tuple, index_to_id: dict,
                 object_allowed=None) -> None:
        self.element, self.name = element, name
        self.clock = SampleGate(sample_interval)
        self.operate_on = int(element.get_property("infer-on-gie-id"))
        self._mux, self._cameras, self._allowed = mux_size, index_to_id, object_allowed

    def take(self, source: int, buf_pts: int):
        """The sampling decision made for this frame, once (see SampleGate.take)."""
        return self.clock.take(source, buf_pts)

    @classmethod
    def install(cls, element, name: str, engine, mux_size: tuple, index_to_id: dict, object_allowed=None):
        """Gate `element`, or None (with a warning) when the pipeline cannot be gated: the gate cannot see which
        frames matter while the preview draws on every frame or process_fps skips some, and an element on every
        gie cannot be steered by id."""
        import logging
        logger = logging.getLogger("iris.sidecar")
        if engine.pipeline_preview or engine.infer_interval > 1:
            logger.warning("%s sample_only ignored: needs pipeline_preview=false and no process_fps throttle "
                           "(pipeline_preview=%s, infer_interval=%d)", name, engine.pipeline_preview,
                           engine.infer_interval)
            return None
        if int(element.get_property("infer-on-gie-id")) < 0:
            logger.warning("%s sample_only ignored: it has no operate-on-gie-id", name)
            return None
        gate = cls(element, name, engine.sample_interval, mux_size, index_to_id, object_allowed)
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        element.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, gate._hide)
        element.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, gate._restore)
        logger.info("%s on sampled frames only (every %.3fs per source, operate-on-gie-id=%d)", name,
                    engine.sample_interval, gate.operate_on)
        return gate

    def _hide(self, pad, info):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        from .nvds import batch_objects
        mux_w, mux_h = self._mux
        for frame_meta, objs in batch_objects(info):
            due = self.clock.decide(frame_meta.pad_index, frame_meta.buf_pts, frame_meta.buf_pts / 1e9)
            cam = self._cameras.get(frame_meta.pad_index)
            for obj in objs:
                rp = obj.rect_params
                if not due or (self._allowed is not None and not self._allowed(
                        cam, rp.left / mux_w, rp.top / mux_h, rp.width / mux_w, rp.height / mux_h)):
                    obj.unique_component_id = hidden_id(obj.unique_component_id, self.operate_on)
        return Gst.PadProbeReturn.OK

    def _restore(self, pad, info):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        from .nvds import batch_objects
        for _frame_meta, objs in batch_objects(info):
            for obj in objs:
                obj.unique_component_id = restored_id(obj.unique_component_id)
        return Gst.PadProbeReturn.OK
