"""GPU JPEG encoder (nvjpegenc) for the frame publisher.

Replaces the publisher's cv2.imencode (5-15 ms/frame CPU JPEG at 1280x720) with
DeepStream's nvjpegenc, which encodes an NVMM surface with nvJPEG on the GPU.

The publisher's annotated frame is a CPU BGR array (box drawing happens in
Python — nvdsosd was abandoned for cross-stream box mixing), so the encode path
is a small standalone pipeline owned by the publisher worker thread:

    appsrc (video/x-raw BGR)
        -> nvvideoconvert -> capsfilter(NVMM RGB) -> nvjpegenc -> appsink

The BGR->NVMM upload + convert + GPU encode together cost ~1-3 ms versus 5-15 ms
for cv2.imencode, and they run on the worker thread — never the GStreamer
streaming thread. If the elements are unavailable or the pipeline wedges,
encode() returns None and the publisher falls back to cv2.imencode for that
frame (and permanently after a few consecutive failures).

The mini-pipeline runs its own GLib main context on a private thread: hardware
elements (nvvideoconvert/nvjpegenc) need the context iterated for their async
state changes and stream-start, and a private context guarantees the encoder
never interferes with the sidecar's main pipeline loop.
"""
import logging
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

import numpy as np  # noqa: E402

# Benchmarking-mode per-element timing (the telemetry module is importable when run in
# the container). Guarded: this module stays importable off-GPU and without the
# telemetry module present.
try:
    from . import telemetry  # noqa: E402
    _telemetry = telemetry
    del telemetry
except ImportError:
    _telemetry = None

logger = logging.getLogger("iris.sidecar.jpeg")


def _attach_jpeg_probe(encoder, stage="gpu.jpeg"):
    """Time nvjpegenc as the ``stage`` telemetry stage (default ``gpu.jpeg``; sink/src pad probes).

    No-op when telemetry is disabled or the telemetry module is unavailable.
    The measured delta is the GPU nvJPEG encode time per published frame — the
    element-level counterpart of the ``publish.draw`` Python-side stage.
    """
    if _telemetry is None or not _telemetry.enabled():
        return
    sink = encoder.get_static_pad("sink")
    src = encoder.get_static_pad("src")
    if sink is None or src is None:
        return
    sink.add_probe(Gst.PadProbeType.BUFFER,
                   lambda _p, _i: (_telemetry.gpu_enter(stage),
                                   Gst.PadProbeReturn.OK)[1])
    src.add_probe(Gst.PadProbeType.BUFFER,
                  lambda _p, _i: (_telemetry.gpu_exit(stage),
                                  Gst.PadProbeReturn.OK)[1])

_FALLBACK_AFTER_CONSECUTIVE = 3          # GPU failures before permanently falling back
_PULL_TIMEOUT_NS = int(0.25 * Gst.SECOND)  # ample for a 1-3 ms encode after warmup
_APPSRC_MAX_BYTES = 64 * 1024 * 1024     # never block the push; appsink drop keeps it bounded
_NOMINAL_FPS = 25                        # PTS cadence label only; nvjpegenc is not clock-driven
# nvJPEG's quality scale is not byte-equivalent to libjpeg: on a real 1280x720
# CCTV frame, nvjpegenc q65 == cv2 q80 in bytes (~100 KB). Redis bandwidth is a
# known bottleneck at scale, so map the caller's cv2-style jpeg_quality down by
# this offset to keep published JPEG byte sizes comparable to the old cv2 path.
_NVJPEG_QUALITY_OFFSET = 15


class GpuJpegEncoder:
    """GPU-backed JPEG encoder wrapping a private nvjpegenc mini-pipeline.

    The pipeline is built in __init__ (which raises RuntimeError if the required
    elements are missing or the pipeline cannot reach PLAYING — the caller then
    falls back to cv2.imencode for the whole run). encode() is only ever called
    from the publisher worker thread.
    """

    def __init__(self, width, height, quality=80, gpu_id=0, stage="gpu.jpeg"):
        # `stage`: telemetry name of this encoder; with several encoders each needs its own, because
        # the entry/exit probes pair by name and would otherwise mix up their timings
        self._stage = stage
        self._width = int(width)
        self._height = int(height)
        # Map the caller's cv2-style quality onto nvJPEG's scale (see
        # _NVJPEG_QUALITY_OFFSET) so published JPEG bytes stay comparable.
        self._quality = min(max(int(quality) - _NVJPEG_QUALITY_OFFSET, 0), 100)
        self._gpu_id = int(gpu_id)
        self._dead = False
        self._fails = 0
        self._seq = 0
        self._frame_pts = int(1e9 / _NOMINAL_FPS)
        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._loop = None
        self._loop_thread = None
        self._init()

    def _init(self):
        try:
            Gst.init(None)
            pipeline = Gst.Pipeline()
            appsrc = Gst.ElementFactory.make("appsrc", "jpeg_appsrc")
            convert = Gst.ElementFactory.make("nvvideoconvert", "jpeg_convert")
            capsfilter = Gst.ElementFactory.make("capsfilter", "jpeg_caps")
            encoder = Gst.ElementFactory.make("nvjpegenc", "jpeg_enc")
            sink = Gst.ElementFactory.make("appsink", "jpeg_appsink")
            for el, name in ((appsrc, "appsrc"), (convert, "nvvideoconvert"),
                             (capsfilter, "capsfilter"), (encoder, "nvjpegenc"),
                             (sink, "appsink")):
                if el is None:
                    raise RuntimeError(f"element not available: {name}")

            appsrc.set_property("format", Gst.Format.TIME)
            appsrc.set_property("is-live", False)
            appsrc.set_property("max-bytes", _APPSRC_MAX_BYTES)
            appsrc.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw, format=BGR, width={self._width}, "
                f"height={self._height}, framerate={_NOMINAL_FPS}/1"))
            # Force NVMM RGB out of the converter so nvjpegenc gets exactly its
            # required sink caps (NVMM I420 or RGB).
            capsfilter.set_property("caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=RGB"))
            encoder.set_property("quality", self._quality)
            encoder.set_property("gpu-id", self._gpu_id)
            sink.set_property("sync", False)
            sink.set_property("drop", True)
            sink.set_property("max-buffers", 2)
            sink.set_property("wait-on-eos", False)

            _attach_jpeg_probe(encoder, self._stage)

            for el in (appsrc, convert, capsfilter, encoder, sink):
                pipeline.add(el)
            appsrc.link(convert)
            convert.link(capsfilter)
            capsfilter.link(encoder)
            encoder.link(sink)

            # Private main context so this pipeline's iteration never collides
            # with the sidecar's main-loop context. The loop is created BEFORE
            # the thread starts and captured in the closure — referencing
            # self._loop from the thread races its assignment and can kill the
            # thread (None.run()), silently starving the pipeline's context.
            ctx = GLib.MainContext.new()
            loop = GLib.MainLoop(ctx)
            self._loop = loop

            def _run():
                ctx.push_thread_default()
                try:
                    loop.run()
                finally:
                    ctx.pop_thread_default()

            self._loop_thread = threading.Thread(
                target=_run, daemon=True, name="jpeg-gst-loop")
            self._loop_thread.start()

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("gpu jpeg pipeline failed to reach PLAYING")
            self._pipeline = pipeline
            self._appsrc = appsrc
            self._appsink = sink
            # Warm up so the first real frame doesn't pay the async state-change
            # + caps-negotiation cost (or worse, exceed the pull timeout). A
            # black frame is enough; a failure here means the pipeline is
            # unusable, and surfacing it now lets the publisher fall back to
            # cv2.imencode for the whole run.
            if self.encode(np.zeros((self._height, self._width, 3),
                                    dtype=np.uint8),
                           timeout_ns=int(2 * Gst.SECOND)) is None:
                raise RuntimeError("gpu jpeg warmup encode failed")
        except Exception:
            self.close()
            raise

    def encode(self, img, timeout_ns=None) -> "bytes | None":
        """JPEG-encode a CPU BGR frame on the GPU. None => caller falls back.

        timeout_ns overrides the pull timeout (the warmup encode passes a long
        one; steady-state encodes use the module default).
        """
        if self._dead:
            return None
        try:
            frame = np.ascontiguousarray(img)
            buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
            buf.fill(0, frame.tobytes())
            # No explicit NEWSEGMENT needed: appsrc installs a default segment
            # (start=0, stop=none) when none arrives, which is exactly right for
            # a stream of stills nobody waits on.
            buf.pts = self._seq * self._frame_pts
            buf.dts = buf.pts
            self._seq += 1
            ret = self._appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                self._note_failure()
                return None
            # The factory returns a generic GstElement, so go through the signal
            # (try_pull_sample is only reachable on a GstApp.AppSink cast).
            sample = self._appsink.emit("try-pull-sample",
                                        timeout_ns or _PULL_TIMEOUT_NS)
            if sample is None:
                self._note_failure()
                return None
            self._fails = 0
            out = sample.get_buffer()
            data = out.extract_dup(0, out.get_size())  # plain bytes in this binding
            if not data:
                self._note_failure()
                return None
            return bytes(data)
        except Exception as exc:
            logger.debug("gpu jpeg encode failed: %s", exc)
            self._note_failure()
            return None

    def _note_failure(self):
        self._fails += 1
        if self._fails >= _FALLBACK_AFTER_CONSECUTIVE:
            logger.warning("gpu jpeg encoder failing repeatedly (%d); falling back "
                           "to cv2.imencode for the rest of this run", self._fails)
            self.close()
            self._dead = True

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:
                pass
            self._loop = None
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None
