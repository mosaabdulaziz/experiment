"""GStreamer helpers shared by the engine: log filter, element factory, GPU stage telemetry."""

import sys
import logging
from . import telemetry

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


logger = logging.getLogger("iris.sidecar")


# Sentinel returned by _FramePublisher._worker_for after stop(): submissions
# route here and are dropped instead of spawning a new (unjoinable) worker.
_DISCARD_QUEUE = object()  # type: ignore[assignment]



def _install_log_filter():
    """Drop the benign 'Got data flow before segment event' warning that nvurisrcbin's
    internal decoder emits on dynamic linking / file-loop restarts; forward all other
    GLib/GStreamer log messages to the default handler unchanged."""
    def handler(domain, level, message, _user):
        if message and "Got data flow before segment event" in message:
            return
        GLib.log_default_handler(domain, level, message, None)
    try:
        GLib.log_set_default_handler(handler, None)
        return
    except Exception:
        pass  # PyGObject may not bind it; try the C symbol directly below.

    # PyGObject < 3.50 (DeepStream 9.0 ships 3.48) never bound
    # g_log_set_default_handler, so the call above always fails there and the
    # benign warning spams the log. Fall back to the C symbol directly. The
    # callback receives bytes; forward to the PyGObject default handler, which
    # reproduces the usual '(python3:74): GStreamer-WARNING **: ...' line.
    import ctypes as _ctypes

    _GLogFunc = _ctypes.CFUNCTYPE(None, _ctypes.c_char_p, _ctypes.c_uint,
                                  _ctypes.c_char_p, _ctypes.c_void_p)

    @_GLogFunc
    def _ctypes_handler(domain, level, message, _user):
        if message and b"Got data flow before segment event" in message:
            return
        try:
            GLib.log_default_handler(
                domain.decode() if domain else None,
                level,
                message.decode() if message else None,
                None,
            )
        except Exception:
            pass  # never let a logging hiccup crash the pipeline

    try:
        _glib = _ctypes.CDLL("libglib-2.0.so.0")
        _glib.g_log_set_default_handler(_ctypes_handler, None)
        # Keep the callback alive for the process lifetime; ctypes would
        # otherwise release it, leaving GLib with a dangling function pointer.
        _install_log_filter._ctypes_handler = _ctypes_handler
    except Exception as exc:
        sys.stderr.write(f"log filter setup failed (non-fatal): {exc}\n")



def _make(factory, name):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        sys.stderr.write(f"ERROR: could not create element {factory} ({name})\n")
        sys.exit(1)
    return el



# ── GPU-element telemetry (pad probes) ───────────────────────────────────────
# When telemetry is enabled, BUFFER probes are attached around the GPU elements
# so [telemetry] rows report per-element processing latency:
#   gpu.infer.pgie / gpu.infer.sgie_face  nvinfer, per inference batch
#   gpu.decode.<cam>                      decoder inside each nvurisrcbin
#   gpu.jpeg                              nvjpegenc (publisher mini-pipeline)
# The probes call telemetry.gpu_enter/gpu_exit; telemetry stays pure stdlib.

_DECODER_FACTORY_HINTS = ("nvv4l2decoder", "nvdec", "v4l2decoder",
                          "avdec_h264", "avdec_h265")


# Stages that already have probes attached (per-run). Guards against
# double-attaching when _instrument_decode's pad-added handler fires more than
# once for the same source bin — a second probe set would double-count samples.
_gpu_attached: set = set()



def _gpu_probe_stage(element, stage):
    """Time `element` as `stage` via sink/src BUFFER pad probes.

    No-op unless telemetry is enabled — the probes are never installed when
    off, so the live pipeline keeps its exact zero-overhead path. For nvinfer
    the src pad emits one buffer per inference batch, so the measured delta
    is batch latency.
    """
    if not telemetry.enabled() or stage in _gpu_attached:
        return
    sink = element.get_static_pad("sink")
    src = element.get_static_pad("src")
    if sink is None or src is None:
        logger.debug("gpu probe %s skipped (element has no sink/src pad)", stage)
        return
    sink.add_probe(Gst.PadProbeType.BUFFER,
                   lambda _p, _i, s=stage: (telemetry.gpu_enter(s),
                                            Gst.PadProbeReturn.OK)[1])
    src.add_probe(Gst.PadProbeType.BUFFER,
                  lambda _p, _i, s=stage: (telemetry.gpu_exit(s),
                                           Gst.PadProbeReturn.OK)[1])
    _gpu_attached.add(stage)
    logger.debug("gpu probe %s attached", stage)



def _find_element(bin_, factory_hints):
    """Recursively search `bin_` for the first element whose factory name
    contains any of `factory_hints`. Returns None if not found. Runs on the
    streaming thread (pad probes) while the bin may still be mid-construction,
    so any iteration error degrades to "not found" — never an exception."""
    try:
        it = bin_.iterate_elements()
        while True:
            ret, el = it.next()
            if ret != Gst.IteratorResult.OK:
                break
            fac = el.get_factory()
            fname = fac.get_name() if fac is not None else ""
            if any(h in fname for h in factory_hints):
                return el
            if isinstance(el, Gst.Bin):
                found = _find_element(el, factory_hints)
                if found is not None:
                    return found
    except Exception:
        return None
    return None



def _instrument_decode(srcbin, cam_id, attempts=0):
    """Attach gpu.decode.<cam_id> probes to the decoder inside `srcbin`.

    The decoder element only exists once the source bin has reached
    READY/PAUSED, so this installs a one-shot src-pad probe that retries the
    search the moment the first decoded frame flows; while the bin's ghost
    src pad itself does not exist yet (bin still building), it waits on
    ``pad-added`` instead. Gives up after a few attempts (exotic URIs that
    never expose a decoder)."""
    if not telemetry.enabled():
        return
    dec = _find_element(srcbin, _DECODER_FACTORY_HINTS)
    if dec is not None:
        _gpu_probe_stage(dec, f"gpu.decode.{cam_id}")
        return
    if attempts >= 20:
        logger.debug("gpu probe gpu.decode.%s not attached (no decoder found)", cam_id)
        return
    srcpad = srcbin.get_static_pad("src")
    if srcpad is None:
        # Bin is still constructing (ghost pad not yet created) — retry when
        # the pad appears. Connect first, then re-check, to avoid missing the
        # signal in the window between the two calls.
        srcbin.connect("pad-added",
                       lambda _b, _p, s=srcbin, c=cam_id, a=attempts + 1:
                       _instrument_decode(s, c, a))
        return
    holder = {}

    def _retry(_pad, _info):
        srcpad.remove_probe(holder["id"])
        _instrument_decode(srcbin, cam_id, attempts + 1)
        return Gst.PadProbeReturn.OK

    holder["id"] = srcpad.add_probe(Gst.PadProbeType.BUFFER, _retry)
