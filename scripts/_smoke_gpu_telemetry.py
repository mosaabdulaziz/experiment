"""Standalone smoke test for the GPU-element pad-probe telemetry.

Run inside the deepstream container (perception/pipeline/ on sys.path). Exercises the
real sidecar probe helpers against live GStreamer elements WITHOUT touching
the running sidecar pipeline:
  1. _gpu_probe_stage on a plain nvvideoconvert -> [telemetry] gpu.* rows
  2. _instrument_decode self-install on an nvurisrcbin file source -> gpu.decode.*
"""
import glob
import logging
import sys
import time

sys.path.insert(0, "/workspace/iris/sidecar")

import telemetry  # noqa: E402
import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import sidecar  # noqa: E402  (the module under test)


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


lg = logging.getLogger("iris.telemetry")
h = Cap()
lg.addHandler(h)
lg.setLevel(logging.INFO)

telemetry.reset()
telemetry.configure(enabled=True, interval=60.0)
assert telemetry.enabled()

Gst.init(None)

# ── 1) plain element pad-probe instrumentation ───────────────────────────────
pipe = Gst.Pipeline.new("t1")
src = Gst.ElementFactory.make("videotestsrc", "t1_src")
conv = Gst.ElementFactory.make("nvvideoconvert", "t1_conv")
sink = Gst.ElementFactory.make("fakesink", "t1_sink")
for e in (src, conv, sink):
    pipe.add(e)
src.link(conv)
conv.link(sink)
sidecar._gpu_probe_stage(conv, "gpu.test.convert")
pipe.set_state(Gst.State.PLAYING)
time.sleep(1.2)
pipe.set_state(Gst.State.NULL)
telemetry.flush(force=True)

elem_lines = [r for r in h.records if "gpu.test.convert" in r]
print("plain-element lines:", elem_lines)
assert len(elem_lines) == 1 and "thread=gpu" in elem_lines[0], "plain element probe failed"
assert "n=" in elem_lines[0]
print("OK: plain element probe produced:", elem_lines[0][:120])

# ── 2) decode self-install inside nvurisrcbin (file source) ──────────────────
vids = sorted(glob.glob("/workspace/iris/data/uploads/**/*.mp4", recursive=True))
assert vids, "no upload fixtures found"
uri = "file://" + vids[0]
print("decoding:", uri)

bin2 = Gst.ElementFactory.make("nvurisrcbin", "src_smoke")
bin2.set_property("uri", uri)
sink2 = Gst.ElementFactory.make("fakesink", "sink_smoke")
pipe2 = Gst.Pipeline.new("t2")
pipe2.add(bin2)
pipe2.add(sink2)
# _instrument_decode BEFORE the pipeline exists/starts, like build_pipeline does.
sidecar._instrument_decode(bin2, "smoke-cam")
# nvurisrcbin's ghost src pad appears only when its internal source is built,
# so link via pad-added exactly like build_pipeline's _on_pad_added.
bin2.connect("pad-added", lambda _b, pad: pad.link(sink2.get_static_pad("sink")))
pipe2.set_state(Gst.State.PLAYING)
time.sleep(3.0)
pipe2.set_state(Gst.State.NULL)
telemetry.flush(force=True)

dec_lines = [r for r in h.records if "gpu.decode" in r]
print("decode lines:", dec_lines)
assert len(dec_lines) == 1 and "thread=gpu" in dec_lines[0], "decode self-install failed"
assert "smoke-cam" in dec_lines[0]
print("OK: decode self-install produced:", dec_lines[0][:120])

print("SMOKE PASS")
