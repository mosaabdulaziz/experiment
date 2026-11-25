#!/usr/bin/env python3
"""Probe the maximum number of concurrent NVENC encode sessions this GPU and
driver accept inside a single process.

Why: nvv4l2h264enc opens one NVENC session per instance, and how many
simultaneous sessions the hardware/driver allows is GPU-specific (measured ~12
on an RTX 5050 laptop; ~8 on older consumer parts; more on datacenter GPUs).
Opening more than the ceiling is dangerous in TWO ways: sometimes the driver
cleanly rejects the extra session (an S_FMT bus error — recoverable), but
sometimes it wedges the whole NVENC engine (sessions open but never encode a
frame, freezing the entire pipeline). A configured session cap would be wrong
on any other GPU, so the limit must be measured at runtime.

This script measures it on a THROWAWAY pipeline, in a separate process, so a
wedge during probing cannot touch the real pipeline. It mirrors the sidecar's
serialized branch creation: one videotestsrc -> nvvideoconvert -> nvv4l2h264enc
-> fakesink branch at a time, each brought to PLAYING synchronously, and waits
for the branch's encoder to actually ENCODE its first frame (the S_FMT /
session activation happens on the first frame, which is where the driver
rejects or wedges). A branch that opens but never encodes within the timeout
means we are past the ceiling.

Protocol on stdout:
  OK <n>       after branch n (1-based) encoded its first frame
  LIMIT <n>    branch n (1-based) failed to encode within the timeout
  DONE <n>     all requested branches encoded

Exit is via os._exit() so a wedged driver cannot hang the parent on a clean
GStreamer teardown.

Usage: nvenc_probe.py <max_branches> [--timeout-ms 4000]
"""
import os
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
TIMEOUT_MS = 4000
if "--timeout-ms" in sys.argv:
    TIMEOUT_MS = int(sys.argv[sys.argv.index("--timeout-ms") + 1])


def main():
    Gst.init(None)
    pipe = Gst.Pipeline()
    pipe.set_state(Gst.State.PLAYING)  # empty pipeline: no-op, legal
    counters = []

    def make_branch(i):
        src = Gst.ElementFactory.make("videotestsrc", f"vsrc_{i}")
        src.set_property("is-live", False)
        cvt = Gst.ElementFactory.make("nvvideoconvert", f"cvt_{i}")
        enc = Gst.ElementFactory.make("nvv4l2h264enc", f"enc_{i}")
        enc.set_property("bitrate", 1000 * 1000)
        sink = Gst.ElementFactory.make("fakesink", f"sink_{i}")
        sink.set_property("sync", False)
        c = {"n": 0}

        def _count(_pad, _info, _c=c):
            _c["n"] += 1
            return Gst.PadProbeReturn.OK

        sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, _count)
        for el in (src, cvt, enc, sink):
            pipe.add(el)
        src.link(cvt)
        cvt.link(enc)
        enc.link(sink)
        return (src, cvt, enc, sink), c

    for i in range(N):
        els, c = make_branch(i)
        counters.append(c)
        # Serialized open: sync_state_with_parent blocks until this branch
        # reaches PLAYING (the NVENC session open for the encoder).
        for el in els:
            ret = el.sync_state_with_parent()
            if ret == Gst.StateChangeReturn.FAILURE:
                print(f"LIMIT {i + 1}", flush=True)
                os._exit(0)
        # Wait for this branch's encoder to actually encode a frame. If the
        # driver wedged (session opened but never encodes), the whole engine
        # stalls and no counter advances — treat as the ceiling.
        deadline = time.monotonic() + TIMEOUT_MS / 1000.0
        while time.monotonic() < deadline:
            if c["n"] > 0:
                print(f"OK {i + 1}", flush=True)
                break
            time.sleep(0.1)
        else:
            print(f"LIMIT {i + 1}", flush=True)
            os._exit(0)

    print(f"DONE {N}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
