#!/usr/bin/env python3
"""Standalone Smart Video Record probe — isolates SVR from the IRIS pipeline.

Builds ONE nvurisrcbin on a given rtsp:// URI with exactly the smart-record
properties sidecar.py sets, waits for the cache to fill, emits `start-sr`, and
reports whether a session opened and a file landed.

Run it inside the DeepStream container:

    docker compose exec deepstream-sidecar python3 perception/pipeline/svr_probe.py \
        "rtsp://user:pass@host:554/stream1"

Add DeepStream's own logging to see what NvDsSRStart does internally:

    docker compose exec -e GST_DEBUG='*nvurisrc*:6,*smart*:6,*nvdssr*:6' \
        deepstream-sidecar python3 perception/pipeline/svr_probe.py "rtsp://..." 2>&1 | tail -80

Why this exists: `start-sr` is an action signal returning void, so the caller
cannot tell a no-op from a success, and this pyds build exposes no Smart Record
bindings to introspect with. When the in-pipeline path fails, the only way to
tell an IRIS integration bug from a DeepStream/camera issue is to reproduce it
with nothing else in the graph.
"""
import ctypes
import os
import sys
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def _start_sr_forms(session):
    """Candidate marshallings for start-sr's sessionId out-parameter.

    DeepStream asserts `sessionId != NULL` inside NvDsSRStart, and PyGObject
    refuses a plain Python int ("could not convert type int to gpointer"), so the
    argument has to be a real pointer OBJECT. PyGObject's documented gpointer path
    accepts a PyCapsule, which can be built around a ctypes buffer through the
    CPython API. The int/None forms are kept last only to prove they still fail.
    """
    forms = []
    try:
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        forms.append(("pycapsule", PyCapsule_New(ctypes.addressof(session), None, None)))
    except Exception as exc:
        print(f"  (PyCapsule form unavailable: {exc})")
    forms.append(("ctypes-byref", ctypes.byref(session)))
    forms.append(("ctypes-pointer", ctypes.pointer(session)))
    forms.append(("ctypes-address", ctypes.addressof(session)))
    forms.append(("none", None))
    return forms


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    uri = sys.argv[1]
    cache = int(os.environ.get("SVR_CACHE", "15"))
    duration = int(os.environ.get("SVR_DURATION", "10"))
    start_time = int(os.environ.get("SVR_START_TIME", "5"))
    # 0 = MP4 (qtmux, what sidecar.py ships), 1 = MKV (matroskamux). qtmux
    # refuses any caps change after a track is configured, so this knob is how
    # you test whether a source whose caps change mid-stream can be recorded
    # at all -- see the go2rtc SDP-vs-in-band parameter-set finding.
    container = int(os.environ.get("SVR_CONTAINER", "0"))
    out_dir = os.environ.get("SVR_DIR", "/tmp/svr_probe")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass

    Gst.init(None)
    pipeline = Gst.Pipeline()
    src = Gst.ElementFactory.make("nvurisrcbin", "src")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    if not src or not sink:
        print("FAIL: could not create nvurisrcbin/fakesink")
        return 1
    sink.set_property("sync", 0)

    src.set_property("uri", uri)
    src.set_property("type", 2)                      # rtsp
    src.set_property("smart-record", 2)              # cloud + local events
    src.set_property("smart-rec-cache", cache)
    src.set_property("smart-rec-container", container)   # 0 MP4 / 1 MKV
    src.set_property("smart-rec-mode", 1)            # video only
    src.set_property("smart-rec-dir-path", out_dir)
    src.set_property("smart-rec-file-prefix", "probe_")
    src.set_property("smart-rec-default-duration", duration)

    print("--- properties as the element reports them ---")
    for p in ("type", "smart-record", "smart-rec-cache", "smart-rec-container",
              "smart-rec-mode", "smart-rec-dir-path", "smart-rec-file-prefix",
              "smart-rec-default-duration", "smart-rec-status"):
        try:
            print(f"  {p:28} {src.get_property(p)}")
        except Exception as exc:
            print(f"  {p:28} <unreadable: {exc}>")

    done = {"fired": False, "info": None}

    def _on_sr_done(_bin, *args):
        done["fired"] = True
        done["info"] = args
        print(f"*** sr-done fired, args={args!r}")
    src.connect("sr-done", _on_sr_done)

    def _on_pad_added(_bin, pad):
        if not pad.is_linked():
            pad.link(sink.get_static_pad("sink"))
    src.connect("pad-added", _on_pad_added)

    pipeline.add(src)
    pipeline.add(sink)
    pipeline.set_state(Gst.State.PLAYING)

    bus = pipeline.get_bus()

    def pump(seconds, label):
        deadline = time.time() + seconds
        while time.time() < deadline:
            msg = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"[{label}] BUS ERROR: {err} | {dbg}")
                elif msg.type == Gst.MessageType.WARNING:
                    err, dbg = msg.parse_warning()
                    print(f"[{label}] bus warning: {err}")
            GLib.MainContext.default().iteration(False)

    print(f"\n--- filling the {cache}s cache (waiting {cache + 5}s) ---")
    pump(cache + 5, "warmup")
    print("  smart-rec-status before start:", src.get_property("smart-rec-status"))

    session = ctypes.c_uint32(0)
    forms = [(label, arg) for label, arg in _start_sr_forms(session)]
    opened = None
    for label, arg0 in forms:
        try:
            src.emit("start-sr", arg0, start_time, duration, None)
        except Exception as exc:
            print(f"  start-sr({label}) RAISED: {type(exc).__name__}: {exc}")
            continue
        status = src.get_property("smart-rec-status")
        print(f"  start-sr({label}) -> smart-rec-status={status} session={session.value}")
        if status:
            opened = label
            break
        # Status may be set asynchronously on the streaming thread rather than
        # inside the signal handler -- give it a beat before calling it a failure.
        pump(2, "post-emit")
        status = src.get_property("smart-rec-status")
        if status:
            opened = f"{label} (after 2s)"
            print(f"  ...status went true late: {opened}")
            break

    if opened:
        print(f"\nSESSION OPENED via arg0={opened}")
    else:
        print("\nNO SESSION OPENED by any form")

    wait = duration + 15
    print(f"--- waiting {wait}s for the recording to finalize ---")
    pump(wait, "record")

    files = sorted(os.listdir(out_dir))
    print(f"\n--- result ---")
    print(f"  sr-done fired : {done['fired']}")
    print(f"  files in {out_dir}: {files}")
    for f in files:
        print(f"    {f}  {os.path.getsize(os.path.join(out_dir, f))} bytes")
    pipeline.set_state(Gst.State.NULL)
    return 0 if files else 1


if __name__ == "__main__":
    sys.exit(main())
