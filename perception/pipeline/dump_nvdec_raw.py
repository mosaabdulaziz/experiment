#!/usr/bin/env python3
"""Dump raw NVDEC-decoded frames from RTSP sources, bypassing nvstreammux.

For each camera, builds a standalone GStreamer pipeline:
  nvurisrcbin → tee → nvvideoconvert → videoconvert → appsink

Captures N raw frames at native resolution and saves as high-quality JPEGs.
Compare these with the DeepStream pipeline frames to isolate whether
corruption happens in the decoder or the mux.

Usage (inside sidecar container):
    python3 perception/pipeline/dump_nvdec_raw.py --count 30

Or from host (needs GStreamer + NVDEC):
    python3 perception/pipeline/dump_nvdec_raw.py --count 30
"""
import argparse
import os
import sys
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstApp, GLib, GstVideo  # noqa: E402

Gst.init(None)

CAMERAS = {
    "cam-1": "rtsp://go2rtc:8554/raw-cam-1",
    "cam-2": "rtsp://go2rtc:8554/raw-cam-2",
    "cam-3": "rtsp://go2rtc:8554/raw-cam-3",
}

OUT_DIR = Path("/workspace/iris/data/nvdec_raw")


def dump_frames_for_source(rtsp_url, cam_id, count, out_dir):
    """Build a pipeline, capture `count` decoded frames, save as JPEG."""
    cam_dir = out_dir / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Gst.Pipeline()

    # Source with same tuning as the sidecar
    src = Gst.ElementFactory.make("nvurisrcbin", f"src_{cam_id}")
    src.set_property("uri", rtsp_url)
    try:
        src.set_property("rtsp-reconnect-interval", 10)
        src.set_property("latency", 100)
        src.set_property("drop-on-latency", True)
        src.set_property("select-rtp-protocol", 0x4)  # TCP only
        src.set_property("udp-buffer-size", 2097152)
        src.set_property("num-extra-surfaces", 2)
    except Exception:
        pass

    # NVMM → system-memory RGBA → RGB for CPU capture
    conv_nvmm = Gst.ElementFactory.make("nvvideoconvert", f"conv_{cam_id}")
    # nvbuf-memory-type=0 (NVBUF_MEM_DEFAULT) ensures system-memory output
    # when downstream (videoconvert) can't accept NVMM
    conv_nvmm.set_property("nvbuf-memory-type", 0)

    # videoconvert: RGBA (system) → RGB
    vconv = Gst.ElementFactory.make("videoconvert", f"vconv_{cam_id}")

    # Force RGB output
    rgb_caps = Gst.ElementFactory.make("capsfilter", f"rgb_caps_{cam_id}")
    rgb_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))

    # appsink for CPU frame capture
    sink = Gst.ElementFactory.make("appsink", f"sink_{cam_id}")
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers", 2)
    sink.set_property("drop", True)

    for el in (src, conv_nvmm, vconv, rgb_caps, sink):
        pipeline.add(el)
    src.link(conv_nvmm)
    conv_nvmm.link(vconv)
    vconv.link(rgb_caps)
    rgb_caps.link(sink)

    # Frame counter
    saved = [0]
    frames_info = []

    def on_new_sample(appsink):
        if saved[0] >= count:
            return Gst.FlowReturn.OK
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Parse resolution from caps
        struct = caps.get_structure(0)
        w = struct.get_value("width")
        h = struct.get_value("height")

        # Map buffer
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            import numpy as np
            import cv2

            # appsink gives us video/x-raw,format=RGB after videoconvert
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
            # Save as JPEG
            fname = cam_dir / f"nvdec_raw_{saved[0]:04d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            pts = buf.pts / Gst.SECOND if buf.pts != Gst.CLOCK_TIME_NONE else 0
            frames_info.append({
                "file": str(fname),
                "width": w,
                "height": h,
                "pts": pts,
            })
            saved[0] += 1
            if saved[0] % 10 == 0:
                print(f"  [{cam_id}] saved {saved[0]}/{count} frames ({w}x{h})")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    sink.connect("new-sample", on_new_sample)

    # Set up a bus to watch for errors
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"  [{cam_id}] ERROR: {err.message}")
            if debug:
                print(f"    {debug}")
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    bus.connect("message", on_message)

    # Start
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print(f"  [{cam_id}] Failed to set pipeline to PLAYING")
        pipeline.set_state(Gst.State.NULL)
        return 0

    print(f"  [{cam_id}] Pipeline PLAYING, waiting for frames...")

    # Run until we have enough frames or timeout
    deadline = time.time() + 60  # 60s max per camera
    while saved[0] < count and time.time() < deadline:
        try:
            loop.iteration(timeout=0.5)
        except Exception as e:
            print(f"  [{cam_id}] loop error: {e}")
            break

    pipeline.set_state(Gst.State.NULL)

    if saved[0] == 0:
        print(f"  [{cam_id}] WARNING: Zero frames captured. Checking pipeline state...")
        # Try once more with a simple pipeline to verify NVDEC works
        print(f"  [{cam_id}] Trying simple nvurisrcbin → nvvideoconvert → videoconvert → fakesink...")
        simple = Gst.Pipeline()
        s_src = Gst.ElementFactory.make("nvurisrcbin", "s_src")
        s_src.set_property("uri", rtsp_url)
        s_src.set_property("select-rtp-protocol", 0x4)
        s_conv = Gst.ElementFactory.make("nvvideoconvert", "s_conv")
        s_vconv = Gst.ElementFactory.make("videoconvert", "s_vconv")
        s_sink = Gst.ElementFactory.make("fakesink", "s_sink")
        for el in (s_src, s_conv, s_vconv, s_sink):
            simple.add(el)
        s_src.link(s_conv)
        s_conv.link(s_vconv)
        s_vconv.link(s_sink)
        simple.set_state(Gst.State.PLAYING)
        time.sleep(3)
        simple.set_state(Gst.State.NULL)
        print(f"  [{cam_id}] Simple pipeline test done")

    if saved[0] > 0:
        print(f"  [{cam_id}] Done: {saved[0]}/{count} frames saved to {cam_dir}")
        # Print resolution info
        if frames_info:
            w, h = frames_info[0]["width"], frames_info[0]["height"]
            print(f"  [{cam_id}] Native resolution: {w}x{h}")
    else:
        print(f"  [{cam_id}] No frames captured!")

    return saved[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=30,
                   help="Number of frames per camera (default: 30)")
    p.add_argument("--cam", nargs="*", default=None,
                   help="Cameras to capture (default: all)")
    p.add_argument("--out", default=str(OUT_DIR),
                   help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    cameras = args.cam or list(CAMERAS.keys())

    print(f"Dumping {args.count} raw NVDEC frames per camera to {out_dir}")
    print(f"Cameras: {cameras}")
    print()

    for cam_id in cameras:
        if cam_id not in CAMERAS:
            print(f"Unknown camera: {cam_id}")
            continue
        url = CAMERAS[cam_id]
        print(f"[{cam_id}] Capturing from {url}...")
        n = dump_frames_for_source(url, cam_id, args.count, out_dir)
        print()

    print(f"\nDone! Frames saved to {out_dir}")
    print("Compare these NVDEC-decoded frames with the pipeline frames to isolate corruption.")


if __name__ == "__main__":
    main()
