"""The hot-path buffer probe and the helpers it calls."""

import os
import time
import json
import pyds
import redis
import cv2
import numpy as np
from .sources import preview_plan
from . import telemetry

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .frame import FrameOutput, FrameView, ObjectView, detection_rows
from .gst_util import logger
from .modules import ModuleSet
from .objects import extract_objects
from .svr import _maybe_start_svr
from .segment_index import arrival_wall




class ProbeCtx:
    def __init__(self, gcfg, states, index_to_id, publisher=None):
        self.g = gcfg
        self.states = states          # {pad_index -> StreamState}
        self.index_to_id = index_to_id
        self.publisher = publisher    # Optional[_FramePublisher]
        self.gie_names = {}           # nvinfer unique component id -> element name, set by build_pipeline
        self.modules = ModuleSet([])  # the configured modules, set by build_pipeline



_extract_objects = telemetry.timeit("probe.frame_objects")(extract_objects)


def _map_frame_bgr(gst_buffer, frame_meta, gcfg):
    """Copy THIS frame's slice of the batch surface to a CPU BGR array.

    Runs on the streaming thread: the NVMM buffer cannot safely outlive the probe
    callback, so the GPU->CPU copy happens here — once per rendered frame. The
    publisher worker and every module that wants pixels then share this single copy.
    """
    surf = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    frame = np.array(surf, copy=True, order="C")
    frame = frame[:gcfg.capture_h, :gcfg.capture_w, :]  # crop any pitch padding
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)



@telemetry.timeit("probe")
def probe(pad, info, ctx):
    """
    GstPadProbe callback on the capsfilter after the tracker. Reads each frame's metadata
    and publishes per-source detections + annotated JPEGs to Redis.

    The probe is called once per batch, so we iterate the batch's frames and process each
    frame's metadata individually. We do not assume that the batch is complete or that all
    sources are present in every batch — some sources may be paused, disconnected, or
    unavailable.

    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    now = time.monotonic()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        st = ctx.states.get(frame_meta.pad_index)
        if st is None:
            l_frame = l_frame.next
            continue
        # A frame of this source made it through the mux and inference: the
        # stall watch counts it here, after the mux, because a source can stall
        # on either side of it (decoder starved, or frames swallowed by the mux).
        ctx.g.stall_watch.note_buffer(frame_meta.pad_index, now)
        st._last_frame_t = now         # the fps gauge reads 0 once this goes stale

        # per-source FPS (log roughly every 5s of wall-clock). When processing is
        # throttled (infer_interval > 1) the log also reports processed-fps, so the
        # realtime source rate and the analysis rate are both visible.
        st._fcount = getattr(st, "_fcount", 0) + 1
        st._last_frame_mono = now      # read by the stalled-source watchdog
        st._frames_total = getattr(st, "_frames_total", 0) + 1   # frames that got through inference
        _fps_t0 = getattr(st, "_fps_t0", None)
        if _fps_t0 is None:
            st._fps_t0 = now
        elif now - _fps_t0 >= 5.0:
            fps = st._fcount / (now - _fps_t0)
            st._fps_now = fps          # sampled by the telemetry fps_<id> gauge
            if ctx.g.infer_interval > 1:
                logger.info("[%s] %.1f fps (process %.1f fps)", st.id, fps,
                            getattr(st, "_pwin", 0) / (now - _fps_t0))
                st._pwin = 0
            else:
                logger.info("[%s] %.1f fps", st.id, fps)
            st._fps_t0 = now
            st._fcount = 0

        pts = frame_meta.buf_pts / 1e9

        if (ctx.g.clock_offsets.needs_probe(st.id)
                and frame_meta.buf_pts != Gst.CLOCK_TIME_NONE):
            # Wall clock, not the monotonic `now` above -- the recorder side samples
            # time.time() and the two constants must be comparable. Stops once frozen.
            # An unset buf_pts would freeze a garbage offset for the whole run, so it
            # is skipped rather than observed (mirrors _sample_pts on the other side).
            # The frame's arrival at the mux, not the probe's own time: see arrival_wall.
            ctx.g.clock_offsets.observe_probe(st.id, pts, arrival_wall(
                frame_meta.ntp_timestamp, not ctx.g.rtcp_camera_clock, time.time()))

        # Processing throttle: only every Nth frame (N = infer_interval, from
        # process_fps) runs the HEAVY analysis pass (zones, faces, meta feed).
        # The preview publish is deliberately NOT gated on it: nvinfer's
        # `interval` skips the inference itself while propagating its last
        # detections to the skipped batches, so the probe can still draw boxes
        # and publish a smooth full-rate preview while inference runs at
        # process_fps. (analysis_fps — the videorate pre-mux cap — is the
        # alternative: it also reduces tracker/convert work but caps the preview
        # at the analysis rate; see sidecar.yaml.) Skipped frames are counted
        # for the realtime fps log above, then only the cheap preview path runs.
        analyze = True
        if ctx.g.infer_interval > 1:
            st._pcount = getattr(st, "_pcount", 0) + 1
            if st._pcount % ctx.g.infer_interval != 0:
                analyze = False
            else:
                st._pwin = getattr(st, "_pwin", 0) + 1

        if analyze and ctx.g.pts_parity_logging:
            _pt0 = getattr(st, "_parity_t0", 0.0)
            if now - _pt0 >= 5.0:
                st._parity_t0 = now
                logger.info("[%s] parity probe pts-wall=%.3f", st.id, pts - time.time())

        # One FrameView per frame; its objects are read off the metadata once, the first time anything asks, and
        # shared by every module and the detections pass below. What a module needs read off each object's metadata
        # (its `extract` hook) is asked at that first use, so it can depend on what the frame turned out to be.
        view = FrameView(
            stream_id=st.id, state=st, pad_index=frame_meta.pad_index, frame_num=frame_meta.frame_num,
            pts=pts, now=now, buf_pts=frame_meta.buf_pts, analyze=analyze, mux_size=(ctx.g.mux_w, ctx.g.mux_h),
            arrived=arrival_wall(frame_meta.ntp_timestamp, not ctx.g.rtcp_camera_clock, time.time()),
            capture_size=(ctx.g.capture_w, ctx.g.capture_h), frame_meta=frame_meta)
        view._objects_loader = lambda fm=frame_meta, v=view: _extract_objects(
            fm, ctx.gie_names, extractors=ctx.modules.collect("extract", v))

        # meta (the detection feed) only advances on analyzed frames —
        # dets only refresh at the inference cadence. The preview (need_stream)
        # runs every frame regardless, which is what keeps the site smooth —
        # unless pipeline_preview is off (inference-only mode), where preview
        # frames are skipped entirely and only the meta cadence does work.
        # A module may own the sampling clock (the face detector's gate decides, before it runs, which frames it
        # runs on, and a frame it sampled has faces); otherwise the engine's own interval does.
        sampled = ctx.modules.first("sample", view)
        if sampled is None:
            sampled = (st._last_pts is None
                       or (pts - st._last_pts) >= ctx.g.sample_interval
                       or pts < st._last_pts)
        need_meta = analyze and sampled

        # Logic that must see every frame, ahead of the early exit below and NOT gated on `analyze` (alerting
        # runs on every frame whether or not this one is an inference cycle or has a publish due), is the
        # modules' on_frame hook.
        out = FrameOutput()
        view.need_meta = need_meta
        ctx.modules.run("on_frame", view, out)

        need_stream = (now - st._last_stream) >= ctx.g.stream_interval
        do_preview = need_stream and ctx.g.pipeline_preview
        if not (need_meta or do_preview):
            l_frame = l_frame.next
            continue

        # Collect THIS frame's detections + boxes, with two deliberately different gates:
        #   - dets (analytics/alert + description feed): detector-confident objects only.
        #   - boxes: detector-confident objects always; shadow tracks the detector missed
        #     this frame (DeepStream reports conf -0.1 while nvtracker keeps localizing them
        #     up to maxShadowTrackingAge) only when draw_shadow_tracks is on. Gating boxes on
        #     detector confidence alone made them flicker off/on as detection dropped in and
        #     out; drawing from the tracker's bbox keeps them stable. For shadow tracks we
        #     show the tracker's own confidence so the label isn't a bogus negative number.
        with telemetry.stage("probe.dets"):
            draw_shadow = ctx.g.draw_shadow_tracks
            for o in view.objects:
                label = o.label
                conf = o.conf
                if ctx.g.relevant is None or label in ctx.g.relevant:
                    detected = conf >= ctx.g.conf_threshold
                    # Detector-confident objects are published (the meta feed). Shadow tracks (tracker-predicted,
                    # detector-missed) are only drawn -- unless a module that follows objects needs them in the feed
                    # (it sets `publish` itself in transform).
                    o.draw = detected or draw_shadow
                    o.publish = detected
                else:
                    o.draw = o.publish = False       # a label this product does not use

        # Modules that change which detections there are and which count (box stabilizer, filter zones), in the
        # order they are listed. Everything below -- the meta feed, pixels, the analyze hooks -- sees the result.
        ctx.modules.run("transform", view)
        dets = detection_rows(view.objects, full=view.dets_with_boxes)

        # ── Offload render + publish to the worker thread ───────────────────
        # The probe does a single CPU copy of this frame's surface (buffer
        # lifetime forces that here), then hands drawing, JPEG encode, and the
        # Redis writes to the publisher worker — cv2.imencode at stream_fps and
        # the network round-trips must not run on the streaming thread.
        if ctx.publisher is None:
            l_frame = l_frame.next
            continue
        # Surface copy only when something needs the CPU frame: the annotated preview (pipeline_preview) or a module
        # (crops of faces, of bodies). Otherwise frame_bgr stays None and the whole draw/encode path is skipped in
        # the publisher.
        _, need_bgr = preview_plan(need_meta, need_stream, ctx.g.pipeline_preview)
        view.need_stream = do_preview
        need_bgr = ctx.modules.any("wants_pixels", view) or need_bgr
        if need_bgr:
            try:
                with telemetry.stage("probe.map_bgr"):
                    frame_bgr = _map_frame_bgr(gst_buffer, frame_meta, ctx.g)
            except Exception as exc:
                logger.warning("[%s] surface map failed (skipping frame): %s", st.id, exc)
                l_frame = l_frame.next
                continue
        else:
            frame_bgr = None
        if frame_bgr is not None:
            ctx.modules.run("on_pixels", view, frame_bgr)

        if need_meta:
            st._last_pts = pts
        if need_stream:
            st._last_stream = now
        view.bgr = frame_bgr
        ctx.modules.run("analyze", view, out)
        ctx.publisher.submit({
            "annotations": out.annotations, "fields": out.fields,
            "st": st, "frame_bgr": frame_bgr,
            "boxes": sorted((o for o in view.objects if o.draw), key=lambda o: o.layer),
            "need_meta": need_meta, "need_stream": do_preview,
            "frame_num": frame_meta.frame_num, "pts": pts,
            # Camera clock (issue 1 Phase 2): the camera's own capture time
            # from RTCP, ns; 0 = none. Only meaningful with rtcp_camera_clock
            # on -- otherwise nvstreammux fills it with THIS machine's clock,
            # which must not be passed off as the camera's.
            "ntp_ts": frame_meta.ntp_timestamp if ctx.g.rtcp_camera_clock else 0,
            "dets": dets, "publish_hooks": out.publish_hooks, "draws": out.draws,
        })

        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK
