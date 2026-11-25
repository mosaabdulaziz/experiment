"""Builds the GStreamer pipeline: sources, mux, inference chain, tracker, module elements, sinks."""

import os
import sys
import time
import pyds
import redis
from .sources import make_stream_state
from .engine_path import assert_distinct_engine_paths, gie_engine_path
from .recorder_kind import recorder_kind
from . import telemetry

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .modules import EngineServices, ModuleConfigError, PipelineHandle, camera_keyed_modules, load_modules
from .svr import _maybe_start_svr
from .gst_util import _gpu_probe_stage, _instrument_decode, _make, logger
from .probe import ProbeCtx, probe
from .publisher import _FramePublisher
from .recording import _RecordBranchScheduler, _SegmentUploader
from .svr import _configure_svr_source




def build_pipeline(cfg, gcfg, sources, all_sources, role):
    """Build the DeepStream pipeline for all sources, returning the GstPipeline object.

    `sources` is what THIS process owns (post role/shard filtering) and drives
    every local index -- streammux pads included. `all_sources` is the full,
    unfiltered list and `role` names this process's slice; modules get both
    (EngineServices) for config that the app indexes by the full list.
    The pipeline is:
        nvstreammux -> nvinfer -> nvtracker -> [secondary nvinfer] -> [module elements] -> nvvideoconvert -> capsfilter -> fakesink

    nvstreammux batches all sources into one stream,
    nvinfer runs the YOLO model,
    nvtracker tracks objects across frames,
    module elements (optional, from `modules:`) add what a capability needs, e.g. zone/ROI occupancy,
    nvvideoconvert converts the video format,
    capsfilter enforces the desired video format and resolution,
    and the capsfilter/fakesink pair allows us to probe the output frames and extract metadata for each source individually.

    Returns (pipeline, ctx, uploader, publisher, record_scheduler) — uploader is the
    segment-upload worker the record branches submit finalized fragments to;
    publisher is the frame render/encode/publish worker the probe submits to.
    main() stops the workers on shutdown so pending work drains before exit.
    """
    Gst.init(None)
    pipeline = Gst.Pipeline()

    streammux = _make("nvstreammux", "mux")
    # Cap the inference batch at max_pgie_batch (default 160). The yolo11s INT8
    # engine cannot be BUILT past ~b160 on this GPU — TensorRT rejects larger
    # batches with "CASK5 CONV offset must not exceed int32_max" (observed at
    # b192) — and the engine batch must equal the pgie batch, so sources beyond
    # the cap simply batch in multiple mux pushes (batched-push-timeout flushes
    # partial batches). This is the documented DeepStream pattern for running
    # more sources than the engine's batch size.
    max_batch = int(cfg.get("max_pgie_batch", 160))
    batch = max(1, min(len(sources), max_batch))
    streammux.set_property("batch-size", batch)
    streammux.set_property("width", int(cfg.get("mux_width", 1280)))
    streammux.set_property("height", int(cfg.get("mux_height", 720)))
    if cfg.get("mux_interpolation_method") is not None:
        streammux.set_property("interpolation-method", int(cfg["mux_interpolation_method"]))
    # batched-push-timeout quantizes when the mux closes a batch. With the
    # analysis branch capped to analysis_fps (period P), a timeout that does NOT
    # divide P rounds every source's delivery up to a whole number of timeout
    # cycles: e.g. af=10 (100 ms period) with the old 40 ms timeout -> 3 cycles
    # = 120 ms -> 8.3 fps measured instead of 10 (the ~1-frame-in-6 loss the
    # capacity benchmark documents). Set the timeout to P/4 (an exact divisor)
    # so the cadence is preserved at every stream count; keep the 40 ms default
    # for the uncapped full-rate pipeline. Overridable via mux_push_timeout_us.
    af = gcfg.analysis_fps
    override = cfg.get("mux_push_timeout_us")
    if override:
        push_timeout = int(override)
    elif af > 0:
        push_timeout = max(int(1_000_000 / (4 * af)), 5000)
    else:
        push_timeout = 40000
    streammux.set_property("batched-push-timeout", push_timeout)
    logger.info("streammux batched-push-timeout=%dus (analysis_fps=%s)",
                push_timeout, af)
    # Time-align one frame per source per batch. With sync-inputs=0, two
    # independently-paced sources drift and the mux emits malformed batches —
    # frames get cross-attributed between sources (a tile flickers between
    # cameras' content; the box/det metadata stays correct because it keys off
    # pad_index, only the surface pixels are scrambled). sync-inputs=1 makes
    # the mux wait for one frame per source and produces well-formed batches.
    # Known caveats that motivated the default below, worth re-testing per DS
    # version: (1) mixed file+RTSP pipelines can stall on live clocks (observed
    # with 4 files + 2 RTSP); (2) sync-inputs=1 + videorate was seen to
    # DEADLOCK (videorate re-times PTS), but on DS 9.0 with all-RTSP sources
    # + analysis_fps=10 it flows fine at the cap (verified 2026-08).
    # mux_sync_inputs: 0/1 overrides the default — set 1 for all-RTSP
    # deployments to fix the cross-stream flicker.
    has_rtsp = any(s["uri"].startswith("rtsp://") for s in sources)
    override_sync = cfg.get("mux_sync_inputs")
    sync_inputs = int(override_sync) if override_sync is not None else (0 if (has_rtsp or gcfg.analysis_fps > 0) else 1)
    try:
        streammux.set_property("sync-inputs", sync_inputs)
        logger.info("streammux sync-inputs=%d (has_rtsp=%s analysis_fps=%s)",
                    sync_inputs, has_rtsp, gcfg.analysis_fps)
    except Exception as exc:
        sys.stderr.write(f"sync-inputs not settable (non-fatal): {exc}\n")
    # Camera clock from RTCP (issue 1 Phase 2, 2026-09-12). An RTSP camera can
    # send "RTCP Sender Reports" alongside its video: small side-channel
    # messages saying "this frame was captured at <my own clock's time>". By
    # default (attach-sys-ts=TRUE) nvstreammux ignores those and stamps every
    # frame's ntp_timestamp with THIS machine's clock instead. FALSE keeps the
    # camera's own time when the camera provides it, and leaves ntp_timestamp
    # at 0 when it doesn't -- the "no camera clock, use the server's current
    # time" signal a consumer falls back on. Nothing else in this
    # file reads ntp_timestamp, so this changes nothing else. Pairs with
    # configure_source_for_ntp_sync() on each RTSP source below (both halves
    # are needed). Process-start-only.
    # NOTE: a source pulled through go2rtc never receives Sender Reports --
    # go2rtc's RTSP server relays RTP only (checked in its v1.9.14 source) --
    # so only a uri pointing straight at a camera can yield a camera clock;
    # every other source falls back to server time.
    rtcp_camera_clock = gcfg.rtcp_camera_clock
    if rtcp_camera_clock:
        streammux.set_property("attach-sys-ts", False)
    logger.info("streammux attach-sys-ts=%s (rtcp_camera_clock=%s)",
                not rtcp_camera_clock, rtcp_camera_clock)
    pipeline.add(streammux)

    uploader = _SegmentUploader(gcfg) if gcfg.record_enabled else None
    publisher = _FramePublisher(gcfg)

    if not gcfg.pipeline_preview:
        logger.info("preview mode: inference-only (pipeline_preview=false) — "
                    "no annotated JPEGs published; frontend plays raw streams "
                    "via iris:status:<id> preview_uri and overlays iris:meta")
        for s in sources:
            if s["uri"].startswith("file://"):
                logger.warning(
                    "[%s] pipeline_preview=false but source is file:// — no raw "
                    "stream exists for it; preview will be unavailable unless "
                    "the upload is also served over RTSP (test_streamer)", s["id"])

    if telemetry.enabled():
        # Backpressure gauges sampled every telemetry flush. The analysis
        # branch's GStreamer mux queues are registered per-source below.
        telemetry.register_gauge("publisher_pending", publisher.pending)
        for _src in sources:
            for _i, _nm in enumerate(("pub_depth", "pub_drops_total", "pub_wait_ms")):
                telemetry.register_gauge(f"{_nm}_{_src['id']}",
                                         lambda c=_src["id"], i=_i: publisher.stream_stats(c)[i])
            telemetry.register_gauge(f"pub_shed_total_{_src['id']}", lambda c=_src["id"]: publisher.shed_count(c))
        telemetry.register_gauge("pub_meta_q_depth", lambda: publisher._meta_q.qsize())
        telemetry.register_gauge("pub_encoders", lambda: len(publisher._encoders))
        telemetry.register_gauge("pub_queue_cap", lambda: publisher._max_pending)
        telemetry.register_gauge("mux_queue_cap", lambda: gcfg.analysis_queue_buffers)
        telemetry.register_gauge("publisher_drops", lambda: publisher._drops)
        if uploader is not None:
            telemetry.register_gauge("segment_uploader_pending",
                                     lambda: uploader._q.qsize())

    states = {}
    index_to_id = {}
    # (source_id, mux pad index, tee) per recordable source — created lazily by
    # _RecordBranchScheduler after the pipeline is PLAYING (see
    # SidecarConfig.record_stagger_ms for why opens must be serialized).
    record_targets = []
    for k, s in enumerate(sources):
        srcbin = _make("nvurisrcbin", f"src_{k}")
        srcbin.set_property("uri", s["uri"])
        if s["uri"].startswith("file://") and bool(cfg.get("file_loop", True)):
            try:
                srcbin.set_property("file-loop", True)
            except Exception:
                pass
        if s["uri"].startswith("rtsp://"):
            try:
                srcbin.set_property("rtsp-reconnect-interval", int(cfg.get("rtsp_reconnect_interval", 10)))
            except Exception:
                pass
            # Jitterbuffer for live RTSP. Was hardcoded 100ms + drop-on-latency
            # (LAN tuning). 2026-09-13: configurable -- over the Tailscale relay
            # to the real cameras, delay swings 0.2-1.5s, so nearly every packet
            # counted as "late", got dropped, and the source looked silent ->
            # reset loop (while an independent ffmpeg pull of the same stream
            # through go2rtc was fine). Defaults keep the old behavior.
            try:
                srcbin.set_property("latency", int(cfg.get("rtsp_latency_ms", 100)))
            except Exception:
                pass
            try:
                srcbin.set_property("drop-on-latency", bool(cfg.get("rtsp_drop_on_latency", True)))
            except Exception:
                pass
            # Request keyframes from the RTSP server so splitmuxsink gets clean
            # GOP boundaries for each segment, avoiding "Queued GOP time is negative".
            try:
                srcbin.set_property("send-keyframe-requests", True)
            except Exception:
                pass
            # Force TCP transport for RTP: UDP multicast doesn't reliably traverse
            # Docker's bridge network, causing "not-negotiated" errors on UDPSrc.
            # Protocol enum: 0x4 = TCP only.
            try:
                srcbin.set_property("select-rtp-protocol", 0x4)
            except Exception:
                pass
            # Increase UDP buffer to reduce packet loss when UDP is used as fallback.
            try:
                srcbin.set_property("udp-buffer-size", 2097152)  # 2 MB
            except Exception:
                pass
            # Extra decode surfaces for live RTSP jitter tolerance.
            try:
                srcbin.set_property("num-extra-surfaces", 2)
            except Exception:
                pass
            # RTCP camera clock, the other half of the attach-sys-ts wiring on
            # nvstreammux above: makes this source turn the camera's Sender
            # Reports into per-frame ntp_timestamps. A camera that sends none
            # just leaves ntp_timestamp at 0 (server-time fallback downstream).
            if rtcp_camera_clock:
                try:
                    pyds.configure_source_for_ntp_sync(hash(srcbin))
                except Exception as exc:
                    logger.warning("[%s] RTCP camera clock not wired (server time "
                                   "will be used): %s", s["id"], exc)
        # Decoder-level frame drop: decode every Nth frame only. Cuts decode GPU
        # work itself (the biggest per-stream pipeline cost), at the cost of also
        # capping the record branch (which taps the same tee). See
        # SidecarConfig.decoder_drop_interval for the tradeoff vs analysis_fps.
        if gcfg.decoder_drop_interval > 1:
            try:
                srcbin.set_property("drop-frame-interval", gcfg.decoder_drop_interval)
                logger.info("src_%d drop-frame-interval=%d", k, gcfg.decoder_drop_interval)
            except Exception as exc:
                logger.warning("src_%d drop-frame-interval failed (%s); decoding full rate",
                               k, exc)
        # Set explicit source-id so reconnect logs show the correct index.
        try:
            srcbin.set_property("source-id", k)
        except Exception:
            pass
        pipeline.add(srcbin)
        # GPU decode telemetry: time the decoder inside this source bin
        # (self-installs via a one-shot src-pad probe once it exists).
        _instrument_decode(srcbin, s["id"])

        tee = _make("tee", f"tee_{k}")
        pipeline.add(tee)
        # Record branches are NOT built here: with a static build every
        # nvv4l2h264enc opens its NVENC session at once during the single
        # set_state(PLAYING), which deadlocks the encoder on some GPUs at 6+
        # branches (see SidecarConfig.record_stagger_ms). Instead, the
        # _RecordBranchScheduler creates each branch after the pipeline is
        # PLAYING, one at a time, serializing the opens.
        rec_branch = None
        kind = recorder_kind(s["uri"], s.get("recorder", gcfg.recorder_default))
        if kind == "svr":
            # Smart Video Record: nvurisrcbin keeps a rolling cache of the source's
            # OWN encoded stream and muxes it to a file on start-sr. No decode ->
            # encode round trip, so NO NVENC SESSION -- which is the point. The
            # encode branch spends one session per stream against a driver ceiling
            # the probe measures at ~8-14, and _RecordBranchScheduler's own docstring
            # notes that streams past it "keep analyzing/previewing but don't record".
            # SVR removes the ceiling for RTSP rather than scheduling around it.
            #
            # Deliberately NOT added to record_targets: no encoder branch, no
            # scheduler slot, and no capacity consumed from the measured budget.
            _configure_svr_source(gcfg, srcbin, s["id"])
        elif gcfg.record_enabled and kind == "encode":
            # NOTE: the single-owner concern that sources.owns_recording solves for the
            # boot-time flushes applies here too (two role processes would both encode a
            # shared untagged camera), but this path is unresolved -- not exercised by
            # the RTSP/`recorder: auto` deployment, and gating it needs more design.
            record_targets.append((s["id"], k, tee))
        # Optional analysis-branch frame-rate cap: a GStreamer-native videorate +
        # capsfilter pair forcing the branch to analysis_fps, so the analysis
        # pipeline (nvinfer -> nvtracker -> nvvideoconvert -> probe) only sees
        # analysis_fps frames/s per source. The record branch (off the same tee)
        # is left untouched, so recordings stay full-rate. See
        # SidecarConfig.analysis_fps. None = disabled (direct tee->mux link).
        #
        # NOTE: videorate converts by PTS; it does not throttle decode work (it
        # is post-decode), but it preserves native PTS so the clock-synced sink
        # paces the chain to realtime (no fast-forward distortion — the old
        # wall-clock drop-probe's PTS re-stamping was what made file sources
        # decode ahead). The preview (probe sits post-mux) runs at the analysis
        # rate: use analysis_fps when per-stream analysis GPU savings matter
        # more than preview smoothness, process_fps for a full-rate smooth
        # preview with inference-only throttling (see sidecar.yaml).
        mux_rate_chain = None
        if gcfg.analysis_fps > 0:
            vrate = _make("videorate", f"vrate_{k}")
            rate_caps = _make("capsfilter", f"rate_caps_{k}")
            rate_caps.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM), framerate={int(gcfg.analysis_fps)}/1"))
            pipeline.add(vrate)
            pipeline.add(rate_caps)
            vrate.link(rate_caps)
            mux_rate_chain = (vrate, rate_caps)

        # The record-branch rate cap (videorate + capsfilter at record_fps) is
        # built by _RecordBranchScheduler alongside each branch.

        # A tee pushes to its src pads SERIALLY, on the upstream thread, and a
        # branch only gets its own thread where a queue sits. videorate is a
        # filter, not a queue -- so without this the analysis branch runs INSIDE
        # the decoder's thread, welded to the record branch hanging off the same
        # tee. Anything that stalls the record branch (an NVENC wedge mid-run, a
        # disk stall in splitmuxsink) therefore stops analysis on this source:
        # the same coupling the "Queued GOP time is negative" comment in
        # _build_record_branch describes as freezing the WHOLE pipeline. That fix
        # removed one trigger; this removes the mechanism.
        #
        # It also removes the leak-vs-block dilemma on rec_queue: with the branches
        # decoupled, the record queue can block safely without reaching the decoder,
        # so it stays non-leaky and segments are never dropped.
        #
        # A queue buffers, it does not isolate -- a SUSTAINED stall still propagates
        # once the buffer drains. It converts "any momentary blip freezes every
        # stream" into "only a sustained problem does".
        #
        # Sized in BUFFERS deliberately: these are NVMM surfaces, so a GstBuffer
        # carries a handle, not the pixels, and max-size-bytes would read a few
        # hundred bytes while the real GPU cost is ~1.4MB/frame at 720p. Depth 2 is
        # ~2.8MB per stream -- at 192 streams that is ~530MB of NVMM, so this is a
        # real budget item, not a free win. 0 disables (direct link, today's
        # behavior).
        mux_q = None
        if gcfg.analysis_queue_buffers > 0:
            mux_q = _make("queue", f"mux_queue_{k}")
            mux_q.set_property("max-size-buffers", gcfg.analysis_queue_buffers)
            mux_q.set_property("max-size-bytes", 0)
            mux_q.set_property("max-size-time", 0)
            pipeline.add(mux_q)
            if telemetry.enabled():
                # Fill level of this source's analysis-branch queue (buffers).
                # A queue pinned at max while the mux is behind is where
                # backpressure originates; see SidecarConfig.analysis_queue_buffers.
                telemetry.register_gauge(
                    f"mux_queue_{k}",
                    lambda el=mux_q: int(el.get_property("current-level-buffers") or 0))

        # Whether this source's tee -> [queue] -> [videorate] -> streammux chain
        # has been built yet. A source RESTART (see _maybe_restart_dead_source)
        # re-fires pad-added, and only the srcbin's own pad may be relinked then:
        # going to NULL removed it. The rest must NOT be rebuilt, because
        # tee.request_pad_simple("src_%u") always creates a NEW pad — a second
        # pass would leave an orphan branch fanning out of the tee forever while
        # the already-requested mux pad hands back nothing.
        wired = {"done": False}

        def _on_pad_added(_bin, pad, idx=k, tee=tee, wired=wired,
                          mux_rate_chain=mux_rate_chain, mux_q=mux_q):
            if pad.is_linked():
                return
            pad.link(tee.get_static_pad("sink"))
            if wired["done"]:
                return
            mux_sink = streammux.request_pad_simple(f"sink_{idx}")
            mux_src = tee.request_pad_simple("src_%u")
            if mux_src and mux_sink:
                wired["done"] = True
                # tee -> [queue] -> [videorate -> capsfilter] -> streammux
                head = mux_src
                if mux_q is not None:
                    ret = head.link(mux_q.get_static_pad("sink"))
                    if ret != Gst.PadLinkReturn.OK:
                        logger.error("[src_%d] tee->mux_queue link failed: %s", idx, ret)
                    head = mux_q.get_static_pad("src")
                if mux_rate_chain is not None:
                    vrate, rate_caps = mux_rate_chain
                    head.link(vrate.get_static_pad("sink"))
                    head = rate_caps.get_static_pad("src")
                ret = head.link(mux_sink)
                if ret != Gst.PadLinkReturn.OK:
                    # A failed link leaves this source silently detached from
                    # inference: the pipeline runs, the stream just never produces
                    # a frame, which is indistinguishable from a dead camera.
                    logger.error("[src_%d] ->streammux link failed: %s", idx, ret)
        srcbin.connect("pad-added", _on_pad_added)

        index_to_id[k] = s["id"]
        states[k] = make_stream_state(s["id"], gcfg.key_id(s["id"]))
        states[k].srcbin = srcbin      # _maybe_start_svr emits start-sr on this
        states[k].mux_q = mux_q        # read by the flight recorder's source-stall evidence

    if telemetry.enabled():
        # Per-source analysis rate (fps), updated by the probe's 5 s fps block;
        # reads 0 until the first window completes. Lets a dashboard show the
        # achieved per-stream rate vs the analysis_fps cap at a glance.
        for k, s in enumerate(sources):
            st = states[k]
            # A source that stopped delivering reads 0, not its last window's
            # rate (2026-09-26: dead cameras kept printing 30.6 fps).
            telemetry.register_gauge(
                f"fps_{s['id']}",
                lambda st=st: telemetry.fresh_or_zero(
                    getattr(st, "_fps_now", 0.0), getattr(st, "_last_frame_t", None),
                    time.monotonic(), max(telemetry.interval(), 5.0)))
            telemetry.register_gauge(
                f"frames_total_{s['id']}", lambda st=st: getattr(st, "_frames_total", 0))
            telemetry.register_gauge(
                f"frame_age_s_{s['id']}",
                lambda st=st: (time.monotonic() - st._last_frame_mono) if getattr(st, "_last_frame_mono", None) else -1.0)
            telemetry.register_gauge(
                f"stall_restarts_total_{s['id']}", lambda st=st: getattr(st, "_stall_restarts", 0))

    # The modules the product asked for (`modules:` in its yaml). Loaded here, once the per-stream states exist, and
    # before the inference chain is built because modules can ask for inference elements of their own.
    ctx_ref: list = []
    states_by_id = {st.id: st for st in states.values()}

    def _emit(stream_key, fields, maxlen=64):
        try:
            gcfg.r.xadd(stream_key, fields, maxlen=maxlen, approximate=True)
            return True
        except redis.RedisError as exc:
            logger.error("redis xadd (%s) failed: %s", stream_key, exc)
            return False

    def _every(interval, fn):
        """Call `fn` every `interval` seconds on the GLib main loop (once it runs); a failure is logged, not fatal."""
        def tick():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("module timer failed: %s", exc)
            return True        # keep the timer alive
        GLib.timeout_add(int(interval * 1000), tick)

    def _request_clip(stream_id, now, age=0.0):
        if ctx_ref and stream_id in states_by_id:
            _maybe_start_svr(ctx_ref[0], states_by_id[stream_id], now, age)

    services = EngineServices(
        redis=gcfg.r, redis_factory=lambda: redis.Redis.from_url(cfg["redis_url"]), emit=_emit,
        request_clip=_request_clip, stream_ids=list(states_by_id), timer=_every,
        telemetry=telemetry, batch_size=batch, config=cfg, states=states_by_id,
        mux_size=(gcfg.mux_w, gcfg.mux_h), sources=sources, all_sources=all_sources, role=role,
        shard_id=getattr(gcfg, "shard_id", 0), engine=gcfg,
        index_to_id=index_to_id, key_id=gcfg.key_id)
    modules = load_modules(cfg.get("modules"), services)
    keyed = camera_keyed_modules(modules) if gcfg.role_scoped_keys else []
    if keyed:
        raise ModuleConfigError(f"role_scoped_keys: these modules write camera-keyed streams and cannot run with "
                                f"it: {', '.join(keyed)}")
    if modules:
        logger.info("modules: %s", ", ".join(modules.names()))

    pgie = None
    if cfg.get("pgie_enabled", True):
        pgie = _make("nvinfer", "pgie")
        pgie.set_property("config-file-path", cfg["pgie_config"])
        pgie.set_property("batch-size", batch)
        # Optional inference throttle: process every Nth batch only (DeepStream's
        # `interval` property; skipped frames propagate the last detections so the
        # tracker stays live). Driven by process_fps so a realtime 30 fps source can
        # be analyzed at e.g. 5 fps, matching the capacity benchmark's 5fps fixtures.
        if gcfg.infer_interval > 1:
            try:
                pgie.set_property("interval", gcfg.infer_interval)
                logger.info("pgie interval=%d (process_fps=%s, stream_fps=%s)",
                            gcfg.infer_interval, cfg.get("process_fps"),
                            cfg.get("stream_fps"))
            except Exception as exc:
                logger.error("pgie interval property failed (%s); "
                             "inference stays every frame", exc)
        # model-engine-file is only nvinfer's DESERIALIZE probe. When it rebuilds,
        # TrtModelBuilder::buildModel() serializes to an auto-derived path BESIDE
        # the ONNX: <onnx-path>_b<batch>_gpu<id>_<mode>.engine (this value is
        # ignored for serialization — same rule as every secondary GIE). So the probe must
        # point at that derived path or the engine is rebuilt from ONNX on EVERY
        # restart. The old code probed a hand-named {engine_dir}/model_b1_gpu0_fp32
        # .engine that only worked because a build had been placed there manually;
        # on a fresh checkout it would rebuild every boot, and in INT8 mode the
        # derived name changes to _int8 so a fixed name would rebuild every boot too.
        engine_path = gie_engine_path(cfg["pgie_config"], batch)
        pgie.set_property("model-engine-file", engine_path)
        logger.info("TRT engine path: %s", engine_path)
        _gpu_probe_stage(pgie, "gpu.infer.pgie")
    else:
        # No yaml file sets pgie_enabled: false today -- the only way to land
        # here is apply_role_overrides forcing it off for a role that has its own
        # detector -- so name the role rather than a key nobody set.
        logger.info("pgie (COCO) disabled (role=%s)", role)

    # Inference elements the modules asked for. A "primary" one runs on the full frame next to pgie, BEFORE the
    # tracker: on a batch where `interval` skips inference it produces no metadata at all, so a module that needs
    # every frame must ask for interval 0. A "secondary" one runs on the tracked objects, after the tracker.
    module_gies = []
    for spec in modules.gies():
        el = _make("nvinfer", spec.name)
        el.set_property("config-file-path", spec.config_path)
        if spec.batch_from_mux:
            el.set_property("batch-size", batch)
        if spec.interval > 0:
            el.set_property("interval", spec.interval)
        if spec.engine_path:
            el.set_property("model-engine-file", spec.engine_path)
        logger.info("%s enabled, config=%s engine=%s interval=%d", spec.name, spec.config_path,
                    spec.engine_path or "(from config)", spec.interval)
        _gpu_probe_stage(el, f"gpu.infer.{spec.name}")
        module_gies.append((spec, el))

    tracker = _make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file", cfg["tracker_lib"])
    tracker.set_property("ll-config-file", cfg["tracker_config"])
    _gpu_probe_stage(tracker, "gpu.tracker")

    # A GIE whose config carries engine-create-func-name derives a <cwd>-based
    # engine name with nothing model-specific in it, so two such configs land on
    # the same file and one silently deserializes the other's model, with no error
    # at any level. Fail the boot instead. Every GIE is included even where its
    # path is hand-built and cannot collide today -- a guard that covers some of
    # them invites the next one to be the exception.
    _engine_entries = []
    if pgie is not None:
        _engine_entries.append(("pgie", cfg["pgie_config"], engine_path))
    for spec, _el in module_gies:
        if spec.engine_path:
            _engine_entries.append((spec.name, spec.config_path, spec.engine_path))
    assert_distinct_engine_paths(_engine_entries)

    # Non-inference elements the modules asked for (e.g. the analytics element zone detection reads ROI membership
    # from), added after every inference element.
    module_elements = []
    for spec in modules.elements():
        el = _make(spec.factory, spec.name)
        for prop, value in spec.properties.items():
            el.set_property(prop, value)
        module_elements.append(el)

    # Single convert to RGBA at capture resolution, CUDA-unified so the probe can map the
    # surface. No nvdsosd — we draw boxes ourselves in the probe from each frame's own
    # metadata, which sidesteps nvdsosd's batched cross-stream box mixing.
    conv = _make("nvvideoconvert", "conv-capture")
    conv.set_property("nvbuf-memory-type", 3)  # NVBUF_MEM_CUDA_UNIFIED
    caps = _make("capsfilter", "caps-capture")
    caps.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM),format=RGBA,width={gcfg.capture_w},height={gcfg.capture_h}"))

    sink = _make("fakesink", "sink")
    # sync = True works for file sources (PTS starts at 0, sink paces at real-time)
    # but BREAKS for RTSP live sources: RTSP PTS comes from the server's clock which
    # is way ahead of the pipeline clock, so the sink stalls waiting forever.
    # Fall back to sync=False whenever any source is RTSP. When analysis_fps is
    # active the videorate re-times kept buffers onto the analysis_fps cadence,
    # so the synced sink paces the branch at ~analysis_fps (real-time decode, no
    # fast-forward). Overridable via sink_sync for special cases.
    has_rtsp = any(s["uri"].startswith("rtsp://") for s in sources)
    sink_sync = cfg.get("sink_sync")
    if sink_sync is None:
        sink_sync = not has_rtsp
    sink.set_property("sync", 1 if sink_sync else 0)

    if pgie is not None:
        pipeline.add(pgie)
    for _spec, el in module_gies:
        pipeline.add(el)
    pipeline.add(tracker)
    for el in module_elements:
        pipeline.add(el)
    for el in (conv, caps, sink):
        pipeline.add(el)

    # `tracker` is appended BEFORE detect_chain[0] is indexed below: that append
    # is what keeps the "no GIE at all" case (no pgie and no primary module detector)
    # from raising IndexError here. Do not reorder these two lines.
    detect_chain = [el for el in [pgie] + [el for spec, el in module_gies if spec.role == "primary"] if el is not None]
    if not detect_chain:
        logger.warning(
            "no GIE in the pipeline: pgie is disabled and no module asked for a primary detector "
            "(role=%s, pgie_enabled=%s) -- streammux feeds the tracker directly and NOTHING will ever be "
            "detected. Role overrides only turn detectors off, never back on, so check the yaml toggles "
            "for this role.", role, cfg.get("pgie_enabled", True))
    detect_chain.append(tracker)
    streammux.link(detect_chain[0])
    for a, b in zip(detect_chain, detect_chain[1:]):
        a.link(b)
    prev = tracker
    for spec, el in module_gies:
        if spec.role == "secondary":
            prev.link(el)
            prev = el
    for el in module_elements:
        prev.link(el)
        prev = el
    prev.link(conv)
    conv.link(caps)
    caps.link(sink)

    ctx = ProbeCtx(gcfg, states, index_to_id, publisher=publisher)
    ctx.modules = modules
    ctx_ref.append(ctx)
    # nvinfer's unique component id -> element name, so a module can tell which GIE produced an object
    for _el in [pgie] + [el for _spec, el in module_gies]:
        if _el is not None:
            ctx.gie_names[int(_el.get_property("unique-id"))] = _el.get_name()
    modules.setup("on_built", PipelineHandle(pipeline=pipeline, get=pipeline.get_by_name))
    caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, ctx)
    record_scheduler = (_RecordBranchScheduler(gcfg, pipeline, uploader,
                                               record_targets)
                        if record_targets else None)
    return (pipeline, ctx, uploader, publisher, record_scheduler)



def _prewarm_gst_discovery():
    """Force GStreamer's typefind + demux/parse plugins to load once, safely,
    single-threaded, before the pipeline goes live.

    Traced root cause of a role's crash loop (plain segfaults, a
    heap-corruption hang, and clean SIGTRAP aborts once G_DEBUG=fatal-warnings
    was set for diagnosis -- all the same underlying bug, confirmed live via
    gdb across 15+ captures): DeepStream's SmartRecord element spawns a
    background GstDiscoverer on every finished recording (gst_discoverer_
    discover_uri, called from libnvdsgst_smartrecord.so), whose typefind step
    dlopen()s libgsttypefindfunctions.so the first time it's needed. Every
    single capture faulted at the identical instruction inside
    ld-linux-x86-64.so.2, in either a "typefind:sink" thread or the main
    thread's own gst_plugin_load_by_name -> dlopen call -- i.e. two threads
    racing each other's FIRST-EVER dlopen of the same plugin.

    A dlopen() of an ALREADY-loaded shared object is cheap and does not
    re-enter that risky first-load path, so running one real, complete
    discovery pass here -- before autostart/PLAYING, before any camera can
    trigger a real SVR completion -- makes every later concurrent discoverer
    call hit the fast, safe, already-loaded path instead of racing on the
    first one. Best-effort: any failure here is logged and swallowed, never
    fatal to startup -- this mitigates a lower-level loader bug, it is not a
    correctness requirement of the pipeline itself.
    """
    import tempfile
    try:
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import GstPbutils
    except Exception as exc:
        logger.warning("prewarm: GstPbutils unavailable (%s) -- skipping", exc)
        return
    path = tempfile.mktemp(suffix=".mkv", prefix="iris_prewarm_")
    try:
        build = Gst.parse_launch(
            "videotestsrc num-buffers=1 ! video/x-raw,width=64,height=64 ! "
            f"matroskamux ! filesink location={path}")
        build.set_state(Gst.State.PLAYING)
        msg = build.get_bus().timed_pop_filtered(
            5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        build.set_state(Gst.State.NULL)
        if msg is None or msg.type == Gst.MessageType.ERROR:
            logger.warning("prewarm: could not build a throwaway sample (%s) -- skipping",
                           msg.parse_error() if msg else "timed out")
            return
        info = GstPbutils.Discoverer.new(5 * Gst.SECOND).discover_uri("file://" + path)
        logger.info("prewarm: GStreamer typefind/discovery plugins loaded (%s)",
                    info.get_result())
    except Exception as exc:
        logger.warning("prewarm: discovery pass failed (%s) -- the dlopen race this "
                       "guards against may still occur", exc)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
