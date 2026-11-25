"""IRIS DeepStream perception sidecar — batched multi-source (Stage 1).

Runs inside the DeepStream 9.0 container. Reads a list of sources from
sidecar.yaml, batches them through one nvstreammux -> nvinfer -> nvtracker
pipeline, and publishes per-source detections + annotated JPEGs to Redis:
  iris:meta:<id> (stream), iris:jpeg:<id> (string), iris:status:<id> (string).
NOT importable off-GPU (needs gi/pyds). Pure config logic lives in sources.py.
"""

import os
import signal
import sys
import threading
import time
import logging
import uuid
from .sources import (known_roles, load_sidecar_config, resolve_sources, select_sources, apply_role_overrides,
    owns_recording, role_scope_problem, stale_status_keys)
from .source_recovery import (MainLoopWatchdog, SourceRestartPolicy,  # noqa: E402
                             SourceStallWatch, dead_source_index)
from . import flight
from . import telemetry

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .config import SidecarConfig
from .control import ControlPoller
from .gst_build import _prewarm_gst_discovery, build_pipeline
from .gst_util import _install_log_filter, logger
from .modules import ModuleConfigError, ModuleStartupError
from .model_check import verify_models
from .model_manifest import ManifestError
from .recording import _flush_segment_index, _flush_svr_dir
from .status import _write_status
from .svr import _maybe_start_svr




def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "perception/pipeline/sidecar.yaml"
    # A product overlay (products/<name>/sidecar.yaml) names its base with `extends:`.
    cfg = load_sidecar_config(cfg_path)

    # INT8 calibration (pgie): the custom YOLO parser's calibrator (yolo.cpp)
    # reads these env vars at engine-build time. Config-driven so calibration
    # runs without editing the container environment. Only consulted when
    # config_infer_primary_yolo11.txt sets network-mode=1 (INT8); inert
    # otherwise.
    if cfg.get("int8_calib_img_path"):
        os.environ["INT8_CALIB_IMG_PATH"] = str(cfg["int8_calib_img_path"])
    if cfg.get("int8_calib_batch_size"):
        # The engine batch is capped at max_pgie_batch (see build_pipeline), and
        # the custom YOLO calibrator segfaults during INT8 calibration when its
        # batch doesn't match the engine batch — clamp so a huge source count
        # can never rebuild the engine with a mismatched calibration batch.
        max_batch = int(cfg.get("max_pgie_batch", 160))
        os.environ["INT8_CALIB_BATCH_SIZE"] = str(
            min(int(cfg["int8_calib_batch_size"]), max_batch))

    _level = str(cfg.get("logging", {}).get("sidecar_level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, _level, logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    _install_log_filter()

    sources = resolve_sources(cfg)
    if not sources:
        logger.error("no sources configured (set 'sources:' or 'uri:' in sidecar.yaml)")
        sys.exit(1)
    all_sources = list(sources)

    # ── Process sharding (M1) + role separation (M2) ────────────────────────
    # Each replica owns a disjoint camera slice: sources[shard_id::shard_count],
    # optionally pre-filtered to one role (sources.known_roles) -- see
    # sources.select_sources. All Redis keys are per-source-id, so processes
    # never collide and StreamManager discovers each one's streams identically.
    # Env overrides config; both unset = single process (today's behavior,
    # byte-for-byte).
    shard_id = int(os.environ.get("IRIS_SHARD_ID", cfg.get("shard_id", 0)))
    shard_count = int(os.environ.get("IRIS_SHARD_COUNT", cfg.get("shard_count", 1)))
    role = os.environ.get("IRIS_SIDECAR_ROLE", cfg.get("sidecar_role")) or None
    if shard_count < 1:
        shard_count = 1
    if not (0 <= shard_id < shard_count):
        logger.error(
            "IRIS_SHARD_ID=%d out of range for IRIS_SHARD_COUNT=%d; if you "
            "started the 'shards' compose profile without exporting "
            "IRIS_SHARD_COUNT, export it (e.g. IRIS_SHARD_COUNT=2) before "
            "docker compose --profile shards up",
            shard_id, shard_count)
        sys.exit(1)
    if role and role not in known_roles(cfg, all_sources):
        # A typo fails silently in BOTH directions at once: select_sources drops
        # every explicitly tagged source (only untagged ones match an unknown
        # role) while apply_role_overrides is a total no-op, so the process boots
        # a FULL pipeline over the WRONG camera subset. Refuse instead.
        logger.error(
            "IRIS_SIDECAR_ROLE=%r is not a known role; valid values are %s, or "
            "unset (single process owning every camera). Note the value is "
            "case-sensitive.", role, ", ".join(sorted(known_roles(cfg, all_sources))) or "(none configured)")
        sys.exit(1)
    if shard_count > 1 or role:
        sources = select_sources(all_sources, shard_id, shard_count, role)
        if not sources:
            logger.error("shard %d/%d role=%s owns no sources", shard_id, shard_count, role)
            sys.exit(1)
    logger.info("sidecar starting shard=%d/%d role=%s sources=%d ids=%s",
                shard_id, shard_count, role, len(sources), [s["id"] for s in sources])
    # One id per process, published in every iris:status:<id> write below -- see
    # _write_status.
    boot_id = uuid.uuid4().hex

    autostart = bool(cfg.get("autostart", False))
    cfg = apply_role_overrides(cfg, role)
    # Model manifest (spike D11): every model file this replica loads must be the registered version its committed
    # manifest lists. Checked here, with the config final for this role and before anything is built or loaded.
    try:
        verify_models(cfg, what=f"{cfg_path} (role {role})")
    except (ManifestError, ModuleConfigError) as exc:
        logger.error("refusing to start: %s", exc)
        sys.exit(1)
    # role_scoped_keys: this process's per-camera Redis keys carry its role (<camera>@<role>), so several role
    # processes can share a camera (sources.stream_key_id). Off = every key name exactly as before.
    _scope_problem = role_scope_problem(cfg, role, sources)
    if _scope_problem:
        logger.error("refusing to start: %s", _scope_problem)
        sys.exit(1)
    gcfg = SidecarConfig(cfg, role=role, shard_id=shard_id)

    # Serve the telemetry as Prometheus metrics (/metrics on
    # telemetry_prometheus_port) for the monitoring stack's prometheus + grafana.
    # Requires telemetry on; safe no-op otherwise. One HTTP server per process,
    # so shard replicas each export their own slice on the same internal port.
    if telemetry.enabled():
        telemetry.start_prometheus(gcfg.telemetry_prometheus_port)

    # --- Purge stale iris:status:* keys from previous runs ---
    # Any key whose source-id is not in the CURRENT config is a ghost from a
    # prior invocation (different camera list, renamed ID, etc.).  Delete them
    # NOW so the StreamManager discovery loop never surfaces a phantom stream.
    # ⚠️ Shard safety: only shard 0 purges, and it must purge against the FULL
    # (pre-slice) source list — otherwise shard 0 would delete shard 1's live
    # status keys at boot (each shard's slice is only a subset of all ids).
    if shard_id == 0:
        _current_ids = {s["id"] for s in all_sources}
        try:
            _existing_keys = gcfg.r.keys("iris:status:*")
            # Role-scoped: only this role's own keys -- another role's belong to another process.
            _stale_keys = stale_status_keys(_existing_keys, _current_ids, role, gcfg.role_scoped_keys)
            if _stale_keys:
                gcfg.r.delete(*_stale_keys)
                logger.info("purged %d stale status key(s): %s",
                            len(_stale_keys),
                            [k.decode() if isinstance(k, bytes) else k for k in _stale_keys])
        except Exception as exc:
            logger.warning("stale status key purge failed (non-fatal): %s", exc)
    else:
        logger.info("shard %d/%d: skipping stale-status purge (shard 0 owns it)",
                    shard_id, shard_count)
    # ---------------------------------------------------------

    # Both flushes DELETE existing recording state (Redis segment index + stored
    # objects; every {id}_* file in the shared svr_dir). An untagged camera is in
    # EVERY role process's `sources` (select_sources' additive safety default), so
    # scoping the flushes to a single owner is what stops the process that boots
    # second from wiping the first one's in-flight recordings. See
    # sources.owns_recording for why its untagged default differs from the above.
    recordable_sources = [s for s in sources if owns_recording(s, role, cfg.get("untagged_recording_owner"))]
    if gcfg.record_enabled:
        _flush_segment_index(gcfg, recordable_sources)
    _flush_svr_dir(gcfg, recordable_sources)

    try:
        pipeline, ctx, uploader, publisher, record_scheduler = build_pipeline(cfg, gcfg, sources, all_sources, role)
    except (ModuleConfigError, ModuleStartupError) as exc:
        # A module refused to start (a bad setting, a missing model file): its message says what to fix, and is the
        # whole story -- log it as one line and exit, instead of ending in a traceback that reads like a crash.
        logger.error("refusing to start: %s", exc)
        sys.exit(1)

    _prewarm_gst_discovery()

    # The configured modules: start their workers and register their gauges.
    ctx.modules.setup("start")
    ControlPoller(gcfg.r, ctx.modules.controls()).start()
    if telemetry.enabled():
        for _m in ctx.modules.modules:
            for _gauge, _fn in _m.gauges().items():
                telemetry.register_gauge(_gauge, _fn)

    initial_state = Gst.State.PLAYING if autostart else Gst.State.PAUSED
    pipeline.set_state(initial_state)
    for s in sources:
        _write_status(gcfg.r, s, running=autostart,
                      pipeline_preview=gcfg.pipeline_preview, boot_id=boot_id, key_id=gcfg.key_id(s["id"]))
    if autostart:
        logger.info("pipeline PLAYING (autostart=true)")
    else:
        logger.info("pipeline PAUSED — send 'playing' to iris:ctrl:pipeline to start")

    # Poll iris:ctrl:pipeline every second and transition the GStreamer pipeline
    # between PAUSED and PLAYING without rebuilding it (TRT engine stays loaded).
    _ctrl_key = "iris:ctrl:pipeline"
    _is_playing = [autostart]

    # Record branches are created lazily after PLAYING, one at a time, to
    # serialize the NVENC session opens (see SidecarConfig.record_stagger_ms).
    # Start the scheduler immediately: with autostart the pipeline is already
    # PLAYING, otherwise it holds the branches until the first 'playing' ctrl.
    if record_scheduler is not None:
        record_scheduler.start(lambda: _is_playing[0])

    def _poll_ctrl():
        try:
            val = gcfg.r.get(_ctrl_key)
            if val is None:
                return True
            val = val.decode() if isinstance(val, bytes) else val
            want_play = (val == "playing")
            if want_play and not _is_playing[0]:
                pipeline.set_state(Gst.State.PLAYING)
                _is_playing[0] = True
                gcfg.stall_watch.reset(time.monotonic())  # paused != stalled
                for s in sources:
                    _write_status(gcfg.r, s, running=True,
                                  pipeline_preview=gcfg.pipeline_preview, boot_id=boot_id,
                                  key_id=gcfg.key_id(s["id"]))
                logger.info("pipeline PLAYING (ctrl)")
            elif not want_play and _is_playing[0]:
                pipeline.set_state(Gst.State.PAUSED)
                _is_playing[0] = False
                gcfg.stall_watch.reset(time.monotonic())
                for s in sources:
                    _write_status(gcfg.r, s, running=False,
                                  pipeline_preview=gcfg.pipeline_preview, boot_id=boot_id,
                                  key_id=gcfg.key_id(s["id"]))
                logger.info("pipeline PAUSED (ctrl)")
        except Exception as exc:
            logger.warning("ctrl poll error: %s", exc)
        return True  # keep GLib timer alive

    GLib.timeout_add(1000, _poll_ctrl)

    # Poll iris:svr_request:<id> for alerts the SIDECAR cannot see. Alerts the modules raise are decided in the
    # buffer probe, which asks for a clip directly; a proactive VLM alert is decided entirely app-side,
    # so without this channel an RTSP stream's VLM alerts queued a clip job that no
    # smart-record file ever arrived for and resolved 'unavailable' every time.
    #
    # The request carries the alert's pts, not a wall clock: the app's anchor comes
    # from the meta stream, which this probe wrote from frame_meta.buf_pts, so age is
    # a subtraction in one clock rather than a comparison across two containers.
    _svr_states = {st.id: st for st in ctx.states.values()
                   if st.id in gcfg.svr_triggers}
    def _svr_tail_id(sid):
        """The stream's newest id, so polling starts after it.

        Not "$": that only means "entries newer than this call" to a BLOCKING xread,
        and this one cannot block without freezing the GLib main loop -- a
        non-blocking xread on "$" returns nothing, forever. Resolving the tail up
        front gets the same intent, and skips requests buffered while the sidecar was
        down: those describe moments long out of the pre-roll cache, so replaying them
        would record unrelated footage under an old alert.
        """
        try:
            tail = gcfg.r.xrevrange(f"iris:svr_request:{gcfg.key_id(sid)}", count=1)
        except Exception as exc:
            logger.warning("[%s] could not read the svr request tail: %s", sid, exc)
            return "0-0"
        if not tail:
            return "0-0"           # no stream yet; any request will be newer
        eid = tail[0][0]
        return eid.decode() if isinstance(eid, bytes) else eid

    _svr_req_ids = {sid: _svr_tail_id(sid) for sid in _svr_states}

    def _poll_svr_requests():
        for sid, st in _svr_states.items():
            try:
                resp = gcfg.r.xread({f"iris:svr_request:{gcfg.key_id(sid)}": _svr_req_ids[sid]},
                                    count=10)
            except Exception as exc:
                logger.warning("[%s] svr request poll error: %s", sid, exc)
                continue
            for _key, entries in resp or []:
                for entry_id, fields in entries:
                    _svr_req_ids[sid] = (entry_id.decode()
                                         if isinstance(entry_id, bytes) else entry_id)
                    raw = fields.get(b"pts") or fields.get("pts")
                    if raw is None:
                        continue
                    try:
                        pts = float(raw)
                    except (TypeError, ValueError):
                        logger.warning("[%s] svr request with unparsable pts %r", sid, raw)
                        continue
                    # _last_pts advances at the meta cadence -- the same samples the app
                    # built its anchor from -- so it is the newest moment the app could
                    # be describing. Unset means no frame has been published yet.
                    latest = getattr(st, "_last_pts", None)
                    age = 0.0 if latest is None else latest - pts
                    logger.info("[%s] smart-record requested for a VLM alert %.1fs back",
                                sid, age)
                    _maybe_start_svr(ctx, st, time.monotonic(), age=age)
        return True  # keep GLib timer alive

    if _svr_states:
        GLib.timeout_add(1000, _poll_svr_requests)

    loop = GLib.MainLoop()

    def _describe_sources():
        now_m = time.monotonic()
        out = []
        for k, src in enumerate(sources):
            st = ctx.states.get(k)
            last = getattr(st, "_last_frame_mono", None)
            cid = src["id"]
            out.append(f"{cid}: last frame {'never' if last is None else f'{now_m - last:.0f}s ago'}, "
                       f"fps {getattr(st, '_fps_now', 0.0):.1f}, stall restarts {getattr(st, '_stall_restarts', 0)}, "
                       f"bus nodata {flight_rec.count(cid, 'nodata')} writefail {flight_rec.count(cid, 'writefail')} "
                       f"error {flight_rec.count(cid, 'error')} warn {flight_rec.count(cid, 'warn')}")
        return out

    flight_rec = flight.FlightRecorder(
        cfg.get("flight_dir", "/workspace/iris/data/flight"), logger,
        stall_s=float(cfg.get("flight_stall_seconds", 10.0)), describe=_describe_sources,
        native_dump=bool(cfg.get("flight_native_dump", False)))   # gdb attach on a stall: opt-in per product
    GLib.timeout_add(1000, flight_rec.beat)

    def _source_ages():
        now_m = time.monotonic()
        out = {}
        for k, src in enumerate(sources):
            last = getattr(ctx.states.get(k), "_last_frame_mono", None)
            out[src["id"]] = None if last is None else now_m - last
        return out

    def _stall_evidence(cam):
        k = next((i for i, src in enumerate(sources) if src["id"] == cam), None)
        st = ctx.states.get(k)
        q = getattr(st, "mux_q", None)
        lvl = int(q.get_property("current-level-buffers")) if q is not None else -1
        depth, _drops, wait = publisher.stream_stats(cam)
        return (f"analysis queue {lvl}/{gcfg.analysis_queue_buffers}, publisher queue {depth} "
                f"(oldest wait {wait:.0f} ms), last fps {getattr(st, '_fps_now', 0.0):.1f}, "
                f"bus 'no data' events so far {flight_rec.count(cam, 'nodata')}")

    flight_rec.watch_sources(_source_ages, _stall_evidence)
    flight_rec.start()
    if telemetry.enabled():
        for _n, _f in (("flight_main_lag_s", lambda: flight_rec.main_lag_s),
                       ("flight_sched_delay_ms", lambda: flight_rec.sched_delay_ms),
                       ("flight_threads", lambda: flight_rec.threads),
                       ("flight_cpu_pct", lambda: flight_rec.process_cpu_pct),
                       ("flight_stalls_total", lambda: flight_rec.stalls)):
            telemetry.register_gauge(_n, _f)
        telemetry.register_gauge("decoder_threads_sleeping", lambda: flight_rec.decoders_sleeping)
        telemetry.register_gauge("source_stalls_open", flight_rec.open_episodes)
        for _s in sources:
            telemetry.register_gauge(f"source_stall_episodes_total_{_s['id']}",
                                     lambda c=_s["id"]: flight_rec.episodes(c))
        telemetry.register_gauge("qos_dropped_total", flight_rec.qos_dropped)
        telemetry.register_gauge("flight_pressure_events_total", lambda: flight_rec.pressure_events)
        # components that must keep up with the others: warn in the log (-> Loki) when they do not
        flight_rec.watch_drops(
            "frame publisher (DATA LOST)", lambda: publisher._drops, 0.05,
            "even the meta-only safety queue overflowed: detection metadata for the tracking app was lost")
        flight_rec.watch_drops(
            "frame publisher (previews shed)", lambda: sum(publisher._shed_by.values()), 5.0,
            "the drawing/JPEG workers cannot keep up with the requested preview rate; previews are skipped "
            "(detections are still published). Add encoders (jpeg_encoders) or lower stream_fps")
        flight_rec.watch_drops(
            "gstreamer QoS", flight_rec.qos_dropped, 0.2,
            "an element is dropping frames as too late: the pipeline behind it is slower than its input")
        for _s in sources:
            flight_rec.watch_depth(
                f"{_s['id']} publisher queue", lambda c=_s["id"]: publisher.stream_stats(c)[0],
                publisher._max_pending, "its publisher worker is not draining the queue")
        for _s in sources:
            for _kind in ("nodata", "writefail", "error", "warn"):
                telemetry.register_gauge(f"bus_{_kind}_total_{_s['id']}",
                                         lambda c=_s["id"], k=_kind: flight_rec.count(c, k))
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    # A Smart Record branch that refuses the stream returns NOT_NEGOTIATED into
    # nvurisrcbin's INTERNAL pre-decode tee, which has no queue to absorb it, so
    # the error travels up and stops rtspsrc. The RTSP connection is dead by then
    # ("Could not write to resource" on every subsequent try_send), so
    # nvurisrcbin's own "Resetting source" loop can never bring it back: observed
    # in the field as the first alert on a camera costing that camera for the life
    # of the process. Restarting the source bin here turns a permanent loss into a
    # few-second gap. Classification and rate-limiting are pure and unit-tested in
    # perception/engine/source_recovery.py; this is only the GStreamer half.
    restart_policy = SourceRestartPolicy(gcfg.source_restart_cooldown_seconds)

    def _maybe_restart_dead_source(dbg):
        idx = dead_source_index(dbg)
        if idx is None:
            return
        _restart_source(idx, "source stopped with a fatal flow error — restarting it "
                             "(nvurisrcbin's own reconnect cannot recover this)",
                        "fatal flow error report")

    def _restart_source(idx, why, kind):
        st = ctx.states.get(idx)
        srcbin = getattr(st, "srcbin", None) if st is not None else None
        if srcbin is None:
            return
        sid = ctx.index_to_id.get(idx, f"src_{idx}")
        if not restart_policy.observe(time.monotonic(), idx):
            logger.info("[%s] already restarting; ignoring the %s", sid, kind)
            return
        logger.error("[%s] %s", sid, why)
        st._stall_restarts = getattr(st, "_stall_restarts", 0) + 1
        flight_rec.note(sid, "fatalrestart" if kind.startswith("fatal") else "stallrestart")
        try:
            srcbin.set_state(Gst.State.NULL)
            srcbin.set_state(Gst.State.PLAYING)
        except Exception as exc:
            logger.error("[%s] source restart failed: %s", sid, exc)

    stall_seconds = gcfg.stall_watch.stall_seconds

    def _check_stalled_sources():
        if not _is_playing[0]:
            return True  # a paused pipeline delivers nothing on purpose
        check = gcfg.stall_watch.stalled(time.monotonic())
        if check.downstream:
            logger.warning("no frame from ANY source came out of the mux for %.0fs — "
                           "stall is downstream of the sources; not restarting them",
                           stall_seconds)
        if check.hopeless or check.hopeless_downstream:
            # Restarting the source in-process did not help (2026-09-26: after
            # the stream group restarted, two cameras stayed dead through every
            # restart until the whole sidecar restarted). Exit like the
            # main-loop watchdog; the container's restart policy brings every
            # camera back in ~15 s.
            if check.hopeless:
                logger.critical("[%s] still no frame out of the mux after %d stall-watch "
                                "restart(s) — exiting so the container restarts the sidecar",
                                ", ".join(ctx.index_to_id.get(i, f"src_{i}")
                                          for i in check.hopeless),
                                gcfg.stall_watch.max_failed_restarts)
            else:
                logger.critical("[every source] still no frame out of the mux after %d "
                                "downstream stall reports in a row — exiting so the "
                                "container restarts the sidecar",
                                gcfg.stall_watch.max_failed_restarts + 1)
            logging.shutdown()
            os._exit(71)
        for idx in check.restart:
            _restart_source(idx, f"no frame came out of the mux for {stall_seconds:.0f}s "
                                 "(or longer, backing off) while nothing reported an error "
                                 "— restarting the source",
                            "stall report")
        return True  # keep GLib timer alive

    if stall_seconds > 0:
        GLib.timeout_add(1000, _check_stalled_sources)

    def on_message(_bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            logger.warning("EOS — pipeline reached end of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error("bus error: %s: %s", err, dbg)
            flight_rec.note(flight.cam_of(msg.src.get_name() if msg.src else "", ctx.index_to_id),
                            flight.classify_bus(f"{err} {dbg}", True))
            # A record-branch encoder that the NVENC driver rejects (e.g. "S_FMT
            # failed ... Unknown error -1" past the GPU's concurrent-session
            # ceiling) would otherwise cascade into its source chain. Stop
            # creating branches and tear down the failed one so the pipeline
            # keeps running.
            src = msg.src.get_name() if msg.src else ""
            if src.startswith("rec_enc_") and record_scheduler is not None:
                try:
                    _fidx = int(src.rsplit("_", 1)[-1])
                except ValueError:
                    _fidx = None
                record_scheduler.note_failure(idx=_fidx, reason=f"{err}")
            _maybe_restart_dead_source(dbg)
        elif t == Gst.MessageType.QOS:
            try:
                _fmt, _processed, _dropped = msg.parse_qos_stats()
                if _dropped > 0 and msg.src is not None:
                    flight_rec.note_qos(msg.src.get_name(), int(_dropped))
            except Exception:
                pass
        elif t == Gst.MessageType.WARNING:
            # nvurisrcbin posts "No data from source since last N sec. Trying reconnect"
            # here, immediately before it resets a source. Dropping these left the
            # "Resetting source" loop undiagnosable — the reset line itself names no cause.
            err, dbg = msg.parse_warning()
            src = msg.src.get_name() if msg.src else "?"
            logger.warning("bus warning from %s: %s (%s)", src, err, dbg)
            flight_rec.note(flight.cam_of(src, ctx.index_to_id), flight.classify_bus(f"{err} {dbg}"))
            # rtspsrc posts its fatal "streaming stopped" as a WARNING, ~10s
            # before the record bin re-reports the same incident as an ERROR.
            # Waiting for the ERROR would double the outage for no benefit.
            _maybe_restart_dead_source(dbg)
    bus.connect("message", on_message)

    # Every recovery path above runs on this main loop, and nvurisrcbin's own
    # reconnect can deadlock it (see MainLoopWatchdog). A thread outside the
    # loop ends the process when it stops beating; the container restarts it.
    watchdog = MainLoopWatchdog(float(cfg.get("main_loop_stall_seconds", 60.0)),
                                now=time.monotonic())

    def _beat():
        watchdog.beat(time.monotonic())
        return True  # keep GLib timer alive

    def _watch_main_loop():
        while True:
            time.sleep(watchdog.poll_seconds)
            now = time.monotonic()
            if watchdog.wedged(now):
                logger.critical("GLib main loop has not run for %.0fs (wedged, e.g. in a "
                                "source reset) — exiting so the container restarts the sidecar",
                                watchdog.silent_for(now))
                logging.shutdown()
                os._exit(70)

    if watchdog.timeout_seconds > 0:
        GLib.timeout_add(1000, _beat)
        threading.Thread(target=_watch_main_loop, name="main-loop-watchdog",
                         daemon=True).start()

    # `docker stop` sends SIGTERM. Without a handler Python dies on it at once and the cleanup below never runs, so
    # this process's iris:status keys (no expiry) would keep its streams listed as running after it stopped.
    def _on_sigterm():
        logger.info("SIGTERM: stopping the pipeline")
        loop.quit()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _on_sigterm)

    try:
        loop.run()
    finally:
        if record_scheduler is not None:
            # Cancel any record branches still pending staggered creation; the
            # pipeline is about to be torn down.
            record_scheduler.stop()
        ctx.modules.run("stop")
        pipeline.set_state(Gst.State.NULL)
        if uploader is not None:
            # Drain pending segment uploads so no finalized fragment is lost on exit.
            uploader.stop()
        # Drain pending render/publish jobs (last frames + face-recognition
        # enqueues) so the tail of the stream is not lost on shutdown.
        publisher.stop()
        # Delete status keys on clean shutdown so the discovery loop in StreamManager
        # does not see these sources on the next run (or until the sidecar is back up).
        for s in sources:
            try:
                gcfg.r.delete(f"iris:status:{gcfg.key_id(s['id'])}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
