"""Engine configuration (SidecarConfig): every knob read from the sidecar yaml."""

import os
import redis
from .source_recovery import (MainLoopWatchdog, SourceRestartPolicy,  # noqa: E402
                             SourceStallWatch, dead_source_index)
from .segment_index import (SegmentBoundaryTracker, StreamClockOffsets,  # noqa: E402
                           has_parameter_sets)
from . import telemetry
from .sources import stream_key_id

from .gst_util import logger




class SidecarConfig:
    """Shared (non-per-source) runtime config + the Redis client."""
    def __init__(self, cfg, role=None, shard_id=0):
        self.r = redis.Redis.from_url(cfg["redis_url"])
        self.shard_id = shard_id    # this replica's IRIS_SHARD_ID (0 when not sharded)
        # The role this process runs as and whether its per-camera Redis keys carry it (`role_scoped_keys`, off by
        # default): see key_id / sources.stream_key_id.
        self.role = role
        self.role_scoped_keys = bool(cfg.get("role_scoped_keys", False))
        # Per-stage timing telemetry (benchmarking mode). Off by default — the
        # hot path then costs one bool check per stage. Enabled by yaml
        # telemetry_enabled or env IRIS_TELEMETRY; the benchmark harness
        # (stream_capacity.py --telemetry) rewrites the yaml per cell.
        _env_tlm = os.environ.get("IRIS_TELEMETRY", "")
        self.telemetry_enabled = (bool(cfg.get("telemetry_enabled", False))
                                  or _env_tlm not in ("", "0", "false", "False"))
        telemetry.configure(enabled=self.telemetry_enabled,
                            interval=float(cfg.get("telemetry_interval", 5.0)),
                            logger=logger)
        if self.telemetry_enabled:
            logger.info("telemetry ENABLED — per-stage [telemetry] lines will be "
                        "logged every %ss", cfg.get("telemetry_interval", 5.0))
        # Prometheus /metrics exporter port (compose-internal; scraped by the
        # monitoring stack's prometheus service). Only used when telemetry is on.
        self.telemetry_prometheus_port = int(cfg.get("telemetry_prometheus_port", 9100))
        self.sample_interval = 1.0 / float(cfg.get("sample_fps", 2))
        self.stream_interval = 1.0 / float(cfg.get("stream_fps", 15))
        # Optional processing throttle: when process_fps is set (e.g. 5), pgie
        # inference (nvinfer `interval`) runs on every Nth batch only
        # (N = stream_fps / process_fps); nvinfer propagates its last detections
        # to the skipped batches, so the tracker and the probe still see every
        # frame. The probe runs its heavy analysis pass (zones/faces/meta) on
        # analyzed frames only, but publishes the preview JPEG EVERY frame at
        # stream_fps — this is the knob that gives "native-rate source + smooth
        # preview + lower-fps analysis". (analysis_fps, the alternative
        # videorate pre-mux cap, also cuts tracker/convert work but caps the
        # preview at the analysis rate; see sidecar.yaml.)
        self.process_fps = float(cfg.get("process_fps", 0))
        self.infer_interval = 1
        if self.process_fps > 0:
            _sfps = float(cfg.get("stream_fps", 15))
            self.infer_interval = max(1, int(round(_sfps / self.process_fps)))
        # Optional analysis-branch frame-rate cap. When > 0, a videorate element is
        # inserted on each source's mux branch (between the tee and nvstreammux) so the
        # analysis pipeline (nvinfer -> nvtracker -> nvvideoconvert -> probe) only sees
        # analysis_fps frames per second per source, instead of every decoded frame at
        # the source's native rate. This is the real fix for the capacity bottleneck:
        # process_fps/infer_interval only throttles nvinfer inference, but the tracker,
        # convert, and probe still ran on every full-rate frame. Capping the branch here
        # drops frames before they reach any of that work. The record branch (off the
        # same tee) is NOT throttled, so recordings stay full-rate. NOTE: the preview
        # probe sits on this branch, so the published preview is capped at analysis_fps
        # too (coherent but low-fps for low values). 0 = disabled.
        self.analysis_fps = float(cfg.get("analysis_fps", 0))
        # When the analysis branch is capped to analysis_fps, the branch already only
        # sees that many frames/s, so infer_interval must be 1 (infer every frame that
        # reaches the branch). Otherwise process_fps's infer_interval would compound
        # with the cap and throttle inference down to analysis_fps/interval.
        if self.analysis_fps > 0:
            self.infer_interval = 1
        # Optional decoder-level frame drop: tells nvurisrcbin's internal decoder to
        # decode every Nth frame only (N = decoder_drop_interval). This cuts decode
        # GPU work itself (the single biggest pipeline cost per stream per the
        # capacity benchmark), not just the post-decode analysis. WARNING: the record
        # branch taps the same tee as the mux branch, so recordings are ALSO capped at
        # ~stream_fps/decoder_drop_interval. Prefer analysis_fps (which keeps
        # recordings full-rate) unless decode is the limiting factor. 0 = disabled.
        self.decoder_drop_interval = int(cfg.get("decoder_drop_interval", 0))
        self.jpeg_quality = int(cfg.get("jpeg_quality", 80))
        # GPU JPEG encode (nvjpegenc) for the published preview frames instead
        # of CPU cv2.imencode: moves the ~5-15 ms/frame encode off the CPU onto
        # the GPU. The publisher falls back to cv2.imencode automatically if the
        # element is unavailable or the encoder pipeline wedges.
        self.gpu_jpeg = bool(cfg.get("gpu_jpeg", True))
        # How many GPU JPEG encoders the frame publisher runs in parallel (1 = a single shared one,
        # which saturates at ~150 frames/s: each encode holds its lock ~6 ms). Cameras are spread over them.
        self.jpeg_encoders = max(1, int(cfg.get("jpeg_encoders", 1)))
        # Viewer-gated preview: when true, the frame publisher only encodes +
        # publishes the preview JPEG for streams that currently have a viewer
        # (the web server refreshes iris:watched:<id> per /frame or MJPEG
        # request; see ViewerGate). Unwatched streams skip the GPU encode — the
        # expensive part — while analysis, the meta feed, and recognition still
        # run at full rate. Off by default because chat/VLM context also reads
        # iris:jpeg: with gating on, those only cover recently-viewed cameras.
        self.viewer_gated_preview = bool(cfg.get("viewer_gated_preview", False))
        # Pipeline preview: when true (default) the sidecar draws boxes + JPEG-
        # encodes every preview frame and publishes iris:jpeg:<id> — the classic
        # annotated-preview path (this is what surface_copy/publish.draw cost).
        # When false, DeepStream is INFERENCE-ONLY: the probe skips the surface
        # copy, the publisher skips draw + encode, and the frontend plays the
        # source's raw RTSP (iris:status:<id> `preview_uri`) with boxes overlaid
        # from iris:meta:<id>. iris:meta and face recognition still run at the
        # analysis cadence; recognition only copies the frame when faces exist.
        self.pipeline_preview = bool(cfg.get("pipeline_preview", True))
        # Preview label text: main's small label by default; a product with long labels (vehicle
        # id + global id + a reading) can enlarge it with preview_label_scale / preview_label_thickness.
        self.label_scale = float(cfg.get("preview_label_scale", 0.5))
        self.label_thickness = int(cfg.get("preview_label_thickness", 1))
        self.maxlen = int(cfg.get("stream_maxlen", 64))
        self.conf_threshold = float(cfg.get("conf_threshold", 0.0))
        # Draw boxes for shadow-tracked objects (detector missed them this frame but the
        # tracker is still localizing) so boxes don't flicker as detection drops in/out.
        # False => only detector-confident objects get boxes (the pre-flicker-fix behavior).
        self.draw_shadow_tracks = bool(cfg.get("draw_shadow_tracks", True))
        rc = cfg.get("relevant_classes")
        self.relevant = set(rc) if rc else None
        self.capture_w = int(cfg.get("capture_width", cfg.get("mux_width", 1280)))
        self.capture_h = int(cfg.get("capture_height", cfg.get("mux_height", 720)))
        self.mux_w = int(cfg.get("mux_width", 1280))   # detection coords are in mux resolution
        self.mux_h = int(cfg.get("mux_height", 720))
        # Camera clock from RTCP (issue 1 Phase 2) -- see the attach-sys-ts
        # note where nvstreammux is built. Process-start-only.
        self.rtcp_camera_clock = bool(cfg.get("rtcp_camera_clock", True))
        self.record_enabled = bool(cfg.get("record_enabled", True))
        self.record_seconds = float(cfg.get("record_seconds", 8))
        self.record_bitrate_kbps = int(cfg.get("record_bitrate_kbps", 2000))
        # Optional record-branch frame-rate cap. When > 0, a videorate + capsfilter
        # pair is inserted between each source's tee and its record queue, so the
        # nvv4l2h264enc only encodes record_fps frames/s instead of the source's
        # native full rate. Alert/chat clips only need a few fps for previews, and
        # capping the encode rate cuts concurrent NVENC demand ~3x on multi-stream
        # deployments — the 6+-stream full-rate case can wedge NVENC on consumer
        # GPUs (sessions open but never encode, stalling the whole pipeline).
        # 0 = full rate (legacy behavior).
        self.record_fps = float(cfg.get("record_fps", 0))
        # Stagger between dynamic record-branch creations after the pipeline is
        # PLAYING. nvv4l2h264enc opens its NVENC session during the PLAYING state
        # change; with a static build every encoder opens at once in the single
        # set_state(PLAYING), and on some GPUs (observed on an RTX 5050 laptop)
        # 6+ concurrent opens deadlock the NVENC driver — sessions open in
        # BLOCKING mode but never encode a frame, freezing the whole pipeline.
        # Creating each record branch one at a time (_RecordBranchScheduler)
        # serializes the opens (verified to 14 sources at 2.5s). The cost is a
        # recording startup ramp of ~N * this interval. ms; 0 disables
        # staggering (legacy all-at-once behavior, unsafe above ~5 branches).
        self.record_stagger_ms = int(cfg.get("record_stagger_ms", 2500))
        # Depth of the per-source queue between each tee and nvstreammux, in
        # BUFFERS (NVMM surfaces). Decouples the analysis branch from the record
        # branch hanging off the same tee. 0 = no queue (direct link). See the
        # sizing note at the construction site: ~1.4MB/frame at 720p, per stream.
        self.analysis_queue_buffers = int(cfg.get("analysis_queue_buffers", 2))
        # Smart Video Record (RTSP only). See _configure_svr_source.
        self.recorder_default = cfg.get("recorder", "auto")
        self.svr_cache_seconds = int(cfg.get("svr_cache_seconds", 15))
        self.svr_duration_seconds = int(cfg.get("svr_duration_seconds", 10))
        self.svr_coalesce_seconds = float(cfg.get("svr_coalesce_seconds", 2.5))
        # Mitigation, not correctness -- see SvrTrigger's docstring (svr_trigger.py)
        # for the traced NVIDIA SmartRecord/nvurisrcbin use-after-free this is
        # working around. 0.0 preserves the original instant-rearm behavior.
        self.svr_rearm_cooldown_seconds = float(cfg.get("svr_rearm_cooldown_seconds", 0.0))
        self.svr_dir = cfg.get("svr_dir", "/workspace/iris/data/alert_svr")
        self.svr_dir_app = cfg.get("svr_dir_app", "data/alert_svr")
        # How long to ignore further fatal reports from a source we just
        # restarted. One failure is reported several times over ~10s (see
        # SourceRestartPolicy), and a restart itself takes a few seconds, so
        # this must comfortably exceed both or the source is torn down again
        # while it is still coming up.
        self.source_restart_cooldown_seconds = float(
            cfg.get("source_restart_cooldown_seconds", 30.0))
        # A source whose frames stop coming out of the mux with its RTSP
        # session still up posts nothing on the bus (see SourceStallWatch);
        # after this many seconds without one of its frames at the probe it is
        # restarted like a dead one. 0 disables.
        # source_stall_max_failed_restarts > 0 (default 0 = off) ends the
        # process once that many restarts of one source brought no frame back.
        self.stall_watch = SourceStallWatch(
            float(cfg.get("source_stall_seconds", 15.0)),
            max_backoff_seconds=float(cfg.get("source_stall_max_backoff_seconds", 300.0)),
            max_failed_restarts=int(cfg.get("source_stall_max_failed_restarts", 0)),
            failed_restart_window_seconds=float(
                cfg.get("source_stall_failed_restart_window_seconds", 600.0)))
        # source_id -> SvrTrigger, built per smart-record source in build_pipeline.
        self.svr_triggers = {}
        # source_id -> SvrFileTracker, built alongside svr_triggers. See
        # _on_sr_done: correlates each session's sr-done to the file it actually
        # wrote, fixing the mtime-guess race _newest_svr_file used to have.
        self.svr_file_trackers = {}
        # NOTE: there is deliberately NO record_max_sessions knob. How many
        # concurrent NVENC encode sessions a GPU supports is hardware-specific
        # and even varies between boots on the same GPU (measured 12-14 on an
        # RTX 5050 laptop; ~8 on older consumer parts; far more on datacenter
        # GPUs), so any configured cap would be wrong elsewhere. The scheduler
        # MEASURES the ceiling at startup with a throwaway subprocess probe
        # (perception/engine/nvenc_probe.py) and caps record branches at the measured
        # number; note_failure() is the backstop for a driver that rejects a
        # session below the measured limit.
        self.record_tmp_dir = cfg.get("record_tmp_dir", "/tmp/iris-segments")
        self.storage_backend = cfg.get("storage_backend", "minio")
        self.record_bucket = cfg.get("minio_bucket", "iris-segments")
        self.local_segments_dir = cfg.get(
            "local_segments_dir", "/workspace/iris/data/segments")
        self.segment_debug_logging = bool(cfg.get("segment_debug_logging", True))
        self.pts_parity_logging = bool(cfg.get("pts_parity_logging", False))
        # Per-stream mapping from the recorder's pre-mux pts onto the probe's
        # post-mux clock; measured live. Drift tracking (for recordings whose PTS jumps, i.e.
        # looped test files) is opt-in: `clock_drift_tracking: true` in a product's overlay.
        self.clock_offsets = StreamClockOffsets(
            track_drift=bool(cfg.get("clock_drift_tracking", False)))
        self.minio = None
        if self.record_enabled and self.storage_backend == "minio":
            from minio import Minio
            self.minio = Minio(
                cfg.get("minio_endpoint", "minio:9000"),
                access_key=os.environ.get("MINIO_ACCESS_KEY", cfg.get("minio_access_key", "minioadmin")),
                secret_key=os.environ.get("MINIO_SECRET_KEY", cfg.get("minio_secret_key", "minioadmin")),
                secure=bool(cfg.get("minio_secure", False)),
            )
            if not self.minio.bucket_exists(self.record_bucket):
                self.minio.make_bucket(self.record_bucket)
        elif self.record_enabled and self.storage_backend == "local":
            os.makedirs(self.local_segments_dir, exist_ok=True)

    def key_id(self, source_id: str) -> str:
        """The id this process's Redis keys for `source_id` carry: the camera id, or `<camera>@<role>` with
        role_scoped_keys (sources.stream_key_id)."""
        return stream_key_id(source_id, self.role, self.role_scoped_keys)
