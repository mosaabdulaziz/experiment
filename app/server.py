import asyncio
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

import redis
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool, StrictFloat

from .alert_clips import AlertClipService
from .camera import Camera
from .chat_search import (
    run_sweep, SweepMarkerGate, parse_sweep_marker,
    clamp_answer_timestamps, offsets_bounds, select_result_streams,
    snapshots_newest_offset, segments_for_found,
)
from .chat_sessions import ChatSessionStore
from .clip_builder import build_preview_clip
from .clip_store import ClipStore
from .depth_images import DepthImages
from .cv_enricher import CVEnricher
from .events import EventStore
from .face_recognition import router as face_recognition_router, init_redis as init_face_redis
from .frames_clip import build_frames_clip
from .personas import DEFAULT_PERSONA, PERSONAS
from .local_chat_context import LocalChatContext
from .metrics import MetricsRegistry, render_table
from .mode import VALID_MODES, analysis_enabled
from .scheduler import Scheduler
from .similarity import CommentFilter
from .tracking import TrackingService
from .vlm_live import router as vlm_live_router
from .hazards_api import router as hazards_router
from .features import ZONE_FEATURES, camera_of, ppe_items
from .vehicle_gallery import router as vehicle_gallery_router, init_redis as init_vehicle_redis, init_storage as init_vehicle_storage
from .sidecar_config import module_enabled, module_setting, running_sidecar_config
from .zones import (
    _zones_file_path, mux_resolution,
    _load_zones, _save_zones, stream_zones, zone_label,
    effective_proximity, proximity_defaults, proximity_problem, stored_proximity,
)

logger = logging.getLogger(__name__)
_chat_log = logging.getLogger("iris.chat")
_metrics_log = logging.getLogger("iris.metrics")

UPLOAD_DIR = "data/uploads"


async def _metrics_loop(registry, get_depths, get_worker, get_sweeps, interval):
    """Periodically log the per-stream analysis status table under iris.metrics."""
    while True:
        try:
            await asyncio.sleep(interval)
            rows = registry.rows(now=time.monotonic())
            if not rows:
                continue
            table = render_table(rows, get_depths(), get_worker(), get_sweeps(),
                                 datetime.now().strftime("%H:%M:%S"))
            _metrics_log.info("\n%s", table)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let the loop die
            _metrics_log.warning("metrics table failed: %s", exc)


async def _segment_prune_loop(segment_store, retention_seconds, interval=300.0):
    """Background task: periodically delete recorded segments (+ their index
    entries) past `retention_seconds`. Blocking Redis/MinIO calls run in the
    default executor so they don't stall the event loop."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            await asyncio.sleep(interval)
            removed = await loop.run_in_executor(None, segment_store.expire, retention_seconds)
            if removed:
                logger.info("Pruned %d expired segment(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("segment prune failed: %s", exc)


class PromptUpdate(BaseModel):
    instructions: str


class LanguageUpdate(BaseModel):
    language: str


class ModeUpdate(BaseModel):
    mode: str


class ChatMessage(BaseModel):
    message: str
    session_id: str | None = None
    mode: str = "normal"


class ZonePoint(BaseModel):
    x: float
    y: float


class ZoneIntrusion(BaseModel):
    mode: Literal["restricted", "valid", "off"] = "restricted"
    classes: list[str] = Field(default_factory=lambda: ["person"])


class ZoneOvercrowding(BaseModel):
    enabled: bool = False
    threshold: int = 5
    classes: list[str] = Field(default_factory=lambda: ["person"])


class ZoneProximity(BaseModel):
    """The proximity module's block (perception/modules/proximity/README.md); unset values take the defaults."""
    enabled: bool = False
    person_classes: list[str] | None = None
    vehicle_classes: list[str] | None = None
    max_gap: float | None = None
    stationary_window_s: float | None = None
    stationary_motion: float | None = None
    persist_s: float | None = None
    grace_s: float | None = None
    cooldown_s: float | None = None
    depth_check: bool | None = None
    depth_max_gap: float | None = None      # the depth check's cut-off in this zone (Ticket 6)
    hidden_feet: str | None = None


class ZoneFeature(BaseModel):
    """A safety feature's block on a zone (`ppe`, `fall`, `fire`): on, the feature alerts only inside its zones on that
    camera; a camera with none of them covers the whole frame (perception/lib/feature_zones.py)."""
    enabled: bool = False


class PpeZoneFeature(ZoneFeature):
    """PPE's block may also name the equipment checked in that zone (some of the module's required_items: hard hat,
    vest); without `items`, everything the module checks."""
    items: list[str] | None = None




class ProximityPreviewUpdate(BaseModel):
    """The proximity live preview's layers to switch, how far apart (person heights) a person and a moving vehicle
    still get a line, and how alert clips draw objects; what is left out keeps its state."""
    masks: StrictBool | None = None
    lines: StrictBool | None = None
    boxes: StrictBool | None = None
    depth: StrictBool | None = None     # the depth map layer (cameras with a depth-check zone)
    line_range: Annotated[StrictFloat, Field(ge=0.25, le=10.0)] | None = None
    clip_style: Literal["masks", "boxes"] | None = None     # how alert clips and snapshots draw objects


class ZoneRequest(BaseModel):
    id: str | None = None
    name: str = ""
    points: list[ZonePoint]
    intrusion: ZoneIntrusion = Field(default_factory=ZoneIntrusion)
    overcrowding: ZoneOvercrowding = Field(default_factory=ZoneOvercrowding)
    proximity: ZoneProximity | None = None
    ppe: PpeZoneFeature | None = None
    fall: ZoneFeature | None = None
    fire: ZoneFeature | None = None


class ZonesRequest(BaseModel):
    zones: list[ZoneRequest] = Field(default_factory=list)


class ConnectionManager:
    def __init__(self):
        self._sockets: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._sockets.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self._sockets.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, payload: dict):
        msg = json.dumps(payload, default=str)
        dead = []
        for ws in self._sockets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def _emit_config_warnings(config: dict, alert_clip_cfg: dict, segment_store,
                          storage_backend: str, clip_service_enabled: bool) -> None:
    """Warn about config.yaml / sidecar.yaml key pairs that fail silently.

    Deepstream-only. Raises nothing the caller needs to handle -- create_app wraps
    the call, because a diagnostic must never be able to prevent startup."""
    clips_on = bool(alert_clip_cfg.get("enabled", False))
    max_wait = float(alert_clip_cfg.get("max_wait_seconds", 20.0))
    if clips_on and segment_store is None:
        logger.warning(
            "alerts.clip.enabled is true but segments.enabled is false -- alert clips are "
            "cut from recorded segments, so no alert will get a clip.")
    side_cfg = running_sidecar_config(config)
    # Frame supply for analysis. The sidecar embeds a JPEG in iris:meta only when it
    # encoded one for that frame, so two preview keys can silently starve the VLM.
    # SidecarConfig defaults: pipeline_preview True, viewer_gated_preview False.
    if side_cfg and bool(config.get("monitoring", {}).get("vlm_enabled", False)):
        if not bool(side_cfg.get("pipeline_preview", True)):
            logger.warning(
                "monitoring.vlm_enabled is true but sidecar.yaml has pipeline_preview: "
                "false -- no JPEG is encoded into iris:meta, so analysis will receive no "
                "frames. Set pipeline_preview: true.")
        elif bool(side_cfg.get("viewer_gated_preview", False)):
            logger.warning(
                "monitoring.vlm_enabled is true and sidecar.yaml has viewer_gated_preview: "
                "true -- a stream nobody is watching in the browser encodes no JPEG, so "
                "analysis will receive no frames for it.")
    # An empty dict means sidecar.yaml was unreadable; comparing against its defaults
    # would invent mismatches, so only check when it was actually read. The defaults
    # below mirror SidecarConfig in perception/engine/config.py -- keep them in step.
    if not (side_cfg and segment_store is not None):
        return
    recording = bool(side_cfg.get("record_enabled", True))
    side_backend = side_cfg.get("storage_backend", "minio")
    # "recording is on" is no longer one boolean: RTSP sources record via
    # smart-record (no segments, no NVENC session), file:// sources via the encode
    # branch. Warning about record_enabled on an all-RTSP fleet would be a false
    # alarm; staying silent on a file:// source would hide a real break.
    from perception.engine.recorder_kind import recorder_kind
    from perception.engine.sources import resolve_sources
    rec_default = side_cfg.get("recorder", "auto")
    kinds = {}
    for s in resolve_sources(side_cfg):
        try:
            kinds[s["id"]] = recorder_kind(s["uri"], s.get("recorder", rec_default))
        except ValueError as exc:
            logger.warning("sidecar.yaml source %s: %s", s["id"], exc)
    encode_sources = sorted(i for i, k in kinds.items() if k == "encode")
    svr_sources = sorted(i for i, k in kinds.items() if k == "svr")

    if svr_sources and clips_on:
        svr_cache = float(side_cfg.get("svr_cache_seconds", 15))
        pad_s = float(alert_clip_cfg.get("pad_seconds", 5.0))
        if svr_cache <= pad_s:
            logger.warning(
                "sidecar.yaml's svr_cache_seconds (%gs) is not greater than "
                "alerts.clip.pad_seconds (%gs) -- start-sr cannot reach further back "
                "than the cache holds, so every smart-record clip loses pre-roll and "
                "may open after its own alert.", svr_cache, pad_s)

    if encode_sources and not recording:
        msg = ("segments.enabled is true but record_enabled is false in sidecar.yaml, "
               f"and {len(encode_sources)} source(s) ({', '.join(encode_sources)}) "
               "use the encode recorder -- no segments are written for them, so chat "
               "search FOUND results fall back to a frames preview")
        if clips_on:
            msg += (f"; their fired alerts will also wait {max_wait:g}s before "
                    "resolving 'unavailable'")
        logger.warning("%s.", msg)
    elif recording and side_backend != storage_backend:
        logger.warning(
            "segments.storage_backend is %r but sidecar.yaml's storage_backend is %r -- "
            "the sidecar writes segment bytes where IRIS does not read them, so every "
            "clip build (chat search and fired alerts alike) will fail to fetch them.",
            storage_backend, side_backend)
    if recording and clip_service_enabled:
        pad = float(alert_clip_cfg.get("pad_seconds", 5.0))
        rec_secs = float(side_cfg.get("record_seconds", 8.0))
        # The segment holding T+pad is only indexed once it rotates closed, up to a
        # full record_seconds later, so the wait must outlast pad + record_seconds.
        if max_wait <= pad + rec_secs:
            logger.warning(
                "alerts.clip.max_wait_seconds (%gs) is not greater than "
                "alerts.clip.pad_seconds + sidecar.yaml's record_seconds "
                "(%g + %g = %gs) -- the segment covering the end of the clip window is "
                "only indexed once it rotates closed, so the wait can expire before the "
                "window is ever covered and every alert resolves 'unavailable'.",
                max_wait, pad, rec_secs, pad + rec_secs)


def _detection_classes(config: dict) -> list[str]:
    """Object classes the pipeline surfaces (from sidecar.yaml relevant_classes) — the set
    the zone editor offers for intrusion/overcrowding. Falls back to person."""
    cfg = running_sidecar_config(config)
    classes = cfg.get("relevant_classes")
    if isinstance(classes, list) and classes:
        return [str(c) for c in classes]
    return ["person"]


def _validate_zone(z, allowed: set, prox_defaults: "dict | None" = None, proximity_running: bool = False,
                   ppe_choices: "list | None" = None) -> str | None:
    """Per-zone save validation. Returns an error message, or None when the zone is valid."""
    if len(z.points) < 3:
        return "a zone needs at least 3 points"
    if any(not (0 <= p.x <= 1 and 0 <= p.y <= 1) for p in z.points):
        return "points must be normalized 0-1 coordinates"
    proximity_on = z.proximity is not None and z.proximity.enabled
    feature_on = any(getattr(z, name) is not None and getattr(z, name).enabled for name in ZONE_FEATURES)
    if z.intrusion.mode == "off" and not z.overcrowding.enabled and not proximity_on and not feature_on:
        return "enable at least one detection feature"
    if z.ppe is not None and z.ppe.enabled and z.ppe.items is not None:     # off: kept as it was, not checked
        choices = ppe_choices or []
        if not z.ppe.items:
            return "ppe needs at least one item to check"
        unknown = [i for i in z.ppe.items if i not in choices]
        if unknown:
            return f"ppe items must be from {', '.join(choices)}: not {', '.join(unknown)}"
    for feature, classes, active in (
        ("intrusion", z.intrusion.classes, z.intrusion.mode != "off"),
        ("overcrowding", z.overcrowding.classes, z.overcrowding.enabled),
    ):
        if not active:
            continue
        if not classes:
            return f"{feature} needs at least one object class"
        unknown = [c for c in classes if c not in allowed]
        if unknown:
            return f"unknown classes: {', '.join(unknown)}"
    if z.overcrowding.enabled and z.overcrowding.threshold < 1:
        return "overcrowding threshold must be at least 1"
    if z.proximity is not None:
        # only what the zone sets differs from the (trusted) defaults; the pipeline's classes matter only when this
        # pipeline runs the module and the block is on (elsewhere the section is hidden: nothing could fix it)
        block = stored_proximity(z.proximity.model_dump(exclude_none=True), prox_defaults or {})
        problem = proximity_problem(block, allowed if block["enabled"] and proximity_running else None)
        if problem is not None:
            return problem
    return None


def _mark_watched(src) -> None:
    """Mark a stream as being viewed, feeding the sidecar's viewer-gated preview
    encoding (perception/engine/sources.ViewerGate). Local-backend Camera objects don't
    implement mark_watched — no-op there. Never raises: a marker failure must
    not fail the stream request."""
    mark = getattr(src, "mark_watched", None)
    if mark is not None:
        try:
            mark()
        except Exception:
            pass


def _follow_source_changes(stream_manager):
    """The hook StreamManager calls when the pipeline's sources change (the safety product's Apply rewrites its role
    configs): the zones are announced again, so each zones process re-renders its ROI config for its own cameras."""
    def follow():
        stream_manager.zones_bump()
    return follow


def _safety_snapshot(stream_manager):
    """sid -> the current JPEG of that stream, or of any stream of that camera (a health incident names the camera),
    for the Telegram message; None when there is none."""
    def snapshot(sid):
        if stream_manager is None or not sid:
            return None
        src = stream_manager.source_for_camera(sid)
        try:
            return src.grab_display_frame() if src is not None else None
        except Exception:  # noqa: BLE001 -- a message without a picture is still a message
            return None
    return snapshot


def create_app(config: dict) -> FastAPI:
    perception_backend = config.get("perception", {}).get("backend", "local")
    if perception_backend == "deepstream":
        from .perception_source import PerceptionSource
        camera = PerceptionSource(config)
        logger.info("Perception backend: deepstream (reading Redis %s)",
                    config.get("perception", {}).get("redis_url"))
    else:
        camera = Camera(config)
    store = EventStore(config)
    metrics = MetricsRegistry()
    _active_sweeps = {"n": 0}
    vlm_enabled = bool(config.get("monitoring", {}).get("vlm_enabled", False))
    if vlm_enabled:
        from .analyzer import Analyzer
        analyzer = Analyzer(config, metrics=metrics)
    else:
        analyzer = None
        logger.info("VLM disabled (vlm_enabled=false) — only detection pipeline running")
    manager = ConnectionManager()
    comment_filter = CommentFilter(config)
    enricher = CVEnricher(config)
    scheduler = Scheduler(config, camera, analyzer, store, manager, comment_filter, enricher,
                          metrics=metrics)
    segment_store = None
    _redis_url = config.get("perception", {}).get("redis_url", "redis://localhost:6379/0")
    init_face_redis(_redis_url)
    # The multi-camera tracking use case (tracking service, vehicle gallery, counting lines,
    # truck-tracking feed) is off unless `mtmc.enabled`: attendance, barber and fire run this
    # same app and must not start it. Hazard Detection has its own switch.
    multicam_enabled = bool((config.get("mtmc") or {}).get("enabled", False))
    hazards_enabled = bool((config.get("hazards") or {}).get("enabled", False))
    if multicam_enabled:
        init_vehicle_redis(_redis_url)
        init_vehicle_storage({"storage_backend": "local", **(config.get("vehicle_crops") or {})})
    _seg_cfg = config.get("segments", {})
    segments_retention_seconds = float(_seg_cfg.get("retention_seconds", 1800))
    _storage_backend = _seg_cfg.get("storage_backend", "minio")
    _clip_debug_logging = bool(_seg_cfg.get("clip_debug_logging", True))
    _clip_padding_seconds = float(_seg_cfg.get("clip_padding_seconds", 8.0))
    _clip_trim_enabled = bool(_seg_cfg.get("clip_trim_enabled", False))
    _clip_trim_seconds = float(_seg_cfg.get("clip_trim_seconds", 4.0))
    if perception_backend == "deepstream" and bool(_seg_cfg.get("enabled", True)):
        import redis as _redis_mod
        from .segment_store import SegmentStore

        if _storage_backend == "minio":
            from minio import Minio
            _minio_client = Minio(
                _seg_cfg.get("minio_endpoint", "minio:9000"),
                access_key=os.environ.get("MINIO_ACCESS_KEY", _seg_cfg.get("minio_access_key", "minioadmin")),
                secret_key=os.environ.get("MINIO_SECRET_KEY", _seg_cfg.get("minio_secret_key", "minioadmin")),
                secure=bool(_seg_cfg.get("minio_secure", False)),
            )
            _bucket = _seg_cfg.get("minio_bucket", "iris-segments")
            if not _minio_client.bucket_exists(_bucket):
                _minio_client.make_bucket(_bucket)
            segment_store = SegmentStore(_redis_mod.Redis.from_url(_redis_url),
                                         minio_client=_minio_client, bucket=_bucket)
        else:
            _local_dir = _seg_cfg.get("local_dir", "data/segments")
            segment_store = SegmentStore(_redis_mod.Redis.from_url(_redis_url),
                                         backend="local", root_dir=_local_dir)

    _alert_clip_cfg = (config.get("alerts", {}) or {}).get("clip", {}) or {}
    alert_clips = ClipStore(ttl_seconds=float(_alert_clip_cfg.get("ttl_seconds", 21600)),
                            root_dir="data/alert_clips")
    depth_images = DepthImages(ttl_seconds=float(_alert_clip_cfg.get("ttl_seconds", 21600)))
    alert_clip_service = AlertClipService(config, segment_store, alert_clips, store, manager,
                                          redis_client=segment_store.redis if segment_store is not None else None)

    # config.yaml and sidecar.yaml are read by different processes and share several
    # keys. A mismatch fails silently -- segments written where nothing reads them,
    # clips that wait out their timeout and resolve 'unavailable' -- so surface it at
    # startup. This only ever WARNS, so it must never stop the app from starting: an
    # unguarded `from perception.engine.recorder_kind import ...` here crash-looped the iris
    # container, because the image ships only a subset of perception/engine/ (see the COPY in
    # Dockerfile). A diagnostic that takes the process down with it is worse than no
    # diagnostic.
    if perception_backend == "deepstream":
        try:
            _emit_config_warnings(config, _alert_clip_cfg, segment_store,
                                  _storage_backend, alert_clip_service.enabled)
        except Exception as exc:
            logger.warning("startup config cross-checks skipped (%s: %s) -- config "
                           "mismatches between config.yaml and sidecar.yaml will not "
                           "be reported this boot", type(exc).__name__, exc)

    stream_manager = None
    if perception_backend == "deepstream":
        from .stream_manager import StreamManager
        stream_manager = StreamManager(config, analyzer, store, manager, metrics=metrics,
                                       clip_service=alert_clip_service, depth_images=depth_images)
        stream_manager.on_sources_changed.append(_follow_source_changes(stream_manager))
    if perception_backend == "deepstream":
        chat_context = stream_manager.chat
    else:
        chat_context = LocalChatContext(config, camera)

    # Multi-camera vehicle tracking service (None unless mtmc.enabled)
    tracking_service = TrackingService(config, manager) if multicam_enabled else None

    chat_sessions = ChatSessionStore(config)
    chat_max_images = config.get("chat", {}).get("max_chat_images", 12)
    _search_cfg = config.get("chat", {}).get("search", {})
    search_enabled = bool(_search_cfg.get("enabled", True))
    search_max_calls = int(_search_cfg.get("max_calls", 12))
    search_max_seconds = float(_search_cfg.get("max_seconds", 45))
    found_min_confidence = float(_search_cfg.get("found_min_confidence", 0.0))

    _preview_cfg = config.get("chat", {}).get("preview", {})
    preview_enabled = bool(_preview_cfg.get("enabled", True))
    # Playback fps of a frames-built preview. Read again after a spell as dead
    # config: it governed the original frames-based preview before the segment
    # builder superseded it, and now governs the fallback behind that builder.
    preview_fps = float(_preview_cfg.get("fps", 4))
    clips = ClipStore(ttl_seconds=float(_preview_cfg.get("ttl_seconds", 1800)),
                      root_dir=str(_preview_cfg.get("dir", "data/clips")))

    search_round_seconds = float(_search_cfg.get("round_seconds", 8))

    # Bootstrap per-persona instruction store (each persona tracks its own edits independently)
    _persona_id = config.get("monitoring", {}).get("persona", DEFAULT_PERSONA)
    if _persona_id not in PERSONAS:
        _persona_id = DEFAULT_PERSONA
    monitoring = config.setdefault("monitoring", {})
    monitoring["active_persona"] = _persona_id
    monitoring.setdefault("persona_instructions", {p: PERSONAS[p]["instructions"] for p in PERSONAS})
    monitoring["instructions"] = monitoring["persona_instructions"][_persona_id]

    # Bootstrap working mode (proactive | chat). Runtime switches stay in-memory.
    _mode = config.get("monitoring", {}).get("mode", "proactive")
    if _mode not in VALID_MODES:
        _mode = "proactive"
    monitoring["active_mode"] = _mode
    monitoring["chat_in_flight"] = 0

    async def _stream_annotation_loop():
        cv_cfg = config.get("cv", {})
        fps = max(0.5, float(cv_cfg.get("stream_annotation_fps", 3.0)))
        interval = 1.0 / fps
        loop = asyncio.get_event_loop()
        # Dedicated single-threaded executor for stream YOLO. Annotated frames are also
        # pushed to camera's annotated_buffer so _cycle() can reuse them and skip process_batch.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-annotator") as ex:
            while True:
                await asyncio.sleep(interval)
                if not camera.is_running or camera.is_paused or not enricher.enabled:
                    continue
                frame = camera.grab_frame()
                if frame is None:
                    continue
                try:
                    annotated, detections = await loop.run_in_executor(ex, enricher.annotate_single, frame)
                    camera.set_annotated_frame(annotated)
                    camera.push_annotated_frame(annotated, detections, camera.current_offset)
                except Exception as exc:
                    logger.debug("Stream annotation error: %s", exc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await store.init()
        stale = await store.resolve_stale_pending()
        if stale:
            logger.info("Resolved %d enrichment status(es) stranded by a restart", stale)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        if stream_manager is not None:
            await stream_manager.run()
        else:
            asyncio.create_task(scheduler.run())
            asyncio.create_task(chat_context.run())   # local chat frame sampler
        asyncio.create_task(chat_sessions.run_prune_loop())
        asyncio.create_task(clips.run_prune_loop())
        asyncio.create_task(alert_clips.run_prune_loop())
        # Started unconditionally: run() self-guards and reports whether it is live,
        # so a service that is off says so instead of leaving no trace at all.
        asyncio.create_task(alert_clip_service.run())
        # Start multi-camera vehicle tracking service
        if tracking_service is not None:
            asyncio.create_task(tracking_service.run())
        if segment_store is not None:
            asyncio.create_task(_segment_prune_loop(segment_store, segments_retention_seconds))
        log_cfg = config.get("logging", {})
        if log_cfg.get("metrics_enabled", True):
            interval = float(log_cfg.get("metrics_interval_seconds", 15))
            if stream_manager is not None:
                get_depths, get_worker = stream_manager.queue_depths, stream_manager.worker_state
            else:
                get_depths = lambda: {"local": scheduler.pending_depth()}
                get_worker = lambda: None
            asyncio.create_task(_metrics_loop(metrics, get_depths, get_worker,
                                              lambda: _active_sweeps["n"], interval))
        if perception_backend == "local" and config.get("cv", {}).get("annotate_stream", False):
            asyncio.create_task(_stream_annotation_loop())
            logger.info("Stream annotation enabled at %.1f fps", config.get("cv", {}).get("stream_annotation_fps", 3.0))
        safety_tasks = []
        safety_notifier = getattr(_app.state, "safety_notifier", None)
        if safety_notifier is not None:
            safety_notifier.start()
        safety_health = getattr(_app.state, "safety_health", None)
        if safety_health is not None:
            safety_tasks.append(asyncio.create_task(safety_health.run()))
        logger.info("IRIS has started")
        yield
        for task in safety_tasks:
            task.cancel()
        if safety_notifier is not None:
            safety_notifier.stop()
        if tracking_service is not None:
            await tracking_service.stop()
        alert_clip_service.stop()
        if stream_manager is not None:
            await stream_manager.stop()
            camera.close()  # release the back-compat /stream PerceptionSource's Redis too
        elif perception_backend == "local":
            chat_context.stop()
            await camera.stop()
        else:
            camera.close()  # deepstream: just release Redis; don't tell the sidecar to stop

    app = FastAPI(title="IRIS", lifespan=lifespan)
    # Other UIs (e.g. truck-tracking's) call this API from their own origin.
    _cors = config.get("cors_origins")
    if _cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(_cors),
                           allow_methods=["GET", "PUT", "POST", "DELETE"], allow_headers=["*"])
    app.state.segment_store = segment_store
    app.state.stream_manager = stream_manager
    app.include_router(face_recognition_router)
    if multicam_enabled:
        app.include_router(vehicle_gallery_router)
    if monitoring.get("vlm_enabled", False):
        app.include_router(vlm_live_router)
    if hazards_enabled:
        app.include_router(hazards_router)
    # The safety product (products/safety): its settings, cameras, Apply and acknowledgement queue.
    if "safety" in [p.strip() for p in os.environ.get("IRIS_PRODUCT", "").split(",")]:
        # Imported here: every other product runs this app too and needs none of it.
        import redis as _redis_safety
        from products.safety.service import SafetyService
        from products.safety.api import router as safety_router
        from products.safety.camera_health import CameraHealth, expected_streams
        from products.safety.notifier import TelegramNotifier
        app.state.safety = SafetyService(
            config, store=store,
            redis_client=(stream_manager.redis if stream_manager is not None
                          else _redis_safety.Redis.from_url(_redis_url)))
        app.include_router(safety_router)
        # Its alert clips are smart-record files: drawn on, with the alert's zone and objects (app/alert_clips.py).
        alert_clip_service.svr_overlays = True
        app.state.alert_clip_service = alert_clip_service
        # Telegram for the incidents whose feature asks for it (opened, escalated), with the camera's frame.
        app.state.safety_notifier = TelegramNotifier(app.state.safety, snapshot=_safety_snapshot(stream_manager))
        if stream_manager is not None:
            stream_manager.set_notifier(app.state.safety_notifier)
            # Camera health: a camera sending nothing, a feature not running on it, a camera too slow.
            app.state.safety_health = CameraHealth(
                store, manager, stream_manager.redis,
                expected=lambda: expected_streams(running_sidecar_config(config)),
                notifier=app.state.safety_notifier,
                names=lambda: {c["id"]: c["name"] for c in app.state.safety.cameras()})

    # ── Video stream ───────────────────────────────────────────────────────
    @app.get("/stream")
    async def video_stream():
        return StreamingResponse(
            camera.generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── Camera control ─────────────────────────────────────────────────────
    @app.post("/api/camera/start")
    async def camera_start():
        if camera.is_running:
            return JSONResponse({"status": "already_running", "source": camera.source_label, "is_file": camera.is_file})
        await camera.start()
        return JSONResponse({"status": "started", "source": camera.source_label, "is_file": camera.is_file})

    @app.post("/api/camera/stop")
    async def camera_stop():
        if not camera.is_running:
            return JSONResponse({"status": "already_stopped"})
        await camera.stop()
        return JSONResponse({"status": "stopped"})

    @app.get("/api/camera/status")
    async def camera_status():
        return JSONResponse({
            "running": camera.is_running,
            "paused": camera.is_paused,
            "source": camera.source_label,
            "is_file": camera.is_file,
        })

    @app.post("/api/camera/pause")
    async def camera_pause():
        if not camera.is_running or not camera.is_file:
            return JSONResponse({"error": "only applicable to running video files"}, status_code=400)
        camera.pause()
        await manager.broadcast({"type": "status", "status": "paused"})
        return JSONResponse({"status": "paused"})

    @app.post("/api/camera/resume")
    async def camera_resume():
        camera.resume()
        await manager.broadcast({"type": "status", "status": "idle"})
        return JSONResponse({"status": "resumed"})

    @app.get("/api/camera/time")
    async def camera_time():
        return JSONResponse({"offset": camera.current_offset, "is_file": camera.is_file})

    @app.post("/api/camera/use-camera")
    async def camera_use_webcam():
        await camera.stop()
        camera.reset_to_default()
        await camera.start()
        return JSONResponse({"status": "started", "source": camera.source_label, "is_file": camera.is_file})

    @app.get("/api/products")
    async def list_products():
        """Which products this installation shows.

        The dashboard is one build for every customer (MULTI_PRODUCT_PLAN.md
        decision 10); what a given install shows is decided here, at run time,
        from the same IRIS_PRODUCT value that decides which pipeline and
        services this checkout runs. Unset means "everything this build has",
        which is what a developer wants and what an install that predates this
        endpoint got anyway.

        This is a convenience for the menu, never a boundary: every request the
        dashboard makes is still authorised on its own. Hiding a menu entry is
        not access control.
        """
        raw = os.environ.get("IRIS_PRODUCT", "").strip()
        products = [p.strip() for p in raw.split(",") if p.strip()]
        return JSONResponse({"products": products})

    @app.get("/api/cameras")
    async def list_cameras():
        _no_cache = {"Cache-Control": "no-store, no-cache, must-revalidate"}
        if stream_manager is not None:
            return JSONResponse({
                "cameras": stream_manager.list_streams(),
                "pipeline": stream_manager.pipeline_desired(),
                "zones": stream_manager.zones_status(),
            }, headers=_no_cache)
        loop = asyncio.get_event_loop()
        cameras = await loop.run_in_executor(None, Camera.list_cameras)
        current_source = None if camera.is_file else camera._source
        return JSONResponse({"cameras": cameras, "current": current_source}, headers=_no_cache)

    @app.post("/api/streams/start")
    async def streams_start():
        if stream_manager is None:
            return JSONResponse({"error": "deepstream backend only"}, status_code=400)
        stream_manager.pipeline_start()
        return JSONResponse({"status": "ok", "pipeline": "playing"})

    @app.post("/api/streams/stop")
    async def streams_stop():
        if stream_manager is None:
            return JSONResponse({"error": "deepstream backend only"}, status_code=400)
        stream_manager.pipeline_stop()
        return JSONResponse({"status": "ok", "pipeline": "paused"})

    @app.get("/api/proximity/preview")
    async def get_proximity_preview():
        if stream_manager is None:
            return JSONResponse({"available": False, "masks": False, "lines": False})
        return JSONResponse(stream_manager.proximity_preview())

    @app.post("/api/proximity/preview")
    async def set_proximity_preview(update: ProximityPreviewUpdate):
        if stream_manager is None or not stream_manager.proximity_preview()["available"]:
            return JSONResponse({"error": "the pipeline does not run proximity"}, status_code=400)
        change = update.model_dump(exclude_none=True)
        if not change:
            return JSONResponse({"error": "nothing to change: send masks, lines, boxes, line_range or clip_style"},
                                status_code=400)
        try:
            return JSONResponse(stream_manager.set_proximity_preview(change))
        except redis.RedisError as exc:
            return JSONResponse({"error": f"could not save the preview setting: {exc}"}, status_code=503)

    @app.get("/api/detection-classes")
    async def detection_classes():
        return JSONResponse({"classes": _detection_classes(config)})

    @app.get("/api/streams/{stream_id}/zones")
    async def get_zones(stream_id: str):
        sidecar_cfg = running_sidecar_config(config)
        proximity_on = module_enabled(sidecar_cfg, "proximity")
        defaults = proximity_defaults(sidecar_cfg)
        zones = []
        for z in stream_zones(config, stream_id):
            zone = {"id": z["id"], "name": z["name"], "points": [{"x": x, "y": y} for x, y in z["points"]],
                    "intrusion": z["intrusion"], "overcrowding": z["overcrowding"]}
            if proximity_on or "proximity" in z:
                zone["proximity"] = effective_proximity(z.get("proximity"), defaults)
            zone.update({name: z[name] for name in ZONE_FEATURES if name in z})
            zones.append(zone)
        payload = {"zones": zones}
        if module_enabled(sidecar_cfg, "ppe"):
            payload["ppe"] = {"items": ppe_items(sidecar_cfg)}      # what a zone can ask PPE to check
        if proximity_on:
            payload["proximity"] = {"available": True, "defaults": defaults,
                                    "depth_model": bool(module_setting(sidecar_cfg, "proximity", "depth_model", ""))}
        return JSONResponse(payload)

    @app.post("/api/streams/{stream_id}/zones")
    async def save_zones(stream_id: str, body: ZonesRequest):
        # Zones are per camera: a role sidecar's stream (`<camera>@<role>`) saves its camera's, which every feature on
        # that camera reads (app/features.py).
        stream_id = camera_of(stream_id)
        def _respond(payload: dict) -> JSONResponse:
            # each zones process re-renders its own nvdsanalytics config from the zones file on this bump
            version = stream_manager.zones_bump() if stream_manager is not None else None
            resp = dict(payload)
            if version is not None:
                resp["zones_version"] = version
            if stream_manager is not None and module_enabled(running_sidecar_config(config), "proximity"):
                proximity_version = stream_manager.proximity_bump()
                if proximity_version is not None:
                    resp["proximity_version"] = proximity_version
            return JSONResponse(resp)

        # An empty list deletes all of this stream's zones.
        if len(body.zones) == 0:
            path = _zones_file_path(config)
            zones = _load_zones(path)
            zones.pop(stream_id, None)
            _save_zones(path, zones)
            return _respond({"zones": []})

        allowed = set(_detection_classes(config))
        mux_w, mux_h = mux_resolution(config)
        stored_zones, response_zones = [], []
        sidecar_cfg = running_sidecar_config(config)              # once per save: it merges every role's file
        prox_defaults = proximity_defaults(sidecar_cfg)
        proximity_running = module_enabled(sidecar_cfg, "proximity")
        ppe_choices = ppe_items(sidecar_cfg)
        for index, z in enumerate(body.zones):
            err = _validate_zone(z, allowed, prox_defaults, proximity_running, ppe_choices)
            if err is not None:
                return JSONResponse({"error": err}, status_code=400)
            # ids/labels are canonical per-stream position — the sidecar keys ROIs on them.
            zid = zone_label(index)
            name = z.name.strip() or f"Zone {index + 1}"
            intrusion = {"mode": z.intrusion.mode, "classes": list(z.intrusion.classes)}
            overcrowding = {"enabled": z.overcrowding.enabled,
                            "threshold": z.overcrowding.threshold,
                            "classes": list(z.overcrowding.classes)}
            stored = {
                "id": zid, "name": name,
                "points": [[round(p.x * mux_w), round(p.y * mux_h)] for p in z.points],
                "intrusion": intrusion, "overcrowding": overcrowding,
            }
            response = {
                "id": zid, "name": name,
                "points": [{"x": p.x, "y": p.y} for p in z.points],
                "intrusion": intrusion, "overcrowding": overcrowding,
            }
            if z.proximity is not None:
                block = stored_proximity(z.proximity.model_dump(exclude_none=True), prox_defaults)
                stored["proximity"] = block
                response["proximity"] = effective_proximity(block, prox_defaults)
            for name in ZONE_FEATURES:
                block = getattr(z, name)
                if block is not None:
                    stored[name] = block.model_dump(exclude_none=True)      # PPE's carries its `items` too
                    response[name] = dict(stored[name])
            stored_zones.append(stored)
            response_zones.append(response)

        path = _zones_file_path(config)
        zones = _load_zones(path)
        zones[stream_id] = {"zones": stored_zones}
        _save_zones(path, zones)
        return _respond({"zones": response_zones})

    # ── Filter zones ────────────────────────────────────────────────────────
    # Per-stream polygons that define the allowed detection region.  Detections
    # whose centre falls outside every polygon are dropped before tracking.

    @app.get("/api/streams/{stream_id}/filter-zones")
    async def get_filter_zones(stream_id: str):
        from app.zones import stream_filter_zones
        polygons = stream_filter_zones(config, stream_id)
        return JSONResponse({"zones": [
            {"points": [{"x": x, "y": y} for x, y in poly]}
            for poly in polygons
        ]})

    class FilterZonePoint(BaseModel):
        x: float
        y: float

    class FilterZoneEntry(BaseModel):
        points: list[FilterZonePoint]

    class FilterZonesRequest(BaseModel):
        zones: list[FilterZoneEntry]

    @app.post("/api/streams/{stream_id}/filter-zones")
    async def save_filter_zones(stream_id: str, body: FilterZonesRequest):
        from app.zones import save_stream_filter_zones, stream_filter_zones
        polygons = [[(p.x, p.y) for p in z.points] for z in body.zones]
        save_stream_filter_zones(config, stream_id, polygons)
        # Bump zones version so the sidecar reloads filter zones
        version = stream_manager.zones_bump() if stream_manager is not None else None
        resp = {"zones": [{"points": [{"x": p.x, "y": p.y} for p in z.points]}
                           for z in body.zones]}
        if version is not None:
            resp["zones_version"] = version
        return JSONResponse(resp)

    # ── Restricted zones: a truck staying in one past its threshold raises an alert ─────────────
    def _restricted_path() -> str:
        return (config.get("restricted_zones") or {}).get("file_path", "data/restricted_zones.json")

    class RestrictedZoneBody(BaseModel):
        id: str | None = None
        name: str
        threshold_s: float
        points: list[FilterZonePoint]

    class RestrictedZonesRequest(BaseModel):
        zones: list[RestrictedZoneBody]

    @app.get("/api/streams/{stream_id}/restricted-zones")
    async def get_restricted_zones(stream_id: str):
        from app.restricted_zones import load_zones
        return JSONResponse({"zones": [
            {"id": z["id"], "name": z["name"], "threshold_s": z["threshold_s"],
             "points": [{"x": x, "y": y} for x, y in z["points"]]}
            for z in load_zones(_restricted_path()).get(stream_id, [])]})

    @app.post("/api/streams/{stream_id}/restricted-zones")
    async def save_restricted_zones(stream_id: str, body: RestrictedZonesRequest):
        import uuid as _uuid
        from app.restricted_zones import load_zones, save_zones
        old = {z["id"] for z in load_zones(_restricted_path()).get(stream_id, [])}
        zones = []
        for z in body.zones:
            zid = z.id if z.id and z.id in old else f"{stream_id}-{_uuid.uuid4().hex[:8]}"
            zones.append({"id": zid, "name": z.name.strip() or "Restricted area",
                          "threshold_s": max(1.0, float(z.threshold_s)),
                          "points": [(p.x, p.y) for p in z.points]})
        save_zones(_restricted_path(), stream_id, zones)
        return JSONResponse({"zones": [
            {"id": z["id"], "name": z["name"], "threshold_s": z["threshold_s"],
             "points": [{"x": x, "y": y} for x, y in z["points"]]} for z in zones
            if len(z["points"]) >= 3]})

    @app.get("/api/streams/{stream_id}/memory")
    async def stream_memory(stream_id: str):
        """The stream's temporal memory, as durations rather than clock stamps.

        Always 200 with valid JSON, never a 404 and never an exception: the SPA
        fallback answers an unknown path with index.html and a 200, which the
        frontend client reads as a parse failure rather than as "memory is off".
        So every no-memory case -- the local backend (no StreamManager), memory
        disabled, an unknown stream id -- is an `enabled: false` body.

        The two halves are independent: `scene_delta: false` is a supported
        configuration, and registers an observer but no SceneMemory.
        """
        payload = {"enabled": False, "stream": stream_id, "situations": [], "areas": [],
                   "entities": [], "scene_notes": None, "tracked": []}
        if stream_manager is None:
            return JSONResponse(payload)
        observer = getattr(stream_manager, "_track_observers", {}).get(stream_id)
        memory = getattr(stream_manager, "_scene_memories", {}).get(stream_id)
        if observer is None and memory is None:
            return JSONResponse(payload)
        now = time.monotonic()
        payload["enabled"] = True
        if memory is not None:
            payload.update(memory.snapshot(now))
        if observer is not None:
            payload.update(observer.snapshot(now))
        return JSONResponse(payload)

    @app.get("/stream/{source_id}")
    async def stream_for(source_id: str):
        if stream_manager is not None:
            src = stream_manager.get_source(source_id)
            if src is None:
                return JSONResponse({"error": "unknown stream"}, status_code=404)
            _mark_watched(src)
            return StreamingResponse(
                src.generate_mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )
        _mark_watched(camera)
        return StreamingResponse(
            camera.generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/{source_id}/frame")
    def stream_frame(source_id: str):
        """Latest frame snapshot for one stream (single JPEG, no held connection).

        The dashboard wall polls this endpoint instead of holding one permanent
        MJPEG connection per tile: browsers cap ~6 concurrent HTTP/1.1 connections
        per origin, so a 21-tile wall of long-lived multipart streams renders most
        tiles black. Snapshot requests are short-lived and multiplex over the same
        connection pool, so they scale to any tile count. Deliberately a sync def:
        FastAPI runs it in a worker thread, keeping the blocking Redis read off
        the event loop.
        """
        if stream_manager is not None:
            src = stream_manager.get_source(source_id)
            if src is None:
                return JSONResponse({"error": "unknown stream"}, status_code=404)
        else:
            src = camera
        _mark_watched(src)
        frame = src.grab_display_frame()
        if not frame:
            # Retry-After guides the frontend backoff during sidecar cold-start.
            # The sidecar typically needs 5-10s after reporting running=true to
            # publish its first iris:jpeg key.  Short delay keeps steady-state
            # responsive if a frame is momentarily delayed.
            return JSONResponse({"error": "no frame yet"}, status_code=503,
                                headers={"Retry-After": "3"})
        return Response(content=frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/camera/switch/{index}")
    async def switch_camera(index: int):
        await camera.stop()
        camera.set_source(index)
        await camera.start()
        return JSONResponse({"status": "started", "source": camera.source_label, "is_file": camera.is_file})

    # ── Video upload ───────────────────────────────────────────────────────
    @app.post("/api/upload-video")
    async def upload_video(file: UploadFile):
        if not file.filename:
            return JSONResponse({"error": "No file provided"}, status_code=400)

        ext = os.path.splitext(file.filename)[1].lower()
        allowed = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        if ext not in allowed:
            return JSONResponse(
                {"error": f"Unsupported format. Allowed: {', '.join(allowed)}"},
                status_code=400,
            )
    
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        if perception_backend == "deepstream":
            # Sidecar owns capture: tell it to switch to the uploaded file using the path
            # as the sidecar sees it on its mount. switch_source builds + plays it.
            camera.set_source(camera.upload_uri(file.filename))
            return JSONResponse({"status": "streaming", "source": camera.source_label, "is_file": camera.is_file})

        # Local backend: stop current camera, switch source to uploaded video, restart
        await camera.stop()
        camera.set_source(save_path)
        await camera.start()

        return JSONResponse({"status": "streaming", "source": camera.source_label, "is_file": camera.is_file})

    # ── Chat (conversational, with event log + session memory) ────────────
    @app.post("/api/chat")
    async def chat_endpoint(body: ChatMessage):
        if not vlm_enabled:
            return JSONResponse(
                {"error": "VLM is disabled (vlm_enabled=false in config.yaml). Chat is not available."},
                status_code=503,
            )
        per_camera_frames, per_camera_offsets, included_ids = chat_context.gather_chat_context()
        if not per_camera_frames:
            return JSONResponse(
                {"error": "No camera frame available. Start the camera first."},
                status_code=503,
            )

        _t_turn = time.monotonic()
        _frame_counts = ",".join(f"{cid}:{len(f)}" for cid, f in per_camera_frames.items())
        _chat_log.info('[chat] turn msg="%s" frames=%s total=%d',
                       body.message[:60], _frame_counts,
                       sum(len(f) for f in per_camera_frames.values()))

        session = await chat_sessions.get_or_create(body.session_id)
        prior_turns = list(session.turns)  # snapshot before appending the new turn
        # The recent-event log is intentionally NOT injected into the chat prompt:
        # the operator may have been chatting for a while with proactive analysis
        # paused, so a "recent" log would be stale and misleading. Historical
        # questions go through the query_events tool; aggregate analytics
        # (cumulative, time-stamped) stay in the prompt.
        history = None
        analytics = await store.analytics_summary()

        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        async def generate_inner():
            reply_parts: list[str] = []
            debug_calls: list = []  # one entry per model call; dumped after the turn
            clip_frames: dict | None = None   # frames of the FOUND request -> preview clip
            clip_offsets: dict | None = None
            try:
                if body.mode == "search":
                    # ── First pass: sweep-prompted, tools on, streamed through the gate ──
                    gate = SweepMarkerGate(min_confidence=found_min_confidence)
                    _raw_first_pass: list[str] = []  # DEBUG: capture post-think text the gate resolves
                    _fp_parts: list[str] = []  # buffer the answer so timestamps can be validated before emit
                    async for item in analyzer.chat_multi(
                        per_camera_frames, body.message, store=store, history=history,
                        analytics=analytics, prior_turns=prior_turns, max_images=chat_max_images,
                        use_tools=True, sweep=True, per_camera_offsets=per_camera_offsets,
                        debug_record=debug_calls,
                    ):
                        if item["type"] == "status":
                            yield _sse("status", {"stage": item["stage"]})
                        elif item["type"] == "token":
                            _raw_first_pass.append(item["delta"])  # DEBUG
                            emit = gate.feed(item["delta"])
                            if emit:
                                _fp_parts.append(emit)
                    tail = gate.flush()
                    if tail:
                        _fp_parts.append(tail)

                    # FOUND / CONVERSATION / PLAIN are answered here. Clamp any
                    # out-of-range timestamp the model emitted to the footage window
                    # we actually showed, then emit the validated answer.
                    if gate.decision != "NOT_FOUND":
                        first_answer = clamp_answer_timestamps(
                            "".join(_fp_parts), offsets_bounds(per_camera_offsets))
                        if first_answer:
                            reply_parts.append(first_answer)
                            yield _sse("token", {"delta": first_answer})
                        if gate.decision == "FOUND":
                            clip_frames, clip_offsets = per_camera_frames, per_camera_offsets

                    _chat_log.info("[chat] first-pass %.2fs -> %s%s",
                                   time.monotonic() - _t_turn, gate.decision or "PLAIN",
                                   " (sweeping)" if gate.decision == "NOT_FOUND" else "")
                    _chat_log.warning("[chat][DEBUG] first-pass raw first line=%r full=%r",
                                      "".join(_raw_first_pass).split("\n", 1)[0][:120],
                                      "".join(_raw_first_pass)[:300])

                    # ── Sweep ONLY when the model explicitly reports the queried object/action
                    #    is not in the current frames. FOUND / CONVERSATION / PLAIN are answered in place. ──
                    if gate.decision == "NOT_FOUND" and search_enabled and hasattr(chat_context, "sweep_history_snapshots"):
                        _active_sweeps["n"] += 1
                        try:
                            yield _sse("status", {"stage": "searching_history"})
                            prior_sweep = await chat_sessions.get_sweep(session.session_id)
                            snapshots = chat_context.sweep_history_snapshots()

                            _match = {}  # frames/offsets of the round that returned FOUND

                            async def probe(frames_by_cam, offsets_by_cam=None):
                                parts: list[str] = []
                                async for it in analyzer.chat_multi(
                                    frames_by_cam, body.message, store=store, history=history,
                                    analytics=analytics, prior_turns=prior_turns,
                                    max_images=chat_max_images, use_tools=False, sweep=True,
                                    per_camera_offsets=offsets_by_cam, debug_record=debug_calls,
                                ):
                                    if it["type"] == "token":
                                        parts.append(it["delta"])
                                found, answer = parse_sweep_marker(
                                    "".join(parts), min_confidence=found_min_confidence)
                                if found:
                                    answer = clamp_answer_timestamps(
                                        answer, offsets_bounds(offsets_by_cam))
                                    _match["frames"] = frames_by_cam
                                    _match["offsets"] = offsets_by_cam
                                return found, answer

                            result = None
                            async for it in run_sweep(
                                snapshots=snapshots, live_window=chat_context.sweep_live_window,
                                probe=probe, fps=chat_context._fps,
                                round_seconds=chat_context._round_seconds,
                                window_frames=chat_context._window_frames,
                                max_calls=search_max_calls, max_seconds=search_max_seconds,
                                start_index=(prior_sweep or {}).get("history_index"),
                                # Seed the live cursor at "now" (newest buffered offset) so the
                                # first live round doesn't dump the whole history buffer; a resume
                                # carries its own saved cursor.
                                start_since=(prior_sweep or {}).get(
                                    "since", snapshots_newest_offset(snapshots)),
                            ):
                                if it["type"] == "status":
                                    yield _sse("status", {"stage": it["stage"], "detail": it.get("detail", "")})
                                elif it["type"] == "token":
                                    reply_parts.append(it["delta"])
                                    yield _sse("token", {"delta": it["delta"]})
                                elif it["type"] == "result":
                                    result = it
                            if result and result["found"] and _match.get("frames"):
                                clip_frames, clip_offsets = _match["frames"], _match["offsets"]
                            if result and not result["found"] and not result["exhausted"]:
                                await chat_sessions.set_sweep(session.session_id,
                                                              result["history_index"], result["since"])
                            else:
                                await chat_sessions.clear_sweep(session.session_id)
                        finally:
                            _active_sweeps["n"] -= 1
                    else:
                        await chat_sessions.clear_sweep(session.session_id)
                else:
                    # ── Normal mode: answer from current frames only. No marker prompt,
                    #    no gate, no history sweep. Tools stay on for event-DB questions. ──
                    async for item in analyzer.chat_multi(
                        per_camera_frames, body.message, store=store, history=history,
                        analytics=analytics, prior_turns=prior_turns, max_images=chat_max_images,
                        use_tools=True, sweep=False, per_camera_offsets=per_camera_offsets,
                        debug_record=debug_calls,
                    ):
                        if item["type"] == "status":
                            yield _sse("status", {"stage": item["stage"]})
                        elif item["type"] == "token":
                            reply_parts.append(item["delta"])
                            yield _sse("token", {"delta": item["delta"]})
                    await chat_sessions.clear_sweep(session.session_id)
                    _chat_log.info("[chat] first-pass %.2fs -> NORMAL",
                                   time.monotonic() - _t_turn)
            except Exception as exc:
                logger.error("Chat failed: %s", exc)
                yield _sse("error", {"message": str(exc)})
                return

            reply = "".join(reply_parts).strip()
            analyzer.save_chat_debug(body.mode, body.message, debug_calls, reply,
                                     session_id=session.session_id)
            await chat_sessions.append_turn(session.session_id, "user", body.message)
            await chat_sessions.append_turn(session.session_id, "assistant", reply)

            # Build a preview clip from the FOUND request's footage window (search
            # mode). Scope it to the stream the answer names (exact stream-name
            # match); if none is named, fall back to all matched cameras.
            #
            # Recorded segments only exist on the deepstream backend and only for
            # streams the sidecar records continuously. Where none cover the window
            # — the local backend (no recorder), a gap in the index, or an RTSP
            # source under smart-record — the camera falls back to a clip built from
            # the frames the model was shown. Per camera, not per turn: a mixed
            # fleet gives real footage for one stream and frames for another in the
            # same payload, rather than collapsing to whichever is worse off.
            clip_payload = None
            try:
                if preview_enabled and clip_frames:
                    wanted = select_result_streams(reply, list(clip_frames)) or list(clip_frames)
                    cam_offsets = {c: (clip_offsets or {}).get(c) for c in wanted}
                    rng = offsets_bounds(cam_offsets)
                    if rng is not None:
                        lo, hi = rng
                        loop = asyncio.get_running_loop()
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            cam_paths = {}
                            cam_sources = {}  # camera -> "footage" | "frames"
                            cam_ranges = []  # (actual_start, actual_end) per camera that produced a clip
                            for cid in wanted:
                                result = None
                                if segment_store is not None:
                                    _t_fetch = time.monotonic()
                                    refs = await loop.run_in_executor(
                                        None, segments_for_found, segment_store, cid, lo, hi,
                                        _clip_padding_seconds)
                                    if refs:
                                        _chat_log.info(
                                            "[chat] clip lookup cam=%s window=[%.2f,%.2f] "
                                            "found=%d segs=[%.2f,%.2f]", cid, lo, hi, len(refs),
                                            refs[0].start_pts, refs[-1].end_pts)
                                        if _clip_debug_logging:
                                            _chat_log.info("[chat] fetched %d segment(s) for %s in %.2fs",
                                                           len(refs), cid, time.monotonic() - _t_fetch)
                                    else:
                                        _chat_log.warning(
                                            "[chat] clip lookup cam=%s window=[%.2f,%.2f] "
                                            "found=0 segments (no overlap in the segment index)",
                                            cid, lo, hi)
                                    _t_build = time.monotonic()
                                    result = await loop.run_in_executor(
                                        None, build_preview_clip, segment_store, cid, lo, hi,
                                        _clip_padding_seconds, _clip_trim_enabled, _clip_trim_seconds, tmp_dir)
                                if result:
                                    path, actual_start, actual_end = result
                                    if _clip_debug_logging:
                                        size_kb = os.path.getsize(path) / 1024
                                        _chat_log.info(
                                            "[chat] clip built %s -> %.1fKB in %.2fs range=[%.2f,%.2f]",
                                            cid, size_kb, time.monotonic() - _t_build,
                                            actual_start, actual_end)
                                    cam_paths[cid] = path
                                    cam_sources[cid] = "footage"
                                    cam_ranges.append((actual_start, actual_end))
                                elif clip_frames.get(cid):
                                    # No recorded footage covers this window. Fall back to
                                    # the frames the model was actually shown -- low frame
                                    # rate, but it is the evidence behind the answer.
                                    fpath = await loop.run_in_executor(
                                        None, build_frames_clip, clip_frames[cid],
                                        preview_fps, tmp_dir, cid)
                                    if fpath:
                                        frames_rng = offsets_bounds(
                                            {cid: (clip_offsets or {}).get(cid)})
                                        cam_paths[cid] = fpath
                                        cam_sources[cid] = "frames"
                                        if frames_rng is not None:
                                            cam_ranges.append(frames_rng)
                                        _chat_log.info(
                                            "[chat] frames preview cam=%s frames=%d fps=%g "
                                            "(no segments covered the window)",
                                            cid, len(clip_frames[cid]), preview_fps)
                            if cam_paths:
                                rng_out = [min(s for s, _ in cam_ranges), max(e for _, e in cam_ranges)]
                                clip_id = clips.add(cam_paths, {"cameras": list(cam_paths),
                                                                "range": rng_out,
                                                                "sources": cam_sources,
                                                                "message": body.message})
                                clip_payload = {"id": clip_id, "cameras": list(cam_paths),
                                                "range": rng_out, "sources": cam_sources}
                                _chat_log.info("[chat] preview clip %s cams=%s sources=%s",
                                               clip_id, list(cam_paths), cam_sources)
            except Exception as exc:
                _chat_log.warning("[chat] preview clip build failed: %s", exc)
                clip_payload = None

            _chat_log.info("[chat] turn done %.2fs cameras=%s",
                           time.monotonic() - _t_turn, included_ids)
            yield _sse("done", {"session_id": session.session_id, "cameras": included_ids,
                                "clip": clip_payload})

        async def generate():
            # Auto-pause proactive analysis for the duration of the chat turn.
            monitoring["chat_in_flight"] = monitoring.get("chat_in_flight", 0) + 1
            try:
                async for chunk in generate_inner():
                    yield chunk
            finally:
                monitoring["chat_in_flight"] = max(0, monitoring.get("chat_in_flight", 1) - 1)

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── FOUND-result preview clips ──────────────────────────────────────────
    @app.get("/api/chat/clip/{clip_id}")
    async def get_clip_meta(clip_id: str):
        meta = clips.get(clip_id)
        if meta is None:
            return JSONResponse({"error": "Clip not found or expired"}, status_code=404)
        return JSONResponse(meta)

    @app.get("/api/chat/clip/{clip_id}/{camera}")
    async def get_clip_stream(clip_id: str, camera: str):
        path = clips.get_clip_path(clip_id, camera)
        if path is None:
            return JSONResponse({"error": "Clip not found or expired"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")

    def _clip_response(store_, clip_id: str, camera: str | None):
        if camera is None:
            meta = store_.get(clip_id)
            if meta is None:
                return JSONResponse({"error": "Clip not found or expired"}, status_code=404)
            return JSONResponse(meta)
        path = store_.get_clip_path(clip_id, camera)
        if path is None:
            return JSONResponse({"error": "Clip not found or expired"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/alerts/clip/{clip_id}")
    async def get_alert_clip_meta(clip_id: str):
        return _clip_response(alert_clips, clip_id, None)

    @app.get("/api/alerts/clip/{clip_id}/{camera}")
    async def get_alert_clip_stream(clip_id: str, camera: str):
        return _clip_response(alert_clips, clip_id, camera)

    @app.get("/api/alerts/snapshot/{clip_id}/{camera}/{event_id}")
    async def get_alert_snapshot(clip_id: str, camera: str, event_id: int):
        path = alert_clips.get_snapshot_path(clip_id, camera, event_id)
        if path is None:
            return JSONResponse({"error": "Snapshot not found or expired"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/alerts/depth/{event_id}")
    async def get_alert_depth_image(event_id: int):
        path = depth_images.get(event_id)
        if path is None:
            return JSONResponse({"error": "Depth image not found or expired"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    # ── System prompt ──────────────────────────────────────────────────────
    @app.get("/api/prompt")
    async def get_prompt():
        instructions = config.get("monitoring", {}).get("instructions", "")
        return JSONResponse({"instructions": instructions})

    @app.post("/api/prompt")
    async def update_prompt(body: PromptUpdate):
        monitoring = config.setdefault("monitoring", {})
        monitoring["instructions"] = body.instructions
        active = monitoring.get("active_persona", DEFAULT_PERSONA)
        monitoring.setdefault("persona_instructions", {})[active] = body.instructions
        return JSONResponse({"status": "updated"})

    # ── Personas ───────────────────────────────────────────────────────────
    @app.get("/api/personas")
    async def get_personas():
        active = config.get("monitoring", {}).get("active_persona", DEFAULT_PERSONA)
        return JSONResponse({
            "personas": [{"id": k, "name": v["name"]} for k, v in PERSONAS.items()],
            "active": active,
        })

    @app.post("/api/persona/{persona_id}")
    async def set_persona(persona_id: str):
        if persona_id not in PERSONAS:
            return JSONResponse({"error": "Unknown persona"}, status_code=400)
        monitoring = config.setdefault("monitoring", {})
        monitoring["active_persona"] = persona_id
        instructions = monitoring.setdefault("persona_instructions", {}).get(
            persona_id, PERSONAS[persona_id]["instructions"]
        )
        monitoring["instructions"] = instructions
        return JSONResponse({
            "status": "ok",
            "id": persona_id,
            "name": PERSONAS[persona_id]["name"],
            "instructions": instructions,
        })

    # ── Language ───────────────────────────────────────────────────────────
    @app.get("/api/language")
    async def get_language():
        lang = config.get("monitoring", {}).get("output_language", "english")
        return JSONResponse({"language": lang})

    @app.post("/api/language")
    async def set_language(body: LanguageUpdate):
        allowed = {"english", "arabic"}
        if body.language not in allowed:
            return JSONResponse({"error": "unsupported language"}, status_code=400)
        config.setdefault("monitoring", {})["output_language"] = body.language
        return JSONResponse({"status": "updated", "language": body.language})

    # ── Working mode (proactive | chat) ─────────────────────────────────────
    @app.get("/api/mode")
    async def get_mode():
        return JSONResponse({"mode": monitoring.get("active_mode", "proactive")})

    @app.post("/api/mode")
    async def set_mode(body: ModeUpdate):
        if body.mode not in VALID_MODES:
            return JSONResponse({"error": "unknown mode"}, status_code=400)
        monitoring["active_mode"] = body.mode
        await manager.broadcast({"type": "mode", "mode": body.mode})
        return JSONResponse({"status": "updated", "mode": body.mode})

    # ── Events ─────────────────────────────────────────────────────────────
    @app.get("/api/events")
    async def get_events(limit: int = 100, camera: str | None = None):
        events = await store.get_recent(limit=limit, camera=camera)
        return JSONResponse(events)

    @app.post("/api/events/{event_id}/acknowledge")
    async def acknowledge_event(event_id: int):
        """The one write action the fire/smoke operator UI offers: mark an
        incident acknowledged. Idempotent -- acknowledging twice keeps the
        original timestamp (see EventStore.acknowledge)."""
        record = await store.acknowledge(event_id)
        if record is None:
            return JSONResponse({"error": "event not found"}, status_code=404)
        await manager.broadcast({"type": "event_update", "id": event_id,
                                 "acknowledged_at": record["acknowledged_at"]})
        return JSONResponse(record)

    @app.delete("/api/events")
    async def clear_events():
        deleted = await store.clear_all()
        enricher.clear_log()
        if analyzer is not None:
            analyzer.clear_prompt_log()
            analyzer.clear_response_log()
            analyzer.clear_call_debug()
            analyzer.clear_chat_debug()
            analyzer.clear_description_debug()
        alert_clips.clear()
        await manager.broadcast({"type": "cleared"})
        return JSONResponse({"status": "cleared", "deleted": deleted})

    @app.delete("/api/events/tag/{tag}")
    async def clear_events_by_tag(tag: str):
        """Scoped counterpart to DELETE /api/events above (2026-09-06):
        that one wipes every alert kind plus chat/debug logs and alert
        clips, which is far more than the attendance dashboard's Fire/Smoke
        page "Clear" button should be allowed to touch -- it only shows
        fire_smoke-tagged incidents, so it only clears those, leaving other
        alert kinds (zone intrusion, overcrowding, ...) and every clip/log
        untouched."""
        deleted = await store.clear_by_tag(tag)
        await manager.broadcast({"type": "cleared", "tag": tag})
        return JSONResponse({"status": "cleared", "deleted": deleted, "tag": tag})

    @app.delete("/api/chat/session/{session_id}")
    async def delete_chat_session(session_id: str):
        deleted = await chat_sessions.delete(session_id)
        return JSONResponse({"status": "deleted" if deleted else "not_found"})

    @app.get("/api/config")
    async def get_config():
        safe = {k: v for k, v in config.items() if k != "api"}
        api_cfg = config.get("api", {})
        backend = api_cfg.get("backend", "cloud")
        safe["api"] = {
            "backend": backend,
            "model": api_cfg.get("local_model") if backend == "local" else api_cfg.get("model"),
            "local_visual_format": api_cfg.get("local_visual_format") if backend == "local" else None,
        }
        return JSONResponse(safe)

    # ── Debug ──────────────────────────────────────────────────────────────
    @app.get("/api/debug/frame")
    async def debug_frame():
        path = config.get("cv", {}).get("debug_frame_path", "data/debug_frame.jpg")
        if not os.path.isfile(path):
            return JSONResponse({"error": "No debug frame yet — CV enrichment may be disabled or no cycle has run."}, status_code=404)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # Counting and tracking routes exist only when the tracking service does (mtmc.enabled).
    if tracking_service is not None:
        # ── Entrance / exit counting ──────────────────────────────────────────
        class CountLineBody(BaseModel):
            a: list[float] = Field(min_length=2, max_length=2)
            b: list[float] = Field(min_length=2, max_length=2)
            enter_positive: bool

        @app.get("/api/count/{cam_id}")
        async def count_get(cam_id: str):
            lc = tracking_service.line_counter
            return JSONResponse({"line": lc.get_line(cam_id), **lc.totals(cam_id)})

        @app.put("/api/count/{cam_id}/line")
        async def count_set_line(cam_id: str, body: CountLineBody):
            try:
                line = tracking_service.line_counter.set_line(cam_id, body.model_dump())
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            return JSONResponse({"line": line})

        @app.delete("/api/count/{cam_id}/line")
        async def count_delete_line(cam_id: str):
            tracking_service.line_counter.delete_line(cam_id)
            return JSONResponse({"line": None})

        @app.post("/api/count/{cam_id}/reset")
        async def count_reset(cam_id: str):
            tracking_service.line_counter.reset(cam_id)
            return JSONResponse({"ok": True})

        # ── Multi-camera vehicle tracking ─────────────────────────────────────
        @app.get("/api/tracking/tracks")
        async def tracking_tracks(limit: int = 100):
            return JSONResponse(tracking_service.get_tracks(limit=limit))

        @app.get("/api/tracking/transitions")
        async def tracking_transitions(limit: int = 100):
            return JSONResponse(tracking_service.get_transitions(limit=limit))

        @app.get("/api/vehicles/{gid}/sightings")
        async def vehicle_sightings(gid: int):
            return JSONResponse(tracking_service.get_sightings(gid))

        @app.get("/metrics")
        async def prometheus_metrics():
            from app.prom_text import render
            return PlainTextResponse(render(tracking_service.health_samples()),
                                     media_type="text/plain; version=0.0.4")

        @app.get("/api/tracking/stats")
        async def tracking_stats():
            return JSONResponse(tracking_service.get_stats())

        @app.get("/api/tracking/cameras")
        async def tracking_cameras():
            return JSONResponse(tracking_service.get_camera_summary())

        @app.get("/api/tracking/config")
        async def tracking_config():
            return JSONResponse(tracking_service.get_config())

    # ── Map layout ────────────────────────────────────────────────────────
    MAP_FILE = config.get("map", {}).get("file_path", "data/map_layout.json")

    @app.get("/api/map")
    async def get_map():
        if not os.path.isfile(MAP_FILE):
            return JSONResponse({"cameras": [], "shapes": []})
        with open(MAP_FILE) as f:
            return JSONResponse(json.load(f))



    @app.post("/api/map")
    async def save_map_post(request: Request):
        body = await request.json()
        os.makedirs(os.path.dirname(MAP_FILE) or ".", exist_ok=True)
        tmp = MAP_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(body, f, indent=2)
        os.replace(tmp, MAP_FILE)
        return JSONResponse({"status": "saved"})

    # ── WebSocket ──────────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await manager.connect(ws)
        try:
            history = await store.get_recent(limit=50)
            await ws.send_text(json.dumps({
                "type": "history",
                "events": history,
                "mode": monitoring.get("active_mode", "proactive"),
            }))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(ws)
        except Exception as exc:
            logger.debug("WebSocket closed: %s", exc)
            manager.disconnect(ws)

    # ── Static frontend (must be last so API routes match first) ──────────
    static_dir = "static"
    if os.path.isdir(static_dir):
        index_file = os.path.join(static_dir, "index.html")
        assets_dir = os.path.join(static_dir, "assets")
        if os.path.isdir(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            candidate = os.path.join(static_dir, full_path)
            if full_path and os.path.isfile(candidate):
                return FileResponse(candidate)
            return FileResponse(index_file)
    else:
        logger.warning("static/ directory not found — frontend will not be served")

    return app
