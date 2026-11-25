import asyncio
import json
import logging
import time

import redis

from perception.engine.sources import split_stream_key_id

from .features import camera_of, feature_of
from .mode import analysis_enabled
from .overlay_track import CLIP_STYLES, PREVIEW_KEY
from .perception_source import PerceptionSource
from .redis_streams import STATUS_PREFIX
from .sidecar_config import module_enabled, module_setting, running_sidecar_config
from .usecases.alert_events import AlertEvents
from .usecases.chat import ChatHistory
from .usecases.count_alerts import CountAlerts, fire_sampler_needed
from .usecases.descriptions import Descriptions
from .usecases.fire_incidents import FireIncidents
from .usecases.memory import MeasuredMemory
from .usecases.recordings import Recordings
from .usecases.vlm_analysis import VlmAnalysis, build_frame_metas  # noqa: F401  (re-exported)
from .usecases.module_alerts import ModuleAlerts
from .usecases.zone_alerts import ZoneAlerts
from .usecases.zone_incidents import ZoneIncidents

logger = logging.getLogger("iris.stream")


def pick_source(pending_ids, last_serviced: dict):
    """Return the pending source_id serviced longest ago (fair, anti-starvation), or None."""
    if not pending_ids:
        return None
    return min(pending_ids, key=lambda i: last_serviced.get(i, 0.0))


_PIPELINE_KEY = "iris:ctrl:pipeline"
_ZONES_VERSION_KEY = "iris:ctrl:zones_version"
_ZONES_APPLIED_KEY = "iris:analytics:zones_applied"
_PROXIMITY_VERSION_KEY = "iris:ctrl:proximity_version"         # the proximity module's VERSION_KEY / APPLIED_KEY
_PROXIMITY_APPLIED_KEY = "iris:analytics:proximity_applied"    # (not imported: the module package loads the pipeline)
_PROXIMITY_PREVIEW_KEY = PREVIEW_KEY       # the module's PREVIEW_KEY: the live preview's layers, and the clips' style


class StreamManager:
    """Runs the use cases for every camera the pipeline reports. It discovers
    cameras from Redis, starts and stops each use case's per-camera task
    (app/usecases/), and runs the one VLM worker that VLM analysis and alert
    descriptions share. The use cases hold the logic; this only wires them up.
    Without a VLM, everything except VLM analysis and descriptions still runs."""

    def __init__(self, config, analyzer, store, conn_manager, redis_client=None, metrics=None,
                 clip_service=None, depth_images=None):
        self._config = config
        self._vlm_enabled = analyzer is not None
        self._conn = conn_manager
        self._worker_current: str | None = None
        perc = config.get("perception", {})
        self._redis = redis_client or redis.Redis.from_url(perc.get("redis_url", "redis://localhost:6379/0"))
        self._discovery_interval = float(perc.get("discovery_interval_seconds", 5))

        # Measured temporal memory (monitoring.memory): off unless enabled, then a no-op everywhere.
        self.memory = MeasuredMemory(config)
        self.chat = ChatHistory(config, running=lambda: self._running,
                                on_tick=self.memory.observe_tick)

        api_cfg = config.get("api", {})
        self._serve_batch = 1
        if api_cfg.get("backend", "cloud") == "local":
            self._serve_batch = max(int(api_cfg.get("local_serve_batch", 1)), 1)

        sidecar_cfg = running_sidecar_config(config)
        self._analytics_enabled = module_enabled(sidecar_cfg, "zones")
        self._proximity_enabled = module_enabled(sidecar_cfg, "proximity")
        self._proximity_preview_defaults = {      # the module's Settings.preview_* unless the yaml sets them
            "masks": bool(module_setting(sidecar_cfg, "proximity", "preview_masks", False)),
            "lines": bool(module_setting(sidecar_cfg, "proximity", "preview_lines", True)),
            "boxes": bool(module_setting(sidecar_cfg, "proximity", "preview_boxes", True)),
            "depth": bool(module_setting(sidecar_cfg, "proximity", "preview_depth", False)),
            "line_range": float(module_setting(sidecar_cfg, "proximity", "preview_line_range", 2.0)),
            "clip_style": CLIP_STYLES[0]}         # the app's own: how alert clips and snapshots draw objects
        self._fire_alert_enabled = module_enabled(sidecar_cfg, "fire_alerts")
        # Overcrowding is evaluated by the zones module: without it no overcrowding alert can fire.
        self._overcrowding_enabled = (self._analytics_enabled
                                      and bool(module_setting(sidecar_cfg, "zones", "overcrowding_enabled", True)))

        # Per-camera role tags (fire/attendance/...), for the UI to know which
        # cameras to preview where. Untagged here means "not tagged" -- unlike
        # the sidecar's own process-filtering default, this must NOT report
        # "every role" for an untagged camera, or every camera would show up
        # in every role-scoped UI section regardless of whether it was ever
        # meant to.
        # `.get("sources", [])` only supplies its default when the KEY is absent:
        # a `sources:` line with nothing under it parses to None and would make
        # this iteration raise, taking the whole StreamManager (and app) down --
        # so guard it the same way usecases/recordings.py does for its sources.
        self._camera_roles = {}
        self._camera_meta = {}
        self._read_camera_tags(sidecar_cfg)
        # The pipeline's sources as last read: a change (the safety product's Apply rewrites its role configs) is
        # picked up by refresh_config() on the next discovery pass.
        self._source_ids = self._ids_of(sidecar_cfg)
        self.on_sources_changed: list = []      # called with no argument after a change (the server: the ROI file)

        self.fire = FireIncidents(store, conn_manager, clip_service, self._redis)
        self.recordings = Recordings(
            sidecar_cfg, self._redis, clip_service, running=lambda: self._running,
            on_unclaimed_file=self.fire.continue_clip)
        self.descriptions = Descriptions(config, analyzer, store, conn_manager, self.chat,
                                         vlm_enabled=self._vlm_enabled)
        self.alerts = AlertEvents(config, store, conn_manager, clip_service,
                                  self.recordings, self.descriptions)
        # Intrusion and overcrowding as incidents with an end, when the zones module says when they end (its
        # end_markers setting: the safety product); otherwise one alert per entry / crossing, as before.
        self.zone_incidents = (ZoneIncidents(self.alerts)
                               if module_setting(sidecar_cfg, "zones", "end_markers", False) else None)
        # Each alert path checks its stream's boot first: a sidecar's first alert after a restart can come before the
        # discovery pass sees the new boot_id, and the reconcile would then close the new boot's incident too.
        before = self.fire.check_sidecar_boot_id
        self.zone_alerts = ZoneAlerts(config, self._redis, self.alerts, self.descriptions,
                                      running=lambda: self._running, incidents=self.zone_incidents, before=before)
        self.count_alerts = CountAlerts(config, self._redis, self.alerts, self.descriptions,
                                        self.fire, running=lambda: self._running, incidents=self.zone_incidents,
                                        before=before)
        self.module_alerts = ModuleAlerts(self._redis, self.alerts, lambda: list(self._sources),
                                          running=lambda: self._running, depth_images=depth_images, before=before)
        self.fire.on_restart.append(self.module_alerts.sidecar_restarted)
        if self.zone_incidents is not None:
            self.fire.on_restart.append(self.zone_incidents.sidecar_restarted)
        self.analysis = VlmAnalysis(config, analyzer, store, conn_manager, clip_service,
                                    self.recordings, metrics, running=lambda: self._running,
                                    memory=self.memory)


        self._sources: dict = {}
        self._last_serviced: dict = {}
        # Every long-running per-camera task, keyed (use case, camera): the VLM
        # producer, chat sampler, alert samplers, smart-record sampler. One
        # registry so starting, restarting and cancelling is written once.
        self._tasks: dict = {}
        self._running = False
        self._worker_task = None
        self._discovery_task = None
        self._alerts_task = None

        # Chat reads the same memory the analysis prompt does. The chat prompt is
        # assembled inside the Analyzer, which has no view of the streams, so hand it a
        # callable it can pull from at turn time (it only fires on normal turns; search
        # turns drop the block, as they already drop analytics).
        if self.memory.scene_delta and analyzer is not None:
            analyzer.chat_scene_provider = self.chat_scene_memory

    def _read_camera_tags(self, sidecar_cfg: dict) -> None:
        # Optional human-readable name/zone per camera, purely cosmetic (the UI's
        # fire/smoke live wall and history views). Absent for any camera that
        # doesn't set them in sidecar.yaml -- the frontend falls back to the raw
        # camera id, same as it already does for roles.
        try:
            from perception.engine.sources import normalize_roles
            for s in (sidecar_cfg.get("sources") or []):
                if s.get("id"):
                    self._camera_roles[s["id"]] = normalize_roles(s.get("roles"))
                    self._camera_meta[s["id"]] = {
                        "label": s.get("label"), "zone": s.get("zone")}
        except Exception as exc:
            logger.warning(
                "could not read per-camera roles from sidecar.yaml: %s -- every camera "
                "will report an empty roles list to the UI", exc)

    @staticmethod
    def _ids_of(sidecar_cfg: dict) -> tuple:
        return tuple(str(s.get("id")) for s in (sidecar_cfg.get("sources") or []) if s.get("id"))

    def refresh_config(self) -> bool:
        """Re-read the pipeline's settings (cached until a file changes); when its sources changed, refresh what
        depends on them -- which streams get smart-record clips, the cameras' role tags -- and tell the
        on_sources_changed hooks. True when something changed."""
        cfg = running_sidecar_config(self._config)
        ids = self._ids_of(cfg)
        if ids == self._source_ids:
            return False
        self._source_ids = ids
        self._read_camera_tags(cfg)
        self.recordings.refresh(cfg)
        logger.info("pipeline sources changed: %s", list(ids))
        for hook in self.on_sources_changed:
            try:
                hook()
            except Exception:  # noqa: BLE001 -- one hook failing must not stop discovery
                logger.exception("a sources-changed hook failed")
        return True

    # The memory use case owns these; the manager keeps the names its callers and tests use.
    @property
    def _track_observers(self) -> dict:
        return self.memory.observers

    @property
    def _scene_memories(self) -> dict:
        return self.memory.memories

    def _observe_tick(self, sid: str, src, last_meta_id):
        return self.memory.observe_tick(sid, src, last_meta_id)

    def chat_scene_memory(self) -> str:
        return self.memory.chat_block()

    # ── discovery ─────────────────────────────────────────────────────────────
    def discover(self) -> set:
        try:
            keys = self._redis.keys(f"{STATUS_PREFIX}*")
        except redis.RedisError:
            return set(self._sources)
        out = set()
        for k in keys:
            k = k.decode() if isinstance(k, bytes) else k
            out.add(k[len(STATUS_PREFIX):])
        return out

    def _ensure_stream_state(self, sid: str):
        if sid in self._sources:
            return
        self._sources[sid] = PerceptionSource(self._config, redis_client=self._redis, source_id=sid)
        self.analysis.add_camera(sid)
        self._last_serviced.setdefault(sid, 0.0)
        self.chat.add_camera(sid)
        # Seed the watched key so the sidecar's ViewerGate immediately sees this
        # stream as watched and encodes the first JPEG frame. Without this, the
        # first browser request sets the key, but the ViewerGate's 2-second cache
        # misses it -- the sidecar skips encoding -> 503 loop until the cache
        # refreshes. The short TTL means the key expires if no real viewer arrives.
        try:
            self._redis.set(f"iris:watched:{sid}", "1", ex=10)
        except Exception:
            pass
        self.memory.add_camera(sid)

    def _ensure_task(self, kind: str, sid: str, make) -> None:
        """Start `make()` for this camera unless it is already running. Also
        restarts one that has died: _discovery_loop calls this for every live
        camera on every pass, so a crashed sampler comes back by itself."""
        task = self._tasks.get((kind, sid))
        if task is None or task.done():
            self._tasks[(kind, sid)] = asyncio.create_task(make())

    def _add_stream(self, sid: str):
        self._ensure_stream_state(sid)
        if self._vlm_enabled:
            producer = self._tasks.get(("producer", sid))
            if producer is None or producer.done():
                logger.info("stream added: %s", sid)
            self._ensure_task("producer", sid, lambda: self.analysis.run_for(sid, self._sources[sid]))
        self._ensure_task("chat", sid, lambda: self.chat.run_for(sid, self._sources[sid]))
        if self._analytics_enabled:
            self._ensure_task("zone_alerts", sid, lambda: self.zone_alerts.run_for(sid))
        # Overcrowding is enabled per-zone (in the sidecar), so run the sampler whenever
        # analytics is on and let it sit idle for streams without overcrowding — the
        # sidecar only xadds overcrowd alerts for zones that opted in. Fire/smoke shares
        # this same count-based stream but uses no ROI, so the gate must also open on
        # fire_alert_enabled alone (see fire_sampler_needed) — otherwise a deployment
        # with analytics off gets fire alerts XADDed that nothing ever reads.
        if fire_sampler_needed(self._analytics_enabled, self._fire_alert_enabled):
            self._ensure_task("count_alerts", sid, lambda: self.count_alerts.run_for(sid))
        # Not gated on analytics_enabled: the sidecar announces any finished
        # smart-record file, and only smart-record streams ever produce one.
        if sid in self.recordings.svr_sources:
            self._ensure_task("svr_done", sid, lambda: self.recordings.run_for(sid))

    def _remove_stream(self, sid: str):
        for key in [k for k in self._tasks if k[1] == sid]:
            self._tasks.pop(key).cancel()
        self.descriptions.cancel_for(sid)
        self.chat.remove_camera(sid)
        self.memory.remove_camera(sid)
        self._sources.pop(sid, None)
        self.analysis.remove_camera(sid)
        logger.info("stream removed: %s", sid)

    async def _discovery_loop(self):
        # On cold start, poll aggressively (1 s) until at least one stream is
        # discovered, then settle to the normal interval. This cuts the
        # sidecar-startup -> first-frame latency by up to 4 seconds (the gap
        # between the sidecar publishing iris:status:* and the next slow poll).
        fast_interval = 1.0
        interval = fast_interval
        while self._running:
            self.refresh_config()
            live = self.discover()
            # Every live stream, not just newly-appeared ones: _add_stream's checks are
            # idempotent and restart any producer/sampler that has died (each is guarded
            # by `.done()`). Passing only `live - self._sources` meant those guards could
            # never fire for a stream already known, so one crashed sampler stayed dead
            # for the life of the process with nothing to bring it back.
            for sid in live:
                self._add_stream(sid)
                await self.fire.check_sidecar_boot_id(sid)
            for sid in set(self._sources) - live:
                self._remove_stream(sid)
                await self.fire.stream_gone(sid)        # a stopped role's open incidents (safety) are closed
            if live and interval == fast_interval:
                interval = self._discovery_interval
            await asyncio.sleep(interval)

    # ── producer (per stream) ─────────────────────────────────────────────────
    # ── single VLM worker ─────────────────────────────────────────────────────
    async def _worker(self):
        while self._running:
            proactive_enabled = analysis_enabled(self._config.get("monitoring", {}))
            pending = ([sid for sid, q in self.analysis.queues.items() if not q.empty()]
                      if proactive_enabled else [])
            pending += list(self.descriptions.jobs)
            if not pending:
                await asyncio.sleep(0.05)
                continue

            # Anti-starvation: sort by last_serviced, pick up to serve_batch streams.
            chosen = sorted(pending, key=lambda i: self._last_serviced.get(i, 0.0))[:self._serve_batch]

            items = []       # proactive: (sid, frames, dets_list, offsets, wall_ts)
            desc_jobs = []   # (pseudo_id, event_id, sid, frames, prompt_kind, count)
            for pid in chosen:
                if pid.startswith("__desc__:"):
                    job = self.descriptions.jobs.pop(pid, None)
                    if job is None:
                        continue
                    sid, frames, prompt_kind, count = job
                    desc_jobs.append((pid, int(pid.split(":", 1)[1]), sid, frames, prompt_kind, count))
                    self._last_serviced[pid] = time.monotonic()
                else:
                    try:
                        q_item = self.analysis.queues[pid].get_nowait()
                        _sid, _ts, frames, dets_list, offsets = q_item[:5]
                        wall_ts = q_item[5] if len(q_item) > 5 else time.time()
                        items.append((pid, frames, dets_list, offsets, wall_ts))
                        self._last_serviced[pid] = time.monotonic()
                    except asyncio.QueueEmpty:
                        pass

            if not items and not desc_jobs:
                await asyncio.sleep(0.05)
                continue

            label = ",".join([sid for sid, *_ in items] + [pid for pid, *_ in desc_jobs])
            self._worker_current = label
            logger.info("worker -> %s", label)
            try:
                results = await asyncio.gather(
                    *(self.analysis.process(sid, frames, dets_list, offsets, wall_ts=wall_ts)
                      for sid, frames, dets_list, offsets, wall_ts in items),
                    *(self.descriptions.process(event_id, sid, frames, prompt_kind, count)
                      for _pid, event_id, sid, frames, prompt_kind, count in desc_jobs),
                    return_exceptions=True,
                )
                n = len(items)
                for (sid, *_), exc in zip(items, results[:n]):
                    if isinstance(exc, Exception):
                        logger.error("[%s] analysis failed: %s", sid, exc, exc_info=exc)
                        await self._conn.broadcast({"type": "error", "message": str(exc), "camera": sid})
                for (pid, _event_id, sid, _frames, _kind, _count), exc in zip(desc_jobs, results[n:]):
                    if isinstance(exc, Exception):
                        # descriptions.process resolves via its own fallback internally and
                        # should never raise; log defensively if it somehow does.
                        logger.error("[%s] description job %s crashed unexpectedly: %s",
                                     sid, pid, exc, exc_info=exc)
            finally:
                self._worker_current = None
                # A finished description job is not a stream to keep fair -- drop its
                # bookkeeping (descriptions used to do this from inside the worker's state).
                for pid, *_ in desc_jobs:
                    self._last_serviced.pop(pid, None)

    # ── lifecycle + accessors ─────────────────────────────────────────────────
    async def run(self):
        self._running = True
        if self._vlm_enabled:
            self._worker_task = asyncio.create_task(self._worker())
        else:
            logger.info("VLM worker disabled — only detection + zone alerts + recording active")
            self._worker_task = None
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        # One reader for every camera's iris:alert:<id> (alerts any pipeline module raises); see ModuleAlerts.
        self._alerts_task = asyncio.create_task(self.module_alerts.run())

    async def stop(self):
        self._running = False
        # Every registered per-camera task. Listing the dicts by hand here missed
        # the smart-record samplers, which kept running after stop() (2026-10-01).
        tasks = (list(self._tasks.values())
                 + self.descriptions.all_tasks()
                 + [self._discovery_task, self._alerts_task])
        if self._worker_task is not None:
            tasks = list(tasks) + [self._worker_task]
        for t in tasks:
            if t:
                t.cancel()
        try:
            self._redis.close()
        except Exception:
            pass

    def list_streams(self) -> list:
        out = []
        for sid in sorted(self._sources):
            src = self._sources[sid]
            meta = self._camera_meta.get(sid, {})
            out.append({"id": sid, "running": src.is_running, "is_file": src.is_file,
                        "source": src.source_label, "offset": src.current_offset,
                        # Raw-preview support: in pipeline_preview: false the sidecar
                        # publishes no iris:jpeg, so the frontend plays preview_uri
                        # (raw RTSP via WebRTC) and overlays iris:meta instead of
                        # polling annotated frames. preview_mode is "pipeline" by
                        # default (annotated JPEG polling).
                        "preview_uri": src.preview_uri,
                        "preview_mode": src.preview_mode,
                        "roles": self._camera_roles.get(sid, []),
                        "label": meta.get("label"), "zone": meta.get("zone"),
                        # Beside several role sidecars a stream is `<camera>@<role>` (app/features.py).
                        "camera": camera_of(sid), "feature": feature_of(sid, ())})
        return out

    def get_source(self, sid: str):
        return self._sources.get(sid)

    def source_for_camera(self, camera_or_sid: str):
        """The stream itself, or -- given a camera id -- any of its role streams (`<camera>@<role>`); None if none."""
        src = self._sources.get(camera_or_sid)
        if src is None:
            src = next((s for sid, s in list(self._sources.items()) if camera_of(sid) == camera_or_sid), None)
        return src

    @property
    def redis(self):
        """The Redis client every use case here shares (the safety product's API and health read it too)."""
        return self._redis

    def set_notifier(self, notifier) -> None:
        """Hand every alert's opening and escalation to `notifier` (the safety product's Telegram)."""
        self.alerts.notifier = notifier

    def queue_depths(self) -> dict:
        return {sid: q.qsize() for sid, q in self.analysis.queues.items()}

    def worker_state(self) -> str | None:
        return self._worker_current

    # ── pipeline control (write to sidecar via Redis) ─────────────────────────
    def pipeline_start(self) -> None:
        self._redis.set(_PIPELINE_KEY, "playing")

    def pipeline_stop(self) -> None:
        self._redis.set(_PIPELINE_KEY, "paused")

    def pipeline_desired(self) -> str:
        v = self._redis.get(_PIPELINE_KEY)
        if v is None:
            return "unknown"
        return v.decode() if isinstance(v, bytes) else v

    # ── zone config versioning (sidecar hot-reload handshake) ─────────────────
    def zones_bump(self) -> "int | None":
        """Announce a new generation of the nvdsanalytics config file. The sidecar
        polls this counter and live-reloads the ROI config when it changes."""
        try:
            return int(self._redis.incr(_ZONES_VERSION_KEY))
        except redis.RedisError as exc:
            logger.warning("zones version bump failed: %s", exc)
            return None

    def proximity_bump(self) -> "int | None":
        """Announce new proximity settings in the zones file; the proximity module reloads them and acknowledges."""
        try:
            return int(self._redis.incr(_PROXIMITY_VERSION_KEY))
        except redis.RedisError as exc:
            logger.warning("proximity version bump failed: %s", exc)
            return None

    # ── proximity live preview layers (the module reads the key within a second) ──
    def proximity_preview(self) -> dict:
        """What the proximity preview draws: the UI's setting over the pipeline's defaults; `available` when the
        pipeline runs the module."""
        layers = dict(self._proximity_preview_defaults)
        try:
            chosen = json.loads(self._redis.get(_PROXIMITY_PREVIEW_KEY) or "{}")
        except (redis.RedisError, ValueError) as exc:
            logger.warning("proximity preview setting unreadable: %s", exc)
            chosen = {}
        if isinstance(chosen, dict):
            for k, default in layers.items():
                v = chosen.get(k)
                if isinstance(v, bool) and isinstance(default, bool):
                    layers[k] = v
                elif k == "line_range" and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                    layers[k] = float(v)
                elif k == "clip_style" and v in CLIP_STYLES:
                    layers[k] = v
        return {"available": self._proximity_enabled, **layers}

    def set_proximity_preview(self, change: dict) -> dict:
        """Change preview layers ({"masks"/"lines"/"boxes": bool, "line_range": person heights}) and the clips' style
        ("masks" | "boxes") for every camera and viewer; the others keep their state."""
        state = self.proximity_preview()
        available = state.pop("available")
        state.update(change)
        self._redis.set(_PROXIMITY_PREVIEW_KEY, json.dumps(state))
        return {"available": available, **state}

    def _zones_applied_by_camera(self) -> dict:
        """{camera: version its zones process applied}: each zones process acknowledges the cameras it runs, so the
        zone editor waits on the camera it saved, not on whichever process answered last. A camera no process has
        acknowledged (an older sidecar) is absent; the editor then falls back to applied_version."""
        cameras = sorted({split_stream_key_id(sid)[0] for sid in self._sources})
        if not cameras:
            return {}
        values = self._redis.mget([f"{_ZONES_APPLIED_KEY}:{c}" for c in cameras])
        return {c: int(v) for c, v in zip(cameras, values) if v is not None}

    def zones_status(self) -> dict:
        """The zones' and proximity's versions saved and acknowledged, and whether anything will acknowledge them.
        Beside role sidecars (`<camera>@<role>`, the safety stack) every module is in the config whether or not a
        camera runs it: a part counts as on only while its role streams, so the zone editor never waits for a sidecar
        that is not running."""
        def _int_key(key: str) -> int:
            v = self._redis.get(key)
            return int(v) if v is not None else 0
        roles = {split_stream_key_id(sid)[1] for sid in self._sources} - {None, ""}
        analytics_on = self._analytics_enabled and (not roles or "zones" in roles)
        try:
            status = {
                "enabled": analytics_on,
                "version": _int_key(_ZONES_VERSION_KEY),
                "applied_version": _int_key(_ZONES_APPLIED_KEY),
                "applied_by_camera": self._zones_applied_by_camera(),
            }
            if self._proximity_enabled:
                status["proximity"] = {"version": _int_key(_PROXIMITY_VERSION_KEY),
                                       "applied_version": _int_key(_PROXIMITY_APPLIED_KEY)}
        except redis.RedisError:
            status = {"enabled": analytics_on, "version": 0, "applied_version": 0}
            if self._proximity_enabled:
                status["proximity"] = {"version": 0, "applied_version": 0}
        return status
