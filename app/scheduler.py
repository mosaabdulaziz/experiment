import asyncio
import logging
import time

from .analyzer import Analyzer
from .camera import Camera
from .cv_enricher import CVBatchContext, CVEnricher, FrameMeta
from .events import EventStore, save_kwargs_from_result
from .mode import analysis_enabled
from .similarity import CommentFilter

logger = logging.getLogger("iris.scheduler")


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


class Scheduler:
    def __init__(self, config: dict, camera: Camera, analyzer: Analyzer, store: EventStore, manager,
                 comment_filter: CommentFilter, enricher: CVEnricher, metrics=None):
        self._config = config
        self._metrics = metrics
        self._camera = camera
        self._analyzer = analyzer
        self._store = store
        self._manager = manager  # ConnectionManager
        self._comment_filter = comment_filter
        self._enricher = enricher

        sampling = config.get("sampling", {})
        self._interval = sampling.get("interval_seconds", 6)

        monitoring = config.get("monitoring", {})
        self._cooldown = monitoring.get("cooldown_seconds", 30)
        self._alert_cooldown = monitoring.get("alert_cooldown_seconds", 8)
        self._max_history = monitoring.get("max_history", 20)

        self._max_pending: int = monitoring.get("pending_queue_depth", 2)

        self._last_comment_time: float = 0.0
        self._last_alert_time: float = 0.0
        self._analyzing = False
        self._pending: int = 0

    async def run(self):
        while True:
            await asyncio.sleep(self._interval)
            if not self._camera.is_running or self._camera.is_paused:
                continue
            if not analysis_enabled(self._config.get("monitoring", {})):
                continue
            if self._analyzing:
                if self._pending >= self._max_pending:
                    logger.info("tick missed — pending queue full (%d/%d), dropping", self._pending, self._max_pending)
                else:
                    self._pending += 1
                    logger.info("tick queued — pending %d/%d (analysis still running)", self._pending, self._max_pending)
            else:
                asyncio.create_task(self._cycle())

    async def _cycle(self):
        self._analyzing = True
        await self._manager.broadcast({"type": "status", "status": "analyzing"})
        try:
            annotated_batch = self._camera.grab_annotated_batch()
            if annotated_batch is not None:
                # Stream annotator already ran YOLO — reuse its frames and detections
                frames, detections_list, offsets = annotated_batch
                frame_metas = [
                    FrameMeta(offset=off, detections=dets)
                    for off, dets in zip(offsets, detections_list)
                ]
                cv_ctx = CVBatchContext(frames=frames, frame_metas=frame_metas)
                self._enricher.save_debug_jpeg(frames[len(frames) // 2])
            else:
                frames, offsets = self._camera.grab_frame_batch()
                if not frames:
                    return
                loop = asyncio.get_event_loop()
                cv_ctx = await loop.run_in_executor(
                    None, self._enricher.process_batch, frames, offsets
                )

            offset_start = offsets[0] if offsets else None
            offset_end   = offsets[-1] if offsets else None
            ts_range = (
                f"[{_fmt_ts(offset_start)}-{_fmt_ts(offset_end)}]"
                if offset_start is not None and offset_end is not None
                else ""
            )
            logger.info("Analyzing %d frame(s) %s", len(frames), ts_range)
            self._enricher.log_cycle(cv_ctx)

            history = await self._store.get_recent(limit=self._max_history, types=("comment", "alert"))
            results = await self._analyzer.analyze(
                cv_ctx.frames, history,
                offset_start=offset_start, offset_end=offset_end,
            )

            for result in results:
                event_type = result.get("type", "")
                message = result.get("message", "")
                tags = result.get("tags", [])

                if not self._should_emit(event_type):
                    #logger.info("cooldown active — skipping %s: %s", event_type, message)
                    continue

                if await self._comment_filter.is_duplicate(message, event_type):
                    if event_type == "alert":
                        logger.info("alert suppressed by similarity filter: %s", message)
                    continue

                record = await self._store.save(
                    event_type, message, tags,
                    offset_start=offset_start, offset_end=offset_end,
                    **save_kwargs_from_result(result),
                )
                self._comment_filter.record_emitted(event_type)
                self._update_cooldown(event_type)
                await self._manager.broadcast({"type": "event", "event": record})
                if self._metrics is not None:
                    self._metrics.note_event("local")

        except Exception as exc:
            logger.error("Analysis cycle failed: %s", exc, exc_info=True)
            await self._manager.broadcast({"type": "error", "message": str(exc)})
        finally:
            self._analyzing = False
            run_catchup = (
                self._pending > 0
                and self._camera.is_running
                and not self._camera.is_paused  # a paused camera must not keep draining stale queued frames
                and analysis_enabled(self._config.get("monitoring", {}))
            )
            if run_catchup:
                self._pending -= 1
                logger.info("running catch-up cycle from pending queue (%d remaining)", self._pending)
                asyncio.create_task(self._cycle())
            else:
                self._pending = 0
                await self._manager.broadcast({"type": "status", "status": "idle"})

    def pending_depth(self) -> int:
        return self._pending

    def _should_emit(self, event_type: str) -> bool:
        now = time.monotonic()
        if event_type == "alert":
            return now - self._last_alert_time >= self._alert_cooldown
        if event_type == "comment":
            return now - self._last_comment_time >= self._cooldown
        return False

    def _update_cooldown(self, event_type: str):
        now = time.monotonic()
        if event_type == "alert":
            self._last_alert_time = now
        elif event_type == "comment":
            self._last_comment_time = now
