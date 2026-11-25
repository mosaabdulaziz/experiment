"""Periodic VLM analysis: each camera queues a batch of recent frames every few
seconds, the shared VLM worker (StreamManager._worker) asks the model about it,
and what the model reports is saved -- through a cooldown and a duplicate
filter per camera -- as a comment or an alert."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from ..cv_enricher import CVBatchContext, Detection, FrameMeta
from ..events import save_kwargs_from_result
from ..mode import analysis_enabled
from ..similarity import CommentFilter

logger = logging.getLogger("iris.stream")


def enqueue_drop_oldest(queue: "asyncio.Queue", item) -> bool:
    """Put item; if the queue is full, drop the oldest first. Returns True if a drop happened."""
    dropped = False
    if queue.full():
        try:
            queue.get_nowait()
            dropped = True
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)
    return dropped


def should_emit(event_type: str, last_times: dict, cooldown: float,
                alert_cooldown: float, now: float) -> bool:
    """Cooldown gate for one stream: enough time since the last event of this type?"""
    if event_type == "alert":
        return now - last_times.get("alert", 0.0) >= alert_cooldown
    if event_type == "comment":
        return now - last_times.get("comment", 0.0) >= cooldown
    return False


def build_frame_metas(offsets, dets_list) -> list[FrameMeta]:
    """Pair each frame's stream offset with its detections.

    dets_list entries are the wire tuples from PerceptionSource.parse_dets;
    this is where they become the app's Detection domain model.
    """
    return [FrameMeta(offset=o, detections=[Detection(*d) for d in dets])
            for o, dets in zip(offsets, dets_list)]


class VlmAnalysis:
    def __init__(self, config: dict, analyzer, store, conn_manager, clip_service, recordings,
                 metrics, running: Callable[[], bool], memory=None):
        self._memory = memory           # MeasuredMemory | None: tracked/scene prompt context
        self._config = config
        self._analyzer = analyzer
        self._store = store
        self._conn = conn_manager
        self._clip_service = clip_service
        self._recordings = recordings
        self._metrics = metrics
        self._running = running

        self._queue_depth = int(config.get("perception", {}).get("queue_depth", 3))
        sampling = config.get("sampling", {})
        self._interval = float(sampling.get("interval_seconds", 4))
        mon = config.get("monitoring", {})
        self._cooldown = float(mon.get("cooldown_seconds", 8))
        self._alert_cooldown = float(mon.get("alert_cooldown_seconds", 4))
        self._max_history = int(mon.get("max_history", 3))

        self.queues: dict = {}          # camera -> bounded queue of batches for the worker
        self.filters: dict = {}         # camera -> duplicate-message filter
        self.cooldown_times: dict = {}  # camera -> last emit time per event type

    def add_camera(self, sid: str) -> None:
        self.queues[sid] = asyncio.Queue(maxsize=self._queue_depth)
        self.filters[sid] = CommentFilter(self._config)
        self.cooldown_times[sid] = {"comment": 0.0, "alert": 0.0}

    def remove_camera(self, sid: str) -> None:
        self.queues.pop(sid, None)
        self.filters.pop(sid, None)
        self.cooldown_times.pop(sid, None)

    async def run_for(self, sid: str, src) -> None:
        """Per-camera producer: every `interval` seconds, queue the latest annotated
        batch for the shared VLM worker, dropping the oldest when it falls behind."""
        try:
            while self._running():
                await asyncio.sleep(self._interval)
                if not src.is_running:
                    continue
                if not analysis_enabled(self._config.get("monitoring", {})):
                    continue
                batch = src.grab_annotated_batch()      # (frames, dets_list, offsets) | None
                if batch is None:
                    continue
                frames, dets_list, offsets = batch
                # time.time() rides along so the alert clip can be aligned to wall-clock
                if enqueue_drop_oldest(self.queues[sid], (sid, time.monotonic(), frames, dets_list, offsets, time.time())):
                    logger.warning("queue full for %s — dropped oldest (raise interval_seconds)", sid)
        except asyncio.CancelledError:
            pass

    async def process(self, sid, frames, dets_list, offsets, wall_ts=None):
        """Ask the VLM about one batch and save what it reports, through the
        per-camera cooldown and duplicate filter."""
        frame_metas = build_frame_metas(offsets, dets_list)
        cv_ctx = CVBatchContext(frames=frames, frame_metas=frame_metas)
        offset_start = offsets[0] if offsets else None
        offset_end = offsets[-1] if offsets else None
        history = await self._store.get_recent(limit=self._max_history,
                                               types=("comment", "alert"), camera=sid)
        # Empty strings / None when memory is off: the rendered prompt is then byte-identical
        # to the pre-memory one.
        memory_ctx = self._memory.analysis_context(sid) if self._memory is not None else {}
        results = await self._analyzer.analyze(
            cv_ctx.frames, history,
            offset_start=offset_start, offset_end=offset_end, source_id=sid,
            **memory_ctx,
        )
        times = self.cooldown_times[sid]
        cfilter = self.filters[sid]
        for result in results:
            etype = result.get("type", "")
            message = result.get("message", "")
            tags = result.get("tags", [])
            if not should_emit(etype, times, self._cooldown, self._alert_cooldown, time.monotonic()):
                continue
            if await cfilter.is_duplicate(message, etype):
                continue
            clip_status = ("pending" if etype == "alert" and self._recordings.clips_enabled() else None)
            record = await self._store.save(etype, message, tags,
                                            offset_start=offset_start, offset_end=offset_end,
                                            camera=sid, clip_status=clip_status,
                                            **save_kwargs_from_result(result))
            cfilter.record_emitted(etype)
            times[etype] = time.monotonic()
            await self._conn.broadcast({"type": "event", "event": record, "camera": sid})
            if clip_status == "pending":
                # The VLM judged the whole batch, so centre the clip on its midpoint.
                anchor = (None if offset_start is None or offset_end is None
                          else (offset_start + offset_end) / 2.0)
                if anchor is not None and sid in self._recordings.svr_sources:
                    # Ask before submitting: the recording is the long pole, and the
                    # sidecar can only reach back as far as its pre-roll cache holds.
                    self._recordings.request_svr(sid, anchor)
                await self._clip_service.submit(record["id"], sid, anchor,
                                                wall_time=wall_ts or time.time())
            if self._metrics is not None:
                self._metrics.note_event(sid)
