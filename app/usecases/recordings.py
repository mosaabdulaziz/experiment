"""Smart-record clips: which cameras record with NVIDIA smart-record (SVR), asking
the pipeline to record for an alert decided in this app, and receiving the
finished files it announces."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import redis

from ..redis_streams import read_stream

logger = logging.getLogger("iris.stream")


class Recordings:
    def __init__(self, sidecar_cfg: dict, redis_client, clip_service,
                 running: Callable[[], bool],
                 on_unclaimed_file: Callable[[str, str], Awaitable[None]]):
        """`on_unclaimed_file(camera, path)` gets a finished recording that no
        queued alert was waiting for -- fire incidents attach those as follow-up
        clips while an incident stays open."""
        self._redis = redis_client
        self._clip_service = clip_service
        self._running = running
        self._on_unclaimed_file = on_unclaimed_file

        self.svr_sources: set = set()
        self.refresh(sidecar_cfg)

    def refresh(self, sidecar_cfg: dict) -> None:
        """Streams whose alert clips arrive as a finished smart-record file rather than from the segment index. The
        sidecar owns recording, so its config is authoritative about which recorder each source got. Called again when
        the pipeline's sources change (the safety product's Apply): new ones are added, none forgotten."""
        try:
            from perception.engine.recorder_kind import recorder_kind
            from perception.engine.sources import resolve_sources
            rec_default = sidecar_cfg.get("recorder", "auto")
            for s in resolve_sources(sidecar_cfg):
                if recorder_kind(s["uri"], s.get("recorder", rec_default)) == "svr":
                    self.svr_sources.add(s["id"])
                    if self._clip_service is not None:
                        self._clip_service.mark_svr_source(s["id"])
            if self.svr_sources:
                logger.info("smart-record streams (alert clips resolve via sr-done): %s",
                            sorted(self.svr_sources))
        except Exception as exc:
            logger.warning(
                "could not determine recorder kinds from sidecar.yaml: %s -- alert clips "
                "will wait on the segment index for every stream, which is wrong for RTSP "
                "sources and resolves their alerts 'unavailable'", exc)

    def clips_enabled(self) -> bool:
        return self._clip_service is not None and self._clip_service.enabled

    def request_svr(self, sid: str, pts: float) -> None:
        """Ask the sidecar to open a smart-record session for an app-side alert.

        Zone and overcrowding alerts are decided inside the sidecar's own buffer probe,
        which starts the recording on the spot. A proactive VLM alert is decided here,
        and the nvurisrcbin that owns smart-record lives in the other container -- so
        without this request its clip job waits out the deadline and resolves
        'unavailable' every time.

        `pts` is the sidecar's own frame clock (it reached us through the meta stream),
        which lets the sidecar age the request by subtraction instead of comparing wall
        clocks across two containers. Best-effort: a lost request costs this alert its
        clip, which is what used to happen to all of them.
        """
        try:
            self._redis.xadd(f"iris:svr_request:{sid}", {"pts": f"{pts:.3f}"},
                             maxlen=32, approximate=True)
        except redis.RedisError as exc:
            logger.warning("[%s] could not request a smart-record clip: %s", sid, exc)

    async def run_for(self, sid: str) -> None:
        """Watches this stream's iris:svr_done:<id> Redis stream and hands each
        finished smart-record file to the clip service.

        Smart-record streams write nothing continuously, so their alert jobs never
        become 'covered' in the segment index -- this announcement is the only thing
        that resolves them. A lost entry is not fatal: the job's own deadline expires
        it to 'unavailable' rather than leaving the event on 'pending'.

        Both alert kinds are covered: the sidecar's probe starts a recording for zone
        and overcrowding alerts itself, and proactive VLM alerts reach it through
        request_svr. A VLM alert older than the sidecar's smart-rec-cache still gets
        no clip -- the footage is genuinely gone by then -- and resolves 'unavailable'.
        """
        try:
            async for f in read_stream(self._redis, f"iris:svr_done:{sid}", self._running):
                try:
                    path = f.get("path")
                    if not path:
                        continue
                    if self._clip_service is None:
                        logger.warning("[%s] smart-record file %s arrived but no "
                                       "clip service is configured", sid, path)
                        continue
                    try:
                        recorded_at = float(f.get("recorded_at"))     # the file's end: places its overlay boxes
                    except (TypeError, ValueError):
                        recorded_at = None
                    if not self._clip_service.attach_svr_file(sid, path, recorded_at=recorded_at):
                        # No queued alert job -- normal once a fire/smoke incident's
                        # first clip has already resolved but the incident is still
                        # open (sidecar.py keeps re-arming smart-record for as long
                        # as it stays open). Attach as a follow-up clip instead of
                        # dropping it as an orphan.
                        await self._on_unclaimed_file(sid, path, recorded_at)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("[%s] svr_done sampler iteration failed: %s", sid, exc,
                                 exc_info=True)
        except asyncio.CancelledError:
            pass
