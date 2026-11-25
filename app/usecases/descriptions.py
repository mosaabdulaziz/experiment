"""Alert descriptions: when an alert should be described, frames around the
moment it fired are picked from the chat history, the zone is drawn on them,
and the shared VLM worker turns them into one sentence. Every path ends in a
description -- the real one, or a fallback -- so the UI never waits forever."""
from __future__ import annotations

import asyncio
import logging

from ..chat_select import frames_near_offset
from ..frame_extract import annotate_zone
from ..zones import zone_by_id

logger = logging.getLogger("iris.stream")

_DESCRIPTION_FALLBACK = "Description unavailable."


class Descriptions:
    def __init__(self, config: dict, analyzer, store, conn_manager, chat, vlm_enabled: bool):
        self._config = config
        self._analyzer = analyzer
        self._store = store
        self._conn = conn_manager
        self._chat = chat
        zones_cfg = config.get("zones", {})
        self.describe_intrusions = bool(zones_cfg.get("describe_intrusions", False))
        self.describe_overcrowding = bool(zones_cfg.get("describe_overcrowding", False))
        alerts_cfg = config.get("alerts", {})
        self.describe_fire = bool(alerts_cfg.get("describe_fire", False))
        # A description is a VLM call, so it is impossible without an analyzer. Forced off
        # here rather than at the emit sites: without it the alert saved with
        # description_status='pending' and its job went into self.jobs, which only the VLM
        # worker drains -- and run() never starts that worker when the VLM is off. Nothing
        # resolved the event, so the UI showed "preparing description" forever.
        if not vlm_enabled and (self.describe_intrusions or self.describe_overcrowding
                                or self.describe_fire):
            logger.warning(
                "zones.describe_intrusions/describe_overcrowding/alerts.describe_fire are "
                "set but monitoring.vlm_enabled is false -- a description is a VLM call, so "
                "alerts will be emitted without one. Set vlm_enabled: true to get "
                "descriptions.")
            self.describe_intrusions = False
            self.describe_overcrowding = False
            self.describe_fire = False
        self.frame_count = int(zones_cfg.get("describe_frame_count", 6))
        self.window_padding = float(zones_cfg.get("describe_window_padding_seconds", 2.0))
        # "__desc__:<event id>" -> (camera, frames, prompt_kind, count): prepared jobs
        # waiting for the shared VLM worker, which drains them alongside analysis.
        self.jobs: dict = {}
        # event id -> (camera, task) for frame preparation still in progress.
        self.tasks: dict = {}

    def start(self, sid: str, event_id: int, pts: float, prompt_kind: str,
              count: "int | None", zone_id: "str | None") -> None:
        """Begin describing a just-saved alert: prepare its frames in the background."""
        task = asyncio.create_task(self.prepare(sid, event_id, pts, prompt_kind, count, zone_id))
        self.tasks[event_id] = (sid, task)

    async def finish(self, event_id: int, description: str) -> None:
        """Store the description and tell the UI -- the last step of every path."""
        record = await self._store.update_description(event_id, "ready", description)
        if record is not None:
            await self._conn.broadcast({"type": "event_update", "id": event_id,
                                        "description_status": "ready",
                                        "description": description})

    async def finish_with_fallback(self, event_id: int) -> None:
        await self.finish(event_id, _DESCRIPTION_FALLBACK)

    def cancel_for(self, sid: str) -> None:
        for eid, (dsid, dtask) in list(self.tasks.items()):
            if dsid == sid:
                dtask.cancel()
                self.tasks.pop(eid, None)

    def all_tasks(self) -> list:
        return [dtask for _sid, dtask in self.tasks.values()]

    async def prepare(self, sid: str, event_id: int, pts: float,
                                    prompt_kind: str = "intrusion", count: "int | None" = None,
                                    zone_id: "str | None" = None):
        """Selects sample frames from this stream's live chat/analysis buffer around
        `pts`, annotates them with the zone that fired (resolved from zone_id; no
        annotation at all when zone_id is None -- there is nothing to draw, and drawing
        an arbitrary unrelated zone would mislead a zone-less alert kind's description,
        e.g. fire/smoke), and hands them off via self.jobs for the shared worker to
        describe. If no frames fall in the window, resolves the event directly with the
        fallback text -- no VLM call is ever attempted without frames in hand. `prompt_kind`/
        `count` are carried through to the eventual Analyzer call -- 'intrusion' jobs ignore
        count."""
        frames = None
        try:
            buf = self._chat.buffers.get(sid)
            snapshot = buf.snapshot() if buf is not None else []
            frames = frames_near_offset(snapshot, pts, self.window_padding,
                                        self.frame_count)
            if frames and zone_id:
                zone = zone_by_id(self._config, sid, zone_id)
                if zone and zone["points"]:
                    frames = await asyncio.to_thread(
                        lambda: [annotate_zone(f, zone["points"], zone["intrusion"]["mode"])
                                 for f in frames])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[%s] description frame prep failed for event %s", sid, event_id,
                           exc_info=True)
            frames = None
        finally:
            self.tasks.pop(event_id, None)

        if not frames:
            await self.finish(event_id, _DESCRIPTION_FALLBACK)
            return

        self.jobs[f"__desc__:{event_id}"] = (sid, frames, prompt_kind, count)

    async def process(self, event_id: int, sid: str, frames: list,
                                    prompt_kind: str = "intrusion", count: "int | None" = None):
        try:
            description = await self._analyzer.describe_alert(prompt_kind, frames, sid, count)
        except Exception:
            logger.warning("[%s] describe_%s failed for event %s", sid, prompt_kind, event_id,
                           exc_info=True)
            description = _DESCRIPTION_FALLBACK
        await self.finish(event_id, description)
