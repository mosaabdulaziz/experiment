"""Measured temporal memory of a camera: how long each tracked object has been there
(TrackObserver, fed from the detections already on iris:meta) and, optionally, what the
model said it was doing (SceneMemory, the interpreted half).

Off unless `monitoring.memory.enabled`. When off, every method here is a no-op and
`analysis_context` returns empty strings, which is what keeps the rendered prompt
byte-identical to the one built before memory existed -- so a deployment that does not
turn it on (attendance, barber, fire) behaves exactly as before.
"""
from __future__ import annotations

import logging
import time

from ..scene_memory import SceneMemory
from ..track_observer import TrackObserver
from ..zones import mux_resolution, stream_zones

logger = logging.getLogger("iris.stream")


class MeasuredMemory:
    def __init__(self, config: dict):
        self._config = config
        mem_cfg = (config.get("monitoring", {}) or {}).get("memory", {}) or {}
        self._cfg = mem_cfg
        self.enabled = bool(mem_cfg.get("enabled", False))
        # The interpreted half. `enabled` is the master for both halves, so scene_delta
        # alone is inert: the measured half is free at inference time, this one costs
        # output tokens on every call and changes the response shape.
        self.scene_delta = self.enabled and bool(mem_cfg.get("scene_delta", False))
        # Detections are in mux pixels, so the observer needs the mux frame size to
        # turn stationary_radius (a fraction of the diagonal) into a pixel threshold.
        self._mux_w, self._mux_h = mux_resolution(config)

        self.observers: dict = {}   # camera -> TrackObserver
        self.memories: dict = {}    # camera -> SceneMemory (one per stream, never shared)

    def add_camera(self, sid: str) -> None:
        if not self.enabled:
            return
        m = self._cfg
        self.observers[sid] = TrackObserver(
            frame_width=self._mux_w, frame_height=self._mux_h,
            window_seconds=float(m.get("entity_window_seconds", 120)),
            stale_seconds=float(m.get("entity_stale_seconds", 30)),
            stationary_radius=float(m.get("stationary_radius", 0.02)),
            max_rendered=int(m.get("max_entities_rendered", 20)))
        # One SceneMemory per stream, never one shared: the model's delta describes
        # the scene in front of ONE camera, and a shared instance would render cam0's
        # situations into cam1's prompt.
        if self.scene_delta:
            self.memories[sid] = SceneMemory(
                max_situations=int(m.get("max_situations", 8)),
                ttl_cycles=int(m.get("situation_ttl_cycles", 3)),
                notes_max_chars=int(m.get("scene_notes_max_chars", 400)))

    def remove_camera(self, sid: str) -> None:
        self.observers.pop(sid, None)
        self.memories.pop(sid, None)

    def observe_tick(self, sid: str, src, last_meta_id):
        """Feed one tick's detections to the stream's TrackObserver; return the new cursor.

        Never raises. Memory is an aid and the chat buffer is the sampler's actual
        product, so a bad read costs one tick of duration, never the frame history.

        Skips a meta entry already consumed: the sampler and the sidecar run at the
        same nominal rate but not in lockstep, so the newest entry is often one
        already seen -- and re-observing it would let a stalled sidecar go on
        accruing duration for tracks nobody has actually seen since.
        """
        observer = self.observers.get(sid)
        if observer is None:
            return last_meta_id
        try:
            reading = src.grab_detections()
            if reading is None:
                return last_meta_id
            eid, dets = reading
            if eid == last_meta_id:
                return last_meta_id
            observer.observe(time.monotonic(), dets)
            return eid
        except Exception as exc:
            logger.warning("[%s] track observer skipped a tick: %s", sid, exc)
            return last_meta_id

    def known_zones(self, sid: str) -> set:
        """The stream's configured zone names -- what SceneMemory filters areas against.

        An empty set is the honest answer for a stream with no zones drawn: every area
        the model then reports names a zone that does not exist. Never raises; the
        zones file is read from disk on every call.
        """
        try:
            return {name for z in stream_zones(self._config, sid)
                    if (name := (z.get("name") or "").strip())}
        except Exception as exc:
            logger.warning("[%s] could not read zone names for scene memory: %s", sid, exc)
            return set()

    def analysis_context(self, sid: str) -> dict:
        """Keyword arguments for Analyzer.analyze: "" / None for everything when off.

        `time.monotonic()` throughout, because `analyze` stamps the merge with it too. A
        time.time() here would render every duration as the seconds since 1970 without
        raising anything.
        """
        observer = self.observers.get(sid)
        memory = self.memories.get(sid)
        return dict(
            tracked=observer.render(time.monotonic()) if observer is not None else "",
            scene=memory.render(time.monotonic()) if memory is not None else "",
            scene_memory=memory,
            # Without these two the merge keeps every entity note the model ever wrote,
            # dead track or not, and every zone name it ever invented.
            live_track_ids=observer.live_ids() if observer is not None else None,
            known_zones=self.known_zones(sid),
        )

    def chat_block(self) -> str:
        """Every live stream's SCENE MEMORY block, labeled by camera. `""` when empty.

        Pulled by the Analyzer on a NORMAL chat turn only (search turns drop it, as
        they already drop the analytics block: its durations are a second clock the
        model confuses with the footage HH:MM:SS headers it must cite). Always labeled
        with the camera id, even for a single stream -- an unattributed block of
        situations in a cross-stream prompt is worse than none.
        """
        now = time.monotonic()
        parts = []
        for sid in sorted(self.memories):
            block = self.memories[sid].render(now)
            if block:
                parts.append(f"=== Camera {sid} ==={block}")
        return "\n" + "\n".join(parts) if parts else ""
