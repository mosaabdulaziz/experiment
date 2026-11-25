"""Chat history: a rolling buffer of recent frames per camera, so the chat can
look at what the cameras saw. The same three methods LocalChatContext offers,
so server.py can use either one as its chat context."""
from __future__ import annotations

import asyncio
from typing import Callable

from ..chat_frame_buffer import ChatFrameBuffer
from ..chat_select import _stride_pick, recent_strided, select_chat_frames


class ChatHistory:
    def __init__(self, config: dict, running: Callable[[], bool],
                 on_tick: "Callable | None" = None):
        # on_tick(sid, src, last_meta_id) -> last_meta_id, called once per sampler tick
        # (measured memory feeds its track observer from here); None = nothing extra.
        self._on_tick = on_tick
        chat_cfg = config.get("chat", {})
        self.chat_interval = float(chat_cfg.get("chat_interval_seconds", 10))
        self.chat_buffer_size = int(chat_cfg.get("chat_buffer_size", 4))
        self.max_chat_images = int(chat_cfg.get("max_chat_images", 12))

        hist_cfg = chat_cfg.get("history", {})
        self.fps = max(float(hist_cfg.get("fps", 2)), 0.1)
        window_seconds = float(hist_cfg.get("window_seconds", 300))
        self.history_size = max(int(window_seconds * self.fps), 1)
        search_cfg = chat_cfg.get("search", {})
        self.window_frames = max(int(search_cfg.get("window_frames", 4)), 1)
        self.sample_interval = 1.0 / self.fps

        self.buffers: dict = {}
        self._running = running

    def add_camera(self, sid: str) -> None:
        self.buffers[sid] = ChatFrameBuffer(self.history_size)

    def remove_camera(self, sid: str) -> None:
        self.buffers.pop(sid, None)

    async def run_for(self, sid: str, src) -> None:
        """Slow sampler: snapshot the latest annotated JPEG into the chat buffer
        on the chat cadence (independent of the fast analysis producer), and hand the same
        tick to `on_tick` when one is set."""
        last_meta_id = None
        try:
            while self._running():
                await asyncio.sleep(self.sample_interval)
                if not src.is_running:
                    continue
                # Before the jpeg guard: a stream can have detections on iris:meta while
                # its latest JPEG is momentarily absent, and the durations should not
                # stall for that.
                if self._on_tick is not None:
                    last_meta_id = self._on_tick(sid, src, last_meta_id)
                jpeg = src.grab_frame()
                if jpeg is None:
                    continue
                buf = self.buffers.get(sid)
                if buf is not None:
                    buf.append(src.current_offset, jpeg)
        except asyncio.CancelledError:
            pass

    def gather_chat_context(self):
        """Build each live stream's normal-chat context from the shared history
        buffer: chat_buffer_size frames spaced ~chat_interval_seconds apart, capped
        across streams. Returns ({camera_id: [jpeg]}, {camera_id: [offset]},
        included_ids) — the offsets are each frame's stream position so search
        mode can timestamp the frames it shows the model."""
        step = max(round(self.chat_interval * self.fps), 1)
        raw: dict = {}
        for sid in sorted(self.buffers):
            buf = self.buffers.get(sid)
            if buf is None:
                continue
            try:
                raw[sid] = recent_strided(buf.snapshot(), self.chat_buffer_size, step,
                                          with_offsets=True)
            except Exception:
                continue
        paired, ids = select_chat_frames(raw, self.max_chat_images)
        frames = {cid: [j for _o, j in v] for cid, v in paired.items()}
        offsets = {cid: [o for o, _j in v] for cid, v in paired.items()}
        return frames, offsets, ids

    def sweep_history_snapshots(self) -> dict:
        out: dict = {}
        for sid in sorted(self.buffers):
            buf = self.buffers.get(sid)
            out[sid] = buf.snapshot() if buf is not None else []
        return out

    def sweep_live_window(self, since: float) -> tuple:
        out: dict = {}
        offsets: dict = {}
        new_since = since
        for sid in sorted(self.buffers):
            buf = self.buffers.get(sid)
            if buf is None:
                continue
            newer = [(o, j) for o, j in buf.snapshot() if o > since]
            if newer:
                new_since = max(new_since, newer[-1][0])
            picked = _stride_pick(newer, self.window_frames)
            if picked:
                out[sid] = [j for _o, j in picked]
                offsets[sid] = [o for o, _j in picked]
        return out, offsets, new_since
