import asyncio
import logging

from .chat_frame_buffer import ChatFrameBuffer
from .chat_select import recent_strided, select_chat_frames

logger = logging.getLogger(__name__)


class LocalChatContext:
    """Single-camera analogue of StreamManager's chat context for the local
    backend. Slow-samples one Camera into a ChatFrameBuffer and exposes the same
    gather_chat_context() shape (one camera, keyed "camera")."""

    def __init__(self, config: dict, camera):
        chat_cfg = config.get("chat", {})
        hist_cfg = chat_cfg.get("history", {})
        self._fps = max(float(hist_cfg.get("fps", 2)), 0.1)
        self._interval = 1.0 / self._fps
        self._max_images = int(chat_cfg.get("max_chat_images", 12))
        self._chat_buffer_size = int(chat_cfg.get("chat_buffer_size", 4))
        self._chat_interval = float(chat_cfg.get("chat_interval_seconds", 10))
        history_size = max(int(float(hist_cfg.get("window_seconds", 300)) * self._fps), 1)
        self._buffer = ChatFrameBuffer(history_size)
        search_cfg = chat_cfg.get("search", {})
        self._window_frames = max(int(search_cfg.get("window_frames", 4)), 1)
        self._round_seconds = float(search_cfg.get("round_seconds", 30))
        self._camera = camera
        self._running = False

    def sample_once(self) -> None:
        cam = self._camera
        if not getattr(cam, "is_running", False) or getattr(cam, "is_paused", False):
            return
        jpeg = cam.grab_frame()
        if jpeg is None:
            return
        self._buffer.append(getattr(cam, "current_offset", 0.0), jpeg)

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    await asyncio.sleep(self._interval)
                    self.sample_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Local chat sampler error: %s", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def gather_chat_context(self):
        step = max(round(self._chat_interval * self._fps), 1)
        raw = {"camera": recent_strided(self._buffer.snapshot(), self._chat_buffer_size,
                                        step, with_offsets=True)}
        paired, ids = select_chat_frames(raw, self._max_images)
        frames = {cid: [j for _o, j in v] for cid, v in paired.items()}
        offsets = {cid: [o for o, _j in v] for cid, v in paired.items()}
        return frames, offsets, ids

    def sweep_history_snapshots(self) -> dict:
        return {"camera": self._buffer.snapshot()}

    def sweep_live_window(self, since: float) -> tuple[dict, dict, float]:
        from .chat_select import _stride_pick
        newer = [(o, j) for o, j in self._buffer.snapshot() if o > since]
        new_since = newer[-1][0] if newer else since
        picked = _stride_pick(newer, self._window_frames)
        if not picked:
            return {}, {}, new_since
        return ({"camera": [j for _o, j in picked]},
                {"camera": [o for o, _j in picked]}, new_since)
