from collections import deque


class ChatFrameBuffer:
    """Slow-sampled ring buffer of (offset, jpeg) for one stream's chat context.

    Distinct from the analysis buffers (Camera._frame_buffer / the Redis meta
    stream): sampled on the slow chat cadence so a small batch spans a long
    history window. Holds the latest `maxlen` frames, oldest->newest.
    """

    def __init__(self, maxlen: int):
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=max(int(maxlen), 1))

    def append(self, offset: float, jpeg: bytes) -> None:
        self._frames.append((offset, jpeg))

    def snapshot(self) -> list[tuple[float, bytes]]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)
