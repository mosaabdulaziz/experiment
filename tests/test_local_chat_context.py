import asyncio

import pytest

from app.local_chat_context import LocalChatContext


class _FakeCam:
    def __init__(self, frame=b"f", running=True, paused=False):
        self._frame = frame
        self.is_running = running
        self.is_paused = paused
        self.current_offset = 1.0

    def grab_frame(self):
        return self._frame


class _RaisingCam(_FakeCam):
    """Camera whose grab_frame always raises, to exercise loop error handling."""

    def __init__(self):
        super().__init__(frame=b"a")
        self.calls = 0

    def grab_frame(self):
        self.calls += 1
        raise RuntimeError("boom")


def _cfg(interval=10):
    return {"chat": {"history": {"window_seconds": 30, "fps": 1.0 / interval},
                    "search": {"window_frames": 4},
                    "max_chat_images": 12}}


def test_sample_and_gather():
    ctx = LocalChatContext(_cfg(), _FakeCam(frame=b"a"))
    ctx.sample_once()
    ctx.sample_once()
    frames, offsets, ids = ctx.gather_chat_context()
    assert ids == ["camera"]
    assert frames["camera"] == [b"a", b"a"]
    assert offsets["camera"] == [1.0, 1.0]  # _FakeCam.current_offset


def test_sample_skips_when_not_running():
    ctx = LocalChatContext(_cfg(), _FakeCam(running=False))
    ctx.sample_once()
    assert ctx.gather_chat_context() == ({}, {}, [])


def test_sample_skips_when_no_frame():
    ctx = LocalChatContext(_cfg(), _FakeCam(frame=None))
    ctx.sample_once()
    assert ctx.gather_chat_context() == ({}, {}, [])


def test_run_propagates_cancelled_error_and_resets_running():
    async def go():
        ctx = LocalChatContext(_cfg(interval=0.001), _FakeCam(frame=b"a"))
        task = asyncio.create_task(ctx.run())
        await asyncio.sleep(0.01)  # let it tick at least once
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return ctx

    ctx = asyncio.run(go())
    assert ctx._running is False         # finally reset the flag
    assert len(ctx._buffer) > 0          # it sampled before cancellation


def test_run_swallows_sample_exceptions_and_keeps_looping():
    async def go():
        cam = _RaisingCam()
        ctx = LocalChatContext(_cfg(interval=0.001), cam)
        task = asyncio.create_task(ctx.run())
        await asyncio.sleep(0.02)
        alive = ctx._running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return cam, alive

    cam, alive = asyncio.run(go())
    assert alive is True       # loop stayed alive despite repeated exceptions
    assert cam.calls > 1       # it kept ticking past the first failure


class _FakeCam2:
    is_running = True
    is_paused = False
    current_offset = 0.0
    def grab_frame(self):
        return b"x"


def _ctx():
    cfg = {"chat": {"history": {"window_seconds": 10, "fps": 2},
                    "search": {"round_seconds": 4, "window_frames": 2},
                    "max_chat_images": 6}}
    return LocalChatContext(cfg, _FakeCam2())


def test_local_history_size_and_windows():
    ctx = _ctx()
    assert ctx._buffer._frames.maxlen == 20  # 10s * 2fps
    assert ctx._round_seconds == 4  # from _ctx() search config
    for i in range(4):
        ctx._buffer.append(float(i), bytes([i]))
    assert [o for o, _ in ctx.sweep_history_snapshots()["camera"]] == [0.0, 1.0, 2.0, 3.0]
    frames, offsets, new_since = ctx.sweep_live_window(1.0)
    assert new_since == 3.0
    assert frames["camera"] == [b"\x02", b"\x03"]
    assert offsets["camera"] == [2.0, 3.0]


def test_local_chat_buffer_config_read():
    cfg = {"chat": {"chat_buffer_size": 5, "chat_interval_seconds": 3,
                    "history": {"window_seconds": 10, "fps": 2}, "max_chat_images": 6}}
    ctx = LocalChatContext(cfg, _FakeCam2())
    assert ctx._chat_buffer_size == 5
    assert ctx._chat_interval == 3


def test_local_gather_chat_context_uses_chat_buffer_knobs():
    ctx = _ctx()
    ctx._chat_buffer_size = 3
    ctx._chat_interval = 1
    ctx._fps = 2          # step = round(1*2) = 2; window = size*step = 6 frames
    for i in range(10):
        ctx._buffer.append(float(i), bytes([i]))
    frames, offsets, ids = ctx.gather_chat_context()
    picked = frames["camera"]
    assert offsets["camera"] == [4.0, 6.0, 9.0]  # aligned timestamps of the picked frames
    assert len(picked) == 3
    assert picked[0] == bytes([4])    # start of the last 6-frame window (4..9)
    assert picked[-1] == bytes([9])   # newest frame
    assert bytes([0]) not in picked
