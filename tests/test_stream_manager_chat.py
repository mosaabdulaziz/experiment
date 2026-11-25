import json
import fakeredis
from app.stream_manager import StreamManager
from app.chat_frame_buffer import ChatFrameBuffer


def _cfg():
    return {
        "perception": {"redis_url": "redis://x", "queue_depth": 2, "discovery_interval_seconds": 1},
        "sampling": {"interval_seconds": 1, "frames_per_batch": 4},
        "monitoring": {},
        "chat": {"chat_interval_seconds": 10, "chat_buffer_size": 4, "max_chat_images": 6},
    }


def _mgr():
    r = fakeredis.FakeRedis()
    return StreamManager(_cfg(), analyzer=None, store=None, conn_manager=None, redis_client=r)


def test_chat_config_read():
    m = _mgr()
    assert m.chat.chat_interval == 10
    assert m.chat.chat_buffer_size == 4
    assert m.chat.max_chat_images == 6


def test_gather_chat_context_caps_and_skips_empty():
    m = _mgr()
    m.chat.buffers = {
        "cam0": ChatFrameBuffer(4),
        "cam1": ChatFrameBuffer(4),
        "cam2": ChatFrameBuffer(4),   # left empty -> skipped
    }
    for b in (b"0", b"1", b"2", b"3"):
        m.chat.buffers["cam0"].append(0.0, b)
        m.chat.buffers["cam1"].append(0.0, b)
    frames, offsets, ids = m.chat.gather_chat_context()
    assert ids == ["cam0", "cam1"]
    total = sum(len(v) for v in frames.values())
    assert total <= 6
    # offsets are aligned per camera with the returned frames
    assert {cid: len(o) for cid, o in offsets.items()} == {cid: len(f) for cid, f in frames.items()}


def test_gather_chat_context_empty_when_no_frames():
    m = _mgr()
    m.chat.buffers = {"cam0": ChatFrameBuffer(4)}
    assert m.chat.gather_chat_context() == ({}, {}, [])


def _cfg_hist():
    c = _cfg()
    c["chat"].update({"history": {"window_seconds": 10, "fps": 2},
                      "search": {"round_seconds": 4, "window_frames": 2}})
    return c


def test_history_size_from_window_and_fps():
    import fakeredis
    from app.stream_manager import StreamManager
    m = StreamManager(_cfg_hist(), analyzer=None, store=None, conn_manager=None,
                      redis_client=fakeredis.FakeRedis())
    assert m.chat.history_size == 20  # 10s * 2fps
    assert m.chat.fps == 2


def test_sweep_history_snapshots_per_camera():
    m = _mgr()
    m.chat.buffers = {"cam0": ChatFrameBuffer(20), "cam1": ChatFrameBuffer(20)}
    for i in range(5):
        m.chat.buffers["cam0"].append(float(i), bytes([i]))
    snaps = m.chat.sweep_history_snapshots()
    assert [o for o, _ in snaps["cam0"]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert snaps["cam1"] == []


def test_sweep_live_window_returns_only_newer():
    m = _mgr()
    m.chat.window_frames = 2
    m.chat.buffers = {"cam0": ChatFrameBuffer(20)}
    for i in range(4):
        m.chat.buffers["cam0"].append(float(i), bytes([i]))
    frames, offsets, new_since = m.chat.sweep_live_window(1.0)
    assert new_since == 3.0
    assert frames["cam0"] == [b"\x02", b"\x03"]
    assert offsets["cam0"] == [2.0, 3.0]


def test_gather_chat_context_selects_by_chat_buffer_knobs():
    m = _mgr()
    m.chat.chat_buffer_size = 3
    m.chat.chat_interval = 1
    m.chat.fps = 2            # step = round(1*2) = 2; window = size*step = 6 frames
    m.chat.buffers = {"cam0": ChatFrameBuffer(100)}
    for i in range(10):
        m.chat.buffers["cam0"].append(float(i), bytes([i]))
    frames, offsets, ids = m.chat.gather_chat_context()
    picked = frames["cam0"]
    assert offsets["cam0"] == [4.0, 6.0, 9.0]  # aligned timestamps of the picked frames
    assert len(picked) == 3
    assert picked[0] == bytes([4])    # start of the last 6-frame window (4..9)
    assert picked[-1] == bytes([9])   # newest frame
    assert bytes([0]) not in picked
