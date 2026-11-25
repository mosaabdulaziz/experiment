from app.chat_frame_buffer import ChatFrameBuffer


def test_append_and_snapshot_oldest_to_newest():
    buf = ChatFrameBuffer(maxlen=3)
    buf.append(1.0, b"a")
    buf.append(2.0, b"b")
    assert buf.snapshot() == [(1.0, b"a"), (2.0, b"b")]
    assert len(buf) == 2


def test_append_past_maxlen_drops_oldest():
    buf = ChatFrameBuffer(maxlen=2)
    buf.append(1.0, b"a")
    buf.append(2.0, b"b")
    buf.append(3.0, b"c")
    assert buf.snapshot() == [(2.0, b"b"), (3.0, b"c")]
    assert len(buf) == 2


def test_snapshot_is_a_copy():
    buf = ChatFrameBuffer(maxlen=3)
    buf.append(1.0, b"a")
    snap = buf.snapshot()
    snap.append((9.0, b"z"))
    assert len(buf) == 1  # mutating the snapshot does not affect the buffer


def test_clear():
    buf = ChatFrameBuffer(maxlen=3)
    buf.append(1.0, b"a")
    buf.clear()
    assert buf.snapshot() == []
    assert len(buf) == 0


def test_maxlen_floor_is_one():
    buf = ChatFrameBuffer(maxlen=0)
    buf.append(1.0, b"a")
    buf.append(2.0, b"b")
    assert buf.snapshot() == [(2.0, b"b")]
