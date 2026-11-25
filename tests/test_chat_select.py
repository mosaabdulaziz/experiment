from app.chat_select import recent_strided, frames_near_offset


def _snap(n):
    """Oldest->newest [(offset, jpeg)] with jpeg = bytes([i])."""
    return [(float(i), bytes([i])) for i in range(n)]


def test_returns_size_frames_from_recent_window():
    # 20 frames, size=4, step=2 -> window = last 8 (indices 12..19),
    # stride-picked to 4, oldest->newest, first+last of the window preserved.
    picked = recent_strided(_snap(20), size=4, step=2)
    assert len(picked) == 4
    assert picked[0] == bytes([12])   # start of the recent window
    assert picked[-1] == bytes([19])  # newest frame


def test_step_below_one_clamps_to_every_frame():
    # step=0 -> treated as 1; window = last `size` frames, all kept.
    picked = recent_strided(_snap(10), size=3, step=0)
    assert picked == [bytes([7]), bytes([8]), bytes([9])]


def test_fewer_frames_than_size_returns_all():
    picked = recent_strided(_snap(3), size=10, step=2)
    assert picked == [bytes([0]), bytes([1]), bytes([2])]


def test_empty_snapshot_returns_empty():
    assert recent_strided([], size=5, step=2) == []


def test_size_zero_returns_empty():
    assert recent_strided(_snap(5), size=0, step=2) == []


def test_with_offsets_returns_offset_jpeg_pairs():
    # Same selection as test_returns_size_frames_from_recent_window, but keeping offsets.
    picked = recent_strided(_snap(20), size=4, step=2, with_offsets=True)
    assert len(picked) == 4
    assert picked[0] == (12.0, bytes([12]))
    assert picked[-1] == (19.0, bytes([19]))


def test_frames_near_offset_filters_to_window_and_strides():
    # 20 frames at offsets 0..19; target=10, padding=3 -> window is offsets 7..13 (7 frames),
    # stride-picked to 4.
    snapshot = _snap(20)
    picked = frames_near_offset(snapshot, target=10.0, padding=3.0, count=4)
    assert len(picked) == 4
    assert picked[0] == bytes([7])    # start of the window
    assert picked[-1] == bytes([13])  # end of the window


def test_frames_near_offset_returns_all_when_fewer_than_count_in_window():
    snapshot = _snap(20)
    picked = frames_near_offset(snapshot, target=10.0, padding=1.0, count=10)
    assert picked == [bytes([9]), bytes([10]), bytes([11])]


def test_frames_near_offset_returns_empty_when_nothing_in_window():
    snapshot = _snap(20)
    assert frames_near_offset(snapshot, target=100.0, padding=2.0, count=4) == []


def test_frames_near_offset_returns_empty_for_empty_snapshot():
    assert frames_near_offset([], target=10.0, padding=2.0, count=4) == []
