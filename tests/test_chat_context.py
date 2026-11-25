from app.chat_select import select_chat_frames, _stride_pick


def test_stride_pick_keeps_first_and_last():
    assert _stride_pick([1, 2, 3, 4, 5], 3) == [1, 3, 5]
    assert _stride_pick([1, 2, 3], 5) == [1, 2, 3]   # fewer than k -> all
    assert _stride_pick([1, 2, 3, 4], 1) == [4]       # k==1 -> most recent
    assert _stride_pick([], 3) == []


def test_select_drops_empty_cameras():
    frames, ids = select_chat_frames({"cam0": [b"a", b"b"], "cam1": []}, max_images=12)
    assert ids == ["cam0"]
    assert frames == {"cam0": [b"a", b"b"]}


def test_select_caps_total_images_via_per_camera_stride():
    buffers = {
        "cam0": [b"0", b"1", b"2", b"3"],
        "cam1": [b"0", b"1", b"2", b"3"],
        "cam2": [b"0", b"1", b"2", b"3"],
    }
    frames, ids = select_chat_frames(buffers, max_images=6)
    # 6 // 3 = 2 per camera
    assert ids == ["cam0", "cam1", "cam2"]
    total = sum(len(v) for v in frames.values())
    assert total <= 6
    assert all(len(v) == 2 for v in frames.values())
    assert frames["cam0"] == [b"0", b"3"]  # first + last preserved


def test_select_empty_input():
    assert select_chat_frames({}, max_images=12) == ({}, [])
    assert select_chat_frames({"cam0": []}, max_images=12) == ({}, [])
