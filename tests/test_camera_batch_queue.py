"""Batch queue: file playback stays live, but no batch is silently skipped.

The first fix froze playback during each VLM call — the visible stream
stalled for the whole call. The queue replaces the freeze: footage that
plays during a slow call is analyzed NEXT (oldest batch first); on
overflow the oldest batch is dropped, prioritizing recent footage.
"""
import asyncio

from app.camera import Camera


def _camera(batch_queue_depth=3):
    cfg = {
        "camera": {"source": 0},
        "sampling": {"interval_seconds": 1.0, "frames_per_batch": 3, "overlap_frames": 1,
                      "batch_queue_depth": batch_queue_depth},
    }
    return Camera(cfg)


def _feed(cam, n, start=0):
    """Push n sampled frames, bypassing the wall-clock rate limiter.

    new_frames_per_cycle = frames_per_batch - overlap = 2 per batch here.
    Frame payloads encode the sequence number so assertions can check order.
    """
    for i in range(n):
        k = start + i
        cam._accept_frame(f"f{k}".encode(), float(k))


def test_frames_accumulate_into_complete_batches():
    cam = _camera()
    _feed(cam, 2)
    assert len(cam._frame_queue) == 1
    assert not cam._frame_pending


def test_incomplete_batch_waits():
    cam = _camera()
    _feed(cam, 1)
    assert not cam._frame_queue
    assert len(cam._frame_pending) == 1


def test_grab_consumes_oldest_first():
    cam = _camera()
    _feed(cam, 4)  # batches: [f0,f1] [f2,f3]
    frames, offsets = cam.grab_frame_batch()
    assert frames == [b"f0", b"f1"]
    assert offsets == [0.0, 1.0]
    frames, offsets = cam.grab_frame_batch()
    # overlap tail of the previous batch ([f1]) prepended to [f2,f3]
    assert frames == [b"f1", b"f2", b"f3"]
    assert offsets == [1.0, 2.0, 3.0]


def test_grab_falls_back_to_pending_then_live():
    cam = _camera()
    cam._latest_frame = b"live"
    frames, offsets = cam.grab_frame_batch()  # nothing queued or pending
    assert frames == [b"live"] and offsets == [0.0]

    _feed(cam, 1)  # only an incomplete batch
    frames, _ = cam.grab_frame_batch()
    assert frames == [b"f0"]


def test_overflow_drops_oldest_batch():
    cam = _camera(batch_queue_depth=2)
    _feed(cam, 8)  # four batches into a queue of 2 -> oldest two dropped
    assert len(cam._frame_queue) == 2
    frames, _ = cam.grab_frame_batch()
    assert frames == [b"f4", b"f5"]
    frames, _ = cam.grab_frame_batch()
    assert frames == [b"f5", b"f6", b"f7"]  # overlap tail (f5) prepended by design


def test_annotated_path_mirrors_queue_discipline():
    cam = _camera()
    for i in range(2):
        cam._accept_annotated(f"a{i}".encode(), [("person", 0.9)], float(i))
    assert len(cam._annotated_queue) == 1
    batch = cam.grab_annotated_batch()
    assert batch is not None
    frames, dets, offs = batch
    assert frames == [b"a0", b"a1"]
    assert dets == [[("person", 0.9)]] * 2
    assert offs == [0.0, 1.0]
    assert cam.grab_annotated_batch() is None  # nothing left


def test_annotated_grab_falls_back_to_incomplete():
    cam = _camera()
    cam._accept_annotated(b"a0", [("truck", 0.5)], 0.0)
    batch = cam.grab_annotated_batch()
    assert batch is not None and batch[0] == [b"a0"]
    assert cam._annotated_pending == []


def test_stop_clears_queues():
    cam = _camera()
    _feed(cam, 4)
    cam._annotated_pending = [(b"x", [], 0.0)]
    asyncio.run(cam.stop())
    assert not cam._frame_queue and not cam._frame_pending
    assert not cam._annotated_queue and not cam._annotated_pending
