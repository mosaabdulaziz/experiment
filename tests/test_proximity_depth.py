"""proximity.depth: letterbox, the relative depth gap of a pair, and the worker thread (with a stand-in model)."""
import time

import cv2
import numpy as np
import pytest

from perception.modules.proximity.camera import DepthRequest
from perception.modules.proximity.depth import DepthNotOnGpu, DepthWorker, depth_gap, ensure_on_gpu, letterbox
from perception.modules.proximity.shapes import Blob


def test_letterbox_centres_the_frame_on_grey_in_rgb_0_1():
    bgr = np.zeros((720, 1280, 3), np.uint8)
    bgr[..., 2] = 255                                          # red, in BGR
    x, s, top, left = letterbox(bgr, (736, 1280))
    assert x.shape == (1, 3, 736, 1280) and x.dtype == np.float32 and (s, top, left) == (1.0, 8, 0)
    assert x[0, 0, 100, 100] == 1.0 and x[0, 2, 100, 100] == 0.0
    assert x[0, 0, 0, 0] == pytest.approx(114 / 255)


def test_a_smaller_frame_is_scaled_up_to_fit():
    _x, s, top, left = letterbox(np.zeros((360, 640, 3), np.uint8), (736, 1280))
    assert (s, top, left) == (2.0, 8, 0)


def _scene(vehicle_depth):
    """A person (30 x 100 px at 600, 300) in front of a vehicle (300 x 150 at 500, 250), on a far background."""
    depth = np.full((720, 1280), 20.0, np.float32)
    depth[250:400, 500:800] = vehicle_depth
    depth[300:400, 600:630] = 5.0
    request = DepthRequest(("z", 1, 2), Blob(np.ones((100, 30), bool), 600, 300, 1.0, 1.0), (600, 300, 30, 100),
                           Blob(np.ones((150, 300), bool), 500, 250, 1.0, 1.0), (500, 250, 300, 150))
    return depth, request


def test_same_depth_gives_a_small_gap_and_a_vehicle_far_behind_a_large_one():
    near, req = _scene(5.3)
    assert depth_gap(near, (1.0, 1.0), req) == pytest.approx(0.06, abs=0.01)
    far, req = _scene(9.0)
    assert depth_gap(far, (1.0, 1.0), req) == pytest.approx(0.8, abs=0.01)


def test_a_half_resolution_map_gives_the_same_gap():
    depth, req = _scene(5.3)
    assert depth_gap(depth[::2, ::2], (0.5, 0.5), req) == pytest.approx(0.06, abs=0.01)


def test_a_region_outside_the_map_gives_none():
    depth, req = _scene(5.3)
    req.person_box, req.person_blob = (2000, 300, 30, 100), Blob(np.ones((100, 30), bool), 2000, 300, 1.0, 1.0)
    assert depth_gap(depth, (1.0, 1.0), req) is None


class _Model:
    def __init__(self, depth, fail_first=False):
        self.depth, self.fail_first, self.calls = depth, fail_first, 0

    def __call__(self, bgr):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("cuda hiccup")
        return self.depth, 1.0


def _wait_idle(worker, sid="cam-1"):
    deadline = time.monotonic() + 5
    while worker.busy(sid) and time.monotonic() < deadline:
        time.sleep(0.01)


def test_the_worker_measures_queued_frames_and_serves_recent_results():
    depth, req = _scene(5.3)
    w = DepthWorker(lambda: _Model(depth))
    w.start()
    try:
        assert w.submit("cam-1", 10.0, np.zeros((720, 1280, 3), np.uint8), (1.0, 1.0), [req])
        _wait_idle(w)
        measured_t, gap = w.latest("cam-1", req.key)
        assert measured_t == 10.0 and gap == pytest.approx(0.06, abs=0.01)   # with its time, however old
        assert w.latest("cam-2", req.key) is None                              # another camera
    finally:
        w.stop()


def test_a_full_queue_drops_the_new_frame():
    depth, req = _scene(5.3)
    w = DepthWorker(lambda: _Model(depth), max_pending=1)                    # not started: nothing drains it
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert w.submit("cam-1", 1.0, frame, (1.0, 1.0), [req])
    assert not w.submit("cam-1", 1.5, frame, (1.0, 1.0), [req])
    assert w.dropped == 1 and w.busy("cam-1")


def test_a_failing_frame_is_counted_and_the_next_one_is_measured():
    depth, req = _scene(5.3)
    w = DepthWorker(lambda: _Model(depth, fail_first=True))
    w.start()
    try:
        frame = np.zeros((720, 1280, 3), np.uint8)
        w.submit("cam-1", 1.0, frame, (1.0, 1.0), [req])
        _wait_idle(w)
        w.submit("cam-1", 2.0, frame, (1.0, 1.0), [req])
        _wait_idle(w)
        assert w.failed == 1 and w.latest("cam-1", req.key)[0] == 2.0
    finally:
        w.stop()


def test_a_model_that_cannot_load_refuses_start():
    def boom():
        raise RuntimeError("no such file")
    with pytest.raises(RuntimeError):
        DepthWorker(boom).start()


class _Session:
    def __init__(self, providers):
        self.providers = providers

    def get_providers(self):
        return self.providers


def test_a_session_that_fell_back_to_the_cpu_is_refused():
    with pytest.raises(DepthNotOnGpu, match="CPUExecutionProvider"):
        ensure_on_gpu(_Session(["CPUExecutionProvider"]))
    ensure_on_gpu(_Session(["CUDAExecutionProvider", "CPUExecutionProvider"]))


def test_the_worker_keeps_each_cameras_latest_depth_map_near_warm():
    depth, req = _scene(5.3)                       # person at 5 (near), vehicle 5.3, background 20 (far)
    w = DepthWorker(lambda: _Model(depth))
    w.start()
    try:
        w.submit("cam-1", 10.0, np.zeros((720, 1280, 3), np.uint8), (1.0, 1.0), [req])
        _wait_idle(w)
        t, img = w.latest_map("cam-1")
        assert t == 10.0 and img.shape == (360, 640, 3) and img.dtype == np.uint8
        near, far = img[175, 307].astype(int), img[20, 20].astype(int)          # person (x 600-630, y 300-400) / far
        assert near[2] > far[2] and near[0] < far[0]                             # red-ish near, blue-ish far (BGR)
        assert w.latest_map("cam-2") is None
    finally:
        w.stop()


def test_the_depth_crop_is_the_pairs_area_as_a_small_jpeg():
    import base64
    from perception.modules.proximity.camera import Person, Vehicle
    from perception.modules.proximity.depth import CROP_W, depth_crop
    from perception.modules.proximity.shapes import PersonShape, vehicle_shape_from_box
    m = np.zeros((360, 640, 3), np.uint8)
    m[:, :, 2] = 200
    person = Person(1, "person", 0.9, (600.0, 300.0, 30.0, 100.0), PersonShape((615.0, 400.0), (615.0, 300.0), False))
    truck = Vehicle(2, "truck", (500.0, 250.0, 300.0, 150.0), vehicle_shape_from_box((500.0, 250.0, 300.0, 150.0)))
    jpg = base64.b64decode(depth_crop(m, (1280, 720), person, truck))
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert jpg[:2] == b"\xff\xd8" and img.shape[1] == CROP_W and len(jpg) < 40_000



def test_the_depth_crop_leaves_the_cameras_map_untouched():
    # a pair spanning the frame's width used to make the crop a view, and its outlines were drawn into the shared map
    from perception.modules.proximity.camera import Person, Vehicle
    from perception.modules.proximity.depth import depth_crop
    from perception.modules.proximity.shapes import PersonShape, vehicle_shape_from_box
    m = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    before = m.copy()
    person = Person(1, "person", 0.9, (600.0, 300.0, 30.0, 100.0), PersonShape((615.0, 400.0), (615.0, 300.0), False))
    truck = Vehicle(2, "truck", (0.0, 100.0, 1280.0, 400.0), vehicle_shape_from_box((0.0, 100.0, 1280.0, 400.0)))
    depth_crop(m, (1280, 720), person, truck)
    assert np.array_equal(m, before)


def test_the_worker_is_idle_only_with_nothing_queued_for_any_camera():
    depth, req = _scene(5.3)
    w = DepthWorker(lambda: _Model(depth))                                    # not started: nothing drains it
    assert w.idle()
    w.submit("cam-2", 1.0, np.zeros((720, 1280, 3), np.uint8), (1.0, 1.0), [req])
    assert not w.idle() and not w.busy("cam-1")                              # another camera's frame waits
