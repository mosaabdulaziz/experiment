"""The depth check: YOLO26m-depth on the frames that have a close pair, in a worker thread, so the streaming thread
never waits for it (RESEARCH 4.7, 5.3 level 3).

It runs through onnxruntime on the GPU, not as a DeepStream GIE: `perception.engine.gie_gate` steers a secondary GIE
per object, and a full-frame model on chosen frames only cannot be gated that way. The model is exported once by
`just proximity-models` (input 736x1280 RGB in 0..1, letterboxed on grey 114; output the depth map at input size --
checked equal to ultralytics' own prediction).

Per pair the check compares the person's depth at their feet with the vehicle's next to them, as measured in the
analysis (S6): |dp - dv| / dp. Small: same depth, the image overlap is a real near case. Large: one is well behind the
other (near 24/25 and far 0/12 of the labelled pairs at 0.2, a thin margin).
"""
from __future__ import annotations

import base64
import logging
import queue
import threading
from typing import Callable, Optional

import cv2
import numpy as np

from .shapes import outline_polygon

logger = logging.getLogger("iris.sidecar")

PAD = 114
FOOT_BAND = 0.15        # the person's depth is the median over the lowest 15 % of their mask
FOOT_X_BAND = 0.05      # the foot's x: mean over the lowest 5 %
MAP_W = 640             # px: the colourised map kept per camera (the preview's depth layer, the card's depth image)
CROP_W = 320            # px: the incident card's depth image
CROP_GROW = 0.2         # the pair's area: the union of the two boxes, grown by this share


def letterbox(bgr: np.ndarray, hw: tuple) -> tuple:
    """(1x3xHxW float32 RGB 0..1 tensor, scale, top, left): `bgr` resized to fit `hw`, centred on grey."""
    H, W = hw
    h, w = bgr.shape[:2]
    s = min(H / h, W / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = bgr if (nh, nw) == (h, w) else cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (H - nh) // 2, (W - nw) // 2
    canvas = np.full((H, W, 3), PAD, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, s, top, left


class DepthNotOnGpu(RuntimeError):
    """The depth session ended up on the CPU: ~30x slower, and it would take every core of the host."""


def ensure_on_gpu(session) -> None:
    """Raise DepthNotOnGpu unless the session runs on CUDA. ONNX Runtime falls back to the CPU on its own when CUDA or
    cuDNN fails to load, and only logs it (as face_recognition's verify_gpu_inference found); check after a warm-up."""
    providers = list(session.get_providers())
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise DepthNotOnGpu(f"after a warm-up inference the session runs on {providers}")


class DepthModel:
    """An exported YOLO26m-depth ONNX through onnxruntime on CUDA."""

    def __init__(self, path: str, providers=("CUDAExecutionProvider", "CPUExecutionProvider")) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=list(providers))
        inp = self.session.get_inputs()[0]
        self.input, self.hw = inp.name, (int(inp.shape[2]), int(inp.shape[3]))

    def __call__(self, bgr: np.ndarray) -> tuple:
        """(depth map over the resized frame, the frame-to-map scale)."""
        x, s, top, left = letterbox(bgr, self.hw)
        d = self.session.run(None, {self.input: x})[0].reshape(self.hw)
        h, w = bgr.shape[:2]
        return d[top:top + int(round(h * s)), left:left + int(round(w * s))], s


def region(blob, box, map_hw: tuple, kx: float, ky: float) -> np.ndarray:
    """A bool mask over the depth map: the object's mask blob, or its box when it has none. kx, ky: mux px -> map px."""
    H, W = map_hw
    out = np.zeros((H, W), bool)
    if blob is not None:
        rows, cols = blob.grid.shape
        x0, y0, w, h = blob.left * kx, blob.top * ky, cols * blob.cell_w * kx, rows * blob.cell_h * ky
        patch = blob.grid
    else:
        left, top, bw, bh = box
        x0, y0, w, h = left * kx, top * ky, bw * kx, bh * ky
        patch = np.ones((1, 1), bool)
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    iw, ih = max(1, int(round(w))), max(1, int(round(h)))
    patch = cv2.resize(patch.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
    xa, ya, xb, yb = max(0, ix0), max(0, iy0), min(W, ix0 + iw), min(H, iy0 + ih)
    if xa < xb and ya < yb:
        out[ya:yb, xa:xb] = patch[ya - iy0:yb - iy0, xa - ix0:xb - ix0]
    return out


def depth_gap(depth: np.ndarray, k: tuple, req) -> Optional[float]:
    """|dp - dv| / dp for one DepthRequest: dp = median depth over the lowest 15 % of the person's mask, dv = median over
    the vehicle's mask within one (visible) person height of the person's lowest point, or over all of it when none is
    that close. k = (kx, ky): mux px -> map px. None when a region falls outside the map."""
    kx, ky = k
    pm = region(req.person_blob, req.person_box, depth.shape, kx, ky)
    vm = region(req.vehicle_blob, req.vehicle_box, depth.shape, kx, ky)
    ys, xs = np.nonzero(pm)
    if len(ys) < 5 or not vm.any():
        return None
    top, bottom = ys.min(), ys.max()
    height = bottom - top + 1
    fx = xs[ys >= bottom - FOOT_X_BAND * height].mean()
    rows, cols = np.ogrid[0:depth.shape[0], 0:depth.shape[1]]
    foot = pm & (rows >= bottom - FOOT_BAND * height)
    near = vm & ((rows - bottom) ** 2 + (cols - fx) ** 2 <= height ** 2)
    if not near.any():
        near = vm
    dp, dv = float(np.median(depth[foot])), float(np.median(depth[near]))
    return abs(dp - dv) / max(dp, 1e-3)


def colourise(depth: np.ndarray) -> np.ndarray:
    """A depth map as a BGR image MAP_W wide: TURBO over the frame's own range, near warm and far cool."""
    lo, hi = float(np.min(depth)), float(np.max(depth))
    norm = np.zeros(depth.shape, np.uint8) if hi - lo < 1e-6 else ((hi - depth) * (255.0 / (hi - lo))).astype(np.uint8)
    h = max(1, int(round(depth.shape[0] * MAP_W / depth.shape[1])))
    return cv2.applyColorMap(cv2.resize(norm, (MAP_W, h), interpolation=cv2.INTER_AREA), cv2.COLORMAP_TURBO)


def depth_crop(depth_map: np.ndarray, mux_size: tuple, person, vehicle) -> Optional[str]:
    """The pair's area of a colourised depth map (over the whole frame) with the vehicle's outline in white and the
    person's in black, CROP_W wide, as a base64 JPEG; None when the area is empty."""
    mw, mh = mux_size
    kx, ky = depth_map.shape[1] / mw, depth_map.shape[0] / mh
    boxes = (person.box, vehicle.box)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    gx, gy = (x1 - x0) * CROP_GROW / 2, (y1 - y0) * CROP_GROW / 2
    ix0, iy0 = int(max(0.0, x0 - gx) * kx), int(max(0.0, y0 - gy) * ky)
    ix1, iy1 = int(np.ceil(min(mw, x1 + gx) * kx)), int(np.ceil(min(mh, y1 + gy) * ky))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    crop = depth_map[iy0:iy1, ix0:ix1].copy()            # the camera's map is shared: never draw into it
    for obj, color in ((vehicle, (255, 255, 255)), (person, (0, 0, 0))):
        if obj.shape.blob is not None:
            pts = outline_polygon(obj.shape.blob)
        else:
            left, top, w, h = obj.box
            pts = [[left, top], [left + w, top], [left + w, top + h], [left, top + h]]
        if pts:
            poly = np.round(np.array(pts, np.float32) * (kx, ky) - (ix0, iy0)).astype(np.int32)
            cv2.polylines(crop, [poly], True, color, 1, cv2.LINE_AA)
    out = cv2.resize(crop, (CROP_W, max(1, int(round(crop.shape[0] * CROP_W / crop.shape[1])))))
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None


class DepthWorker:
    """One thread, one model. `submit` hands it a frame and the pairs to measure and never blocks: with `max_pending`
    frames waiting the new one is dropped, and the next candidate frame asks again. `latest` reads the latest gap
    measured for a pair, with the time it was measured."""

    def __init__(self, model_factory: Callable[[], Callable], max_pending: int = 2, keep_s: float = 10.0) -> None:
        self._factory, self._keep_s = model_factory, keep_s
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._results: dict = {}        # (stream id, pair key) -> (t, gap)
        self._busy: dict = {}           # stream id -> frames queued or being measured
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self.dropped = 0
        self.failed = 0
        self._maps: dict = {}           # stream id -> (t, colourised map)

    def start(self) -> None:
        """Load the model (a failure here refuses startup) and start the thread."""
        self._model = self._factory()
        self._thread = threading.Thread(target=self._run, name="proximity-depth", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put(None, timeout=5)
        except queue.Full:
            return
        self._thread.join(timeout=5)

    def idle(self) -> bool:
        """No frame is queued or being measured, for any camera."""
        with self._lock:
            return not any(self._busy.values())

    def busy(self, stream_id: str) -> bool:
        with self._lock:
            return self._busy.get(stream_id, 0) > 0

    def submit(self, stream_id: str, t: float, bgr: np.ndarray, mux_to_frame: tuple, requests: list) -> bool:
        with self._lock:
            self._busy[stream_id] = self._busy.get(stream_id, 0) + 1
        try:
            self._queue.put_nowait((stream_id, t, bgr, mux_to_frame, requests))
            return True
        except queue.Full:
            with self._lock:
                self._busy[stream_id] -= 1
            self.dropped += 1
            return False

    def latest(self, stream_id: str, key: tuple) -> Optional[tuple]:
        """(the frame time it was measured on, the relative depth gap) of the pair's latest measurement, or None."""
        with self._lock:
            return self._results.get((stream_id, key))

    def latest_map(self, stream_id: str) -> Optional[tuple]:
        """(the frame time, the colourised map over the whole frame) of the camera's latest depth frame, or None."""
        with self._lock:
            return self._maps.get(stream_id)

    def measure(self, stream_id: str, t: float, bgr: np.ndarray, mux_to_frame: tuple, requests: list) -> None:
        """Run the model on one frame and keep each pair's gap (the thread's work)."""
        depth, s = self._model(bgr)
        k = (mux_to_frame[0] * s, mux_to_frame[1] * s)
        gaps = {r.key: depth_gap(depth, k, r) for r in requests}
        image = colourise(depth)
        with self._lock:
            self._maps[stream_id] = (t, image)
            for key, gap in gaps.items():
                if gap is not None:
                    self._results[(stream_id, key)] = (t, gap)
            for old in [rk for rk, (rt, _g) in self._results.items() if rk[0] == stream_id and t - rt > self._keep_s]:
                del self._results[old]

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self.measure(*job)
            except Exception as exc:  # noqa: BLE001 -- one bad frame must not end the checks
                self.failed += 1
                logger.warning("proximity depth check failed (%d so far): %s", self.failed, exc)
            finally:
                with self._lock:
                    self._busy[job[0]] -= 1
