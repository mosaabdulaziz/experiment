import asyncio
import logging
import os
import threading
import time
from collections import deque
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

import cv2
from PIL import Image


class Camera:
    def __init__(self, config: dict):
        self._config = config
        cam_cfg = config.get("camera", {})
        self._source = cam_cfg.get("source", 0)
        self._width = cam_cfg.get("width", 1280)
        self._height = cam_cfg.get("height", 720)

        sampling = config.get("sampling", {})
        self._quality = sampling.get("jpeg_quality", 82)
        interval = max(float(sampling.get("interval_seconds", 6)), 0.1)
        batch_size = max(int(sampling.get("frames_per_batch", 4)), 1)
        overlap = min(max(int(sampling.get("overlap_frames", 1)), 0), batch_size - 1)
        self._overlap = overlap
        self._batch_size = batch_size
        # New frames per cycle = batch_size - overlap; the rest come from the previous batch.
        new_frames_per_cycle = batch_size - overlap
        self._buffer_interval = interval / new_frames_per_cycle

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._annotated_frame: bytes | None = None
        self._current_offset: float = 0.0
        self._new_frames_per_cycle = new_frames_per_cycle
        self._last_buffer_push: float = 0.0
        self._prev_tail: list[tuple[bytes, float]] = []   # frame-path overlap, prepended to the next grabbed batch
        self._annotated_prev_tail: list[tuple[bytes, list, float]] = []  # same, annotated path (triples)
        # Batch queue: frames accumulate into complete batches while the VLM works,
        # and each analysis consumes the OLDEST queued batch — so footage that plays
        # during a slow cycle is analyzed next instead of being skipped. deque(maxlen)
        # drops the oldest batch on overflow (prefer recent footage over stale).
        self._batch_queue_depth = max(int(sampling.get("batch_queue_depth", 3)), 1)
        self._frame_pending: list[tuple[bytes, float]] = []
        self._frame_queue: deque = deque(maxlen=self._batch_queue_depth)
        self._annotated_last_push: float = 0.0
        self._annotated_pending: list[tuple[bytes, list, float]] = []
        self._annotated_queue: deque = deque(maxlen=self._batch_queue_depth)
        self._stream_start_time: float = 0.0
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._thread: threading.Thread | None = None
        self._is_file = False  # True when streaming from a video file

        pre = config.get("preprocessing", {})
        self._max_width = int(pre.get("max_width", 0))
        if pre.get("clahe", False):
            clip = float(pre.get("clahe_clip_limit", 2.0))
            tile = int(pre.get("clahe_tile_size", 8))
            self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        else:
            self._clahe = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def current_offset(self) -> float:
        with self._lock:
            return self._current_offset

    @property
    def source_label(self) -> str:
        if self._is_file:
            return f"Video: {os.path.basename(str(self._source))}"
        return f"Camera: {self._source}"

    def reset_to_default(self):
        """Reset source back to the configured default (webcam)."""
        cam_cfg = self._config.get("camera", {})
        self._source = cam_cfg.get("source", 0)

    async def start(self):
        if self._running:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._start_capture)

    def _start_capture(self):
        source = self._source
        # Numeric sources stay as int; string sources (RTSP, HTTP) stay as str
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        # Detect whether the source is a local video file
        self._is_file = isinstance(source, str) and os.path.isfile(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source!r}")

        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # For video files, read the native FPS to pace playback
        if self._is_file:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_delay = 1.0 / fps if fps and fps > 0 else 0.033
        else:
            self._frame_delay = 0.033  # ~30 fps

        self._stream_start_time = time.monotonic()
        self._running = True

        if self._is_file:
            first = []
            for _ in range(self._batch_size):
                ret, frame = self._cap.read()
                if not ret:
                    break
                offset = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                first.append((self._encode_jpeg(frame), offset))
            if first:
                self._frame_queue.append(first)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def pause(self):
        if not self._paused:
            self._paused = True
            self._pause_event.clear()
            logger.info("Video playback paused")

    def resume(self):
        if self._paused:
            self._paused = False
            self._pause_event.set()
            logger.info("Video playback resumed")

    def _capture_loop(self):
        while self._running:
            self._pause_event.wait()  # block here while paused
            if not self._running:
                break
            t0 = time.monotonic()
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    # Loop video files back to the beginning
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            encoded = self._encode_jpeg(frame)
            if self._is_file:
                stream_offset = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                stream_offset = time.monotonic() - self._stream_start_time
            now = time.monotonic()
            with self._lock:
                self._latest_frame = encoded
                self._current_offset = stream_offset
                if now - self._last_buffer_push >= self._buffer_interval:
                    self._accept_frame(encoded, stream_offset)
                    self._last_buffer_push = now
            # Sleep only the remaining time so processing overhead doesn't slow playback
            remaining = self._frame_delay - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _accept_frame(self, encoded: bytes, stream_offset: float) -> None:
        """Buffer one sampled frame; complete batches join the analysis queue.

        Called from the reader loop with the lock held. A full queue drops its
        oldest batch: when the VLM is slower than playback, analyzing stale
        footage late is worse than analyzing the newest footage now.
        """
        self._frame_pending.append((encoded, stream_offset))
        if len(self._frame_pending) < self._new_frames_per_cycle:
            return
        if len(self._frame_queue) == self._frame_queue.maxlen:
            logger.info("batch queue full — dropping oldest batch (%.0fs of footage "
                        "analyzed late or never)",
                        self._buffer_interval * self._new_frames_per_cycle)
        self._frame_queue.append(self._frame_pending)
        self._frame_pending = []

    def _encode_jpeg(self, frame) -> bytes:
        if self._max_width > 0:
            h, w = frame.shape[:2]
            if w > self._max_width:
                scale = self._max_width / w
                frame = cv2.resize(
                    frame, (self._max_width, int(h * scale)), interpolation=cv2.INTER_AREA
                )
        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return buf.tobytes()

    def grab_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def set_annotated_frame(self, frame: bytes | None) -> None:
        with self._lock:
            self._annotated_frame = frame

    def grab_display_frame(self) -> bytes | None:
        """Return annotated frame for stream display, falling back to raw if not available."""
        with self._lock:
            return self._annotated_frame if self._annotated_frame is not None else self._latest_frame

    def grab_frame_batch(self) -> tuple[list[bytes], list[float]]:
        """Return (frames, offsets) for the next analysis batch.

        Consumes the OLDEST queued batch (frames queued while previous analyses
        ran), prepending the overlap tail from the previous batch. Falls back to
        incomplete pending frames, then to the live frame — a cycle never fires
        on nothing.
        """
        with self._lock:
            if self._frame_queue:
                batch = self._frame_queue.popleft()
                combined = self._prev_tail + batch
                self._prev_tail = batch[-self._overlap:] if self._overlap else []
                return [f for f, _ in combined], [o for _, o in combined]
            pending = self._frame_pending
            self._frame_pending = []
            fallback = self._latest_frame

        if pending:
            combined = self._prev_tail + pending
            self._prev_tail = pending[-self._overlap:] if self._overlap else []
            return [f for f, _ in combined], [o for _, o in combined]
        return ([fallback], [0.0]) if fallback else ([], [])

    def push_annotated_frame(self, annotated: bytes, detections: list, offset: float) -> None:
        """Called by stream annotation loop; samples annotated frames at buffer_interval cadence."""
        now = time.monotonic()
        with self._lock:
            if now - self._annotated_last_push >= self._buffer_interval:
                self._accept_annotated(annotated, detections, offset)
                self._annotated_last_push = now

    def _accept_annotated(self, annotated: bytes, detections: list, offset: float) -> None:
        """Buffer one annotated frame; complete batches join the analysis queue.

        Called with the lock held; mirrors _accept_frame for the annotated path.
        """
        self._annotated_pending.append((annotated, detections, offset))
        if len(self._annotated_pending) < self._new_frames_per_cycle:
            return
        if len(self._annotated_queue) == self._annotated_queue.maxlen:
            logger.info("annotated batch queue full — dropping oldest batch")
        self._annotated_queue.append(self._annotated_pending)
        self._annotated_pending = []

    def grab_annotated_batch(self) -> tuple[list[bytes], list[list], list[float]] | None:
        """Return (frames, detections_list, offsets) from the oldest queued batch, or None.

        Mirrors grab_frame_batch's queue discipline so the scheduler consumes
        analysis batches in order; None only when nothing has accumulated yet.
        """
        with self._lock:
            if self._annotated_queue:
                batch = self._annotated_queue.popleft()
                combined = self._annotated_prev_tail + batch
                self._annotated_prev_tail = batch[-self._overlap:] if self._overlap else []
                return (
                    [f for f, _, _ in combined],
                    [d for _, d, _ in combined],
                    [o for _, _, o in combined],
                )
            pending = self._annotated_pending
            if not pending:
                return None
            self._annotated_pending = []
            combined = self._annotated_prev_tail + pending
            self._annotated_prev_tail = pending[-self._overlap:] if self._overlap else []
            return (
                [f for f, _, _ in combined],
                [d for _, d, _ in combined],
                [o for _, _, o in combined],
            )

    async def generate_mjpeg(self) -> AsyncGenerator[bytes, None]:
        """Yields MJPEG boundary frames for browser streaming."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = self.grab_display_frame()
            if frame:
                yield boundary + frame + b"\r\n"
            await asyncio.sleep(0.04)  # ~25 fps to browser

    def set_source(self, source):
        """Change the camera source (device index, URL, or file path)."""
        self._source = source

    @staticmethod
    def list_cameras(max_index: int = 8) -> list[dict]:
        """Probe camera indices 0‥max_index-1 and return the ones that open.

        On Linux, restricts the probe to indices with a matching /dev/videoN
        file so we don't fire OpenCV errors for nonexistent devices. Also
        silences OpenCV's logger during the probe — failed opens are
        expected and handled by checking `cap.isOpened()`."""
        if os.name == "posix":
            candidates = [i for i in range(max_index) if os.path.exists(f"/dev/video{i}")]
        else:
            candidates = list(range(max_index))

        prior_level = None
        try:
            prior_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except AttributeError:
            pass

        available = []
        try:
            for i in candidates:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append({"index": i, "label": f"Camera {i}"})
                    cap.release()
        finally:
            if prior_level is not None:
                cv2.utils.logging.setLogLevel(prior_level)
        return available

    async def stop(self):
        self._pause_event.set()  # unblock capture loop if paused so it can exit
        self._paused = False
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_frame = None
            self._annotated_frame = None
            self._frame_pending = []
            self._frame_queue.clear()
            self._last_buffer_push = 0.0
            self._annotated_pending = []
            self._annotated_queue.clear()
            self._annotated_last_push = 0.0
        self._prev_tail = []
        self._annotated_prev_tail = []
