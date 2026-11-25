"""Per-camera bounded frame queue, worker, and Redis stream consumer driving `ConcealmentMonitor`.

Provenance: adapted from `products/fall_detection/consumer.py` (`FrameQueue`, `CameraWorker`,
`Consumer`). Owned independently by `products/concealment_detection`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import queue
import threading
import time
import redis

import acl
from concealment.state_machine import ConcealmentMonitor
from evidence import build_candidate_record
from footage_clock import FootageClock, footage_clock
from storage import ReviewStore

log = logging.getLogger("concealment_detection.consumer")

QUEUE_MAXSIZE = 128
WARN_INTERVAL_S = 60.0


class FrameQueue:
    """Bounded queue that drops the oldest entry when full."""

    def __init__(self, maxsize: int = QUEUE_MAXSIZE) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0:
            raise ValueError("Queue capacity must be a positive integer")
        self.maxsize = maxsize
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, entry) -> None:
        while True:
            try:
                self._q.put_nowait(entry)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout: float | None = None):
        return self._q.get(timeout=timeout)

    def get_nowait(self):
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()


class CameraWorker:
    """Processes one camera's `iris:meta:<cam>` stream entries through `ConcealmentMonitor`."""

    def __init__(
        self,
        camera: str,
        monitor: ConcealmentMonitor,
        redis_client,
        store: ReviewStore,
        clock: FootageClock | None = None,
        queue: FrameQueue | None = None,
        *,
        mux_shape: tuple[int, int] = (1920, 1080),
        footage_start: datetime | None = None,
        clip_duration_s: float | None = None,
        location: str | None = None,
    ) -> None:
        self.camera = camera
        self.location = location or camera
        self.monitor = monitor
        self.redis = redis_client
        self.store = store
        self.clock = clock
        self.queue = queue if queue is not None else FrameQueue()
        self._warned: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._lock = threading.RLock()
        self.mux_shape = mux_shape
        self.footage_start = footage_start
        self.clip_duration_s = clip_duration_s
        self._replay_start: float | None = None
        self._replay_lap: int | None = None
        self._last_received = time.monotonic()
        self._last_pts = 0.0
        self._health = {"camera_ok": False, "reason": "awaiting_frames"}

    @property
    def health(self) -> dict:
        with self._lock:
            return dict(self._health)

    def _assess_camera(self, reason: str) -> list[dict]:
        self.monitor.invalidate(self._last_pts)
        if self._health["reason"] == reason:
            return []
        self._health = {"camera_ok": False, "reason": reason}
        row = {"camera": self.camera, "track_id": -1, "pts": self._last_pts,
               "state": "insufficient_evidence", "reason": reason}
        row["id"] = self.store.add_assessment(row)
        return [row]

    def check_health(self, now: float | None = None) -> dict:
        with self._lock:
            now = time.monotonic() if now is None else now
            if now - self._last_received > self.monitor.settings.camera_stale_s:
                self._assess_camera("stream_stale")
            return dict(self._health)

    def set_replay_start(self, start: float | None) -> None:
        with self._lock:
            if start != self._replay_start:
                self.monitor.invalidate(self._last_pts)
                self._replay_lap = None
            self._replay_start = start
            self.clock = None

    def _warn(self, kind: str, message: str, *args) -> None:
        """Rate-limited warning for repeated stream malformations (at most once per `WARN_INTERVAL_S`)."""
        now = time.monotonic()
        if now - self._warned.get(kind, -WARN_INTERVAL_S) >= WARN_INTERVAL_S:
            self._warned[kind] = now
            log.warning(message, *args)

    def submit(self, entry: dict) -> None:
        self.queue.put(entry)

    def process_entry(self, entry: dict) -> dict:
        with self._lock:
            return self._process_entry(entry)

    def _process_entry(self, entry: dict) -> dict:
        self._last_received = time.monotonic()
        try:
            frame = acl.parse_pose_frame(self.camera, entry, mux_shape=self.mux_shape)
            raw_id = acl.text(entry.get("id", ""))
            device_dt = (datetime.fromtimestamp(int(raw_id.split("-")[0]) / 1000, timezone.utc)
                         if raw_id else datetime.now(timezone.utc))
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            self._warn("parse", "malformed iris:meta entry on %s (%s) -- dropped", self.camera, exc)
            assessments = self._assess_camera("malformed_pose")
            return {"camera": self.camera, "pts": self._last_pts, "camera_ok": False,
                    "candidates": [], "assessments": assessments}
        if self.footage_start is not None:
            if self._replay_start is None or self.clip_duration_s is None:
                assessments = self._assess_camera("replay_clock_unavailable")
                return {"camera": self.camera, "pts": frame.pts, "camera_ok": False,
                        "candidates": [], "assessments": assessments}
            lap = math.floor((device_dt.timestamp() - self._replay_start) / self.clip_duration_s)
            if self._replay_lap is not None and lap != self._replay_lap:
                self.monitor.invalidate(frame.pts)
            self._replay_lap = lap
            self.clock = footage_clock(self.footage_start,
                                       {self.camera: [self._replay_start, self.clip_duration_s, lap]})
        self._last_pts = frame.pts
        step_res = self.monitor.update(frame)
        previous_reason = self._health["reason"]
        camera_reason = next((a.reason for a in step_res.assessments if a.track_id == -1), "")
        self._health = {"camera_ok": step_res.camera_ok, "reason": camera_reason}
        created_candidates: list[dict] = []
        created_assessments: list[dict] = []
        for ass in step_res.assessments:
            if ass.track_id == -1 and ass.reason == previous_reason:
                continue
            ass_row = {"camera": ass.camera, "track_id": ass.track_id, "pts": ass.pts,
                       "state": ass.state, "reason": ass.reason}
            ass_row["id"] = self.store.add_assessment(ass_row)
            created_assessments.append(ass_row)
        for cand in step_res.candidates:
            rec = build_candidate_record(cand, clock=self.clock, device_dt=device_dt, location=self.location)
            cid = self.store.add_candidate(rec)
            rec["id"] = cid
            if self.redis is not None:
                acl.write_alert(self.redis, self.camera, severity=rec["severity"],
                                message=rec["description"], pts=cand.pts, kind=rec["kind"],
                                category=rec["category"], zone=cand.zone_id)
            created_candidates.append(rec)
        return {"camera": self.camera, "pts": frame.pts, "camera_ok": step_res.camera_ok,
                "candidates": created_candidates, "assessments": created_assessments}

    def drain(self) -> list[dict]:
        """Process all currently queued entries, oldest first."""
        results: list[dict] = []
        while True:
            try:
                entry = self.queue.get_nowait()
            except queue.Empty:
                return results
            results.append(self.process_entry(entry))

    def run(self) -> None:
        """Drain the queue continuously until stopped."""
        while not self._stopped:
            try:
                entry = self.queue.get(timeout=0.2)
            except queue.Empty:
                self.check_health()
                continue
            try:
                self.process_entry(entry)
                self.check_health()
            except Exception:
                log.exception("frame of %s could not be handled", self.camera)
                with self._lock:
                    self._assess_camera("processing_error")

    def start(self) -> None:
        """Run the camera worker on its own daemon thread."""
        self._thread = threading.Thread(
            target=self.run,
            name=f"concealment-worker-{self.camera}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        if self._thread is not None:
            return self._thread.is_alive()
        return not self._stopped


class Consumer:
    """Redis reader thread polling `iris:meta:<cam>` and dispatching entries to per-camera workers."""

    def __init__(
        self,
        redis_client: redis.Redis,
        workers: dict[str, CameraWorker],
        batch: int = 64,
        interval: float = 0.04,
        retry_interval: float = 1.0,
    ) -> None:
        self.workers = workers
        self.batch = batch
        self.interval = interval
        self.retry_interval = retry_interval
        self._reader = acl.MetaStreamReader(redis_client)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failing = False

    def _read_once(self) -> None:
        for camera, worker in self.workers.items():
            for entry in self._reader.read(camera, self.batch):
                worker.submit(entry)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_once()
            except redis.RedisError as exc:
                if not self._failing:
                    self._failing = True
                    log.warning(
                        "meta streams unreachable -- retrying every %.1f s: %s",
                        self.retry_interval,
                        exc,
                    )
                self._stop.wait(self.retry_interval)
                continue
            except Exception:
                log.exception("meta entry could not be queued")
            else:
                if self._failing:
                    self._failing = False
                    log.info("meta streams readable again")
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="concealment-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
