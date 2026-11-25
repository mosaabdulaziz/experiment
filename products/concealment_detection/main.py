from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
import threading

import redis
import uvicorn

from concealment.state_machine import ConcealmentMonitor
from config import ServiceConfig, load_config
from consumer import CameraWorker, Consumer, FrameQueue
from server import create_app
from storage import ReviewStore, connect

log = logging.getLogger("concealment_detection.service")


class ServiceRuntime:
    def __init__(self, config: ServiceConfig, redis_client=None):
        self.config = config
        self.redis = redis_client if redis_client is not None else redis.Redis.from_url(
            config.redis_url, socket_connect_timeout=3, socket_timeout=3,
        )
        self.store = ReviewStore(connect(config.db_path))
        self.workers = {
            camera.id: CameraWorker(
                camera.id, ConcealmentMonitor(camera.id, config.detector, config.camera_zones[camera.id]),
                self.redis, self.store, queue=FrameQueue(config.queue_maxsize),
                mux_shape=config.mux_shape, footage_start=config.footage_start,
                clip_duration_s=camera.clip_duration_s,
                location=camera.location,
            ) for camera in config.cameras
        }
        self.consumer = Consumer(self.redis, self.workers)
        self.running = False
        self._stop = threading.Event()
        self._maintenance: threading.Thread | None = None

    def refresh_replay_starts(self):
        if self.config.footage_start is None:
            return
        starts = {}
        try:
            if self.config.stream_starts_path is not None:
                starts = json.loads(self.config.stream_starts_path.read_text(encoding="utf-8"))
                if not isinstance(starts, dict):
                    raise ValueError("Stream starts must map camera IDs to epoch timestamps")
            for camera, worker in self.workers.items():
                value = starts.get(camera)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value)):
                    raise ValueError("Invalid stream start timestamp")
                worker.set_replay_start(value)
        except (OSError, ValueError):
            for worker in self.workers.values():
                worker.set_replay_start(None)
            log.warning("Replay start times unavailable; replay assessment is suspended")

    def _maintain(self):
        ticks = 0
        while not self._stop.wait(1.0):
            try:
                self.refresh_replay_starts()
                ticks += 1
                if ticks >= 60:
                    self.store.prune(keep_candidates=self.config.keep_candidates,
                                     review_days=self.config.review_days)
                    ticks = 0
            except Exception:
                log.exception("Concealment maintenance failed")

    def start(self):
        self.redis.ping()
        self.refresh_replay_starts()
        self.consumer._read_once()
        self.store.prune(keep_candidates=self.config.keep_candidates, review_days=self.config.review_days)
        for worker in self.workers.values():
            worker.start()
        self.consumer.start()
        self._maintenance = threading.Thread(target=self._maintain, name="concealment-maintenance", daemon=True)
        self._maintenance.start()
        self.running = True

    def stop(self):
        self.running = False
        self._stop.set()
        self.consumer.stop()
        self.consumer.join()
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join()
        if self._maintenance is not None:
            self._maintenance.join()
        self.store.close()
        self.redis.close()

    def status(self):
        cameras = {camera: worker.health for camera, worker in self.workers.items()}
        running = self.running and self.consumer.alive and all(w.alive for w in self.workers.values())
        return {
            "running": running,
            "ready": running and all(c["camera_ok"] for c in cameras.values()),
            "cameras": cameras,
            "capabilities": {"cart_detection": False, "bag_context_requires_boxed_detections": True,
                             "durable_clips": False},
        }


def build_app(config: ServiceConfig | None = None, *, redis_client=None):
    runtime = ServiceRuntime(config or load_config(), redis_client=redis_client)

    @asynccontextmanager
    async def lifespan(app):
        try:
            runtime.start()
            yield
        finally:
            runtime.stop()

    app = create_app(runtime.status, runtime.store,
                     cameras=[{"id": c.id, "location": c.location} for c in runtime.config.cameras],
                     lifespan=lifespan)
    app.state.runtime = runtime
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the IRIS suspected-concealment backend")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    uvicorn.run(build_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
