"""The service's entry point: the gate, then the consumer, the workers, the API.

Nothing runs before the gate passes. The classifier is built first (design §8), and only then
do the Redis consumer, the per-camera workers and the HTTP API start -- so a deployment
without a GPU, or with a corrupt checkpoint, exits before it can raise an alert it cannot
stand behind.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import redis
import uvicorn

import acl
import boot
import config
import consumer
import monitor
import server
import storage

logger = logging.getLogger("fall_detection")

STOP_TIMEOUT_S = 5.0

# The retention sweep: the history table keeps its newest rows and terminal review rows age
# out; 'candidate' rows are the operator's work list and are never swept.
SWEEP_PERIOD_S = 6 * 3600
SWEEP_KEEP_EVENTS = 5000
SWEEP_REVIEW_DAYS = 30


def _retention_sweep(queue: storage.ReviewQueue, stop: threading.Event) -> None:
    """Sweep once at boot, then every SWEEP_PERIOD_S until the service stops.

    Best effort: a failed sweep cannot take the service down, so it logs and waits for the
    next period.
    """
    while True:
        try:
            dropped = queue.prune(keep_events=SWEEP_KEEP_EVENTS, review_days=SWEEP_REVIEW_DAYS)
            if any(dropped):
                logger.info("the retention sweep dropped %d events and %d reviews", *dropped)
        except Exception:
            logger.exception("the retention sweep failed")
        if stop.wait(SWEEP_PERIOD_S):
            return


def _start_sweeper(queue: storage.ReviewQueue) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    thread = threading.Thread(target=_retention_sweep, args=(queue, stop),
                              name="fall-retention-sweep", daemon=True)
    thread.start()
    return thread, stop


@dataclass
class Service:
    """What the running service is made of, and how it stops."""

    app: object
    reader: consumer.Consumer
    workers: dict[str, consumer.CameraWorker]
    client: redis.Redis
    queue: storage.ReviewQueue
    sweeper: threading.Thread | None = None
    sweeper_stop: threading.Event | None = None

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        """Stop the reader and the workers, wait for them, then close what they held.

        A thread that outlived the wait keeps its handles: closing the queue's connection or the
        Redis client under a live thread turns a slow shutdown into an exception in that thread.
        """
        self.reader.stop()
        for worker in self.workers.values():
            worker.stop()
        self.reader.join(timeout)
        for worker in self.workers.values():
            worker.join(timeout)
        if self.sweeper_stop is not None:
            self.sweeper_stop.set()
        if self.sweeper is not None:
            self.sweeper.join(timeout)
        if self.reader.alive or any(worker.alive for worker in self.workers.values()):
            logger.warning("a thread of the fall service did not stop within %.1f s; "
                           "its handles stay open", timeout)
            return
        self.queue.close()
        self.client.close()


def _connect_redis(cfg: config.ServiceConfig) -> redis.Redis:
    """The platform's bus. Checked now, because a service that cannot read frames or write
    alerts would otherwise run silently and log an exception every poll.

    Decoding stays off: a meta entry carries the frame's `jpeg` as raw bytes beside the text
    fields (perception/engine/publisher.py:350), and a strict UTF-8 decode raises on the first
    real frame. The ACL turns every field of the entry into text, errors replaced (acl.read).

    The socket timeouts are what keep a half-open connection from parking the reader inside a
    read (and a worker inside a write) forever, with `/status` still saying the service is up.
    """
    client = redis.Redis.from_url(cfg.redis_url, decode_responses=False,
                                  socket_timeout=5.0, socket_connect_timeout=5.0,
                                  health_check_interval=30)
    try:
        client.ping()
    except redis.RedisError as exc:
        client.close()
        raise SystemExit(
            f"fall detection: refusing to start -- Redis at {cfg.redis_url} is unreachable: {exc}",
        ) from None
    return client


def build_service(cfg: config.ServiceConfig) -> Service:
    """Everything the service runs, in the design §8 order."""
    classifier, gate_state = boot.boot(cfg.classifier, cfg.model_manifest)
    vendor_cfg = monitor.vendor_blocks()
    policy = {"scoring": cfg.scoring, "suppress_after_fall_s": cfg.suppress_after_fall_s,
              "alert_details": cfg.alert_details}

    client = _connect_redis(cfg)
    try:
        queue = storage.ReviewQueue(storage.connect(cfg.db_path))
    except Exception:
        client.close()
        raise
    try:
        workers = consumer.build_workers(
            [c.id for c in cfg.cameras],
            monitor_factory=lambda _camera: monitor.build_monitor(classifier,
                                                                 device=cfg.classifier["device"]),
            redis=client, storage=queue, policy=policy, vendor_cfg=vendor_cfg,
            key_ids={c.id: c.key_id for c in cfg.cameras},
            zones=acl.ZonesFile(cfg.zones_path, "fall") if cfg.zones_path else None, zones_mux=cfg.zones_mux)
        for worker in workers.values():
            worker.start()

        reader = consumer.Consumer(client, workers)
        reader.start()
        app = server.create_app(status=boot.status(gate_state), storage=queue,
                                cameras=[c.id for c in cfg.cameras])
        sweeper, sweeper_stop = _start_sweeper(queue)
    except Exception:
        queue.close()
        client.close()
        raise
    return Service(app=app, reader=reader, workers=workers, client=client, queue=queue,
                   sweeper=sweeper, sweeper_stop=sweeper_stop)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = config.load_config()
    service = build_service(cfg)
    logger.info("the fall product is running on %s", ", ".join(service.workers))
    try:
        uvicorn.run(service.app, host=cfg.host, port=cfg.port)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
