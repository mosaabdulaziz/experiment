# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/broker.py. Copied unchanged below this header: fix it there first, then copy it again.
"""RabbitMQ client for the IRIS contract (guide §3, §4, §8), using pika.

One thread owns the pika BlockingConnection (pika is not thread-safe). It
  - consumes the inbound queue with manual acks and prefetch 20, acking after the handler ran;
  - publishes to iris.ai with publisher confirms, user_id, persistent JSON;
  - on every (re)connect publishes catalog.snapshot (cloud only) then state.sync.request;
  - reconnects with exponential backoff (1, 2, 4 ... max 30 s).

Other threads (camera workers, later) call `publish()`, which only appends to an outbox.
The broker thread drains the outbox between I/O rounds. A message is removed from the outbox
only after the broker confirmed it, so after a reconnect it is re-sent unchanged (same eventId,
which IRIS de-duplicates).
"""
import collections
import json
import threading
import time
import uuid

import pika
import pika.exceptions

from .log import log

AI_EXCHANGE = "iris.ai"


def message_properties(settings, message_id=None, now=None):
    """AMQP properties every message to IRIS must carry (guide §3.3)."""
    return pika.BasicProperties(
        content_type="application/json",
        content_encoding="utf-8",
        delivery_mode=pika.DeliveryMode.Persistent,
        user_id=settings.amqp_user,
        app_id=settings.app_id,
        message_id=message_id or str(uuid.uuid4()),
        timestamp=int(now if now is not None else time.time()),
    )


def encode(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Broker:
    def __init__(self, settings, on_message, on_connected=None, on_tick=None):
        """on_message(routing_key, body_dict) is called on the broker thread; it publishes via publish().
        on_connected() returns a list of (routing_key, body) to publish first after every connect."""
        self.settings = settings
        self.on_message = on_message
        self.on_connected = on_connected or (lambda: [])
        self.on_tick = on_tick or (lambda: None)  # broker thread, every loop (~0.2 s)
        self._outbox = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._conn = self._ch = None
        self.connected = threading.Event()

    # ---------- called from any thread ----------
    def publish(self, routing_key, body):
        with self._lock:
            self._outbox.append((routing_key, body, str(uuid.uuid4())))

    def stop(self):
        self._stop.set()

    def pending(self):
        with self._lock:
            return len(self._outbox)

    # ---------- broker thread ----------
    def run(self):
        delay = 1.0
        while not self._stop.is_set():
            try:
                self._connect()
                delay = 1.0
                self._loop()
            except (pika.exceptions.AMQPError, OSError) as e:
                log("broker connection lost", error=f"{type(e).__name__}: {e}", retry_in_s=delay,
                    unsent=self.pending())
            finally:
                self.connected.clear()
                self._close()
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, self.settings.reconnect_max_s)
        log("broker stopped", unsent=self.pending())

    def _connect(self):
        s = self.settings
        params = pika.URLParameters(s.amqp_url)
        params.heartbeat = 30
        params.blocked_connection_timeout = 60
        params.client_properties = {"connection_name": f"{s.app_id} ({s.target})"}
        log("broker connecting", url=s.safe_url, queue=s.inbound_queue, target=s.target)
        self._conn = pika.BlockingConnection(params)
        self._ch = self._conn.channel()
        self._ch.confirm_delivery()
        self._ch.basic_qos(prefetch_count=s.prefetch)
        # IRIS declares everything; our user has no configure permission -> passive check only.
        self._ch.queue_declare(s.inbound_queue, passive=True)
        self._ch.basic_consume(s.inbound_queue, on_message_callback=self._deliver, auto_ack=False)
        self.connected.set()
        log("broker connected", queue=s.inbound_queue)
        # Startup sequence (guide §4): catalog (cloud only), then state.sync.request. Ahead of the outbox.
        first = [(rk, body, str(uuid.uuid4())) for rk, body in self.on_connected()]
        with self._lock:
            self._outbox.extendleft(reversed(first))

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.on_tick()
            except Exception as e:
                log("tick error", error=f"{type(e).__name__}: {e}")
            self._drain()
            self._conn.process_data_events(time_limit=0.2)
        self._drain()

    def _drain(self):
        while True:
            with self._lock:
                if not self._outbox:
                    return
                rk, body, mid = self._outbox[0]
            try:
                self._ch.basic_publish(AI_EXCHANGE, rk, encode(body),
                                       properties=message_properties(self.settings, mid), mandatory=True)
            except pika.exceptions.UnroutableError:
                log("publish unroutable, dropped", routing_key=rk, message_id=mid)
            except pika.exceptions.NackError:
                log("publish nacked by broker, will retry", routing_key=rk, message_id=mid)
                raise pika.exceptions.AMQPError("publish nacked")
            else:
                log("published", routing_key=rk, message_id=mid, **_summary(body))
            with self._lock:
                self._outbox.popleft()

    def _deliver(self, ch, method, props, raw):
        rk = method.routing_key
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            log("inbound message is not JSON, dropped", routing_key=rk, error=str(e))
            ch.basic_ack(method.delivery_tag)
            return
        try:
            self.on_message(rk, body)
        except Exception as e:  # never let one bad message kill the consumer; IRIS re-sends upserts
            log("handler error", routing_key=rk, error=f"{type(e).__name__}: {e}")
        ch.basic_ack(method.delivery_tag)  # ack after applying (guide §3.4)

    def _close(self):
        try:
            if self._conn is not None and self._conn.is_open:
                self._conn.close()
        except Exception:
            pass
        self._conn = self._ch = None


def _summary(body):
    """A few safe fields for the publish log line (never credentials)."""
    keys = ("pipelineId", "cameraId", "status", "detail", "state", "eventId", "target")
    out = {k: body[k] for k in keys if isinstance(body, dict) and k in body}
    if isinstance(body, dict) and "pipelines" in body:
        out["pipelines"] = [p.get("id") for p in body["pipelines"]]
    return out
