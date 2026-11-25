"""Shelf -> iris-platform bridge:  python -m shelf_iris_bridge   (README.md, "Shelf -> iris-platform")

Reads the shelf_void module's alerts from Redis (iris:alert:<camera>, or iris:alert:<camera>@<role> when the sidecar
runs with role_scoped_keys; its own consumer group) and publishes them to
iris-platform over RabbitMQ as an AI backend: catalog.snapshot, state.sync.request, pipeline.status, event.raised /
event.cleared. Run EITHER this bridge OR the shelf-demo service against one iris-platform, never both: they publish
the same catalog id and consume the same queue.

Environment (.env.example):
  AMQP_URL          amqp(s)://user:password@host:port/vhost           (required; never logged)
  AMQP_USER         user_id on published messages (default: the user in AMQP_URL)
  TARGET            cloud | edge.<org-slug>                            (default cloud)
  AMQP_QUEUE        inbound queue (default ai.cloud.inbound / edge.<slug>.inbound)
  PUBLISH_CATALOG   true | false: send catalog.snapshot (cloud only)  (default true)
  CONFIDENCE_MODE   stability | detector                               (default stability)
  REDIS_URL         the stack's Redis                                  (default redis://redis:6379/0)
  MAPPING_FILE      DeepStream camera + region -> iris-platform ids    (default /config/shelf-iris-mapping.yaml)
  CONSUMER_GROUP    the bridge's own group on the alert streams        (default shelf-iris-bridge)
  CONSUMER_NAME     (default: the host name)
  SIDECAR_CONFIG    the shelf sidecar's preset yaml, as the sidecar runs it  (required: role_scoped_keys, role)
  IRIS_SIDECAR_ROLE the sidecar's role, only when the sidecar gets one from its environment (else the yaml's)
  IRIS_ROOT         the checkout a relative SIDECAR_CONFIG starts from  (default: the working directory)
"""
import os
import signal
import sys
import threading

import redis

from .alerts import AlertReader
from .bridge import Bridge
from .broker import Broker
from .config import Settings
from .keys import KeyScopeError, load_alert_keys
from .log import log
from .mapping import Mapping, MappingError


def main() -> None:
    try:
        settings = Settings.from_env()
    except ValueError as e:
        sys.exit(f"bad setting: {e}")
    if not settings.amqp_url:
        sys.exit("AMQP_URL is required")
    try:
        mapping = Mapping.load(settings.mapping_file)
    except MappingError as e:
        log("cannot start", error=str(e))
        sys.exit(2)
    try:
        keys = load_alert_keys(mapping.cameras, settings.sidecar_config, settings.sidecar_role, settings.iris_root)
    except KeyScopeError as e:
        log("cannot start", error=str(e))
        sys.exit(2)

    broker = Broker(settings, on_message=None)
    bridge = Bridge(settings, mapping, broker.publish, mapping_name=os.path.basename(settings.mapping_file))
    broker.on_message = bridge.handle
    broker.on_connected = bridge.startup_messages
    reader = AlertReader(redis.Redis.from_url(settings.redis_url, socket_timeout=10), keys.streams,
                         settings.group, settings.consumer, bridge.on_alert)

    def on_signal(signum, _frame):
        log("stopping", signal=signal.Signals(signum).name)
        reader.stop()
        broker.stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    log("starting", target=settings.target, queue=settings.inbound_queue, user=settings.amqp_user,
        broker=settings.safe_url, redis=settings.safe_redis_url, group=settings.group, consumer=settings.consumer,
        cameras=mapping.cameras, regions=len(mapping.targets), publish_catalog=settings.publish_catalog,
        confidence_mode=settings.confidence_mode)
    log("alert streams", mode=keys.mode, role_scoped_keys=keys.scoped, role=keys.role,
        sidecar_config=settings.sidecar_config, streams=list(keys.streams))
    threading.Thread(target=reader.run, name="alerts", daemon=True).start()
    broker.run()


if __name__ == "__main__":
    main()
