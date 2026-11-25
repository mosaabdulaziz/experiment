"""Polls the Redis keys modules ask to be told about (`Module.control()`), on a thread of its own.

A thread, not a GLib timer: when the main loop is wedged (a source reconnect that never returns, see flight.py) a
timer never fires, and a setting saved in the UI would not be applied until the next restart.
"""
import logging
import threading
import time

logger = logging.getLogger("iris.sidecar")


class ControlPoller:
    """Every `interval` seconds reads each key and calls its handler with the new value when it changed.

    The value present at start counts as already seen: the pipeline was just built from whatever it announces. A
    handler that raises is retried on the next tick (the new value is not recorded as seen), so a change that could
    not be applied is never lost."""

    def __init__(self, redis, handlers: dict, interval: float = 1.0) -> None:
        self.redis = redis
        self.handlers = dict(handlers)
        self.interval = interval
        self._seen: dict = {}
        self._thread = None

    def prime(self) -> None:
        for key in self.handlers:
            try:
                self._seen[key] = self.redis.get(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("control key %s unreadable at start: %s", key, exc)
                self._seen[key] = None

    def poll(self) -> None:
        for key, handler in self.handlers.items():
            try:
                value = self.redis.get(key)
                if value == self._seen.get(key):
                    continue
                handler(value)
                self._seen[key] = value
            except Exception as exc:  # noqa: BLE001
                logger.warning("control %s: %s", key, exc)

    def start(self) -> None:
        if not self.handlers or self._thread is not None:
            return
        self.prime()
        self._thread = threading.Thread(target=self._run, name="module-control", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            self.poll()
