"""Reads the shelf sidecar's alert streams (iris:alert:<camera>, or iris:alert:<camera>@<role> with role_scoped_keys:
keys.py) through the bridge's OWN consumer group.

The app reads the same streams with a plain XREAD (app/usecases/module_alerts.py), not a group, so a group here takes
nothing away from it: every entry still reaches the app. The group remembers where the bridge stopped, so a restart
resumes after the last acknowledged entry: first its own pending entries (read, not yet acked), then new ones.
A group made the first time starts at the stream's end ("$"): alerts from before the bridge ever ran are not sent.
"""
import threading

import redis

from .keys import camera_of
from .log import log


class AlertReader:
    def __init__(self, client, streams, group: str, consumer: str, on_alert, block_ms: int = 1000, count: int = 50):
        """`streams`: the alert stream names (keys.AlertKeys.streams). `on_alert(camera, entry id, fields)` gets the
        mapping's camera, the stream's @<role> suffix stripped."""
        self.r, self.streams, self.group, self.consumer = client, list(streams), group, consumer
        self.on_alert, self.block_ms, self.count = on_alert, block_ms, count
        self._stop = threading.Event()
        self._pending_done = False

    def ensure_groups(self) -> None:
        for stream in self.streams:
            try:
                self.r.xgroup_create(stream, self.group, id="$", mkstream=True)
                log("consumer group created", stream=stream, group=self.group)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    def poll(self) -> int:
        """One read round: this consumer's pending entries until there are none, then new ones. Returns the number
        of entries handled (each acked after its handler ran)."""
        start = ">" if self._pending_done else "0"
        streams = {stream: start for stream in self.streams}
        resp = self.r.xreadgroup(self.group, self.consumer, streams, count=self.count,
                                 block=self.block_ms if self._pending_done else None)
        n = 0
        for key, entries in resp or []:
            key = key.decode() if isinstance(key, bytes) else key
            cam = camera_of(key)
            for entry_id, fields in entries:
                entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                if fields is None:          # pending entry trimmed from the stream (MAXLEN): nothing to send
                    self.r.xack(key, self.group, entry_id)
                    continue
                f = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                     for k, v in fields.items()}
                try:
                    self.on_alert(cam, entry_id, f)
                except Exception as e:      # a bad entry must not stop the stream
                    log("alert handler error", camera=cam, entry=entry_id, error=f"{type(e).__name__}: {e}")
                self.r.xack(key, self.group, entry_id)
                n += 1
        if not self._pending_done and n == 0:
            self._pending_done = True
        return n

    def run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                self.ensure_groups()
                delay = 1.0
                while not self._stop.is_set():
                    self.poll()
            except (redis.ConnectionError, redis.TimeoutError) as e:
                log("redis connection lost", error=f"{type(e).__name__}: {e}", retry_in_s=delay)
                self._pending_done = False
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, 30.0)
        log("alert reader stopped")

    def stop(self) -> None:
        self._stop.set()
