"""The depth image a proximity alert carries (its pair's area of the depth map, a base64 JPEG), kept per event under
data/alert_depth/ and served to the incident card. It lives as long as the alert clips (alerts.clip.ttl_seconds)."""
import base64
import binascii
import logging
import os
import time

logger = logging.getLogger(__name__)

MAX_BYTES = 256 * 1024


class DepthImages:
    def __init__(self, root_dir: str = "data/alert_depth", ttl_seconds: float = 21600) -> None:
        self._root, self._ttl = root_dir, float(ttl_seconds)

    def _path(self, event_id: int) -> str:
        return os.path.join(self._root, f"{int(event_id)}.jpg")

    def put(self, event_id: int, b64: str, clock=time.time) -> bool:
        """Keep (or replace) an event's depth image; False, with a warning, when it is not a JPEG of at most MAX_BYTES."""
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            data = b""
        if data[:2] != b"\xff\xd8" or len(data) > MAX_BYTES:
            logger.warning("event %s: depth image refused (not a JPEG of at most %d bytes)", event_id, MAX_BYTES)
            return False
        os.makedirs(self._root, exist_ok=True)
        tmp = self._path(event_id) + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, self._path(event_id))
        self.prune(clock)
        return True

    def get(self, event_id: int, clock=time.time):
        """The path of an event's depth image, or None (none kept, or expired)."""
        path = self._path(event_id)
        try:
            return path if clock() - os.path.getmtime(path) <= self._ttl else None
        except OSError:
            return None

    def prune(self, clock=time.time) -> int:
        removed = 0
        for name in os.listdir(self._root) if os.path.isdir(self._root) else []:
            path = os.path.join(self._root, name)
            try:
                if clock() - os.path.getmtime(path) > self._ttl:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
        return removed
