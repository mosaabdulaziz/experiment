"""Hot-reloadable configuration the app edits and a module applies live, with an acknowledgement.

The app saves the configuration (zones, a proximity limit, a PPE rule) to a file or Redis, then bumps an integer
`version_key`. The module's control hook sees the bump, reloads, and writes the version it applied to `applied_key`;
the app compares the two and warns the operator while applied < version (a reload that keeps failing keeps them
apart, which is the signal). Pure stdlib, so a module and its test need no GStreamer.

    cfg = VersionedConfig(redis, "iris:ctrl:thing_version", "iris:thing:applied")
    def control(self): return cfg.control(self._reload)       # _reload(version) raises to retry on the next tick
    def on_built(self, p): self._load(); cfg.acknowledge()    # what exists at start is already applied
"""
import logging
from typing import Callable

logger = logging.getLogger("iris.sidecar")


class VersionedConfig:
    def __init__(self, redis, version_key: str, applied_key: str) -> None:
        self.redis = redis
        self.version_key, self.applied_key = version_key, applied_key

    def version(self) -> int:
        v = self.redis.get(self.version_key)
        return int(v) if v is not None else 0

    def acknowledge(self, version: "int | None" = None) -> None:
        """Mark `version` (default: the current one) applied. At start this also heals changes saved while the
        process was down; a failure is only logged."""
        try:
            self.redis.set(self.applied_key, self.version() if version is None else version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: startup ack failed: %s", self.applied_key, exc)

    def control(self, reload: Callable[[int], None]) -> dict:
        """The `Module.control()` mapping: on a bump, `reload(version)`, then acknowledge it. `reload` raising means
        no acknowledgement and a retry on the next poll."""
        def on_bump(_value) -> None:
            version = self.version()
            reload(version)
            self.redis.set(self.applied_key, version)
        return {self.version_key: on_bump}
