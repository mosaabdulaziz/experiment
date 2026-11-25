import asyncio
import json
import logging
import os
import shutil
import time
import uuid

logger = logging.getLogger(__name__)


class ClipStore:
    """Disk-backed store of FOUND search-result preview clips. Each entry is a
    directory `<root_dir>/<clip_id>/` holding one mp4 per camera plus a
    `meta.json`. Entries persist across restarts and are kept until
    `ttl_seconds` past creation, then pruned (a background loop plus lazy
    pruning on access)."""

    def __init__(self, ttl_seconds: float = 1800.0, root_dir: str = "data/clips"):
        self._ttl = max(float(ttl_seconds), 1.0)
        self._root = root_dir
        os.makedirs(self._root, exist_ok=True)

    def _dir(self, clip_id: str) -> str:
        return os.path.join(self._root, clip_id)

    def add(self, cameras_mp4_paths: dict, meta: dict | None = None, clock=time.time) -> str:
        clip_id = uuid.uuid4().hex
        clip_dir = self._dir(clip_id)
        os.makedirs(clip_dir, exist_ok=True)
        cameras = {}
        for cam, src_path in cameras_mp4_paths.items():
            dest = os.path.join(clip_dir, f"{cam}.mp4")
            shutil.move(src_path, dest)
            cameras[cam] = dest
        record = {"cameras": list(cameras), "meta": dict(meta or {}), "created": clock()}
        with open(os.path.join(clip_dir, "meta.json"), "w") as fh:
            json.dump(record, fh)
        return clip_id

    def _load(self, clip_id: str, now: float):
        meta_path = os.path.join(self._dir(clip_id), "meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as fh:
                record = json.load(fh)
            if now - record["created"] > self._ttl:
                shutil.rmtree(self._dir(clip_id), ignore_errors=True)  # lazy prune
                return None
        except (OSError, ValueError, KeyError):
            # Corrupt or partially written metadata; treat as missing
            shutil.rmtree(self._dir(clip_id), ignore_errors=True)
            return None
        return record

    def get(self, clip_id: str, clock=time.time):
        record = self._load(clip_id, clock())
        if record is None:
            return None
        return {"cameras": list(record["cameras"]), "meta": dict(record["meta"])}

    def get_clip_path(self, clip_id: str, camera: str, clock=time.time):
        record = self._load(clip_id, clock())
        if record is None or camera not in record["cameras"]:
            return None
        path = os.path.join(self._dir(clip_id), f"{camera}.mp4")
        return path if os.path.exists(path) else None

    def add_snapshot(self, clip_id: str, camera: str, src_path: str, event_id: int, clock=time.time) -> bool:
        """Keep an incident's still, taken from this clip, with it (so it expires with it); one per event, since
        incidents coalesced onto one clip each have their own. False when the clip is gone."""
        if self._load(clip_id, clock()) is None:
            return False
        shutil.move(src_path, os.path.join(self._dir(clip_id), f"{camera}.{int(event_id)}.snapshot.jpg"))
        return True

    def get_snapshot_path(self, clip_id: str, camera: str, event_id: int, clock=time.time):
        if self._load(clip_id, clock()) is None:
            return None
        path = os.path.join(self._dir(clip_id), f"{camera}.{int(event_id)}.snapshot.jpg")
        return path if os.path.exists(path) else None

    def prune_expired(self, clock=time.time) -> int:
        now = clock()
        removed = 0
        if not os.path.isdir(self._root):
            return 0
        for clip_id in os.listdir(self._root):
            clip_dir = self._dir(clip_id)
            meta_path = os.path.join(clip_dir, "meta.json")
            expired = True
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as fh:
                        record = json.load(fh)
                    expired = now - record["created"] > self._ttl
                except (OSError, ValueError, KeyError):
                    expired = True
            if expired:
                shutil.rmtree(clip_dir, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("Pruned %d expired preview clip(s)", removed)
        return removed

    def clear(self) -> int:
        """Delete every stored clip. Used by the events-clear endpoint, which wipes
        the rest of the debug trail alongside the events themselves."""
        removed = 0
        if os.path.isdir(self._root):
            for clip_id in os.listdir(self._root):
                shutil.rmtree(self._dir(clip_id), ignore_errors=True)
                removed += 1
        os.makedirs(self._root, exist_ok=True)
        return removed

    def __len__(self) -> int:
        if not os.path.isdir(self._root):
            return 0
        return len(os.listdir(self._root))

    async def run_prune_loop(self, check_interval: float = 60.0) -> None:
        while True:
            try:
                await asyncio.sleep(check_interval)
                self.prune_expired()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the loop die silently
                logger.error("Clip prune failed: %s", exc, exc_info=True)
