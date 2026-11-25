"""Segment metadata lives in a per-source Redis stream (`iris:segments:<id>`);
segment video bytes live either in a MinIO bucket or on local disk, selected
by `backend`. Both clients/paths are injected so this is testable against
fakeredis + a fake MinIO client, with no live services."""
import os
import shutil
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentRef:
    start_pts: float
    end_pts: float
    object_key: str
    recorded_at: float
    # the segment's first frame time on the pipeline's clock (the index's own start_pts); in wall mode start/end_pts
    # are wall labels (indexed time minus duration), which can be seconds off -- this is what places frames exactly
    pts_start: "float | None" = None


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


class SegmentStore:
    def __init__(self, redis_client, backend: str = "minio", *,
                minio_client=None, bucket: str | None = None,
                root_dir: str | None = None):
        self._r = redis_client
        self._backend = backend
        self._minio = minio_client
        self._bucket = bucket
        self._root_dir = root_dir
        if backend == "local":
            os.makedirs(root_dir, exist_ok=True)

    def _index_key(self, source_id: str) -> str:
        return f"iris:segments:{source_id}"

    def _local_path(self, object_key: str) -> str:
        return os.path.join(self._root_dir, object_key)

    def put(self, source_id: str, start_pts: float, end_pts: float, local_path: str,
           clock=time.time) -> str:
        object_key = f"{source_id}/{start_pts:.3f}_{end_pts:.3f}_{uuid.uuid4().hex[:8]}.ts"
        if self._backend == "minio":
            self._minio.fput_object(self._bucket, object_key, local_path)
        else:
            dest = self._local_path(object_key)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(local_path, dest)
        recorded_at = clock()
        try:
            self._r.xadd(self._index_key(source_id), {
                "start_pts": f"{start_pts:.3f}",
                "end_pts": f"{end_pts:.3f}",
                "object_key": object_key,
                "recorded_at": f"{recorded_at:.3f}",
            })
        except Exception:
            # If xadd fails after the bytes are already stored, clean up the
            # orphan. Swallow any secondary exception from cleanup so the
            # original error propagates.
            try:
                if self._backend == "minio":
                    self._minio.remove_object(self._bucket, object_key)
                else:
                    os.remove(self._local_path(object_key))
            except Exception:
                pass
            raise
        return object_key

    @property
    def redis(self):
        """The Redis client the index lives in (the alert clip service reads the overlay stream from it)."""
        return self._r

    def list_range(self, source_id: str, lo: float, hi: float, time_mode: str = "auto") -> list:
        """
        List all segments for a source that overlap the given time range.

        Args:
            source_id: The ID of the source to query.
            lo: The lower bound of the time range (inclusive).
            hi: The upper bound of the time range (inclusive).
            time_mode: 'auto', 'wall', or 'pts'. In 'auto' mode, timestamps > 1e8
                       (Unix epoch seconds) are evaluated against each segment's
                       recorded_at wall-clock time span.
        """
        entries = self._r.xrange(self._index_key(source_id), "-", "+")
        out = []
        is_wall = (time_mode == "wall") or (time_mode == "auto" and lo > 1e8)
        for _entry_id, fields in entries:
            f = {_decode(k): _decode(v) for k, v in fields.items()}
            start_pts, end_pts = float(f["start_pts"]), float(f["end_pts"])
            recorded_at = float(f.get("recorded_at", 0.0) or 0.0)
            if is_wall:
                duration = end_pts - start_pts
                if duration <= 0:
                    duration = 6.0
                wall_start = recorded_at - duration
                wall_end = recorded_at
                if wall_end >= lo and wall_start <= hi:
                    out.append(SegmentRef(wall_start, wall_end, f["object_key"], recorded_at, start_pts))
            else:
                if end_pts >= lo and start_pts <= hi:
                    out.append(SegmentRef(start_pts, end_pts, f["object_key"], recorded_at, start_pts))
        return sorted(out, key=lambda s: s.start_pts)


    def fetch(self, ref: SegmentRef, dest_dir: str) -> str:
        if self._backend == "minio":
            local_path = f"{dest_dir}/{ref.object_key.replace('/', '_')}"
            self._minio.fget_object(self._bucket, ref.object_key, local_path)
            return local_path
        return self._local_path(ref.object_key)

    def expire(self, retention_seconds: float, clock=time.time) -> int:
        cutoff = clock() - retention_seconds
        removed = 0
        for key in self._r.keys("iris:segments:*"):
            key = _decode(key)
            entries = self._r.xrange(key, "-", "+")
            for entry_id, fields in entries:
                f = {_decode(k): _decode(v) for k, v in fields.items()}
                if float(f["recorded_at"]) < cutoff:
                    if self._backend == "minio":
                        self._minio.remove_object(self._bucket, f["object_key"])
                    else:
                        try:
                            os.remove(self._local_path(f["object_key"]))
                        except OSError:
                            pass
                    self._r.xdel(key, entry_id)
                    removed += 1
        return removed
