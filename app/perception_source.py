import asyncio
import json
import logging

import redis

logger = logging.getLogger("iris.perception")


# DeepStream's UNTRACKED_OBJECT_ID — what int(obj_meta.object_id) yields when the tracker
# never assigned an id. Duplicated from app/stream_manager.py's _UNTRACKED_OBJECT_ID:
# importing it from there would be circular (stream_manager imports PerceptionSource).
# Keep both copies in sync if this value ever changes.
_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


def parse_dets(raw: list) -> list[tuple]:
    """Wire-format detections -> (cls, conf, track_id, bbox) tuples.

    The sidecar publishes [label, conf, object_id, [l, t, w, h]] on iris:meta.
    Older sidecars publish fewer fields; a newer app must keep working against
    one, so every field past conf is optional. bbox is in mux pixel coordinates.

    object_id == UNTRACKED_OBJECT_ID (the tracker never assigned an id) normalises
    to track_id=None rather than being passed through as a real id. A bbox with any
    length other than 4 is treated as absent rather than accepted malformed.
    """
    out = []
    for d in raw:
        track_id = int(d[2]) if len(d) > 2 and d[2] is not None else None
        if track_id == _UNTRACKED_OBJECT_ID:
            track_id = None
        if len(d) >= 7:
            bbox = (float(d[3]), float(d[4]), float(d[5]), float(d[6]))
        elif len(d) > 3 and isinstance(d[3], (list, tuple)) and len(d[3]) == 4:
            bbox = tuple(float(v) for v in d[3])
        else:
            bbox = None
        out.append((str(d[0]), float(d[1]), track_id, bbox))
    return out


class PerceptionSource:
    """Redis-backed stand-in for Camera, fed by the DeepStream sidecar.

    Reads the data plane (iris:meta / iris:jpeg) for the scheduler + /stream, and
    drives the control plane: publishes commands to iris:ctrl:<id> and reads state
    from iris:status:<id> so the UI reflects what the sidecar is actually doing.
    """

    def __init__(self, config: dict, redis_client=None, source_id: str | None = None):
        perc = config.get("perception", {})
        self._source_id = source_id or perc.get("source_id", "cam0")
        self._meta_key = f"iris:meta:{self._source_id}"
        self._jpeg_key = f"iris:jpeg:{self._source_id}"
        self._ctrl_key = f"iris:ctrl:{self._source_id}"
        self._status_key = f"iris:status:{self._source_id}"
        self._uploads_prefix = perc.get(
            "sidecar_uploads_uri_prefix",
            "file:///workspace/iris/data/uploads",
        )
        self._batch_size = max(int(config.get("sampling", {}).get("frames_per_batch", 8)), 1)  # max frames/dets returned by grab_annotated_batch, not a cap on what the sidecar pushes
        self._r = redis_client or redis.Redis.from_url(perc.get("redis_url", "redis://localhost:6379/0"))
        self._source = f"deepstream:{self._source_id}"

    # ── control plane (commands out, status in) ──────────────────────────────
    def _status(self) -> dict:
        try:
            raw = self._r.get(self._status_key)
        except Exception as exc:
            logger.warning("[%s] redis status read failed: %s", self._source_id, exc)
            return {}
        logger.debug("[%s] status read hit=%s", self._source_id, raw is not None)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}

    def _publish(self, cmd: dict) -> None:
        self._r.rpush(self._ctrl_key, json.dumps(cmd))

    def upload_uri(self, filename: str) -> str:
        """Build the URI for an uploaded file as the *sidecar* sees it on disk."""
        return f"{self._uploads_prefix.rstrip('/')}/{filename}"

    # ── status surface (read from the sidecar's status key) ───────────────────
    @property
    def is_running(self) -> bool:
        return bool(self._status().get("running", False))

    @property
    def is_paused(self) -> bool:
        return False

    @property
    def is_file(self) -> bool:
        return bool(self._status().get("is_file", False))

    @property
    def current_offset(self) -> float:  # timestamp of the most recent sampled frame (video position); 0.0 if none/malformed
        entries = self._r.xrevrange(self._meta_key, count=1)
        if not entries:
            return 0.0
        return float(entries[0][1].get(b"pts", b"0") or 0.0)

    @property
    def source_label(self) -> str:
        s = self._status()
        if not s:
            return "sidecar unavailable"
        return s.get("source") or "No source"

    @property
    def preview_uri(self) -> str | None:
        """Raw source URI (e.g. rtsp://go2rtc:8554/cam-1) the sidecar exposes for
        raw-preview mode (`pipeline_preview: false`). The frontend plays this over
        WebRTC (WHEP via go2rtc) instead of polling iris:jpeg. None in
        pipeline-preview mode or when the sidecar hasn't published a status yet."""
        return self._status().get("preview_uri")

    @property
    def preview_mode(self) -> str:
        """How the frontend should render this stream's preview: "pipeline" (poll
        iris:jpeg — annotated frames from DeepStream) or "raw" (play preview_uri
        directly; boxes come from iris:meta). Defaults to pipeline when the sidecar
        hasn't said otherwise."""
        return self._status().get("preview_mode", "pipeline")

    # ── batch + frame reads (data plane) ──────────────────────────────────────
    def grab_detections(self):
        """Return (entry_id, detections) from the NEWEST iris:meta entry, or None.

        Detections only — the JPEG on that entry is never read. This runs on every
        sampler tick purely to count durations, and a frame per tick would be Redis
        bandwidth spent on bytes nobody decodes.

        The entry id comes back so the caller can skip an entry it already consumed:
        the sampler polls faster than the sidecar publishes, and re-observing the
        same frame would let a stalled sidecar accrue duration forever. None when
        the stream is empty, unreadable, or carries malformed dets — memory is an
        aid, so a bad read is a skipped tick, never a raise into the caller's loop.
        """
        try:
            entries = self._r.xrevrange(self._meta_key, count=1)
        except Exception as exc:
            logger.warning("[%s] redis detections read failed: %s", self._source_id, exc)
            return None
        if not entries:
            return None
        eid, fields = entries[0]
        try:
            dets = parse_dets(json.loads(fields.get(b"dets", b"[]") or b"[]"))
        except Exception as exc:
            logger.warning("[%s] malformed dets on iris:meta: %s", self._source_id, exc)
            return None
        return (eid.decode() if isinstance(eid, bytes) else eid), dets

    def grab_annotated_batch(self):
        """Return (frames, detections_list, offsets) from the last N stream entries,
        oldest->newest. detections_list entries are
        [(class, conf, track_id, bbox), ...]; track_id and bbox are None when the
        sidecar is older than this app."""
        entries = self._r.xrevrange(self._meta_key, count=self._batch_size)
        if not entries:
            return None
        entries = list(reversed(entries))  # oldest -> newest
        frames, dets_list, offsets = [], [], []
        for _id, fields in entries:
            jpeg = fields.get(b"jpeg")
            if jpeg is None:
                continue
            frames.append(jpeg)
            offsets.append(float(fields.get(b"pts", b"0") or 0.0))
            dets_list.append(parse_dets(json.loads(fields.get(b"dets", b"[]"))))
        if not frames:
            return None
        return frames, dets_list, offsets

    def grab_frame_batch(self):
        batch = self.grab_annotated_batch()
        if batch is None:
            return [], []
        frames, _dets, offsets = batch
        return frames, offsets

    def grab_frame(self):
        return self._r.get(self._jpeg_key)

    def grab_display_frame(self):
        return self._r.get(self._jpeg_key)

    def mark_watched(self, ttl: int = 10) -> None:
        """Refresh the "watched" marker the sidecar uses for viewer-gated
        preview encoding (perception/engine/sources.ViewerGate). Called on every snapshot
        or MJPEG request; the short TTL means a stream stops being encoded
        ~TTL seconds after the last viewer looks away. Never raises — a marker
        failure must not fail the frame request itself."""
        try:
            self._r.set(f"iris:watched:{self._source_id}", "1", ex=ttl)
        except Exception:
            pass

    async def generate_mjpeg(self):  # yields multipart MJPEG frames for /stream; 40ms poll keeps CPU reasonable
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        loop = asyncio.get_running_loop()
        while True:
            # The Redis read is blocking; run it in the executor so a busy stream
            # (or several concurrent MJPEG viewers) never stalls the event loop.
            frame = await loop.run_in_executor(None, self.grab_display_frame)
            if frame:
                yield boundary + frame + b"\r\n"
            await asyncio.sleep(0.04)

    # lifecycle -> control commands
    def set_source(self, source) -> None:
        self._publish({"cmd": "switch_source", "uri": source})
    async def start(self):
        self._publish({"cmd": "start"})
    async def stop(self):
        self._publish({"cmd": "stop"})
    def close(self) -> None:
        # Release the Redis connection on app shutdown (does NOT command the sidecar).
        try:
            self._r.close()
        except Exception:
            pass
    def pause(self):
        return None
    def resume(self):
        return None
