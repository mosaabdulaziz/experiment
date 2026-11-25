"""Anti-Corruption Layer: the only module that knows surveillance-camera-sys's
Redis wire format. Nothing here imports that project's code -- the shapes are
re-declared, so a change on that side breaks loudly here.

Streams read (2026-09-24, three cameras):
  iris:meta:<cam>       every frame: `person_boxes` [[left, top, w, h, track_id]]
                        in mux pixels (+ `ntp_ts` for the camera clock);
  iris:appearance:<cam> sampled crops (perception/modules/appearance/worker.py): `items`
                        [{"track_id", "reid"?: base64 float16, "staff_p"?}].
A record's time is its Redis entry id (ms, this server's clock at publish):
one pipeline publishes all three cameras, so entry ids share one clock --
what the paired-zone handoff needs. Face-ID streams are no longer read
(staff are told by uniform).

Forked from products/attendance/acl.py @ 1f5b27b -- kept: the camera clock
and LiveCameraWatch; the rest is new.
"""

import base64
import json
import logging
import math
import struct
import time
from datetime import datetime

import redis

from domain import Appearance, FaceCandidate, Frame, TrackObs

logger = logging.getLogger("barber.acl")

META_STREAM_PREFIX = "iris:meta:"
APPEARANCE_STREAM_PREFIX = "iris:appearance:"
CUSTOMER_FACE_STREAM_PREFIX = "iris:customer_face:"


# ── Camera clock (issue 1 Phase 2, 2026-09-12) ─────────────────────────────
# The sidecar attaches each frame's camera-side capture time (from the
# camera's RTCP Sender Reports) to iris:meta as `ntp_ts`, in nanoseconds --
# 0 when the camera never sent one (see sidecar.py's attach-sys-ts note).
# Seconds from the NTP epoch (1900-01-01) to the Unix epoch (1970-01-01).
_NTP_TO_UNIX_SECONDS = 2208988800
# A seconds value above this can only be NTP-epoch-based (it's ~2065 in Unix
# time, ~1995 in NTP time). DeepStream reports Unix-epoch ns; accepting both
# costs one comparison.
_NTP_EPOCH_THRESHOLD = 3.0e9
# A camera whose clock was never set reports e.g. 1970 or 2000. Anything
# before this counts as "no usable camera clock" rather than dating
# sightings decades back (which would also drop them out of today's table).
_MIN_PLAUSIBLE_CAMERA_TIME = datetime(2020, 1, 1).timestamp()


def camera_clock_offset_seconds(ntp_ts_ns: int, record_id: bytes | str) -> float | None:
    """(camera clock - server clock) in seconds for one iris:meta entry, or
    None if it carries no usable camera time. The server-side reference is
    the entry's own Redis stream id, whose first half is the millisecond time
    Redis stored it -- this machine's clock at publish time, a fraction of a
    second after the frame was captured."""
    if ntp_ts_ns <= 0:
        return None
    camera_s = ntp_ts_ns / 1e9
    if camera_s > _NTP_EPOCH_THRESHOLD:
        camera_s -= _NTP_TO_UNIX_SECONDS
    if camera_s < _MIN_PLAUSIBLE_CAMERA_TIME:
        return None
    rid = record_id.decode() if isinstance(record_id, bytes) else record_id
    return camera_s - int(rid.split("-", 1)[0]) / 1000.0


class CameraClock:
    """Latest camera-vs-server clock offset per camera, shared by the consumer
    thread (stamps sightings with it) and the API (/api/recordings, which
    feeds the camera tile's clock) -- both run in this one process. A single
    dict assignment/read is atomic under the GIL, so no lock."""

    # No fresh reading for this long = the camera stopped providing its clock
    # (or its stream dropped): fall back to server time.
    MAX_AGE_SECONDS = 10.0

    def __init__(self) -> None:
        self._latest: dict[str, tuple[float, float]] = {}  # cam -> (offset_s, monotonic learned)

    def note(self, source_id: str, offset_seconds: float) -> None:
        self._latest[source_id] = (offset_seconds, time.monotonic())

    def offset(self, source_id: str) -> float | None:
        """Seconds to add to this server's clock to get the camera's, or None
        when there's no fresh camera clock (the caller uses server time)."""
        entry = self._latest.get(source_id)
        if entry is None or time.monotonic() - entry[1] > self.MAX_AGE_SECONDS:
            return None
        return entry[0]


camera_clock = CameraClock()


class LiveCameraWatch:
    """Which LIVE cameras are actually delivering video right now, and since
    when (2026-09-13). data/stream_starts.json is written only by the test
    streamer, for the recordings it plays -- so without this a live camera
    never counted as streaming: the consumer skipped it and nobody seen on it
    ever reached the attendance table (or got a tile clock).

    "Delivering video" = the sidecar published a frame record for it
    (iris:meta:<cam>) within FRESH_SECONDS. The sidecar's own status flag says
    running even for a camera with no video (checked 2026-09-13), so it can't
    be used. Refreshed by the consumer thread every cycle; read by the API
    (same process) for the camera tile's clock. refresh() publishes a new dict
    by plain assignment, so snapshot() never sees one being modified."""

    FRESH_SECONDS = 10.0

    def __init__(self) -> None:
        self._since: dict[str, datetime] = {}

    def refresh(self, r: redis.Redis, cameras: list[str]) -> dict[str, datetime]:
        now_ms = time.time() * 1000
        old, new = self._since, {}
        for cam in cameras:
            try:
                tip = r.xrevrange(f"{META_STREAM_PREFIX}{cam}", count=1)
            except redis.RedisError:
                if cam in old:  # Redis hiccup: keep the last known state
                    new[cam] = old[cam]
                continue
            if tip:
                rid = tip[0][0]
                rid = rid.decode() if isinstance(rid, bytes) else rid
                if now_ms - int(rid.split("-", 1)[0]) <= self.FRESH_SECONDS * 1000:
                    new[cam] = old.get(cam, datetime.now())
        self._since = new
        return dict(new)

    def snapshot(self) -> dict[str, datetime]:
        return dict(self._since)


live_watch = LiveCameraWatch()


def _entry_ts(record_id) -> float:
    rid = record_id.decode() if isinstance(record_id, bytes) else record_id
    return int(rid.split("-", 1)[0]) / 1000.0


def decode_embedding(s: str) -> tuple[float, ...]:
    raw = base64.b64decode(s)
    v = struct.unpack(f"<{len(raw) // 2}e", raw)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


class _Cursor:
    """Per-camera XREAD cursor, primed at the stream's tip: a non-blocking
    XREAD from "$" never returns anything (found 2026-09-01)."""

    def __init__(self, r: redis.Redis, prefix: str):
        self.r, self.prefix = r, prefix
        self._last: dict[str, str] = {}

    def read(self, source_id: str, count: int):
        key = f"{self.prefix}{source_id}"
        last = self._last.get(source_id)
        if last is None:
            tip = self.r.xrevrange(key, count=1)
            self._last[source_id] = tip[0][0].decode() if tip else "0-0"
            return []
        response = self.r.xread({key: last}, count=count)
        records = [rec for _key, recs in (response or []) for rec in recs]
        if records:
            rid = records[-1][0]
            self._last[source_id] = rid.decode() if isinstance(rid, bytes) else rid
        return records



def _outline(raw) -> tuple[float, ...] | None:
    if not isinstance(raw, list) or len(raw) < 6 or len(raw) % 2:
        return None
    try:
        vals = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return vals if all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals) else None


def tracks_from(person_boxes, person_masks) -> tuple[TrackObs, ...]:
    """person_boxes (left, top, w, h, id) merged with person_masks [id, conf, outline] by id. A
    malformed mask entry is dropped (its box stays); a malformed field is ignored -- the consumer
    thread must never die on one bad message."""
    boxes = json.loads(person_boxes) if person_boxes else []
    extra: dict[int, tuple[float | None, tuple[float, ...] | None]] = {}
    try:
        for item in (json.loads(person_masks) if person_masks else []):
            if not isinstance(item, list) or len(item) != 3:
                continue
            try:
                oid, conf = int(item[0]), float(item[1])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(conf):
                continue
            extra[oid] = (conf, _outline(item[2]))
    except (ValueError, TypeError):
        extra = {}
    out = []
    for left, top, w, h, oid in boxes:
        conf, outline = extra.get(int(oid), (None, None))
        out.append(TrackObs(int(oid), (left, top, w, h), conf, outline))
    return tuple(out)


class MetaStreamReader:
    """One Frame per iris:meta record, oldest first. count=64 = the stream's
    maxlen: one call reads the whole retained window, so the cursor can't
    fall behind the trim."""

    def __init__(self, r: redis.Redis):
        self._cursor = _Cursor(r, META_STREAM_PREFIX)

    def read_frames(self, source_id: str) -> list[Frame]:
        frames = []
        for record_id, fields in self._cursor.read(source_id, count=64):
            ntp_raw = fields.get(b"ntp_ts")
            if ntp_raw:
                offset = camera_clock_offset_seconds(int(ntp_raw), record_id)
                if offset is not None:
                    camera_clock.note(source_id, offset)
            tracks = tracks_from(fields.get(b"person_boxes"), fields.get(b"person_masks"))
            frames.append(Frame(source_id, _entry_ts(record_id), tracks))
        return frames


class AppearanceStreamReader:
    def __init__(self, r: redis.Redis):
        self._cursor = _Cursor(r, APPEARANCE_STREAM_PREFIX)

    def read(self, source_id: str) -> list[Appearance]:
        out = []
        for record_id, fields in self._cursor.read(source_id, count=256):
            ts = _entry_ts(record_id)
            for item in json.loads(fields[b"items"]):
                out.append(Appearance(source_id, int(item["track_id"]), ts,
                                      decode_embedding(item["reid"]) if item.get("reid") else None,
                                      item.get("staff_p"),
                                      item.get("crop_path"), item.get("crop_quality")))
        return out


class FaceCandidateReader:
    def __init__(self, r: redis.Redis):
        self._cursor = _Cursor(r, CUSTOMER_FACE_STREAM_PREFIX)

    def read(self, source_id: str) -> list[FaceCandidate]:
        out = []
        for record_id, fields in self._cursor.read(source_id, count=256):
            item = json.loads(fields[b"item"])
            if not item.get("embedding"):
                continue
            out.append(FaceCandidate(source_id, int(item["track_id"]), _entry_ts(record_id), item["crop_path"],
                                     decode_embedding(item["embedding"]), float(item.get("quality", 0.0)),
                                     face_box_of(item)))
        return out


def face_box_of(item: dict) -> tuple[float, float, float, float] | None:
    """A customer-face item's face_box (left, top, width, height), or None when absent or malformed."""
    box = item.get("face_box")
    try:
        left, top, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return (left, top, w, h) if w > 0 and h > 0 else None
