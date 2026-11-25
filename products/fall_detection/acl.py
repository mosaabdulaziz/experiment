"""Anti-Corruption Layer: the only module that knows the platform's Redis wire format.

The shapes are re-declared here, not imported: the product shares no code with another
product, so a change on the platform's side breaks loudly in this file instead of silently
inside the service (design §5).

Three streams, all existing platform contracts (design §7):

  iris:meta:<cam>        read   the sidecar's frames; the `pose` field carries the people
  iris:alert:<cam>       write  the platform's alert -- the app turns it into an event
  iris:svr_request:<cam> write  the clip request -- the sidecar answers with the clip

<cam> is the camera's key id: the camera id, or `<camera>@<role>` when the fall sidecar runs with role-scoped keys
beside other role sidecars on the same camera (the safety product; perception/engine/sources.py stream_key_id).

One file, also a platform contract: the zones file (data/zones_config.txt) the app writes, read by ZonesFile for the
zones whose `fall` block is on.
"""
from __future__ import annotations

import json
import logging
import math
import os

import redis

logger = logging.getLogger("fall_detection.acl")

META_STREAM_PREFIX = "iris:meta:"
ALERT_STREAM_PREFIX = "iris:alert:"
SVR_REQUEST_PREFIX = "iris:svr_request:"

# The platform's own alert stream keeps a bounded history (perception/engine/modules.py:111).
ALERT_MAXLEN = 256

# The separator of a role-scoped key id (perception/engine/sources.py ROLE_KEY_SEPARATOR).
ROLE_KEY_SEPARATOR = "@"


def key_id(camera: str, role: str = "") -> str:
    """The id a camera's streams carry: the camera id, or `<camera>@<role>` for a role-scoped sidecar."""
    return f"{camera}{ROLE_KEY_SEPARATOR}{role}" if role else camera


def _inside(x: float, y: float, polygon: list) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _polygon(points) -> "list | None":
    """A zone's points as [(x, y), ...], or None unless they are at least 3 pairs of finite numbers: a malformed point
    (a hand-edited file) skips its zone, never the candidate being scored."""
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None
    try:
        polygon = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError):
        return None
    return polygon if all(math.isfinite(v) for point in polygon for v in point) else None


class ZonesFile:
    """The zones whose `<feature>` block is on, per camera, re-read when the file changes.

    The file is the app's (`{camera: {"zones": [{"id", "points": [[x, y], ...] in mux pixels, "fall": {"enabled":
    true}}]}}`, or a legacy single-zone entry). A camera with such zones is in scope only inside them; a camera with
    none is in scope everywhere. A missing or unreadable file means no zones.
    """

    def __init__(self, path, feature: str):
        self.path, self.feature = str(path), feature
        self._mtime: float | None = None
        self._by_camera: dict = {}

    def _refresh(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime, self._by_camera = None, {}
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            logger.exception("the zones file %s could not be read -- no %s zones", self.path, self.feature)
            self._by_camera = {}
            return
        by_camera = {}
        for camera, entry in (data.items() if isinstance(data, dict) else ()):
            if not isinstance(entry, dict):
                continue
            raw_zones = entry.get("zones")
            if raw_zones is None:
                raw_zones = [entry] if entry.get("points") else []
            zones = []
            for i, raw in enumerate(raw_zones if isinstance(raw_zones, list) else ()):
                block = raw.get(self.feature) if isinstance(raw, dict) else None
                if not (isinstance(block, dict) and block.get("enabled")):
                    continue
                polygon = _polygon(raw.get("points"))
                if polygon is None:
                    logger.error("the zones file %s: camera %s zone %s needs at least 3 [x, y] points -- skipped",
                                 self.path, camera, raw.get("id") or f"z{i}")
                    continue
                zones.append((str(raw.get("id") or f"z{i}"), polygon))
            if zones:
                by_camera[camera] = zones
        self._by_camera = by_camera

    def covering(self, camera: str, x: float, y: float) -> tuple[bool, str]:
        """(in scope, zone id) for a point in mux pixels: (True, "") on a camera with no zones."""
        self._refresh()
        zones = self._by_camera.get(camera)
        if not zones:
            return True, ""
        for zone_id, polygon in zones:
            if _inside(x, y, polygon):
                return True, zone_id
        return False, ""


def text(value) -> str:
    """A stream field as text: Redis hands back bytes unless the client decodes them."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def format_pts(pts) -> str:
    """The frame's own clock, as the stream carries it.

    The platform's alert writer rounds to three decimals; the fall product writes the clock it
    was given, so the alert and the clip request carry the same string (AC-F19.1).
    """
    return f"{pts}"


class _Cursor:
    """Per-camera XREAD cursor, primed at the stream's tip.

    A non-blocking XREAD from "$" never returns anything, so a consumer that starts now reads
    the tip first and treats the history before it as history (design §5.1).
    """

    def __init__(self, r: redis.Redis, prefix: str):
        self.r = r
        self.prefix = prefix
        self._last: dict[str, str] = {}

    def read(self, source_id: str, count: int) -> list:
        key = f"{self.prefix}{source_id}"
        last = self._last.get(source_id)
        if last is None:
            tip = self.r.xrevrange(key, count=1)
            self._last[source_id] = text(tip[0][0]) if tip else "0-0"
            return []
        response = self.r.xread({key: last}, count=count)
        records = [rec for _key, recs in (response or []) for rec in recs]
        if records:
            self._last[source_id] = text(records[-1][0])
        return records


class MetaStreamReader:
    """One camera's frames, oldest first."""

    def __init__(self, r: redis.Redis):
        self._cursor = _Cursor(r, META_STREAM_PREFIX)

    def read(self, source_id: str, count: int = 64) -> list[dict]:
        """The frames since the last read, as {"id": <entry id>, "fields": {...}}."""
        return [{"id": text(record_id), "fields": {text(k): text(v) for k, v in fields.items()}}
                for record_id, fields in self._cursor.read(source_id, count)]


def write_alert(r: redis.Redis, camera: str, *, severity: str, message: str, pts,
                kind: str = "fall", category: str = "safety", zone: str = "",
                count=None, person_ids: str = "", extra: "dict | None" = None) -> bool:
    """Raise the platform's alert on a camera.

    The field set is the platform's own (`perception/engine/modules.py:99-111`): the app turns
    the entry into an event -- saved, broadcast, clip queued -- with no code of this product's
    own. False when the write failed; the alert is never worth killing the consumer over.
    `person_ids`: the track of the person who fell, so the alert's clip draws that person's box (the sidecar's pose
    module writes each person's box by track id; app/overlay_track.py).
    `extra`: more fields, kept on the event's details (app/usecases/module_alerts.py) -- what the message says, as data
    a console can word in its own language.
    """
    fields = {"v": "1", "kind": kind, "severity": severity, "category": category,
              "message": message, "zone": zone, "pts": format_pts(pts),
              "count": "" if count is None else str(count)}
    if person_ids:
        fields["person_ids"] = person_ids
    fields.update({k: str(v) for k, v in (extra or {}).items() if k not in fields})
    try:
        r.xadd(f"{ALERT_STREAM_PREFIX}{camera}", fields, maxlen=ALERT_MAXLEN)
    except redis.RedisError:
        logger.exception("the alert on %s could not be written", camera)
        return False
    return True


def request_clip(r: redis.Redis, camera: str, pts) -> bool:
    """Ask the sidecar for the clip around a frame.

    The sidecar answers with the clip centred on that frame (perception/engine/main.py:258-289).
    """
    try:
        r.xadd(f"{SVR_REQUEST_PREFIX}{camera}", {"pts": format_pts(pts)})
    except redis.RedisError:
        logger.exception("the clip request on %s could not be written", camera)
        return False
    return True
