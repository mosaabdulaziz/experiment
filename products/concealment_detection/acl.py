"""Anti-Corruption Layer for Redis streams (`iris:meta:<cam>`, `iris:alert:<cam>`, `iris:svr_request:<cam>`).

Provenance: adapted from `products/fall_detection/acl.py` (`_Cursor`, `MetaStreamReader`,
`write_alert`, `request_clip`). The shapes are re-declared here rather than imported so this
product shares no code with other products.

Wire contract (`perception/engine/frame.py:157-172`, `perception/engine/publisher.py:337-353`):
- `iris:meta:<cam>` carries `v="1"`, `frame_num`, `pts`, `dets`, and module-owned JSON fields:
  - `pose`: `{"shape": [capture_w, capture_h], "items": [{"id", "box", "kps"}]}` in capture pixels.
  - `dets`: `[[label, conf, id], ...]` by default (`dets_with_boxes=False`), or 7-field rows
    `[[label, shown_conf, id, left, top, width, height], ...]` when a module sets
    `frame.dets_with_boxes = True`.
"""
from __future__ import annotations

import json
import logging
import math
import time
import redis

from concealment.types import COCO_KEYPOINTS, PoseFrame, TrackObservation
from evidence import validate_non_accusatory_text

logger = logging.getLogger("concealment_detection.acl")

META_STREAM_PREFIX = "iris:meta:"
ALERT_STREAM_PREFIX = "iris:alert:"
SVR_REQUEST_PREFIX = "iris:svr_request:"
ALERT_MAXLEN = 256
BAG_LABELS = ("handbag", "backpack", "suitcase", "bag")
CART_LABELS = ("shopping_cart", "cart", "basket", "trolley", "stroller")


def text(value) -> str:
    """Decode a Redis field value as UTF-8 text."""
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value or "")


def format_pts(pts) -> str:
    """Format `pts` as a string for Redis alert and clip streams."""
    return f"{pts:.3f}".rstrip("0").rstrip(".") if isinstance(pts, float) else str(pts)


def _decode_fields(raw_fields: dict) -> dict[str, str]:
    # JPEG is binary and unused by the detector; never decode or copy its payload.
    return {key: text(v) for k, v in raw_fields.items() if (key := text(k)) != "jpeg"}


def _finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric data")
    return number


def _integer(value, name: str) -> int:
    number = _finite(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _shape(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain width and height")
    width, height = (_integer(v, name) for v in value)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return width, height


def _box(value, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    x1, y1, x2, y2 = (_finite(v, name) for v in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return x1, y1, x2, y2


def _parse_context_boxes(
    dets_str: str | None,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Extract context boxes in capture pixels; only repo seven-field rows use mux pixels.

    Six-field XYWH and dict XYXY fixtures already use capture pixels. Three-field
    repo detections carry no box and cannot contribute spatial context.
    """
    if not dets_str:
        return ([], [])
    dets = json.loads(dets_str)
    if not isinstance(dets, list):
        raise ValueError("dets must be a list")

    bag_boxes: list[tuple[float, float, float, float]] = []
    cart_boxes: list[tuple[float, float, float, float]] = []
    for item in dets:
        if isinstance(item, (list, tuple)) and len(item) in (3, 6, 7):
            label = str(item[0]).lower()
            if len(item) == 3:
                continue
            confidence = item[1]
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("class") or "").lower()
            confidence = item.get("conf", 1.0)
        else:
            raise ValueError("Invalid detection row")
        if label not in CART_LABELS and label not in BAG_LABELS:
            continue
        if not 0 <= _finite(confidence, "detection confidence") <= 1:
            raise ValueError("detection confidence must be in [0, 1]")
        if isinstance(item, dict):
            rect = _box(item.get("box", item.get("bbox")), "context box")
        else:
            offset = 3 if len(item) == 7 else 2
            x, y, w, h = (_finite(v, "context box") for v in item[offset:offset + 4])
            sx, sy = scale if len(item) == 7 else (1.0, 1.0)
            rect = _box((x * sx, y * sy, (x + w) * sx, (y + h) * sy), "context box")
        if label in CART_LABELS:
            cart_boxes.append(rect)
        else:
            bag_boxes.append(rect)
    return (bag_boxes, cart_boxes)


def _match_bag_to_person(
    person_box: tuple[float, float, float, float],
    bag_boxes: list[tuple[float, float, float, float]],
) -> tuple[bool, tuple[float, float, float, float] | None]:
    if not bag_boxes:
        return (False, None)
    px1, py1, px2, py2 = person_box
    pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    best_bag = None
    best_dist = float("inf")
    for b in bag_boxes:
        bx1, by1, bx2, by2 = b
        bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        dx = abs(bcx - pcx) / pw
        dy = abs(bcy - pcy) / ph
        if dx <= 0.85 and dy <= 0.65:
            d = math.hypot(bcx - pcx, bcy - pcy)
            if d < best_dist:
                best_dist = d
                best_bag = b
    return (best_bag is not None, best_bag)


def _match_carts_to_person(
    person_box: tuple[float, float, float, float],
    cart_boxes: list[tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float, float], ...]:
    if not cart_boxes:
        return ()
    px1, py1, px2, py2 = person_box
    pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    matched: list[tuple[float, float, float, float]] = []
    for c in cart_boxes:
        cx1, cy1, cx2, cy2 = c
        ccx, ccy = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)
        if abs(ccx - pcx) / pw <= 1.5 and abs(ccy - pcy) / ph <= 0.85:
            matched.append(c)
    return tuple(matched)


def parse_pose_frame(
    camera: str, entry: dict, *, mux_shape: tuple[int, int] | None = None,
) -> PoseFrame:
    """Parse metadata; malformed observed pose raises ValueError, absent pose is unobserved.

    When mux_shape is supplied, repo seven-field context boxes are scaled into
    pose capture coordinates. Without it, legacy capture-coordinate behavior is kept.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("fields", entry), dict):
        raise ValueError("Metadata entry fields must be a mapping")
    raw_fields = entry.get("fields", entry)
    fields = _decode_fields(raw_fields)
    frame_num = _integer(fields.get("frame_num") or 0, "frame_num")
    pts = _finite(fields.get("pts") or 0.0, "pts")
    pose_str = fields.get("pose")
    if pose_str is None:
        return PoseFrame(
            camera=camera,
            frame_num=frame_num,
            pts=pts,
            shape=(1920, 1080),
            tracks=(),
            observed=False,
        )

    payload = json.loads(pose_str)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("pose must be an object with an items list")
    shape = _shape(payload.get("shape"), "pose shape")
    scale = (1.0, 1.0)
    if mux_shape is not None:
        mux_width, mux_height = _shape(mux_shape, "mux_shape")
        scale = (shape[0] / mux_width, shape[1] / mux_height)
    bag_boxes, cart_boxes = _parse_context_boxes(fields.get("dets"), scale)

    tracks: list[TrackObservation] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("pose items must be objects")
        tid = _integer(item.get("id", item.get("track_id")), "track id")
        box = _box(item.get("box"), "person box")
        kps_raw = item.get("kps")
        if not isinstance(kps_raw, list) or len(kps_raw) != COCO_KEYPOINTS:
            raise ValueError(f"pose must contain exactly {COCO_KEYPOINTS} keypoints")
        kps: list[tuple[float, float, float]] = []
        for kp in kps_raw:
            if not isinstance(kp, (list, tuple)) or len(kp) != 3:
                raise ValueError("keypoints must contain x, y, confidence")
            x, y, confidence = (_finite(v, "keypoint") for v in kp)
            if not 0 <= confidence <= 1:
                raise ValueError("keypoint confidence must be in [0, 1]")
            kps.append((x, y, confidence))

        has_bag, matched_bag = _match_bag_to_person(box, bag_boxes)
        matched_carts = _match_carts_to_person(box, cart_boxes)
        if item.get("has_bag"):
            has_bag = True
        if item.get("bag_box") is not None:
            matched_bag = _box(item["bag_box"], "bag box")

        tracks.append(
            TrackObservation(
                track_id=tid,
                box=box,
                kps=tuple(kps),
                has_bag=has_bag,
                bag_box=matched_bag,
                cart_boxes=matched_carts,
            )
        )

    return PoseFrame(
        camera=camera,
        frame_num=frame_num,
        pts=pts,
        shape=shape,
        tracks=tuple(tracks),
        observed=True,
    )


class MetaStreamReader:
    """Per-camera XREAD cursor primed at the stream tip."""

    def __init__(self, r: redis.Redis) -> None:
        self.r = r
        self._cursors: dict[str, str] = {}

    def _prime(self, key: str) -> str:
        latest = self.r.xrevrange(key, count=1)
        return text(latest[0][0]) if latest else "0-0"

    def read(self, source_id: str, count: int = 64) -> list[dict]:
        key = f"{META_STREAM_PREFIX}{source_id}"
        if source_id not in self._cursors:
            self._cursors[source_id] = self._prime(key)
            return []
        res = self.r.xread({key: self._cursors[source_id]}, count=count)
        if not res:
            return []
        entries = res[0][1]
        self._cursors[source_id] = text(entries[-1][0])
        return [{"id": text(eid), "fields": _decode_fields(f)} for eid, f in entries]


def write_alert(
    r: redis.Redis,
    camera: str,
    *,
    severity: str,
    message: str,
    pts,
    kind: str = "suspected_concealment",
    category: str = "security",
    zone: str = "",
    count=None,
) -> bool:
    """Validate non-accusatory wording and write one alert entry to `iris:alert:<camera>`."""
    validate_non_accusatory_text(message)
    if severity not in ("critical", "warning", "info"):
        raise ValueError(
            f"Concealment alert severity must be 'critical', 'warning', or 'info', got '{severity}'"
        )

    fields = {
        "v": "1",
        "kind": kind,
        "zone": zone,
        "severity": severity,
        "category": category,
        "message": message,
        "ts": f"{time.time():.3f}",
        "pts": format_pts(pts),
        "count": "" if count is None else str(count),
    }
    try:
        r.xadd(f"{ALERT_STREAM_PREFIX}{camera}", fields, maxlen=ALERT_MAXLEN)
        return True
    except redis.RedisError:
        logger.exception("the alert on %s could not be written", camera)
        return False


def request_clip(r: redis.Redis, camera: str, pts) -> bool:
    """Request an SVR clip around `pts` on `iris:svr_request:<camera>`."""
    try:
        r.xadd(f"{SVR_REQUEST_PREFIX}{camera}", {"pts": format_pts(pts)}, maxlen=ALERT_MAXLEN)
        return True
    except redis.RedisError:
        logger.exception("the clip request on %s could not be written", camera)
        return False
