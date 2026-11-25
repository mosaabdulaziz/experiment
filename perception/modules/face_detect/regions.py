"""Which persons the face SGIE may see (barber face register, 2026-10-05): only those whose box bottom-centre lies
inside a door area of their own camera. The areas come from Redis (barber-service publishes the centre camera's
door_area zones as JSON {camera: [polygon, ...]}, normalised) and are re-read every refresh_s. Anything unreadable
-- Redis down, a bad value, no entry for the camera -- allows nothing: faces are never detected "everywhere".
Seated faces (2026-10-15): {"match": "box", "regions": {camera: [polygon, ...]}} lets through every person whose
box overlaps an area instead -- face areas are drawn where a seated customer's face shows, his feet are below them.
{"match": "foot", "regions": ...} is the plain form's test.
Pure stdlib (no gi), so it is unit-testable on the host; the engine's GIE gate applies it in its sink-pad probe."""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

logger = logging.getLogger("sidecar.face_regions")


def point_in_polygon(x: float, y: float, poly) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


MATCHES = ("foot", "box")


def rect_overlaps_polygon(left: float, top: float, right: float, bottom: float, poly) -> bool:
    """True if the axis-aligned rectangle and the polygon share any point."""
    if any(left <= x <= right and top <= y <= bottom for x, y in poly):
        return True
    if point_in_polygon((left + right) / 2, (top + bottom) / 2, poly):
        return True

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def crosses(p1, p2, q1, q2):
        return (cross(q1, q2, p1) > 0) != (cross(q1, q2, p2) > 0) and (cross(p1, p2, q1) > 0) != (cross(p1, p2, q2) > 0)

    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    edges = [(corners[i], corners[i - 1]) for i in range(4)]
    return any(crosses(poly[i], poly[i - 1], a, b) for i in range(len(poly)) for a, b in edges)


def _parse(raw) -> tuple[str, dict[str, list]]:
    """-> (match, {camera: [polygon, ...]}) from the plain form or the {"match", "regions"} form."""
    if not raw:
        return "foot", {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("regions must be an object")
    match = "foot"
    if "match" in data or "regions" in data:
        match, data = data.get("match"), data.get("regions")
        if match not in MATCHES or not isinstance(data, dict):
            raise ValueError(f"bad face regions envelope (match {match!r})")
    out = {}
    for cam, polys in data.items():
        if not isinstance(polys, list):
            raise ValueError(f"regions of {cam} must be a list")
        out[str(cam)] = [[(float(p[0]), float(p[1])) for p in poly] for poly in polys if len(poly) >= 3]
    return match, out


class FaceRegions:
    def __init__(self, fetch: Callable[[], str | bytes | None], refresh_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch, self._refresh_s, self._clock = fetch, refresh_s, clock
        self._regions: dict[str, list] = {}
        self._match = "foot"
        self._read_at: float | None = None
        self._logged_empty = False

    def _refresh(self) -> None:
        now = self._clock()
        if self._read_at is not None and now - self._read_at < self._refresh_s:
            return
        self._read_at = now
        try:
            self._match, self._regions = _parse(self._fetch())
        except Exception as exc:          # never let a bad value or Redis cost a frame
            if self._regions:
                logger.warning("face regions unreadable (%s): no faces until they are", exc)
            self._regions = {}
        if not self._regions and not self._logged_empty:
            self._logged_empty = True
            logger.info("face regions: none readable yet (no faces until the key holds door areas or face areas)")

    def allows(self, camera: str | None, left: float, top: float, width: float, height: float) -> bool:
        """The person box (normalised left, top, width, height): its bottom-centre inside an area of its camera
        ("foot"), or the box overlapping one ("box")."""
        self._refresh()
        if camera is None:
            return False
        polys = self._regions.get(camera, ())
        if self._match == "box":
            return any(rect_overlaps_polygon(left, top, left + width, top + height, p) for p in polys)
        return any(point_in_polygon(left + width / 2, top + height, p) for p in polys)
