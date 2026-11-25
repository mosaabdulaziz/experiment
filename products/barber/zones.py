"""Zone geometry and storage for the barber shop (three cameras).

Forked from products/attendance/zones.py @ 1f5b27b -- changes: barber zone
types; the entrance is a directed LINE (DoorLine); the frame size is a
parameter (FrameGeometry); zones are validated on write; zones are paired
across cameras (2026-09-24).

zones.json shape: {cam_id: [zone, ...]}, normalized 0..1 coordinates:
  waiting / chair / washing: {"id", "name", "type", "points": [[x, y], ... 3+],
                              "pair"?: str, "owner"?: true}
  chair:         ... "kind"?: "pedicure" (a pedicure chair: its sittings are pedicures, never customers; 2026-11-10)
  seat:          {"id", "name", "type": "seat", "points", "seat_of": <chair zone id, same camera>}
  face_area:     {"id", "name", "type": "face_area", "points", "chair": <a chair's area id, any camera>}
  entrance_line: {"id", "name", "type": "entrance_line", "points": [[x1, y1], [x2, y2]], "inside": [x, y]}
A shop AREA is what a zone stands for: its `pair` id when the same place is
drawn in two cameras (a chair in the centre view and in the side camera
facing it), else the zone's own id. The pair's `owner` zone's camera decides
what happens there (the side camera for a chair: seated, haircut); an
unpaired zone owns itself. Seats live in the owner camera's chair.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from jsonstore import JsonFileStore

ENTRANCE_LINE, WAITING, CHAIR, SEAT, WASHING, ROW = "entrance_line", "waiting", "chair", "seat", "washing", "row"
DOOR_AREA = "door_area"   # where the centre camera detects faces of people coming in (face register)
FACE_AREA = "face_area"   # where a chair's seated customer's face shows (any camera; branch 2, 2026-10-14)
ZONE_TYPES = (ENTRANCE_LINE, WAITING, CHAIR, SEAT, WASHING, ROW, DOOR_AREA, FACE_AREA)   # row: a side row of chairs seen by the centre camera (waits end there)
HAIRCUT, PEDICURE = "haircut", "pedicure"   # a chair zone's kind (2026-11-10): no kind = a barber chair
CHAIR_KINDS = (PEDICURE,)
AREA_TYPES = (WAITING, CHAIR, WASHING)   # zones that are places a person is "in"
PAIRABLE = AREA_TYPES

# A crossing only counts within the doorway: up to this fraction of the
# line's length beyond either end (a drawn line is rarely exactly the door).
EXTENT_TOLERANCE = 0.25


def zone_store(data_dir: Path) -> JsonFileStore:
    return JsonFileStore(data_dir / "zones.json", dict)


def zones_of_type(zones: list[dict], zone_type: str) -> list[dict]:
    return [z for z in zones if z.get("type") == zone_type]


def area_of(zone: dict) -> str:
    """The shop area a zone stands for: its pair id, else its own id."""
    return zone.get("pair") or zone["id"]


def owner_cameras(zones_by_cam: dict[str, list[dict]]) -> dict[str, str]:
    """area -> the camera that decides what happens there: the camera of the
    zone marked owner, or the only camera drawing an unpaired zone."""
    out: dict[str, str] = {}
    for cam, zones in zones_by_cam.items():
        for z in zones:
            if z["type"] in AREA_TYPES and (z.get("owner") or not z.get("pair")):
                out.setdefault(area_of(z), cam)
    return out


def zone_set_problems(zones_by_cam: dict[str, list[dict]], parked: dict[str, list[dict]] | None = None) -> list[dict]:
    """What's wrong with the zones of all cameras taken together (each
    camera's list is already valid on its own). Empty = consistent. Each
    problem is a dict {"code": str, ...} -- codes the UI translates.
    parked: the stored zones of cameras the site leaves out (ph-4 on the live
    Palm run, 2026-10-15): a pair whose other half is there is not single."""
    problems: list[dict] = []
    members: dict[str, list[tuple[str, dict]]] = {}
    for cam, zones in zones_by_cam.items():
        for z in zones:
            if z.get("pair"):
                members.setdefault(z["pair"], []).append((cam, z))
    elsewhere = {z["pair"] for zones in (parked or {}).values() for z in zones if z.get("pair")}
    for pair, zs in sorted(members.items()):
        cams = sorted({cam for cam, _ in zs})
        if len(cams) < 2 and pair not in elsewhere:
            problems.append({"code": "pair_single_camera", "pair": pair, "camera": cams[0]})
        if len(zs) != len(cams):
            problems.append({"code": "pair_twice_in_camera", "pair": pair})
        owners = sum(1 for _, z in zs if z.get("owner"))
        if owners != 1:
            problems.append({"code": "pair_owner_count", "pair": pair, "count": owners})
        if len({z.get("kind") for _, z in zs if z["type"] == CHAIR}) > 1:
            problems.append({"code": "pair_kind_mismatch", "pair": pair})
    for cam, zones in zones_by_cam.items():
        seated = {z["seat_of"] for z in zones if z["type"] == SEAT}
        for z in zones:
            if z["type"] == CHAIR and (z.get("owner") or not z.get("pair")) and z["id"] not in seated:
                problems.append({"code": "chair_without_seat", "camera": cam, "zone": z["name"]})
    for cam, zones in zones_by_cam.items():
        if any(z["type"] == DOOR_AREA for z in zones) and not any(z["type"] == ENTRANCE_LINE for z in zones):
            problems.append({"code": "door_area_ignored", "camera": cam})
    chair_areas = {area_of(z) for zones in zones_by_cam.values() for z in zones if z["type"] == CHAIR}
    for cam, zones in zones_by_cam.items():
        for z in zones:
            if z["type"] == FACE_AREA and z["chair"] not in chair_areas:
                problems.append({"code": "face_area_unknown_chair", "camera": cam, "zone": z["name"]})
    return problems


# ── geometry ────────────────────────────────────────────────────────────────

def point_in_polygon(x: float, y: float, pts: list[list[float]]) -> bool:
    """Ray casting; pts normalized [[x, y], ...]."""
    inside = False
    n = len(pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def rect_overlaps_polygon(left: float, top: float, right: float, bottom: float, pts: list[list[float]]) -> bool:
    """True if the axis-aligned rectangle and the polygon share any point (normalized). The sidecar face gate's
    test (perception/modules/face_detect/regions.py; this service can't import it): a polygon vertex inside the rect,
    the rect's centre inside the polygon, or two edges crossing."""
    if len(pts) < 3:
        return False
    if any(left <= x <= right and top <= y <= bottom for x, y in pts):
        return True
    if point_in_polygon((left + right) / 2, (top + bottom) / 2, pts):
        return True
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    edges = [(corners[i], corners[i - 1]) for i in range(4)]

    def crosses(p1, p2, q1, q2) -> bool:
        return ((_cross(q1, q2, p1) > 0) != (_cross(q1, q2, p2) > 0)
                and (_cross(p1, p2, q1) > 0) != (_cross(p1, p2, q2) > 0))

    return any(crosses(pts[i], pts[i - 1], a, b) for i in range(len(pts)) for a, b in edges)


@dataclass(frozen=True)
class FrameGeometry:
    """Coordinate space of the sidecar's person_boxes (its mux size). Zones
    are stored normalized so they survive a resolution change; only this
    has to follow it."""

    width: float
    height: float

    def anchor(self, box: tuple[float, float, float, float]) -> tuple[float, float]:
        """Normalized zone-membership test point: horizontal centre, 80% down
        the box. A seated person's feet are hidden by the chair, so the true
        bottom is unreliable; the lower torso sits inside a chair or bench
        zone whether the person is seated or standing at it."""
        left, top, w, h = box
        return ((left + w / 2.0) / self.width, (top + 0.8 * h) / self.height)

    def middle(self, box: tuple[float, float, float, float]) -> tuple[float, float]:
        """Normalized centre of the box: where stillness is measured. The
        box's bottom comes and goes as a barber works in front of a seated
        customer; its middle moves least (2026-09-24, Task 16 tune)."""
        left, top, w, h = box
        return ((left + w / 2.0) / self.width, (top + h / 2.0) / self.height)

    def seat_points(self, box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
        """Normalized points down the box's vertical centre line, from its
        middle (50%) to the anchor (80%): a body sits in a seat if any of
        them is inside it. A reclined or caped customer's anchor lands on
        the knees or footrest, below the seat, while the middle of the box
        is on it (2026-09-24, Task 16 tune)."""
        left, top, w, h = box
        return [((left + w / 2.0) / self.width, (top + f * h) / self.height) for f in (0.5, 0.65, 0.8)]

    def foot(self, box: tuple[float, float, float, float]) -> tuple[float, float]:
        """Normalized bottom-centre of the box: where a walking person meets
        the floor -- the point tested against the entrance line, which is
        drawn on the floor across the threshold."""
        left, top, w, h = box
        return ((left + w / 2.0) / self.width, (top + h) / self.height)

    def rect(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Normalized (left, top, right, bottom) of the box."""
        left, top, w, h = box
        return (left / self.width, top / self.height, (left + w) / self.width, (top + h) / self.height)


def zone_hit(point: tuple[float, float], zones: list[dict]) -> str | None:
    """Id of the first polygon zone containing the normalized point, else None."""
    x, y = point
    for z in zones:
        if point_in_polygon(x, y, z.get("points", [])):
            return z["id"]
    return None


def centroid(pts: list[list[float]]) -> tuple[float, float]:
    """Vertex mean -- good enough for the small, convex-ish zones drawn here."""
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def expand(pts: list[list[float]], margin: float) -> list[list[float]]:
    """The polygon scaled about its centroid by (1 + margin): margin 0.5 makes
    a chair's proximity area 1.5x its size in every direction."""
    cx, cy = centroid(pts)
    k = 1.0 + margin
    return [[cx + (x - cx) * k, cy + (y - cy) * k] for x, y in pts]


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass(frozen=True)
class DoorLine:
    """A directed entrance line, drawn on the floor across the doorway."""

    id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    inside_sign: int   # which sign of the cross product is the shop side

    @classmethod
    def from_zone(cls, zone: dict) -> "DoorLine":
        p1, p2 = tuple(zone["points"][0]), tuple(zone["points"][1])
        return cls(zone["id"], p1, p2, 1 if _cross(p1, p2, tuple(zone["inside"])) > 0 else -1)

    @property
    def length(self) -> float:
        return distance(self.p1, self.p2)

    def signed(self, pt: tuple[float, float]) -> float:
        """Distance from the (infinite) line, positive on the shop side. The
        infinite line also sorts people seen through a shop window: the
        storefront usually runs along the door's line."""
        return self.inside_sign * _cross(self.p1, self.p2, pt) / self.length

    def along(self, pt: tuple[float, float]) -> float:
        """Projection onto p1 -> p2: 0 at p1, 1 at p2."""
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        return ((pt[0] - self.p1[0]) * dx + (pt[1] - self.p1[1]) * dy) / (dx * dx + dy * dy)

    def within_extent(self, pt: tuple[float, float]) -> bool:
        return -EXTENT_TOLERANCE <= self.along(pt) <= 1.0 + EXTENT_TOLERANCE

    def distance_to(self, pt: tuple[float, float]) -> float:
        """Distance to the segment itself."""
        t = min(1.0, max(0.0, self.along(pt)))
        return distance(pt, (self.p1[0] + t * (self.p2[0] - self.p1[0]),
                             self.p1[1] + t * (self.p2[1] - self.p1[1])))

    def just_inside(self, pt: tuple[float, float], band: float) -> bool:
        """On the shop side, within `band` of the doorway itself."""
        return self.within_extent(pt) and 0.0 < self.signed(pt) <= band


# ── validation (PUT /api/monitoring/zones/{cam}) ────────────────────────────

class ZoneValidationError(ValueError):
    pass


def _clean_point(p, zid: str) -> list[float]:
    try:
        return [min(1.0, max(0.0, float(p[0]))), min(1.0, max(0.0, float(p[1])))]
    except (TypeError, ValueError, IndexError):
        raise ZoneValidationError(f"zone {zid}: points must be [x, y] numbers") from None


def validate_zones(raw: list) -> list[dict]:
    """Clean one camera's zone list, or raise ZoneValidationError. Unlike
    attendance (which drops a bad zone and saves the rest), a bad zone
    rejects the whole write: a silently vanished chair would silently stop
    counting haircuts."""
    if not isinstance(raw, list):
        raise ZoneValidationError("expected a list of zones")
    cleaned, seen = [], set()
    for i, z in enumerate(raw):
        if not isinstance(z, dict):
            raise ZoneValidationError(f"zone {i}: expected an object")
        zid = str(z.get("id") or f"zone-{uuid.uuid4().hex[:8]}")
        if zid in seen:
            raise ZoneValidationError(f"zone {zid}: duplicate id")
        seen.add(zid)
        ztype = z.get("type")
        if ztype not in ZONE_TYPES:
            raise ZoneValidationError(f"zone {zid}: type must be one of {', '.join(ZONE_TYPES)}")
        points = z.get("points")
        zone: dict = {"id": zid, "name": str(z.get("name") or zid), "type": ztype}

        if ztype == ENTRANCE_LINE:
            if not isinstance(points, list) or len(points) != 2:
                raise ZoneValidationError(f"zone {zid}: an entrance line needs exactly 2 points")
            pts = [_clean_point(p, zid) for p in points]
            if distance(pts[0], pts[1]) < 0.01:
                raise ZoneValidationError(f"zone {zid}: the entrance line is too short")
            if z.get("inside") is None:
                raise ZoneValidationError(f"zone {zid}: an entrance line needs an `inside` point "
                                          "on the shop side")
            inside = _clean_point(z["inside"], zid)
            if abs(_cross(pts[0], pts[1], inside)) / distance(pts[0], pts[1]) < 1e-3:
                raise ZoneValidationError(f"zone {zid}: the `inside` point must not lie on the line")
            zone.update(points=pts, inside=inside)
        else:
            if not isinstance(points, list) or len(points) < 3:
                raise ZoneValidationError(f"zone {zid}: needs at least 3 points")
            zone["points"] = [_clean_point(p, zid) for p in points]
            if ztype == SEAT:
                if not z.get("seat_of"):
                    raise ZoneValidationError(f"zone {zid}: a seat needs `seat_of` (its chair's id)")
                zone["seat_of"] = str(z["seat_of"])
            if ztype == FACE_AREA:
                if not z.get("chair"):
                    raise ZoneValidationError(f"zone {zid}: a face area needs `chair` (the chair whose customer it shows)")
                zone["chair"] = str(z["chair"])
            if ztype == CHAIR and z.get("kind") is not None:
                if z["kind"] not in CHAIR_KINDS:
                    raise ZoneValidationError(f"zone {zid}: kind must be one of {', '.join(CHAIR_KINDS)}")
                zone["kind"] = z["kind"]
        if z.get("pair") is not None:
            if ztype not in PAIRABLE:
                raise ZoneValidationError(f"zone {zid}: a {ztype} can't be paired")
            zone["pair"] = str(z["pair"])
        if z.get("owner") and ztype in PAIRABLE:
            zone["owner"] = True
        cleaned.append(zone)
    chairs = {z["id"] for z in cleaned if z["type"] == CHAIR}
    for z in cleaned:
        if z["type"] == SEAT and z["seat_of"] not in chairs:
            raise ZoneValidationError(f"zone {z['id']}: seat_of {z['seat_of']} is not a chair in this camera")
    return cleaned
