"""Zone geometry and storage (2026-09-02, extracted from productivity.py --
same reasoning as jsonstore.py). Zones are genuinely shared between the two
features: a seat zone feeds productivity's desk-monitoring, a restricted
zone feeds attendance's violation detection, and both kinds live together
in the SAME zones.json file, edited in the same zone editor (design
decision, /grilling round 2 Q5: "zone editor stays where it is, gains a
restricted-zone option") -- so this module belongs to neither feature,
same as the geometry it holds.

zones.json shape: {cam_id: [{"id", "name", "points": [[x,y],...],
"type": "seat" | "restricted", "allowed_roles": [...]}, ...]} (normalized
0..1 coordinates). "type" defaults to "seat" for zones saved before this
field existed -- see seat_zones()/restricted_zones() below.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonstore import JsonFileStore

# Coordinate space of the sidecar's person_boxes = its capture size
# (sidecar.yaml capture_width/height). Zones are stored normalized 0..1 so
# they survive a resolution change; only this constant has to follow it.
FRAME_W, FRAME_H = 1080, 720

SEAT, RESTRICTED = "seat", "restricted"


def zone_store(data_dir: Path, seed_path: Path | None = None) -> JsonFileStore:
    """First run copies the committed seed (seat zones drawn from the CHIRLA
    footage) so the page works before anyone draws anything."""
    def seed():
        if seed_path and seed_path.is_file():
            with open(seed_path) as f:
                return json.load(f)
        return {}
    return JsonFileStore(data_dir / "zones.json", seed)


def seat_zones(zones_for_cam: list[dict]) -> list[dict]:
    """Zones that count for desk-monitoring -- everything saved before
    "type" existed is a seat zone, so this stays backward-compatible."""
    return [z for z in zones_for_cam if z.get("type", SEAT) == SEAT]


def restricted_zones(zones_for_cam: list[dict]) -> list[dict]:
    return [z for z in zones_for_cam if z.get("type", SEAT) == RESTRICTED]


# Whole-stream access check (2026-11-02). A camera listed in the attendance policy's `access_control`
# ({cam_id: {"allowed_roles": [...]}}) is checked for who is allowed to be seen on it. Where the check applies:
#   * restricted zones drawn on that camera -> only inside them, each with its own allowed_roles (as always);
#   * none drawn                            -> the whole stream, with the camera's allowed_roles.
# A camera NOT listed keeps the old behaviour (checked only inside restricted zones, if it has any). The whole-stream
# zone is a polygon far larger than the frame so a body on the frame's edge, whose anchor can fall a little outside
# 0..1, is still inside it.
WHOLE_STREAM = "whole-stream"
_BEYOND_FRAME = [[-1.0, -1.0], [2.0, -1.0], [2.0, 2.0], [-1.0, 2.0]]


def effective_restricted_zones(cam_id: str, zones_for_cam: list[dict], access_control: dict | None) -> list[dict]:
    """The restricted zones the violation check runs against for one camera."""
    drawn = restricted_zones(zones_for_cam)
    if drawn:
        return drawn
    cfg = (access_control or {}).get(cam_id)
    if cfg is None:
        return []
    return [{"id": WHOLE_STREAM, "name": "whole stream", "points": _BEYOND_FRAME, "type": RESTRICTED,
             "allowed_roles": list(cfg.get("allowed_roles") or [])}]


def role_allowed(role: str | None, zone: dict) -> bool:
    """Fail-closed both ways: no role on the employee, or no roles listed on
    the zone yet, means not allowed -- an empty allow-list is the safe
    default for something called "restricted", not "allow everyone"."""
    if role is None:
        return False
    return role in (zone.get("allowed_roles") or [])


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


def box_anchor(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Normalized test point for zone membership: lower-middle of the person
    box (x center, 80% down). Seated people's feet are hidden under the desk,
    so the true bottom is unreliable; the lower torso sits inside a seat zone
    whether they're seated or standing at it."""
    left, top, w, h = box
    return ((left + w / 2.0) / FRAME_W, (top + 0.8 * h) / FRAME_H)


def zone_hit(box, zones: list[dict]) -> str | None:
    """Id of the first zone containing the box anchor, else None."""
    x, y = box_anchor(box)
    for z in zones:
        if point_in_polygon(x, y, z.get("points", [])):
            return z["id"]
    return None
