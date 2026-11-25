"""The areas the sidecar's face SGIE is limited to (face register, 2026-10-05): the entrance-line camera's
door_area zones (register_source 1), or every camera's face_area zones (register_source 2, 2026-10-14), published to Redis as JSON {camera: [polygon, ...]} (perception/modules/face_detect/regions.py reads them)."""
from __future__ import annotations

import json

from zones import DOOR_AREA, ENTRANCE_LINE, FACE_AREA

REDIS_KEY = "barber:face_regions"


def face_regions_payload(zones_by_cam: dict[str, list[dict]], register_source: int = 1) -> str:
    """register_source 2 (seated faces, 2026-10-14): every camera's face_area polygons, in the {"match": "box"} form
    (the sidecar lets through a person whose box overlaps one: the face area is where the seated customer's face
    shows, his feet are below it); otherwise the entrance-line camera's door_area polygons (the door register, the
    person's feet inside), as before."""
    if register_source == 2:
        polys = {cam: [z["points"] for z in zs if z.get("type") == FACE_AREA] for cam, zs in zones_by_cam.items()}
        return json.dumps({"match": "box", "regions": {cam: p for cam, p in polys.items() if p}})
    door_cam = next((cam for cam, zs in zones_by_cam.items() if any(z.get("type") == ENTRANCE_LINE for z in zs)), None)
    polys = [z["points"] for z in zones_by_cam.get(door_cam, []) if z.get("type") == DOOR_AREA] if door_cam else []
    return json.dumps({door_cam: polys} if polys else {})
