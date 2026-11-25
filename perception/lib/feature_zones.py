"""Active zones for one feature, read from the shared zones file (data/zones_config.txt).

A zone carries a block per feature that uses it -- `proximity` (its own loader), and `ppe`, `fire`, `fall` read here:
{"id", "name", "points": [[x, y], ...] in mux pixels, "ppe": {"enabled": true, "items": ["helmet"]}}. A camera with at
least one zone whose block for the feature is on alerts only inside those zones (and the alert names the zone); a camera
with none covers the whole frame, so nothing needs drawing to start. A block may name what the feature checks in that
zone (`items`: PPE's equipment); without it the feature checks everything it is configured for. Pure stdlib, so a
module and its test need no GStreamer.
"""
import json
import logging
import math
import os
from dataclasses import dataclass

from perception.lib.versioned import VersionedConfig

logger = logging.getLogger("iris.sidecar")


def _inside(x: float, y: float, polygon: tuple) -> bool:
    """Ray casting; a point on an edge may land either way."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def polygon_of(points) -> "tuple | None":
    """A zone's points as ((x, y), ...), or None unless they are at least 3 pairs of finite numbers: a malformed point
    (a hand-edited file) skips its zone instead of failing the whole load."""
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None
    try:
        polygon = tuple((float(x), float(y)) for x, y in points)
    except (TypeError, ValueError):
        return None
    return polygon if all(math.isfinite(v) for point in polygon for v in point) else None


@dataclass(frozen=True)
class FeatureZone:
    zone_id: str
    name: str
    polygon: tuple                 # ((x, y), ...) in mux pixels
    items: "tuple | None" = None   # what the feature checks in this zone (the block's `items`); None: everything

    def contains(self, x: float, y: float) -> bool:
        return _inside(float(x), float(y), self.polygon)


class FeatureZones:
    """{camera id: [FeatureZone, ...]} for one feature; `covering` answers whether a point is in scope."""

    def __init__(self, feature: str, by_camera: "dict | None" = None) -> None:
        self.feature = feature
        self.by_camera = by_camera or {}

    @classmethod
    def load(cls, path: str, feature: str) -> "FeatureZones":
        """A missing or unreadable file means no zones (whole frame everywhere); a broken zone is skipped."""
        try:
            if not os.path.isfile(path):
                return cls(feature)
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001 -- a broken file must not stop the pipeline
            logger.error("%s zones: cannot read %s: %s", feature, path, exc)
            return cls(feature)
        by_camera = {}
        for camera, entry in (data.items() if isinstance(data, dict) else ()):
            if not isinstance(entry, dict):
                continue
            raw_zones = entry.get("zones")
            if raw_zones is None:
                raw_zones = [entry] if entry.get("points") else []       # a legacy single-zone entry
            if not isinstance(raw_zones, list):
                logger.error("%s zones: camera %s has no list of zones; skipped", feature, camera)
                continue
            zones = []
            for i, raw in enumerate(raw_zones):
                block = raw.get(feature) if isinstance(raw, dict) else None
                if not (isinstance(block, dict) and block.get("enabled")):
                    continue
                polygon = polygon_of(raw.get("points"))
                if polygon is None:
                    logger.error("%s zones: camera %s zone %s needs at least 3 [x, y] points; skipped",
                                 feature, camera, raw.get("id") or f"z{i}")
                    continue
                items = block.get("items")
                items = (tuple(items) if isinstance(items, list) and all(isinstance(i, str) for i in items)
                         else None)
                zones.append(FeatureZone(str(raw.get("id") or f"z{i}"), str(raw.get("name") or f"Zone {i + 1}"),
                                         polygon, items))
            if zones:
                by_camera[camera] = zones
        return cls(feature, by_camera)

    def zones_at(self, camera: str, x: float, y: float) -> "tuple[bool, list]":
        """(in scope, the zones holding the point, in file order): (True, []) for a camera with no zones for this
        feature (whole frame), (False, []) for a point outside them all."""
        zones = self.by_camera.get(camera)
        if not zones:
            return True, []
        holding = [zone for zone in zones if zone.contains(x, y)]
        return bool(holding), holding

    def zone_at(self, camera: str, x: float, y: float) -> "tuple[bool, FeatureZone | None]":
        """(in scope, zone): zones_at's first zone (None for the whole frame or outside every zone)."""
        covered, holding = self.zones_at(camera, x, y)
        return covered, holding[0] if holding else None

    def covering(self, camera: str, x: float, y: float) -> "tuple[bool, str]":
        """(in scope, zone id): zone_at with the zone's id ("" for the whole frame or outside every zone)."""
        covered, zone = self.zone_at(camera, x, y)
        return covered, zone.zone_id if zone is not None else ""


ZONES_VERSION_KEY = "iris:ctrl:zones_version"      # the app bumps it on every zone save (app/stream_manager.py)


class LiveFeatureZones:
    """One feature's zones in a running module: loaded at start, reloaded whenever the app saves zones (the zones
    version), each load acknowledged on iris:analytics:<feature>_zones_applied. A module keeps one, calls start() from
    on_built, returns control() from its control hook and asks covering()."""

    def __init__(self, feature: str, path: str, redis) -> None:
        self.feature, self.path = feature, path
        self.zones = FeatureZones(feature)
        self._versions = (VersionedConfig(redis, ZONES_VERSION_KEY, f"iris:analytics:{feature}_zones_applied")
                          if redis is not None else None)

    def load(self) -> None:
        self.zones = FeatureZones.load(self.path, self.feature)
        if self.zones.by_camera:
            logger.info("%s zones: %s", self.feature,
                        {cam: [z.zone_id for z in zs] for cam, zs in self.zones.by_camera.items()})

    def start(self) -> None:
        self.load()
        if self._versions is not None:
            self._versions.acknowledge()

    def control(self) -> dict:
        return self._versions.control(lambda _version: self.load()) if self._versions is not None else {}

    def covering(self, camera: str, x: float, y: float) -> "tuple[bool, str]":
        return self.zones.covering(camera, x, y)

    def zone_at(self, camera: str, x: float, y: float) -> "tuple[bool, FeatureZone | None]":
        return self.zones.zone_at(camera, x, y)

    def zones_at(self, camera: str, x: float, y: float) -> "tuple[bool, list]":
        return self.zones.zones_at(camera, x, y)
