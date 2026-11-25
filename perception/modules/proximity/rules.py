"""The zones with proximity on, read from the zones file the app writes (`data/zones_config.txt`).

A zone is the app's zone ({"id", "name", "points", "intrusion", "overcrowding", ...}); a "proximity" block turns this
rule on for it and may override the module's defaults:

    "proximity": {"enabled": true, "person_classes": ["person"], "vehicle_classes": ["car", "truck"],
                  "max_gap": 0.75, "stationary_window_s": 2.0, "stationary_motion": 0.75, "persist_s": 0.25,
                  "grace_s": 0.5, "cooldown_s": 10, "depth_check": true, "depth_max_gap": 0.25,
                  "hidden_feet": "unconfirmed"}

Points are mux pixels, as the app stores them. The module tests membership itself (not through nvdsanalytics),
because a hidden person's foot point is one it recovered, not their box's (RESEARCH 5.9).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .zone_defaults import DEFAULTS, HIDDEN_FEET
from .zone_defaults import NON_NEGATIVE as _NON_NEGATIVE
from .zone_defaults import POSITIVE as _POSITIVE

logger = logging.getLogger("iris.sidecar")


@dataclass(frozen=True, eq=False)
class ZoneRule:
    zone_id: str
    name: str
    polygon: np.ndarray             # (n, 2) float32, mux px
    person_classes: frozenset
    vehicle_classes: frozenset
    max_gap: float                  # person heights
    stationary_window_s: float
    stationary_motion: float        # person heights
    persist_s: float
    grace_s: float
    cooldown_s: float
    depth_check: bool
    depth_max_gap: float            # relative depth gap at or under which the depth check says "the same depth"
    hidden_feet: str                # "unconfirmed" | "suppress": hidden feet with nothing to confirm them

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0


def zone_rule(raw: dict, index: int, defaults) -> "ZoneRule | None":
    """One zone's rule, None when proximity is off for it. `defaults` is the module's Settings. Raises ValueError for
    a block that is on but cannot be applied."""
    prox = raw.get("proximity")
    if not isinstance(prox, dict) or not prox.get("enabled"):
        return None
    zone_id = str(raw.get("id") or f"z{index}")
    points = raw.get("points") or []
    if len(points) < 3:
        raise ValueError(f"zone {zone_id}: a polygon needs 3 points, has {len(points)}")
    values = {k: float(prox.get(k, getattr(defaults, k))) for k in _POSITIVE + _NON_NEGATIVE}
    bad = [k for k in _POSITIVE if not values[k] > 0] + [k for k in _NON_NEGATIVE if not values[k] >= 0]
    if bad:
        raise ValueError(f"zone {zone_id}: {', '.join(f'{k}={values[k]}' for k in bad)} out of range")
    hidden = prox.get("hidden_feet", defaults.hidden_feet)
    if hidden not in HIDDEN_FEET:
        raise ValueError(f"zone {zone_id}: hidden_feet must be one of {HIDDEN_FEET}, got {hidden!r}")
    return ZoneRule(
        zone_id=zone_id, name=str(raw.get("name") or f"Zone {index + 1}"),
        polygon=np.asarray(points, np.float32).reshape(-1, 2),
        person_classes=frozenset(prox.get("person_classes") or defaults.person_classes),
        vehicle_classes=frozenset(prox.get("vehicle_classes") or defaults.vehicle_classes),
        depth_check=bool(prox.get("depth_check", DEFAULTS["depth_check"])), hidden_feet=hidden, **values)


def load_zone_rules(path: str, defaults) -> dict:
    """{camera id: [ZoneRule, ...]} for the zones with proximity on. A missing or unreadable file means no zones; a
    zone whose block cannot be applied is skipped with an error logged, and the camera's other zones still apply."""
    try:
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 -- a broken file must not stop the pipeline
        logger.error("proximity: cannot read %s: %s", path, exc)
        return {}
    out: dict = {}
    for sid, entry in (data.items() if isinstance(data, dict) else ()):
        if not isinstance(entry, dict):
            continue
        raw_zones = entry.get("zones")
        if raw_zones is None:
            raw_zones = [entry] if entry.get("points") else []       # a legacy single-zone entry
        rules = []
        for i, raw in enumerate(raw_zones):
            if not isinstance(raw, dict):
                continue
            try:
                rule = zone_rule(raw, i, defaults)
            except (TypeError, ValueError) as exc:
                logger.error("proximity: camera %s: %s; zone skipped", sid, exc)
                continue
            if rule is not None:
                rules.append(rule)
        if rules:
            out[sid] = rules
    return out
