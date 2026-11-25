import json
import os
from pathlib import Path

from .features import ZONE_FEATURES, camera_of
from .sidecar_config import running_sidecar_config

_DEFAULT_CLASSES = ["person"]
_DEFAULT_THRESHOLD = 5


def _zones_file_path(config: dict) -> str:
    return config.get("zones", {}).get("file_path", "data/zones_config.txt")


def mux_resolution(config: dict) -> tuple[int, int]:
    """The pipeline's mux frame size (detection coordinates). Public: the tracking code uses it too."""
    cfg = running_sidecar_config(config)
    try:
        return int(cfg["mux_width"]), int(cfg["mux_height"])
    except Exception:
        return 1280, 720


_mux_resolution = mux_resolution   # main's private name, kept for its callers


def _load_zones(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_zones(path: str, zones: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(zones, f)
    os.replace(tmp, path)


def intrusion_settings(entry: dict) -> dict:
    """Normalize a stored zone's intrusion config. Handles both the current schema
    ({"intrusion": {"mode", "classes"}}) and the legacy {"type"} shape (type becomes the
    mode; classes default to person)."""
    intr = entry.get("intrusion")
    if intr is not None:
        return {"mode": intr.get("mode", "restricted"),
                "classes": list(intr.get("classes") or _DEFAULT_CLASSES)}
    # a zone written for the proximity module alone is not an intrusion zone (perception/modules/zones/config.py)
    mode = entry.get("type", "off" if "proximity" in entry else "restricted")
    return {"mode": mode, "classes": list(_DEFAULT_CLASSES)}


def overcrowding_settings(entry: dict) -> dict:
    """Normalize a stored zone's overcrowding config. Legacy zones (no "overcrowding" key)
    default to disabled."""
    oc = entry.get("overcrowding")
    if oc is not None:
        return {"enabled": bool(oc.get("enabled", False)),
                "threshold": int(oc.get("threshold", _DEFAULT_THRESHOLD)),
                "classes": list(oc.get("classes") or _DEFAULT_CLASSES)}
    return {"enabled": False, "threshold": _DEFAULT_THRESHOLD, "classes": list(_DEFAULT_CLASSES)}


def zone_label(index: int) -> str:
    """Per-stream ROI label / stable id for the zone at this position. Also the nvdsanalytics
    roi-<label> key the sidecar reads back from each object's roiStatus."""
    return f"z{index}"


def _zone_default_name(index: int) -> str:
    return f"Zone {index + 1}"


def _normalize_zone(raw: dict, index: int) -> dict:
    """Normalize one raw zone (a new-schema list element, or a wrapped legacy single-zone
    entry) into {id, name, points, intrusion, overcrowding} (+ proximity when it has one). Points are returned as stored
    (mux pixels); callers that need 0-1 fractions convert via the mux resolution."""
    zone = {
        "id": raw.get("id") or zone_label(index),
        "name": raw.get("name") or _zone_default_name(index),
        "points": list(raw.get("points") or []),
        "intrusion": intrusion_settings(raw),
        "overcrowding": overcrowding_settings(raw),
    }
    if isinstance(raw.get("proximity"), dict):
        zone["proximity"] = dict(raw["proximity"])        # the proximity module reads it; the app keeps it on save
    for name in ZONE_FEATURES:                            # the safety features' blocks (perception/lib/feature_zones)
        if isinstance(raw.get(name), dict):
            zone[name] = {"enabled": bool(raw[name].get("enabled"))}
            items = raw[name].get("items")
            if isinstance(items, list) and all(isinstance(i, str) for i in items):
                zone[name]["items"] = list(items)                # PPE: the equipment this zone checks
    return zone


def normalize_stream_entry(entry: dict | None) -> list[dict]:
    """Return a stream's zones (pixel points) from its stored entry, handling the current
    {"zones": [...]} shape and both legacy single-zone shapes ({points, intrusion,
    overcrowding} and {points, type}). Missing/zoneless entries yield []."""
    if not entry:
        return []
    raw_zones = entry.get("zones")
    if raw_zones is None:
        # Legacy single-zone entry — wrap it so downstream code is list-uniform.
        raw_zones = [entry] if entry.get("points") else []
    return [_normalize_zone(z, i) for i, z in enumerate(raw_zones) if z.get("points")]


def stream_zones(config: dict, stream_id: str) -> list[dict]:
    """All zones for a stream with points normalized to 0-1 (the API / frame-annotation
    shape). [] when the stream has no saved zones."""
    # Zones are drawn per camera: a role sidecar's stream (`<camera>@<role>`) reads its camera's.
    zones = normalize_stream_entry(_load_zones(_zones_file_path(config)).get(camera_of(stream_id)))
    if not zones:
        return []
    mux_w, mux_h = mux_resolution(config)
    for z in zones:
        z["points"] = [(px / mux_w, py / mux_h) for px, py in z["points"]]
    return zones


def zone_by_id(config: dict, stream_id: str, zone_id: str) -> dict | None:
    """The single zone (normalized points) matching zone_id, or None. Used by the alert
    samplers to resolve which named zone fired for message copy + frame annotation."""
    if not zone_id:
        return None
    for z in stream_zones(config, stream_id):
        if z["id"] == zone_id:
            return z
    return None


# ── Filter zones ─────────────────────────────────────────────────────────────
# Filter zones define regions where detections are allowed.  Detections whose
# centre falls OUTSIDE every filter zone polygon for a stream are dropped
# before reaching the tracking pipeline.

def _filter_zones_file_path(config: dict) -> str:
    return config.get("zones", {}).get("filter_file_path", "data/filter_zones.json")


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm.  polygon is a list of (x, y) tuples in any order."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def stream_filter_zones(config: dict, stream_id: str) -> list[list[tuple[float, float]]]:
    """Return a list of polygons (each a list of (x, y) in 0-1 normalised coords)
    for the given stream's filter zones.  An empty list means no filter is active
    (all detections pass through)."""
    stream_id = camera_of(stream_id)      # per camera, like the zones: what every feature's sidecar on it reads
    path = _filter_zones_file_path(config)
    try:
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    entry = data.get(stream_id)
    if not entry:
        return []
    raw = entry if isinstance(entry, list) else entry.get("zones", [])
    if not raw:
        return []
    mux_w, mux_h = mux_resolution(config)
    polygons: list[list[tuple[float, float]]] = []
    for z in raw:
        pts = z.get("points") or []
        if len(pts) < 3:
            continue
        polygons.append([(px / mux_w, py / mux_h) for px, py in pts])
    return polygons


def save_stream_filter_zones(config: dict, stream_id: str,
                              polygons: list[list[tuple[float, float]]]) -> None:
    """Save filter zones for a stream.  Each polygon is a list of (x, y) in 0-1 coords."""
    stream_id = camera_of(stream_id)
    path = _filter_zones_file_path(config)
    try:
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}
    mux_w, mux_h = mux_resolution(config)
    stored = [{"points": [[round(px * mux_w), round(py * mux_h)] for px, py in poly]}
              for poly in polygons if len(poly) >= 3]
    if stored:
        data[stream_id] = {"zones": stored}
    else:
        data.pop(stream_id, None)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


_PROXIMITY_FILE = Path(__file__).resolve().parents[1] / "perception" / "modules" / "proximity" / "zone_defaults.json"


def proximity_rules() -> dict:
    """The proximity module's per-zone defaults and allowed values, read by path: importing the module would load
    the pipeline (perception/modules/proximity/zone_defaults.json)."""
    with open(_PROXIMITY_FILE) as f:
        return json.load(f)


def proximity_defaults(sidecar_cfg: dict) -> dict:
    """The values a zone gets for what it does not set: the module's file, with the running pipeline's
    `modules.proximity` settings on top."""
    mod = (sidecar_cfg.get("modules") or {}).get("proximity") or {}
    return {k: mod.get(k, v) for k, v in proximity_rules()["defaults"].items()}


def proximity_problem(block: dict, allowed_classes: "set | None") -> "str | None":
    """Why a zone's proximity block cannot be saved (the module's own checks, rules.py), or None. `allowed_classes`:
    the running pipeline's classes, None to check vehicle classes against the module's list only."""
    rules = proximity_rules()
    for k in rules["positive"]:
        if block.get(k) is not None and not block[k] > 0:
            return f"proximity {k} must be greater than 0"
    for k in rules["non_negative"]:
        if block.get(k) is not None and block[k] < 0:
            return f"proximity {k} must not be negative"
    if block.get("hidden_feet") is not None and block["hidden_feet"] not in rules["hidden_feet"]:
        return f"proximity hidden_feet must be one of {', '.join(rules['hidden_feet'])}"
    classes = block.get("vehicle_classes")
    if classes is not None:
        if not classes:
            return "proximity needs at least one vehicle class"
        unknown = [c for c in classes if c not in rules["vehicle_classes"]
                   or (allowed_classes is not None and c not in allowed_classes)]
        if unknown:
            return f"unknown vehicle classes: {', '.join(unknown)}"
    return None


def stored_proximity(block: dict, defaults: dict) -> dict:
    """What the zones file keeps: `enabled` and only the settings that differ from the defaults, so a zone follows
    later default changes for everything its operator left alone."""
    out = {"enabled": bool(block.get("enabled"))}
    for k, v in block.items():
        if k != "enabled" and v is not None and v != defaults.get(k):
            out[k] = v
    return out


def effective_proximity(stored: "dict | None", defaults: dict) -> dict:
    """A zone's proximity settings as they run: the defaults with its stored values on top."""
    stored = stored or {}
    return {**defaults, **stored, "enabled": bool(stored.get("enabled"))}
