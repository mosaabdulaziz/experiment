"""Per-stream detection config loaded from data/zones_config.txt.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU. The app writes the
zones file; the sidecar re-reads it on each zones version bump and rebuilds each StreamState's
zone list. Handles the current shape ({"zones": [{"id", "name", "points", "intrusion",
"overcrowding"}, ...]}) and the legacy single-zone shapes ({"points", "intrusion",
"overcrowding"} and {"points", "type"}).
"""
import json
import os


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray-casting algorithm.  polygon is a list of (x, y) tuples/lists."""
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


def _zone_config(raw: dict, index: int, default_intrusion_classes,
                 default_overcrowding_classes, default_threshold) -> dict:
    """One zone's runtime config. `label` mirrors the app-assigned id (and the nvdsanalytics
    roi-<label> key); it falls back to z<index> so it stays aligned with the render side for
    legacy data that predates explicit ids."""
    intr = raw.get("intrusion")
    if intr is not None:
        mode = intr.get("mode", "restricted")
        intrusion_classes = set(intr.get("classes") or default_intrusion_classes)
    else:
        # a legacy zone (points only, or {"points", "type"}) is restricted; a zone written for the proximity module
        # alone is not an intrusion zone (multicam reads the same file and camera ids)
        mode = raw.get("type", "off" if "proximity" in raw else "restricted")
        intrusion_classes = set(default_intrusion_classes)
    oc = raw.get("overcrowding")
    if oc is None:
        oc_enabled, oc_threshold, oc_classes = False, default_threshold, set(default_overcrowding_classes)
    else:
        oc_enabled = bool(oc.get("enabled", False))
        oc_threshold = int(oc.get("threshold", default_threshold))
        oc_classes = set(oc.get("classes") or default_overcrowding_classes)
    return {
        "label": raw.get("id") or f"z{index}",
        "name": raw.get("name") or f"Zone {index + 1}",
        "mode": mode,
        "intrusion_classes": intrusion_classes,
        "overcrowding_enabled": oc_enabled,
        "overcrowding_threshold": oc_threshold,
        "overcrowding_classes": oc_classes,
    }


def load_stream_detection_configs(path: str, default_intrusion_classes,
                                  default_overcrowding_classes, default_threshold) -> dict:
    """Read the zones file and return {stream_id: [zone_config, ...]}, where each zone_config
    is {label, name, mode, intrusion_classes, overcrowding_enabled, overcrowding_threshold,
    overcrowding_classes}. Streams absent from the file (or with no zones) are absent from the
    result. Returns {} if the file is missing or unreadable, so the caller falls back to "no
    zones configured"."""
    try:
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            zones = json.load(f)
    except Exception:
        return {}
    out = {}
    for sid, entry in zones.items():
        if not isinstance(entry, dict):
            continue
        raw_zones = entry.get("zones")
        if raw_zones is None:
            # Legacy single-zone entry — wrap it so the runtime is list-uniform.
            raw_zones = [entry] if entry.get("points") else []
        configs = [
            _zone_config(z, i, default_intrusion_classes, default_overcrowding_classes,
                         default_threshold)
            for i, z in enumerate(raw_zones)
            if isinstance(z, dict) and z.get("points")
        ]
        if configs:
            out[sid] = configs
    return out


def load_stream_filter_zones(filter_path: str, mux_width: int, mux_height: int) -> dict:
    """Read the filter-zones file and return {stream_id: [[(x,y), ...], ...]} where each
    inner list is a polygon in 0-1 normalised coords.  Streams without filter zones are
    absent from the result."""
    try:
        if not os.path.isfile(filter_path):
            return {}
        with open(filter_path) as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for sid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        raw = entry.get("zones", [])
        if not isinstance(raw, list):
            continue
        polygons = []
        for z in raw:
            pts = z.get("points") or []
            if len(pts) < 3 or not isinstance(pts, list):
                continue
            polygons.append([(px / mux_width, py / mux_height) for px, py in pts])
        if polygons:
            out[sid] = polygons
    return out
